"""Guards for one-shot held-out Test scoring and label isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import creditrisk.final_test as final_test


def test_test_metrics_use_locked_threshold_and_require_index_alignment():
    indices = np.array([20, 10, 30, 40])
    labels = pd.Series([0, 0, 1, 1], index=indices)
    scores = pd.Series([0.1, 0.7, 0.8, 0.4], index=indices)
    result = final_test.compute_test_metrics(labels, scores, indices)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["brier_score"] == pytest.approx(0.225)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["test_prevalence"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="align"):
        final_test.compute_test_metrics(labels, scores.iloc[::-1], indices)
    with pytest.raises(ValueError, match="align"):
        final_test.compute_test_metrics(labels, scores, indices, threshold=0.6)
    with pytest.raises(ValueError, match="invalid"):
        final_test.compute_test_metrics(labels, scores.where(scores != 0.8, 1.1), indices)
    with pytest.raises(ValueError, match="invalid"):
        final_test.compute_test_metrics(labels, scores.where(scores != 0.8, np.nan), indices)


def test_preflight_does_not_load_raw_rows_or_test_labels(tmp_path, monkeypatch):
    config = final_test.load_final_test_config()
    config_file = tmp_path / "configs/final_test.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    candidates = {name: {"study_name": "fixed", "trial": number,
                         "source_role": name, "params": {"random_state": 42}}
                  for number, name in enumerate(config["candidate_names"])}
    monkeypatch.setattr(final_test, "load_locked_candidates", lambda root: candidates)
    monkeypatch.setattr(final_test, "protocol_fingerprint", lambda: "hpo")
    monkeypatch.setattr(final_test, "sensitivity_fingerprint", lambda root: "sensitivity")
    monkeypatch.setattr(final_test, "_validate_sensitivity_run", lambda *args: None)
    monkeypatch.setattr(final_test, "load_data_config", lambda path: SimpleNamespace(splits_dir="data/splits"))
    monkeypatch.setattr(final_test, "load_splits", lambda path: {
        "dev_indices": np.arange(24000), "test_indices": np.arange(24000, 30000),
    })
    monkeypatch.setattr(final_test, "_sha256_file", lambda path: "file-hash")
    monkeypatch.setattr(final_test, "package_versions", lambda: {"xgboost": "locked"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not parse raw rows or Test labels")
    monkeypatch.setattr(final_test, "load_raw_data", forbidden)
    plan = final_test.preflight_final_test(tmp_path)
    assert plan["config"]["expected_samples"]["test"] == 6000
    assert len(plan["candidate_params_sha256"]) == 5


def test_final_score_fits_all_models_on_development_before_accessing_test(tmp_path, monkeypatch):
    raw = pd.DataFrame({"x": np.arange(8), "label": [0, 1, 0, 1, 0, 0, 1, 1]})
    events = []
    monkeypatch.setattr(final_test, "load_raw_data", lambda *args, **kwargs: raw)
    monkeypatch.setattr(final_test, "extract_features_and_target",
                        lambda frame, config: (frame[["x"]], frame["label"]))

    class FakePreprocessor:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, X, y):
            events.append(("preprocess_fit", X.index.tolist()))
            return X
        def transform(self, X):
            events.append(("preprocess_transform", X.index.tolist()))
            return X

    class FakeModel:
        classes_ = np.array([0, 1])
        def fit(self, X, y):
            events.append(("model_fit", X.index.tolist()))
        def predict_proba(self, X):
            events.append(("test_predict", X.index.tolist()))
            values = np.array([0.8 if number in {6, 7} else 0.2 for number in X.index])
            return np.column_stack((1 - values, values))

    monkeypatch.setattr(final_test, "CreditRiskPreprocessor", FakePreprocessor)
    monkeypatch.setattr(final_test, "get_model", lambda *args: FakeModel())
    config = SimpleNamespace(raw_data_path="raw.csv", columns=SimpleNamespace(all_feature_columns=["x"]))
    plan = {
        "root": tmp_path, "data_config": config,
        "config": {"expected_samples": {"total": 8, "development": 4, "test": 4}, "threshold": 0.5},
        "splits": {"dev_indices": np.array([0, 1, 2, 3]), "test_indices": np.array([6, 4, 7, 5])},
        "candidates": {
            "a": {"study_name": "locked", "trial": 1, "source_role": "a", "params": {"seed": 42}},
            "b": {"study_name": "locked", "trial": 2, "source_role": "b", "params": {"seed": 42}},
        },
        "candidate_params_sha256": {"a": "hash-a", "b": "hash-b"},
    }
    fitted = final_test._fit_development_models(plan)
    table = final_test._score_final_test(plan, fitted)
    assert table["candidate"].tolist() == ["a", "b"]
    assert table["roc_auc"].tolist() == pytest.approx([1.0, 1.0])
    assert events == [
        ("preprocess_fit", [0, 1, 2, 3]),
        ("model_fit", [0, 1, 2, 3]),
        ("model_fit", [0, 1, 2, 3]),
        ("preprocess_transform", [6, 4, 7, 5]),
        ("test_predict", [6, 4, 7, 5]),
        ("test_predict", [6, 4, 7, 5]),
    ]


def test_one_shot_guard_refuses_existing_output(tmp_path, monkeypatch):
    path = tmp_path / final_test.MANIFEST_PATH
    path.parent.mkdir(parents=True)
    path.write_text('{"status": "started", "final_test_evaluated": false}', encoding="utf-8")
    monkeypatch.setattr(final_test, "preflight_final_test", lambda root: {"root": Path(root)})
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing output must prevent rescoring")
    monkeypatch.setattr(final_test, "_score_final_test", forbidden)
    with pytest.raises(ValueError, match="one-shot"):
        final_test.run_final_test(root=tmp_path)


def test_incomplete_sensitivity_manifest_blocks_test_preflight(tmp_path):
    path = tmp_path / "artifacts/tables/xgb_sensitivity_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "running", "final_test_evaluated": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible status"):
        final_test._validate_sensitivity_run(
            tmp_path, {"status": "complete", "final_test_evaluated": False}, {}, "hpo", "sensitivity"
        )


def _fake_completed_plan(tmp_path):
    candidates = {
        "a": {"study_name": "fixed", "trial": 1, "source_role": "a", "params": {"seed": 42}},
        "b": {"study_name": "fixed", "trial": 2, "source_role": "b", "params": {"seed": 42}},
    }
    return {
        "root": tmp_path,
        "config": {"protocol_version": "held_out_test_v1", "threshold": 0.5,
                   "candidate_names": list(candidates),
                   "expected_samples": {"development": 4, "test": 4}},
        "candidates": candidates,
        "hpo_protocol_fingerprint": "hpo", "sensitivity_fingerprint": "sensitivity",
        "final_test_fingerprint": "final", "input_sha256": {"raw.csv": "locked"},
        "candidate_params_sha256": {"a": "hash-a", "b": "hash-b"},
        "package_versions": {"xgboost": "locked"},
    }


def _summary_row(plan, candidate, y, probabilities):
    indices = np.array([5, 3, 4, 2])
    metrics = final_test.compute_test_metrics(pd.Series(y, index=indices),
                                               pd.Series(probabilities, index=indices), indices)
    source = plan["candidates"][candidate]
    return {
        "candidate": candidate, "study_name": source["study_name"],
        "trial": source["trial"], "source_role": source["source_role"],
        "candidate_params_sha256": plan["candidate_params_sha256"][candidate],
        "n_development": 4, "threshold": 0.5, **metrics,
        "fit_seconds": 0.1, "score_seconds": 0.01,
    }


def test_summary_rechecks_confusion_metrics_and_shared_prevalence(tmp_path):
    plan = _fake_completed_plan(tmp_path)
    same = _summary_row(plan, "a", [0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4])
    table = pd.DataFrame([same, _summary_row(plan, "b", [0, 0, 1, 1],
                                              [0.1, 0.7, 0.8, 0.4])])
    final_test._validate_summary(table, plan)
    bad_metric = table.copy()
    bad_metric.loc[1, "f1"] = 0.99
    with pytest.raises(ValueError, match="invalid row"):
        final_test._validate_summary(bad_metric, plan)
    drifted = pd.DataFrame([same, _summary_row(plan, "b", [0, 1, 1, 1],
                                                [0.1, 0.7, 0.8, 0.4])])
    with pytest.raises(ValueError, match="prevalence differs"):
        final_test._validate_summary(drifted, plan)
    empty_class = table.copy()
    empty_class.loc[0, "n_positive"] = 0
    with pytest.raises(ValueError, match="invalid class counts"):
        final_test._validate_summary(empty_class, plan)


def test_training_error_leaves_no_test_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test, "preflight_final_test", lambda root: {"root": Path(root)})
    def training_failure(plan):
        raise RuntimeError("fit failed before Test")
    monkeypatch.setattr(final_test, "_fit_development_models", training_failure)
    with pytest.raises(RuntimeError, match="fit failed"):
        final_test.run_final_test(root=tmp_path)
    assert not (tmp_path / final_test.MANIFEST_PATH).exists()
    assert not (tmp_path / final_test.SUMMARY_PATH).exists()


def test_finalize_complete_atomic_summary_without_rescoring(tmp_path, monkeypatch):
    plan = _fake_completed_plan(tmp_path)
    manifest_path = tmp_path / final_test.MANIFEST_PATH
    summary_path = tmp_path / final_test.SUMMARY_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest = {**final_test._manifest_identity(plan), "status": "started",
                "final_test_evaluated": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    table = pd.DataFrame([
        _summary_row(plan, name, [0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4])
        for name in plan["candidates"]
    ])
    table.to_csv(summary_path, index=False)
    monkeypatch.setattr(final_test, "preflight_final_test", lambda root: plan)
    def forbidden(*args, **kwargs):
        raise AssertionError("Finalize must not refit or rescore Test")
    monkeypatch.setattr(final_test, "_fit_development_models", forbidden)
    monkeypatch.setattr(final_test, "_score_final_test", forbidden)
    final_test.run_final_test(finalize=True, root=tmp_path)
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["final_test_evaluated"] is True
    assert completed["summary_sha256"] == final_test._sha256_file(summary_path)
    with pytest.raises(ValueError, match="Only a started"):
        final_test.run_final_test(finalize=True, root=tmp_path)
