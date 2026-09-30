"""Guards for locked candidates, paired resampling, and safe result resumption."""

import json

import pandas as pd
import pytest

from creditrisk.sensitivity import (
    _params_hash,
    _validate_against_trial,
    _validate_existing_rows,
    load_sensitivity_config,
    paired_differences,
    summarize_repeats,
)


def _metric_row(seed, candidate, auc, stability, jaccard):
    return {
        "seed": seed, "candidate": candidate,
        "roc_auc_cv": auc, "shap_stability": stability,
        "top_5_jaccard": jaccard,
        "duration_seconds": 1.0, "n_model_fits": 5,
    }


def test_paired_differences_match_seed_instead_of_row_order():
    rows = pd.DataFrame([
        _metric_row(102, "pareto_auc", 0.81, 0.91, 0.7),
        _metric_row(101, "tpe_best", 0.75, 0.80, 0.5),
        _metric_row(102, "tpe_best", 0.80, 0.85, 0.6),
        _metric_row(101, "pareto_auc", 0.79, 0.90, 0.6),
    ])
    paired, summary = paired_differences(rows, candidates=("pareto_auc",),
                                         comparators=("tpe_best",))
    assert paired["seed"].tolist() == [101, 102]
    assert paired["delta_roc_auc_cv"].tolist() == pytest.approx([0.04, 0.01])
    assert paired["delta_shap_stability"].tolist() == pytest.approx([0.10, 0.06])
    assert summary.iloc[0]["delta_roc_auc_cv_mean"] == pytest.approx(0.025)
    assert summary.iloc[0]["delta_roc_auc_cv_std"] == pytest.approx(0.015)
    aggregate = summarize_repeats(rows.loc[rows["candidate"] == "pareto_auc"])
    assert aggregate.iloc[0]["roc_auc_cv_mean"] == pytest.approx(0.80)
    assert aggregate.iloc[0]["roc_auc_cv_std"] == pytest.approx(0.01)


def test_paired_differences_reject_unmatched_seeds():
    rows = pd.DataFrame([
        _metric_row(101, "pareto_auc", 0.8, 0.9, 0.7),
        _metric_row(102, "tpe_best", 0.8, 0.9, 0.7),
    ])
    with pytest.raises(ValueError, match="Seed sets differ"):
        paired_differences(rows, candidates=("pareto_auc",), comparators=("tpe_best",))


def test_locked_trial_validation_rejects_changed_identity_or_params():
    specs = {"max_depth": {"type": "int", "low": 2, "high": 10, "step": 1}}
    values = {
        "trial": 44, "state": "COMPLETE", "roc_auc_cv": 0.78,
        "shap_stability": 0.93, "top_5_jaccard": 0.7,
        "fold_roc_auc": json.dumps([0.77, 0.78, 0.79, 0.78, 0.78]),
        "n_model_fits": 5, "param_max_depth": 4,
    }
    row = pd.Series(values)
    trials = pd.DataFrame([values])
    assert _validate_against_trial(row, trials, 44, specs) == {"max_depth": 4}
    with pytest.raises(ValueError, match="replaced"):
        _validate_against_trial(row, trials, 57, specs)
    tampered = row.copy()
    tampered["param_max_depth"] = 5
    with pytest.raises(ValueError, match="conflicting max_depth"):
        _validate_against_trial(tampered, trials, 44, specs)


def test_resume_rejects_changed_fingerprint_candidate_or_folds():
    params = {"max_depth": 4, "random_state": 42}
    identity = {"study_name": "multi", "trial": 44, "source_role": "auc", "params": params}
    row = {
        **_metric_row(101, "pareto_auc", 0.78, 0.9, 0.7),
        "study_name": "multi", "trial": 44, "source_role": "auc",
        "sensitivity_fingerprint": "locked-run",
        "candidate_params_sha256": _params_hash(params), "folds_sha256": "locked-folds",
        "fold_roc_auc": json.dumps([0.77, 0.78, 0.79, 0.78, 0.78]),
        "n_reference": 1000,
    }
    table = pd.DataFrame([row])
    args = ({"pareto_auc": identity}, [101], 5, 1000, "locked-run", {101: "locked-folds"})
    _validate_existing_rows(table, *args)
    for field, replacement in (
        ("sensitivity_fingerprint", "other-run"),
        ("candidate_params_sha256", "other-candidate"),
        ("folds_sha256", "other-folds"),
    ):
        bad = table.copy()
        bad.loc[0, field] = replacement
        with pytest.raises(ValueError, match="changed protocol, candidate, or folds"):
            _validate_existing_rows(bad, *args)
    duplicate = pd.concat([table, table], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _validate_existing_rows(duplicate, *args)


def test_config_rejects_nonlocked_resampling_seed(tmp_path):
    original = load_sensitivity_config()
    altered = {**original, "cv_seeds": [101, 102]}
    path = tmp_path / "sensitivity.yaml"
    import yaml
    path.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="locked cv_seeds"):
        load_sensitivity_config(path)
