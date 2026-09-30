"""Locked, one-shot evaluation on the untouched hold-out Test partition.

Preflight and verification inspect split indices, run manifests, result tables,
and file hashes. They never parse Test labels. The default command is the only
path that reads Test labels, and it refuses to run once an output exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from creditrisk.config import DataConfig, load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data
from creditrisk.hpo import package_versions, protocol_fingerprint
from creditrisk.metrics import compute_classification_metrics
from creditrisk.models import get_model
from creditrisk.preprocessing import CreditRiskPreprocessor
from creditrisk.sensitivity import (
    EXPECTED_TRIALS,
    load_locked_candidates,
    paired_differences,
    sensitivity_fingerprint,
    summarize_repeats,
)
from creditrisk.splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = "artifacts/tables/xgb_final_test_summary.csv"
MANIFEST_PATH = "artifacts/tables/xgb_final_test_manifest.json"
SENSITIVITY_ARTIFACTS = (
    "artifacts/tables/xgb_sensitivity_manifest.json",
    "artifacts/tables/xgb_sensitivity_repeats.csv",
    "artifacts/tables/xgb_sensitivity_summary.csv",
    "artifacts/tables/xgb_sensitivity_paired.csv",
    "artifacts/tables/xgb_sensitivity_paired_summary.csv",
)
INPUT_FILES = (
    "configs/final_test.yaml",
    "configs/data.yaml",
    "configs/baseline.yaml",
    "data/raw/default_credit_card.csv",
    "data/splits/outer_split.npz",
    "data/splits/cv_folds.json",
    "data/splits/shap_reference_indices.npy",
    "src/creditrisk/final_test.py",
    "src/creditrisk/config.py",
    "src/creditrisk/data.py",
    "src/creditrisk/models.py",
    "src/creditrisk/preprocessing.py",
    "src/creditrisk/metrics.py",
    "src/creditrisk/splits.py",
    "src/creditrisk/sensitivity.py",
    "scripts/08_run_final_test.py",
    *SENSITIVITY_ARTIFACTS,
)
METRIC_COLUMNS = (
    "roc_auc", "average_precision", "precision", "recall", "f1",
    "balanced_accuracy", "brier_score",
)


def load_final_test_config(path: Path = ROOT / "configs/final_test.yaml") -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    expected = {
        "protocol_version": "held_out_test_v1",
        "threshold": 0.5,
        "expected_samples": {"total": 30000, "development": 24000, "test": 6000},
        "candidate_names": list(EXPECTED_TRIALS),
        "required_sensitivity": {
            "protocol_version": "core_cv_resampling_v1",
            "mode": "full", "status": "complete",
            "cv_seeds": list(range(101, 111)),
            "completed_evaluations": 50, "completed_model_fits": 250,
            "final_test_evaluated": False,
        },
    }
    if config != expected:
        raise ValueError("Final Test configuration differs from the locked protocol.")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_sensitivity_run(
    root: Path, required: dict[str, Any], candidates: dict[str, Any],
    hpo_fingerprint: str, sensitivity_hash: str,
) -> None:
    """Check that the fixed candidates completed their Development-only audit."""
    tables = root / "artifacts/tables"
    with (tables / "xgb_sensitivity_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Sensitivity run has incompatible {key}.")
    expected_names = list(candidates)
    if (manifest.get("hpo_protocol_fingerprint") != hpo_fingerprint
            or manifest.get("sensitivity_fingerprint") != sensitivity_hash
            or manifest.get("candidates") != expected_names
            or manifest.get("candidate_sources") != candidates
            or manifest.get("planned_evaluations") != 50
            or manifest.get("planned_model_fits") != 250
            or manifest.get("n_development") != 24000
            or manifest.get("n_final_test_reserved") != 6000):
        raise ValueError("Sensitivity manifest does not match the locked candidates and protocol.")
    repeats = pd.read_csv(tables / "xgb_sensitivity_repeats.csv")
    expected_pairs = {(seed, name) for seed in required["cv_seeds"] for name in expected_names}
    needed = {"seed", "candidate", "sensitivity_fingerprint", "candidate_params_sha256",
              "n_model_fits", "n_reference", "roc_auc_cv", "shap_stability",
              "top_5_jaccard", "fold_roc_auc"}
    if (len(repeats) != 50 or not needed.issubset(repeats.columns)
            or repeats.duplicated(["seed", "candidate"]).any()
            or set(zip(repeats["seed"], repeats["candidate"])) != expected_pairs):
        raise ValueError("Sensitivity repeats do not contain exactly 50 locked evaluations.")
    for row in repeats.itertuples(index=False):
        if (row.sensitivity_fingerprint != sensitivity_hash
                or row.candidate_params_sha256 != _params_hash(candidates[row.candidate]["params"])
                or row.n_model_fits != 5 or row.n_reference != 1000):
            raise ValueError("Sensitivity repeat has an incompatible candidate or protocol.")
        fold_auc = json.loads(row.fold_roc_auc)
        if (len(fold_auc) != 5 or not np.isfinite(fold_auc).all()
                or not all(0 <= value <= 1 for value in fold_auc)
                or not math.isclose(row.roc_auc_cv, float(np.mean(fold_auc)), abs_tol=1e-12)
                or not -1 <= row.shap_stability <= 1
                or not 0 <= row.top_5_jaccard <= 1):
            raise ValueError("Sensitivity repeat has invalid Development metrics.")
    expected_tables = {
        "xgb_sensitivity_summary.csv": summarize_repeats(repeats),
    }
    paired, paired_summary = paired_differences(repeats)
    expected_tables["xgb_sensitivity_paired.csv"] = paired
    expected_tables["xgb_sensitivity_paired_summary.csv"] = paired_summary
    for filename, expected in expected_tables.items():
        observed = pd.read_csv(tables / filename)
        sort_columns = [name for name in ("candidate", "comparator", "seed") if name in expected.columns]
        observed = observed.sort_values(sort_columns).reset_index(drop=True)
        expected = expected.sort_values(sort_columns).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(observed, expected, check_exact=False,
                                          rtol=1e-10, atol=1e-12)
        except AssertionError as exc:
            raise ValueError(f"Sensitivity table {filename} does not match repeats.") from exc


def preflight_final_test(root: Path = ROOT) -> dict[str, Any]:
    """Validate the one-shot plan without loading raw rows or Test labels."""
    root = Path(root)
    config = load_final_test_config(root / "configs/final_test.yaml")
    candidates = load_locked_candidates(root)
    hpo_hash = protocol_fingerprint()
    sensitivity_hash = sensitivity_fingerprint(root)
    _validate_sensitivity_run(root, config["required_sensitivity"], candidates,
                              hpo_hash, sensitivity_hash)
    data_config = load_data_config(root / "configs/data.yaml")
    splits = load_splits(root / data_config.splits_dir)
    expected = config["expected_samples"]
    dev = np.asarray(splits["dev_indices"])
    test = np.asarray(splits["test_indices"])
    if (len(dev) != expected["development"] or len(test) != expected["test"]
            or set(dev).union(test) != set(range(expected["total"]))):
        raise ValueError("Outer split no longer partitions all 30,000 rows as locked.")
    input_hashes = {relative: _sha256_file(root / relative) for relative in INPUT_FILES}
    params_hashes = {name: _params_hash(candidate["params"])
                     for name, candidate in candidates.items()}
    fingerprint_material = {
        "hpo_protocol_fingerprint": hpo_hash,
        "sensitivity_fingerprint": sensitivity_hash,
        "candidate_params_sha256": params_hashes,
        "input_sha256": input_hashes,
        "package_versions": package_versions(),
    }
    final_hash = hashlib.sha256(json.dumps(fingerprint_material, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "root": root, "config": config, "data_config": data_config,
        "splits": splits, "candidates": candidates,
        "hpo_protocol_fingerprint": hpo_hash,
        "sensitivity_fingerprint": sensitivity_hash,
        "candidate_params_sha256": params_hashes,
        "input_sha256": input_hashes,
        "package_versions": fingerprint_material["package_versions"],
        "final_test_fingerprint": final_hash,
    }


def compute_test_metrics(
    y_true: pd.Series, probabilities: pd.Series, test_indices: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Score one aligned probability vector at the locked decision threshold."""
    expected_index = pd.Index(test_indices)
    if (threshold != 0.5 or not expected_index.is_unique
            or not y_true.index.equals(expected_index)
            or not probabilities.index.equals(expected_index)):
        raise ValueError("Test labels and probabilities must align with locked Test indices.")
    y = y_true.to_numpy()
    scores = probabilities.to_numpy(dtype=float)
    if (y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores)
            or set(np.unique(y)) != {0, 1}
            or not np.isfinite(scores).all()
            or not ((0 <= scores) & (scores <= 1)).all()):
        raise ValueError("Test labels or predicted probabilities are invalid.")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    metrics = compute_classification_metrics(y, scores, threshold=threshold)
    return {
        **metrics, "n_test": len(y), "n_positive": int(np.sum(y)),
        "n_negative": int(len(y) - np.sum(y)), "test_prevalence": float(np.mean(y)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _manifest_identity(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan["config"]
    return {
        "protocol_version": config["protocol_version"],
        "threshold": config["threshold"],
        "hpo_protocol_fingerprint": plan["hpo_protocol_fingerprint"],
        "sensitivity_fingerprint": plan["sensitivity_fingerprint"],
        "final_test_fingerprint": plan["final_test_fingerprint"],
        "input_sha256": plan["input_sha256"],
        "candidate_params_sha256": plan["candidate_params_sha256"],
        "candidate_sources": plan["candidates"],
        "package_versions": plan["package_versions"],
        "candidate_names": config["candidate_names"],
        "n_development": config["expected_samples"]["development"],
        "n_test": config["expected_samples"]["test"],
        "planned_model_fits": len(config["candidate_names"]),
        "test_used_for_selection": False,
        "shap_computed_on_test": False,
    }


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            table.to_csv(file, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fit_development_models(plan: dict[str, Any]) -> dict[str, Any]:
    """Fit all five locked models before creating the one-shot Test marker."""
    root = plan["root"]
    config: DataConfig = plan["data_config"]
    splits = plan["splits"]
    raw = load_raw_data(root / config.raw_data_path, config=config)
    if len(raw) != plan["config"]["expected_samples"]["total"]:
        raise ValueError("Raw data row count changed after preflight.")
    development = raw.loc[splits["dev_indices"]]
    X_development, y_development = extract_features_and_target(development, config=config)
    if (len(X_development) != plan["config"]["expected_samples"]["development"]
            or not X_development.index.equals(y_development.index)
            or set(X_development.columns) != set(config.columns.all_feature_columns)
            or set(np.unique(y_development)) != {0, 1}):
        raise ValueError("Development training data are misaligned or incomplete.")
    preprocessor = CreditRiskPreprocessor(scale_numerical=False, one_hot_categorical=True,
                                          config=config)
    X_train = preprocessor.fit_transform(X_development, y_development)
    trained = {}
    fit_seconds = {}
    for name, candidate in plan["candidates"].items():
        model = get_model("xgboost", dict(candidate["params"]))
        started = time.perf_counter()
        model.fit(X_train, y_development)
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError(f"Candidate {name} changed the binary class order.")
        fit_seconds[name] = time.perf_counter() - started
        trained[name] = model
    return {"raw": raw, "preprocessor": preprocessor, "trained": trained,
            "fit_seconds": fit_seconds, "n_development": len(X_development)}


def _score_final_test(plan: dict[str, Any], fitted: dict[str, Any]) -> pd.DataFrame:
    """Select and score held-out Test rows only after the one-shot marker."""
    config: DataConfig = plan["data_config"]
    splits = plan["splits"]
    raw = fitted["raw"]
    preprocessor = fitted["preprocessor"]
    trained = fitted["trained"]
    fit_seconds = fitted["fit_seconds"]
    test = raw.loc[splits["test_indices"]]
    X_test, y_test = extract_features_and_target(test, config=config)
    if not X_test.index.equals(y_test.index):
        raise ValueError("Test features and labels are misaligned.")
    X_test_transformed = preprocessor.transform(X_test)
    rows = []
    for name, candidate in plan["candidates"].items():
        started = time.perf_counter()
        probabilities = trained[name].predict_proba(X_test_transformed)
        if probabilities.shape != (len(X_test), 2):
            raise ValueError(f"Candidate {name} returned malformed class probabilities.")
        scores = pd.Series(probabilities[:, 1], index=X_test.index)
        metrics = compute_test_metrics(y_test, scores, splits["test_indices"],
                                       threshold=plan["config"]["threshold"])
        rows.append({
            "candidate": name, "study_name": candidate["study_name"],
            "trial": candidate["trial"], "source_role": candidate["source_role"],
            "candidate_params_sha256": plan["candidate_params_sha256"][name],
            "n_development": fitted["n_development"], "threshold": plan["config"]["threshold"],
            **metrics, "fit_seconds": fit_seconds[name],
            "score_seconds": time.perf_counter() - started,
        })
    return pd.DataFrame(rows)


def _read_output_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    path = plan["root"] / MANIFEST_PATH
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if any(manifest.get(key) != value for key, value in _manifest_identity(plan).items()):
        raise ValueError("Final Test manifest no longer matches the locked inputs and candidates.")
    return manifest


def _validate_summary(table: pd.DataFrame, plan: dict[str, Any]) -> None:
    """Check locked identities and metrics derivable from confusion counts."""
    required = {"candidate", "study_name", "trial", "source_role", "candidate_params_sha256",
                "n_development", "n_test", "threshold", "n_positive", "n_negative",
                "test_prevalence", "tn", "fp", "fn", "tp", "fit_seconds", "score_seconds",
                *METRIC_COLUMNS}
    if (not required.issubset(table.columns) or len(table) != len(plan["candidates"])
            or table["candidate"].tolist() != list(plan["candidates"])):
        raise ValueError("Final Test summary lacks the five locked candidates or required columns.")
    shared_counts = None
    for row in table.itertuples(index=False):
        candidate = plan["candidates"][row.candidate]
        counts = (row.n_positive, row.n_negative, row.tn, row.fp, row.fn, row.tp)
        if not all(math.isfinite(float(value)) and float(value).is_integer() and value >= 0
                   for value in counts):
            raise ValueError(f"Final Test summary has invalid confusion counts for {row.candidate}.")
        positives, negatives, tn, fp, fn, tp = (int(value) for value in counts)
        if positives <= 0 or negatives <= 0:
            raise ValueError(f"Final Test summary has invalid class counts for {row.candidate}.")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        balanced = (recall + tn / negatives) / 2
        derived = {"precision": precision, "recall": recall, "f1": f1,
                   "balanced_accuracy": balanced}
        if (row.study_name != candidate["study_name"] or row.trial != candidate["trial"]
                or row.source_role != candidate["source_role"]
                or row.candidate_params_sha256 != plan["candidate_params_sha256"][row.candidate]
                or row.n_development != plan["config"]["expected_samples"]["development"]
                or row.n_test != plan["config"]["expected_samples"]["test"]
                or row.threshold != plan["config"]["threshold"]
                or positives <= 0 or negatives <= 0 or positives + negatives != row.n_test
                or tn + fp != negatives or fn + tp != positives
                or not math.isclose(row.test_prevalence, positives / row.n_test,
                                    rel_tol=0, abs_tol=1e-12)
                or not all(math.isfinite(getattr(row, name)) and 0 <= getattr(row, name) <= 1
                           for name in METRIC_COLUMNS)
                or not all(math.isclose(getattr(row, name), value, rel_tol=0, abs_tol=1e-12)
                           for name, value in derived.items())
                or not math.isfinite(row.fit_seconds) or row.fit_seconds < 0
                or not math.isfinite(row.score_seconds) or row.score_seconds < 0):
            raise ValueError(f"Final Test summary has an invalid row for {row.candidate}.")
        if shared_counts is None:
            shared_counts = (positives, negatives, row.test_prevalence)
        elif (positives, negatives) != shared_counts[:2] or not math.isclose(
                row.test_prevalence, shared_counts[2], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Test prevalence differs across the five locked candidates.")


def verify_final_test(plan: dict[str, Any]) -> pd.DataFrame:
    """Verify persisted outputs and current inputs without parsing Test labels."""
    root = plan["root"]
    manifest_path = root / MANIFEST_PATH
    summary_path = root / SUMMARY_PATH
    if not manifest_path.exists() or not summary_path.exists():
        raise ValueError("Final Test manifest and summary must both exist for verification.")
    manifest = _read_output_manifest(plan)
    if (manifest.get("status") != "complete" or manifest.get("final_test_evaluated") is not True
            or manifest.get("completed_model_fits") != len(plan["candidates"])
            or manifest.get("completed_test_evaluations") != len(plan["candidates"])
            or manifest.get("summary_sha256") != _sha256_file(summary_path)):
        raise ValueError("Final Test manifest is incomplete or its summary changed.")
    table = pd.read_csv(summary_path)
    _validate_summary(table, plan)
    print(f"Verified {summary_path}: {len(table)} locked Test evaluations; fingerprint "
          f"{plan['final_test_fingerprint']}.")
    return table


def finalize_final_test(plan: dict[str, Any]) -> pd.DataFrame:
    """Complete a started manifest from an already atomic, complete summary only."""
    manifest_path = plan["root"] / MANIFEST_PATH
    summary_path = plan["root"] / SUMMARY_PATH
    if not manifest_path.exists() or not summary_path.exists():
        raise ValueError("Finalization requires both the started manifest and complete summary.")
    manifest = _read_output_manifest(plan)
    if manifest.get("status") != "started" or manifest.get("final_test_evaluated") is not False:
        raise ValueError("Only a started Final Test manifest can be finalized.")
    table = pd.read_csv(summary_path)
    _validate_summary(table, plan)
    manifest.update({
        "status": "complete", "final_test_evaluated": True,
        "completed_model_fits": len(table), "completed_test_evaluations": len(table),
        "summary_sha256": _sha256_file(summary_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(manifest, manifest_path)
    return verify_final_test(plan)


def run_final_test(*, dry_run: bool = False, verify: bool = False,
                   finalize: bool = False, root: Path = ROOT) -> None:
    if sum((dry_run, verify, finalize)) > 1:
        raise ValueError("Choose one of --dry-run, --verify, or --finalize.")
    plan = preflight_final_test(root)
    manifest_path = plan["root"] / MANIFEST_PATH
    summary_path = plan["root"] / SUMMARY_PATH
    if verify:
        verify_final_test(plan)
        return
    if finalize:
        finalize_final_test(plan)
        return
    if dry_run:
        if manifest_path.exists() or summary_path.exists():
            verify_final_test(plan)
        else:
            print("Final Test preflight passed: five locked candidates, 24,000 Development rows, "
                  "6,000 reserved Test rows; no Test labels parsed or models fit.")
            print(f"Final Test fingerprint: {plan['final_test_fingerprint']}")
        return
    if manifest_path.exists() or summary_path.exists():
        raise ValueError("Final Test output already exists; one-shot scoring is disabled. Use --verify.")
    fitted = _fit_development_models(plan)
    # Raw CSV is parsed here, but only Development rows enter preprocessing and fitting.
    # Training errors leave no Test marker, so fitting can be retried safely.
    if {relative: _sha256_file(plan["root"] / relative) for relative in INPUT_FILES} != plan["input_sha256"]:
        raise ValueError("Final Test input changed during Development fitting; Test was not scored.")
    if manifest_path.exists() or summary_path.exists():
        raise ValueError("Final Test output was created during fitting; Test was not scored.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_manifest_identity(plan), "status": "started", "final_test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Exclusive creation stops a second process before any Test row is extracted.
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    table = _score_final_test(plan, fitted)
    _validate_summary(table, plan)
    # Refuse to publish scores if any input changed while Test was scored.
    if {relative: _sha256_file(plan["root"] / relative) for relative in INPUT_FILES} != plan["input_sha256"]:
        raise ValueError("Final Test input changed during scoring; output remains incomplete.")
    _atomic_csv(table, summary_path)
    finalize_final_test(plan)
