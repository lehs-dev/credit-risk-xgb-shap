"""Development-only resampling sensitivity check for five locked XGBoost models.

The 10 resampling seeds change only the five Core CV partitions. Development,
Core, the SHAP reference, model seeds, and all candidate parameters remain fixed.
Each candidate is evaluated on the same folds for a given seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from creditrisk.config import load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data
from creditrisk.hpo import _check_result, load_search_space, package_versions, protocol_fingerprint
from creditrisk.objectives import evaluate_xgb_configuration
from creditrisk.splits import load_splits, make_cv_folds, validate_split_protocol

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TRIALS = {
    "pareto_auc": 44,
    "pareto_stability": 57,
    "pareto_balanced": 61,
    "tpe_best": 50,
    "xgb_default": None,
}
PARETO_ROLES = {"auc": "pareto_auc", "stability": "pareto_stability", "balanced": "pareto_balanced"}
METRICS = ("roc_auc_cv", "shap_stability", "top_5_jaccard")
RUN_INPUTS = (
    "configs/sensitivity.yaml",
    "configs/baseline.yaml",
    "artifacts/tables/xgb_pareto_selected.csv",
    "artifacts/tables/xgb_pareto_checks.csv",
    "artifacts/tables/xgb_multi_hpo_trials.csv",
    "artifacts/tables/xgb_multi_hpo_manifest.csv",
    "artifacts/tables/xgb_single_hpo_best.csv",
    "artifacts/tables/xgb_single_hpo_trials.csv",
    "artifacts/tables/xgb_single_hpo_manifest.csv",
    "src/creditrisk/sensitivity.py",
    "scripts/07_run_sensitivity.py",
)


def load_sensitivity_config(path: Path = ROOT / "configs/sensitivity.yaml") -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Sensitivity configuration must be a mapping.")
    expected = {
        "protocol_version": "core_cv_resampling_v1",
        "cv_seeds": list(range(101, 111)),
        "n_splits": 5,
        "top_k": 5,
        "model_random_state": 42,
        "expected_samples": {
            "development": 24000,
            "core": 23000,
            "shap_reference": 1000,
            "final_test": 6000,
        },
        "candidate_trials": EXPECTED_TRIALS,
        "comparators": ["tpe_best", "xgb_default"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Sensitivity configuration changed locked {key}.")
    return config


def _read_one(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError(f"Expected one row in {path}.")
    return table.iloc[0]


def _equal_number(a: Any, b: Any) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), rel_tol=0.0, abs_tol=1e-12
    )


def _validated_search_params(row: pd.Series, specs: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in specs.items():
        raw = row[f"param_{name}"]
        value = float(raw)
        low, high = float(spec["low"]), float(spec["high"])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Candidate has invalid {name}={raw}.")
        if "step" in spec:
            steps = (value - low) / float(spec["step"])
            if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Candidate has off-grid {name}={raw}.")
        if spec["type"] == "int":
            if not value.is_integer():
                raise ValueError(f"Candidate has non-integer {name}={raw}.")
            params[name] = int(value)
        else:
            params[name] = value
    return params


def _validate_against_trial(
    candidate: pd.Series,
    trials: pd.DataFrame,
    expected_trial: int,
    specs: dict[str, Any],
) -> dict[str, Any]:
    """Reject a selected row whose trial, metrics, or parameters drifted."""
    if int(candidate["trial"]) != expected_trial:
        raise ValueError(f"Locked trial {expected_trial} was replaced.")
    matched = trials.loc[trials["trial"] == expected_trial]
    if len(matched) != 1 or matched.iloc[0]["state"] != "COMPLETE":
        raise ValueError(f"Locked trial {expected_trial} is missing or incomplete.")
    source = matched.iloc[0]
    for name in METRICS:
        if not _equal_number(candidate[name], source[name]):
            raise ValueError(f"Locked trial {expected_trial} has conflicting {name}.")
    for name, value in _validated_search_params(candidate, specs).items():
        if not _equal_number(value, source[f"param_{name}"]):
            raise ValueError(f"Locked trial {expected_trial} has conflicting {name}.")
    candidate_folds = json.loads(candidate["fold_roc_auc"])
    source_folds = json.loads(source["fold_roc_auc"])
    if len(candidate_folds) != 5 or len(source_folds) != 5 or not np.allclose(
        candidate_folds, source_folds, rtol=0, atol=1e-12
    ):
        raise ValueError(f"Locked trial {expected_trial} has conflicting fold AUC values.")
    if int(candidate["n_model_fits"]) != 5 or int(source["n_model_fits"]) != 5:
        raise ValueError(f"Locked trial {expected_trial} lacks five CV fits.")
    return _validated_search_params(candidate, specs)


def _validate_manifest(path: Path, study: str, mode: str, fingerprint: str) -> None:
    row = _read_one(path)
    if (row["study_name"] != study or row["mode"] != mode
            or str(row["pilot"]).lower() != "false"
            or int(row["complete_trials"]) != 64
            or int(row["failed_trials"]) != 0
            or row["protocol_fingerprint"] != fingerprint):
        raise ValueError(f"Study manifest no longer matches locked {mode} HPO protocol.")


def load_locked_candidates(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load fixed candidate identities from audited HPO exports; never reselect."""
    root = Path(root)
    fingerprint = protocol_fingerprint()
    tables = root / "artifacts/tables"
    _validate_manifest(tables / "xgb_multi_hpo_manifest.csv", "xgb_auc_shap_nsgaii", "multi", fingerprint)
    _validate_manifest(tables / "xgb_single_hpo_manifest.csv", "xgb_auc_tpe", "single", fingerprint)
    checks = _read_one(tables / "xgb_pareto_checks.csv")
    if (checks["protocol_fingerprint"] != fingerprint
            or int(checks["selected_count"]) != 3
            or str(checks["pareto_manual_equals_optuna"]).lower() != "true"
            or str(checks["test_accessed"]).lower() != "false"):
        raise ValueError("Independent Pareto audit no longer matches locked HPO protocol.")
    search = load_search_space(root / "configs/xgb_search_space.yaml")
    specs = search["search_space"]
    fixed = search["fixed_params"]
    if len(specs) != 9 or fixed.get("random_state") != 42:
        raise ValueError("The HPO search space or model seed changed.")
    multi_trials = pd.read_csv(tables / "xgb_multi_hpo_trials.csv")
    single_trials = pd.read_csv(tables / "xgb_single_hpo_trials.csv")
    if len(multi_trials) != 64 or len(single_trials) != 64:
        raise ValueError("Expected 64 HPO trials in each completed study export.")
    if multi_trials["trial"].duplicated().any() or single_trials["trial"].duplicated().any():
        raise ValueError("HPO trial exports contain duplicate trial numbers.")
    selected = pd.read_csv(tables / "xgb_pareto_selected.csv")
    if len(selected) != 3 or set(selected["selection_role"]) != set(PARETO_ROLES):
        raise ValueError("Expected exactly the three locked Pareto roles.")
    candidates: dict[str, dict[str, Any]] = {}
    for _, row in selected.iterrows():
        key = PARETO_ROLES[row["selection_role"]]
        trial = EXPECTED_TRIALS[key]
        if (row["protocol_fingerprint"] != fingerprint
                or row["study_name"] != "xgb_auc_shap_nsgaii"
                or int(row["n_reference"]) != 1000):
            raise ValueError(f"Pareto candidate {key} has incompatible protocol metadata.")
        params = _validate_against_trial(row, multi_trials, trial, specs)
        candidates[key] = {
            "study_name": "xgb_auc_shap_nsgaii", "trial": trial,
            "source_role": row["selection_role"], "params": {**fixed, **params},
        }
    best = _read_one(tables / "xgb_single_hpo_best.csv")
    if best["state"] != "COMPLETE":
        raise ValueError("The locked TPE best trial is incomplete.")
    params = _validate_against_trial(best, single_trials, 50, specs)
    candidates["tpe_best"] = {
        "study_name": "xgb_auc_tpe", "trial": 50,
        "source_role": "auc_best_tpe", "params": {**fixed, **params},
    }
    with (root / "configs/baseline.yaml").open(encoding="utf-8") as file:
        baseline = yaml.safe_load(file)["models"]["xgboost_default"]
    if baseline.get("enabled") is not True or baseline["params"].get("random_state") != 42:
        raise ValueError("The default XGBoost baseline or model seed changed.")
    candidates["xgb_default"] = {
        "study_name": "baseline", "trial": -1,
        "source_role": "default", "params": dict(baseline["params"]),
    }
    if set(candidates) != set(EXPECTED_TRIALS):
        raise ValueError("The candidate list no longer matches the locked five configurations.")
    return {key: candidates[key] for key in EXPECTED_TRIALS}


def sensitivity_fingerprint(root: Path = ROOT) -> str:
    """Bind resumable results to HPO inputs, fixed selections, and runner code."""
    digest = hashlib.sha256(protocol_fingerprint().encode("ascii"))
    for relative in RUN_INPUTS:
        digest.update(relative.encode("utf-8"))
        with (root / relative).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


def _folds_hash(folds: list[dict[str, Any]]) -> str:
    partition = [
        {"train_indices": fold["train_indices"], "val_indices": fold["val_indices"]}
        for fold in folds
    ]
    return hashlib.sha256(json.dumps(partition, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_data_shapes(splits: dict[str, Any], expected: dict[str, int]) -> None:
    for key, split_key in (
        ("development", "dev_indices"), ("core", "core_indices"),
        ("shap_reference", "shap_reference_indices"), ("final_test", "test_indices"),
    ):
        if len(splits[split_key]) != expected[key]:
            raise ValueError(f"Locked {key} size changed.")
    if not np.array_equal(
        validate_split_protocol(
            splits["dev_indices"], splits["test_indices"],
            splits["cv_folds"], splits["shap_reference_indices"],
        ), splits["core_indices"]
    ):
        raise ValueError("Locked Core indices changed.")


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            tmp_path = Path(file.name)
            table.to_csv(file, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            tmp_path = Path(file.name)
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _validate_result(result: dict[str, Any], n_splits: int, n_reference: int) -> None:
    _check_result(result, n_splits)
    if result["n_repeats"] != n_splits or result["n_reference"] != n_reference:
        raise ValueError("Evaluator used the wrong folds or SHAP reference set.")
    if not _equal_number(result["roc_auc_cv"], np.mean(result["fold_roc_auc"])):
        raise ValueError("Mean AUC disagrees with fold AUC values.")


def _validate_existing_rows(
    rows: pd.DataFrame, candidates: dict[str, Any], seeds: list[int], n_splits: int,
    n_reference: int, fingerprint: str, fold_hashes: dict[int, str],
) -> None:
    if rows.empty:
        return
    required = {"seed", "candidate", "study_name", "trial", "source_role", *METRICS,
                "fold_roc_auc", "duration_seconds", "n_model_fits", "n_reference",
                "sensitivity_fingerprint", "candidate_params_sha256", "folds_sha256"}
    if not required.issubset(rows.columns):
        raise ValueError("Existing sensitivity rows have missing columns.")
    if rows.duplicated(["seed", "candidate"]).any():
        raise ValueError("Existing sensitivity rows contain duplicate seed/candidate pairs.")
    for _, row in rows.iterrows():
        candidate = str(row["candidate"])
        if candidate not in candidates or int(row["seed"]) not in seeds:
            raise ValueError("Existing sensitivity rows contain an unknown seed or candidate.")
        identity = candidates[candidate]
        if (row["sensitivity_fingerprint"] != fingerprint
                or row["candidate_params_sha256"] != _params_hash(identity["params"])
                or row["folds_sha256"] != fold_hashes[int(row["seed"])]) :
            raise ValueError("Existing sensitivity row has a changed protocol, candidate, or folds.")
        if (row["study_name"] != identity["study_name"]
                or int(row["trial"]) != identity["trial"]
                or row["source_role"] != identity["source_role"]):
            raise ValueError("Existing sensitivity rows have a changed candidate identity.")
        fold_auc = json.loads(row["fold_roc_auc"])
        result = {"roc_auc_cv": float(row["roc_auc_cv"]),
                  "shap_stability": float(row["shap_stability"]),
                  "top_k_jaccard": float(row["top_5_jaccard"])}
        result.update({"fold_roc_auc": fold_auc, "n_repeats": int(row["n_model_fits"]),
                       "n_reference": int(row["n_reference"])})
        _validate_result(result, n_splits, n_reference)
        if not math.isfinite(float(row["duration_seconds"])) or float(row["duration_seconds"]) < 0:
            raise ValueError("Existing sensitivity row has invalid duration.")


def summarize_repeats(rows: pd.DataFrame) -> pd.DataFrame:
    """Across-seed summary; population standard deviation (ddof=0)."""
    if rows.empty:
        raise ValueError("Cannot summarize an empty sensitivity table.")
    summaries = []
    for candidate, group in rows.groupby("candidate", sort=False):
        record: dict[str, Any] = {"candidate": candidate, "n_seeds": len(group),
                                  "n_model_fits": int(group["n_model_fits"].sum())}
        for name in (*METRICS, "duration_seconds"):
            values = group[name].to_numpy(dtype=float)
            for stat, value in (
                ("mean", np.mean(values)), ("std", np.std(values, ddof=0)),
                ("min", np.min(values)), ("max", np.max(values)),
            ):
                record[f"{name}_{stat}"] = float(value)
        summaries.append(record)
    return pd.DataFrame(summaries)


def paired_differences(
    rows: pd.DataFrame,
    candidates: tuple[str, ...] = ("pareto_auc", "pareto_stability", "pareto_balanced"),
    comparators: tuple[str, ...] = ("tpe_best", "xgb_default"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract matched-seed baselines; refuse mismatched or incomplete pairs."""
    if rows.duplicated(["seed", "candidate"]).any():
        raise ValueError("Cannot pair duplicate seed/candidate rows.")
    indexed = rows.set_index(["seed", "candidate"])
    output = []
    for candidate in candidates:
        for comparator in comparators:
            candidate_seeds = set(rows.loc[rows["candidate"] == candidate, "seed"])
            comparator_seeds = set(rows.loc[rows["candidate"] == comparator, "seed"])
            if not candidate_seeds or candidate_seeds != comparator_seeds:
                raise ValueError(f"Seed sets differ for {candidate} and {comparator}.")
            for seed in sorted(candidate_seeds):
                first = indexed.loc[(seed, candidate)]
                second = indexed.loc[(seed, comparator)]
                output.append({
                    "seed": int(seed), "candidate": candidate, "comparator": comparator,
                    **{f"delta_{name}": float(first[name] - second[name]) for name in METRICS},
                })
    paired = pd.DataFrame(output)
    summaries = []
    for (candidate, comparator), group in paired.groupby(["candidate", "comparator"], sort=False):
        record: dict[str, Any] = {"candidate": candidate, "comparator": comparator,
                                  "n_paired_seeds": len(group)}
        for name in METRICS:
            values = group[f"delta_{name}"].to_numpy(dtype=float)
            for stat, value in (
                ("mean", np.mean(values)), ("std", np.std(values, ddof=0)),
                ("min", np.min(values)), ("max", np.max(values)),
            ):
                record[f"delta_{name}_{stat}"] = float(value)
        summaries.append(record)
    return paired, pd.DataFrame(summaries)


def run_sensitivity(*, dry_run: bool = False, smoke: bool = False, root: Path = ROOT) -> None:
    """Run or resume an exact 10×5 sweep; smoke uses a separate 1×1 artifact set."""
    if dry_run and smoke:
        raise ValueError("Choose either --dry-run or --smoke.")
    root = Path(root)
    config = load_sensitivity_config(root / "configs/sensitivity.yaml")
    candidates = load_locked_candidates(root)
    data_config = load_data_config(root / "configs/data.yaml")
    splits = load_splits(root / data_config.splits_dir)
    _validate_data_shapes(splits, config["expected_samples"])
    # Extract labels and features only after selecting Development rows.
    raw = load_raw_data(root / data_config.raw_data_path, config=data_config)
    development = raw.loc[splits["dev_indices"]]
    X, y = extract_features_and_target(development, config=data_config)
    if not X.index.equals(y.index) or len(X) != config["expected_samples"]["development"]:
        raise ValueError("Development features and labels are misaligned.")
    core = development.loc[splits["core_indices"]]
    seeds = [config["cv_seeds"][0]] if smoke else config["cv_seeds"]
    candidate_names = ["pareto_auc"] if smoke else list(candidates)
    folds_by_seed = {}
    fold_hashes = {}
    for seed in seeds:
        folds = make_cv_folds(core, target_col=data_config.columns.target_column,
                              n_splits=config["n_splits"], random_state=seed)
        validate_split_protocol(splits["dev_indices"], splits["test_indices"],
                                folds, splits["shap_reference_indices"])
        folds_by_seed[seed] = folds
        fold_hashes[seed] = _folds_hash(folds)
    if dry_run:
        print(f"Validated Development={len(development)}, Core={len(core)}, "
              f"reference={len(splits['shap_reference_indices'])}; "
              f"{len(config['cv_seeds'])} seeds × {len(candidates)} locked candidates "
              f"= {len(config['cv_seeds']) * len(candidates) * config['n_splits']} fits.")
        print(f"HPO fingerprint: {protocol_fingerprint()}")
        print(f"Sensitivity fingerprint: {sensitivity_fingerprint(root)}")
        return

    out = root / "artifacts/tables"
    suffix = "_smoke" if smoke else ""
    prefix = f"xgb_sensitivity{suffix}"
    paths = {name: out / f"{prefix}_{name}.{extension}" for name, extension in (
        ("repeats", "csv"), ("summary", "csv"), ("paired", "csv"),
        ("paired_summary", "csv"), ("manifest", "json"),
    )}
    fingerprint = sensitivity_fingerprint(root)
    planned = {(seed, candidate) for seed in seeds for candidate in candidate_names}
    if paths["manifest"].exists():
        with paths["manifest"].open(encoding="utf-8") as file:
            manifest = json.load(file)
        expected_manifest = {
            "protocol_version": config["protocol_version"],
            "mode": "smoke" if smoke else "full",
            "hpo_protocol_fingerprint": protocol_fingerprint(),
            "sensitivity_fingerprint": fingerprint,
            "cv_seeds": seeds,
            "candidates": candidate_names,
            "candidate_sources": {key: candidates[key] for key in candidate_names},
            "comparators": [] if smoke else config["comparators"],
            "n_splits": config["n_splits"], "top_k": config["top_k"],
            "model_random_state": config["model_random_state"],
            "n_development": len(development), "n_core": len(core),
            "n_shap_reference": len(splits["shap_reference_indices"]),
            "n_final_test_reserved": len(splits["test_indices"]),
            "final_test_evaluated": False,
            "planned_evaluations": len(planned),
            "planned_model_fits": len(planned) * config["n_splits"],
            "std_ddof": 0,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise ValueError("Existing sensitivity manifest is incompatible; use a new output name.")
    else:
        if paths["repeats"].exists():
            raise ValueError("Sensitivity rows exist without a matching manifest.")
        manifest = {
            "protocol_version": config["protocol_version"],
            "mode": "smoke" if smoke else "full",
            "status": "running",
            "hpo_protocol_fingerprint": protocol_fingerprint(),
            "sensitivity_fingerprint": fingerprint,
            "cv_seeds": seeds,
            "candidates": candidate_names,
            "candidate_sources": {key: candidates[key] for key in candidate_names},
            "comparators": [] if smoke else config["comparators"],
            "n_splits": config["n_splits"], "top_k": config["top_k"],
            "model_random_state": config["model_random_state"],
            "n_development": len(development), "n_core": len(core),
            "n_shap_reference": len(splits["shap_reference_indices"]),
            "n_final_test_reserved": len(splits["test_indices"]),
            "final_test_evaluated": False,
            "planned_evaluations": len(planned),
            "planned_model_fits": len(planned) * config["n_splits"],
            "std_ddof": 0,
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(manifest, paths["manifest"])

    rows = pd.read_csv(paths["repeats"]) if paths["repeats"].exists() else pd.DataFrame()
    _validate_existing_rows(rows, candidates, seeds, config["n_splits"],
                            len(splits["shap_reference_indices"]), fingerprint, fold_hashes)
    done = {(int(row["seed"]), str(row["candidate"])) for _, row in rows.iterrows()}
    if not done.issubset(planned):
        raise ValueError("Existing sensitivity results contain unplanned evaluations.")
    records = rows.to_dict("records") if not rows.empty else []
    for seed in seeds:
        repeat_splits = {**splits, "cv_folds": folds_by_seed[seed]}
        for key in candidate_names:
            if (seed, key) in done:
                continue
            identity = candidates[key]
            if identity["params"].get("random_state") != config["model_random_state"]:
                raise ValueError(f"Candidate {key} changed the locked model seed.")
            started = time.perf_counter()
            result = evaluate_xgb_configuration(
                X, y, repeat_splits, identity["params"], data_config, top_k=config["top_k"]
            )
            duration = time.perf_counter() - started
            _validate_result(result, config["n_splits"], len(splits["shap_reference_indices"]))
            records.append({
                "seed": seed, "candidate": key,
                "study_name": identity["study_name"], "trial": identity["trial"],
                "source_role": identity["source_role"],
                "sensitivity_fingerprint": fingerprint,
                "candidate_params_sha256": _params_hash(identity["params"]),
                "folds_sha256": fold_hashes[seed],
                "roc_auc_cv": result["roc_auc_cv"],
                "shap_stability": result["shap_stability"],
                "top_5_jaccard": result["top_k_jaccard"],
                "fold_roc_auc": json.dumps(result["fold_roc_auc"]),
                "duration_seconds": duration,
                "n_model_fits": config["n_splits"],
                "n_reference": result["n_reference"],
            })
            _atomic_csv(pd.DataFrame(records), paths["repeats"])
            done.add((seed, key))
            manifest["completed_evaluations"] = len(done)
            manifest["completed_model_fits"] = len(done) * config["n_splits"]
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(manifest, paths["manifest"])
            print(f"{prefix}: seed={seed}, {key}, AUC={result['roc_auc_cv']:.6f}, "
                  f"SHAP={result['shap_stability']:.6f} ({len(done)}/{len(planned)})", flush=True)
    if done != planned:
        raise RuntimeError("Sensitivity sweep ended before all planned evaluations completed.")
    final = pd.DataFrame(records).sort_values(["seed", "candidate"]).reset_index(drop=True)
    _validate_existing_rows(final, candidates, seeds, config["n_splits"],
                            len(splits["shap_reference_indices"]), fingerprint, fold_hashes)
    _atomic_csv(final, paths["repeats"])
    _atomic_csv(summarize_repeats(final), paths["summary"])
    if not smoke:
        paired, paired_summary = paired_differences(final)
        _atomic_csv(paired, paths["paired"])
        _atomic_csv(paired_summary, paths["paired_summary"])
    manifest["status"] = "complete"
    manifest["completed_evaluations"] = len(done)
    manifest["completed_model_fits"] = len(done) * config["n_splits"]
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest, paths["manifest"])
    print(f"Saved {prefix}: {len(done)} evaluations, {len(done) * config['n_splits']} model fits.")
