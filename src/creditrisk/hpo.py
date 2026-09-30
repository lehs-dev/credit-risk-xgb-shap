"""Fixed-protocol Optuna searches for XGBoost with persistent trial history."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.trial import TrialState

from creditrisk.config import load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data
from creditrisk.objectives import evaluate_xgb_configuration
from creditrisk.splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
SEARCH_PATH = ROOT / "configs/xgb_search_space.yaml"
PROTOCOL_FILES = (
    "configs/data.yaml",
    "configs/xgb_search_space.yaml",
    "data/raw/default_credit_card.csv",
    "data/splits/outer_split.npz",
    "data/splits/cv_folds.json",
    "data/splits/shap_reference_indices.npy",
    "src/creditrisk/config.py",
    "src/creditrisk/data.py",
    "src/creditrisk/models.py",
    "src/creditrisk/hpo.py",
    "src/creditrisk/objectives.py",
    "src/creditrisk/preprocessing.py",
    "src/creditrisk/shap_utils.py",
    "src/creditrisk/splits.py",
    "src/creditrisk/stability.py",
)


def load_search_space(path: Path = SEARCH_PATH) -> dict[str, Any]:
    """Validate the versioned parameter ranges shared by both searches."""
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("search_space"), dict):
        raise ValueError("XGBoost search configuration must contain search_space.")
    if not isinstance(config.get("fixed_params"), dict):
        raise ValueError("XGBoost search configuration must contain fixed_params.")
    if "random_state" not in config["fixed_params"]:
        raise ValueError("XGBoost random_state must be fixed.")
    for name, spec in config["search_space"].items():
        if not isinstance(spec, dict) or spec.get("type") not in {"int", "float"}:
            raise ValueError(f"Invalid parameter type for {name}.")
        low, high = spec.get("low"), spec.get("high")
        if low is None or high is None or not low < high:
            raise ValueError(f"Invalid range for {name}.")
        if spec.get("log") and (low <= 0 or "step" in spec):
            raise ValueError(f"Log-scale parameter {name} needs positive bounds and no step.")
        if "step" in spec and spec["step"] <= 0:
            raise ValueError(f"Parameter {name} needs a positive step.")
    return config


def suggest_xgb_params(trial: optuna.Trial, search_config: dict[str, Any]) -> dict[str, Any]:
    """Draw one candidate from the proposal's nine-dimensional search space."""
    params = dict(search_config["fixed_params"])
    for name, spec in search_config["search_space"].items():
        if name in params:
            raise ValueError(f"Parameter {name} occurs in fixed_params and search_space.")
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]),
                step=int(spec.get("step", 1)), log=bool(spec.get("log", False)),
            )
        else:
            options = {"log": bool(spec.get("log", False))}
            if "step" in spec:
                options["step"] = float(spec["step"])
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), **options
            )
    return params


def package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna", "PyYAML")
    return {name: version(name) for name in names}


def protocol_fingerprint() -> str:
    """Reject reuse of a study after its input data or objective code changes."""
    digest = hashlib.sha256()
    for relative_path in PROTOCOL_FILES:
        path = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(package_versions(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _check_result(result: dict[str, Any], n_folds: int) -> None:
    auc = result["roc_auc_cv"]
    stability = result["shap_stability"]
    jaccard = result["top_k_jaccard"]
    fold_auc = result["fold_roc_auc"]
    if len(fold_auc) != n_folds or not np.isfinite(fold_auc).all():
        raise ValueError("Fold ROC-AUC values must be finite and cover every fold.")
    if not all(0 <= value <= 1 for value in fold_auc):
        raise ValueError("Fold ROC-AUC values are outside [0, 1].")
    if not np.isfinite([auc, stability, jaccard]).all():
        raise ValueError("Trial objectives must be finite.")
    if not (0 <= auc <= 1 and -1 <= stability <= 1 and 0 <= jaccard <= 1):
        raise ValueError("Trial objective or diagnostic is outside its valid range.")


def _trial_row(trial: optuna.trial.FrozenTrial, mode: str) -> dict[str, Any]:
    values = trial.values or []
    attrs = trial.user_attrs
    return {
        "trial": trial.number,
        "state": trial.state.name,
        "roc_auc_cv": values[0] if values else attrs.get("roc_auc_cv"),
        "shap_stability": values[1] if mode == "multi" and len(values) > 1 else attrs.get("shap_stability"),
        "top_5_jaccard": attrs.get("top_5_jaccard"),
        "fold_roc_auc": json.dumps(attrs.get("fold_roc_auc")),
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "n_model_fits": attrs.get("n_model_fits"),
        "failure_type": attrs.get("failure_type"),
        "failure_message": attrs.get("failure_message"),
        **{f"param_{name}": value for name, value in trial.params.items()},
    }


def save_study_tables(study: optuna.Study, mode: str, suffix: str = "") -> None:
    """Export trial and best/Pareto tables; SQLite remains the source of truth."""
    out_dir = ROOT / "artifacts/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"xgb_{mode}_hpo{suffix}"
    trials = sorted(study.get_trials(deepcopy=True), key=lambda trial: trial.number)
    table = pd.DataFrame([_trial_row(trial, mode) for trial in trials])
    table.to_csv(out_dir / f"{prefix}_trials.csv", index=False)
    selected = study.best_trials if mode == "multi" else (
        [study.best_trial] if any(trial.state == TrialState.COMPLETE for trial in trials) else []
    )
    selected_table = pd.DataFrame([_trial_row(trial, mode) for trial in selected])
    selected_table.to_csv(out_dir / f"{prefix}_{'pareto' if mode == 'multi' else 'best'}.csv", index=False)
    complete = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    manifest = {
        "study_name": study.study_name,
        "mode": mode,
        "pilot": study.user_attrs.get("pilot"),
        "target_complete_trials": study.user_attrs.get("target_complete_trials"),
        "complete_trials": len(complete),
        "failed_trials": sum(trial.state == TrialState.FAIL for trial in trials),
        "attempted_trials": sum(trial.state != TrialState.WAITING for trial in trials),
        "model_fits_complete": sum(trial.user_attrs.get("n_model_fits", 0) for trial in complete),
        "shap_evaluations_complete": sum(trial.user_attrs.get("n_model_fits", 0) for trial in complete),
        "trial_duration_seconds_total": sum(
            trial.duration.total_seconds() for trial in trials if trial.duration is not None
        ),
        "protocol_fingerprint": study.user_attrs.get("protocol_fingerprint"),
        "sampler_name": study.user_attrs.get("sampler_name"),
        "sampler_seed": study.user_attrs.get("sampler_seed"),
        "sampler_settings": json.dumps(study.user_attrs.get("sampler_settings"), sort_keys=True),
        "package_versions": json.dumps(study.user_attrs.get("package_versions"), sort_keys=True),
        "hardware": json.dumps(study.user_attrs.get("hardware"), sort_keys=True),
    }
    pd.DataFrame([manifest]).to_csv(out_dir / f"{prefix}_manifest.csv", index=False)


def run_search(mode: str, pilot: bool = False, target_trials: int | None = None) -> optuna.Study:
    """Run to a target number of COMPLETE trials, preserving each trial in SQLite."""
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'.")
    with (ROOT / f"configs/{mode}_hpo.yaml").open(encoding="utf-8") as file:
        run_config = yaml.safe_load(file)
    search_config = load_search_space()
    selected = run_config["pilot" if pilot else "main"]
    target = int(target_trials if target_trials is not None else selected["target_complete_trials"])
    if target < 1:
        raise ValueError("Target COMPLETE trial count must be positive.")
    max_attempts = int(selected["max_attempts"])
    if max_attempts < target:
        raise ValueError("max_attempts must be at least target_complete_trials.")
    seed = int(run_config["sampler_seed"])
    study_name = run_config["study_name"] + ("_pilot" if pilot else "")
    storage_path = ROOT / run_config["storage_path"]
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.resolve()}"
    if mode == "single":
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=int(selected["n_startup_trials"])
        )
        directions = ["maximize"]
    else:
        sampler = optuna.samplers.NSGAIISampler(
            seed=seed, population_size=int(selected["population_size"])
        )
        directions = ["maximize", "maximize"]

    config = load_data_config(ROOT / "configs/data.yaml")
    data = load_raw_data(ROOT / config.raw_data_path, config=config)
    X, y = extract_features_and_target(data, config=config)
    splits = load_splits(ROOT / config.splits_dir)
    # The trial evaluator only receives Development rows. Final Test labels stay out.
    X = X.loc[splits["dev_indices"]]
    y = y.loc[splits["dev_indices"]]
    fingerprint = protocol_fingerprint()
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        sampler=sampler, directions=directions,
    )
    identity = {
        "mode": mode,
        "pilot": pilot,
        "protocol_fingerprint": fingerprint,
        "sampler_seed": seed,
        "sampler_name": type(sampler).__name__,
        "sampler_settings": {
            key: selected[key] for key in ("n_startup_trials", "population_size") if key in selected
        },
    }
    if study.user_attrs:
        for key, expected in identity.items():
            if study.user_attrs.get(key) != expected:
                raise ValueError(f"Study {study_name} has incompatible {key}; use a new study name.")
    elif study.trials:
        raise ValueError(f"Study {study_name} contains trials without protocol metadata.")
    else:
        for key, value in identity.items():
            study.set_user_attr(key, value)
        study.set_user_attr("package_versions", package_versions())
        study.set_user_attr("hardware", {
            "platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        })
    study.set_user_attr("target_complete_trials", target)
    if pilot and not study.trials:
        # Boundaries make the pilot useful for timing, not only smoke testing.
        for n_estimators, max_depth in ((100, 2), (1000, 10)):
            candidate = {
                "n_estimators": n_estimators, "max_depth": max_depth,
                "learning_rate": 0.1, "min_child_weight": 1,
                "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.0,
                "reg_alpha": 0.0001, "reg_lambda": 0.001,
            }
            study.enqueue_trial(candidate)

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        params = suggest_xgb_params(trial, search_config)
        try:
            result = evaluate_xgb_configuration(X, y, splits, params, config, top_k=5)
            _check_result(result, len(splits["cv_folds"]))
        except Exception as exc:
            trial.set_user_attr("failure_type", type(exc).__name__)
            trial.set_user_attr("failure_message", str(exc)[:1000])
            raise
        trial.set_user_attr("roc_auc_cv", result["roc_auc_cv"])
        trial.set_user_attr("shap_stability", result["shap_stability"])
        trial.set_user_attr("top_5_jaccard", result["top_k_jaccard"])
        trial.set_user_attr("fold_roc_auc", result["fold_roc_auc"])
        trial.set_user_attr("n_model_fits", result["n_repeats"])
        if mode == "single":
            return result["roc_auc_cv"]
        return result["roc_auc_cv"], result["shap_stability"]

    try:
        while True:
            trials = study.get_trials(deepcopy=False)
            running = [trial.number for trial in trials if trial.state == TrialState.RUNNING]
            if running:
                raise RuntimeError(
                    f"Study {study_name} has RUNNING trial(s) {running}; check for an active "
                    "process or recover stale trials before resuming."
                )
            complete = sum(trial.state == TrialState.COMPLETE for trial in trials)
            if complete >= target:
                break
            if len(trials) >= max_attempts:
                raise RuntimeError(
                    f"Study {study_name} reached {max_attempts} attempts with only {complete} COMPLETE trials."
                )
            study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True)
            save_study_tables(study, mode, "_pilot" if pilot else "")
    finally:
        save_study_tables(study, mode, "_pilot" if pilot else "")

    complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    print(f"{study_name}: {len(complete_trials)} COMPLETE / {len(study.trials)} attempts")
    if mode == "multi":
        print(f"Pareto configurations: {len(study.best_trials)}")
    else:
        print(f"Best CV ROC-AUC: {study.best_value:.6f} (trial {study.best_trial.number})")
    return study
