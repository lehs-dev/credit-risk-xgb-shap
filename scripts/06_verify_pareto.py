"""Verify the completed Development-study Pareto front and select representatives.

Reads the two main Optuna studies and the default-XGBoost summary. It never
loads the raw data, split indices, or Final Test observations.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import optuna
import pandas as pd
import yaml
from optuna.study import StudyDirection
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from creditrisk.pareto import (  # noqa: E402
    dominance_counts,
    dominates,
    pareto_front,
    select_representatives,
)

EXPECTED_COMPLETE = 64
EXPECTED_FOLDS = 5
EXPECTED_REFERENCE = 1000
EXPECTED_SEED = 42


def _load_config(mode: str) -> dict:
    with (ROOT / "configs" / f"{mode}_hpo.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_study(mode: str) -> tuple[optuna.Study, dict]:
    config = _load_config(mode)
    storage_path = ROOT / config["storage_path"]
    if not storage_path.is_file():
        raise FileNotFoundError(f"HPO SQLite database is missing: {storage_path}")
    study_name = config["study_name"]
    if study_name.endswith("_pilot"):
        raise ValueError("The Pareto check must use a main study, not a pilot study.")
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path.resolve()}",
    )
    attrs = study.user_attrs
    if attrs.get("mode") != mode or attrs.get("pilot") is not False:
        raise ValueError(f"Study {study_name} has incorrect mode/pilot metadata.")
    if attrs.get("sampler_seed") != EXPECTED_SEED or config["sampler_seed"] != EXPECTED_SEED:
        raise ValueError(f"Study {study_name} has an unexpected sampler seed.")
    if attrs.get("target_complete_trials") != EXPECTED_COMPLETE:
        raise ValueError(f"Study {study_name} has an unexpected COMPLETE-trial target.")
    if config["main"]["target_complete_trials"] != EXPECTED_COMPLETE:
        raise ValueError(f"Config for {study_name} no longer specifies 64 COMPLETE trials.")
    expected_sampler = "TPESampler" if mode == "single" else "NSGAIISampler"
    setting_key = "n_startup_trials" if mode == "single" else "population_size"
    expected_setting = {setting_key: 16}
    if attrs.get("sampler_name") != expected_sampler:
        raise ValueError(f"Study {study_name} has an unexpected sampler.")
    if attrs.get("sampler_settings") != expected_setting:
        raise ValueError(f"Study {study_name} has unexpected sampler settings.")
    if config["main"].get(setting_key) != 16:
        raise ValueError(f"Config for {study_name} no longer matches locked sampler settings.")
    if not attrs.get("protocol_fingerprint"):
        raise ValueError(f"Study {study_name} has no protocol fingerprint.")
    expected_directions = [StudyDirection.MAXIMIZE] * (1 if mode == "single" else 2)
    if study.directions != expected_directions:
        raise ValueError(f"Study {study_name} has unexpected objective directions.")

    states = [trial.state for trial in study.get_trials(deepcopy=False)]
    if any(state not in {TrialState.COMPLETE, TrialState.FAIL} for state in states):
        raise ValueError(f"Study {study_name} is still active or contains pruned trials.")
    if states.count(TrialState.COMPLETE) != EXPECTED_COMPLETE:
        raise ValueError(
            f"Study {study_name} has {states.count(TrialState.COMPLETE)} COMPLETE trials; "
            f"expected {EXPECTED_COMPLETE}."
        )
    return study, config


def _search_specs() -> dict:
    with (ROOT / "configs/xgb_search_space.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)["search_space"]


def _check_params(params: dict, specs: dict, trial_number: int) -> None:
    if set(params) != set(specs):
        raise ValueError(f"Trial {trial_number} does not contain the locked nine parameters.")
    for name, spec in specs.items():
        value = params[name]
        low, high = float(spec["low"]), float(spec["high"])
        if not math.isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f"Trial {trial_number} has invalid {name}.")
        if spec["type"] == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"Trial {trial_number} has a non-integer {name}.")
        if "step" in spec:
            steps = (float(value) - low) / float(spec["step"])
            if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Trial {trial_number} has off-grid {name}.")


def _completed_rows(study: optuna.Study, mode: str, specs: dict) -> list[dict]:
    rows = []
    for trial in study.get_trials(deepcopy=True):
        if trial.state != TrialState.COMPLETE:
            continue
        values = trial.values
        if values is None or len(values) != (1 if mode == "single" else 2):
            raise ValueError(f"Trial {trial.number} has incomplete objective values.")
        auc = float(values[0])
        stability = float(trial.user_attrs["shap_stability"])
        jaccard = float(trial.user_attrs["top_5_jaccard"])
        fold_auc = trial.user_attrs["fold_roc_auc"]
        if mode == "multi" and not math.isclose(float(values[1]), stability, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Trial {trial.number} has conflicting SHAP-stability values.")
        if not math.isclose(float(trial.user_attrs["roc_auc_cv"]), auc, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Trial {trial.number} has conflicting ROC-AUC values.")
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError(f"Trial {trial.number} has invalid ROC-AUC.")
        if not math.isfinite(stability) or not -1 <= stability <= 1:
            raise ValueError(f"Trial {trial.number} has invalid SHAP stability.")
        if not math.isfinite(jaccard) or not 0 <= jaccard <= 1:
            raise ValueError(f"Trial {trial.number} has invalid top-5 Jaccard.")
        if len(fold_auc) != EXPECTED_FOLDS or any(
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in fold_auc
        ):
            raise ValueError(f"Trial {trial.number} lacks five valid fold AUC values.")
        if not math.isclose(sum(fold_auc) / EXPECTED_FOLDS, auc, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Trial {trial.number} has inconsistent fold/mean AUC.")
        if trial.user_attrs.get("n_model_fits") != EXPECTED_FOLDS:
            raise ValueError(f"Trial {trial.number} does not record five model fits.")
        _check_params(trial.params, specs, trial.number)
        rows.append({
            "study_name": study.study_name,
            "trial": trial.number,
            "roc_auc_cv": auc,
            "shap_stability": stability,
            "top_5_jaccard": jaccard,
            "fold_roc_auc": json.dumps(fold_auc),
            "fold_roc_auc_std": pd.Series(fold_auc).std(ddof=0),
            "n_model_fits": EXPECTED_FOLDS,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            **{f"param_{name}": trial.params[name] for name in specs},
        })
    return sorted(rows, key=lambda row: row["trial"])


def _load_default() -> dict:
    path = ROOT / "artifacts/tables/xgb_default_objectives.csv"
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError("Expected one default-XGBoost summary row.")
    row = table.iloc[0].to_dict()
    for key, low, high in (
        ("roc_auc_cv", 0, 1), ("shap_stability", -1, 1), ("top_5_jaccard", 0, 1),
    ):
        value = float(row[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Default-XGBoost summary has invalid {key}.")
    if (int(row["n_repeats"]) != EXPECTED_FOLDS
            or int(row["n_reference"]) != EXPECTED_REFERENCE
            or int(row["model_seed"]) != EXPECTED_SEED
            or row["shap_output"] != "raw_log_odds_margin"
            or row["shap_feature_perturbation"] != "tree_path_dependent"
            or row["stability_variation"] != "five_fixed_core_cv_training_subsets"):
        raise ValueError("Default-XGBoost summary does not match the locked SHAP protocol.")
    return row


def _normalized(value: float, low: float, high: float) -> float:
    return 1.0 if high == low else (value - low) / (high - low)


def main() -> None:
    single, _ = _load_study("single")
    multi, _ = _load_study("multi")
    fingerprint = single.user_attrs["protocol_fingerprint"]
    if fingerprint != multi.user_attrs["protocol_fingerprint"]:
        raise ValueError("Single- and multi-objective studies use different protocols.")
    specs = _search_specs()
    if len(specs) != 9:
        raise ValueError("Expected the locked nine-parameter search space.")
    single_rows = _completed_rows(single, "single", specs)
    multi_rows = _completed_rows(multi, "multi", specs)
    default = _load_default()

    counts = dominance_counts(multi_rows)
    manual_front = pareto_front(multi_rows)
    manual_numbers = {row["trial"] for row in manual_front}
    optuna_numbers = {trial.number for trial in multi.best_trials}
    if manual_numbers != optuna_numbers:
        raise ValueError(
            f"Independent Pareto check disagrees with Optuna: "
            f"manual={sorted(manual_numbers)}, Optuna={sorted(optuna_numbers)}"
        )
    selected = select_representatives(multi_rows)
    selected_by_trial = {row["trial"]: row["role"] for row in selected}
    point_counts = Counter((row["roc_auc_cv"], row["shap_stability"]) for row in manual_front)
    auc_min = min(row["roc_auc_cv"] for row in manual_front)
    auc_max = max(row["roc_auc_cv"] for row in manual_front)
    stability_min = min(row["shap_stability"] for row in manual_front)
    stability_max = max(row["shap_stability"] for row in manual_front)

    audit_rows = []
    for row in multi_rows:
        on_front = row["trial"] in manual_numbers
        auc_scaled = _normalized(row["roc_auc_cv"], auc_min, auc_max) if on_front else None
        stability_scaled = (
            _normalized(row["shap_stability"], stability_min, stability_max)
            if on_front else None
        )
        audit_rows.append({
            **row,
            "dominated_by_count": counts[row["trial"]],
            "pareto_manual": on_front,
            "pareto_optuna": row["trial"] in optuna_numbers,
            "duplicate_objective_count": (
                point_counts[(row["roc_auc_cv"], row["shap_stability"])] if on_front else 0
            ),
            "selected_role": selected_by_trial.get(row["trial"]),
            "normalized_auc": auc_scaled,
            "normalized_stability": stability_scaled,
            "ideal_distance": (
                math.hypot(1 - auc_scaled, 1 - stability_scaled) if on_front else None
            ),
        })

    selected_table = pd.DataFrame([
        {"selection_role": row["role"], "protocol_fingerprint": fingerprint,
         "n_reference": EXPECTED_REFERENCE, **{k: v for k, v in row.items() if k != "role"}}
        for row in selected
    ])
    best_single = next(row for row in single_rows if row["trial"] == single.best_trial.number)
    comparison_rows = [
        {
            "source": "xgboost_default", "selection_role": "default", "trial": None,
            "roc_auc_cv": float(default["roc_auc_cv"]),
            "shap_stability": float(default["shap_stability"]),
            "top_5_jaccard": float(default["top_5_jaccard"]),
            "fold_roc_auc_std": float(default["roc_auc_cv_std"]),
        },
        {
            "source": single.study_name, "selection_role": "auc_best_tpe",
            "trial": best_single["trial"],
            **{key: best_single[key] for key in (
                "roc_auc_cv", "shap_stability", "top_5_jaccard", "fold_roc_auc_std"
            )},
        },
    ]
    comparison_rows.extend({
        "source": multi.study_name, "selection_role": row["role"],
        "trial": row["trial"],
        **{key: row[key] for key in (
            "roc_auc_cv", "shap_stability", "top_5_jaccard", "fold_roc_auc_std"
        )},
    } for row in selected)
    for row in comparison_rows:
        row["dominated_by_multi_pareto"] = any(
            dominates(
                (candidate["roc_auc_cv"], candidate["shap_stability"]),
                (row["roc_auc_cv"], row["shap_stability"]),
            )
            for candidate in manual_front
        )
        row["delta_auc_vs_tpe_best"] = row["roc_auc_cv"] - best_single["roc_auc_cv"]
        row["delta_stability_vs_tpe_best"] = (
            row["shap_stability"] - best_single["shap_stability"]
        )

    out_dir = ROOT / "artifacts/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(out_dir / "xgb_pareto_audit.csv", index=False)
    selected_table.to_csv(out_dir / "xgb_pareto_selected.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out_dir / "xgb_pareto_comparison.csv", index=False)
    pd.DataFrame([{
        "single_study": single.study_name,
        "multi_study": multi.study_name,
        "single_complete": len(single_rows),
        "multi_complete": len(multi_rows),
        "single_failed": sum(t.state == TrialState.FAIL for t in single.trials),
        "multi_failed": sum(t.state == TrialState.FAIL for t in multi.trials),
        "protocol_fingerprint": fingerprint,
        "pareto_manual_equals_optuna": True,
        "pareto_trial_count": len(manual_front),
        "pareto_unique_objective_count": len(point_counts),
        "selected_count": len(selected),
        "default_n_repeats": EXPECTED_FOLDS,
        "default_n_reference": EXPECTED_REFERENCE,
        "test_accessed": False,
    }]).to_csv(out_dir / "xgb_pareto_checks.csv", index=False)
    print(f"Verified {len(single_rows)} TPE and {len(multi_rows)} NSGA-II COMPLETE trials.")
    print(f"Independent Pareto check matched Optuna: {len(manual_front)} trials.")
    print(selected_table[["selection_role", "trial", "roc_auc_cv", "shap_stability"]].to_string(index=False))
    print(f"Saved Pareto audit, selected, comparison, and checks to {out_dir}")


if __name__ == "__main__":
    main()
