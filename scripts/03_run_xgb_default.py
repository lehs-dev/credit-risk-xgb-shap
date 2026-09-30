"""Evaluate default XGBoost on ROC-AUC and fixed-reference SHAP stability."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.config import load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data
from creditrisk.objectives import evaluate_xgb_configuration
from creditrisk.splits import load_splits


def main() -> None:
    config = load_data_config("configs/data.yaml")
    data = load_raw_data(config.raw_data_path, config=config)
    X, y = extract_features_and_target(data, config=config)
    splits = load_splits(config.splits_dir)
    with open("configs/baseline.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)["models"]["xgboost_default"]["params"]

    result = evaluate_xgb_configuration(X, y, splits, params, config, top_k=5)
    summary = {
        "model": "xgboost_default",
        "roc_auc_cv": result["roc_auc_cv"],
        "roc_auc_cv_std": float(np.std(result["fold_roc_auc"])),
        "shap_stability": result["shap_stability"],
        "top_5_jaccard": result["top_k_jaccard"],
        "n_repeats": result["n_repeats"],
        "n_reference": result["n_reference"],
        "model_seed": params["random_state"],
        "shap_output": "raw_log_odds_margin",
        "shap_feature_perturbation": "tree_path_dependent",
        "stability_variation": "five_fixed_core_cv_training_subsets",
    }
    table_dir = Path("artifacts/tables")
    shap_dir = Path("artifacts/shap")
    table_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(table_dir / "xgb_default_objectives.csv", index=False)
    importance = pd.DataFrame(result["importance_matrix"], columns=result["feature_names"])
    importance.insert(0, "fold", [fold["fold"] for fold in splits["cv_folds"]])
    importance.to_csv(shap_dir / "xgb_default_importance_by_fold.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Fold ROC-AUC: {[round(value, 6) for value in result['fold_roc_auc']]}")
    print(f"Saved {table_dir / 'xgb_default_objectives.csv'}")
    print(f"Saved {shap_dir / 'xgb_default_importance_by_fold.csv'}")


if __name__ == "__main__":
    main()
