"""Benchmark 6 baseline models on 5-fold CV under strict no-leakage protocol."""

from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.config import load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data
from creditrisk.metrics import compute_classification_metrics
from creditrisk.models import get_model
from creditrisk.preprocessing import CreditRiskPreprocessor
from creditrisk.splits import load_splits


def run_cv_for_model(
    model_name: str,
    model_cfg: dict,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cv_folds: list,
    data_config,
) -> dict:
    requires_scaling = model_cfg.get("requires_scaling", False)
    params = model_cfg.get("params", {})

    fold_metrics_list = []

    for fold_info in cv_folds:
        f_idx = fold_info["fold"]
        train_idx = fold_info["train_indices"]
        val_idx = fold_info["val_indices"]

        X_train, y_train = X_dev.loc[train_idx], y_dev.loc[train_idx]
        X_val, y_val = X_dev.loc[val_idx], y_dev.loc[val_idx]

        # Fit preprocessor strictly on train fold
        preprocessor = CreditRiskPreprocessor(
            scale_numerical=requires_scaling,
            scaler_type="standard",
            one_hot_categorical=True,
            config=data_config,
        )
        X_train_trans = preprocessor.fit_transform(X_train, y_train)
        X_val_trans = preprocessor.transform(X_val)

        # Train model
        model = get_model(model_name, params)
        model.fit(X_train_trans, y_train)

        # Predict probabilities
        y_val_prob = model.predict_proba(X_val_trans)[:, 1]
        metrics = compute_classification_metrics(y_val.values, y_val_prob)
        fold_metrics_list.append(metrics)

    # Aggregate across folds
    keys = fold_metrics_list[0].keys()
    agg = {"model": model_name}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics_list]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    return agg


def main():
    print("==================================================")
    print("STEP 5: Benchmarking 6 Baseline Classifiers       ")
    print("==================================================")

    data_config = load_data_config("configs/data.yaml")
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        baseline_cfg = yaml.safe_load(f)

    # 1. Load data and splits
    df_raw = load_raw_data(data_config.raw_data_path, config=data_config)
    X, y = extract_features_and_target(df_raw, config=data_config)

    splits = load_splits(data_config.splits_dir)
    dev_idx = splits["dev_indices"]
    cv_folds = splits["cv_folds"]

    X_dev = X.loc[dev_idx]
    y_dev = y.loc[dev_idx]

    print(f"Development Set: {len(X_dev):,} samples across {len(cv_folds)} folds.")

    # 2. Benchmark each model
    results = []
    models_dict = baseline_cfg.get("models", {})

    for model_name, model_cfg in models_dict.items():
        if not model_cfg.get("enabled", True):
            print(f"Skipping {model_name} (disabled)...")
            continue

        print(f"\nEvaluating: {model_name}...")
        start_time = datetime.now()
        res = run_cv_for_model(model_name, model_cfg, X_dev, y_dev, cv_folds, data_config)
        elapsed = (datetime.now() - start_time).total_seconds()
        res["duration_sec"] = elapsed
        results.append(res)
        print(
            f"   -> ROC-AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f} "
            f"| PR-AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f} ({elapsed:.1f}s)"
        )

    # 3. Format and save results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="roc_auc_mean", ascending=False).reset_index(drop=True)

    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_benchmark.csv"
    df_res.to_csv(out_file, index=False)

    print("\n==================================================")
    print("BENCHMARK RESULTS SUMMARY (Sorted by 5-fold CV ROC-AUC)")
    print("==================================================")
    cols_to_print = [
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "balanced_accuracy_mean",
        "duration_sec",
    ]
    print(df_res[cols_to_print].to_string(index=False))
    print(f"\nSaved benchmark table to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
