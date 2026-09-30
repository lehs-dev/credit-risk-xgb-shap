"""Script to generate reproducible stratified outer split, 5-fold CV, and SHAP reference set."""

from datetime import datetime
from pathlib import Path
import sys

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.config import load_data_config
from creditrisk.data import load_raw_data
from creditrisk.splits import make_core_indices, make_cv_folds, make_outer_split, sample_shap_reference, save_splits


def main():
    print("==================================================")
    print("STEP 2: Generating Anti-Leakage Splits Protocol   ")
    print("==================================================")

    config = load_data_config("configs/data.yaml")
    df_raw = load_raw_data(config.raw_data_path, config=config)
    target_col = config.columns.target_column

    # 1. Outer Split (80% Development, 20% Final Test)
    test_size = config.splits.test_size
    seed = config.splits.random_state
    print(f"\n1. Generating Outer Split (test_size={test_size:.0%}, seed={seed})...")
    dev_idx, test_idx = make_outer_split(
        df_raw,
        target_col=target_col,
        test_size=test_size,
        random_state=seed,
    )

    df_dev = df_raw.loc[dev_idx]
    df_test = df_raw.loc[test_idx]

    dev_default_rate = df_dev[target_col].mean()
    test_default_rate = df_test[target_col].mean()

    print(f"   -> Development set: {len(df_dev):,} samples (default rate: {dev_default_rate:.2%})")
    print(f"   -> Final Test set:   {len(df_test):,} samples (default rate: {test_default_rate:.2%})")

    # 2. Reserve a fixed SHAP reference set before creating CV folds.
    ref_size = config.splits.shap_reference_size
    ref_seed = config.splits.shap_reference_seed
    print(f"\n2. Sampling SHAP Reference Set ({ref_size} samples from Dev, seed={ref_seed})...")
    shap_ref_idx = sample_shap_reference(
        df_dev,
        target_col=target_col,
        n_samples=ref_size,
        random_state=ref_seed,
    )
    df_ref = df_dev.loc[shap_ref_idx]
    ref_default_rate = df_ref[target_col].mean()
    print(f"   -> SHAP Reference: {len(shap_ref_idx):,} samples (default rate: {ref_default_rate:.2%})")

    # 3. Core is the remainder of Development; only Core enters model evaluation.
    core_idx = make_core_indices(dev_idx, shap_ref_idx)
    df_core = df_dev.loc[core_idx]
    core_default_rate = df_core[target_col].mean()
    print(f"\n3. Core: {len(df_core):,} samples (default rate: {core_default_rate:.2%})")

    # 4. Fixed stratified folds on Core, with no reference or Test observations.
    n_splits = config.splits.cv_n_splits
    print(f"\n4. Generating {n_splits}-fold Stratified CV on Core (seed={seed})...")
    cv_folds = make_cv_folds(
        df_core,
        target_col=target_col,
        n_splits=n_splits,
        random_state=seed,
    )

    for fold in cv_folds:
        f_idx = fold["fold"]
        val_sub = df_core.loc[fold["val_indices"]]
        val_rate = val_sub[target_col].mean()
        print(f"   -> Fold {f_idx}: train={fold['n_train']:,}, val={fold['n_val']:,} (val default rate: {val_rate:.2%})")

    # 5. Save artifacts; save_splits validates all disjointness and fold invariants.
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_samples": len(df_raw),
        "dev_samples": len(dev_idx),
        "core_samples": len(core_idx),
        "test_samples": len(test_idx),
        "test_size": test_size,
        "n_cv_folds": n_splits,
        "shap_reference_size": ref_size,
        "random_state": seed,
        "shap_reference_seed": ref_seed,
        "overall_default_rate": float(df_raw[target_col].mean()),
        "dev_default_rate": float(dev_default_rate),
        "core_default_rate": float(core_default_rate),
        "test_default_rate": float(test_default_rate),
        "shap_reference_default_rate": float(ref_default_rate),
    }

    out_dir = Path(config.splits_dir)
    print(f"\nSaving split files to: {out_dir.resolve()}...")
    save_splits(
        output_dir=out_dir,
        dev_indices=dev_idx,
        test_indices=test_idx,
        cv_folds=cv_folds,
        shap_reference_indices=shap_ref_idx,
        metadata=metadata,
    )

    print("All split files successfully generated and saved!")


if __name__ == "__main__":
    main()
