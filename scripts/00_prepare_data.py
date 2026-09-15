"""Script to validate raw dataset and generate processed summary."""

import json
from pathlib import Path
import sys

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.config import load_data_config
from creditrisk.data import get_dataset_summary, load_raw_data, standardize_columns


def main():
    print("==================================================")
    print("STEP 1: Validating Raw Credit Card Default Dataset")
    print("==================================================")

    config = load_data_config("configs/data.yaml")
    raw_path = Path(config.raw_data_path)

    if not raw_path.exists():
        print(f"Error: Raw data file does not exist at '{raw_path.resolve()}'")
        sys.exit(1)

    print(f"Loading raw data from: {raw_path}")
    df_raw = load_raw_data(raw_path, config=config)
    print(f"Raw shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    summary = get_dataset_summary(df_raw, config=config)
    print("\n--- Dataset Summary ---")
    print(f"Total observations: {summary['n_samples']:,}")
    print(f"Total features:     {summary['n_features']}")
    print(f"Default rate (y=1): {summary['default_rate']:.2%}")
    print(f"Non-default (y=0):  {summary['non_default_rate']:.2%}")
    print(f"Class counts:       {summary['class_counts']}")

    # Save summary report in data/processed
    out_dir = Path(config.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {summary_file}")
    print("Raw data validation completed successfully!")


if __name__ == "__main__":
    main()
