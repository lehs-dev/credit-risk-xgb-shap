"""Run two-objective NSGA-II search for CV ROC-AUC and SHAP rank stability."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.hpo import run_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Use the separate pilot study.")
    parser.add_argument("--target-trials", type=int, help="Target number of COMPLETE trials.")
    args = parser.parse_args()
    run_search("multi", pilot=args.pilot, target_trials=args.target_trials)


if __name__ == "__main__":
    main()
