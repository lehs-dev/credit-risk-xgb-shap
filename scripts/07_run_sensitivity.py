"""Check fixed XGBoost candidates across ten new Development Core CV partitions.

Default: 10 seeds × 5 locked candidates × 5 folds = 250 model fits.
Use --dry-run to validate inputs without fitting, or --smoke for one candidate
on one seed (5 fits) with a separate output prefix.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.sensitivity import run_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate locks, data, and folds; fit no models.")
    mode.add_argument("--smoke", action="store_true", help="Run one seed and one candidate (five fits).")
    args = parser.parse_args()
    run_sensitivity(dry_run=args.dry_run, smoke=args.smoke)


if __name__ == "__main__":
    main()
