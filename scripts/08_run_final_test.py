"""One-shot held-out Test evaluation of five configurations locked on Development.

Use --dry-run to validate all locks without parsing Test labels. The default
command fits five models on all Development and scores Test once per model.
Use --verify to check completed outputs; --finalize repairs a started
manifest only when an atomic, complete summary already exists.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.final_test import run_final_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs; do not parse Test labels or fit models.")
    mode.add_argument("--verify", action="store_true", help="Verify a completed run; do not refit or score.")
    mode.add_argument("--finalize", action="store_true", help="Finalize an already written summary; never refit or score.")
    args = parser.parse_args()
    run_final_test(dry_run=args.dry_run, verify=args.verify, finalize=args.finalize)


if __name__ == "__main__":
    main()
