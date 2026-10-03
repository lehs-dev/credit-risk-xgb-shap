"""Export Chapter 3 PNG/PDF figures from saved artifacts without model fitting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from creditrisk.visualization import generate_chapter3_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/figures")
    args = parser.parse_args()
    manifest = generate_chapter3_figures(ROOT, args.output_dir)
    print(f"Saved {len(manifest['figures'])} PNG/PDF artifacts to {args.output_dir}")
    print("ROC/PR curves unavailable: individual probabilities were not saved; figure 06 uses saved metrics and confusion counts.")
    print(f"Provenance and limitations: {args.output_dir / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
