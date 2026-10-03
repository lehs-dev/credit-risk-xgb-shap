#!/usr/bin/env python3
"""Execute Chapter 3 analysis notebooks without rerunning model experiments.

Executed copies go to artifacts/notebooks by default. Source notebooks remain
small and reviewable; --inplace is available for an explicit interactive check.
Existing tables, SHAP, predictions, Optuna storage, models, raw data and splits
are hashed before and after execution to detect accidental experiment writes.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAMES = (
    "04_baseline_benchmarking.ipynb",
    "05_xgb_hpo_and_pareto_analysis.ipynb",
    "06_sensitivity_and_final_evaluation.ipynb",
)
PROTECTED_DIRS = (
    "artifacts/tables", "artifacts/shap", "artifacts/predictions",
    "artifacts/optuna", "artifacts/models", "data/raw", "data/splits",
    "configs",
)


def artifact_snapshot() -> dict[str, str]:
    """Include names as well as content, so new/deleted artifacts are detected."""
    snapshot = {}
    for directory in PROTECTED_DIRS:
        for path in sorted((PROJECT_ROOT / directory).rglob("*")):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                snapshot[str(path.relative_to(PROJECT_ROOT))] = digest.hexdigest()
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/notebooks")
    parser.add_argument("--inplace", action="store_true", help="Save executed outputs in source notebooks.")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum seconds per code cell.")
    parser.add_argument("--notebook", choices=NOTEBOOK_NAMES, action="append", help="Select one or more Chapter 3 notebooks.")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        if find_spec("ipykernel") is None:
            raise ImportError("No module named 'ipykernel'")
    except ImportError as exc:
        print("Notebook execution requires nbformat, nbclient and ipykernel. "
              "Install the project's development requirements.", file=sys.stderr)
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mpl-cache"))
    snapshot = artifact_snapshot()
    with tempfile.TemporaryDirectory(prefix="creditrisk-ch3-kernel-") as kernel_dir:
        kernel_name = "creditrisk-ch3"
        kernel_path = Path(kernel_dir) / kernel_name
        kernel_path.mkdir()
        (kernel_path / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "CreditRisk Chapter 3", "language": "python",
        }))
        kernel_specs = KernelSpecManager(kernel_dirs=[kernel_dir])
        for name in args.notebook or NOTEBOOK_NAMES:
            source = PROJECT_ROOT / "notebooks" / name
            notebook = nbformat.read(source, as_version=4)
            nbformat.validate(notebook)
            print(f"Executing {name} (analysis of saved artifacts only)", flush=True)
            manager = KernelManager(
                kernel_name=kernel_name, kernel_spec_manager=kernel_specs,
                transport="ipc" if os.name == "posix" else "tcp",
            )
            try:
                NotebookClient(
                    notebook, km=manager, timeout=args.timeout, kernel_name=kernel_name,
                    allow_errors=False, resources={"metadata": {"path": str(PROJECT_ROOT)}},
                ).execute()
            finally:
                # Externally supplied managers are not shut down by nbclient.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                current = artifact_snapshot()
                if current != snapshot:
                    changed = sorted(key for key in snapshot.keys() | current.keys()
                                     if snapshot.get(key) != current.get(key))
                    raise RuntimeError("Notebook changed protected experimental artifacts: " + ", ".join(changed))
            destination = source if args.inplace else output_dir / name
            nbformat.write(notebook, destination)
            cells = sum(cell.cell_type == "code" for cell in notebook.cells)
            print(f"Verified {cells} code cells; saved {destination}", flush=True)
    print("All requested notebooks executed successfully; experimental artifacts are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
