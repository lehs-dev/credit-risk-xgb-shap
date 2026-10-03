#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

case "${1:-}" in
  data)
    if [ -f "data/raw/default_credit_card.csv" ]; then
      echo "[INFO] Using data/raw/default_credit_card.csv"
    else
      echo "[ERROR] Place your UCI CSV at data/raw/default_credit_card.csv"; exit 1
    fi
    ;;
  prepare)
    "$PYTHON" scripts/00_prepare_data.py
    ;;
  split)
    "$PYTHON" scripts/01_make_splits.py
    ;;
  baseline)
    "$PYTHON" scripts/02_run_baselines.py
    ;;
  figures)
    "$PYTHON" scripts/09_generate_figures.py
    ;;
  notebooks)
    "$PYTHON" scripts/10_execute_notebooks.py
    ;;
  test)
    "$PYTHON" -m pytest tests/ -v
    ;;
  clean)
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    echo "[SUCCESS] Cleaned cache files."
    ;;
  *)
    echo "Usage: ./run.sh {data|prepare|split|baseline|figures|notebooks|test|clean}"
    exit 1
    ;;
esac
