#!/usr/bin/env bash
set -e

VENV_PYTHON=".venv/bin/python"
VENV_PYTEST=".venv/bin/pytest"

case "$1" in
  data)
    if [ -f "data/raw/default_credit_card.csv" ]; then
      echo "[INFO] Using data/raw/default_credit_card.csv"
    else
      echo "[ERROR] Place your UCI CSV at data/raw/default_credit_card.csv"; exit 1
    fi
    ;;
  prepare)
    $VENV_PYTHON scripts/00_prepare_data.py
    ;;
  split)
    $VENV_PYTHON scripts/01_make_splits.py
    ;;
  baseline)
    $VENV_PYTHON scripts/02_run_baselines.py
    ;;
  test)
    $VENV_PYTEST tests/ -v
    ;;
  clean)
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    echo "[SUCCESS] Cleaned cache files."
    ;;
  *)
    echo "Usage: ./run.sh {data|prepare|split|baseline|test|clean}"
    exit 1
    ;;
esac
