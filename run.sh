#!/usr/bin/env bash
set -e

VENV_PYTHON=".venv/bin/python"
VENV_PYTEST=".venv/bin/pytest"

case "$1" in
  data)
    mkdir -p data/raw
    if [ -f "../data-set /UCI_Credit_Card.csv" ]; then
      cp "../data-set /UCI_Credit_Card.csv" data/raw/default_credit_card.csv
      echo "[SUCCESS] Copied dataset to data/raw/default_credit_card.csv"
    elif [ -f "data/raw/default_credit_card.csv" ]; then
      echo "[INFO] data/raw/default_credit_card.csv already exists."
    else
      echo "[ERROR] UCI_Credit_Card.csv not found in ../data-set /"; exit 1
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
