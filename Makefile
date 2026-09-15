.PHONY: help install data prepare split baseline test clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest

help:
	@echo "Available commands:"
	@echo "  make install   - Install required packages into $(VENV)"
	@echo "  make data      - Ingest raw dataset into data/raw/"
	@echo "  make prepare   - Validate raw dataset & print summary statistics"
	@echo "  make split     - Generate outer 80/20 split, 5-fold CV and SHAP reference set"
	@echo "  make baseline  - Run 6 baseline models benchmark"
	@echo "  make test      - Run automated unit and anti-leakage tests"
	@echo "  make clean     - Remove Python bytecode and test cache files"

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

data:
	@mkdir -p data/raw
	@if [ -f "../data-set /UCI_Credit_Card.csv" ]; then \
		cp "../data-set /UCI_Credit_Card.csv" data/raw/default_credit_card.csv; \
		echo "Dataset successfully copied to data/raw/default_credit_card.csv"; \
	elif [ -f "data/raw/default_credit_card.csv" ]; then \
		echo "data/raw/default_credit_card.csv already exists."; \
	else \
		echo "Error: UCI_Credit_Card.csv not found in ../data-set /"; exit 1; \
	fi

prepare: data
	$(PYTHON) scripts/00_prepare_data.py

split:
	$(PYTHON) scripts/01_make_splits.py

baseline:
	$(PYTHON) scripts/02_run_baselines.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
