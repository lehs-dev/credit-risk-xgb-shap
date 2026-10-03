.PHONY: help install data prepare split baseline figures notebooks test clean

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP ?= $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  make install   - Install required packages into $(VENV)"
	@echo "  make data      - Check the existing CSV in data/raw/"
	@echo "  make prepare   - Validate raw dataset & print summary statistics"
	@echo "  make split     - Generate 80/20 split, SHAP reference and 5-fold CV on Core"
	@echo "  make baseline  - Run 6 baseline models benchmark"
	@echo "  make figures   - Export Chapter 3 PNG/PDF figures from saved artifacts"
	@echo "  make notebooks - Execute Chapter 3 notebooks into artifacts/notebooks/"
	@echo "  make test      - Run automated unit and anti-leakage tests"
	@echo "  make clean     - Remove Python bytecode and test cache files"

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

data:
	@if [ -f "data/raw/default_credit_card.csv" ]; then \
		echo "Using data/raw/default_credit_card.csv"; \
	else \
		echo "Place your UCI CSV at data/raw/default_credit_card.csv"; exit 1; \
	fi

prepare: data
	$(PYTHON) scripts/00_prepare_data.py

split:
	$(PYTHON) scripts/01_make_splits.py

baseline:
	$(PYTHON) scripts/02_run_baselines.py

figures:
	$(PYTHON) scripts/09_generate_figures.py

notebooks:
	$(PYTHON) scripts/10_execute_notebooks.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
