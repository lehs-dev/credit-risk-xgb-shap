"""Credit Risk Modeling with XGBoost, Multi-Objective HPO, and SHAP Stability."""

__version__ = "0.1.0"

from creditrisk.config import DataConfig, load_data_config
from creditrisk.data import extract_features_and_target, load_raw_data, standardize_columns
from creditrisk.metrics import compute_classification_metrics
from creditrisk.preprocessing import CleanCategoricalTransformer, CreditRiskPreprocessor
from creditrisk.splits import load_splits, make_cv_folds, make_outer_split, sample_shap_reference

__all__ = [
    "DataConfig",
    "load_data_config",
    "load_raw_data",
    "standardize_columns",
    "extract_features_and_target",
    "CleanCategoricalTransformer",
    "CreditRiskPreprocessor",
    "compute_classification_metrics",
    "make_outer_split",
    "make_cv_folds",
    "sample_shap_reference",
    "load_splits",
]
