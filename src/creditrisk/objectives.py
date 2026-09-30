"""Joint evaluation of predictive performance and SHAP ranking stability."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditrisk.config import DataConfig
from creditrisk.models import get_model
from creditrisk.preprocessing import CreditRiskPreprocessor
from creditrisk.shap_utils import group_shap_by_original_feature, xgboost_tree_shap_values
from creditrisk.splits import validate_split_protocol
from creditrisk.stability import (
    calculate_global_shap_importance,
    compute_mean_spearman_rank_stability,
    compute_top_k_jaccard_stability,
)


def evaluate_xgb_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Dict[str, Any],
    model_params: Dict[str, Any],
    config: DataConfig,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate both objectives using the same five fixed Core CV models.

    Each fold model is trained only on its Core training rows, scored on its
    validation rows, and explained on the same held-out SHAP reference rows.
    Thus the five different training subsets are the source of variation for
    the ranking-stability estimate. The final Test set is never accessed.
    """
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X and y must have identical, unique row indices.")
    if "random_state" not in model_params:
        raise ValueError("XGBoost random_state must be fixed across folds and trials.")
    original_features = config.columns.all_feature_columns
    if set(X.columns) != set(original_features):
        raise ValueError("X must contain exactly the 23 configured predictors.")
    validate_split_protocol(
        splits["dev_indices"],
        splits["test_indices"],
        splits["cv_folds"],
        splits["shap_reference_indices"],
    )

    X_reference = X.loc[splits["shap_reference_indices"]]
    fold_auc = []
    importance_rows = []
    for fold in splits["cv_folds"]:
        train_indices = fold["train_indices"]
        validation_indices = fold["val_indices"]
        preprocessor = CreditRiskPreprocessor(
            scale_numerical=False,
            one_hot_categorical=True,
            config=config,
        )
        X_train = preprocessor.fit_transform(X.loc[train_indices], y.loc[train_indices])
        X_validation = preprocessor.transform(X.loc[validation_indices])
        X_reference_transformed = preprocessor.transform(X_reference)

        model = get_model("xgboost", dict(model_params))
        model.fit(X_train, y.loc[train_indices])
        probabilities = model.predict_proba(X_validation)[:, 1]
        fold_auc.append(float(roc_auc_score(y.loc[validation_indices], probabilities)))

        encoded_shap = xgboost_tree_shap_values(model, X_reference_transformed)
        original_shap = group_shap_by_original_feature(
            encoded_shap,
            preprocessor.get_feature_names_out(),
            original_features,
            config.columns.categorical_features,
        )
        importance_rows.append(calculate_global_shap_importance(original_shap))

    importance_matrix = np.vstack(importance_rows)
    return {
        "roc_auc_cv": float(np.mean(fold_auc)),
        "shap_stability": compute_mean_spearman_rank_stability(importance_matrix),
        "top_k_jaccard": compute_top_k_jaccard_stability(importance_matrix, k=top_k),
        "fold_roc_auc": fold_auc,
        "importance_matrix": importance_matrix,
        "feature_names": original_features,
        "n_reference": len(X_reference),
        "n_repeats": len(splits["cv_folds"]),
    }
