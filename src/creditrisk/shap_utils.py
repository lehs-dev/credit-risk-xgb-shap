"""TreeSHAP calculations on a fixed reference set for XGBoost classifiers."""

from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


def xgboost_tree_shap_values(
    model: XGBClassifier, X_reference: pd.DataFrame
) -> np.ndarray:
    """Return exact TreeSHAP values on the raw log-odds margin.

    XGBoost's final contribution column is the bias term and is excluded here.
    Tree-path-dependent expectations are used; no separate background set is
    supplied. The reference rows are the observations being explained.
    """
    if not isinstance(X_reference, pd.DataFrame) or X_reference.empty:
        raise ValueError("X_reference must be a nonempty pandas DataFrame.")
    if X_reference.columns.has_duplicates:
        raise ValueError("Reference feature names must be unique.")

    booster = model.get_booster()
    reference_matrix = xgb.DMatrix(X_reference, feature_names=list(X_reference.columns))
    contributions = booster.predict(
        reference_matrix,
        pred_contribs=True,
        approx_contribs=False,
        strict_shape=True,
    )
    expected_shape = (len(X_reference), 1, X_reference.shape[1] + 1)
    if contributions.shape != expected_shape:
        raise ValueError(f"Expected SHAP contribution shape {expected_shape}, got {contributions.shape}.")

    raw_margin = booster.predict(reference_matrix, output_margin=True, strict_shape=True)
    if not np.allclose(contributions.sum(axis=-1), raw_margin, rtol=1e-5, atol=1e-4):
        raise ValueError("TreeSHAP contributions do not sum to the raw model margin.")
    return np.asarray(contributions[:, 0, :-1], dtype=float)


def group_shap_by_original_feature(
    shap_values: np.ndarray,
    encoded_feature_names: Sequence[str],
    original_feature_names: Sequence[str],
    categorical_features: Sequence[str],
) -> np.ndarray:
    """Sum one-hot SHAP contributions per row before taking absolute values.

    The output always has one column per original predictor, in the supplied
    original-feature order. Unknown or missing columns fail explicitly.
    """
    values = np.asarray(shap_values, dtype=float)
    encoded_names = list(encoded_feature_names)
    original_names = list(original_feature_names)
    categorical_names = set(categorical_features)
    if values.ndim != 2 or values.shape[1] != len(encoded_names):
        raise ValueError("SHAP columns and encoded feature names must match.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite.")
    if len(set(encoded_names)) != len(encoded_names) or len(set(original_names)) != len(original_names):
        raise ValueError("Feature names must be unique.")
    if not categorical_names.issubset(original_names):
        raise ValueError("Categorical features must be original predictors.")

    original_positions = {name: index for index, name in enumerate(original_names)}
    grouped = np.zeros((values.shape[0], len(original_names)), dtype=float)
    seen = set()
    for column, encoded_name in enumerate(encoded_names):
        if encoded_name in original_positions:
            original_name = encoded_name
        else:
            matches = [
                name for name in categorical_names if encoded_name.startswith(f"{name}_")
            ]
            if len(matches) != 1:
                raise ValueError(f"Cannot map encoded feature '{encoded_name}' to an original predictor.")
            original_name = matches[0]
        grouped[:, original_positions[original_name]] += values[:, column]
        seen.add(original_name)

    if seen != set(original_names):
        raise ValueError(f"Original predictors missing from encoded data: {set(original_names) - seen}")
    return grouped
