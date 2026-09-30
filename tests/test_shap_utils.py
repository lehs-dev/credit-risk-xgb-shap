"""Checks for the SHAP values and original-feature aggregation used by objective 2."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from creditrisk.shap_utils import group_shap_by_original_feature, xgboost_tree_shap_values
from creditrisk.stability import calculate_global_shap_importance


def test_group_one_hot_contributions_before_absolute_value():
    encoded = np.array([[1.0, -2.0, 3.0], [-1.0, 1.0, -2.0]])
    grouped = group_shap_by_original_feature(
        encoded,
        ["EDUCATION_1", "EDUCATION_2", "AGE"],
        ["EDUCATION", "AGE"],
        ["EDUCATION"],
    )
    np.testing.assert_allclose(grouped, [[-1.0, 3.0], [0.0, -2.0]])
    np.testing.assert_allclose(calculate_global_shap_importance(grouped), [0.5, 2.5])
    np.testing.assert_allclose(grouped.sum(axis=1), encoded.sum(axis=1))


def test_xgboost_tree_shap_shape_and_finite_values():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = ((X["a"] + X["b"]) > 0).astype(int)
    model = XGBClassifier(n_estimators=8, max_depth=2, random_state=42, n_jobs=1)
    model.fit(X, y)

    values = xgboost_tree_shap_values(model, X.iloc[:12])
    assert values.shape == (12, 3)
    assert np.isfinite(values).all()
