"""Guard trial validation so invalid objectives cannot enter a Pareto study."""

import math

import pytest

from creditrisk.hpo import _check_result


def _valid_result():
    return {
        "roc_auc_cv": 0.78,
        "shap_stability": 0.91,
        "top_k_jaccard": 0.7,
        "fold_roc_auc": [0.77, 0.79, 0.78, 0.80, 0.76],
    }


def test_hpo_accepts_five_finite_fold_scores():
    _check_result(_valid_result(), n_folds=5)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("roc_auc_cv", math.nan),
        ("shap_stability", math.inf),
        ("shap_stability", 1.1),
        ("top_k_jaccard", -0.1),
        ("fold_roc_auc", [0.77, 0.79]),
        ("fold_roc_auc", [0.77, 0.79, 0.78, math.nan, 0.76]),
    ],
)
def test_hpo_rejects_incomplete_or_invalid_objectives(field, bad_value):
    result = _valid_result()
    result[field] = bad_value
    with pytest.raises(ValueError):
        _check_result(result, n_folds=5)
