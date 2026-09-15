"""Unit tests for SHAP stability metrics."""

import numpy as np
import pytest

from creditrisk.stability import (
    calculate_global_shap_importance,
    compute_kendall_w,
    compute_mean_spearman_rank_stability,
    compute_top_k_jaccard_stability,
)


def test_calculate_global_shap_importance():
    shap_matrix = np.array(
        [
            [-1.0, 2.0, 0.0],
            [3.0, -2.0, 1.0],
        ]
    )
    # abs:
    # [1.0, 2.0, 0.0]
    # [3.0, 2.0, 1.0]
    # mean: [2.0, 2.0, 0.5]
    imp = calculate_global_shap_importance(shap_matrix)
    np.testing.assert_allclose(imp, np.array([2.0, 2.0, 0.5]))


def test_spearman_stability_identical_rankings():
    # 3 runs, 4 features with identical ordering
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.5, 1.0, 1.5, 2.0],
        ]
    )
    stab = compute_mean_spearman_rank_stability(matrix)
    assert np.isclose(stab, 1.0)


def test_spearman_stability_reversed_rankings():
    # 2 runs with exact opposite rankings
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    stab = compute_mean_spearman_rank_stability(matrix)
    assert np.isclose(stab, -1.0)


def test_top_k_jaccard_stability():
    # 2 runs, top-2 features
    # Run 0 top-2: features 2, 3 (values 3.0, 4.0)
    # Run 1 top-2: features 2, 3 (values 3.0, 4.0)
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    jaccard = compute_top_k_jaccard_stability(matrix, k=2)
    assert np.isclose(jaccard, 1.0)

    # Disjoint top-2
    # Run 0 top-2: features 2, 3
    # Run 1 top-2: features 0, 1
    matrix_disjoint = np.array(
        [
            [0.1, 0.2, 0.9, 0.8],
            [0.9, 0.8, 0.1, 0.2],
        ]
    )
    jaccard_disjoint = compute_top_k_jaccard_stability(matrix_disjoint, k=2)
    assert np.isclose(jaccard_disjoint, 0.0)


def test_kendall_w_concordance():
    # Perfect concordance
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    w = compute_kendall_w(matrix)
    assert np.isclose(w, 1.0)
