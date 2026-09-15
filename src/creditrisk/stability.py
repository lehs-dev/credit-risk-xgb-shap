"""Mathematical metrics for quantifying SHAP explanation stability."""

from itertools import combinations
from typing import List, Union
import numpy as np
from scipy.stats import rankdata, spearmanr


def calculate_global_shap_importance(shap_values: np.ndarray) -> np.ndarray:
    """
    Calculate global mean absolute SHAP importance vector across reference instances.
    I_j = (1 / N) * sum_{i=1}^N |phi_{ij}|

    Parameters
    ----------
    shap_values : np.ndarray
        Array of shape (N, M) where N is number of instances and M is number of features.

    Returns
    -------
    np.ndarray
        Vector of shape (M,) containing mean absolute importance per feature.
    """
    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2D array of SHAP values (N, M), got shape {shap_values.shape}")
    return np.mean(np.abs(shap_values), axis=0)


def compute_mean_spearman_rank_stability(importance_matrix: np.ndarray) -> float:
    """
    Compute average pairwise Spearman rank correlation across R repeated runs.
    S(theta) = 2 / [R * (R - 1)] * sum_{a < b} rho_s(I^(a), I^(b))

    Parameters
    ----------
    importance_matrix : np.ndarray
        Array of shape (R, M) where R is number of repeated runs and M is number of features.

    Returns
    -------
    float
        Average Spearman rank correlation in [-1, 1]. Value close to 1 means highly stable.
    """
    R, M = importance_matrix.shape
    if R < 2:
        raise ValueError("At least 2 repeated runs are required to compute stability.")

    correlations = []
    for a, b in combinations(range(R), 2):
        corr, _ = spearmanr(importance_matrix[a], importance_matrix[b])
        # Handle constant ranking edge case
        if np.isnan(corr):
            corr = 0.0
        correlations.append(corr)

    return float(np.mean(correlations))


def compute_top_k_jaccard_stability(importance_matrix: np.ndarray, k: int = 5) -> float:
    """
    Compute average pairwise Jaccard similarity of the top-k most important features across R runs.
    J(A, B) = |A cap B| / |A cup B|

    Parameters
    ----------
    importance_matrix : np.ndarray
        Shape (R, M) of feature importances across R runs.
    k : int
        Top-k features to compare (default: 5).

    Returns
    -------
    float
        Average Jaccard index in [0, 1].
    """
    R, M = importance_matrix.shape
    if R < 2:
        raise ValueError("At least 2 repeated runs are required.")
    if k > M or k <= 0:
        raise ValueError(f"k must be between 1 and {M}, got {k}")

    # For each run, get set of indices of top-k features (highest importance)
    top_k_sets = [set(np.argsort(importance_matrix[r])[-k:]) for r in range(R)]

    jaccards = []
    for a, b in combinations(range(R), 2):
        set_a, set_b = top_k_sets[a], top_k_sets[b]
        jaccard = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        jaccards.append(jaccard)

    return float(np.mean(jaccards))


def compute_kendall_w(importance_matrix: np.ndarray) -> float:
    """
    Compute Kendall's W (Coefficient of Concordance) across R repeated runs (judges)
    evaluating M features (objects).

    W = 12 * S / [R^2 * (M^3 - M)]
    where S is sum of squared deviations of feature rank sums from the mean rank sum.

    Parameters
    ----------
    importance_matrix : np.ndarray
        Shape (R, M).

    Returns
    -------
    float
        Kendall's W in [0, 1]. 1 means complete agreement across runs, 0 means no agreement.
    """
    R, M = importance_matrix.shape
    if R < 2 or M < 2:
        return 1.0

    # Rank features within each run (higher importance = higher rank)
    ranks = np.zeros((R, M))
    for r in range(R):
        ranks[r] = rankdata(importance_matrix[r])

    # Sum of ranks for each feature across R runs
    rank_sums = np.sum(ranks, axis=0)
    mean_rank_sum = R * (M + 1) / 2.0
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    W = 12.0 * S / (R**2 * (M**3 - M))
    return float(np.clip(W, 0.0, 1.0))
