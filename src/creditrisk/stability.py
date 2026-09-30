"""Mathematical metrics for quantifying SHAP explanation stability."""

from itertools import combinations
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


def _validate_importance_matrix(importance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(importance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("Importance matrix must contain at least two runs and two features.")
    if not np.isfinite(matrix).all():
        raise ValueError("Importance matrix must contain only finite values.")
    return matrix


def compute_mean_spearman_rank_stability(importance_matrix: np.ndarray) -> float:
    """Average pairwise Spearman correlation of global importance rankings.

    SciPy assigns average ranks to tied values. A constant importance vector
    has no defined rank correlation, so pairs containing one score 0 rather
    than being treated as perfectly stable.
    """
    matrix = _validate_importance_matrix(importance_matrix)
    correlations = []
    for a, b in combinations(range(matrix.shape[0]), 2):
        if np.all(matrix[a] == matrix[a, 0]) or np.all(matrix[b] == matrix[b, 0]):
            correlations.append(0.0)
        else:
            correlations.append(float(spearmanr(matrix[a], matrix[b]).statistic))
    return float(np.mean(correlations))


def compute_top_k_jaccard_stability(importance_matrix: np.ndarray, k: int = 5) -> float:
    """Mean pairwise top-k Jaccard, breaking boundary ties by feature order."""
    matrix = _validate_importance_matrix(importance_matrix)
    n_features = matrix.shape[1]
    if not 1 <= k <= n_features:
        raise ValueError(f"k must be between 1 and {n_features}, got {k}")

    feature_order = np.arange(n_features)
    top_k_sets = [
        set(np.lexsort((feature_order, -row))[:k]) for row in matrix
    ]
    jaccards = []
    for a, b in combinations(range(matrix.shape[0]), 2):
        if np.all(matrix[a] == matrix[a, 0]) or np.all(matrix[b] == matrix[b, 0]):
            jaccards.append(0.0)
            continue
        first, second = top_k_sets[a], top_k_sets[b]
        jaccards.append(len(first.intersection(second)) / len(first.union(second)))
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
