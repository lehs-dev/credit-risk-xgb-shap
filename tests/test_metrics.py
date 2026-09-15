"""Unit tests for predictive performance metrics."""

import numpy as np
import pytest

from creditrisk.metrics import compute_classification_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_compute_metrics_imbalanced_edge_case():
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.random.uniform(0, 1, size=100)

    metrics = compute_classification_metrics(y_true, y_prob)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1" in metrics
    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
