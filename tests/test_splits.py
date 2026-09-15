"""Unit tests for dataset splitting and stratification."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from creditrisk.config import DataConfig
from creditrisk.splits import (
    load_splits,
    make_cv_folds,
    make_outer_split,
    sample_shap_reference,
    save_splits,
)


@pytest.fixture
def mock_dataset():
    """Create a mock credit dataset with identical schema and ~22% default rate."""
    np.random.seed(42)
    n = 1000
    y = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": np.random.randint(10000, 500000, size=n),
        "SEX": np.random.choice([1, 2], size=n),
        "EDUCATION": np.random.choice([1, 2, 3, 4], size=n),
        "MARRIAGE": np.random.choice([1, 2, 3], size=n),
        "AGE": np.random.randint(20, 70, size=n),
        "default.payment.next.month": y,
    }
    return pd.DataFrame(data)


def test_outer_split_disjoint_and_stratified(mock_dataset):
    target_col = "default.payment.next.month"
    dev_idx, test_idx = make_outer_split(
        mock_dataset, target_col=target_col, test_size=0.20, random_state=42
    )

    # 1. No overlap
    assert len(set(dev_idx).intersection(set(test_idx))) == 0

    # 2. Total size matches
    assert len(dev_idx) + len(test_idx) == len(mock_dataset)
    assert len(test_idx) == int(len(mock_dataset) * 0.20)

    # 3. Stratification preserved
    overall_rate = mock_dataset[target_col].mean()
    dev_rate = mock_dataset.loc[dev_idx, target_col].mean()
    test_rate = mock_dataset.loc[test_idx, target_col].mean()

    assert abs(dev_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.02


def test_cv_folds_partition_and_disjoint(mock_dataset):
    target_col = "default.payment.next.month"
    dev_idx, _ = make_outer_split(mock_dataset, target_col=target_col, test_size=0.20, random_state=42)
    df_dev = mock_dataset.loc[dev_idx]

    folds = make_cv_folds(df_dev, target_col=target_col, n_splits=5, random_state=42)

    assert len(folds) == 5
    all_val_indices = []

    for f in folds:
        train_set = set(f["train_indices"])
        val_set = set(f["val_indices"])

        # Train and val in each fold are disjoint
        assert len(train_set.intersection(val_set)) == 0

        # Union covers all dev indices
        assert train_set.union(val_set) == set(df_dev.index)

        all_val_indices.extend(f["val_indices"])

    # Across all folds, each sample is validated exactly once
    assert len(all_val_indices) == len(df_dev)
    assert len(set(all_val_indices)) == len(df_dev)


def test_shap_reference_strictly_from_dev(mock_dataset):
    target_col = "default.payment.next.month"
    dev_idx, test_idx = make_outer_split(mock_dataset, target_col=target_col, test_size=0.20, random_state=42)
    df_dev = mock_dataset.loc[dev_idx]

    ref_idx = sample_shap_reference(df_dev, target_col=target_col, n_samples=100, random_state=42)

    assert len(ref_idx) == 100
    # Must be subset of dev
    assert set(ref_idx).issubset(set(dev_idx))
    # Must NOT overlap with test set
    assert len(set(ref_idx).intersection(set(test_idx))) == 0


def test_save_and_load_splits_roundtrip(mock_dataset, tmp_path):
    target_col = "default.payment.next.month"
    dev_idx, test_idx = make_outer_split(mock_dataset, target_col=target_col, test_size=0.20, random_state=42)
    df_dev = mock_dataset.loc[dev_idx]
    folds = make_cv_folds(df_dev, target_col=target_col, n_splits=5, random_state=42)
    ref_idx = sample_shap_reference(df_dev, target_col=target_col, n_samples=100, random_state=42)

    save_splits(
        output_dir=tmp_path,
        dev_indices=dev_idx,
        test_indices=test_idx,
        cv_folds=folds,
        shap_reference_indices=ref_idx,
    )

    loaded = load_splits(tmp_path)
    np.testing.assert_array_equal(loaded["dev_indices"], dev_idx)
    np.testing.assert_array_equal(loaded["test_indices"], test_idx)
    np.testing.assert_array_equal(loaded["shap_reference_indices"], ref_idx)
    assert len(loaded["cv_folds"]) == 5
