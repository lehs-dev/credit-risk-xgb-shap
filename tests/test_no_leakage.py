"""Verification tests for strict anti-leakage guarantees."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.config import load_data_config
from creditrisk.preprocessing import CleanCategoricalTransformer, CreditRiskPreprocessor
from creditrisk.splits import make_cv_folds, make_outer_split, sample_shap_reference


@pytest.fixture
def sample_features_df():
    """Create sample tabular data with categorical and numerical values."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "LIMIT_BAL": np.random.uniform(10000, 100000, n),
            "SEX": np.random.choice([1, 2], n),
            "EDUCATION": np.random.choice([0, 1, 2, 3, 4, 5, 6], n),
            "MARRIAGE": np.random.choice([0, 1, 2, 3], n),
            "AGE": np.random.randint(20, 60, n),
            "PAY_0": np.random.choice([-1, 0, 1, 2], n),
            "PAY_2": np.random.choice([-1, 0, 1, 2], n),
            "PAY_3": np.random.choice([-1, 0, 1, 2], n),
            "PAY_4": np.random.choice([-1, 0, 1, 2], n),
            "PAY_5": np.random.choice([-1, 0, 1, 2], n),
            "PAY_6": np.random.choice([-1, 0, 1, 2], n),
            "BILL_AMT1": np.random.uniform(0, 50000, n),
            "BILL_AMT2": np.random.uniform(0, 50000, n),
            "BILL_AMT3": np.random.uniform(0, 50000, n),
            "BILL_AMT4": np.random.uniform(0, 50000, n),
            "BILL_AMT5": np.random.uniform(0, 50000, n),
            "BILL_AMT6": np.random.uniform(0, 50000, n),
            "PAY_AMT1": np.random.uniform(0, 10000, n),
            "PAY_AMT2": np.random.uniform(0, 10000, n),
            "PAY_AMT3": np.random.uniform(0, 10000, n),
            "PAY_AMT4": np.random.uniform(0, 10000, n),
            "PAY_AMT5": np.random.uniform(0, 10000, n),
            "PAY_AMT6": np.random.uniform(0, 10000, n),
            "default.payment.next.month": np.random.choice([0, 1], n, p=[0.78, 0.22]),
        }
    )


def test_scaler_parameters_not_affected_by_val_set(sample_features_df):
    """Test that preprocessor learned parameters are strictly derived from training data."""
    train_df = sample_features_df.iloc[:150].copy()
    val_df = sample_features_df.iloc[150:].copy()

    # Preprocessor fitted ONLY on train
    preprocessor = CreditRiskPreprocessor(scale_numerical=True, scaler_type="standard")
    train_transformed = preprocessor.fit_transform(train_df)

    # Extract learned scaler mean for LIMIT_BAL
    # Now artificially inject extreme values into val_df
    val_df_corrupted = val_df.copy()
    val_df_corrupted["LIMIT_BAL"] = val_df_corrupted["LIMIT_BAL"] * 1000

    # Transforming val_df_corrupted should NOT change preprocessor internal state
    val_transformed_1 = preprocessor.transform(val_df)
    val_transformed_corrupted = preprocessor.transform(val_df_corrupted)

    # The transformation of training data should remain exactly identical
    train_transformed_again = preprocessor.transform(train_df)
    pd.testing.assert_frame_equal(train_transformed, train_transformed_again)


def test_categorical_cleaner_unknown_mapping():
    """Verify unknown category codes (0, 5, 6 for EDUCATION and 0 for MARRIAGE) are cleanly mapped."""
    df = pd.DataFrame(
        {
            "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
            "MARRIAGE": [0, 1, 2, 3, 0, 1, 2],
        }
    )
    cleaner = CleanCategoricalTransformer(
        education_unknowns=[0, 5, 6],
        education_replacement=4,
        marriage_unknowns=[0],
        marriage_replacement=3,
    )
    cleaned = cleaner.transform(df)

    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["MARRIAGE"].unique()).issubset({1, 2, 3})


def test_shap_reference_leakage_guard(sample_features_df):
    """Verify no sample in SHAP reference set can ever originate from Test set."""
    dev_idx, test_idx = make_outer_split(
        sample_features_df,
        target_col="default.payment.next.month",
        test_size=0.20,
        random_state=42,
    )
    df_dev = sample_features_df.loc[dev_idx]

    shap_ref_idx = sample_shap_reference(
        df_dev,
        target_col="default.payment.next.month",
        n_samples=50,
        random_state=42,
    )

    leakage = set(shap_ref_idx).intersection(set(test_idx))
    assert len(leakage) == 0, f"Detected leakage in SHAP reference set: {leakage}"
