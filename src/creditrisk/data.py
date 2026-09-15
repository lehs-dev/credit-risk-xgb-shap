"""Data loading, validation, and schema definitions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from creditrisk.config import DataConfig, load_data_config


def load_raw_data(
    file_path: Union[str, Path] = "data/raw/default_credit_card.csv",
    config: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """
    Load raw UCI Credit Card Default dataset.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to raw CSV file.
    config : Optional[DataConfig]
        Optional DataConfig instance.

    Returns
    -------
    pd.DataFrame
        Loaded and validated raw DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    validate_raw_data(df, config=config)
    return df


def validate_raw_data(df: pd.DataFrame, config: Optional[DataConfig] = None) -> None:
    """
    Validate expected columns and minimal sanity checks for the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    config : Optional[DataConfig]
        Config specifying expected target and ID columns.
    """
    if config is None:
        config = load_data_config()

    target = config.columns.target_column
    if target not in df.columns:
        raise ValueError(
            f"Expected target column '{target}' not found. Available columns: {df.columns.tolist()}"
        )

    # Check ID column if present
    id_col = config.columns.id_column
    if id_col in df.columns:
        if df[id_col].nunique() != len(df):
            raise ValueError(f"ID column '{id_col}' is not strictly unique.")

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        non_zero = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Unexpected missing values detected: {non_zero}")

    # Check expected rows and target values
    unique_targets = set(df[target].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column '{target}' contains invalid classes: {unique_targets}")


def standardize_columns(
    df: pd.DataFrame,
    config: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """
    Standardize target column name from 'default.payment.next.month' to clean format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : Optional[DataConfig]
        Data configuration.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with standardized target column name.
    """
    if config is None:
        config = load_data_config()

    df = df.copy()
    target_raw = config.columns.target_column
    target_clean = config.columns.target_column_clean

    if target_raw in df.columns:
        df = df.rename(columns={target_raw: target_clean})

    return df


def extract_features_and_target(
    df: pd.DataFrame,
    config: Optional[DataConfig] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract predictor features X and target y.
    Drops the ID column and separates the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    config : Optional[DataConfig]
        Data configuration.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) tuple where X has all 23 features and y has binary labels {0, 1}.
    """
    if config is None:
        config = load_data_config()

    df_clean = standardize_columns(df, config=config)
    target_col = config.columns.target_column_clean
    id_col = config.columns.id_column

    drop_cols = [col for col in [id_col, target_col] if col in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)
    y = df_clean[target_col].astype(int)

    return X, y


def get_dataset_summary(df: pd.DataFrame, config: Optional[DataConfig] = None) -> Dict[str, Union[int, float, dict]]:
    """
    Produce summary metadata for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    config : Optional[DataConfig]
        Data configuration.

    Returns
    -------
    dict
        Summary metrics including row count, class balance, and categorical distributions.
    """
    if config is None:
        config = load_data_config()

    df_clean = standardize_columns(df, config=config)
    target_col = config.columns.target_column_clean

    y = df_clean[target_col]
    val_counts = y.value_counts(normalize=True).to_dict()

    return {
        "n_samples": len(df_clean),
        "n_features": len(df_clean.columns) - 2 if config.columns.id_column in df_clean.columns else len(df_clean.columns) - 1,
        "default_rate": float(val_counts.get(1, 0.0)),
        "non_default_rate": float(val_counts.get(0, 0.0)),
        "class_counts": {int(k): int(v) for k, v in y.value_counts().to_dict().items()},
    }
