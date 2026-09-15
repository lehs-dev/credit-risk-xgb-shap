"""Dataset splitting protocols and fixed reference set generation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from creditrisk.config import DataConfig, load_data_config


def make_outer_split(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the full dataset into Development (1 - test_size) and Final Test (test_size) sets
    using stratified sampling to preserve class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Target label column name.
    test_size : float
        Fraction reserved for final independent testing (default: 0.20).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (dev_indices, test_indices) arrays corresponding to df.index values.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[target_col].values
    dev_pos, test_pos = next(sss.split(df, y))

    dev_indices = df.index.values[dev_pos]
    test_indices = df.index.values[test_pos]

    return dev_indices, test_indices


def make_cv_folds(
    df_dev: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate fixed stratified K-fold cross-validation splits on the development set.

    Parameters
    ----------
    df_dev : pd.DataFrame
        Development set subset.
    target_col : str
        Target column name.
    n_splits : int
        Number of cross-validation folds (default: 5).
    random_state : int
        Random seed.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts containing fold index, train_indices, and val_indices.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_dev = df_dev[target_col].values

    folds = []
    for fold_idx, (train_pos, val_pos) in enumerate(skf.split(df_dev, y_dev)):
        train_indices = df_dev.index.values[train_pos].tolist()
        val_indices = df_dev.index.values[val_pos].tolist()
        folds.append(
            {
                "fold": int(fold_idx),
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "train_indices": train_indices,
                "val_indices": val_indices,
            }
        )

    return folds


def sample_shap_reference(
    df_dev: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    n_samples: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Extract a stratified reference sample strictly from the Development set.
    This reference set is held constant across all hyperparameter trials for TreeSHAP.

    Parameters
    ----------
    df_dev : pd.DataFrame
        Development set.
    target_col : str
        Target column name.
    n_samples : int
        Number of reference background samples (default: 1000).
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Array of indices from df_dev.index representing the SHAP reference set.
    """
    if n_samples >= len(df_dev):
        raise ValueError(f"Reference size ({n_samples}) must be smaller than dev size ({len(df_dev)}).")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_samples,
        random_state=random_state,
    )
    y_dev = df_dev[target_col].values
    ref_pos, _ = next(sss.split(df_dev, y_dev))

    return df_dev.index.values[ref_pos]


def save_splits(
    output_dir: Union[str, Path],
    dev_indices: np.ndarray,
    test_indices: np.ndarray,
    cv_folds: List[Dict[str, Any]],
    shap_reference_indices: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save all split artifacts to directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Outer split (dev vs test)
    np.savez_compressed(
        out_path / "outer_split.npz",
        dev_indices=dev_indices,
        test_indices=test_indices,
    )

    # 2. CV Folds
    with open(out_path / "cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(cv_folds, f, indent=2)

    # 3. SHAP Reference indices
    np.save(out_path / "shap_reference_indices.npy", shap_reference_indices)

    # 4. Metadata
    if metadata:
        with open(out_path / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def load_splits(splits_dir: Union[str, Path] = "data/splits") -> Dict[str, Any]:
    """
    Load all precomputed split artifacts.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: 'dev_indices', 'test_indices', 'cv_folds', 'shap_reference_indices'.
    """
    path = Path(splits_dir)

    outer_file = path / "outer_split.npz"
    folds_file = path / "cv_folds.json"
    shap_file = path / "shap_reference_indices.npy"

    if not (outer_file.exists() and folds_file.exists() and shap_file.exists()):
        raise FileNotFoundError(f"Missing one or more split files in {path.resolve()}")

    outer = np.load(outer_file)
    with open(folds_file, "r", encoding="utf-8") as f:
        cv_folds = json.load(f)
    shap_ref = np.load(shap_file)

    return {
        "dev_indices": outer["dev_indices"],
        "test_indices": outer["test_indices"],
        "cv_folds": cv_folds,
        "shap_reference_indices": shap_ref,
    }
