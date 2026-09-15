"""Preprocessing pipelines with strict anti-leakage guarantees."""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from creditrisk.config import DataConfig, load_data_config


class CleanCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Clean undocumented/unknown categorical values in EDUCATION and MARRIAGE.
    According to Yeh & Lien (2009):
    - EDUCATION: 0, 5, 6 are grouped into 4 (Others).
    - MARRIAGE: 0 is grouped into 3 (Others).
    """

    def __init__(
        self,
        education_col: str = "EDUCATION",
        education_unknowns: Optional[List[int]] = None,
        education_replacement: int = 4,
        marriage_col: str = "MARRIAGE",
        marriage_unknowns: Optional[List[int]] = None,
        marriage_replacement: int = 3,
    ):
        self.education_col = education_col
        self.education_unknowns = education_unknowns if education_unknowns is not None else [0, 5, 6]
        self.education_replacement = education_replacement
        self.marriage_col = marriage_col
        self.marriage_unknowns = marriage_unknowns if marriage_unknowns is not None else [0]
        self.marriage_replacement = marriage_replacement

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.education_col in X_out.columns:
            X_out[self.education_col] = X_out[self.education_col].replace(
                {val: self.education_replacement for val in self.education_unknowns}
            )
        if self.marriage_col in X_out.columns:
            X_out[self.marriage_col] = X_out[self.marriage_col].replace(
                {val: self.marriage_replacement for val in self.marriage_unknowns}
            )
        return X_out


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Unified preprocessor for credit risk models.
    Guarantees strict isolation (no leakage):
    - Fits statistics only on training data.
    - Preserves explicit column names for SHAP feature importance tracking.
    """

    def __init__(
        self,
        scale_numerical: bool = True,
        scaler_type: str = "standard",  # 'standard' or 'robust'
        one_hot_categorical: bool = True,
        config: Optional[DataConfig] = None,
    ):
        self.scale_numerical = scale_numerical
        self.scaler_type = scaler_type
        self.one_hot_categorical = one_hot_categorical
        self.config = config if config is not None else load_data_config()

        self.cleaner_ = CleanCategoricalTransformer(
            education_col="EDUCATION",
            education_unknowns=self.config.preprocessing.education_unknown_values,
            education_replacement=self.config.preprocessing.education_replacement,
            marriage_col="MARRIAGE",
            marriage_unknowns=self.config.preprocessing.marriage_unknown_values,
            marriage_replacement=self.config.preprocessing.marriage_replacement,
        )

        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None

    def _build_column_transformer(self) -> ColumnTransformer:
        transformers = []

        # Categorical columns: SEX, EDUCATION, MARRIAGE
        cat_cols = self.config.columns.categorical_features
        if self.one_hot_categorical:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                    cat_cols,
                )
            )
        else:
            transformers.append(("cat", "passthrough", cat_cols))

        # Continuous numerical columns
        num_cols = (
            self.config.columns.numerical_features
            + self.config.columns.bill_amount_features
            + self.config.columns.payment_amount_features
        )

        if self.scale_numerical:
            scaler = (
                RobustScaler() if self.scaler_type == "robust" else StandardScaler()
            )
            transformers.append(("num", scaler, num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

        # Payment status columns (can be kept ordinal / integer)
        status_cols = self.config.columns.payment_status_features
        transformers.append(("status", "passthrough", status_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit preprocessor only on the training fold."""
        X_cleaned = self.cleaner_.fit_transform(X)
        self.column_transformer_ = self._build_column_transformer()
        self.column_transformer_.fit(X_cleaned, y)

        # Cache output feature names
        try:
            self.feature_names_out_ = list(
                self.column_transformer_.get_feature_names_out()
            )
        except Exception:
            # Fallback if scikit-learn version doesn't support get_feature_names_out on pass-through
            self.feature_names_out_ = [f"f_{i}" for i in range(self.column_transformer_.n_features_in_)]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using parameters fitted on the training fold."""
        if self.column_transformer_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        X_cleaned = self.cleaner_.transform(X)
        arr = self.column_transformer_.transform(X_cleaned)

        return pd.DataFrame(
            arr,
            columns=self.feature_names_out_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor not fitted.")
        return list(self.feature_names_out_)
