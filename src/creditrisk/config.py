"""Configuration loader and validation utilities for credit-risk-xgb-shap."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ColumnConfig:
    id_column: str = "ID"
    target_column: str = "default.payment.next.month"
    target_column_clean: str = "default_payment_next_month"
    categorical_features: List[str] = field(default_factory=lambda: ["SEX", "EDUCATION", "MARRIAGE"])
    numerical_features: List[str] = field(default_factory=lambda: ["LIMIT_BAL", "AGE"])
    payment_status_features: List[str] = field(
        default_factory=lambda: ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    )
    bill_amount_features: List[str] = field(
        default_factory=lambda: [f"BILL_AMT{i}" for i in range(1, 7)]
    )
    payment_amount_features: List[str] = field(
        default_factory=lambda: [f"PAY_AMT{i}" for i in range(1, 7)]
    )

    @property
    def all_feature_columns(self) -> List[str]:
        """All predictor columns in order."""
        return (
            self.categorical_features
            + self.numerical_features
            + self.payment_status_features
            + self.bill_amount_features
            + self.payment_amount_features
        )


@dataclass
class PreprocessingConfig:
    education_unknown_values: List[int] = field(default_factory=lambda: [0, 5, 6])
    education_replacement: int = 4
    marriage_unknown_values: List[int] = field(default_factory=lambda: [0])
    marriage_replacement: int = 3


@dataclass
class SplitConfig:
    test_size: float = 0.20
    cv_n_splits: int = 5
    random_state: int = 42
    stratify: bool = True
    shap_reference_size: int = 1000
    shap_reference_seed: int = 42


@dataclass
class DataConfig:
    raw_data_path: str = "data/raw/default_credit_card.csv"
    processed_data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)


def load_data_config(config_path: str = "configs/data.yaml") -> DataConfig:
    """Load configuration from a YAML file into typed DataConfig."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to defaults if file not found at specific path
        return DataConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = raw.get("paths", {})
    cols = raw.get("columns", {})
    prep = raw.get("preprocessing", {})
    spl = raw.get("splits", {})

    column_config = ColumnConfig(
        id_column=cols.get("id_column", "ID"),
        target_column=cols.get("target_column", "default.payment.next.month"),
        target_column_clean=cols.get("target_column_clean", "default_payment_next_month"),
        categorical_features=cols.get("categorical_features", ["SEX", "EDUCATION", "MARRIAGE"]),
        numerical_features=cols.get("numerical_features", ["LIMIT_BAL", "AGE"]),
        payment_status_features=cols.get(
            "payment_status_features", ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        ),
        bill_amount_features=cols.get(
            "bill_amount_features", [f"BILL_AMT{i}" for i in range(1, 7)]
        ),
        payment_amount_features=cols.get(
            "payment_amount_features", [f"PAY_AMT{i}" for i in range(1, 7)]
        ),
    )

    preprocessing_config = PreprocessingConfig(
        education_unknown_values=prep.get("education_unknown_values", [0, 5, 6]),
        education_replacement=prep.get("education_replacement", 4),
        marriage_unknown_values=prep.get("marriage_unknown_values", [0]),
        marriage_replacement=prep.get("marriage_replacement", 3),
    )

    split_config = SplitConfig(
        test_size=float(spl.get("test_size", 0.20)),
        cv_n_splits=int(spl.get("cv_n_splits", 5)),
        random_state=int(spl.get("random_state", 42)),
        stratify=bool(spl.get("stratify", True)),
        shap_reference_size=int(spl.get("shap_reference_size", 1000)),
        shap_reference_seed=int(spl.get("shap_reference_seed", 42)),
    )

    return DataConfig(
        raw_data_path=paths.get("raw_data_path", "data/raw/default_credit_card.csv"),
        processed_data_dir=paths.get("processed_data_dir", "data/processed"),
        splits_dir=paths.get("splits_dir", "data/splits"),
        columns=column_config,
        preprocessing=preprocessing_config,
        splits=split_config,
    )
