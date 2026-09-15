"""Model factories and wrappers for benchmark baseline algorithms."""

from typing import Any, Dict
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_model(model_name: str, params: Dict[str, Any]):
    """
    Factory function to instantiate models based on name and parameters.

    Supported model names:
    - 'logistic_regression'
    - 'random_forest'
    - 'gradient_boosting'
    - 'lightgbm'
    - 'catboost'
    - 'xgboost' or 'xgboost_default'

    Parameters
    ----------
    model_name : str
        Algorithm name identifier.
    params : Dict[str, Any]
        Hyperparameter dictionary.

    Returns
    -------
    BaseEstimator
        Instantiated classifier.
    """
    name = model_name.lower()

    if name == "logistic_regression":
        return LogisticRegression(**params)

    elif name == "random_forest":
        return RandomForestClassifier(**params)

    elif name == "gradient_boosting":
        return GradientBoostingClassifier(**params)

    elif name == "lightgbm":
        return LGBMClassifier(**params)

    elif name == "catboost":
        return CatBoostClassifier(**params)

    elif name in ["xgboost", "xgboost_default"]:
        return XGBClassifier(**params)

    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported models: "
            f"['logistic_regression', 'random_forest', 'gradient_boosting', 'lightgbm', 'catboost', 'xgboost']"
        )
