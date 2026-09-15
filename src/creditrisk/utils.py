"""Common helper utilities for logging, seeding, and serialization."""

from contextlib import contextmanager
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Generator, Union
import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(name: str = "Execution") -> Generator[None, None, None]:
    """Context manager to log elapsed time of code blocks."""
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] Completed in {t1 - t0:.2f} seconds.")


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, path: Union[str, Path]) -> None:
    """Save dictionary to YAML file with directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
