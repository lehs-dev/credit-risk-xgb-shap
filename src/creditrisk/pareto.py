"""Independent two-objective Pareto checks and deterministic representative selection.

Both objectives are maximized. Selection uses only completed Development-study
values; this module has no data-loading or model-evaluation dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import hypot, isfinite
from typing import Any


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return whether two-dimensional point ``a`` strictly Pareto-dominates ``b``."""
    if len(a) != 2 or len(b) != 2:
        raise ValueError("Pareto dominance requires two objectives per point.")
    if not all(isfinite(float(value)) for value in (*a, *b)):
        raise ValueError("Pareto objectives must be finite.")
    return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])


def _prepare_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    seen_trials = set()
    for row in rows:
        trial = int(row["trial"])
        auc = float(row["roc_auc_cv"])
        stability = float(row["shap_stability"])
        if trial < 0 or trial in seen_trials:
            raise ValueError("Trial numbers must be unique nonnegative integers.")
        if not isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError(f"Invalid ROC-AUC for trial {trial}.")
        if not isfinite(stability) or not -1 <= stability <= 1:
            raise ValueError(f"Invalid SHAP stability for trial {trial}.")
        seen_trials.add(trial)
        prepared.append({**row, "trial": trial, "roc_auc_cv": auc, "shap_stability": stability})
    return sorted(prepared, key=lambda row: row["trial"])


def dominance_counts(rows: Iterable[Mapping[str, Any]]) -> dict[int, int]:
    """Count evaluated trials that dominate each trial; equal points do not dominate."""
    prepared = _prepare_rows(rows)
    return {
        row["trial"]: sum(
            dominates(
                (other["roc_auc_cv"], other["shap_stability"]),
                (row["roc_auc_cv"], row["shap_stability"]),
            )
            for other in prepared
            if other["trial"] != row["trial"]
        )
        for row in prepared
    }


def pareto_front(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return all nondominated trials, including trials with equal objective pairs."""
    prepared = _prepare_rows(rows)
    counts = dominance_counts(prepared)
    return [row for row in prepared if counts[row["trial"]] == 0]


def select_representatives(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select up to three distinct Pareto points using a locked Development rule.

    AUC is maximized first, with stability and then earlier trial as tie-breaks.
    Stability is maximized next, with AUC and then earlier trial as tie-breaks.
    The balanced point is the remaining Pareto point nearest (1, 1) after
    min-max scaling each objective over the unique Pareto points. Distance
    ties favor AUC, then stability, then earlier trial. Equal objective pairs
    are represented by the earliest trial. A collapsed front yields fewer than
    three representatives rather than selecting dominated configurations.
    """
    front = pareto_front(rows)
    unique_points: dict[tuple[float, float], dict[str, Any]] = {}
    for row in front:
        point = (row["roc_auc_cv"], row["shap_stability"])
        unique_points.setdefault(point, row)
    candidates = list(unique_points.values())
    if not candidates:
        return []

    auc_min = min(row["roc_auc_cv"] for row in candidates)
    auc_max = max(row["roc_auc_cv"] for row in candidates)
    stability_min = min(row["shap_stability"] for row in candidates)
    stability_max = max(row["shap_stability"] for row in candidates)

    def normalized(value: float, low: float, high: float) -> float:
        return 1.0 if high == low else (value - low) / (high - low)

    def with_selection_fields(row: dict[str, Any], role: str) -> dict[str, Any]:
        auc_scaled = normalized(row["roc_auc_cv"], auc_min, auc_max)
        stability_scaled = normalized(row["shap_stability"], stability_min, stability_max)
        return {
            **row,
            "role": role,
            "normalized_auc": auc_scaled,
            "normalized_stability": stability_scaled,
            "ideal_distance": hypot(1.0 - auc_scaled, 1.0 - stability_scaled),
        }

    auc_choice = max(
        candidates,
        key=lambda row: (row["roc_auc_cv"], row["shap_stability"], -row["trial"]),
    )
    stability_choice = max(
        candidates,
        key=lambda row: (row["shap_stability"], row["roc_auc_cv"], -row["trial"]),
    )
    selected = [with_selection_fields(auc_choice, "auc")]
    chosen_trials = {auc_choice["trial"]}
    if stability_choice["trial"] not in chosen_trials:
        selected.append(with_selection_fields(stability_choice, "stability"))
        chosen_trials.add(stability_choice["trial"])

    remaining = [row for row in candidates if row["trial"] not in chosen_trials]
    if remaining:
        balanced = min(
            remaining,
            key=lambda row: (
                with_selection_fields(row, "balanced")["ideal_distance"],
                -row["roc_auc_cv"],
                -row["shap_stability"],
                row["trial"],
            ),
        )
        selected.append(with_selection_fields(balanced, "balanced"))
    return selected
