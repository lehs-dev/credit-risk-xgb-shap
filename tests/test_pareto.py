"""Checks for independent Pareto filtering and representative selection."""

import math

import pytest

from creditrisk.pareto import dominates, pareto_front, select_representatives


def _row(trial, auc, stability):
    return {
        "trial": trial,
        "roc_auc_cv": auc,
        "shap_stability": stability,
    }


def _by_role(rows):
    by_role = {row["role"]: row for row in rows}
    assert len(by_role) == len(rows)
    return by_role


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0.8, 0.9), (0.7, 0.8), True),
        ((0.8, 0.9), (0.8, 0.8), True),
        ((0.8, 0.9), (0.7, 0.9), True),
        ((0.8, 0.9), (0.8, 0.9), False),
        ((0.8, 0.7), (0.7, 0.9), False),
        ((0.7, 0.8), (0.8, 0.9), False),
    ],
)
def test_dominance_requires_no_worse_and_one_strict_gain(a, b, expected):
    assert dominates(a, b) is expected


def test_pareto_front_keeps_all_nondominated_trials_including_equal_points():
    rows = [
        _row(3, 0.75, 0.75),  # dominated by trial 2
        _row(4, 0.70, 0.90),
        _row(1, 0.83, 0.70),
        _row(5, 0.80, 0.80),  # same objective pair as trial 2
        _row(6, 0.60, 0.60),  # dominated by several trials
        _row(2, 0.80, 0.80),
    ]

    front = pareto_front(rows)

    assert {row["trial"] for row in front} == {1, 2, 4, 5}
    assert len(front) == 4
    assert pareto_front([]) == []


def test_representatives_use_unique_front_extremes_and_normalized_ideal_distance():
    rows = [
        _row(9, 0.91, 0.60),  # tied AUC but lower stability
        _row(2, 0.91, 0.63),
        _row(1, 0.91, 0.63),  # tied objective pair; lower trial wins
        _row(8, 0.63, 0.92),  # tied stability but lower AUC
        _row(4, 0.66, 0.92),
        _row(3, 0.66, 0.92),  # tied objective pair; lower trial wins
        _row(6, 0.80, 0.82),  # closest interior point to the ideal
        _row(7, 0.84, 0.75),
        _row(10, 0.20, 0.20),  # must not affect front normalization
    ]

    chosen = _by_role(select_representatives(rows))

    assert {role: row["trial"] for role, row in chosen.items()} == {
        "auc": 1,
        "stability": 3,
        "balanced": 6,
    }
    assert chosen["auc"]["normalized_auc"] == pytest.approx(1.0)
    assert chosen["auc"]["normalized_stability"] == pytest.approx(0.0)
    assert chosen["stability"]["normalized_auc"] == pytest.approx(0.0)
    assert chosen["stability"]["normalized_stability"] == pytest.approx(1.0)

    balanced = chosen["balanced"]
    expected_auc = (0.80 - 0.66) / (0.91 - 0.66)
    expected_stability = (0.82 - 0.63) / (0.92 - 0.63)
    assert balanced["normalized_auc"] == pytest.approx(expected_auc)
    assert balanced["normalized_stability"] == pytest.approx(expected_stability)
    assert balanced["ideal_distance"] == pytest.approx(
        math.hypot(1.0 - expected_auc, 1.0 - expected_stability)
    )


def test_balanced_choice_excludes_the_two_extreme_trials():
    # The interior point lies farther from (1, 1) than either extreme.
    rows = [_row(1, 0.90, 0.80), _row(2, 0.80, 0.90), _row(3, 0.82, 0.82)]

    chosen = _by_role(select_representatives(rows))

    assert {role: row["trial"] for role, row in chosen.items()} == {
        "auc": 1,
        "stability": 2,
        "balanced": 3,
    }


def test_balanced_distance_tie_prefers_higher_auc():
    rows = [
        _row(1, 0.875, 0.500),
        _row(2, 0.500, 0.875),
        _row(3, 0.78125, 0.6875),  # normalized (0.75, 0.50)
        _row(4, 0.6875, 0.78125),  # normalized (0.50, 0.75)
    ]

    chosen = _by_role(select_representatives(rows))

    assert chosen["balanced"]["trial"] == 3


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [_row(10, 0.80, 0.70), _row(3, 0.80, 0.70), _row(5, 0.70, 0.80), _row(4, 0.60, 0.60)],
            {"auc": 3, "stability": 5},
        ),
        (
            [_row(8, 0.80, 0.70), _row(1, 0.80, 0.70), _row(4, 0.50, 0.40)],
            {"auc": 1},
        ),
        ([], {}),
    ],
)
def test_representatives_do_not_invent_missing_distinct_points(rows, expected):
    chosen = _by_role(select_representatives(rows))
    assert {role: row["trial"] for role, row in chosen.items()} == expected
