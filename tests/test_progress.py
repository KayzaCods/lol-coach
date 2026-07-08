"""Pure math for the behavioral-progress metric (#10): Wilson intervals, the
rolling window series, and the block comparison. No DB — the I/O layer feeds
(deaths, incidents) pairs per match in chronological order."""
from __future__ import annotations

from lol_coach.progress import (
    BLOCK,
    GOAL_PROP,
    LEAD_POWER_INDEX,
    OVEREXT_CLASSES,
    WINDOW,
    block_comparison,
    rolling_series,
    wilson_interval,
)


# ------------------------------------------------------------ criterion consts
def test_criterion_constants_are_the_specced_ones():
    assert OVEREXT_CLASSES == ("engage_blind", "disengage_failed")
    assert LEAD_POWER_INDEX == 0.55
    assert 10 <= WINDOW <= 15
    assert GOAL_PROP == 0.05          # 1 incident per 20 deaths (provisional)
    assert 20 <= BLOCK <= 30


# ------------------------------------------------------------- wilson_interval
def test_wilson_none_when_no_trials():
    assert wilson_interval(0, 0) is None


def test_wilson_zero_successes_starts_at_zero_but_high_is_positive():
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert 0.05 < high < 0.09         # ~0.071 for 0/50


def test_wilson_all_successes_ends_at_one():
    low, high = wilson_interval(20, 20)
    assert high == 1.0
    assert 0.80 < low < 0.88          # ~0.839 for 20/20


def test_wilson_known_value_real_window_shape():
    # 5 incidents over 69 deaths — a real window from today's measurement.
    low, high = wilson_interval(5, 69)
    assert 0.025 < low < 0.040        # ~0.031
    assert 0.150 < high < 0.170       # ~0.159
    assert low < 5 / 69 < high


# -------------------------------------------------------------- rolling_series
def test_rolling_series_windows_and_sums():
    per_match = [(5, 1), (4, 0), (6, 2), (3, 0), (2, 0)]
    pts = rolling_series(per_match, window=3)
    assert len(pts) == 3              # ends at match 3, 4, 5
    assert pts[0] == {
        "end": 3, "games": 3, "deaths": 15, "incidents": 3,
        "rate_per_game": 1.0, "deaths_per_game": 5.0,
        "prop": round(3 / 15, 4),
        "ci_low": pts[0]["ci_low"], "ci_high": pts[0]["ci_high"],
    }
    assert pts[0]["ci_low"] is not None and pts[0]["ci_low"] < 0.2 < pts[0]["ci_high"]
    assert pts[1]["deaths"] == 13 and pts[1]["incidents"] == 2
    assert pts[2]["deaths"] == 11 and pts[2]["incidents"] == 2


def test_rolling_series_empty_when_history_shorter_than_window():
    assert rolling_series([(5, 1)] * 4, window=12) == []


def test_rolling_series_zero_deaths_window_has_null_prop():
    pts = rolling_series([(0, 0)] * 3, window=3)
    assert pts[0]["prop"] is None and pts[0]["ci_low"] is None


# ------------------------------------------------------------ block_comparison
def test_block_comparison_prev_vs_last():
    per_match = [(4, 2), (4, 2), (4, 0), (4, 0)]
    res = block_comparison(per_match, block=2)
    assert res["block"] == 2
    assert res["prev"]["incidents"] == 4 and res["prev"]["deaths"] == 8
    assert res["last"]["incidents"] == 0 and res["last"]["deaths"] == 8
    assert res["prev"]["prop"] == 0.5 and res["last"]["prop"] == 0.0
    assert res["last"]["ci_high"] > 0.0  # 0/8 still has an upper bound


def test_block_comparison_none_without_two_blocks():
    assert block_comparison([(4, 1)] * 3, block=2) is None
