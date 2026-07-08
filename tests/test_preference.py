"""Characterization tests for the HEF weight learner (preference.py).

The learner is a hand-rolled Bradley-Terry MAP fit with an L2 prior toward the
default weights, source balancing, and a cross-validated accuracy that the
apply-weights guardrail trusts. These tests pin the contracts that matter:
- degenerate input (ties, no-signal, empty) yields None, never a crash;
- a cleanly separable signal is learned and scores at accuracy 1.0;
- the prior keeps weights from collapsing a dimension to 0;
- the two sources are balanced to equal total mass;
- CV accuracy is deterministic (fixed seed) so the guardrail is reproducible.
"""
from __future__ import annotations

import pytest

from lol_coach import preference as P


def _terms(power=None, info=None, action_fit=None, wave=None) -> dict:
    """Build a terms dict with only the given dims present (None = absent)."""
    out = {}
    if power is not None:
        out["power"] = power
    if info is not None:
        out["info"] = info
    if action_fit is not None:
        out["action_fit"] = action_fit
    if wave is not None:
        out["wave"] = wave
    return out


# ------------------------------------------------------------------- _delta
def test_delta_missing_dim_on_either_side_votes_zero():
    # info present on both -> real delta; power present on only one -> 0 (no signal).
    row = P._delta(_terms(power=0.8, info=0.9), _terms(info=0.4))
    dims = P.DIMS
    assert row[dims.index("info")] == pytest.approx(0.5)
    assert row[dims.index("power")] == 0.0


# ------------------------------------------------------------- _rows_from_pairs
def test_rows_from_pairs_default_source_is_pref():
    rows, n_no_signal = P._rows_from_pairs([(_terms(power=1.0), _terms(power=0.0), "a")])
    assert n_no_signal == 0
    assert len(rows) == 1
    assert rows[0][1] == "pref"


def test_rows_from_pairs_explicit_source_kept():
    rows, _ = P._rows_from_pairs([(_terms(power=1.0), _terms(power=0.0), "a", "dec")])
    assert rows[0][1] == "dec"


def test_rows_from_pairs_winner_b_flips_sign():
    a, b = _terms(power=0.0), _terms(power=1.0)
    rows, _ = P._rows_from_pairs([(a, b, "b")])
    # winner (b) goes first -> power delta = 1.0 - 0.0 = +1.0
    assert rows[0][0][P.DIMS.index("power")] == pytest.approx(1.0)


def test_rows_from_pairs_ties_and_no_signal_excluded():
    pairs = [
        (_terms(power=0.5), _terms(power=0.5), "a"),   # identical -> no signal
        (_terms(power=1.0), _terms(power=0.0), "tie"),  # tie -> skipped
    ]
    rows, n_no_signal = P._rows_from_pairs(pairs)
    assert rows == []
    assert n_no_signal == 1   # only the identical-terms row counts as no-signal


# --------------------------------------------------------------- _row_weights
def test_row_weights_single_source_all_one():
    rows = [([1.0, 0, 0, 0], "pref"), ([0.5, 0, 0, 0], "pref")]
    assert P._row_weights(rows) == [1.0, 1.0]


def test_row_weights_two_sources_balanced_to_equal_mass():
    # 1 pref + 2 dec rows. Each SOURCE should end with equal total mass.
    rows = [([1.0, 0, 0, 0], "pref"), ([1.0, 0, 0, 0], "dec"), ([1.0, 0, 0, 0], "dec")]
    w = P._row_weights(rows)
    by_source = {"pref": 0.0, "dec": 0.0}
    for (_, src), wt in zip(rows, w):
        by_source[src] += wt
    assert by_source["pref"] == pytest.approx(by_source["dec"])
    assert sum(w) == pytest.approx(len(rows))   # mean weight is 1.0


# --------------------------------------------------------------------- _acc
def test_acc_skips_zero_signal_rows():
    # state-only weights, an action-only row -> z == 0 -> "can't judge", not a miss.
    rows = [([0.0, 0.0, 1.0, 0.0], "dec")]            # only action_fit signal
    w = [1.0, 1.0, 0.0, 1.0]                          # action_fit weight 0 -> z == 0
    assert P._acc(w, rows) is None


def test_acc_source_filter():
    rows = [([1.0, 0, 0, 0], "pref"), ([-1.0, 0, 0, 0], "dec")]
    w = [1.0, 0, 0, 0]
    assert P._acc(w, rows, source="pref") == 1.0   # ordered right
    assert P._acc(w, rows, source="dec") == 0.0    # ordered wrong


# --------------------------------------------------------------- learn_weights
def test_learn_weights_empty_is_none():
    assert P.learn_weights([]) is None


def test_learn_weights_only_ties_is_none():
    assert P.learn_weights([(_terms(power=1.0), _terms(power=0.0), "tie")]) is None


def _separable_power_pairs(n: int):
    """n pairs where the winner always has the higher power term and nothing else."""
    return [(_terms(power=1.0), _terms(power=0.0), "a") for _ in range(n)]


def test_learn_weights_separable_signal_is_learned():
    res = P.learn_weights(_separable_power_pairs(12))
    assert res is not None
    assert res["n"] == 12
    assert res["accuracy"] == 1.0
    # weights are a normalized distribution
    assert sum(res["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    # the learned power weight should dominate, but the L2 prior keeps the
    # other dims off exactly 0 (no collapse).
    assert res["weights"]["power"] == max(res["weights"].values())
    assert all(v > 0.0 for v in res["weights"].values())


def test_learn_weights_enough_flag_tracks_min_pairs():
    assert P.learn_weights(_separable_power_pairs(P.MIN_PAIRS))["enough"] is True
    assert P.learn_weights(_separable_power_pairs(P.MIN_PAIRS - 1))["enough"] is False


def test_learn_weights_reports_no_signal_count():
    pairs = _separable_power_pairs(8) + [(_terms(power=0.5), _terms(power=0.5), "a")]
    res = P.learn_weights(pairs)
    assert res["n"] == 8            # the identical-terms row is excluded from the fit
    assert res["n_no_signal"] == 1


def test_cv_accuracy_is_deterministic():
    rows, _ = P._rows_from_pairs(_separable_power_pairs(15))
    a1 = P._cv_accuracy(rows, lr=0.5)
    a2 = P._cv_accuracy(rows, lr=0.5)
    assert a1 == a2                 # fixed CV_SEED -> reproducible guardrail number


def test_cv_accuracy_none_below_three_rows():
    rows, _ = P._rows_from_pairs(_separable_power_pairs(2))
    acc, scored = P._cv_accuracy(rows, lr=0.5)
    assert acc is None
    assert scored == 0


def test_learn_weights_two_sources_reports_per_source_accuracy():
    pairs = (
        [(_terms(power=1.0), _terms(power=0.0), "a", "pref") for _ in range(6)]
        + [(_terms(action_fit=1.0), _terms(action_fit=0.0), "a", "dec") for _ in range(6)]
    )
    res = P.learn_weights(pairs)
    assert set(res["n_by_source"]) == {"pref", "dec"}
    assert res["n_by_source"] == {"pref": 6, "dec": 6}
    assert set(res["accuracy_by_source"]) == {"pref", "dec"}


# ------------------------------------------------------------ persisted weights
def test_weights_roundtrip(tmp_path):
    db = tmp_path / "coach.db"
    weights = {"power": 0.4, "info": 0.25, "action_fit": 0.25, "wave": 0.1}
    P.save_weights(db, weights)
    loaded = P.load_weights(db)
    assert loaded == pytest.approx(weights)


def test_load_weights_absent_is_none(tmp_path):
    assert P.load_weights(tmp_path / "missing.db") is None
