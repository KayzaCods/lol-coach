"""Integration tests for the trade detector (trade_v1 / v2).

trade_v2 reads the RAW Riot timeline JSON (cumulative per-minute damageStats) to
judge bot-lane exchanges. These tests write a minimal timeline where exactly one
minute is a lopsided exchange and assert: a lost trade is flagged, a won trade is
recorded as positive (hidden from the list but kept for Calibrar), and a minute
below the damage threshold produces nothing.
"""
from __future__ import annotations

from lol_coach.decisions import trade as TR

from conftest import damage_frame, set_timeline_json, ten_player_match

# Our bot duo is pid 1 (BOTTOM) + 5 (UTILITY); enemy duo is pid 6 + 10.
BAD_MINUTE = 9               # a conftest position frame exists at 9*60s = 540k


def _timeline_with_minute9(conn, match_id, tmp_path, *, us_taken, enemy_taken):
    """11 frames (0..10), all zero except a single cumulative jump at minute 9."""
    frames = [damage_frame() for _ in range(9)]                       # minutes 0..8: nothing
    jump = damage_frame(**{"1": us_taken, "6": enemy_taken})          # our pid1 vs enemy pid6
    frames.append(jump)                                               # frame 9: the exchange
    frames.append(jump)                                               # frame 10: no new damage
    set_timeline_json(conn, match_id, tmp_path / "timeline.json", frames)


def test_lost_trade_is_flagged(conn, tmp_path):
    ids = ten_player_match(conn)
    # We took 600, the enemy duo 100: a clearly lost exchange.
    _timeline_with_minute9(conn, ids["match_id"], tmp_path, us_taken=600, enemy_taken=100)

    decisions = TR.analyze_trades(conn, ids["match_id"])

    assert len(decisions) == 1
    d = decisions[0]
    assert d.detector_id == "trade_v1"
    assert d.game_time_ms == BAD_MINUTE * 60_000
    assert d.context["positive_trade"] is False
    assert d.context["duo_damage_taken"] == 600
    assert d.context["enemy_duo_damage_taken"] == 100
    assert "perdido" in d.moment.lower()


def test_lost_trade_has_wellformed_options(conn, tmp_path):
    ids = ten_player_match(conn)
    _timeline_with_minute9(conn, ids["match_id"], tmp_path, us_taken=600, enemy_taken=100)

    d = TR.analyze_trades(conn, ids["match_id"])[0]

    assert len(d.options) == 2
    taken = [o for o in d.options if "hiciste" in o.label.lower()]
    assert len(taken) == 1
    assert set(d.context["state_features"]) == {"power", "info_risk", "wave_tempo", "objectives"}


def test_won_trade_is_recorded_positive(conn, tmp_path):
    ids = ten_player_match(conn)
    # Mirror image: the enemy duo took 600, we took 100 -> a won exchange. These
    # are persisted flagged positive (hidden from the list, kept for Calibrar).
    _timeline_with_minute9(conn, ids["match_id"], tmp_path, us_taken=100, enemy_taken=600)

    decisions = TR.analyze_trades(conn, ids["match_id"])

    assert len(decisions) == 1
    assert decisions[0].context["positive_trade"] is True
    assert "ganado" in decisions[0].moment.lower()


def test_minute_below_damage_threshold_is_ignored(conn, tmp_path):
    ids = ten_player_match(conn)
    # 60 + 40 = 100 combined < MIN_EXCHANGE_DMG (250): not a real exchange.
    _timeline_with_minute9(conn, ids["match_id"], tmp_path, us_taken=60, enemy_taken=40)

    assert TR.analyze_trades(conn, ids["match_id"]) == []


def test_no_timeline_json_returns_empty(conn):
    ids = ten_player_match(conn)
    # ten_player_match leaves timeline_json_path NULL -> nothing to analyze.
    assert TR.analyze_trades(conn, ids["match_id"]) == []
