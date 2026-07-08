"""Unit tests for the shared respawn/death-timer helpers (extracted from tempo)."""
from __future__ import annotations

import json
import sqlite3

from lol_coach.decisions.respawn import (
    _respawn_seconds, player_dead_at, last_death_before,
)


def _conn(death_ms=None, level=11, victim=7, pos=None):
    con = sqlite3.connect(":memory:"); con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE timeline_events (match_id TEXT, timestamp_ms INT, "
                "type TEXT, payload_json TEXT)")
    con.execute("CREATE TABLE timeline_frames (match_id TEXT, participant_id INT, "
                "timestamp_ms INT, level INT)")
    if death_ms is not None:
        payload = {"victimId": victim}
        if pos is not None:
            payload["position"] = {"x": pos[0], "y": pos[1]}
        con.execute("INSERT INTO timeline_events VALUES (?,?,?,?)",
                    ("M", death_ms, "CHAMPION_KILL", json.dumps(payload)))
        con.execute("INSERT INTO timeline_frames VALUES (?,?,?,?)",
                    ("M", victim, death_ms, level))
    return con


def test_respawn_seconds_no_factor_before_15min():
    assert _respawn_seconds(6, 10 * 60_000) == 16.0          # BRW[5]=16, tif=0


def test_respawn_seconds_scales_at_30min():
    assert round(_respawn_seconds(11, 30 * 60_000), 2) == 45.5   # 35 * 1.30


def test_respawn_seconds_handles_none_level():
    assert _respawn_seconds(None, 0) == 10.0                  # clamps to level 1


def test_last_death_before_returns_pos_or_none():
    con = _conn(death_ms=600_000, pos=(3000, 12000))
    d = last_death_before(con, "M", 7, 620_000)
    assert d == {"death_ms": 600_000, "x": 3000, "y": 12000}
    assert last_death_before(con, "M", 7, 500_000) is None   # before any death


def test_player_dead_at_true_within_respawn():
    con = _conn(death_ms=600_000, level=11)
    assert player_dead_at(con, "M", 7, 620_000) is True       # 600 + 35 + 12 = 647 > 620


def test_player_dead_at_false_after_respawn():
    con = _conn(death_ms=600_000, level=11)
    assert player_dead_at(con, "M", 7, 700_000) is False      # 647 < 700


def test_player_dead_at_false_when_never_died():
    con = _conn(death_ms=None)
    assert player_dead_at(con, "M", 7, 600_000) is False
