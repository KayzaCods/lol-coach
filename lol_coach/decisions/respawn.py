"""Respawn / death-timer helpers shared by detectors.

Extracted verbatim from tempo_v1 (where the death-timer first lived) so
objective_readiness_v1 can reuse the same 'was the player dead at T' check and
the death position without duplicating the respawn formula. Pure stdlib + a
sqlite conn.
"""
from __future__ import annotations

import json

# Base respawn wait (seconds) by champion level 1..18 — standard LoL death timer.
_BRW = [10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5]

RESPAWN_GRACE_S = 12   # just-respawned (still at fountain): no decision window yet


def _respawn_seconds(level: int | None, game_time_ms: int) -> float:
    """Approximate death-timer length: base wait by level scaled by a game-time
    factor (deaths get longer late game). Close enough to decide 'was I still dead'."""
    lvl = max(1, min(18, level or 1))
    minutes = game_time_ms / 60000.0
    if minutes <= 15:
        tif = 0.0
    elif minutes <= 30:
        tif = (minutes - 15) * 0.02      # up to +30% at 30 min
    else:
        tif = min(0.50, 0.30 + (minutes - 30) * 0.01)
    return _BRW[lvl - 1] * (1.0 + tif)


def last_death_before(conn, match_id, us_id, t_ms) -> dict | None:
    """The player's most recent death (as victim) strictly before t_ms, with its
    position: {"death_ms", "x", "y"} (x/y may be None), or None if they never died."""
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events WHERE match_id=? AND "
        "type='CHAMPION_KILL' AND timestamp_ms < ? ORDER BY timestamp_ms DESC",
        (match_id, t_ms),
    ):
        ev = json.loads(row["payload_json"])
        if ev.get("victimId") != us_id:
            continue
        pos = ev.get("position") or {}
        return {"death_ms": row["timestamp_ms"], "x": pos.get("x"), "y": pos.get("y")}
    return None


def player_dead_at(conn, match_id, us_id, t_ms) -> bool:
    """True if the player was dead — or had JUST respawned (still at the fountain) —
    at t_ms, so there was no decision to make at that instant."""
    d = last_death_before(conn, match_id, us_id, t_ms)
    if d is None:
        return False
    death_ms = d["death_ms"]
    lvl = conn.execute(
        "SELECT level FROM timeline_frames WHERE match_id=? AND participant_id=? AND "
        "timestamp_ms<=? ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, us_id, death_ms),
    ).fetchone()
    respawn_ms = death_ms + _respawn_seconds(lvl["level"] if lvl else None, death_ms) * 1000
    return respawn_ms + RESPAWN_GRACE_S * 1000 > t_ms
