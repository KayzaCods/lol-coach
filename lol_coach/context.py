"""Build a 'world state' snapshot at any moment of a match.

Combines participants (static info), timeline_frames (per-minute positions
and stats), and timeline_events (kills, wards, objectives) into a single
object that decision evaluators can consume.

The position data from Riot's timeline is a god-view (visible regardless
of fog of war). For "what we actually saw" approximations, see
last_event_appearance(): it returns the most recent event involving an
enemy, which is a lower-bound proxy for visibility.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


# LoL map is 0..15000 in both x and y. Blue side (team 100) base is at
# bottom-left, red side (team 200) base at top-right.
MAP_MAX = 15000

# Visibility / proximity thresholds (LoL distance units).
ALLY_NEAR_UNITS = 2500          # close enough to peel/help in a fight
ENEMY_CLOSE_UNITS = 1200        # within most spell ranges
UNSEEN_DANGEROUS_S = 20.0       # enemy unseen for this long = potential gank


@dataclass
class PlayerSnap:
    participant_id: int
    puuid: str
    champion: str
    team_id: int
    position: Optional[str]
    is_us: bool
    pos_x: Optional[int] = None
    pos_y: Optional[int] = None
    pos_at_ms: Optional[int] = None
    level: Optional[int] = None
    total_gold: Optional[int] = None
    current_gold: Optional[int] = None
    minions_killed: Optional[int] = None
    jungle_minions_killed: Optional[int] = None
    kills: int = 0
    deaths: int = 0
    assists: int = 0


@dataclass
class WorldState:
    match_id: str
    game_time_ms: int
    us_id: int
    our_team: int
    players: dict[int, PlayerSnap] = field(default_factory=dict)


def get_match_meta(conn: sqlite3.Connection, match_id: str) -> dict:
    match = conn.execute(
        "SELECT * FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    if not match:
        raise ValueError(f"Match {match_id} not in DB")
    us = conn.execute(
        "SELECT participant_id, team_id FROM participants "
        "WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    if not us:
        raise ValueError(f"Our participant row missing for {match_id}")
    return {"match": dict(match), "us_id": us["participant_id"], "our_team": us["team_id"]}


def build_world_state(
    conn: sqlite3.Connection, match_id: str, game_time_ms: int
) -> WorldState:
    """Snapshot of all 10 players + score state at the given game time."""
    meta = get_match_meta(conn, match_id)
    state = WorldState(
        match_id=match_id,
        game_time_ms=game_time_ms,
        us_id=meta["us_id"],
        our_team=meta["our_team"],
    )

    participants = conn.execute(
        "SELECT participant_id, puuid, champion_name, team_id, team_position "
        "FROM participants WHERE match_id = ?",
        (match_id,),
    ).fetchall()
    for p in participants:
        pid = p["participant_id"]
        state.players[pid] = PlayerSnap(
            participant_id=pid,
            puuid=p["puuid"],
            champion=p["champion_name"],
            team_id=p["team_id"],
            position=p["team_position"],
            is_us=(pid == state.us_id),
        )

    # Latest frame at or before game_time_ms, per participant.
    for pid, snap in state.players.items():
        row = conn.execute(
            "SELECT * FROM timeline_frames "
            "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
            "ORDER BY timestamp_ms DESC LIMIT 1",
            (match_id, pid, game_time_ms),
        ).fetchone()
        if row is None:
            continue
        snap.pos_x = row["position_x"]
        snap.pos_y = row["position_y"]
        snap.pos_at_ms = row["timestamp_ms"]
        snap.level = row["level"]
        snap.total_gold = row["total_gold"]
        snap.current_gold = row["current_gold"]
        snap.minions_killed = row["minions_killed"]
        snap.jungle_minions_killed = row["jungle_minions_killed"]

    # KDA accumulated from CHAMPION_KILL events up to game_time_ms.
    kill_rows = conn.execute(
        "SELECT payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'CHAMPION_KILL' AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms",
        (match_id, game_time_ms),
    ).fetchall()
    for r in kill_rows:
        ev = json.loads(r["payload_json"])
        killer = ev.get("killerId")
        victim = ev.get("victimId")
        assists = ev.get("assistingParticipantIds") or []
        if killer and killer in state.players:
            state.players[killer].kills += 1
        if victim and victim in state.players:
            state.players[victim].deaths += 1
        for a in assists:
            if a in state.players:
                state.players[a].assists += 1

    return state


# ------------------------------------------------------------------- helpers

def distance(a: PlayerSnap, b: PlayerSnap) -> Optional[float]:
    if a.pos_x is None or b.pos_x is None:
        return None
    return math.hypot(a.pos_x - b.pos_x, a.pos_y - b.pos_y)


# Event types where a participant's position is implicitly visible to all
# (i.e., a lower-bound for "we saw them recently").
_VISIBLE_EVENT_TYPES = (
    "CHAMPION_KILL",
    "CHAMPION_SPECIAL_KILL",
    "WARD_KILL",
    "BUILDING_KILL",
    "TURRET_PLATE_DESTROYED",
    "ELITE_MONSTER_KILL",
)


def last_event_appearance(
    conn: sqlite3.Connection, match_id: str, pid: int, before_ms: int
) -> Optional[int]:
    """Most recent timestamp_ms of an event involving `pid`, strictly before `before_ms`.

    Lower-bound proxy for visibility — undercounts because it ignores
    lane-presence visibility (we can't tell from minute-frames whether
    the enemy was within ally vision range continuously).
    """
    placeholders = ",".join("?" * len(_VISIBLE_EVENT_TYPES))
    rows = conn.execute(
        f"SELECT timestamp_ms, payload_json FROM timeline_events "
        f"WHERE match_id = ? AND type IN ({placeholders}) AND timestamp_ms < ? "
        f"ORDER BY timestamp_ms DESC LIMIT 50",
        (match_id, *_VISIBLE_EVENT_TYPES, before_ms),
    ).fetchall()
    for row in rows:
        ev = json.loads(row["payload_json"])
        if ev.get("killerId") == pid or ev.get("victimId") == pid:
            return row["timestamp_ms"]
        if pid in (ev.get("assistingParticipantIds") or []):
            return row["timestamp_ms"]
    return None


def map_lane(x: Optional[int], y: Optional[int]) -> str:
    """Rough lane classification on Summoner's Rift (15000x15000 map)."""
    if x is None or y is None:
        return "desconocido"
    if x > 9500 and y < 5500:
        return "bot lane"
    if x < 5500 and y > 9500:
        return "top lane"
    if 5500 <= x <= 9500 and 5500 <= y <= 9500 and abs(x - y) < 1500:
        return "mid lane"
    if x > 7500 and y > 7500:
        return "rio"
    if x < 7500 and y < 7500:
        return "rio (lado azul)"
    # Otherwise jungle. Crude: top-right jungle vs bottom-left jungle.
    return "jungla"


def map_side(x: Optional[int], y: Optional[int], our_team: int) -> str:
    """Whose half of the map this is on, from our team's perspective."""
    if x is None or y is None:
        return "desconocido"
    # Blue side (team 100) controls bottom-left. The river is roughly y = MAP_MAX - x.
    # We consider "enemy side" to be the opposite team's half.
    if our_team == 100:
        return "lado enemigo" if x + y > MAP_MAX else "nuestro lado"
    return "lado enemigo" if x + y < MAP_MAX else "nuestro lado"


def find_clip_for_event(
    conn: sqlite3.Connection, match_id: str, event_ms: int
) -> tuple[Optional[str], Optional[float]]:
    """Find the clip whose window contains `event_ms` for this match, if any.

    Returns (clip_path, offset_into_clip_seconds) or (None, None).
    """
    # First check per-event clips that cover this exact moment.
    row = conn.execute(
        "SELECT path, in_match_start_s FROM clips "
        "WHERE match_id = ? AND in_match_start_s IS NOT NULL "
        "ORDER BY ABS(in_match_start_s * 1000 - ?) ASC LIMIT 1",
        (match_id, event_ms),
    ).fetchone()
    if row is None:
        return (None, None)
    # If the closest clip starts within the match, treat it as covering ~30s
    # before/after its start. For full-match recordings the offset is simply
    # event_ms/1000.
    clip_path = row["path"]
    clip_start_s = row["in_match_start_s"] or 0.0
    offset_s = (event_ms / 1000.0) - clip_start_s
    return (clip_path, offset_s)
