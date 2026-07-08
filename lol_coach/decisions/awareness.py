"""Map/camera awareness analyzer (Phase 2) — needs Ascent input_events.

The player's #1 priority pattern: camera/map awareness. We can't see eye
glances, but Ascent's input log gives us active map usage:
  - map_command : a right-click issued on the minimap
  - camera      : a mouse-wheel (zoom) tick
  - Space        : centre camera on champion

Trigger: each of the user's own deaths that had ZERO of those interactions in
the 10s before dying — i.e. you died without actively consulting the map. That's
the actionable awareness deficit (the rest of the deaths, where you did check,
are left to death_v1). Pairs the deficit with how long the enemy jungler had
been unseen, since a blind death to an off-screen ganker is the classic case.

Requires input_events for the match (scripts/ingest_ascent_events.py).
"""
from __future__ import annotations

import json
import math
import sqlite3

from ..context import build_world_state
from .base import Decision, Option
from .features import ACTION_CHECK_MAP, ACTION_FIGHT_ENTRY, build_action, build_state_features

DETECTOR_ID = "awareness_v1"

AWARENESS_WINDOW_MS = 10_000   # look this far before death for map interactions
MAP_INTENTS = ("map_command", "camera")
FIGHT_NEAR_UNITS = 3000        # another kill this close (±8s) = a fight, not a gank
SKIRMISH_RADIUS_UNITS = 3000   # allies this close to the death = team was fighting here
MIN_ALLIES_SKIRMISH = 2        # this many allies present = a skirmish, not an isolated gank


LANE_GROUP = {"TOP": "top", "JUNGLE": "jg", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}


def _is_gank(parts, us_pos, ev) -> bool:
    """A gank/roam death: at least one attacker came from OUTSIDE your lane (or is
    the jungler). If only your direct lane matchup killed you, it was a trade/duel,
    not a map-awareness deficit. (Player feedback: lane trades flagged as 'no map
    awareness'.)"""
    mine = LANE_GROUP.get(us_pos)
    if mine is None:
        return True  # unknown role: don't suppress
    for a in [ev.get("killerId")] + (ev.get("assistingParticipantIds") or []):
        p = parts.get(a)
        if p and LANE_GROUP.get(p["team_position"]) != mine:
            return True
    return False


def _death_in_fight(conn, match_id, death_ms, death_pos, us_id) -> bool:
    """True if the death happened inside a fight (another champion kill within ±8s
    and ~3000u). Then it wasn't an awareness deficit — you were committed to a
    fight, not ganked for not watching the map. (Player feedback: trades and
    teamfights were being flagged as 'no map awareness'.)"""
    dx, dy = death_pos.get("x"), death_pos.get("y")
    for r in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events WHERE match_id=? AND "
        "type='CHAMPION_KILL' AND timestamp_ms BETWEEN ? AND ?",
        (match_id, death_ms - 8000, death_ms + 8000)):
        ev = json.loads(r["payload_json"])
        if r["timestamp_ms"] == death_ms and ev.get("victimId") == us_id:
            continue  # our own death
        pos = ev.get("position") or {}
        if dx is None or pos.get("x") is None:
            return True
        if math.hypot(pos["x"] - dx, pos["y"] - dy) <= FIGHT_NEAR_UNITS:
            return True
    return False


def _allies_near(conn, match_id, our_team, us_id, death_ms, death_pos) -> int:
    """How many teammates (excluding you) were near the death spot. 2+ means your team
    was committed to a fight right there — a skirmish, not an isolated gank you'd have
    avoided by glancing at the minimap. (Player feedback: a 3v1 skirmish after taking
    objectives was flagged as 'no map awareness'.) Positions are the nearest per-minute
    frame, so this is approximate."""
    dx, dy = death_pos.get("x"), death_pos.get("y")
    if dx is None:
        return 0
    n = 0
    for p in conn.execute(
        "SELECT participant_id FROM participants WHERE match_id=? AND team_id=? AND participant_id<>?",
        (match_id, our_team, us_id),
    ):
        fr = conn.execute(
            "SELECT position_x x, position_y y FROM timeline_frames WHERE match_id=? AND "
            "participant_id=? ORDER BY ABS(timestamp_ms-?) LIMIT 1",
            (match_id, p["participant_id"], death_ms),
        ).fetchone()
        if fr and fr["x"] is not None and math.hypot(fr["x"] - dx, fr["y"] - dy) <= SKIRMISH_RADIUS_UNITS:
            n += 1
    return n


def _mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def _map_interactions(conn, match_id, t0, t1) -> dict:
    """Count active map interactions in [t0, t1]: minimap clicks, zoom, centre-camera."""
    counts = {"map_command": 0, "camera": 0, "center_camera": 0}
    for r in conn.execute(
        "SELECT classified_intent, key, COUNT(*) c FROM input_events "
        "WHERE match_id = ? AND game_time_ms BETWEEN ? AND ? "
        "AND (classified_intent IN ('map_command','camera') OR key = 'Space') "
        "GROUP BY classified_intent, key",
        (match_id, t0, t1),
    ):
        if r["key"] == "Space":
            counts["center_camera"] += r["c"]
        elif r["classified_intent"] in MAP_INTENTS:
            counts[r["classified_intent"]] += r["c"]
    counts["total"] = counts["map_command"] + counts["camera"] + counts["center_camera"]
    return counts


def _last_map_interaction_ms(conn, match_id, before_ms) -> int | None:
    r = conn.execute(
        "SELECT MAX(game_time_ms) m FROM input_events "
        "WHERE match_id = ? AND game_time_ms < ? "
        "AND (classified_intent IN ('map_command','camera') OR key = 'Space')",
        (match_id, before_ms),
    ).fetchone()
    return r["m"] if r and r["m"] is not None else None


def analyze_awareness(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    if not conn.execute(
        "SELECT 1 FROM input_events WHERE match_id = ? LIMIT 1", (match_id,)
    ).fetchone():
        return []

    match = conn.execute("SELECT our_puuid FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    us = conn.execute(
        "SELECT participant_id, team_id, team_position FROM participants WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    us_id, our_team, us_pos = us["participant_id"], us["team_id"], us["team_position"]
    parts = {r["participant_id"]: r for r in conn.execute(
        "SELECT participant_id, team_position FROM participants WHERE match_id = ?", (match_id,))}

    decisions: list[Decision] = []
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'CHAMPION_KILL' ORDER BY timestamp_ms",
        (match_id,),
    ):
        ev = json.loads(row["payload_json"])
        if ev.get("victimId") != us_id:
            continue
        t = row["timestamp_ms"]
        inter = _map_interactions(conn, match_id, t - AWARENESS_WINDOW_MS, t)
        if inter["total"] > 0:
            continue  # you did consult the map; not an awareness deficit
        if _death_in_fight(conn, match_id, t, ev.get("position") or {}, us_id):
            continue  # died in a fight, not an isolated gank — awareness not the issue
        if _allies_near(conn, match_id, our_team, us_id, t, ev.get("position") or {}) >= MIN_ALLIES_SKIRMISH:
            continue  # 2+ allies fighting right there — a team skirmish, not a blind gank
        if not _is_gank(parts, us_pos, ev):
            continue  # killed by your own lane matchup = a trade/duel, not a gank
        decisions.append(_build(conn, match_id, us_id, our_team, t, inter))
    return decisions


def _build(conn, match_id, us_id, our_team, death_ms, inter) -> Decision:
    ws = build_world_state(conn, match_id, death_ms)
    sf = build_state_features(conn, match_id, ws, us_id, our_team)
    jg_unseen = sf["info_risk"].get("enemy_jg_unseen_s")
    enemies_unseen = sf["info_risk"].get("enemies_unseen")

    last_ms = _last_map_interaction_ms(conn, match_id, death_ms)
    since_s = round((death_ms - last_ms) / 1000.0, 1) if last_ms is not None else None

    options = [
        Option(
            "Revisar minimap/cámara antes de comprometer",
            "Ves la posición del jungla y aliados; ajustas el riesgo antes de extender.",
            0.80,
        ),
        Option(
            "Extender/pelear sin info del mapa (lo que hiciste)",
            "Sin lectura del mapa, no anticipas rotaciones; mueres a un ganker o flanqueo.",
            0.20,
        ),
    ]

    context = {
        "time_mmss": _mmss(death_ms),
        "map_interactions_in_window": inter["total"],
        "window_s": AWARENESS_WINDOW_MS // 1000,
        "interaction_breakdown": {
            "minimap_clicks": inter["map_command"],
            "camera_zoom": inter["camera"],
            "center_camera": inter["center_camera"],
        },
        "seconds_since_last_map_interaction": since_s,
        "enemy_jg_unseen_s": jg_unseen,
        "enemies_unseen": enemies_unseen,
        "state_features": sf,
        "action": build_action(DETECTOR_ID, options, [ACTION_CHECK_MAP, ACTION_FIGHT_ENTRY]),
    }

    jg_str = (
        f"El jungla enemigo llevaba {int(jg_unseen)}s sin aparecer"
        if isinstance(jg_unseen, (int, float)) else
        "El jungla enemigo no era visible"
    )
    since_str = (
        f"tu última interacción con el mapa fue hace {since_s}s"
        if since_s is not None else
        "no se registró ninguna interacción con el mapa en toda la partida previa"
    )
    argument = (
        f"A las {context['time_mmss']} moriste sin haber revisado el mapa en los "
        f"{context['window_s']}s previos (0 clics de minimapa, 0 zoom, 0 centrar cámara) — "
        f"{since_str}. {jg_str}: la información para anticipar el peligro estaba a un "
        f"vistazo del minimapa. Este no es un error mecánico sino de hábito: el chequeo "
        f"periódico del mapa, sobre todo al extender o empujar, es lo que separa morir "
        f"a un gank de respetarlo a tiempo."
    )

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=match_id,
        game_time_ms=death_ms,
        moment=f"Muerte sin awareness de mapa a las {context['time_mmss']}",
        outcome=(
            f"Moriste con 0 interacciones de mapa en {context['window_s']}s previos. "
            + (f"Jungla enemigo sin verse {int(jg_unseen)}s." if isinstance(jg_unseen, (int, float)) else "")
        ),
        context=context,
        options=options,
        argument=argument,
    )
