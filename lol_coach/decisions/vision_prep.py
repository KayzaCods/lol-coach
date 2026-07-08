"""Vision preparation analyzer for major neutral objectives.

For each ELITE_MONSTER_KILL of a contestable objective (dragon, herald,
baron, atakhan), evaluates what was done for vision in the 90 seconds
leading up to the kill:

  - Team wards placed near the pit (approximated via placer's frame position)
  - Team sweeps of enemy wards near the pit (via killer's frame position)
  - Your personal contribution as the configured user
  - Enemy wards placed near the pit (proxy for enemy vision setup)

Limitations:
  - Riot's WARD_PLACED and WARD_KILL events do NOT carry positions
    (only creatorId / killerId + wardType). Ward locations are
    approximated from the placer/killer's position at the closest
    timeline_frame (1-minute granularity).
  - This approximation works best for sustained presence (support
    hovering river around objective time stays near the pit).
"""
from __future__ import annotations

import json
import math
import sqlite3

from ..context import build_world_state, map_lane
from .base import Decision, Option
from .features import (
    ACTION_IGNORE_OBJECTIVE,
    ACTION_VISION_SETUP,
    build_action,
    build_state_features,
    obj_kind,
)


DETECTOR_ID = "vision_prep_v1"
PREP_WINDOW_MS = 90_000          # 1:30 window before the objective kill
NEAR_PIT_UNITS = 3500            # ward counted as "near pit" if placer within this radius

# Objectives we evaluate. HORDE (Void Grubs) is excluded because Riot
# emits one event per individual grub killed, which generates redundant
# decisions within the same 90s prep window. They're a soft objective
# anyway — not a teamfight focal point.
EVAL_OBJECTIVES = {"DRAGON", "RIFTHERALD", "BARON_NASHOR", "ATAKHAN"}

OBJECTIVE_NAMES = {
    ("DRAGON", "HEXTECH_DRAGON"): "Dragón Hextech",
    ("DRAGON", "INFERNAL_DRAGON"): "Dragón Infernal",
    ("DRAGON", "FIRE_DRAGON"): "Dragón Infernal",
    ("DRAGON", "OCEAN_DRAGON"): "Dragón Oceánico",
    ("DRAGON", "WATER_DRAGON"): "Dragón Oceánico",
    ("DRAGON", "MOUNTAIN_DRAGON"): "Dragón de Montaña",
    ("DRAGON", "EARTH_DRAGON"): "Dragón de Montaña",
    ("DRAGON", "CLOUD_DRAGON"): "Dragón Nube",
    ("DRAGON", "AIR_DRAGON"): "Dragón Nube",
    ("DRAGON", "CHEMTECH_DRAGON"): "Dragón Químico",
    ("DRAGON", "ELDER_DRAGON"): "Dragón Ancestro",
    ("RIFTHERALD", None): "Heraldo",
    ("BARON_NASHOR", None): "Barón Nashor",
    ("HORDE", None): "Gusanos del Vacío",
    ("ATAKHAN", None): "Atakhan",
}


def _mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def _objective_name(mt: str | None, sub: str | None) -> str:
    if (mt, sub) in OBJECTIVE_NAMES:
        return OBJECTIVE_NAMES[(mt, sub)]
    if (mt, None) in OBJECTIVE_NAMES:
        return OBJECTIVE_NAMES[(mt, None)]
    return mt or "objetivo"


def _player_position_at(
    conn: sqlite3.Connection, match_id: str, pid: int, ts_ms: int
) -> tuple[int, int] | None:
    """Closest timeline_frame position at or before ts_ms (fallback to after)."""
    row = conn.execute(
        "SELECT position_x, position_y FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, pid, ts_ms),
    ).fetchone()
    if row and row["position_x"] is not None:
        return (row["position_x"], row["position_y"])
    row = conn.execute(
        "SELECT position_x, position_y FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms > ? "
        "ORDER BY timestamp_ms ASC LIMIT 1",
        (match_id, pid, ts_ms),
    ).fetchone()
    if row and row["position_x"] is not None:
        return (row["position_x"], row["position_y"])
    return None


def analyze_vision_prep(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    match = conn.execute(
        "SELECT our_puuid FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    us = conn.execute(
        "SELECT participant_id, team_id, team_position FROM participants "
        "WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    us_id = us["participant_id"]
    our_team = us["team_id"]
    user_is_support = us["team_position"] == "UTILITY"

    parts = {
        r["participant_id"]: r
        for r in conn.execute(
            "SELECT participant_id, team_id, champion_name, team_position "
            "FROM participants WHERE match_id = ?",
            (match_id,),
        )
    }

    kills = []
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'ELITE_MONSTER_KILL' ORDER BY timestamp_ms",
        (match_id,),
    ):
        ev = json.loads(row["payload_json"])
        if ev.get("monsterType") not in EVAL_OBJECTIVES:
            continue
        ev["_ts"] = row["timestamp_ms"]
        kills.append(ev)

    decisions = []
    for kill in kills:
        d = _evaluate_objective(
            conn, match_id, kill, us_id, our_team, user_is_support, parts
        )
        if d is not None:
            decisions.append(d)
    return decisions


def _evaluate_objective(
    conn, match_id, kill, us_id, our_team, user_is_support, parts
) -> Decision | None:
    kill_ts = kill["_ts"]
    pit_pos = kill.get("position") or {}
    pit_x = pit_pos.get("x")
    pit_y = pit_pos.get("y")
    if pit_x is None:
        return None

    window_start = kill_ts - PREP_WINDOW_MS

    ward_placed_rows = conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'WARD_PLACED' "
        "AND timestamp_ms BETWEEN ? AND ? ORDER BY timestamp_ms",
        (match_id, window_start, kill_ts),
    ).fetchall()
    ward_killed_rows = conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'WARD_KILL' "
        "AND timestamp_ms BETWEEN ? AND ? ORDER BY timestamp_ms",
        (match_id, window_start, kill_ts),
    ).fetchall()

    team_wards: list[dict] = []
    enemy_wards: list[dict] = []
    team_sweeps: list[dict] = []
    your_wards: list[dict] = []
    your_sweeps: list[dict] = []

    for row in ward_placed_rows:
        ev = json.loads(row["payload_json"])
        cid = ev.get("creatorId")
        # Skip non-ward entities (Yorick ghouls, Bel'Veth voidlings, etc.) that
        # the Riot API also emits as WARD_PLACED with wardType=UNDEFINED.
        if ev.get("wardType") == "UNDEFINED":
            continue
        if not cid or cid not in parts:
            continue
        pos = _player_position_at(conn, match_id, cid, row["timestamp_ms"])
        if not pos:
            continue
        d = math.hypot(pos[0] - pit_x, pos[1] - pit_y)
        if d > NEAR_PIT_UNITS:
            continue
        entry = {
            "by_champion": parts[cid]["champion_name"],
            "by_pid": cid,
            "ts_ms": row["timestamp_ms"],
            "rel_to_kill_s": round((row["timestamp_ms"] - kill_ts) / 1000.0, 1),
            "ward_type": ev.get("wardType"),
            "approx_dist_from_pit": int(d),
        }
        if parts[cid]["team_id"] == our_team:
            team_wards.append(entry)
            if cid == us_id:
                your_wards.append(entry)
        else:
            enemy_wards.append(entry)

    for row in ward_killed_rows:
        ev = json.loads(row["payload_json"])
        kid = ev.get("killerId")
        if not kid or kid not in parts:
            continue
        pos = _player_position_at(conn, match_id, kid, row["timestamp_ms"])
        if not pos:
            continue
        d = math.hypot(pos[0] - pit_x, pos[1] - pit_y)
        if d > NEAR_PIT_UNITS:
            continue
        entry = {
            "by_champion": parts[kid]["champion_name"],
            "by_pid": kid,
            "ts_ms": row["timestamp_ms"],
            "rel_to_kill_s": round((row["timestamp_ms"] - kill_ts) / 1000.0, 1),
            "ward_type": ev.get("wardType"),
        }
        if parts[kid]["team_id"] == our_team:
            team_sweeps.append(entry)
            if kid == us_id:
                your_sweeps.append(entry)

    killer_id = kill.get("killerId")
    obj_taken_by_us = (
        parts[killer_id]["team_id"] == our_team if killer_id in parts
        else kill.get("killerTeamId") == our_team
    )

    objective_name = _objective_name(kill.get("monsterType"), kill.get("monsterSubType"))

    context = {
        "time_mmss": _mmss(kill_ts),
        "objective": objective_name,
        "objective_type": kill.get("monsterType"),
        "objective_subtype": kill.get("monsterSubType"),
        "objective_position": {
            "x": pit_x, "y": pit_y,
            "lane": map_lane(pit_x, pit_y),
        },
        "taken_by": "tu equipo" if obj_taken_by_us else "enemigo",
        "prep_window_s": PREP_WINDOW_MS // 1000,
        "team_wards_near_pit": len(team_wards),
        "your_wards_near_pit": len(your_wards),
        "enemy_wards_near_pit": len(enemy_wards),
        "team_sweeps_near_pit": len(team_sweeps),
        "your_sweeps_near_pit": len(your_sweeps),
        "vision_asymmetry": len(team_wards) - len(enemy_wards),
        "your_role_is_support": user_is_support,
        "your_wards_detail": [
            {"ts_mmss": _mmss(w["ts_ms"]), "type": w["ward_type"], "approx_dist": w["approx_dist_from_pit"]}
            for w in your_wards
        ],
        "team_wards_detail": [
            {"by": w["by_champion"], "ts_mmss": _mmss(w["ts_ms"]), "type": w["ward_type"]}
            for w in team_wards if w["by_pid"] != us_id
        ],
    }

    options = _build_options(context, obj_taken_by_us)

    # Standardized blocks. Taken action is vision_setup if we contributed any
    # vision in the window, else ignore_objective.
    ws = build_world_state(conn, match_id, kill_ts)
    contributed = (context["your_wards_near_pit"] + context["your_sweeps_near_pit"]) > 0
    context["state_features"] = build_state_features(
        conn, match_id, ws, us_id, our_team,
        next_major_obj=obj_kind(kill.get("monsterType")),
        time_to_obj_s=0,
        we_have_setup=context["vision_asymmetry"] >= 1,
    )
    context["action"] = build_action(
        DETECTOR_ID, options,
        [ACTION_VISION_SETUP, ACTION_VISION_SETUP if contributed else ACTION_IGNORE_OBJECTIVE],
    )

    argument = _compose_argument(context, options)

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=match_id,
        game_time_ms=kill_ts,
        moment=f"Preparación de visión — {objective_name}",
        outcome=(
            f"{objective_name} cae a {context['taken_by']}. "
            f"Team: {context['team_wards_near_pit']}w / {context['team_sweeps_near_pit']}sw. "
            f"Tú: {context['your_wards_near_pit']}w / {context['your_sweeps_near_pit']}sw. "
            f"Enemigo: {context['enemy_wards_near_pit']}w."
        ),
        context=context,
        options=options,
        argument=argument,
    )


def _build_options(ctx, taken_by_us) -> list[Option]:
    yw = ctx["your_wards_near_pit"]
    ys = ctx["your_sweeps_near_pit"]
    tw = ctx["team_wards_near_pit"]
    ew = ctx["enemy_wards_near_pit"]
    is_support = ctx["your_role_is_support"]

    # Ideal prep score (active vision setup)
    ideal_ev = 0.85

    # Your actual contribution
    if is_support:
        if yw == 0 and ys == 0:
            actual = 0.15
        elif yw >= 1 and ys == 0:
            actual = 0.45
        elif yw == 0 and ys >= 1:
            actual = 0.45
        elif yw >= 2 and ys >= 1:
            actual = 0.85
        else:
            actual = 0.55
    else:
        # Non-support: lower expectations
        if yw == 0 and ys == 0:
            actual = 0.40
        elif yw >= 1 or ys >= 1:
            actual = 0.65
        else:
            actual = 0.50

    if yw >= 2 and ys >= 1 and taken_by_us:
        actual = max(actual, 0.90)

    return [
        Option(
            label=f"Setup activo: 2+ wards en pit perimeter + 1+ sweep en ventana de {ctx['prep_window_s']}s",
            predicted_consequence=(
                "Cobertura del pit (4 lados) y denegación de info al enemigo. "
                "Permite que tu equipo entre al fight con info asimétrica a favor."
            ),
            ev_score=round(ideal_ev, 2),
        ),
        Option(
            label=f"Lo que hiciste: {yw} wards / {ys} sweeps cerca del pit",
            predicted_consequence=(
                "Sin contribución de visión personal en la ventana."
                if yw + ys == 0
                else f"Contribución parcial. Equipo total: {tw}w / {ctx['team_sweeps_near_pit']}sw. "
                f"Enemigo tenía {ew}w en zona (lo que sabes)."
            ),
            ev_score=round(actual, 2),
        ),
    ]


def _compose_argument(ctx, options) -> str:
    lines: list[str] = []
    lines.append(
        f"Preparación para {ctx['objective']} a las {ctx['time_mmss']} "
        f"(ventana de {ctx['prep_window_s']}s previa al kill)."
    )
    lines.append(
        f"Tu equipo: {ctx['team_wards_near_pit']} wards puestos cerca del pit, "
        f"{ctx['team_sweeps_near_pit']} wards enemigos barridos. "
        f"Enemigo: {ctx['enemy_wards_near_pit']} wards en la zona "
        f"(diferencial de visión: {ctx['vision_asymmetry']:+d})."
    )

    yw = ctx["your_wards_near_pit"]
    ys = ctx["your_sweeps_near_pit"]
    is_support = ctx["your_role_is_support"]

    if is_support:
        if yw == 0 and ys == 0:
            lines.append(
                "Como SUPPORT, no contribuiste con visión al objetivo (0 wards / 0 sweeps "
                "en zona, ventana de 90s). El rol esperaba al menos 1-2 wards en el pit "
                "perimeter + sweep si había wards enemigos."
            )
        elif yw >= 2 and ys >= 1:
            lines.append(
                f"Setup ejecutado correctamente para tu rol: {yw} wards + {ys} sweeps. "
                + ("Convertiste la prep en kill del objetivo." if ctx["taken_by"] == "tu equipo"
                   else "Pese a buena prep, el objetivo se perdió — diagnóstico se mueve a ejecución del fight, no a la preparación.")
            )
        else:
            lines.append(
                f"Contribución parcial como support: {yw} wards / {ys} sweeps. "
                f"Para un objetivo de este peso lo ideal es 2+ wards cubriendo entradas "
                f"al pit + sweep de wards enemigos visibles."
            )
    else:
        if yw + ys > 0:
            lines.append(
                f"Aportaste {yw} wards / {ys} sweeps cerca del pit. "
                f"La responsabilidad principal de la visión del objetivo recae en el support."
            )

    if ctx["vision_asymmetry"] < 0:
        lines.append(
            f"Visión asimétrica DESFAVORABLE: el enemigo tenía más wards en zona "
            f"({ctx['enemy_wards_near_pit']} vs {ctx['team_wards_near_pit']}). "
            f"Sin sweep activo, peleaste con desventaja de info."
        )

    lines.append(
        f"Resultado: {ctx['objective']} cae a {ctx['taken_by']}."
    )

    return " ".join(lines)
