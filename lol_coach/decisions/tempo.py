"""Tempo / map presence analyzer.

Detects moments when the team had a fight you weren't part of, and
argues whether your geographic absence was justified by what you were
doing instead.

Inputs used (no CV needed):
  - timeline_events of type CHAMPION_KILL clustered into fights
  - timeline_frames for your position at the closest minute boundary
  - timeline_events of type ELITE_MONSTER_KILL / BUILDING_KILL near the
    fight location for "objective at stake" inference
  - timeline_frames delta for your CS / jungle CS / gold in that minute
    to infer what you were doing

Limitations:
  - Position is per-minute (you could have already been rotating)
  - Doesn't see pings / chat / cooldowns
  - Travel time assumes constant base movespeed; no boots/MS items
  - Lane phase fights (early game) get evaluated but the could_arrive
    heuristic correctly rejects most as unreachable
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Optional

from ..context import build_world_state, map_lane, map_side
from .base import Decision, Option
from .features import (
    ACTION_FARM_JUNGLE,
    ACTION_PUSH_WAVE,
    ACTION_ROAM,
    build_action,
    build_state_features,
    mmss,
    obj_kind,
)
from .respawn import player_dead_at


DETECTOR_ID = "tempo_v1"

FIGHT_CLUSTER_GAP_S = 15           # Events within this gap form one fight
FIGHT_MIN_KILLS = 2                # Min kills to qualify as a "fight"
FIGHT_MAX_SPREAD_UNITS = 3500      # Cluster kills must be co-located
ASSUMED_MOVE_SPEED_UPS = 380       # ~boots-1
TRAVEL_GRACE_S = 5                 # Champion finishes current action before moving
SKIP_DISTANCE_UNITS = 9000         # Beyond half-map, no realistic rotation
OBJECTIVE_PROXIMITY_UNITS = 3500   # Objective counts as "at stake" within this radius
FIGHT_PRESENCE_UNITS = 2500        # A frame around the fight this close = we were there
SKIP_GAME_TIME_S = 60              # Skip pre-lane shuffling


def analyze_tempo(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    match = conn.execute(
        "SELECT our_puuid FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    us = conn.execute(
        "SELECT participant_id, team_id FROM participants WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    us_id = us["participant_id"]
    our_team = us["team_id"]

    parts = {
        r["participant_id"]: r
        for r in conn.execute(
            "SELECT participant_id, team_id, champion_name FROM participants WHERE match_id = ?",
            (match_id,),
        )
    }

    kills = []
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'CHAMPION_KILL' ORDER BY timestamp_ms",
        (match_id,),
    ):
        ev = json.loads(row["payload_json"])
        ev["_ts"] = row["timestamp_ms"]
        if ev["_ts"] >= SKIP_GAME_TIME_S * 1000:
            kills.append(ev)

    fights = _cluster_fights(kills)

    decisions: list[Decision] = []
    for fight in fights:
        facts = _gather_fight_facts(conn, match_id, fight, us_id, our_team, parts)
        if facts is not None:
            decisions.append(_evaluate(facts))
    return decisions


def _cluster_fights(kills: list[dict]) -> list[list[dict]]:
    """Group kills into fights by temporal AND spatial proximity."""
    fights: list[list[dict]] = []
    current: list[dict] = []

    def _flush():
        nonlocal current
        if len(current) >= FIGHT_MIN_KILLS:
            # Verify spatial proximity
            xs = [k.get("position", {}).get("x") for k in current if k.get("position")]
            ys = [k.get("position", {}).get("y") for k in current if k.get("position")]
            xs = [x for x in xs if x is not None]
            ys = [y for y in ys if y is not None]
            if xs and ys:
                spread = max(
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                )
                if spread <= FIGHT_MAX_SPREAD_UNITS:
                    fights.append(list(current))
        current = []

    for ev in kills:
        if not current:
            current = [ev]
            continue
        if (ev["_ts"] - current[-1]["_ts"]) <= FIGHT_CLUSTER_GAP_S * 1000:
            current.append(ev)
        else:
            _flush()
            current = [ev]
    _flush()
    return fights


def _player_near_fight(conn, match_id, us_id, start_ms, end_ms, fx, fy):
    """Minimum distance from us to the fight across the frames bracketing it.
    Per-minute frames are coarse, so check the closest frame before/after both the
    start and end; if any is close, we were physically present (or arrived)."""
    dists = []
    for ts in (start_ms, end_ms):
        for op, order in ((" <= ", "DESC"), (" >= ", "ASC")):
            r = conn.execute(
                "SELECT position_x, position_y FROM timeline_frames WHERE match_id=? AND "
                "participant_id=? AND timestamp_ms" + op + "? ORDER BY timestamp_ms " + order + " LIMIT 1",
                (match_id, us_id, ts)).fetchone()
            if r and r["position_x"] is not None:
                dists.append(math.hypot(r["position_x"] - fx, r["position_y"] - fy))
    return min(dists) if dists else None


@dataclass
class TempoFacts:
    """Plain, DB-free snapshot of a fight you weren't in — the boundary between
    _gather_fight_facts (all I/O, incl. the None-guards) and _evaluate (pure)."""
    match_id: str
    fight_start_ms: int
    duration_s: float
    fight_x: int
    fight_y: int
    our_team: int
    me_x: int
    me_y: int
    our_kills: int
    our_deaths: int
    our_champs: list
    enemy_champs: list
    distance_u: float
    travel_s: float
    could_arrive: bool
    activity: dict
    objective: Optional[dict]
    stay_action: str
    state_features: dict


def _gather_fight_facts(
    conn: sqlite3.Connection,
    match_id: str,
    fight: list[dict],
    us_id: int,
    our_team: int,
    parts: dict,
) -> Optional[TempoFacts]:
    """The I/O layer for one fight. Returns None when there was no rotate/stay
    decision to make (you were in it, dead, present, or it was unreachable)."""
    fight_start_ms = fight[0]["_ts"]
    fight_end_ms = fight[-1]["_ts"]
    duration_s = (fight_end_ms - fight_start_ms) / 1000.0

    # Did we participate?
    for ev in fight:
        kid = ev.get("killerId")
        vid = ev.get("victimId")
        assists = ev.get("assistingParticipantIds") or []
        if us_id in (kid, vid) or us_id in assists:
            return None  # We were there. Tempo not the issue.

    # Were we dead when the fight started? Then there was no rotate/stay decision.
    if player_dead_at(conn, match_id, us_id, fight_start_ms):
        return None

    # Centroid of kill positions
    xs = [k.get("position", {}).get("x") for k in fight]
    ys = [k.get("position", {}).get("y") for k in fight]
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if not xs:
        return None
    fight_x = sum(xs) // len(xs)
    fight_y = sum(ys) // len(ys)

    # Were we physically AT the fight? Per-minute frames are coarse, so check the
    # frames bracketing it; if any puts us within FIGHT_PRESENCE_UNITS we were
    # there / arrived — not a tempo problem, even if we didn't score a kill/assist
    # (a support shields/CCs without appearing in CHAMPION_KILL). Player feedback:
    # "the fight was right in front of me but it detected me as 'rotate'."
    near = _player_near_fight(conn, match_id, us_id, fight_start_ms, fight_end_ms, fight_x, fight_y)
    if near is not None and near <= FIGHT_PRESENCE_UNITS:
        return None

    # World state at fight start
    ws = build_world_state(conn, match_id, fight_start_ms)
    me = ws.players[us_id]
    if me.pos_x is None:
        return None

    distance_u = math.hypot(me.pos_x - fight_x, me.pos_y - fight_y)
    if distance_u > SKIP_DISTANCE_UNITS:
        return None  # Cross-map, no actionable insight

    travel_s = distance_u / ASSUMED_MOVE_SPEED_UPS
    could_arrive = travel_s <= (duration_s + TRAVEL_GRACE_S)
    if not could_arrive:
        return None  # unreachable even heading straight there: no decision existed
                     # (player 2026-06-10: stop flagging fights I could never reach)

    # Fight outcome from our perspective
    our_kills = 0
    our_deaths = 0
    our_pids: set[int] = set()
    enemy_pids: set[int] = set()
    for ev in fight:
        kid = ev.get("killerId")
        vid = ev.get("victimId")
        assists = ev.get("assistingParticipantIds") or []
        if kid in parts:
            (our_pids if parts[kid]["team_id"] == our_team else enemy_pids).add(kid)
            if parts[kid]["team_id"] == our_team:
                our_kills += 1
        if vid in parts:
            (our_pids if parts[vid]["team_id"] == our_team else enemy_pids).add(vid)
            if parts[vid]["team_id"] == our_team:
                our_deaths += 1
        for a in assists:
            if a in parts:
                (our_pids if parts[a]["team_id"] == our_team else enemy_pids).add(a)

    # Objective at stake?
    objective = _find_objective_near(
        conn, match_id, fight_start_ms, fight_end_ms, fight_x, fight_y, our_team, parts
    )

    # Our activity in the minute containing the fight
    activity = _infer_our_activity(conn, match_id, us_id, fight_start_ms)

    # "Stay" maps to farm_jungle if we were taking camps, else push_wave.
    stay_action = (
        ACTION_FARM_JUNGLE if (activity.get("jungle_delta_in_minute") or 0) > 0
        else ACTION_PUSH_WAVE
    )
    state_features = build_state_features(
        conn, match_id, ws, us_id, our_team,
        next_major_obj=obj_kind((objective or {}).get("subtype")),
        time_to_obj_s=(objective or {}).get("rel_to_fight_s"),
        we_have_setup=could_arrive,
    )

    return TempoFacts(
        match_id=match_id, fight_start_ms=fight_start_ms, duration_s=duration_s,
        fight_x=fight_x, fight_y=fight_y, our_team=our_team,
        me_x=me.pos_x, me_y=me.pos_y, our_kills=our_kills, our_deaths=our_deaths,
        our_champs=sorted(parts[p]["champion_name"] for p in our_pids),
        enemy_champs=sorted(parts[p]["champion_name"] for p in enemy_pids),
        distance_u=distance_u, travel_s=travel_s, could_arrive=could_arrive,
        activity=activity, objective=objective, stay_action=stay_action,
        state_features=state_features,
    )


def _evaluate(facts: TempoFacts) -> Decision:
    """Pure decision logic: options, context, argument, Decision. No conn."""
    context = {
        "time_mmss": mmss(facts.fight_start_ms),
        "fight_duration_s": round(facts.duration_s, 1),
        "fight_position": {
            "x": facts.fight_x,
            "y": facts.fight_y,
            "lane": map_lane(facts.fight_x, facts.fight_y),
            "side": map_side(facts.fight_x, facts.fight_y, facts.our_team),
        },
        "fight_outcome_for_us": f"{facts.our_kills}-{facts.our_deaths}",
        "fight_participants": {
            "our_team": facts.our_champs,
            "enemy": facts.enemy_champs,
        },
        "your_position_at_fight_start": {
            "x": facts.me_x,
            "y": facts.me_y,
            "lane": map_lane(facts.me_x, facts.me_y),
            "side": map_side(facts.me_x, facts.me_y, facts.our_team),
        },
        "distance_to_fight_units": int(facts.distance_u),
        "estimated_travel_s": round(facts.travel_s, 1),
        "could_arrive_in_time": facts.could_arrive,
        "your_activity": facts.activity,
        "objective_at_stake": facts.objective,
    }

    options = _build_options(
        facts.could_arrive, facts.our_kills, facts.our_deaths, facts.objective,
        facts.activity, facts.distance_u, facts.duration_s
    )

    context["state_features"] = facts.state_features
    context["action"] = build_action(
        DETECTOR_ID, options, [ACTION_ROAM, facts.stay_action]
    )

    argument = _compose_argument(context, options)

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=facts.match_id,
        game_time_ms=facts.fight_start_ms,
        moment=(
            f"Pelea {facts.our_kills}-{facts.our_deaths} en "
            f"{context['fight_position']['lane']} ({context['fight_position']['side']}) sin ti"
        ),
        outcome=(
            f"Tu equipo "
            f"{'ganó' if facts.our_kills > facts.our_deaths else 'perdió' if facts.our_kills < facts.our_deaths else 'empató'} "
            f"{facts.our_kills}-{facts.our_deaths}. Estabas a {int(facts.distance_u)}u "
            f"en {context['your_position_at_fight_start']['lane']}."
        ),
        context=context,
        options=options,
        argument=argument,
    )


def _find_objective_near(
    conn: sqlite3.Connection,
    match_id: str,
    fight_start_ms: int,
    fight_end_ms: int,
    fight_x: int,
    fight_y: int,
    our_team: int,
    parts: dict,
) -> dict | None:
    rows = conn.execute(
        "SELECT timestamp_ms, type, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type IN ('ELITE_MONSTER_KILL', 'BUILDING_KILL') "
        "AND timestamp_ms BETWEEN ? AND ?",
        (match_id, fight_start_ms - 90_000, fight_end_ms + 90_000),
    ).fetchall()
    best = None
    best_d = None
    for row in rows:
        ev = json.loads(row["payload_json"])
        pos = ev.get("position") or {}
        if pos.get("x") is None:
            continue
        d = math.hypot(pos["x"] - fight_x, pos["y"] - fight_y)
        if d > OBJECTIVE_PROXIMITY_UNITS:
            continue
        if best_d is None or d < best_d:
            best_d = d
            killer_id = ev.get("killerId")
            taken_by_team = (
                parts[killer_id]["team_id"]
                if killer_id and killer_id in parts
                else ev.get("killerTeamId")
            )
            best = {
                "type": row["type"],
                "ts_ms": row["timestamp_ms"],
                "rel_to_fight_s": round((row["timestamp_ms"] - fight_start_ms) / 1000.0, 1),
                "subtype": (
                    ev.get("monsterType")
                    or ev.get("monsterSubType")
                    or ev.get("buildingType")
                ),
                "taken_by": "tu equipo" if taken_by_team == our_team else "enemigo",
            }
    return best


def _infer_our_activity(
    conn: sqlite3.Connection, match_id: str, us_id: int, fight_start_ms: int
) -> dict:
    before = conn.execute(
        "SELECT minions_killed, jungle_minions_killed, total_gold, current_gold, timestamp_ms "
        "FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, us_id, fight_start_ms),
    ).fetchone()
    after = conn.execute(
        "SELECT minions_killed, jungle_minions_killed, total_gold, current_gold, timestamp_ms "
        "FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms > ? "
        "ORDER BY timestamp_ms ASC LIMIT 1",
        (match_id, us_id, fight_start_ms),
    ).fetchone()
    if not before or not after:
        return {"description": "actividad desconocida (datos incompletos)"}

    cs = (after["minions_killed"] or 0) - (before["minions_killed"] or 0)
    jg = (after["jungle_minions_killed"] or 0) - (before["jungle_minions_killed"] or 0)
    gold = (after["total_gold"] or 0) - (before["total_gold"] or 0)

    if cs >= 8:
        desc = f"farmeando línea (+{cs} CS en el minuto)"
    elif jg >= 3:
        desc = f"camps de jungla (+{jg})"
    elif gold > 400:
        desc = f"actividad alta sin farm directo (+{gold} oro, posible ward/objetivo)"
    elif gold < 100:
        desc = "muy poca actividad (rotando, vagando, o esperando recall)"
    else:
        desc = f"actividad mixta (+{cs} CS, +{jg} jg, +{gold} oro)"

    return {
        "description": desc,
        "cs_delta_in_minute": cs,
        "jungle_delta_in_minute": jg,
        "gold_delta_in_minute": gold,
    }


def _build_options(
    could_arrive: bool,
    our_kills: int,
    our_deaths: int,
    objective: dict | None,
    activity: dict,
    distance_u: float,
    duration_s: float,
) -> list[Option]:
    fight_lost = our_kills < our_deaths
    fight_dominant_win = our_kills >= our_deaths + 2

    # Option: rotate to the fight
    rotate = 0.45
    if could_arrive:
        rotate += 0.20
    if objective and objective.get("taken_by") == "enemigo":
        rotate += 0.20
    if fight_lost:
        rotate += 0.10
    if fight_dominant_win:
        rotate -= 0.15  # Team won decisively; your absence didn't cost
    rotate = max(0.05, min(1.0, rotate))

    # Option: stay (what you did)
    stay = 0.45
    if not could_arrive:
        stay += 0.30  # You literally couldn't help
    if fight_dominant_win:
        stay += 0.15
    if fight_lost and objective and objective.get("taken_by") == "enemigo":
        stay -= 0.25
    if "farmeando" in activity.get("description", "").lower() and (objective or fight_lost):
        stay -= 0.10
    if "muy poca actividad" in activity.get("description", "").lower():
        stay -= 0.20  # No excuse to not be there
    stay = max(0.05, min(1.0, stay))

    travel_str = f"{round(distance_u / ASSUMED_MOVE_SPEED_UPS, 1)}s de travel"
    if not could_arrive:
        travel_str += f" — más largo que los {duration_s:.1f}s que duró la pelea"
    obj_str = ""
    if objective:
        who = objective["taken_by"]
        obj_str = f" Objetivo cercano: {objective['subtype'] or objective['type']} (se lo llevó {who})."

    return [
        Option(
            label="Rotar al fight",
            predicted_consequence=f"{travel_str}.{obj_str}",
            ev_score=round(rotate, 2),
        ),
        Option(
            label="Permanecer donde estabas (lo que hiciste)",
            predicted_consequence=(
                activity.get("description", "actividad desconocida")
                + (". Pelea perdida sin ti." if fight_lost else ". Pelea ganada igual sin ti." if fight_dominant_win else ".")
            ),
            ev_score=round(stay, 2),
        ),
    ]


def _compose_argument(ctx: dict, options: list[Option]) -> str:
    lines: list[str] = []
    fp = ctx["fight_position"]
    yp = ctx["your_position_at_fight_start"]

    outcome_word = (
        "ganó" if int(ctx["fight_outcome_for_us"].split("-")[0]) > int(ctx["fight_outcome_for_us"].split("-")[1])
        else "perdió" if int(ctx["fight_outcome_for_us"].split("-")[0]) < int(ctx["fight_outcome_for_us"].split("-")[1])
        else "empató"
    )
    lines.append(
        f"A las {ctx['time_mmss']} hubo una pelea de tu equipo en {fp['lane']} ({fp['side']}), "
        f"duración {ctx['fight_duration_s']}s, resultado {ctx['fight_outcome_for_us']} "
        f"({outcome_word})."
    )

    lines.append(
        f"Tú estabas en {yp['lane']} ({yp['side']}) a {ctx['distance_to_fight_units']} unidades "
        f"del fight (~{ctx['estimated_travel_s']}s de travel a movespeed base con botas)."
    )

    if not ctx["could_arrive_in_time"]:
        lines.append(
            "El travel time excede la duración de la pelea — no podías haber llegado "
            "a tiempo aunque hubieras decidido rotar al primer kill."
        )

    lines.append(f"Tu actividad: {ctx['your_activity']['description']}.")

    if ctx["objective_at_stake"]:
        obj = ctx["objective_at_stake"]
        rel_s = obj["rel_to_fight_s"]
        when = (
            f"durante la pelea" if -5 <= rel_s <= ctx['fight_duration_s'] + 5
            else f"{rel_s:+.0f}s respecto al fight"
        )
        lines.append(
            f"Objetivo en juego: {obj['subtype'] or obj['type']} ({when}). "
            f"Se lo llevó {obj['taken_by']}."
        )

    parts = ctx["fight_participants"]
    if parts["our_team"] or parts["enemy"]:
        lines.append(
            f"Composición de la pelea: tu equipo [{', '.join(parts['our_team']) or '—'}] "
            f"vs [{', '.join(parts['enemy']) or '—'}]."
        )

    rotate, stay = options[0], options[1]
    if rotate.ev_score > stay.ev_score:
        lines.append(
            f"Opción de mayor EV: \"{rotate.label}\" (EV {rotate.ev_score:.2f}) vs "
            f"lo que hiciste \"{stay.label}\" (EV {stay.ev_score:.2f}). "
            f"Diferencial +{rotate.ev_score - stay.ev_score:.2f}."
        )
    elif stay.ev_score > rotate.ev_score:
        lines.append(
            f"Tu decisión de quedarte estaba justificada según la heurística "
            f"(stay EV {stay.ev_score:.2f} vs rotate EV {rotate.ev_score:.2f}). "
            "El sistema no identifica problema de tempo aquí."
        )

    return " ".join(lines)
