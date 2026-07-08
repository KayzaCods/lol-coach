"""Objective-readiness analyzer.

For each major neutral objective taken in the match, evaluates your state
at the moment of the kill:

  - Distance from the pit (were you there?)
  - HP / mana % from Live Client snapshot at the kill instant
  - Recall in the prep window (gold spike + base-area position)
  - Items completion progress
  - Number of completed items vs match-average pace

Limitations:
  - Live Client snapshots are 1Hz; "at the kill instant" finds the
    closest snapshot.
  - We don't have ability cooldown precision (Live Client exposes
    abilityLevel and championStats but not "secondsRemaining").
  - Recall detection is heuristic: large gold drop + position close to
    base area within a short window suggests recall completed.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..context import build_world_state
from .base import Decision, Option
from .features import (
    ACTION_FIGHT_ENTRY,
    ACTION_IGNORE_OBJECTIVE,
    ACTION_VISION_SETUP,
    build_action,
    build_state_features,
    mmss,
    obj_kind,
)
from ..game_data import has_meaningful_resource
from ..snapshots import load_snapshots
from .respawn import player_dead_at, last_death_before


DETECTOR_ID = "objective_readiness_v1"

PREP_WINDOW_MS = 90_000
NEAR_PIT_UNITS = 2500            # "at the pit" radius
CONTEST_DEATH_UNITS = 2500       # died within this of the pit = died contesting (not a spectator)
WARNING_DISTANCE_UNITS = 5000    # beyond this you needed a long rotation
HP_LOW_PCT = 0.50
MANA_LOW_PCT = 0.40
RECALL_GOLD_DROP = 600           # heuristic: gold spent on recall is at least this

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
    ("ATAKHAN", None): "Atakhan",
}


def _objective_name(mt, sub) -> str:
    if (mt, sub) in OBJECTIVE_NAMES:
        return OBJECTIVE_NAMES[(mt, sub)]
    if (mt, None) in OBJECTIVE_NAMES:
        return OBJECTIVE_NAMES[(mt, None)]
    return mt or "objetivo"


def _snapshot_at(snapshots, target_t_s: float) -> dict | None:
    """Find snapshot closest in time to target_t_s."""
    if not snapshots:
        return None
    best = None
    best_delta = None
    for t, data in snapshots:
        d = abs(t - target_t_s)
        if best_delta is None or d < best_delta:
            best_delta = d
            best = data
        if t > target_t_s + 5:
            break
    return best if best_delta is not None and best_delta <= 5 else best  # within 5s ideally


def _our_items_count(snapshot: dict, us_riot_name: str) -> int:
    """Count distinct legendary items in our inventory (slot 0-5, item id != consumables/wards)."""
    if not snapshot:
        return 0
    for p in snapshot.get("allPlayers") or []:
        if p.get("riotIdGameName") == us_riot_name:
            count = 0
            for it in p.get("items") or []:
                slot = it.get("slot")
                iid = it.get("itemID")
                if slot is None or slot >= 6:
                    continue  # trinket / overflow
                if iid is None:
                    continue
                count += 1
            return count
    return 0


VISION_NEAR_PIT_UNITS = 3500      # ward counted near the pit if placer within this


def _player_position_at(conn, match_id, pid, ts_ms):
    """Closest timeline_frame position at/before ts_ms (fallback to after)."""
    for op, order in ((" <= ", "DESC"), (" > ", "ASC")):
        row = conn.execute(
            "SELECT position_x, position_y FROM timeline_frames "
            "WHERE match_id=? AND participant_id=? AND timestamp_ms" + op + "? "
            "ORDER BY timestamp_ms " + order + " LIMIT 1",
            (match_id, pid, ts_ms)).fetchone()
        if row and row["position_x"] is not None:
            return (row["position_x"], row["position_y"])
    return None


def _vision_near_pit(conn, match_id, kill_ts, pit_x, pit_y, us_id, our_team, parts):
    """Vision setup near the pit in the 90s before the kill (absorbed from the old
    vision_prep detector): team/your/enemy wards placed + sweeps near the pit."""
    ws = kill_ts - PREP_WINDOW_MS
    tw = yw = ew = swt = sws = 0
    ydetail = []
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events WHERE match_id=? "
        "AND type='WARD_PLACED' AND timestamp_ms BETWEEN ? AND ?", (match_id, ws, kill_ts)):
        ev = json.loads(row["payload_json"])
        if ev.get("wardType") == "UNDEFINED":
            continue
        cid = ev.get("creatorId")
        if not cid or cid not in parts:
            continue
        pos = _player_position_at(conn, match_id, cid, row["timestamp_ms"])
        if not pos or math.hypot(pos[0] - pit_x, pos[1] - pit_y) > VISION_NEAR_PIT_UNITS:
            continue
        if parts[cid]["team_id"] == our_team:
            tw += 1
            if cid == us_id:
                yw += 1
                ydetail.append({"ts": mmss(row["timestamp_ms"]), "type": ev.get("wardType")})
        else:
            ew += 1
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events WHERE match_id=? "
        "AND type='WARD_KILL' AND timestamp_ms BETWEEN ? AND ?", (match_id, ws, kill_ts)):
        ev = json.loads(row["payload_json"])
        kid = ev.get("killerId")
        if not kid or kid not in parts:
            continue
        pos = _player_position_at(conn, match_id, kid, row["timestamp_ms"])
        if not pos or math.hypot(pos[0] - pit_x, pos[1] - pit_y) > VISION_NEAR_PIT_UNITS:
            continue
        if parts[kid]["team_id"] == our_team:
            swt += 1
            if kid == us_id:
                sws += 1
    return {
        "team_wards_near_pit": tw, "your_wards_near_pit": yw,
        "enemy_wards_near_pit": ew, "team_sweeps_near_pit": swt,
        "your_sweeps_near_pit": sws, "vision_diff": tw - ew,
        "your_wards_detail": ydetail,
    }


def _team_gold_diff(conn, match_id, our_team, at_ms):
    """Our team's total gold minus the enemy team's at the nearest frame <= at_ms."""
    frame_ts = conn.execute(
        "SELECT MAX(timestamp_ms) m FROM timeline_frames WHERE match_id=? AND timestamp_ms<=?",
        (match_id, at_ms)).fetchone()["m"]
    if frame_ts is None:
        return None
    tg = {}
    for r in conn.execute(
        "SELECT p.team_id tid, tf.total_gold g FROM timeline_frames tf "
        "JOIN participants p ON p.match_id=tf.match_id AND p.participant_id=tf.participant_id "
        "WHERE tf.match_id=? AND tf.timestamp_ms=?", (match_id, frame_ts)):
        tg[r["tid"]] = tg.get(r["tid"], 0) + (r["g"] or 0)
    enemy = 200 if our_team == 100 else 100
    return tg.get(our_team, 0) - tg.get(enemy, 0)


def _objective_contested(conn, match_id, kill_ts, pit_x, pit_y, our_team=None) -> bool:
    """Was the objective actually disputed? A champion kill near the pit (~3000u)
    in [-30s,+10s] = a fight. ALSO: enemies merely PRESENT near the pit count as
    contested-by-presence — they were poised to fight and your team's presence
    deterred them (player feedback: 'no fue gratis, se acercaron intentando
    contestarlo pero no pudieron; mi presencia sí tenía valor')."""
    for r in conn.execute(
        "SELECT payload_json FROM timeline_events WHERE match_id=? AND type='CHAMPION_KILL' "
        "AND timestamp_ms BETWEEN ? AND ?", (match_id, kill_ts - 30000, kill_ts + 10000)):
        ev = json.loads(r["payload_json"])
        pos = ev.get("position") or {}
        if pos.get("x") is not None and math.hypot(pos["x"] - pit_x, pos["y"] - pit_y) <= 3000:
            return True
    if our_team is not None:
        for p in conn.execute(
            "SELECT participant_id FROM participants WHERE match_id=? AND team_id<>?",
            (match_id, our_team),
        ):
            fr = conn.execute(
                "SELECT position_x x, position_y y FROM timeline_frames WHERE match_id=? AND "
                "participant_id=? ORDER BY ABS(timestamp_ms-?) LIMIT 1",
                (match_id, p["participant_id"], kill_ts),
            ).fetchone()
            if fr and fr["x"] is not None and math.hypot(fr["x"] - pit_x, fr["y"] - pit_y) <= 3200:
                return True  # an enemy was hovering the pit: deterred, not free
    return False


_COUNTERPART_NAMES = {
    "HORDE": "las larvas (voidgrubs)", "RIFTHERALD": "el Heraldo", "DRAGON": "un dragón",
    "BARON_NASHOR": "el Barón", "ATAKHAN": "Atakhan",
    "TOWER_BUILDING": "una torre", "INHIBITOR_BUILDING": "un inhibidor",
}


def _counterpart_objective(conn, match_id, kill_ts, our_team, exclude_ts=None) -> str | None:
    """Something OUR team took around the same time (±75s): voidgrubs, herald,
    a dragon, a tower... If present, ceding the evaluated objective was an
    objective-for-objective TRADE, not a giveaway."""
    our_pids = {r["participant_id"] for r in conn.execute(
        "SELECT participant_id FROM participants WHERE match_id=? AND team_id=?",
        (match_id, our_team))}
    for r in conn.execute(
        "SELECT timestamp_ms, type, payload_json FROM timeline_events WHERE match_id=? AND "
        "type IN ('ELITE_MONSTER_KILL','BUILDING_KILL') AND timestamp_ms BETWEEN ? AND ?",
        (match_id, kill_ts - 75000, kill_ts + 75000),
    ):
        if exclude_ts is not None and r["timestamp_ms"] == exclude_ts:
            continue
        ev = json.loads(r["payload_json"])
        if r["type"] == "ELITE_MONSTER_KILL":
            if ev.get("killerId") not in our_pids and ev.get("killerTeamId") != our_team:
                continue
            kind = ev.get("monsterType")
        else:
            # BUILDING_KILL: teamId is the OWNER of the destroyed building
            if ev.get("teamId") == our_team:
                continue
            kind = ev.get("buildingType")
        name = _COUNTERPART_NAMES.get(kind)
        if name:
            return name
    return None


def _objective_value(conn, match_id: str, kill: dict) -> float:
    """How much this objective is worth right now (0..1), per the player's own
    scaling: 3rd dragon > the first two, soul (4th) >> , Elder is double-edged,
    Baron high. ('close-out > objective' by game state is a later refinement.)"""
    mt = kill.get("monsterType")
    if mt == "BARON_NASHOR":
        return 0.80
    if mt == "RIFTHERALD":
        return 0.50
    if mt == "DRAGON":
        if kill.get("monsterSubType") == "ELDER_DRAGON":
            return 0.70  # double-edged: strong, but losing it / overstaying hurts
        prior = conn.execute(
            "SELECT COUNT(*) c FROM timeline_events WHERE match_id = ? "
            "AND type = 'ELITE_MONSTER_KILL' AND timestamp_ms < ? "
            "AND payload_json LIKE '%\"monsterType\":\"DRAGON\"%'",
            (match_id, kill["_ts"]),
        ).fetchone()["c"]
        n = prior + 1
        return 0.40 if n <= 2 else 0.60 if n == 3 else 0.85  # 4th ~ soul
    return 0.50


def analyze_objective_readiness(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    match = conn.execute(
        "SELECT our_puuid, session_dir FROM matches WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    if not match:
        return []
    session_dir = Path(match["session_dir"]) if match["session_dir"] else None
    snapshots = load_snapshots(session_dir) if session_dir else []

    us = conn.execute(
        "SELECT participant_id, team_id, team_position, champion_name FROM participants "
        "WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    us_id = us["participant_id"]
    our_team = us["team_id"]
    champion_name = us["champion_name"] or ""

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
        if not ev.get("killerId") and ev.get("killerTeamId") not in (100, 200):
            continue  # natural despawn (e.g. Herald at 19:45: killerId 0, teamId 300) — nobody took it
        ev["_ts"] = row["timestamp_ms"]
        kills.append(ev)

    decisions = []
    for kill in kills:
        facts = _gather_objective_facts(
            conn, match_id, kill, us_id, our_team, parts, snapshots, champion_name
        )
        if facts is not None:
            decisions.append(_evaluate(facts))
    return decisions


@dataclass
class ObjectiveFacts:
    """Plain, DB-free snapshot of an epic objective — the boundary between
    _gather_objective_facts (all I/O + signals, incl. the must_defend/dominating
    flags) and _evaluate (pure branching into options)."""
    match_id: str
    context: dict
    taken_by_us: bool
    has_resource: bool
    is_support: bool
    must_defend: bool
    presence_optional: bool
    name: str
    kill_ts: int
    distance_u: float
    hp_pct: Optional[float]
    mana_pct: Optional[float]
    items_count: Optional[int]


def _gather_objective_facts(
    conn, match_id, kill, us_id, our_team, parts, snapshots, champion_name: str = ""
) -> Optional[ObjectiveFacts]:
    """The I/O layer for one objective: position, snapshot, recall, vision, gold
    diff, contest/counterpart signals and state_features. Returns None with no
    usable position. Computes the branch flags the pure _evaluate consumes."""
    kill_ts = kill["_ts"]
    pit_pos = kill.get("position") or {}
    pit_x = pit_pos.get("x")
    pit_y = pit_pos.get("y")
    if pit_x is None:
        return None

    # Were you a spectator? Dead when the objective fell AND you did NOT die in the
    # pit contesting it -> there was no objective decision to coach (death_v1 owns the
    # death). Use the death event's exact position, not the per-minute frame (which
    # for a dead player reflects the corpse / stale last position).
    if player_dead_at(conn, match_id, us_id, kill_ts):
        d = last_death_before(conn, match_id, us_id, kill_ts)
        died_contesting = (
            d is not None and d["x"] is not None
            and math.hypot(d["x"] - pit_x, d["y"] - pit_y) <= CONTEST_DEATH_UNITS
        )
        if not died_contesting:
            return None   # spectator: dead when it fell and didn't die contesting

    # Our position at closest frame to kill
    pos_row = conn.execute(
        "SELECT position_x, position_y, current_gold, total_gold FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, us_id, kill_ts),
    ).fetchone()
    if not pos_row or pos_row["position_x"] is None:
        return None

    our_x = pos_row["position_x"]
    our_y = pos_row["position_y"]
    distance_u = math.hypot(our_x - pit_x, our_y - pit_y)

    # Live Client snapshot at kill time
    snap_at_kill = _snapshot_at(snapshots, kill_ts / 1000.0) if snapshots else None
    hp_pct = mana_pct = items_count = level = None
    riot_name = None
    # Determine if this champion has a meaningful mana/energy resource.
    # Prefer static game data (reliable) over snapshot's resourceType field.
    _has_resource = has_meaningful_resource(champion_name) if champion_name else True

    if snap_at_kill:
        ap = snap_at_kill.get("activePlayer") or {}
        cs = ap.get("championStats") or {}
        if cs.get("maxHealth"):
            hp_pct = cs["currentHealth"] / cs["maxHealth"]
        if _has_resource and cs.get("resourceMax"):
            mana_pct = cs["resourceValue"] / cs["resourceMax"]
        level = ap.get("level")
        riot_name = ap.get("riotIdGameName")
        items_count = _our_items_count(snap_at_kill, riot_name)

    # Recall in prep window?
    recall = _detect_recall_in_window(
        conn, match_id, us_id, kill_ts, snapshots
    )

    # Outcome
    killer_id = kill.get("killerId")
    taken_by_us = (
        parts[killer_id]["team_id"] == our_team if killer_id in parts
        else kill.get("killerTeamId") == our_team
    )

    # Position assessment
    at_pit = distance_u <= NEAR_PIT_UNITS
    too_far = distance_u > WARNING_DISTANCE_UNITS

    name = _objective_name(kill.get("monsterType"), kill.get("monsterSubType"))

    context = {
        "time_mmss": mmss(kill_ts),
        "objective": name,
        "objective_type": kill.get("monsterType"),
        "taken_by": "tu equipo" if taken_by_us else "enemigo",
        "your_position_at_kill_frame": {"x": our_x, "y": our_y},
        "distance_to_pit_units": int(distance_u),
        "at_pit": at_pit,
        "too_far": too_far,
        "your_state_at_kill": {
            "hp_pct": round(hp_pct, 3) if hp_pct is not None else None,
            "mana_pct": round(mana_pct, 3) if mana_pct is not None else None,
            "level": level,
            "items_in_inventory": items_count,
            "current_gold": pos_row["current_gold"],   # spendable (rule #7)
            "total_gold": pos_row["total_gold"],        # lifetime accumulated, stat only
        },
        "recall_in_prep_window": recall,
    }

    # Vision prep near the pit (absorbed from the former vision_prep detector).
    context.update(_vision_near_pit(conn, match_id, kill_ts, pit_x, pit_y, us_id, our_team, parts))

    is_support = us_id in parts and parts[us_id]["team_position"] == "UTILITY"
    context["your_role_is_support"] = is_support
    taken_is_setup = at_pit or (is_support and not too_far)

    ws = build_world_state(conn, match_id, kill_ts)
    context["state_features"] = build_state_features(
        conn, match_id, ws, us_id, our_team,
        entry_hp_pct=hp_pct,
        next_major_obj=obj_kind(kill.get("monsterType")),
        time_to_obj_s=0,
        we_have_setup=bool(taken_is_setup and (is_support or (hp_pct or 0) >= 0.5)),
        objective_value=_objective_value(conn, match_id, kill),
    )

    # Could we even contest this objective? In a clear disadvantage, or while a
    # base siege is on (own inhibitor already down), the right call is to CEDE and
    # defend/regroup, not to go set vision. (Player feedback: the "go set vision"
    # ideal was offered even when contesting was impossible.)
    pidx = (context["state_features"].get("power") or {}).get("power_index")
    inhibs_lost = conn.execute(
        "SELECT COUNT(*) c FROM timeline_events WHERE match_id=? AND type='BUILDING_KILL' "
        "AND timestamp_ms < ? AND payload_json LIKE '%INHIBITOR_BUILDING%' AND payload_json LIKE ?",
        (match_id, kill_ts, f'%"teamId":{our_team}%'),
    ).fetchone()["c"]
    team_gold_diff = _team_gold_diff(conn, match_id, our_team, kill_ts)
    context["team_gold_diff"] = team_gold_diff
    behind = isinstance(team_gold_diff, (int, float)) and team_gold_diff <= -3000
    # Only "must defend / cede" if the objective actually fell to the ENEMY. If
    # our team took it, we contested it (often on the back of pressure) — not a cede.
    must_defend = (not taken_by_us) and (
        inhibs_lost >= 1 or behind or (isinstance(pidx, (int, float)) and pidx < 0.35))
    context["can_contest"] = not must_defend
    context["own_inhibitors_lost"] = inhibs_lost
    # Opposite of must_defend: a big lead and we took it → objective was controlled,
    # the support's presence wasn't decisive (player: "mucha presión, no hacía falta
    # mi presencia"). Don't demand hovering with vision.
    dominating = bool(taken_by_us and isinstance(team_gold_diff, (int, float)) and team_gold_diff >= 3000)
    uncontested = bool(taken_by_us and not _objective_contested(conn, match_id, kill_ts, pit_x, pit_y, our_team))
    presence_optional = dominating or uncontested
    context["objective_dominated"] = dominating
    context["objective_uncontested"] = uncontested
    # Allies (not you) near the pit at the kill: did the TEAM play this objective?
    # If nobody showed, a "you gave it away" framing is wrong — it was a team-level cede.
    allies_near_pit = 0
    for pid, pdata in parts.items():
        if pid == us_id or pdata["team_id"] != our_team:
            continue
        fr = conn.execute(
            "SELECT position_x x, position_y y FROM timeline_frames WHERE match_id=? AND "
            "participant_id=? ORDER BY ABS(timestamp_ms-?) LIMIT 1",
            (match_id, pid, kill_ts),
        ).fetchone()
        # 4000u: per-minute frames lag behind a rotating team (player: "estábamos la
        # mayoría" while the nearest frame still had them approaching).
        if fr and fr["x"] is not None and math.hypot(fr["x"] - pit_x, fr["y"] - pit_y) <= 4000:
            allies_near_pit += 1
    context["allies_near_pit"] = allies_near_pit
    # Objective-for-objective trade: did OUR team take something else around the same
    # time (voidgrubs/herald/dragon/tower)? Then ceding this one was a TRADE, not a
    # giveaway (player: "fue dragón por las larvas, no me detecta el otro objetivo").
    context["traded_for"] = None if taken_by_us else _counterpart_objective(
        conn, match_id, kill_ts, our_team, exclude_ts=kill_ts)

    return ObjectiveFacts(
        match_id=match_id, context=context, taken_by_us=taken_by_us,
        has_resource=_has_resource, is_support=is_support, must_defend=must_defend,
        presence_optional=presence_optional, name=name, kill_ts=kill_ts,
        distance_u=distance_u, hp_pct=hp_pct, mana_pct=mana_pct, items_count=items_count,
    )


def _evaluate(facts: ObjectiveFacts) -> Decision:
    """Pure decision logic: branch into options, build action/argument, assemble
    the Decision. No conn — everything comes from `facts`."""
    context = facts.context
    # Each branch of _build_options returns its own action_ids — the option order
    # and semantics differ per branch (the taken option isn't always in the same slot).
    options, action_ids = _build_options(
        context, facts.taken_by_us, facts.has_resource, facts.is_support,
        facts.must_defend, facts.presence_optional)
    context["action"] = build_action(DETECTOR_ID, options, action_ids)

    argument = _compose_argument(context, options, facts.has_resource, facts.is_support)

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=facts.match_id,
        game_time_ms=facts.kill_ts,
        moment=f"Objetivo — {facts.name}",
        outcome=(
            f"{facts.name} cae a {context['taken_by']}. "
            f"Tú a {int(facts.distance_u)}u del pit. "
            + (f"HP {int(facts.hp_pct*100)}%, " if facts.hp_pct is not None else "")
            + (f"mana {int(facts.mana_pct*100)}%, " if facts.mana_pct is not None else "")
            + (f"items {facts.items_count} en inv. " if facts.items_count is not None else "")
            + f"Visión: tú {context['your_wards_near_pit']}w/{context['your_sweeps_near_pit']}sw, info {context['vision_diff']:+d}."
        ),
        context=context,
        options=options,
        argument=argument,
    )


def _detect_recall_in_window(
    conn, match_id, us_id, kill_ts, snapshots
) -> dict | None:
    """Heuristic recall detection: look for a gold drop in our frames inside the prep window."""
    rows = conn.execute(
        "SELECT timestamp_ms, total_gold, current_gold, position_x, position_y "
        "FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? "
        "AND timestamp_ms BETWEEN ? AND ? "
        "ORDER BY timestamp_ms",
        (match_id, us_id, kill_ts - PREP_WINDOW_MS, kill_ts),
    ).fetchall()
    if len(rows) < 2:
        return None
    # We look for a frame where current_gold drops significantly (item bought after recall)
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        cg_a = a["current_gold"] or 0
        cg_b = b["current_gold"] or 0
        if cg_a - cg_b >= RECALL_GOLD_DROP:
            return {
                "frame_ts_ms": b["timestamp_ms"],
                "rel_to_kill_s": round((b["timestamp_ms"] - kill_ts) / 1000.0, 1),
                "current_gold_before": cg_a,
                "current_gold_after": cg_b,
                "gold_spent": cg_a - cg_b,
            }
    return None


def _build_options(ctx, taken_by_us, has_resource: bool = True, is_support: bool = False,
                   must_defend: bool = False, dominating: bool = False) -> tuple[list[Option], list[str]]:
    """Returns (options, action_ids): each branch knows the action semantics of its
    own option positions, so the HEF's taken-action mapping stays correct."""
    s = ctx["your_state_at_kill"]
    hp = s["hp_pct"]
    mana = s["mana_pct"]
    at_pit = ctx["at_pit"]
    too_far = ctx["too_far"]

    if must_defend:
        # No setup to contest (disadvantage / base siege): ceding+defending is
        # right, forcing the objective is the mistake. The two options must
        # CONTRAST (cede=good vs force=bad), not repeat each other — the one you
        # actually did carries the "(lo que hiciste)" tag.
        reason = ("inhibidor propio caído" if ctx.get("own_inhibitors_lost", 0) >= 1
                  else "desventaja clara")
        ceded = not at_pit
        traded = ctx.get("traded_for")
        trade_txt = (f" Además tu equipo se llevó {traded} mientras tanto: fue objetivo por objetivo."
                     if traded else "")
        return [
            Option(
                label="Ceder el objetivo y defender/reagrupar" + (" (lo que hiciste)" if ceded else ""),
                predicted_consequence=(
                    f"Sin setup para contestar ({reason}): cederlo y no exponerte preserva la base y tu vida."
                    + (" Hiciste lo correcto." if ceded else "") + trade_txt
                ),
                ev_score=0.80,
            ),
            Option(
                label="Forzar el objetivo / exponerte a contestarlo" + ("" if ceded else " (lo que hiciste)"),
                predicted_consequence=(
                    "Sin setup, forzar el objetivo regala una muerte y acelera el asedio a tu base."
                    + ("" if ceded else " Te expusiste sin poder contestarlo.")
                ),
                ev_score=0.25,
            ),
        ], [ACTION_IGNORE_OBJECTIVE, ACTION_FIGHT_ENTRY]

    if dominating:  # param carries presence_optional: controlled (big lead) OR uncontested (free)
        # We took the objective and your presence wasn't decisive. Optimal = put the
        # freed time where it's worth most; the 2nd option describes what you did
        # with an EV that reflects whether THAT was a good use of the time.
        yw = ctx.get("your_wards_near_pit", 0); ys = ctx.get("your_sweeps_near_pit", 0)
        free = ctx.get("objective_uncontested") and not ctx.get("objective_dominated")
        why = ("nadie lo disputó (ningún enemigo cerca)" if free
               else "tu equipo lo tenía asegurado por la ventaja de oro")
        # You did fine either way — ONE clear confirmation + a real contrast, not
        # two near-equal phrasings of the same judgment (player feedback).
        if ctx.get("too_far"):
            return [
                Option(
                    label="Aprovechaste el tiempo fuera del pit: reset/farm/presión (lo que hiciste) — uso correcto del tiempo",
                    predicted_consequence=f"El objetivo {why}; convertir ese tiempo en oro/presión fue lo acertado.",
                    ev_score=0.82,
                ),
                Option(
                    label="Hacer acto de presencia en un objetivo ya asegurado",
                    predicted_consequence="Pararte en el pit no añadía nada: el costo era el tempo que SÍ convertiste en otra parte.",
                    ev_score=0.35,
                ),
            ], [ACTION_IGNORE_OBJECTIVE, ACTION_VISION_SETUP]
        return [
            Option(
                label=f"Acompañaste el objetivo controlado ({yw}w/{ys}sw) (lo que hiciste) — válido",
                predicted_consequence=(
                    f"Como {why}, tu presencia no era crítica; acompañar es correcto, con un margen "
                    f"menor a favor de convertir ese tiempo en presión/farm en otra zona."
                ),
                ev_score=0.78,
            ),
            Option(
                label="Desentenderte del pit antes de que cayera",
                predicted_consequence="Irte demasiado pronto es lo que regala robos de último golpe.",
                ev_score=0.40,
            ),
        ], [ACTION_VISION_SETUP, ACTION_IGNORE_OBJECTIVE]

    # Normal contest decision. The real choice at an objective is whether to CONTEST
    # (with setup: vision + numbers), TRADE it for another play, or DECLINE it.
    # Judged on the DECISION, not the result (player: "peleamos en desventaja y
    # perdimos, era momento de pelear" — contesting can be right and still be lost).
    name = ctx.get("objective", "el objetivo")
    yw = ctx.get("your_wards_near_pit", 0); ys = ctx.get("your_sweeps_near_pit", 0)
    tw = ctx.get("team_wards_near_pit", 0); tsw = ctx.get("team_sweeps_near_pit", 0)
    vdiff = ctx.get("vision_diff", 0) or 0
    allies_near = ctx.get("allies_near_pit", 0)
    # TEAM vision counts (the support wards for the jungler); your own wards aren't
    # the only setup (player: "tengo a mi soporte cerca limpiando y poniendo visión").
    have_vision = (tw + tsw) >= 1 or (yw + ys) >= 1
    near = at_pit or (is_support and not too_far)
    low_res = has_resource and mana is not None and mana < MANA_LOW_PCT
    low_hp = hp is not None and hp < HP_LOW_PCT
    # Setup = could you contest WELL? vision + numbers dominate; being physically
    # near with neither is NOT setup. With most of the team in zone, numbers alone
    # justify contesting (player: "llegamos a tiempo, estábamos la mayoría — era
    # momento de pelear") even if nobody warded first.
    setup = (0.40 if have_vision else 0.0) \
        + (0.15 if vdiff > 0 else 0.0) \
        + (0.30 if allies_near >= 3 else 0.20 if allies_near == 2 else 0.10 if allies_near == 1 else 0.0) \
        + (0.15 if (near and not too_far) else 0.0) \
        + (0.10 if not (low_res or low_hp) else 0.0)
    setup = max(0.0, min(1.0, setup))
    vis_txt = (f"visión {tw}w/{tsw}sw del equipo ({yw}w/{ys}sw tuyas), info {vdiff:+d}"
               if (tw or tsw or yw or ys or vdiff) else "sin visión del equipo")
    num_txt = f"{allies_near} aliado(s) en zona"

    if taken_by_us:
        if setup >= 0.55:
            # You did the right thing — say it ONCE, clearly, and contrast with the
            # real alternative you didn't take. (Player: "si hice bien, confírmalo;
            # no me des dos opciones gemelas que tengo que comparar".)
            crit = Option(
                label=f"Contestaste {name} con buen setup (lo que hiciste) — fue lo correcto",
                predicted_consequence=f"Setup {int(setup * 100)}% ({vis_txt}; {num_txt}). Decisión y ejecución alineadas.",
                ev_score=0.84,
            )
            actual = Option(
                label=f"Cederlo pese a tener el setup",
                predicted_consequence="La alternativa real era regalar un objetivo que tenían ganado.",
                ev_score=0.30,
            )
            ids = [ACTION_FIGHT_ENTRY, ACTION_IGNORE_OBJECTIVE]
        else:
            crit = Option(
                label="Asegurar el objetivo con setup previo: visión + números antes del spawn",
                predicted_consequence=(
                    "Con visión y equipo en zona antes del spawn, el objetivo se toma sin regalar "
                    "una pelea de moneda al aire."
                ),
                ev_score=0.84,
            )
            actual = Option(
                label=f"Contestaste {name} y lo conseguiste (lo que hiciste)",
                predicted_consequence=(
                    f"Setup {int(setup * 100)}% ({vis_txt}; {num_txt}). "
                    "Salió, pero con poco setup era arriesgado: pudo costar más de lo que valía."
                ),
                ev_score=round(0.54 + 0.30 * setup, 2),
            )
            ids = [ACTION_VISION_SETUP, ACTION_FIGHT_ENTRY]
    elif near and not too_far:
        # You were at the pit and it fell to the enemy: you contested and lost it.
        # With setup the CALL was right (loss = execution/coinflip); without, forcing
        # was the mistake. No resultadismo.
        if setup >= 0.5:
            # The CALL was right even though it was lost — confirm it once and
            # contrast with the real alternative (ceding with setup in hand).
            crit = Option(
                label=f"Contestaste {name} con setup y se perdió (lo que hiciste) — la decisión era correcta",
                predicted_consequence=(
                    f"Tenían con qué disputarlo ({vis_txt}; {num_txt}); falló la ejecución o el azar "
                    f"del robo, no la decisión. El ajuste es de ejecución: smite/daño al objetivo, foco de pelea."
                ),
                ev_score=0.70,
            )
            actual = Option(
                label=f"Ceder {name} sin pelearlo, teniendo setup",
                predicted_consequence="La alternativa era regalar un objetivo disputable con el equipo ya posicionado.",
                ev_score=0.40,
            )
            ids = [ACTION_FIGHT_ENTRY, ACTION_IGNORE_OBJECTIVE]
        else:
            crit = Option(
                label=f"Sin setup, ceder {name} o cambiarlo por otra jugada valía más que forzarlo",
                predicted_consequence="Forzar un objetivo de robo sin visión ni números regala más de lo que el objetivo vale.",
                ev_score=0.78,
            )
            actual = Option(
                label=f"Forzaste {name} sin visión/números y lo perdiste (lo que hiciste)",
                predicted_consequence=f"Estabas ahí sin setup ({vis_txt}; {num_txt}) y cayó al enemigo.",
                ev_score=0.42,
            )
            ids = [ACTION_IGNORE_OBJECTIVE, ACTION_FIGHT_ENTRY]
    else:
        # You weren't there and the enemy took it.
        traded = ctx.get("traded_for")
        if traded:
            # Objective-for-objective: your team took something else meanwhile.
            crit = Option(
                label=f"Cambiaste {name} por {traded} (lo que hiciste)",
                predicted_consequence=(
                    f"Trade de objetivos: mientras {name} caía, tu equipo se llevó {traded}. "
                    f"Con setup {int(setup * 100)}% para disputarlo, el cambio fue una jugada válida."
                ),
                ev_score=0.78,
            )
            actual = Option(
                label=f"Disputar {name} en vez del cambio",
                predicted_consequence="Pelear el objetivo implicaba soltar lo que tu equipo sí estaba asegurando.",
                ev_score=0.55,
            )
            return [crit, actual], [ACTION_IGNORE_OBJECTIVE, ACTION_FIGHT_ENTRY]
        if setup < 0.6:
            # No setup: ceding/trading WAS the right play — the recommendation is what
            # you did; the contrast is the bad alternative. (Player: "la recomendada no
            # aplica / ambas redundantes".)
            crit = Option(
                label=f"Cederlo / cambiarlo por otra jugada (lo que hiciste)",
                predicted_consequence=(
                    f"Sin setup para {name} ({vis_txt}; {num_txt}), invertir el tiempo en otra "
                    f"jugada (farm, presión, reset) fue la decisión correcta."
                ),
                ev_score=0.80,
            )
            actual = Option(
                label=f"Forzar {name} sin visión ni números",
                predicted_consequence="La alternativa real era regalarte en un objetivo sin preparación.",
                ev_score=0.30,
            )
            return [crit, actual], [ACTION_IGNORE_OBJECTIVE, ACTION_FIGHT_ENTRY]
        if allies_near == 0:
            crit = Option(
                label="Coordinar el contest antes del spawn (call de equipo, visión previa)",
                predicted_consequence=f"Había setup parcial ({vis_txt}) pero nadie del equipo jugó {name}: la falla fue colectiva, no tuya.",
                ev_score=0.78,
            )
            actual = Option(
                label=f"El equipo entero cedió {name} (lo que hiciste)",
                predicted_consequence="Tu llegada sola no lo cambiaba; el cede fue a nivel de equipo.",
                ev_score=0.62,
            )
            ids = [ACTION_VISION_SETUP, ACTION_IGNORE_OBJECTIVE]
        else:
            crit = Option(
                label=f"Presentarte y contestar {name} con tu equipo",
                predicted_consequence=f"Tu equipo estaba en zona ({num_txt}) con setup ({vis_txt}); faltó tu presencia.",
                ev_score=0.78,
            )
            actual = Option(
                label=f"Lo cediste sin presentarte, con tu equipo en zona (lo que hiciste)",
                predicted_consequence="Con setup y aliados jugándolo, no presentarse sí regala valor.",
                ev_score=0.48,
            )
            ids = [ACTION_FIGHT_ENTRY, ACTION_IGNORE_OBJECTIVE]

    return [crit, actual], ids


def _compose_argument(ctx, options, has_resource: bool = True, is_support: bool = False) -> str:
    lines = []
    s = ctx["your_state_at_kill"]
    name = ctx["objective"]

    if is_support:
        pos_phrase = (" (en el pit)" if ctx["at_pit"]
                      else " (demasiado lejos)" if ctx["too_far"]
                      else " (cerca, en posición de apoyo)")
    else:
        pos_phrase = (" (en posición de pelear)" if ctx["at_pit"]
                      else " (demasiado lejos)" if ctx["too_far"]
                      else " (cerca pero no en el pit)")
    lines.append(
        f"Estado al instante de {name} (a las {ctx['time_mmss']}): "
        f"a {ctx['distance_to_pit_units']} unidades del pit"
        + pos_phrase
        + "."
    )

    state_parts = []
    if s["hp_pct"] is not None:
        state_parts.append(f"HP {int(s['hp_pct']*100)}%")
    if has_resource and s["mana_pct"] is not None:
        state_parts.append(f"maná {int(s['mana_pct']*100)}%")
    if s["level"] is not None:
        state_parts.append(f"nivel {s['level']}")
    if s["items_in_inventory"] is not None:
        state_parts.append(f"{s['items_in_inventory']} items en inventario")
    if state_parts:
        lines.append("Estado: " + ", ".join(state_parts) + ".")

    yw = ctx.get("your_wards_near_pit", 0); ys = ctx.get("your_sweeps_near_pit", 0)
    vd = ctx.get("vision_diff", 0)
    info_word = "a favor" if vd > 0 else "en contra" if vd < 0 else "pareja"
    lines.append(
        f"Visión cerca del objetivo (90s previos): tú {yw} wards / {ys} sweeps; "
        f"info de equipo {info_word} ({vd:+d} vs enemigo)."
        + ("" if not (is_support and yw + ys == 0)
           else " Como support cerca del pit, no pusiste visión: ese es el aporte que se espera de tu rol.")
    )

    if not ctx.get("can_contest", True):
        reason = "inhibidor propio caído" if ctx.get("own_inhibitors_lost", 0) >= 1 else "desventaja clara"
        lines.append(
            f"No había setup para contestar este objetivo ({reason}): lo correcto "
            "era cederlo y defender/reagrupar, no invertir en él."
        )

    if ctx["recall_in_prep_window"]:
        r = ctx["recall_in_prep_window"]
        lines.append(
            f"Recall detectado en la ventana de prep "
            f"({r['rel_to_kill_s']:+.1f}s respecto al kill): "
            f"gastaste {r['gold_spent']} oro. "
            "Buen timing si compraste control ward + componentes para tu item."
        )

    # HP/mana readiness advice only makes sense if you actually contested it; if you
    # ceded/traded the objective, "recall to full before the pit" is beside the point.
    contested = ctx.get("taken_by") == "tu equipo"
    if contested and s["hp_pct"] is not None and s["hp_pct"] < HP_LOW_PCT:
        lines.append(
            f"HP al {int(s['hp_pct']*100)}% es subóptimo para iniciar/responder un fight. "
            "Idealmente recall + back a HP completa antes del spawn."
        )
    if contested and has_resource and s["mana_pct"] is not None and s["mana_pct"] < MANA_LOW_PCT:
        lines.append(
            f"Maná al {int(s['mana_pct']*100)}% limita tu kit en la pelea. "
            "Considera recall para llegar con maná completo antes del objetivo."
        )

    if ctx["too_far"]:
        lines.append(
            "La distancia al pit excede 5000u: el posicionamiento previo al "
            "spawn fue tarde para llegar a tiempo."
        )

    lines.append(f"Resultado: {name} cae a {ctx['taken_by']}.")
    return " ".join(lines)
