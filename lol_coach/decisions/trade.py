"""Trade analyzer v2 — lane-phase damage exchanges in the bot lane.

v1 triggered on every death with snapshots, so each death produced a death_v1
AND a trade_v1 decision (player 2026-06-10: "me está duplicando las muertes").
v2 implements the player's own criterion for a BAD lane trade:

  Exchanging damage (mostly basic abilities) while the ADC farms is normal;
  the detector must flag when the exchange backfires: we lost significantly
  more HP than the enemy duo — or dumped our mana — so the NEXT exchange
  starts in deficit; aggravated when the following ~90s bring a duo death.

Data sources (honest about what each side exposes):
  - Per-minute damageStats from the RAW match timeline (enemy HP is not
    observable; who absorbed more damage is the proxy for who won the trade).
  - The recorder's 1Hz snapshots for OUR exact HP%/mana% when available.

Scope: lane phase only (minutes 2-14), only when we play bot (UTILITY/BOTTOM),
and only while we are physically in the bot half (roam fights belong to
death_v1/tempo_v1). A minute where someone of either duo DIES is skipped:
death_v1 already argues that play (this kills the duplication), and a kill FOR
us is a won trade.

WON trades are persisted flagged (`context.positive_trade = true`) so Calibrar
can pair them against lost ones ("¿cuál se jugó mejor?") but they are hidden
from the decision list — the player should not have to re-judge what went well.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..context import build_world_state, map_lane
from .base import Decision, Option
from .features import (
    ACTION_DISENGAGE,
    ACTION_FIGHT_ENTRY,
    build_action,
    build_state_features,
    mmss,
)
from ..game_data import has_meaningful_resource
from ..snapshots import load_snapshots


DETECTOR_ID = "trade_v1"

LANE_PHASE_FIRST_MIN = 2     # ignore the walk-to-lane minute
LANE_PHASE_LAST_MIN = 14     # after this, fights are skirmishes, not lane trades
MIN_EXCHANGE_DMG = 250       # combined duo damage in a minute to call it a trade
BAD_MIN_DIFF = 300           # we took at least this much MORE than the enemy duo
BAD_RATIO = 1.5              # and at least 1.5x theirs
LOW_MANA_PCT = 0.25          # ended the exchange nearly oom = next fight in deficit
CONSEQUENCE_WINDOW_MS = 90_000


def _our_stats_at(snapshots, t_s, has_resource) -> dict:
    """Our exact HP%/mana% at ~t_s from the closest snapshot (None without recorder)."""
    if not snapshots:
        return {"hp_pct": None, "mana_pct": None}
    snap = min(snapshots, key=lambda s: abs(s[0] - t_s))
    if abs(snap[0] - t_s) > 90:
        return {"hp_pct": None, "mana_pct": None}
    cs = (snap[1].get("activePlayer") or {}).get("championStats") or {}
    hp = (cs["currentHealth"] / cs["maxHealth"]
          if cs.get("currentHealth") is not None and cs.get("maxHealth") else None)
    mp = (cs["resourceValue"] / cs["resourceMax"]
          if has_resource and cs.get("resourceValue") is not None and cs.get("resourceMax") else None)
    return {"hp_pct": round(hp, 2) if hp is not None else None,
            "mana_pct": round(mp, 2) if mp is not None else None}


def _taken_total(pf: dict) -> float:
    d = pf.get("damageStats") or {}
    return (d.get("physicalDamageTaken") or 0) + (d.get("magicDamageTaken") or 0) \
        + (d.get("trueDamageTaken") or 0)


def analyze_trades(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    match = conn.execute(
        "SELECT our_puuid, timeline_json_path FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    if not match or not match["timeline_json_path"]:
        return []
    us = conn.execute(
        "SELECT participant_id, champion_name, team_id, team_position FROM participants "
        "WHERE match_id = ? AND puuid = ?", (match_id, match["our_puuid"]),
    ).fetchone()
    if us is None or us["team_position"] not in ("UTILITY", "BOTTOM"):
        return []  # lane-trade analysis is a bot-lane concept (player's own framing)
    us_id, our_team = us["participant_id"], us["team_id"]
    _has_resource = has_meaningful_resource(us["champion_name"] or "")

    parts = {r["participant_id"]: r for r in conn.execute(
        "SELECT participant_id, team_id, team_position, champion_name FROM participants "
        "WHERE match_id = ?", (match_id,))}
    my_duo = [pid for pid, p in parts.items()
              if p["team_id"] == our_team and p["team_position"] in ("UTILITY", "BOTTOM")]
    enemy_duo = [pid for pid, p in parts.items()
                 if p["team_id"] != our_team and p["team_position"] in ("UTILITY", "BOTTOM")]
    if len(my_duo) < 2 or len(enemy_duo) < 2:
        return []

    try:
        timeline = json.loads(Path(match["timeline_json_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    frames = (timeline.get("info") or {}).get("frames") or []
    if len(frames) < LANE_PHASE_FIRST_MIN + 2:
        return []

    # Cumulative damage-taken per duo per frame -> per-minute deltas.
    def duo_taken(frame, duo):
        pfs = frame.get("participantFrames") or {}
        return sum(_taken_total(pfs.get(str(pid)) or {}) for pid in duo)

    # Our duo deaths / enemy duo deaths per minute (skip = death_v1's turf / a won kill).
    duo_death_min, enemy_death_min = set(), set()
    for r in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events WHERE match_id=? AND type='CHAMPION_KILL'",
        (match_id,),
    ):
        ev = json.loads(r["payload_json"])
        m = r["timestamp_ms"] // 60_000
        if ev.get("victimId") in my_duo:
            duo_death_min.add(m)
        elif ev.get("victimId") in enemy_duo:
            enemy_death_min.add(m)

    # Recorder snapshots (our exact HP/mana). Optional: damage deltas alone suffice.
    session_row = conn.execute(
        "SELECT session_dir FROM sessions WHERE match_id = ?", (match_id,)).fetchone()
    snapshots = (load_snapshots(Path(session_row["session_dir"]))
                 if session_row and session_row["session_dir"] else [])

    last_min = min(LANE_PHASE_LAST_MIN, len(frames) - 1)
    minutes = []   # (minute, kind, d_us, d_enemy, our_stats)
    for m in range(LANE_PHASE_FIRST_MIN, last_min + 1):
        d_us = duo_taken(frames[m], my_duo) - duo_taken(frames[m - 1], my_duo)
        d_en = duo_taken(frames[m], enemy_duo) - duo_taken(frames[m - 1], enemy_duo)
        if d_us + d_en < MIN_EXCHANGE_DMG:
            continue  # no real exchange this minute
        # Were WE in the bot half? Roams/objective fights belong to other detectors.
        fr = conn.execute(
            "SELECT position_x x, position_y y FROM timeline_frames WHERE match_id=? AND "
            "participant_id=? ORDER BY ABS(timestamp_ms-?) LIMIT 1",
            (match_id, us_id, m * 60_000)).fetchone()
        if fr and fr["x"] is not None and map_lane(fr["x"], fr["y"]) not in ("bot lane", "jungla"):
            if not (fr["x"] > 7500 and fr["y"] < 7500):  # bot-side quadrant
                continue
        if m in duo_death_min:
            continue  # someone of ours fell: death_v1 argues that play (no duplicates)
        stats = _our_stats_at(snapshots, m * 60.0, _has_resource)
        lost = (d_us - d_en >= BAD_MIN_DIFF and d_us >= BAD_RATIO * max(d_en, 1))
        oom = (stats["mana_pct"] is not None and stats["mana_pct"] < LOW_MANA_PCT
               and d_us >= d_en)  # dumped mana without winning the exchange
        won = (m in enemy_death_min) or (d_en - d_us >= BAD_MIN_DIFF and d_en >= BAD_RATIO * max(d_us, 1))
        if lost or oom:
            minutes.append((m, "bad", d_us, d_en, stats))
        elif won:
            minutes.append((m, "good", d_us, d_en, stats))

    # Merge consecutive bad minutes: one sustained losing pattern = one decision.
    decisions: list[Decision] = []
    i = 0
    while i < len(minutes):
        m, kind, d_us, d_en, stats = minutes[i]
        j = i
        while (kind == "bad" and j + 1 < len(minutes)
               and minutes[j + 1][1] == "bad" and minutes[j + 1][0] == minutes[j][0] + 1):
            j += 1
            d_us += minutes[j][2]
            d_en += minutes[j][3]
            stats = minutes[j][4]  # state at the END of the run
        facts = _gather_trade_facts(conn, match_id, us_id, our_team, parts, my_duo,
                                    m, minutes[j][0], kind, d_us, d_en, stats)
        decisions.append(_evaluate(facts))
        i = j + 1
    return decisions


@dataclass
class TradeFacts:
    """Plain, DB-free snapshot of a bot-lane trade run — the boundary between
    _gather_trade_facts (consequence query + state_features) and _evaluate (pure)."""
    match_id: str
    t_ms: int
    m_start: int
    m_end: int
    positive: bool
    d_us: int
    d_en: int
    hp_pct: Optional[float]
    mana_pct: Optional[float]
    consequence: Optional[str]
    state_features: dict


def _gather_trade_facts(conn, match_id, us_id, our_team, parts, my_duo,
                        m_start, m_end, kind, d_us, d_en, stats) -> TradeFacts:
    """The I/O layer for one trade run: the lost-trade consequence query and the
    state_features at the run's start."""
    t_ms = m_start * 60_000
    positive = kind == "good"

    # Consequence after a lost trade: a duo death in the next ~90s confirms the
    # deficit got cashed in (the player's own chain: bad trade -> kill/freeze/gank).
    consequence = None
    if not positive:
        for r in conn.execute(
            "SELECT payload_json FROM timeline_events WHERE match_id=? AND type='CHAMPION_KILL' "
            "AND timestamp_ms BETWEEN ? AND ?",
            (match_id, m_end * 60_000, m_end * 60_000 + CONSEQUENCE_WINDOW_MS),
        ):
            ev = json.loads(r["payload_json"])
            if ev.get("victimId") in my_duo:
                victim = parts.get(ev.get("victimId"))
                consequence = f"muerte de {victim['champion_name'] if victim else 'tu duo'} en los 90s siguientes"
                break

    ws = build_world_state(conn, match_id, t_ms)
    sf = build_state_features(conn, match_id, ws, us_id, our_team)

    return TradeFacts(
        match_id=match_id, t_ms=t_ms, m_start=m_start, m_end=m_end, positive=positive,
        d_us=d_us, d_en=d_en, hp_pct=stats["hp_pct"], mana_pct=stats["mana_pct"],
        consequence=consequence, state_features=sf,
    )


def _evaluate(facts: TradeFacts) -> Decision:
    """Pure decision logic: options, argument, context, Decision. No conn."""
    m_start, m_end = facts.m_start, facts.m_end
    d_us, d_en = facts.d_us, facts.d_en
    consequence = facts.consequence
    positive = facts.positive

    span = f"min {m_start}" if m_start == m_end else f"min {m_start}-{m_end}"
    ratio_txt = f"recibieron {int(d_us)} de daño vs {int(d_en)} del duo enemigo"
    hp_txt = (f"HP {int(facts.hp_pct*100)}%" if facts.hp_pct is not None else None)
    mana_txt = (f"maná {int(facts.mana_pct*100)}%" if facts.mana_pct is not None else None)
    state_txt = " · ".join(x for x in (hp_txt, mana_txt) if x)

    if positive:
        options = [
            Option(
                label="Trade ganado (lo que hiciste)",
                predicted_consequence=f"El intercambio salió a favor: {ratio_txt}.",
                ev_score=0.78,
            ),
            Option(
                label="No intercambiar / solo farmear",
                predicted_consequence="Cedías presión gratis en una ventana que tenían ganada.",
                ev_score=0.45,
            ),
        ]
        action_ids = [ACTION_FIGHT_ENTRY, ACTION_DISENGAGE]
        moment = f"Trade ganado en bot ({span})"
        outcome = f"Intercambio a favor: {ratio_txt}." + (f" Tu estado al cierre: {state_txt}." if state_txt else "")
        argument = (
            f"En {span} el intercambio de daño en bot salió a favor ({ratio_txt}). "
            f"Registrado para comparación (Calibrar), no para revisión."
        )
    else:
        why = []
        if d_us - d_en >= BAD_MIN_DIFF:
            why.append("perdieron el intercambio de vida")
        if facts.mana_pct is not None and facts.mana_pct < LOW_MANA_PCT:
            why.append("quedaste casi sin maná para la siguiente pelea")
        why_txt = " y ".join(why) or "el intercambio salió en contra"
        options = [
            Option(
                label="Cambiar daño solo con ventana a favor (maná, cooldowns, posición), o respetar y farmear",
                predicted_consequence=(
                    "El trade del support vale cuando desgasta más de lo que cuesta; sin ventana, "
                    "cada intercambio financia el all-in enemigo."
                ),
                ev_score=0.72,
            ),
            Option(
                label="Tomaste el intercambio y quedaron abajo (lo que hiciste)",
                predicted_consequence=(
                    f"{why_txt.capitalize()} ({ratio_txt})."
                    + (f" {consequence.capitalize()}." if consequence else "")
                ),
                ev_score=0.42,
            ),
        ]
        action_ids = [ACTION_DISENGAGE, ACTION_FIGHT_ENTRY]
        moment = f"Trade perdido en bot ({span})"
        outcome = (f"{why_txt.capitalize()}: {ratio_txt}."
                   + (f" Estado al cierre: {state_txt}." if state_txt else "")
                   + (f" Consecuencia: {consequence}." if consequence else ""))
        argument = (
            f"Entre el {span} el duo recibió {int(d_us)} de daño contra {int(d_en)} del enemigo: {why_txt}. "
            + (f"Tu estado al cierre ({state_txt}) confirma el déficit para la siguiente ventana. " if state_txt else "")
            + (f"La consecuencia llegó: {consequence}. " if consequence else "")
            + "El criterio no es no tradear: es cambiar daño solo cuando maná/cooldowns/posición están a favor; "
              "si no, respetar, farmear y esperar tu ventana."
        )

    context = {
        "time_mmss": mmss(facts.t_ms),
        "minute_span": span,
        "duo_damage_taken": int(d_us),
        "enemy_duo_damage_taken": int(d_en),
        "our_hp_pct_end": facts.hp_pct,
        "our_mana_pct_end": facts.mana_pct,
        "consequence": consequence,
        "positive_trade": positive,
        "state_features": facts.state_features,
    }
    context["action"] = build_action(DETECTOR_ID, options, action_ids)

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=facts.match_id,
        game_time_ms=facts.t_ms,
        moment=moment,
        outcome=outcome,
        context=context,
        options=options,
        argument=argument,
    )
