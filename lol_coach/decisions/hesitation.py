"""Hesitation analyzer (Phase 2) — needs Ascent input_events.

The most differentiating signal in the system: when the player's cursor rested
(a hover_dwell > 800ms) in the seconds before a kill or a death, they were
evaluating the situation. If they then died, the instinct that made them pause
was usually right and got overridden — that's a behavioral pattern (override of
a correct read), not a knowledge gap.

Trigger: each CHAMPION_KILL where the user is victim or killer, with a
hover_dwell in the 3s window before it. Requires input_events for the match
(scripts/ingest_ascent_events.py); yields nothing otherwise.
"""
from __future__ import annotations

import json
import sqlite3

from ..context import build_world_state
from .base import Decision, Option
from .features import ACTION_DISENGAGE, ACTION_FIGHT_ENTRY, build_action, build_state_features

DETECTOR_ID = "hesitation_v1"

DWELL_WINDOW_MS = 3000      # look this far before the event for a dwell
DWELL_MIN_MS = 800          # minimum dwell to count as deliberation


def _mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def analyze_hesitation(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    # No input data -> nothing to say.
    has_inputs = conn.execute(
        "SELECT 1 FROM input_events WHERE match_id = ? LIMIT 1", (match_id,)
    ).fetchone()
    if not has_inputs:
        return []

    match = conn.execute("SELECT our_puuid FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    us = conn.execute(
        "SELECT participant_id, team_id FROM participants WHERE match_id = ? AND puuid = ?",
        (match_id, match["our_puuid"]),
    ).fetchone()
    us_id, our_team = us["participant_id"], us["team_id"]

    decisions: list[Decision] = []
    for row in conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'CHAMPION_KILL' ORDER BY timestamp_ms",
        (match_id,),
    ):
        ev = json.loads(row["payload_json"])
        victim, killer = ev.get("victimId"), ev.get("killerId")
        if us_id not in (victim, killer):
            continue
        died = victim == us_id
        t = row["timestamp_ms"]

        dwell = conn.execute(
            "SELECT game_time_ms, screen_x, screen_y, duration_ms FROM input_events "
            "WHERE match_id = ? AND event_type = 'hover_dwell' "
            "AND game_time_ms BETWEEN ? AND ? AND duration_ms >= ? "
            "ORDER BY duration_ms DESC LIMIT 1",
            (match_id, t - DWELL_WINDOW_MS, t, DWELL_MIN_MS),
        ).fetchone()
        if dwell is None:
            continue

        decisions.append(_build(conn, match_id, us_id, our_team, t, died, dwell))
    return decisions


def _build(conn, match_id, us_id, our_team, event_ms, died, dwell) -> Decision:
    rel_ms = event_ms - dwell["game_time_ms"]        # how long after the dwell the event landed
    dur = dwell["duration_ms"]

    # Count actions taken between the dwell and the event (did they commit fast?).
    actions_after = conn.execute(
        "SELECT COUNT(*) c FROM input_events WHERE match_id = ? "
        "AND event_type IN ('click','key') AND game_time_ms BETWEEN ? AND ?",
        (match_id, dwell["game_time_ms"], event_ms),
    ).fetchone()["c"]

    if died:
        options = [
            Option("Confiar en la evaluación / disengage", "Sales del rango tras dudar; conservas el recurso.", 0.75),
            Option("Override la evaluación y comprometer (lo que hiciste)", "Entras pese a la señal de duda; mueres.", 0.25),
        ]
        action_ids = [ACTION_DISENGAGE, ACTION_FIGHT_ENTRY]
    else:
        options = [
            Option("Comprometer tras evaluar (lo que hiciste)", "Confirmas el read y ejecutas; consigues el kill.", 0.70),
            Option("Seguir dudando / no comprometer", "Dejas pasar una ventana favorable.", 0.45),
        ]
        action_ids = [ACTION_FIGHT_ENTRY, ACTION_DISENGAGE]

    ws = build_world_state(conn, match_id, event_ms)
    state_features = build_state_features(conn, match_id, ws, us_id, our_team)

    context = {
        "time_mmss": _mmss(event_ms),
        "dwell": {
            "duration_ms": dur,
            "ms_before_event": rel_ms,
            "screen_pos": {"x": dwell["screen_x"], "y": dwell["screen_y"]},
        },
        "subsequent_event": "muerte" if died else "kill",
        "actions_between_dwell_and_event": actions_after,
        "outcome": "moriste" if died else "consiguió kill",
        "state_features": state_features,
        "action": build_action(DETECTOR_ID, options, action_ids),
    }

    if died:
        argument = (
            f"A las {context['time_mmss']} tu cursor estuvo {dur}ms evaluando la situación "
            f"({rel_ms/1000:.1f}s antes de morir). Reconociste el riesgo — el instinto que te "
            f"hizo dudar era la señal correcta, y lo overridiste al comprometer. El patrón a "
            f"trabajar no es de conocimiento (ya lo sabías) sino de confiar en esa pausa: "
            f"cuando dudas así, el EV está casi siempre del lado de salir."
        )
    else:
        argument = (
            f"A las {context['time_mmss']} tu cursor estuvo {dur}ms evaluando antes de conseguir "
            f"el kill. La pausa de lectura previa al commit mejoró el resultado: confirmaste el "
            f"read en vez de entrar a ciegas. Este es el uso correcto de la hesitación."
        )

    return Decision(
        detector_id=DETECTOR_ID,
        match_id=match_id,
        game_time_ms=event_ms,
        moment=f"Hesitación ({dur}ms) antes de {'morir' if died else 'matar'} a las {context['time_mmss']}",
        outcome=(
            f"Dwell de {dur}ms, {rel_ms/1000:.1f}s antes del evento. "
            f"{'Moriste.' if died else 'Conseguiste el kill.'} "
            f"{context['actions_between_dwell_and_event']} acciones entre la duda y el evento."
        ),
        context=context,
        options=options,
        argument=argument,
    )
