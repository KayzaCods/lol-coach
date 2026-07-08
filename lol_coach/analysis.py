"""Canonical detector registry + decision persistence.

Single source of truth for WHICH detectors run and in what order. All entry
points (scripts/analyze.py, scripts/sync_ascent.py) import from here so adding
a detector is a one-line change in one place.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .decisions.base import Decision
from .decisions.death import analyze_deaths
from .decisions.objective_readiness import analyze_objective_readiness
from .decisions.tempo import analyze_tempo
from .decisions.trade import analyze_trades

# Order is the display/processing order; analyze afterwards sorts by game time.
# Retired (files kept for reference): hesitation_v1 (2026-06-07) and awareness_v1
# (2026-06-10) — player: with the current input data "mirar el mapa" is too
# subjective to detect; both produced noise, not signal.
DETECTOR_FUNCS = [
    analyze_deaths,
    analyze_trades,
    analyze_tempo,
    analyze_objective_readiness,
]


def run_all_detectors(conn: sqlite3.Connection, match_id: str) -> list[Decision]:
    """Run every detector on a match and return decisions sorted by time."""
    decisions: list[Decision] = []
    for fn in DETECTOR_FUNCS:
        decisions.extend(fn(conn, match_id))
    decisions.sort(key=lambda d: (d.game_time_ms, d.detector_id))
    return decisions


def insert_decision(conn: sqlite3.Connection, d: Decision) -> int:
    cur = conn.execute(
        """
        INSERT INTO decisions (
            match_id, detector_id, game_time_ms, moment, outcome,
            context_json, options_json, argument,
            clip_path, clip_offset_s, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d.match_id, d.detector_id, d.game_time_ms, d.moment, d.outcome,
            json.dumps(d.context, ensure_ascii=False, indent=2),
            json.dumps([o.__dict__ for o in d.options], ensure_ascii=False, indent=2),
            d.argument,
            d.clip_path, d.clip_offset_s,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return cur.lastrowid


def persist_decisions(
    conn: sqlite3.Connection, match_id: str, decisions: list[Decision]
) -> tuple[int, dict]:
    """Idempotent write: delete this match's decisions, reinsert the given list.

    Preserves per-decision user feedback (agree/disagree, written note, chosen
    option, option fingerprint) across the rewrite by matching on the stable key
    (detector_id, game_time_ms) — ids change on reinsert, these don't. So
    re-analysing a match never wipes the feedback the user already gave it.

    Feedback whose key does NOT reappear (a detector changed its triggers or
    timestamps — the trade_v2 / tempo retune already burned pairwise feedback
    this way) CANNOT be preserved. It is dropped, and reported in the returned
    stats so no caller loses it silently again. Policy: if you change WHAT a
    detector triggers on or WHEN (its game_time_ms), bump its detector_id or
    ship a key migration along with the change.

    Returns (n_decisions, feedback_stats) with feedback_stats =
    {"kept": int, "dropped": int, "dropped_keys": [(detector_id, game_time_ms), ...]}
    (dropped_keys capped at 10 for printing).
    """
    saved = {}
    for r in conn.execute(
        "SELECT detector_id, game_time_ms, user_feedback, user_feedback_note, "
        "user_best_option, user_feedback_at_utc, user_feedback_fingerprint, "
        "user_mark_type "
        "FROM decisions WHERE match_id = ? AND "
        "(user_feedback IS NOT NULL OR user_feedback_note IS NOT NULL "
        " OR user_best_option IS NOT NULL OR user_mark_type IS NOT NULL)",
        (match_id,),
    ):
        saved[(r["detector_id"], r["game_time_ms"])] = (
            r["user_feedback"], r["user_feedback_note"],
            r["user_best_option"], r["user_feedback_at_utc"],
            r["user_feedback_fingerprint"], r["user_mark_type"],
        )
    conn.execute("DELETE FROM decisions WHERE match_id = ?", (match_id,))
    kept = 0
    for d in decisions:
        new_id = insert_decision(conn, d)
        fb = saved.pop((d.detector_id, d.game_time_ms), None)
        if fb:
            conn.execute(
                "UPDATE decisions SET user_feedback=?, user_feedback_note=?, "
                "user_best_option=?, user_feedback_at_utc=?, "
                "user_feedback_fingerprint=?, user_mark_type=? "
                "WHERE id=?",
                (*fb, new_id),
            )
            kept += 1
    conn.commit()
    stats = {
        "kept": kept,
        "dropped": len(saved),
        "dropped_keys": sorted(saved.keys())[:10],
    }
    return len(decisions), stats
