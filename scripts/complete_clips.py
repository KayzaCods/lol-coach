"""Unify full-match replays and per-play clips as video context for decisions.

Two steps, both idempotent and run AFTER scripts/analyze.py (analyze rewrites
decisions and resets clip_path/clip_offset_s for most rows):

  1. normalize_full_replays(conn): a Replays.lol full-match recording is a
     timeline whose t=0 is game time 0, so its clips.in_match_start_s must be 0.
     The ingest currently stores a non-zero anchor (~the save instant), which
     produced negative clip_offset_s. This forces full replays to 0.

  2. complete_clips_for_decisions(conn) [= completar_clips_para_decisiones]:
     for every decision, assign the best available video and a correct offset:
       a) an Outplayed/Replays.lol per-play clip aligned to the event
          (|in_match_start_s - event_s| <= ALIGN_TOL_S), offset into that clip; else
       b) the match's full replay (in_match_start_s=0), offset = game_time_ms/1000; else
       c) no clip (clears any stale/broken path).
     It overwrites existing assignments on purpose, to repair the broken
     (negative) offsets left by the old nearest-clip heuristic.

Schema is untouched — only the VALUES of clips.in_match_start_s and
decisions.clip_path / decisions.clip_offset_s change.

Usage:
    .venv\\Scripts\\python.exe scripts\\complete_clips.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# load_config / connect are imported lazily in main() so the pure functions
# (normalize_full_replays, complete_clips_for_decisions, _any_clip_on_disk)
# can be imported and unit-tested without a config.toml or DB connection.

# Full-match recordings (t=0 ~ game start). Replays.lol from Overwolf and Ascent
# both record the whole game. ('Replays.lol' accepted for forward-compat with
# the source name in the task spec; the ingest writes 'replays_lol_full'.)
FULL_REPLAY_SOURCES = ("replays_lol_full", "Replays.lol", "ascent_full")
# Per-play clips that may be aligned to a specific event.
SHORT_CLIP_SOURCES = ("replays_lol_clip", "outplayed_clip")
# A short clip counts as "aligned" to a decision within this many seconds.
ALIGN_TOL_S = 30.0


def _ph(seq) -> str:
    return ",".join("?" * len(seq))


def normalize_full_replays(conn) -> int:
    """Force full-match replays to in_match_start_s = 0 (their t=0 is game 0)."""
    cur = conn.execute(
        f"UPDATE clips SET in_match_start_s = 0 "
        f"WHERE source IN ({_ph(FULL_REPLAY_SOURCES)}) "
        f"AND (in_match_start_s IS NULL OR in_match_start_s <> 0)",
        FULL_REPLAY_SOURCES,
    )
    conn.commit()
    return cur.rowcount


def _first_existing(rows):
    for r in rows:
        if r["path"] and os.path.exists(r["path"]):
            return r
    return None


def _any_clip_on_disk(conn) -> bool:
    """True if at least one tracked clip file is reachable on disk. When the
    video disk is not mounted every os.path.exists fails; without this guard
    complete_clips_for_decisions would NULL every decision clip_path and the
    next run (disk back) would re-assign them all (#3 churn)."""
    for r in conn.execute("SELECT path FROM clips WHERE path IS NOT NULL"):
        if os.path.exists(r["path"]):
            return True
    return False


def complete_clips_for_decisions(conn) -> dict:
    """Assign clip_path + clip_offset_s to every decision (overwriting)."""
    if not _any_clip_on_disk(conn):
        return {"short": 0, "full": 0, "none": 0, "total": 0, "disk_unmounted": True}
    full_cache: dict = {}
    short_cache: dict = {}

    def full_for(match_id):
        if match_id not in full_cache:
            rows = conn.execute(
                f"SELECT path, in_match_start_s FROM clips "
                f"WHERE match_id = ? AND source IN ({_ph(FULL_REPLAY_SOURCES)}) "
                f"ORDER BY (in_match_start_s IS NULL), in_match_start_s",
                (match_id, *FULL_REPLAY_SOURCES),
            ).fetchall()
            full_cache[match_id] = _first_existing(rows)
        return full_cache[match_id]

    def shorts_for(match_id):
        if match_id not in short_cache:
            short_cache[match_id] = conn.execute(
                f"SELECT path, in_match_start_s FROM clips "
                f"WHERE match_id = ? AND source IN ({_ph(SHORT_CLIP_SOURCES)}) "
                f"AND in_match_start_s IS NOT NULL",
                (match_id, *SHORT_CLIP_SOURCES),
            ).fetchall()
        return short_cache[match_id]

    rows = conn.execute("SELECT id, match_id, game_time_ms FROM decisions").fetchall()
    n_short = n_full = n_none = 0
    for d in rows:
        event_s = d["game_time_ms"] / 1000.0

        # a) per-play clip aligned to this event
        best, best_diff = None, None
        for c in shorts_for(d["match_id"]):
            if not (c["path"] and os.path.exists(c["path"])):
                continue
            diff = abs((c["in_match_start_s"] or 0.0) - event_s)
            if diff <= ALIGN_TOL_S and (best_diff is None or diff < best_diff):
                best, best_diff = c, diff
        if best is not None:
            offset = max(0.0, event_s - (best["in_match_start_s"] or 0.0))
            conn.execute(
                "UPDATE decisions SET clip_path = ?, clip_offset_s = ? WHERE id = ?",
                (best["path"], round(offset, 1), d["id"]),
            )
            n_short += 1
            continue

        # b) full-replay fallback
        full = full_for(d["match_id"])
        if full is not None:
            offset = max(0.0, event_s - (full["in_match_start_s"] or 0.0))
            conn.execute(
                "UPDATE decisions SET clip_path = ?, clip_offset_s = ? WHERE id = ?",
                (full["path"], round(offset, 1), d["id"]),
            )
            n_full += 1
            continue

        # c) no usable clip — clear any stale/broken assignment
        conn.execute(
            "UPDATE decisions SET clip_path = NULL, clip_offset_s = NULL WHERE id = ?",
            (d["id"],),
        )
        n_none += 1

    conn.commit()
    return {"short": n_short, "full": n_full, "none": n_none, "total": len(rows)}


def main() -> int:
    from lol_coach.config import load_config
    from lol_coach.db import connect
    cfg = load_config()
    conn = connect(cfg["paths"]["sqlite_db"])
    n = normalize_full_replays(conn)
    print(f"Full replays normalizados a in_match_start_s=0: {n}")
    res = complete_clips_for_decisions(conn)
    print(
        f"Clips asignados: {res['short']} por jugada, {res['full']} por replay completo, "
        f"{res['none']} sin clip (de {res['total']} decisiones)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
