"""Ingest Ascent input logs (events.csv.gz) into the input_events table.

For each Ascent recording whose match is already in our DB, parse its
events.csv.gz into significant input events (clicks, keys, wheel, hover_dwells)
and store them linked by match_id + game_time_ms.

Prereq: run scripts/ingest_ascent.py first (it pulls the match from Riot and
links the video). This script only fills input_events for matches already
present. Idempotent: re-running replaces a match's input_events.

  .venv\\Scripts\\python.exe scripts\\ingest_ascent_events.py
  .venv\\Scripts\\python.exe scripts\\ingest_ascent_events.py LA1_1722354497
"""
from __future__ import annotations

import collections
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.ascent_input import derive_events
from lol_coach.config import load_config
from lol_coach.db import connect


def _candidate_match_ids(game_match_id, game_platform_id) -> list[str]:
    if not game_match_id:
        return []
    gid = str(game_match_id).strip()
    out = [gid]
    if game_platform_id and "_" not in gid:
        plat = str(game_platform_id).strip()
        out += [f"{plat}_{gid}", f"{plat.upper()}_{gid}"]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def _resolve_local(conn, gmi, gpi) -> str | None:
    for cand in _candidate_match_ids(gmi, gpi):
        if conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (cand,)).fetchone():
            return cand
    return None


def _open_ascent_db_readonly(db_path: Path):
    tmp = Path(tempfile.mkdtemp(prefix="ascent_db_"))
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(db_path) + suffix)
        if src.exists():
            shutil.copy2(src, tmp / (db_path.name + suffix))
    conn = sqlite3.connect(str(tmp / db_path.name))
    conn.row_factory = sqlite3.Row
    return conn, tmp


def ingest_one(conn, match_id: str, events_path: Path, width: int, height: int) -> dict:
    events = derive_events(events_path, width or 1920, height or 1080)
    conn.execute("DELETE FROM input_events WHERE match_id = ?", (match_id,))
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO input_events (
            match_id, session_dir, game_time_ms, event_type, button, key,
            screen_x, screen_y, duration_ms, classified_intent, classified_at_utc
        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (match_id, e["game_time_ms"], e["event_type"], e["button"], e["key"],
             e["screen_x"], e["screen_y"], e["duration_ms"], e["classified_intent"], now)
            for e in events
        ],
    )
    conn.commit()
    by_type = collections.Counter(e["event_type"] for e in events)
    return {"total": len(events), "by_type": dict(by_type)}


def ingest_events_for_db(conn, ascent_db: Path, only_match: str | None = None) -> dict:
    """Parse Ascent recordings' input logs into input_events. Returns stats."""
    s = {"seen": 0, "done": 0, "skipped": 0, "per_match": {}}
    adb, tmp = _open_ascent_db_readonly(ascent_db)
    try:
        try:
            rows = adb.execute("SELECT * FROM recordings").fetchall()
        except sqlite3.OperationalError as e:
            print(f"No pude leer 'recordings': {e}")
            return s
        for r in rows:
            row = dict(r)
            s["seen"] += 1
            match_id = _resolve_local(conn, row.get("game_match_id"), row.get("game_platform_id"))
            if not match_id or (only_match and match_id != only_match):
                if not match_id:
                    s["skipped"] += 1
                continue
            ev_path = row.get("events_path")
            if not ev_path or not Path(ev_path).exists():
                print(f"  {match_id}: events.csv.gz no encontrado ({ev_path}); omitido.")
                s["skipped"] += 1
                continue
            res = ingest_one(conn, match_id, Path(ev_path),
                             row.get("output_width"), row.get("output_height"))
            s["done"] += 1
            s["per_match"][match_id] = res["total"]
            print(f"  {match_id}: {res['total']} eventos -> {res['by_type']}")
    finally:
        adb.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return s


def main() -> int:
    cfg = load_config()
    paths = cfg["paths"]
    ascent_db = paths.get("ascent_recordings_db")
    if not ascent_db:
        print("config.toml [paths].ascent_recordings_db no está configurado.")
        return 1
    only_match = sys.argv[1] if len(sys.argv) > 1 else None

    conn = connect(paths["sqlite_db"])
    s = ingest_events_for_db(conn, Path(ascent_db), only_match)

    print(f"\nGrabaciones vistas: {s['seen']} | procesadas: {s['done']} | sin match/eventos: {s['skipped']}")
    if s["done"]:
        print("Siguiente: analyze.py <match> (incluye hesitation_v1) -> compute_reward.py -> complete_clips.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
