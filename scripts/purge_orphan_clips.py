"""Purge orphaned Outplayed clips: per-play clips with no match_id. They belong
to games that were never ingested (their mtime falls in no match window), so they
can never link and just get re-scanned every sync (#3). Dry-run by default.

    .venv\\Scripts\\python.exe scripts\\purge_orphan_clips.py          # dry run (cuenta)
    .venv\\Scripts\\python.exe scripts\\purge_orphan_clips.py --apply  # borra
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WHERE = "source = 'outplayed_clip' AND match_id IS NULL"


def purge(conn, apply: bool) -> int:
    """Count (and, if apply, delete) orphaned Outplayed clips. Pure: takes conn."""
    n = conn.execute(f"SELECT COUNT(*) FROM clips WHERE {WHERE}").fetchone()[0]
    if apply:
        conn.execute(f"DELETE FROM clips WHERE {WHERE}")
        conn.commit()
    return n


def main() -> int:
    from lol_coach.config import load_config
    from lol_coach.db import connect
    apply = "--apply" in sys.argv[1:]
    cfg = load_config()
    conn = connect(cfg["paths"]["sqlite_db"])
    n = purge(conn, apply)
    if apply:
        print(f"Borrados {n} clips huerfanos de Outplayed (source=outplayed_clip, match_id NULL).")
    else:
        print(f"[dry-run] {n} clips huerfanos de Outplayed listos para borrar.")
        print("Haz backup de la DB y corre con --apply para ejecutar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
