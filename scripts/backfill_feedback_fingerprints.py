"""Backfill decisions.user_feedback_fingerprint from DB backups.

The 2026-06-10 full re-analysis refreshed every decision's created_at_utc, so
the learner's timestamp staleness gate killed ALL best_option marks at once —
including the many whose option set never actually changed. This script
reconstructs, for each mark, the option set the user was LOOKING AT when they
marked it, using the DB snapshots in data/backups/ and data/*.bak*:

    the incarnation visible at mark time T is the snapshot row (same stable
    key match_id | detector_id | game_time_ms) with the newest
    created_at_utc <= T.

That row's action_fingerprint (features.action_fingerprint: ordered action_ids
+ taken index; label prose ignored) is written to user_feedback_fingerprint.
The learner gate (_decision_feedback_pairs) then revives the mark iff it
equals the CURRENT fingerprint — the options didn't change — and confirms it
stale otherwise. Marks with no qualifying snapshot stay NULL (conservative:
the timestamp gate keeps treating them as stale).

Residual risk, accepted: if a decision was rewritten between the chosen
snapshot and the mark (possible only for matches re-marked mid-day by sync
step 0), the fingerprint may describe a sibling incarnation. The failure mode
is a fingerprint MISMATCH (mark stays stale), never a wrong revival, unless
that intermediate incarnation changed action ids and changed them back —
not something any detector edit has done.

Dry-run by default; --apply writes. Idempotent: only fills NULL fingerprints.

    .venv\\Scripts\\python.exe scripts\\backfill_feedback_fingerprints.py            # informe
    .venv\\Scripts\\python.exe scripts\\backfill_feedback_fingerprints.py --apply    # escribe
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.config import load_config
from lol_coach.db import connect
from lol_coach.decisions.features import action_fingerprint


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _discover_snapshots(data_dir: Path, db_name: str) -> list[tuple[Path, sqlite3.Connection]]:
    """All usable DB snapshots, newest content first (by MAX(created_at_utc))."""
    candidates = sorted(data_dir.glob("backups/lol_coach_*.db")) + \
        sorted(data_dir.glob(f"{db_name}.bak*"))
    snaps = []
    for p in candidates:
        try:
            c = _open_ro(p)
            mx = c.execute("SELECT MAX(created_at_utc) m FROM decisions").fetchone()["m"]
            if mx:
                snaps.append((mx, p, c))
            else:
                c.close()
        except sqlite3.Error:
            continue
    snaps.sort(key=lambda t: t[0], reverse=True)
    return [(p, c) for _mx, p, c in snaps]


def _fingerprint_of_row(row) -> str | None:
    try:
        ctx = json.loads(row["context_json"])
        opts = json.loads(row["options_json"])
    except (ValueError, TypeError):
        return None
    return action_fingerprint(row["detector_id"], ctx.get("action"), opts)


def _snapshot_fingerprint(snaps, key, fb_at) -> tuple[str | None, str | None]:
    """(old_fingerprint, snapshot_name) for the incarnation visible at fb_at."""
    mid, det, gtms = key
    for path, conn in snaps:
        row = conn.execute(
            "SELECT detector_id, context_json, options_json, created_at_utc "
            "FROM decisions WHERE match_id=? AND detector_id=? AND game_time_ms=? "
            "ORDER BY created_at_utc DESC LIMIT 1",
            (mid, det, gtms),
        ).fetchone()
        if row is None:
            continue
        if row["created_at_utc"] and fb_at and row["created_at_utc"] <= fb_at:
            return _fingerprint_of_row(row), path.name
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill de fingerprints de feedback desde backups.")
    ap.add_argument("--apply", action="store_true", help="escribe en la DB (por defecto: solo informe)")
    ap.add_argument("--db", default=None, help="ruta de la DB principal (por defecto: config.toml)")
    args = ap.parse_args()

    cfg = load_config()
    db_path = Path(args.db) if args.db else Path(cfg["paths"]["sqlite_db"])
    conn = connect(db_path)  # runs migrations: ensures the column exists
    # Snapshots live NEXT TO the DB being filled (works for test copies too).
    data_dir = db_path.parent

    snaps = _discover_snapshots(data_dir, db_path.name)
    if not snaps:
        print("No hay snapshots utilizables en data/backups/ ni data/*.bak*; nada que hacer.")
        return 1
    print("Snapshots (contenido más nuevo primero):")
    for p, c in snaps:
        r = c.execute("SELECT MIN(created_at_utc) mn, MAX(created_at_utc) mx, COUNT(*) n FROM decisions").fetchone()
        print(f"  {p.name}: {r['n']} decisiones, created_at {r['mn'][:19]} .. {r['mx'][:19]}")

    rows = conn.execute(
        "SELECT id, match_id, detector_id, game_time_ms, options_json, context_json, "
        "user_feedback_at_utc FROM decisions "
        "WHERE user_best_option IS NOT NULL AND user_feedback_fingerprint IS NULL"
    ).fetchall()
    print(f"\nMarcas best_option sin fingerprint: {len(rows)}")

    tally = collections.Counter()
    by_detector = collections.defaultdict(collections.Counter)
    by_snapshot = collections.Counter()
    writes = []
    for r in rows:
        det = r["detector_id"]
        fb_at = r["user_feedback_at_utc"]
        if not fb_at:
            tally["sin_timestamp"] += 1
            by_detector[det]["sin_timestamp"] += 1
            continue
        key = (r["match_id"], det, r["game_time_ms"])
        old_fp, snap_name = _snapshot_fingerprint(snaps, key, fb_at)
        if old_fp is None:
            tally["sin_respaldo"] += 1     # no snapshot covers the mark's moment
            by_detector[det]["sin_respaldo"] += 1
            continue
        cur_fp = _fingerprint_of_row(r)
        if old_fp == cur_fp:
            tally["revividas"] += 1        # options unchanged: the mark is valid
            by_detector[det]["revividas"] += 1
        else:
            tally["stale_confirmadas"] += 1  # options really changed: stays out
            by_detector[det]["stale_confirmadas"] += 1
        by_snapshot[snap_name] += 1
        writes.append((old_fp, r["id"]))

    print("\nResultado por detector:")
    for det in sorted(by_detector):
        c = by_detector[det]
        print(f"  {det}: revividas={c['revividas']}  stale_confirmadas={c['stale_confirmadas']}  "
              f"sin_respaldo={c['sin_respaldo']}  sin_timestamp={c['sin_timestamp']}")
    print(f"\nTotal: revividas={tally['revividas']}  stale_confirmadas={tally['stale_confirmadas']}  "
          f"sin_respaldo={tally['sin_respaldo']}  sin_timestamp={tally['sin_timestamp']}")
    if by_snapshot:
        used = ", ".join(f"{k} ({v})" for k, v in by_snapshot.most_common())
        print(f"Snapshots usados: {used}")

    if not args.apply:
        print("\nDRY-RUN: no se escribió nada. Corre con --apply para guardar los fingerprints.")
        return 0

    conn.executemany(
        "UPDATE decisions SET user_feedback_fingerprint=? WHERE id=?", writes)
    conn.commit()
    print(f"\nEscritos {len(writes)} fingerprints. Las marcas 'revividas' vuelven a entrenar el learner; "
          f"las 'stale_confirmadas' quedan excluidas por contenido (no por timestamp).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
