"""Ingest Ascent recordings as clips, linked EXACTLY by Riot match id.

Ascent (the screen recorder that replaces our dependence on Overwolf/Replays.lol)
keeps its own SQLite at %LOCALAPPDATA%\\Ascent\\recordings.db with one row per
recording, including `game_match_id` (the Riot match, e.g. "LA1_1722354497") and
`video_path`. We read that DB and, for each recording:

  1. Resolve the Riot match id (defensive: `game_match_id` as-is, or composed
     with `game_platform_id` — handles both possible Ascent formats).
  2. If that match isn't in our DB yet, pull it straight from Riot match-v5
     (match + timeline) and persist it — no recorder session needed. This makes
     Ascent fully independent of recorder.py: we detect which of the configured
     accounts played by intersecting participants' puuids with ours.
  3. Insert one clip (source='ascent_full', in_match_start_s=0) linked to it.

Coexists with Replays.lol: only ADDS rows; never touches replays_lol_* clips.

Run order: scripts/ingest_ascent.py -> scripts/analyze.py <match> ->
scripts/compute_reward.py <match> -> scripts/complete_clips.py. Idempotent.

Matches with no recorder session won't have Live Client snapshots, so the
snapshot-based detectors (trade_v1, and HP/mana in objective_readiness) yield
less; the timeline-based detectors (death/tempo/vision) work fully.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.accounts import load_accounts
from lol_coach.config import load_config
from lol_coach.db import connect
from lol_coach.ingest import insert_match
from lol_coach.riot_api import RiotAPI, RiotAPIError

SOURCE = "ascent_full"
log = logging.getLogger("ingest_ascent")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_ascent_db_readonly(db_path: Path) -> tuple[sqlite3.Connection, Path]:
    """Copy the Ascent DB (+ WAL/SHM) to a temp dir and open the copy.

    Copying avoids locking / partial-WAL reads while Ascent may be running.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ascent_db_"))
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(db_path) + suffix)
        if src.exists():
            shutil.copy2(src, tmp / (db_path.name + suffix))
    conn = sqlite3.connect(str(tmp / db_path.name))
    conn.row_factory = sqlite3.Row
    return conn, tmp


def _candidate_match_ids(game_match_id, game_platform_id) -> list[str]:
    """Forms of the Riot match id to try (handles prefixed or split formats)."""
    if not game_match_id:
        return []
    gid = str(game_match_id).strip()
    out = [gid]  # already prefixed, e.g. "LA1_1722354497"
    if game_platform_id and "_" not in gid:
        plat = str(game_platform_id).strip()
        out += [f"{plat}_{gid}", f"{plat.upper()}_{gid}"]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _resolve_local(conn, game_match_id, game_platform_id) -> str | None:
    for cand in _candidate_match_ids(game_match_id, game_platform_id):
        if conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (cand,)).fetchone():
            return cand
    return None


def _riot_query_id(game_match_id, game_platform_id) -> str | None:
    """The PLATFORM_gameId form to query Riot match-v5 with."""
    for cand in _candidate_match_ids(game_match_id, game_platform_id):
        if "_" in cand:
            return cand.upper()
    return None


def _resolve_video_path(video_path, ascent_dir: Path | None) -> Path | None:
    if not video_path:
        return None
    p = Path(video_path)
    if not p.is_absolute() and ascent_dir is not None:
        p = ascent_dir / video_path
    return p


def _mtime_ms(video: Path | None, created_at) -> int | None:
    if video is not None and video.exists():
        return int(video.stat().st_mtime * 1000)
    if isinstance(created_at, (int, float)):
        return int(created_at if created_at > 1e12 else created_at * 1000)  # s vs ms
    return None


def _ingest_match_from_riot(conn, api, accounts, raw_dir: Path, qid: str) -> str | None:
    """Pull a match + timeline from Riot and persist it, with no recorder session.

    Returns the match_id on success, or None if none of our accounts played it or
    it's not coachable (normal/remake — tombstoned so it's never re-pulled).
    """
    from lol_coach.ingest import coachable_reason, is_skipped, mark_skipped
    if is_skipped(conn, qid):
        return None
    puuid_to_riot = {accounts.puuid_for(rid): rid for rid in accounts.all_riot_ids()}

    match = api.match(qid)
    reason = coachable_reason(match)
    if reason:
        mark_skipped(conn, qid, reason)
        conn.commit()
        log.info("Skip %s (%s): grabación sin partida coacheable.", qid, reason)
        return None
    participants = match["info"]["participants"]
    ours = next((p for p in participants if p.get("puuid") in puuid_to_riot), None)
    if ours is None:
        log.warning("Ninguna cuenta configurada jugó %s; clip quedará sin vincular.", qid)
        return None
    riot_id = puuid_to_riot[ours["puuid"]]
    cohort = accounts.cohort_for(riot_id)

    timeline = api.match_timeline(qid)
    out_dir = raw_dir / "_ascent" / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "match.json"
    tp = out_dir / "timeline.json"
    mp.write_text(json.dumps(match, indent=2), encoding="utf-8")
    tp.write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    insert_match(conn, match, timeline, ours["puuid"], riot_id, cohort,
                 None, mp, tp)  # session_dir=None: no recorder session
    conn.commit()
    log.info("Ingerida desde Riot: %s (cuenta %s, cohort %s)", qid, riot_id, cohort or "?")
    return qid


def ingest_ascent(conn, ascent_db: Path, ascent_dir, api, accounts, raw_dir: Path) -> dict:
    s = {"recordings": 0, "linked": 0, "unlinked": 0, "missing_file": 0,
         "matches_pulled": 0, "inserted": 0, "relinked": 0, "skipped": 0}
    if not ascent_db.exists():
        print(f"  Ascent DB no encontrada: {ascent_db}")
        return s

    adb, tmp = _open_ascent_db_readonly(ascent_db)
    try:
        try:
            rows = adb.execute("SELECT * FROM recordings").fetchall()
        except sqlite3.OperationalError as e:
            print(f"  No pude leer la tabla 'recordings': {e}")
            return s
        for r in rows:
            row = dict(r)
            s["recordings"] += 1
            video = _resolve_video_path(row.get("video_path"), ascent_dir)
            if video is None:
                continue

            match_id = _resolve_local(conn, row.get("game_match_id"), row.get("game_platform_id"))
            if match_id is None and api is not None and accounts is not None:
                qid = _riot_query_id(row.get("game_match_id"), row.get("game_platform_id"))
                if qid:
                    try:
                        match_id = _ingest_match_from_riot(conn, api, accounts, raw_dir, qid)
                        if match_id:
                            s["matches_pulled"] += 1
                    except RiotAPIError as e:
                        log.warning("Riot API falló para %s: %s", qid, e)

            if match_id:
                s["linked"] += 1
            else:
                s["unlinked"] += 1
            if not video.exists():
                s["missing_file"] += 1

            path_str = str(video)
            existing = conn.execute("SELECT match_id FROM clips WHERE path = ?", (path_str,)).fetchone()
            if existing:
                if existing["match_id"] is None and match_id is not None:
                    conn.execute(
                        "UPDATE clips SET match_id = ?, in_match_start_s = 0 WHERE path = ?",
                        (match_id, path_str),
                    )
                    s["relinked"] += 1
                else:
                    s["skipped"] += 1
                continue

            conn.execute(
                """
                INSERT INTO clips (
                    path, source, file_mtime_ms, file_size, duration_s,
                    match_id, in_match_start_s, champion_hint, event_hint, discovered_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path_str, SOURCE, _mtime_ms(video, row.get("created_at")),
                    (video.stat().st_size if video.exists() else None), row.get("duration_s"),
                    match_id, (0.0 if match_id else None), None, None, _utc_now_iso(),
                ),
            )
            s["inserted"] += 1
        conn.commit()
    finally:
        adb.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return s


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    paths = cfg["paths"]
    ascent_db = paths.get("ascent_recordings_db")
    if not ascent_db:
        print("config.toml [paths].ascent_recordings_db no está configurado.")
        return 1
    ascent_dir = Path(paths["ascent_dir"]) if paths.get("ascent_dir") else None
    raw_dir = Path(paths["data_raw"])

    # Riot API + accounts enable pulling matches that have no recorder session.
    # If the key is dead/missing we still link clips to already-ingested matches.
    api = accounts = None
    try:
        api = RiotAPI(cfg["riot"]["api_key"], cfg["riot"]["routing"], cfg["riot"]["platform"])
        accounts = load_accounts(cfg, api, raw_dir.parent)
    except Exception as e:
        log.warning("Sin Riot API (%s). Solo vincularé clips a partidas ya ingeridas.", e)
        api = accounts = None

    conn = connect(paths["sqlite_db"])
    s = ingest_ascent(conn, Path(ascent_db), ascent_dir, api, accounts, raw_dir)

    print("\n=== Ascent ingest ===")
    print(f"  Grabaciones en Ascent:        {s['recordings']}")
    print(f"  Partidas traídas de Riot:     {s['matches_pulled']}")
    print(f"  Clips vinculados a partida:   {s['linked']}")
    print(f"  Sin match en DB todavía:      {s['unlinked']}")
    print(f"  Archivo no en disco:          {s['missing_file']}")
    print(f"  Insertados / re-link / igual: {s['inserted']} / {s['relinked']} / {s['skipped']}")
    if s["recordings"] == 0:
        print("  (Sin grabaciones. Graba una partida con Ascent y vuelve a correr.)")
    elif s["linked"]:
        print("  Siguiente: analyze.py <match> -> compute_reward.py <match> -> complete_clips.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
