"""One command for the whole Ascent flow.

Replaces the manual chain:
  ingest_ascent.py -> ingest_ascent_events.py -> analyze.py <m>
  -> compute_reward.py <m> -> complete_clips.py

Steps:
  1. Ingest Ascent recordings: link clips, and pull any match not yet in the DB
     straight from Riot (no recorder session needed).
  2. For each Ascent match (new ones by default, or all with --all): parse its
     input log into input_events, run all detectors, persist, compute rewards.
  3. Normalize full replays and (re)assign clips to decisions.

Usage:
    .venv\\Scripts\\python.exe scripts\\sync_ascent.py          # new Ascent matches
    .venv\\Scripts\\python.exe scripts\\sync_ascent.py --all    # reprocess all
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling scripts

from lol_coach.accounts import load_accounts
from lol_coach.config import load_config
from lol_coach.db import connect
from lol_coach.riot_api import RiotAPI

import analyze
import complete_clips
import ingest_ascent
import ingest_ascent_events

ASCENT_SOURCE = "ascent_full"
log = logging.getLogger("sync_ascent")


def _ascent_match_ids(conn, only_new: bool) -> list[str]:
    """Ascent matches to process. "New" = the analyzed_at_utc marker is unset,
    which (unlike the old "has no decisions" test) repairs matches left half-done
    by a crash and doesn't reprocess 0-decision matches forever."""
    q = ("SELECT DISTINCT c.match_id FROM clips c JOIN matches m ON m.match_id = c.match_id "
         "WHERE c.source = ? AND c.match_id IS NOT NULL")
    if only_new:
        q += " AND m.analyzed_at_utc IS NULL"
    return [r["match_id"] for r in conn.execute(q, (ASCENT_SOURCE,))]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    all_flag = "--all" in sys.argv

    cfg = load_config()
    paths = cfg["paths"]
    if not paths.get("ascent_recordings_db"):
        print("config.toml [paths].ascent_recordings_db no está configurado.")
        return 1
    ascent_db = Path(paths["ascent_recordings_db"])
    ascent_dir = Path(paths["ascent_dir"]) if paths.get("ascent_dir") else None
    raw_dir = Path(paths["data_raw"])
    conn = connect(paths["sqlite_db"])

    # Riot API lets step 1 pull matches with no recorder session; optional.
    api = accounts = None
    try:
        api = RiotAPI(cfg["riot"]["api_key"], cfg["riot"]["routing"], cfg["riot"]["platform"])
        accounts = load_accounts(cfg, api, raw_dir.parent)
    except Exception as e:
        log.warning("Sin Riot API (%s): solo proceso partidas ya en la DB.", e)

    print("=" * 70)
    print("0) Ingest del recorder (sesiones Live Client 1Hz -> snapshots HP/maná)")
    # Reconnected 2026-06-09: the recorder kept recording but nothing ingested its
    # sessions since sync_ascent replaced analyze_recent — trade_v1 and HP/mana were
    # silently frozen at the 27 old matches. A session newly linked to an already-
    # analyzed match clears its marker so the analysis step re-runs it (trade_v1
    # fires now that snapshots exist; persist_decisions keeps the feedback).
    if api and accounts:
        from lol_coach.ingest import run_ingest
        linked_before = {r["match_id"] for r in conn.execute(
            "SELECT match_id FROM sessions WHERE match_id IS NOT NULL")}
        replays_dirs = [Path(p) for p in
                        (paths.get("replays_lol_dirs") or [paths.get("replays_lol_dir")]) if p]
        outplayed = Path(paths["outplayed_dir"]) if paths.get("outplayed_dir") else None
        try:
            s0 = run_ingest(conn, api, raw_dir, accounts, replays_dirs, outplayed)
            print(f"   sesiones={s0['sessions_seen']}  nuevas={s0['sessions_new']}  "
                  f"partidas añadidas={s0['matches_added']}")
            newly_linked = {r["match_id"] for r in conn.execute(
                "SELECT match_id FROM sessions WHERE match_id IS NOT NULL")} - linked_before
            if newly_linked:
                qmarks = ",".join("?" * len(newly_linked))
                conn.execute(f"UPDATE matches SET analyzed_at_utc=NULL WHERE match_id IN ({qmarks})",
                             tuple(newly_linked))
                conn.commit()
                print(f"   {len(newly_linked)} partida(s) re-marcadas para análisis (ya tienen snapshots 1Hz)")
        except Exception as e:
            log.warning("run_ingest fallo (%s): sigo con Ascent.", e)
    else:
        print("   (sin Riot API: se salta; las sesiones quedan para la próxima corrida)")

    print()
    print("1) Ingest de grabaciones Ascent (vincula clips + trae partidas de Riot)")
    s1 = ingest_ascent.ingest_ascent(conn, ascent_db, ascent_dir, api, accounts, raw_dir)
    print(f"   grabaciones={s1['recordings']}  traídas de Riot={s1['matches_pulled']}  "
          f"vinculadas={s1['linked']}  sin match={s1['unlinked']}")

    # Analysis covers EVERY pending match (recorder-only, Ascent, reference) — the
    # analyzed_at marker decides, not the clip source. Ascent matches get their
    # input events ingested first.
    if all_flag:
        targets = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches ORDER BY game_start_ms")]
    else:
        targets = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches WHERE analyzed_at_utc IS NULL ORDER BY game_start_ms")]
    ascent_mids = set(_ascent_match_ids(conn, only_new=False))
    print()
    print(f"2) Input events + análisis — {len(targets)} partida(s) "
          f"({'todas' if all_flag else 'pendientes'})")
    if not targets:
        print("   Nada pendiente (usa --all para reprocesar todo).")
    total_dec = 0
    for mid in targets:
        if mid in ascent_mids:
            ingest_ascent_events.ingest_events_for_db(conn, ascent_db, only_match=mid)
        n = analyze.analyze_match(conn, mid)  # detectors + reward + analyzed_at marker
        total_dec += n
        print(f"   {mid}: {n} decisiones + reward")

    print()
    print("3) Clips (normaliza full replays + asigna a decisiones)")
    nfull = complete_clips.normalize_full_replays(conn)
    res = complete_clips.complete_clips_for_decisions(conn)
    print(f"   full normalizados={nfull}  |  {res['short']} por jugada / "
          f"{res['full']} por replay / {res['none']} sin clip")

    print()
    print(f"LISTO. {len(targets)} partida(s), {total_dec} decisiones. "
          f"Abre el dashboard: python scripts/dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
