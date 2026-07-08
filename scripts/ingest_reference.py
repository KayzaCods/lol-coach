"""Ingest a reference player's matches (e.g. a Challenger one-trick) for comparison.

Pulls a player's recent matches from Riot by Riot ID, keeps the ones where they
played the target champion, and ingests them tagged with a distinct cohort so the
dashboard's cohort filter compares them against your own play.

No recorder session and no Ascent recording — only Riot's Match-V5 timeline. So the
input-based detectors (hesitation, awareness) and video clips do NOT apply to these
games; the macro detectors (death, objective_readiness, tempo, trade) and HEF/reward
all run normally. The reference player is NOT added to config.toml, so the automatic
ingest never touches their account.

Usage:
    .venv\\Scripts\\python.exe scripts\\ingest_reference.py "PlayerName#TAG" --champion Sona
    .venv\\Scripts\\python.exe scripts\\ingest_reference.py "PlayerName#TAG" --cohort challenger --champion Sona --keep 20 --queue 420
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling scripts

from lol_coach.analysis import persist_decisions, run_all_detectors
from lol_coach.config import load_config
from lol_coach.db import connect
from lol_coach.ingest import coachable_reason, insert_match
from lol_coach.riot_api import RiotAPI, RiotAPIError

import compute_reward

log = logging.getLogger("ingest_reference")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest a reference player's matches as a cohort.")
    ap.add_argument("riot_id", help='GameName#TagLine, e.g. "PlayerName#TAG"')
    ap.add_argument("--cohort", default="challenger", help="cohort tag for these matches (default: challenger)")
    ap.add_argument("--champion", default=None, help="only keep games on this champion, e.g. Sona")
    ap.add_argument("--keep", type=int, default=20, help="how many matching games to ingest (default: 20)")
    ap.add_argument("--queue", type=int, default=420, help="queue id, 420=ranked solo (default); ignored with --all-queues")
    ap.add_argument("--all-queues", action="store_true", help="don't filter by queue")
    ap.add_argument("--scan", type=int, default=50, help="how many recent matches to scan (default: 50)")
    args = ap.parse_args()

    if "#" not in args.riot_id:
        print('Riot ID debe tener el formato "GameName#TagLine".')
        return 1
    game_name, tag_line = args.riot_id.rsplit("#", 1)

    cfg = load_config()
    paths = cfg["paths"]
    raw_dir = Path(paths["data_raw"])
    conn = connect(paths["sqlite_db"])
    api = RiotAPI(cfg["riot"]["api_key"], cfg["riot"]["routing"], cfg["riot"]["platform"])

    try:
        acc = api.account_by_riot_id(game_name, tag_line)
    except RiotAPIError as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "apikey" in msg.lower():
            print("Riot API key inválida/expirada. Renueva config.toml [riot].api_key y reintenta.")
        else:
            print(f"No se pudo resolver {args.riot_id}: {msg[:200]}")
        return 1
    puuid = acc["puuid"]
    print(f"{args.riot_id} resuelto. Buscando sus partidas recientes...")

    queue = None if args.all_queues else args.queue
    try:
        ids = api.match_ids_by_puuid(puuid, count=args.scan, queue=queue)
    except RiotAPIError as e:
        print(f"No se pudieron listar partidas: {str(e)[:200]}")
        return 1

    out_root = raw_dir / "_reference"
    kept = scanned = skipped_existing = 0
    for mid in ids:
        if kept >= args.keep:
            break
        if conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (mid,)).fetchone():
            skipped_existing += 1
            continue  # already in DB (yours or a previous reference run) — don't re-pull
        try:
            match = api.match(mid)
        except RiotAPIError as e:
            log.warning("skip %s: %s", mid, str(e)[:120])
            continue
        scanned += 1
        if coachable_reason(match):
            continue  # remake/special — no coaching signal
        p = next((x for x in match["info"]["participants"] if x.get("puuid") == puuid), None)
        if p is None:
            continue
        if args.champion and (p.get("championName") or "").lower() != args.champion.lower():
            continue
        timeline = api.match_timeline(mid)
        out_dir = out_root / mid
        out_dir.mkdir(parents=True, exist_ok=True)
        mp, tp = out_dir / "match.json", out_dir / "timeline.json"
        mp.write_text(json.dumps(match, indent=2), encoding="utf-8")
        tp.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        # session_dir=None: no recorder session. our_puuid = the reference player.
        insert_match(conn, match, timeline, puuid, args.riot_id, args.cohort, None, mp, tp)
        decisions = run_all_detectors(conn, mid)
        n, _fb = persist_decisions(conn, mid, decisions)
        try:
            compute_reward.run_for_match(conn, mid)
        except Exception as e:  # reward is best-effort (no recorder data for ref games)
            log.warning("reward %s: %s", mid, str(e)[:120])
        # Mark the full chain done (detectors + reward; reference games have no
        # clips), so the next sync_ascent run doesn't re-analyze them once more.
        conn.execute(
            "UPDATE matches SET analyzed_at_utc=? WHERE match_id=?",
            (datetime.now(timezone.utc).isoformat(), mid),
        )
        kept += 1
        print(f"  {mid}  {'V' if p['win'] else 'D'} {p['kills']}/{p['deaths']}/{p['assists']}  "
              f"{round(match['info']['gameDuration'] / 60)}min  -> {n} decisiones")

    conn.commit()
    champ = args.champion or "cualquier campeón"
    print()
    print(f"LISTO. {kept} partidas de {champ} ingeridas como cohort '{args.cohort}' "
          f"({skipped_existing} ya estaban en la DB).")
    if kept:
        print("Refresca el dashboard y usa el filtro de cohort para comparar.")
    else:
        print("No se ingirió ninguna. Sube --scan, revisa el campeón, o usa --all-queues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
