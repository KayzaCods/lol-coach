"""Run an ingest pass: reconcile recorder sessions with Riot data + scan clips.

Usage:
    .venv\\Scripts\\python.exe scripts\\ingest.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.accounts import load_accounts
from lol_coach.config import load_config
from lol_coach.db import connect
from lol_coach.ingest import run_ingest
from lol_coach.riot_api import RiotAPI


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_raw"]).parent
    api = RiotAPI(
        api_key=cfg["riot"]["api_key"],
        routing=cfg["riot"]["routing"],
        platform=cfg["riot"]["platform"],
    )
    accounts = load_accounts(cfg, api, data_dir)

    conn = connect(cfg["paths"]["sqlite_db"])
    replays_dirs = cfg["paths"].get("replays_lol_dirs") or [cfg["paths"].get("replays_lol_dir")]
    replays_dirs = [Path(p) for p in replays_dirs if p]
    outplayed = cfg["paths"].get("outplayed_dir")
    stats = run_ingest(
        conn=conn,
        api=api,
        raw_dir=Path(cfg["paths"]["data_raw"]),
        accounts=accounts,
        replays_lol_dirs=replays_dirs,
        outplayed_dir=Path(outplayed) if outplayed else None,
    )

    print()
    print("=== Ingest stats ===")
    print(f"  Sessions seen:    {stats['sessions_seen']}")
    print(f"  Sessions new:     {stats['sessions_new']}")
    print(f"  Matches added:    {stats['matches_added']}")
    print(f"  Clips added:      {stats['clips_added']}")
    if stats["errors"]:
        print(f"  Errors ({len(stats['errors'])}):")
        for e in stats["errors"]:
            print(f"    - {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
