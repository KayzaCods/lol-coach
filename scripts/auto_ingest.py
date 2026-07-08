"""Automated Ascent ingestion — run by the LoLCoachIngest scheduled task.

Runs the sync_ascent pipeline for NEW Ascent matches only, appends a timestamped
record to data/ingest.log, and writes data/ingest_status.json so the dashboard
(and you) can see the last run at a glance — including a clear flag when the Riot
dev API key has expired, which is the usual reason ingestion silently does nothing.

Idempotent and cheap: if there is nothing new it does almost no work. Safe to run
on a timer. Designed to be invisible (launch with pythonw.exe).
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # project root
sys.path.insert(0, str(ROOT / "scripts"))     # sibling scripts (sync_ascent, etc.)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from lol_coach.config import load_config  # noqa: E402

DATA = ROOT / "data"
LOG = DATA / "ingest.log"
STATUS = DATA / "ingest_status.json"
LOG_MAX_BYTES = 512 * 1024  # keep the log small; trim oldest when it grows past this


def _key_alive(cfg) -> "bool | None":
    """True/False whether the Riot key can read match-v5; None if undeterminable.

    Tests the exact capability ingestion needs (a match-v5 read) against a match
    already in the DB. If the DB has no match yet, we cannot test -> None.
    """
    riot = cfg.get("riot", {})
    key = riot.get("api_key")
    routing = riot.get("routing", "americas")
    if not key:
        return False
    import sqlite3
    try:
        db = sqlite3.connect(cfg["paths"]["sqlite_db"])
        row = db.execute("SELECT match_id FROM matches LIMIT 1").fetchone()
        db.close()
    except Exception:
        row = None
    if not row:
        return None
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{row[0]}"
    # Use requests, like the rest of the pipeline: the Riot API rejects the
    # default Python-urllib User-Agent with 403, so urllib gives false negatives.
    # Retry on 401/403 in case it's a transient throttle before declaring dead.
    for attempt in range(3):
        try:
            code = requests.get(url, headers={"X-Riot-Token": key}, timeout=10).status_code
        except Exception:
            return None
        if code not in (401, 403):
            return True  # 200 ok; 429/5xx -> key valid, just throttled/down
        if attempt < 2:
            time.sleep(2)
    return False  # 401/403 persisted across retries -> really expired/invalid


def _trim_log() -> None:
    try:
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            tail = LOG.read_text(encoding="utf-8", errors="replace")[-LOG_MAX_BYTES // 2:]
            LOG.write_text("...(recortado)...\n" + tail, encoding="utf-8")
    except Exception:
        pass


BACKUP_KEEP = 7  # daily backups retained


def _daily_db_backup(cfg) -> str | None:
    """One consistent DB backup per day in data/backups/, keeping the last 7.

    Uses sqlite's online backup API (WAL-safe), so it's valid even while the
    dashboard or an analysis is writing. Runs regardless of Riot key state —
    the user's feedback lives only in this DB. Returns the path written today,
    or None if today's backup already existed / on failure (logged by caller).
    """
    import sqlite3
    db_path = cfg["paths"]["sqlite_db"]
    bdir = DATA / "backups"
    bdir.mkdir(exist_ok=True)
    dest = bdir / f"lol_coach_{datetime.now():%Y%m%d}.db"
    if dest.exists():
        return None
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(dest)
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()
    backups = sorted(bdir.glob("lol_coach_*.db"))
    for old in backups[:-BACKUP_KEEP]:
        old.unlink(missing_ok=True)
    return str(dest)


def main() -> int:
    DATA.mkdir(exist_ok=True)
    _trim_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = load_config()
    alive = _key_alive(cfg)
    status = {"last_run": ts, "key_alive": alive, "ingested": 0, "error": None}

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {ts} =====\n")
        try:
            bpath = _daily_db_backup(cfg)
            if bpath:
                f.write(f"backup diario de la DB: {bpath}\n")
        except Exception as e:
            f.write(f"BACKUP FALLO: {e!r}\n")
        if alive is False:
            msg = ("API KEY EXPIRADA/INVALIDA (HTTP 401/403). Renueva en "
                   "https://developer.riotgames.com y pegala en config.toml "
                   "[riot] api_key. La ingesta de partidas nuevas no correra "
                   "hasta entonces.")
            f.write(msg + "\n")
            status["error"] = "api_key_expired"
            STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0

        buf = io.StringIO()
        old_argv = sys.argv
        try:
            import sync_ascent
            sys.argv = ["sync_ascent.py"]  # only-new (no --all)
            with redirect_stdout(buf):
                sync_ascent.main()
            out = buf.getvalue()
            f.write(out)
            m = re.search(r"LISTO\. (\d+) partida", out)
            status["ingested"] = int(m.group(1)) if m else 0
        except Exception as e:  # never let the timer task crash; record it
            f.write(buf.getvalue())
            f.write(f"ERROR: {e!r}\n")
            status["error"] = str(e)
        finally:
            sys.argv = old_argv

    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
