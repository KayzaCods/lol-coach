"""Live Client polling recorder.

Runs as a background process. While no game is running it polls every few
seconds at near-zero cost. When a game is detected it polls /allgamedata
once per second and appends each snapshot to snapshots.jsonl. When the
Live Client stops responding for GAME_END_GRACE_S seconds it closes the
session and writes summary.json.

Output layout:
    <data_raw>/
      2026-05-18_22-13-45/
        snapshots.jsonl   (one JSON per line, one snapshot per second)
        summary.json      (start/end times, snapshot count, game time range)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from . import liveclient

log = logging.getLogger(__name__)

GAME_END_GRACE_S = 10.0
IDLE_POLL_S = 5.0


class RecorderSession:
    def __init__(self, raw_dir: Path):
        self.start_dt = datetime.now(timezone.utc)
        stamp = self.start_dt.strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = raw_dir / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_path = self.dir / "snapshots.jsonl"
        self.summary_path = self.dir / "summary.json"
        self._fh = open(self.snapshots_path, "w", encoding="utf-8")
        self.snapshot_count = 0
        self.first_game_time: float | None = None
        self.last_game_time: float | None = None
        self.active_player_name: str | None = None
        log.info("Session opened at %s", self.dir)

    def write(self, data: dict) -> None:
        record = {"ts": time.time(), "data": data}
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.snapshot_count += 1

        gt = None
        try:
            gt = data["gameData"]["gameTime"]
        except (KeyError, TypeError):
            pass
        if isinstance(gt, (int, float)):
            if self.first_game_time is None:
                self.first_game_time = gt
            self.last_game_time = gt

        if self.active_player_name is None:
            try:
                self.active_player_name = data["activePlayer"]["summonerName"]
            except (KeyError, TypeError):
                pass

    def close(self) -> None:
        self._fh.close()
        end_dt = datetime.now(timezone.utc)
        summary = {
            "start_utc": self.start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "snapshots": self.snapshot_count,
            "first_game_time_s": self.first_game_time,
            "last_game_time_s": self.last_game_time,
            "active_player_name": self.active_player_name,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(
            "Session closed. Snapshots=%d, gameTime=%.1f..%.1f",
            self.snapshot_count,
            self.first_game_time or 0,
            self.last_game_time or 0,
        )


def run_forever(raw_dir: Path, poll_interval_s: float = 1.0) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("Recorder loop started. Output dir: %s", raw_dir)

    session: RecorderSession | None = None
    last_seen = 0.0

    while True:
        try:
            data = liveclient.all_game_data()
        except liveclient.LiveClientUnavailable:
            if session is not None and (time.time() - last_seen) > GAME_END_GRACE_S:
                session.close()
                session = None
            time.sleep(IDLE_POLL_S if session is None else poll_interval_s)
            continue

        if session is None:
            session = RecorderSession(raw_dir)
        try:
            session.write(data)
        except Exception:
            log.exception("Failed to write snapshot")
        last_seen = time.time()
        time.sleep(poll_interval_s)
