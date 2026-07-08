"""Reader for the recorder's per-second Live Client snapshots.

recorder.py writes snapshots.jsonl (one JSON object per line, one snapshot per
second). Detectors that want the player's exact HP/mana/items at a moment read
it back through here — a single loader instead of a near-identical copy in each
detector (trade and objective_readiness had diverged only in name).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_snapshots(session_dir: Path) -> list[tuple[float, dict]]:
    """[(game_time_s, snapshot_data)] from session_dir/snapshots.jsonl, sorted by
    game time. Empty list if the file is absent — the recorder is optional, and
    damage/timeline deltas drive the detectors without it."""
    path = session_dir / "snapshots.jsonl"
    if not path.exists():
        return []
    out: list[tuple[float, dict]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data = obj.get("data") or {}
            gt = (data.get("gameData") or {}).get("gameTime")
            if isinstance(gt, (int, float)):
                out.append((float(gt), data))
    out.sort(key=lambda t: t[0])
    return out
