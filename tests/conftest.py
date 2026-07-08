"""Shared fixtures + low-level builders for detector integration tests.

Detectors take `(conn, match_id)` and read a populated SQLite DB (participants,
timeline_frames, timeline_events). These helpers synthesize a minimal but valid
match in an on-disk temp DB built by the real `db.connect()` (so the schema and
migrations under test are the production ones), letting each test assemble just
the events it needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_coach import db as dbmod


@pytest.fixture
def conn(tmp_path):
    c = dbmod.connect(tmp_path / "test.db")
    try:
        yield c
    finally:
        c.close()


# Standard 10-player layout. Bottom-lane death scene at ~(10000, 4000) on the
# blue (team 100) side. Enemy jungler is pid 8. Positions are chosen so that the
# off-fight enemies (7, 9, 10) sit beyond THREAT_RADIUS of the death and are thus
# NOT local threats — keeping the scene clean for assertions.
_PLAYERS = [
    # pid, team, position, champion, (x, y)
    (1, 100, "BOTTOM", "Jinx", (10000, 4000)),    # us
    (2, 100, "TOP", "Aatrox", (3000, 12000)),
    (3, 100, "MIDDLE", "Ahri", (7500, 7500)),
    (4, 100, "JUNGLE", "LeeSin", (6000, 6000)),
    (5, 100, "UTILITY", "Thresh", (8000, 6500)),
    (6, 200, "BOTTOM", "Caitlyn", (10200, 3800)),  # assister
    (7, 200, "TOP", "Darius", (3000, 12500)),
    (8, 200, "JUNGLE", "Elise", (10100, 4100)),    # killer (ganks from fog)
    (9, 200, "MIDDLE", "Zed", (4000, 11000)),
    (10, 200, "UTILITY", "Lulu", (3500, 11500)),
]

OUR_PUUID = "puuid_1"
DEATH_MS = 600_000          # 10:00
FRAME_TS = (480_000, 540_000, 600_000)


def insert_match(conn, match_id="TEST", our_puuid=OUR_PUUID):
    conn.execute(
        "INSERT INTO matches (match_id, game_creation_ms, game_start_ms, "
        "game_duration_s, queue_id, our_puuid, our_cohort, ingested_at_utc) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (match_id, 0, 0, 1800, 420, our_puuid, "master", "2026-01-01T00:00:00Z"),
    )


def insert_event(conn, match_id, ts_ms, etype, payload, *, frame_index=0, pid=None):
    conn.execute(
        "INSERT INTO timeline_events (match_id, frame_index, timestamp_ms, type, "
        "participant_id, payload_json) VALUES (?,?,?,?,?,?)",
        (match_id, frame_index, ts_ms, etype, pid, json.dumps(payload)),
    )


def ten_player_match(conn, match_id="TEST"):
    """Insert a match, its 10 participants, and per-minute frames (positions,
    level, gold, cs). Returns a dict with useful ids for the test."""
    insert_match(conn, match_id)
    for pid, team, pos, champ, (x, y) in _PLAYERS:
        puuid = OUR_PUUID if pid == 1 else f"puuid_{pid}"
        conn.execute(
            "INSERT INTO participants (match_id, participant_id, puuid, team_id, "
            "team_position, champion_name, win) VALUES (?,?,?,?,?,?,?)",
            (match_id, pid, puuid, team, pos, champ, 1 if team == 100 else 0),
        )
        for fi, ts in enumerate(FRAME_TS, start=8):
            # cs climbs a little each frame so wave/cs helpers have signal.
            cs = 80 + fi * 4
            conn.execute(
                "INSERT INTO timeline_frames (match_id, frame_index, timestamp_ms, "
                "participant_id, position_x, position_y, current_gold, total_gold, "
                "level, minions_killed, jungle_minions_killed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (match_id, fi, ts, pid, x, y, 500, 8000, 11, cs, 0),
            )
    conn.commit()
    return {"match_id": match_id, "us_id": 1, "killer_id": 8, "assister_id": 6,
            "death_ms": DEATH_MS}


def set_timeline_json(conn, match_id, path, frames):
    """Write a raw Riot timeline JSON (only the `info.frames[].participantFrames`
    the trade detector reads) and point matches.timeline_json_path at it."""
    Path(path).write_text(json.dumps({"info": {"frames": frames}}), encoding="utf-8")
    conn.execute("UPDATE matches SET timeline_json_path=? WHERE match_id=?", (str(path), match_id))
    conn.commit()


def damage_frame(**taken_by_pid):
    """One timeline frame: {pid: physicalDamageTaken}. Cumulative across frames
    (Riot's damageStats are running totals)."""
    pfs = {
        str(pid): {"damageStats": {"physicalDamageTaken": v, "magicDamageTaken": 0,
                                   "trueDamageTaken": 0}}
        for pid, v in taken_by_pid.items()
    }
    return {"participantFrames": pfs}
