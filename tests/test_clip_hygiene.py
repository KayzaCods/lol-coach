"""Clip hygiene (#3): ingest skips pre-history clips that can never link to a
match; complete_clips doesn't wipe assignments when the video disk is unmounted."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lol_coach import ingest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import complete_clips  # noqa: E402

EARLIEST = 1_700_000_000_000   # epoch ms ~2023-11, stands in for the first match
_NOW = "2026-01-01T00:00:00Z"


def _add_clip(conn, path, source="replays_lol_full"):
    conn.execute(
        "INSERT INTO clips (path, source, discovered_at_utc) VALUES (?, ?, ?)",
        (path, source, _NOW),
    )
    conn.commit()


# --- ingest: pre-history skip ---

def test_maybe_insert_skips_prehistory_clip(conn, tmp_path):
    f = tmp_path / "old.mp4"; f.write_bytes(b"x")
    os.utime(f, (1_600_000_000, 1_600_000_000))   # mtime ~2020, before EARLIEST
    n = ingest._maybe_insert_clip(conn, f, "outplayed_clip",
                                  lambda m: None, earliest_match_ms=EARLIEST)
    assert n == 0
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 0


def test_maybe_insert_keeps_recent_unmatched_clip(conn, tmp_path):
    # After EARLIEST but no match yet -> still inserted; the re-link loop can
    # attach it once its match is ingested (deferred linking preserved).
    f = tmp_path / "recent.mp4"; f.write_bytes(b"x")
    os.utime(f, (1_700_000_100, 1_700_000_100))
    n = ingest._maybe_insert_clip(conn, f, "outplayed_clip",
                                  lambda m: None, earliest_match_ms=EARLIEST)
    assert n == 1
    assert conn.execute("SELECT match_id FROM clips").fetchone()["match_id"] is None


# --- complete_clips: disk-unmounted guard ---

def test_any_clip_on_disk_false_when_unmounted(conn):
    _add_clip(conn, "/nope/a.mp4")
    assert complete_clips._any_clip_on_disk(conn) is False


def test_any_clip_on_disk_true_when_a_file_exists(conn, tmp_path):
    f = tmp_path / "real.mp4"; f.write_bytes(b"x")
    _add_clip(conn, str(f))
    assert complete_clips._any_clip_on_disk(conn) is True


def test_complete_clips_noop_when_disk_unmounted(conn):
    # A clip whose file does not exist = disk unmounted -> the function must not
    # touch decisions (no mass clip_path NULLing).
    _add_clip(conn, "/nope/a.mp4")
    res = complete_clips.complete_clips_for_decisions(conn)
    assert res.get("disk_unmounted") is True


def test_purge_only_outplayed_unmatched(conn):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import purge_orphan_clips
    conn.execute("INSERT INTO clips (path, source, match_id, discovered_at_utc) "
                 "VALUES ('/o1', 'outplayed_clip', NULL, ?)", (_NOW,))
    conn.execute("INSERT INTO clips (path, source, match_id, discovered_at_utc) "
                 "VALUES ('/o2', 'outplayed_clip', 'M', ?)", (_NOW,))
    conn.execute("INSERT INTO clips (path, source, match_id, discovered_at_utc) "
                 "VALUES ('/r1', 'replays_lol_full', NULL, ?)", (_NOW,))
    conn.commit()
    n = purge_orphan_clips.purge(conn, apply=True)
    assert n == 1
    assert {r["path"] for r in conn.execute("SELECT path FROM clips")} == {"/o2", "/r1"}
