"""Reconcile recorder sessions with Riot match data and persist to SQLite.

For each session directory in data/raw/ that has not been ingested:
1. Read summary.json for start_utc.
2. Ask Riot for match IDs within +/- match_window of that timestamp for our PUUID.
3. Pick the one whose gameStartTimestamp lies closest to session.start_utc.
4. Cache match.json and timeline.json into the session dir.
5. Insert match + participants + timeline_events + timeline_frames into SQLite.
6. Insert/update sessions row with link to match_id.

Also scans the configured video directories for new clips and links them
to matches by mtime range.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .accounts import AccountRegistry
from .riot_api import RiotAPI, RiotAPIError

log = logging.getLogger(__name__)

# Tolerance for matching inferred game start vs Riot's gameStartTimestamp.
# Difference is normally <10s (Live Client responds slightly before/after
# Riot's gameStart depending on polling cadence).
MATCH_TOLERANCE_S = 120

# Common queue IDs -> human names. Riot publishes the full list at
# https://static.developer.riotgames.com/docs/lol/queues.json
QUEUE_NAMES = {
    0: "Custom",
    400: "Normal Draft",
    420: "Ranked Solo/Duo",
    430: "Normal Blind",
    440: "Ranked Flex",
    450: "ARAM",
    700: "Clash",
    720: "ARAM Clash",
    830: "Co-op vs AI Intro",
    840: "Co-op vs AI Beginner",
    850: "Co-op vs AI Intermediate",
    900: "ARURF",
    1020: "One for All",
    1300: "Nexus Blitz",
    1400: "Ultimate Spellbook",
    1700: "Arena",
    1900: "Pick URF",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_epoch_s(iso_str: str) -> int:
    return int(datetime.fromisoformat(iso_str).timestamp())


# ---------------------------------------------------------------- sessions

def discover_sessions(raw_dir: Path) -> list[Path]:
    """Return all session directories under data/raw/ that have a summary.json."""
    if not raw_dir.exists():
        return []
    out = []
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and (d / "summary.json").exists():
            out.append(d)
    return out


def already_ingested(conn: sqlite3.Connection, session_dir: Path) -> bool:
    row = conn.execute(
        "SELECT match_id FROM sessions WHERE session_dir = ?",
        (str(session_dir),),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------- matching

def find_match_for_session(
    api: RiotAPI,
    puuid: str,
    session_start_iso: str,
    first_game_time_s: float | None,
) -> str | None:
    """Find the Riot match whose gameStartTimestamp lines up with this session.

    The recorder may have started mid-game (e.g. user enabled it after the
    match began). The first snapshot's gameTime tells us how far into the
    game we were, so the true game start is approximately:

        session.start_utc - first_game_time_s

    Returns None if no match is within MATCH_TOLERANCE_S (typical for
    custom games / practice tool, which are not in match history).
    """
    session_epoch_s = _iso_to_epoch_s(session_start_iso)
    offset = first_game_time_s or 0.0
    inferred_start = session_epoch_s - offset

    # Empirically, Riot filters startTime/endTime by gameEndTimestamp, not
    # by gameStartTimestamp. So we widen the window to cover possible game
    # durations (5-65 min after start). The MATCH_TOLERANCE_S check below
    # filters out candidates that aren't actually the right match.
    candidate_ids = api.match_ids_by_puuid(
        puuid,
        count=10,
        start_time=int(inferred_start),
        end_time=int(inferred_start + 65 * 60),
    )
    if not candidate_ids:
        return None

    best_id = None
    best_delta = None
    for mid in candidate_ids:
        match = api.match(mid)
        gs_s = match["info"]["gameStartTimestamp"] / 1000
        delta = abs(gs_s - inferred_start)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_id = mid

    if best_delta is not None and best_delta > MATCH_TOLERANCE_S:
        return None
    return best_id


# ---------------------------------------------------------------- persist

COACHABLE_QUEUES = (420, 440)    # ranked solo/duo + ranked flex (user rule 2026-06-10)
MIN_COACHABLE_DURATION_S = 300   # below this it's a remake — no coaching signal


def coachable_reason(match: dict) -> str | None:
    """None if the match is worth coaching; else the rejection reason.
    Normals/specials and remakes pollute the analysis (player request)."""
    info = match.get("info") or {}
    q = info.get("queueId")
    if q not in COACHABLE_QUEUES:
        return f"queue_{q}_not_ranked"
    if (info.get("gameDuration") or 0) < MIN_COACHABLE_DURATION_S:
        return "remake"
    return None


def mark_skipped(conn: sqlite3.Connection, match_id: str, reason: str) -> None:
    """Tombstone so no ingest path ever re-pulls this match from Riot."""
    conn.execute(
        "INSERT OR REPLACE INTO skipped_matches (match_id, reason, at_utc) VALUES (?, ?, ?)",
        (match_id, reason, _utc_now_iso()),
    )


def is_skipped(conn: sqlite3.Connection, match_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM skipped_matches WHERE match_id = ?", (match_id,)
    ).fetchone() is not None


def insert_match(
    conn: sqlite3.Connection,
    match: dict,
    timeline: dict,
    our_puuid: str,
    our_account_riot_id: str | None,
    our_cohort: str | None,
    session_dir: Path | None,
    match_json_path: Path,
    timeline_json_path: Path,
) -> None:
    info = match["info"]
    metadata = match["metadata"]
    match_id = metadata["matchId"]
    platform = match_id.split("_")[0]

    conn.execute(
        """
        INSERT OR REPLACE INTO matches (
            match_id, platform, game_creation_ms, game_start_ms,
            game_duration_s, queue_id, queue_name, game_mode, game_type,
            map_id, game_version, our_puuid, our_account_riot_id, our_cohort,
            session_dir, match_json_path, timeline_json_path, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_id,
            platform,
            info.get("gameCreation"),
            info.get("gameStartTimestamp"),
            info.get("gameDuration"),
            info.get("queueId"),
            QUEUE_NAMES.get(info.get("queueId"), f"Queue {info.get('queueId')}"),
            info.get("gameMode"),
            info.get("gameType"),
            info.get("mapId"),
            info.get("gameVersion"),
            our_puuid,
            our_account_riot_id,
            our_cohort,
            str(session_dir) if session_dir else None,
            str(match_json_path),
            str(timeline_json_path),
            _utc_now_iso(),
        ),
    )

    # Wipe and reinsert participants for idempotency
    conn.execute("DELETE FROM participants WHERE match_id = ?", (match_id,))
    for p in info["participants"]:
        conn.execute(
            """
            INSERT INTO participants (
                match_id, participant_id, puuid, riot_id_game_name, riot_id_tag_line,
                team_id, team_position, champion_name, champion_id, win,
                kills, deaths, assists,
                total_minions_killed, neutral_minions_killed,
                vision_score, wards_placed, wards_killed,
                gold_earned, gold_spent,
                total_damage_dealt_to_champions, total_damage_taken, total_heal,
                longest_time_spent_living
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                p["participantId"],
                p["puuid"],
                p.get("riotIdGameName"),
                p.get("riotIdTagline"),
                p["teamId"],
                p.get("teamPosition") or p.get("individualPosition"),
                p["championName"],
                p.get("championId"),
                int(p["win"]),
                p.get("kills"), p.get("deaths"), p.get("assists"),
                p.get("totalMinionsKilled"),
                p.get("neutralMinionsKilled"),
                p.get("visionScore"),
                p.get("wardsPlaced"),
                p.get("wardsKilled"),
                p.get("goldEarned"),
                p.get("goldSpent"),
                p.get("totalDamageDealtToChampions"),
                p.get("totalDamageTaken"),
                p.get("totalHeal"),
                p.get("longestTimeSpentLiving"),
            ),
        )

    # Wipe and reinsert timeline data for idempotency
    conn.execute("DELETE FROM timeline_events WHERE match_id = ?", (match_id,))
    conn.execute("DELETE FROM timeline_frames WHERE match_id = ?", (match_id,))
    for frame_idx, frame in enumerate(timeline["info"]["frames"]):
        ts = frame["timestamp"]
        for pid_str, pf in frame.get("participantFrames", {}).items():
            pos = pf.get("position") or {}
            conn.execute(
                """
                INSERT INTO timeline_frames (
                    match_id, frame_index, timestamp_ms, participant_id,
                    position_x, position_y, current_gold, total_gold,
                    level, xp, minions_killed, jungle_minions_killed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id, frame_idx, ts, pf["participantId"],
                    pos.get("x"), pos.get("y"),
                    pf.get("currentGold"), pf.get("totalGold"),
                    pf.get("level"), pf.get("xp"),
                    pf.get("minionsKilled"),
                    pf.get("jungleMinionsKilled"),
                ),
            )
        for ev in frame.get("events", []):
            conn.execute(
                """
                INSERT INTO timeline_events (
                    match_id, frame_index, timestamp_ms, type,
                    participant_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id, frame_idx, ev.get("timestamp", ts),
                    ev["type"],
                    ev.get("participantId") or ev.get("killerId"),
                    json.dumps(ev, separators=(",", ":")),
                ),
            )


def insert_session(
    conn: sqlite3.Connection, session_dir: Path, summary: dict, match_id: str | None, note: str | None
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO sessions (
            session_dir, start_utc, end_utc, snapshot_count,
            first_game_time_s, last_game_time_s, active_player_name,
            match_id, note, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(session_dir),
            summary.get("start_utc"),
            summary.get("end_utc"),
            summary.get("snapshots"),
            summary.get("first_game_time_s"),
            summary.get("last_game_time_s"),
            summary.get("active_player_name"),
            match_id,
            note,
            _utc_now_iso(),
        ),
    )


# ---------------------------------------------------------------- clips

# Replays.lol clip filenames look like:
#   "Replays.lol - Insane_Sona_Fight - 2.mp4"
#   "Replays.lol - Godlike_Seraphine_Fight.mp4"
#   "Replays.lol - Impressive_Sona_Triple_Kill.mp4"
_CLIP_RE = re.compile(
    r"^Replays\.lol\s*-\s*(?P<adj>\w+)_(?P<champ>\w+?)_(?P<event>\w+?)(?:\s*-\s*\d+)?\.mp4$"
)


def scan_clips(
    conn: sqlite3.Connection,
    replays_lol_dirs: Iterable[Path],
    outplayed_dir: Path | None,
) -> int:
    """Scan video directories and add new clips to the clips table.

    Match each clip to a match_id by mtime falling within
    [match.game_start_ms, match.game_end_ms + 5min] window.
    """
    matches = list(conn.execute(
        "SELECT match_id, game_start_ms, game_duration_s FROM matches"
    ).fetchall())
    earliest_match_ms = min((m["game_start_ms"] for m in matches), default=None)

    def find_match_for_mtime(mtime_ms: int) -> tuple[str, float] | None:
        """Return (match_id, in_match_start_s) if mtime falls within a known match."""
        for m in matches:
            start = m["game_start_ms"]
            end = start + (m["game_duration_s"] * 1000) + 5 * 60 * 1000
            if start - 60_000 <= mtime_ms <= end:
                return (m["match_id"], max(0.0, (mtime_ms - start) / 1000.0))
        return None

    added = 0

    # First, retry linking any clips that were inserted previously without
    # a match_id but whose mtime now falls within a known match window.
    unlinked = conn.execute(
        "SELECT path, file_mtime_ms FROM clips WHERE match_id IS NULL"
    ).fetchall()
    for row in unlinked:
        link = find_match_for_mtime(row["file_mtime_ms"])
        if link:
            conn.execute(
                "UPDATE clips SET match_id = ?, in_match_start_s = ? WHERE path = ?",
                (link[0], link[1], row["path"]),
            )

    for replays_lol_dir in replays_lol_dirs:
        if not replays_lol_dir.exists():
            continue
        # Full match recordings: directly in replays_lol_dir
        for f in replays_lol_dir.glob("*.mp4"):
            added += _maybe_insert_clip(
                conn, f, "replays_lol_full", find_match_for_mtime, earliest_match_ms
            )
        # Clips: in replays_lol_dir/Clips/
        clips_dir = replays_lol_dir / "Clips"
        if clips_dir.exists():
            for f in clips_dir.glob("*.mp4"):
                added += _maybe_insert_clip(
                    conn, f, "replays_lol_clip", find_match_for_mtime, earliest_match_ms
                )

    if outplayed_dir and outplayed_dir.exists():
        # Outplayed has per-match folders, each with per-play .mp4 files
        for match_folder in outplayed_dir.iterdir():
            if not match_folder.is_dir():
                continue
            for f in match_folder.glob("*.mp4"):
                added += _maybe_insert_clip(
                    conn, f, "outplayed_clip", find_match_for_mtime, earliest_match_ms
                )

    return added


def _maybe_insert_clip(
    conn: sqlite3.Connection,
    path: Path,
    source: str,
    find_match_for_mtime,
    earliest_match_ms: int | None = None,
) -> int:
    existing = conn.execute(
        "SELECT 1 FROM clips WHERE path = ?", (str(path),)
    ).fetchone()
    if existing:
        return 0

    stat = path.stat()
    mtime_ms = int(stat.st_mtime * 1000)
    size = stat.st_size

    champion_hint = None
    event_hint = None
    m = _CLIP_RE.match(path.name)
    if m:
        champion_hint = m.group("champ")
        event_hint = m.group("event")

    match_link = find_match_for_mtime(mtime_ms)
    # Pre-history: a clip older than any ingested match can never link (those
    # games predate tracking). Skip it instead of inserting an orphan that gets
    # re-scanned every sync. (#3 clips huerfanos)
    if (match_link is None and earliest_match_ms is not None
            and mtime_ms < earliest_match_ms - 60_000):
        return 0
    match_id, in_match_start_s = (match_link or (None, None))

    conn.execute(
        """
        INSERT INTO clips (
            path, source, file_mtime_ms, file_size, duration_s,
            match_id, in_match_start_s, champion_hint, event_hint,
            discovered_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(path), source, mtime_ms, size, None,
            match_id, in_match_start_s, champion_hint, event_hint,
            _utc_now_iso(),
        ),
    )
    return 1


# ---------------------------------------------------------------- main entry

def run_ingest(
    conn: sqlite3.Connection,
    api: RiotAPI,
    raw_dir: Path,
    accounts: AccountRegistry,
    replays_lol_dirs: Iterable[Path],
    outplayed_dir: Path | None,
) -> dict:
    """Run a full ingest pass. Returns a stats dict."""
    started = _utc_now_iso()
    sessions = discover_sessions(raw_dir)
    stats = {
        "sessions_seen": len(sessions),
        "sessions_new": 0,
        "matches_added": 0,
        "clips_added": 0,
        "errors": [],
    }

    for session in sessions:
        if already_ingested(conn, session):
            continue
        stats["sessions_new"] += 1

        summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
        active_name = summary.get("active_player_name") or ""
        puuid = accounts.puuid_for(active_name)
        log.info(
            "Processing session %s (start=%s, account=%s)",
            session.name, summary.get("start_utc"), active_name or "?",
        )

        if puuid is None:
            note = (
                f"unknown_account: '{active_name}' is not in config. "
                f"Add a [[riot.accounts]] block for it to enable analysis."
            )
            log.warning(note)
            insert_session(conn, session, summary, None, note)
            conn.commit()
            continue

        match_id = None
        note = None
        try:
            match_id = find_match_for_session(
                api,
                puuid,
                summary["start_utc"],
                summary.get("first_game_time_s"),
            )
        except RiotAPIError as e:
            log.warning("Riot API error finding match for %s: %s", session.name, e)
            stats["errors"].append(f"{session.name}: find_match failed: {e}")
            note = f"riot_api_error: {e}"

        if match_id is None and note is None:
            note = "no_match_in_history (likely custom/practice tool)"
            log.info("No Riot match found for session %s", session.name)

        # Already tombstoned (normal/remake): don't even download it again.
        if match_id and is_skipped(conn, match_id):
            note = "skipped: tombstoned"
            match_id = None

        # Download and cache match + timeline
        if match_id:
            match_json_path = session / "match.json"
            timeline_json_path = session / "timeline.json"
            try:
                if not match_json_path.exists():
                    match = api.match(match_id)
                    match_json_path.write_text(
                        json.dumps(match, indent=2), encoding="utf-8"
                    )
                else:
                    match = json.loads(match_json_path.read_text(encoding="utf-8"))

                if not timeline_json_path.exists():
                    timeline = api.match_timeline(match_id)
                    timeline_json_path.write_text(
                        json.dumps(timeline, indent=2), encoding="utf-8"
                    )
                else:
                    timeline = json.loads(timeline_json_path.read_text(encoding="utf-8"))

                reason = coachable_reason(match)
                if reason:
                    mark_skipped(conn, match_id, reason)
                    log.info("Skipping %s (%s) for session %s", match_id, reason, session.name)
                    note = f"skipped: {reason}"
                    match_id = None
                else:
                    cohort = accounts.cohort_for(active_name)
                    insert_match(
                        conn, match, timeline, puuid, active_name, cohort, session,
                        match_json_path, timeline_json_path,
                    )
                    stats["matches_added"] += 1
                    log.info(
                        "Linked session %s -> match %s (account %s, cohort %s)",
                        session.name, match_id, active_name, cohort or "?",
                    )
            except (RiotAPIError, KeyError, ValueError) as e:
                log.exception("Failed to ingest match %s", match_id)
                stats["errors"].append(f"{session.name} match={match_id}: {e}")
                match_id = None
                note = f"ingest_failed: {e}"

        insert_session(conn, session, summary, match_id, note)
        conn.commit()

    # Scan clips after all matches are persisted
    stats["clips_added"] = scan_clips(conn, replays_lol_dirs, outplayed_dir)

    # Record the run
    conn.execute(
        """
        INSERT INTO ingest_runs (
            started_at_utc, finished_at_utc, sessions_seen, sessions_new,
            matches_added, clips_added, errors_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started, _utc_now_iso(),
            stats["sessions_seen"], stats["sessions_new"],
            stats["matches_added"], stats["clips_added"],
            json.dumps(stats["errors"]) if stats["errors"] else None,
        ),
    )
    conn.commit()
    return stats
