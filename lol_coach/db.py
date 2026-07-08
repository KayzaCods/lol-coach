"""SQLite schema + connection for the ingested match data.

The .jsonl snapshots and the cached match/timeline JSONs from Riot stay
as files on disk (source of truth). SQLite mirrors only the fields we
actually query during analysis: participants, timeline events, frames,
clips, sessions.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    platform TEXT,
    game_creation_ms INTEGER NOT NULL,
    game_start_ms INTEGER NOT NULL,
    game_duration_s INTEGER NOT NULL,
    queue_id INTEGER,
    queue_name TEXT,
    game_mode TEXT,
    game_type TEXT,
    map_id INTEGER,
    game_version TEXT,
    our_puuid TEXT,
    our_account_riot_id TEXT,
    our_cohort TEXT,
    session_dir TEXT,
    match_json_path TEXT,
    timeline_json_path TEXT,
    ingested_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_cohort ON matches(our_cohort);

CREATE TABLE IF NOT EXISTS participants (
    match_id TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    puuid TEXT NOT NULL,
    riot_id_game_name TEXT,
    riot_id_tag_line TEXT,
    team_id INTEGER NOT NULL,
    team_position TEXT,
    champion_name TEXT NOT NULL,
    champion_id INTEGER,
    win INTEGER NOT NULL,
    kills INTEGER, deaths INTEGER, assists INTEGER,
    total_minions_killed INTEGER,
    neutral_minions_killed INTEGER,
    vision_score INTEGER,
    wards_placed INTEGER,
    wards_killed INTEGER,
    gold_earned INTEGER,
    gold_spent INTEGER,
    total_damage_dealt_to_champions INTEGER,
    total_damage_taken INTEGER,
    total_heal INTEGER,
    longest_time_spent_living INTEGER,
    PRIMARY KEY (match_id, participant_id)
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    type TEXT NOT NULL,
    participant_id INTEGER,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_timeline_events_match_time
    ON timeline_events(match_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_timeline_events_match_type
    ON timeline_events(match_id, type);

CREATE TABLE IF NOT EXISTS timeline_frames (
    match_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    participant_id INTEGER NOT NULL,
    position_x INTEGER,
    position_y INTEGER,
    current_gold INTEGER,
    total_gold INTEGER,
    level INTEGER,
    xp INTEGER,
    minions_killed INTEGER,
    jungle_minions_killed INTEGER,
    PRIMARY KEY (match_id, frame_index, participant_id)
);
-- The PK is (match_id, frame_index, participant_id), but the hot access pattern is
-- "latest frame for a (match, participant) at/before T" (build_world_state, _cs_at,
-- _frame_pos, _player_position_at). Without this the planner seeks only on match_id
-- and sorts in a temp b-tree; with it the lookup is a direct index seek (~5.7x).
CREATE INDEX IF NOT EXISTS idx_timeline_frames_lookup
    ON timeline_frames(match_id, participant_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS sessions (
    session_dir TEXT PRIMARY KEY,
    start_utc TEXT NOT NULL,
    end_utc TEXT,
    snapshot_count INTEGER,
    first_game_time_s REAL,
    last_game_time_s REAL,
    active_player_name TEXT,
    match_id TEXT,
    note TEXT,
    ingested_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    path TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    file_mtime_ms INTEGER,
    file_size INTEGER,
    duration_s REAL,
    match_id TEXT,
    in_match_start_s REAL,
    champion_hint TEXT,
    event_hint TEXT,
    discovered_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    detector_id TEXT NOT NULL,
    game_time_ms INTEGER NOT NULL,
    moment TEXT,
    outcome TEXT,
    context_json TEXT NOT NULL,
    options_json TEXT NOT NULL,
    argument TEXT NOT NULL,
    clip_path TEXT,
    clip_offset_s REAL,
    user_feedback TEXT,
    user_feedback_note TEXT,
    user_feedback_at_utc TEXT,
    user_best_option INTEGER,
    user_mark_type TEXT,
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_match
    ON decisions(match_id);
CREATE INDEX IF NOT EXISTS idx_decisions_detector
    ON decisions(detector_id);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    sessions_seen INTEGER DEFAULT 0,
    sessions_new INTEGER DEFAULT 0,
    matches_added INTEGER DEFAULT 0,
    clips_added INTEGER DEFAULT 0,
    errors_json TEXT
);

-- Matches deliberately NOT ingested (normals, remakes, special queues). Tombstones
-- so ingest never re-pulls them from Riot: Ascent recordings of these games would
-- otherwise retry the pull on every 30-minute run forever.
CREATE TABLE IF NOT EXISTS skipped_matches (
    match_id TEXT PRIMARY KEY,
    reason TEXT,
    at_utc TEXT
);

-- Player input events, derived from Ascent's per-recording events.csv.gz.
-- We store significant, derived events (clicks with position, key presses with
-- hold duration, camera zoom, and hover_dwells) rather than the ~200k raw
-- mouse_delta/cursor_pos rows. game_time_ms is synced to the match
-- (Ascent t_ms, with in_match_start_s = 0). Schema follows BEHAVIORAL_PIPELINE.md.
CREATE TABLE IF NOT EXISTS input_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    session_dir TEXT,
    game_time_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,      -- 'click' | 'key' | 'wheel' | 'hover_dwell'
    button TEXT,                   -- 'left' | 'right' | NULL
    key TEXT,                      -- 'Q','W','E','R','D','F','B','Tab','sc_NN' | NULL
    screen_x INTEGER,
    screen_y INTEGER,
    duration_ms INTEGER,           -- key hold / dwell duration | NULL
    classified_intent TEXT,        -- 'move'|'attack'|'spell_cast'|'recall'|'map_command'|'camera'|'hesitation'|NULL
    classified_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_input_events_match_time
    ON input_events(match_id, game_time_ms);
CREATE INDEX IF NOT EXISTS idx_input_events_type
    ON input_events(match_id, event_type);

-- Pairwise preferences (preference learning): the user judges which of two
-- comparable decisions was played better. Used to learn the HEF weights ω so
-- that HEF(winner) > HEF(loser). winner in {'a','b','tie'}.
-- a_key / b_key are STABLE decision keys ("match_id|detector_id|game_time_ms")
-- that survive re-analysis (decision ids change on DELETE+reinsert; these do
-- not). Readers resolve the current decision id from the key, so feedback is
-- never orphaned by re-running the detectors.
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detector_id TEXT,
    decision_a_id INTEGER NOT NULL,
    decision_b_id INTEGER NOT NULL,
    winner TEXT NOT NULL,
    note TEXT,
    created_at_utc TEXT NOT NULL,
    a_key TEXT,
    b_key TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column additions for DBs created before a column existed."""
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(preferences)")}
    for col in ("a_key", "b_key"):
        if col not in pcols:
            conn.execute(f"ALTER TABLE preferences ADD COLUMN {col} TEXT")
    dcols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    if "user_best_option" not in dcols:
        conn.execute("ALTER TABLE decisions ADD COLUMN user_best_option INTEGER")
    if "user_feedback_fingerprint" not in dcols:
        # Identity of the option set the user was looking at when they marked
        # user_best_option (features.action_fingerprint). Lets the learner
        # validate a mark by CONTENT (did the options actually change?) instead
        # of by timestamp — the timestamp gate went stale wholesale on every
        # full re-analysis (2026-06-10: 192/192 marks dead).
        conn.execute("ALTER TABLE decisions ADD COLUMN user_feedback_fingerprint TEXT")
    if "user_mark_type" not in dcols:
        # Por qué el usuario marcó (#12): 'decision' entrena al learner; los otros
        # 4 tipos (execution/mixed/missing_context/wrong_moment) no.
        conn.execute("ALTER TABLE decisions ADD COLUMN user_mark_type TEXT")
    mcols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
    if "analyzed_at_utc" not in mcols:
        # Marker set after the FULL per-match chain (detectors -> reward -> clips).
        # NULL = needs (re)analysis. Replaces the old "has no decisions" heuristic,
        # which both reprocessed 0-decision matches forever and never repaired
        # matches left half-done by a crash between steps.
        conn.execute("ALTER TABLE matches ADD COLUMN analyzed_at_utc TEXT")
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn
