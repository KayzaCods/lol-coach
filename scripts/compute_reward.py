"""Offline reward computation for each decision.

For every row in `decisions`, computes a forward-looking observable reward over
a fixed window (REWARD_WINDOW_S) starting at the decision's game_time_ms, and
writes a `reward` block into context_json. Does NOT change the SQLite schema —
only the JSON content of decisions.context_json.

Run order matters: run AFTER scripts/analyze.py. analyze.py rewrites
context_json (DELETE + reinsert), which wipes any prior reward block, so reward
must be (re)computed after analysis. This script is idempotent — re-running
overwrites the reward block in place.

GOLD RULE (ARCHITECTURE_OPTIMIZED rule #7): gold deltas use current_gold
(spendable). Note this means purchases inside the window subtract from the
delta; that's accepted per the rule (we score spendable-gold swing, not income).

Usage:
    .venv\\Scripts\\python.exe scripts\\compute_reward.py             # all matches
    .venv\\Scripts\\python.exe scripts\\compute_reward.py LA1_1719501835   # one match
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.config import load_config
from lol_coach.context import build_world_state, distance
from lol_coach.db import connect
from lol_coach.decisions.features import LOCAL_RADIUS_UNITS, nearest_ally_tower_dist

REWARD_WINDOW_S = 45
REWARD_WINDOW_MS = REWARD_WINDOW_S * 1000

# Reward weights (Phase 1; calibrated with feedback later).
W_KILL = 1.0          # +1 per enemy our team kills
W_OWN_DEATH = -1.0    # -1 per your own death
W_ALLY_DEATH = -0.5   # -0.5 per ally death
W_OBJECTIVE = 3.0     # +/-3 per major neutral objective
W_GOLD_PER_100 = 0.1  # +0.1 per 100 net spendable gold
W_WAVE_LOSS = -0.5    # -0.5 if we lost a big wave under tower

WAVE_LOSS_TOWER_UNITS = 1800  # died this close to an ally tower => likely a defended wave loss
MAJOR_MONSTERS = {"DRAGON", "RIFTHERALD", "BARON_NASHOR", "ATAKHAN"}


def _frame_val(conn, match_id, pid, ts_ms, col):
    """Value of `col` from the closest frame at/before ts_ms (fallback: after)."""
    row = conn.execute(
        f"SELECT {col} AS v FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, pid, ts_ms),
    ).fetchone()
    if row is None or row["v"] is None:
        row = conn.execute(
            f"SELECT {col} AS v FROM timeline_frames "
            "WHERE match_id = ? AND participant_id = ? AND timestamp_ms > ? "
            "ORDER BY timestamp_ms ASC LIMIT 1",
            (match_id, pid, ts_ms),
        ).fetchone()
    return row["v"] if row and row["v"] is not None else None


def _match_meta(conn, match_id):
    m = conn.execute("SELECT our_puuid FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    if not m:
        raise ValueError("match not in DB")
    us = conn.execute(
        "SELECT participant_id, team_id FROM participants WHERE match_id = ? AND puuid = ?",
        (match_id, m["our_puuid"]),
    ).fetchone()
    if not us:
        raise ValueError("our participant row missing")
    team_of = {
        r["participant_id"]: r["team_id"]
        for r in conn.execute(
            "SELECT participant_id, team_id FROM participants WHERE match_id = ?", (match_id,)
        )
    }
    return us["participant_id"], us["team_id"], team_of


def compute_reward(conn, match_id, decision_game_time_ms, us_id, our_team, team_of) -> dict:
    t0 = decision_game_time_ms
    t1 = t0 + REWARD_WINDOW_MS

    # Kills in the window.
    kills_for = kills_against = own_death = ally_death = 0
    for r in conn.execute(
        "SELECT payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'CHAMPION_KILL' AND timestamp_ms BETWEEN ? AND ?",
        (match_id, t0, t1),
    ):
        ev = json.loads(r["payload_json"])
        if team_of.get(ev.get("killerId")) == our_team:
            kills_for += 1
        if team_of.get(ev.get("victimId")) == our_team:
            kills_against += 1
            if ev.get("victimId") == us_id:
                own_death += 1
            else:
                ally_death += 1

    # Major neutral objectives in the window.
    objective_delta = 0
    for r in conn.execute(
        "SELECT payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'ELITE_MONSTER_KILL' AND timestamp_ms BETWEEN ? AND ?",
        (match_id, t0, t1),
    ):
        ev = json.loads(r["payload_json"])
        if ev.get("monsterType") not in MAJOR_MONSTERS:
            continue
        kteam = team_of.get(ev.get("killerId"), ev.get("killerTeamId"))
        if kteam == our_team:
            objective_delta += 1
        elif kteam is not None:
            objective_delta -= 1

    # Local gold/xp swing (current_gold). "Local" = enemies within radius at t0.
    ws = build_world_state(conn, match_id, t0)
    us = ws.players.get(us_id)
    local_enemy_ids = []
    if us is not None and us.pos_x is not None:
        for p in ws.players.values():
            if p.team_id != our_team and p.pos_x is not None:
                d = distance(us, p)
                if d is not None and d <= LOCAL_RADIUS_UNITS:
                    local_enemy_ids.append(p.participant_id)

    def delta(pid, col):
        v0 = _frame_val(conn, match_id, pid, t0, col)
        v1 = _frame_val(conn, match_id, pid, t1, col)
        return (v1 - v0) if (v0 is not None and v1 is not None) else None

    our_gold = delta(us_id, "current_gold")
    our_xp = delta(us_id, "xp")
    en_gold = [d for pid in local_enemy_ids if (d := delta(pid, "current_gold")) is not None]
    en_xp = [d for pid in local_enemy_ids if (d := delta(pid, "xp")) is not None]
    mean_en_gold = sum(en_gold) / len(en_gold) if en_gold else 0.0
    mean_en_xp = sum(en_xp) / len(en_xp) if en_xp else 0.0
    gold_delta_local = round(our_gold - mean_en_gold, 1) if our_gold is not None else 0.0
    xp_delta_local = round(our_xp - mean_en_xp, 1) if our_xp is not None else 0.0

    # wave_loss proxy: you died in the window while near an ally tower (defending).
    wave_loss = 0
    if own_death >= 1 and us is not None:
        td = nearest_ally_tower_dist(us.pos_x, us.pos_y, our_team)
        if td is not None and td <= WAVE_LOSS_TOWER_UNITS:
            wave_loss = 1

    reward_score = round(
        kills_for * W_KILL
        + own_death * W_OWN_DEATH
        + ally_death * W_ALLY_DEATH
        + objective_delta * W_OBJECTIVE
        + (gold_delta_local / 100.0) * W_GOLD_PER_100
        + wave_loss * W_WAVE_LOSS,
        2,
    )

    return {
        "window_s": REWARD_WINDOW_S,
        "gold_delta_local": gold_delta_local,
        "xp_delta_local": xp_delta_local,
        "kills_for": kills_for,
        "kills_against": kills_against,
        "objective_delta": objective_delta,
        "wave_loss": wave_loss,
        "reward_score": reward_score,
    }


def run_for_match(conn, match_id) -> int:
    """(Re)compute the reward block for every decision of one match. Returns count."""
    try:
        us_id, our_team, team_of = _match_meta(conn, match_id)
    except ValueError:
        return 0
    rows = conn.execute(
        "SELECT id, game_time_ms, context_json FROM decisions WHERE match_id = ?",
        (match_id,),
    ).fetchall()
    for row in rows:
        reward = compute_reward(conn, match_id, row["game_time_ms"], us_id, our_team, team_of)
        ctx = json.loads(row["context_json"])
        ctx["reward"] = reward
        conn.execute(
            "UPDATE decisions SET context_json = ? WHERE id = ?",
            (json.dumps(ctx, ensure_ascii=False, indent=2), row["id"]),
        )
    conn.commit()
    return len(rows)


def main() -> int:
    cfg = load_config()
    conn = connect(cfg["paths"]["sqlite_db"])

    if len(sys.argv) > 1:
        match_ids = [sys.argv[1]]
    else:
        match_ids = [r["match_id"] for r in conn.execute("SELECT match_id FROM matches")]

    total = 0
    for match_id in match_ids:
        try:
            us_id, our_team, team_of = _match_meta(conn, match_id)
        except ValueError as e:
            print(f"  skip {match_id}: {e}")
            continue
        rows = conn.execute(
            "SELECT id, game_time_ms, context_json FROM decisions WHERE match_id = ?",
            (match_id,),
        ).fetchall()
        for row in rows:
            reward = compute_reward(conn, match_id, row["game_time_ms"], us_id, our_team, team_of)
            ctx = json.loads(row["context_json"])
            ctx["reward"] = reward
            conn.execute(
                "UPDATE decisions SET context_json = ? WHERE id = ?",
                (json.dumps(ctx, ensure_ascii=False, indent=2), row["id"]),
            )
            total += 1
        conn.commit()
        if rows:
            print(f"  {match_id}: {len(rows)} decisions")
    print(f"Done. reward block written to {total} decision(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
