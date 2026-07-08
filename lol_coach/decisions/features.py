"""Shared builders for the standardized Decision context blocks.

Every detector emits a consistent schema inside `context`:

  context["state_features"] = {power, info_risk, wave_tempo, objectives}
  context["action"]         = {detector_id, action_id, available_actions}

(The third block, `context["reward"]`, is computed offline in
scripts/compute_reward.py with a forward-looking window — it cannot be known
at detection time.)

GOLD RULE (ARCHITECTURE_OPTIMIZED rule #7): any gold that represents the
player's state/power AT the decision moment is current_gold (spendable),
never total_gold (lifetime accumulated). total_gold is kept only as a
historical stat where useful.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Optional

from ..context import UNSEEN_DANGEROUS_S, WorldState, distance, last_event_appearance, map_lane

# --------------------------------------------------------------------- config
LOCAL_RADIUS_UNITS = 2500        # "local" skirmish radius for power/proximity
WARD_NEARBY_UNITS = 2000         # our wards counted as "near us"
WARD_LOOKBACK_MS = 120_000       # how far back to look for nearby wards
CS_WINDOW_MS = 180_000           # 3-minute window for cs_last_3min_diff

# Marker identifying the option the player actually took. Same convention the
# detectors use in their option labels; re-verified 2026-06-10: matches exactly
# one option in every decision across all active detectors.
TAKEN_MARKER = re.compile(r"hiciste|estado real|lo que hiciste|permaneciste", re.I)


def mmss(ms: int) -> str:
    """mm:ss from a millisecond game time. Each detector used to carry a private
    `_mmss` copy of this; now there is one home. Tolerates None (-> 0:00)."""
    s = int(ms or 0) // 1000
    return f"{s // 60}:{s % 60:02d}"

# ----------------------------------------------------- standard action vocab
ACTION_FIGHT_ENTRY = "fight_entry"        # entrar a la pelea / commit
ACTION_DISENGAGE = "disengage"            # retroceder / no pelear
ACTION_PUSH_WAVE = "push_wave"            # pushear la oleada
ACTION_HOLD_WAVE = "hold_wave"            # mantener / freeze
ACTION_ROAM = "roam"                      # rotar a otra linea / al fight
ACTION_FARM_JUNGLE = "farm_jungle"        # seguir farmeando jungla
ACTION_RESET = "reset"                    # volver a base
ACTION_VISION_SETUP = "vision_setup"      # preparar vision para objetivo
ACTION_IGNORE_OBJECTIVE = "ignore_objective"  # no moverse al objetivo
ACTION_CHECK_MAP = "check_map"            # revisar minimap / camara antes de comprometer

ACTION_VOCAB = {
    ACTION_FIGHT_ENTRY, ACTION_DISENGAGE, ACTION_PUSH_WAVE, ACTION_HOLD_WAVE,
    ACTION_ROAM, ACTION_FARM_JUNGLE, ACTION_RESET, ACTION_VISION_SETUP,
    ACTION_IGNORE_OBJECTIVE, ACTION_CHECK_MAP,
}


# ----------------------------------------------------------------- power index
# A single 0..1 "relative strength" readout from the power block, per the Live
# Feedback (CHI PLAY'22) recommendation to fight information overload by folding
# several metrics into one. 0.5 = even, >0.5 = our advantage, <0.5 = disadvantage.
# Each present sub-metric maps to a signed [-1, 1] advantage; absent ones are
# dropped and the weights renormalized. Gold is current_gold (rule #7).
def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Component weights sum to 1.0. We divide by the FULL weight, not just the
# present components, so a MISSING component counts as neutral (0 in [-1,1]) and
# the score shrinks toward 0.5 the less real signal we have. A decision with no
# recorder/economy data (only a noisy one-frame local head-count) is therefore
# NOT scored as a blowout advantage the way `head-count alone` used to give 1.0.
_POWER_W = {"level": 0.20, "gold": 0.25, "hp": 0.20, "numbers": 0.35}


def power_index(power: Optional[dict]) -> Optional[float]:
    if not power:
        return None
    s = 0.0          # Σ value·weight over present components; missing => 0 (neutral)
    comps = 0
    ld = power.get("local_level_diff")
    if isinstance(ld, (int, float)):
        s += _clamp(ld / 3.0) * _POWER_W["level"]; comps += 1      # ~3 levels = decisive
    gd = power.get("local_gold_diff")
    if isinstance(gd, (int, float)):
        s += _clamp(gd / 1500.0) * _POWER_W["gold"]; comps += 1    # ~1.5k spendable = decisive
    hp = power.get("entry_hp_pct")
    if isinstance(hp, (int, float)):
        s += _clamp(hp * 2 - 1) * _POWER_W["hp"]; comps += 1       # 0%->-1, 50%->0, 100%->+1
    an, en = power.get("ally_count_nearby"), power.get("enemy_count_nearby")
    if isinstance(an, (int, float)) and isinstance(en, (int, float)):
        s += _clamp((an - en) / 2.0) * _POWER_W["numbers"]; comps += 1   # local numbers
    if comps == 0:
        return None
    return round((s + 1) / 2, 2)                          # [-1,1] -> [0,1]


def power_label(idx: Optional[float]) -> Optional[str]:
    if idx is None:
        return None
    if idx < 0.30:
        return "desventaja fuerte"
    if idx < 0.45:
        return "desventaja"
    if idx <= 0.55:
        return "parejo"
    if idx <= 0.70:
        return "ventaja"
    return "ventaja fuerte"


# ------------------------------------------------------- HEF (hierarchical EF)
# Composite, auditable quality score for a decision (arxiv 2508.13057 idea):
# weighted sum of normalized, complementary terms minus penalties Π.
#   F_HEF = Σ ω_i · term_i(0..1)  −  Π_severe  −  Π_progressive
# Each term is 0..1 where 1 = good for the player. Weights are hierarchical
# (power and info dominate). Penalties: severe (invalid state for the action,
# e.g. committing blind) + progressive (crossing a soft threshold, e.g. low HP).
# Returns the score AND the breakdown so the UI can show WHY.
_HEF_WEIGHTS = {"power": 0.35, "info": 0.30, "action_fit": 0.25, "wave": 0.10}
_AGGRESSIVE_ACTIONS = {"fight_entry", "push_wave", "roam"}
_DEFENSIVE_ACTIONS = {"disengage", "reset", "hold_wave"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _info_term(info: dict) -> Optional[float]:
    """0..1 from info_risk: 1 = safe/informed, 0 = blind/exposed."""
    parts = []
    eu = info.get("enemies_unseen")
    if isinstance(eu, (int, float)):
        parts.append(_clamp01(1 - eu / 5.0))           # 0 unseen=1, 5+=0
    jg = info.get("enemy_jg_unseen_s")
    if isinstance(jg, (int, float)):
        parts.append(_clamp01(1 - jg / 90.0))          # 90s+ unseen = 0
    w = info.get("wards_nearby")
    if isinstance(w, (int, float)):
        parts.append(_clamp01(0.4 + w * 0.3))          # 0 wards=0.4, 2+=1
    sr = info.get("safety_ratio")
    if isinstance(sr, (int, float)):
        parts.append(_clamp01(sr))                     # 1 = by our tower (safe)
    elif isinstance(info.get("distance_to_ally_tower_u"), (int, float)):
        parts.append(_clamp01(1 - info["distance_to_ally_tower_u"] / 8000.0))
    lt = info.get("local_threat")
    if isinstance(lt, (int, float)):
        parts.append(_clamp01(1 - lt))                 # high local threat lowers info safety
    return sum(parts) / len(parts) if parts else None


def _wave_term(wave: dict, at_objective: bool = False) -> Optional[float]:
    """Wave value depends on WHAT you're doing (player feedback 2026-06-10):
    - Fighting/surviving (deaths): near your tower = safety = good; shoved into
      the enemy = exposed = bad.
    - Playing an OBJECTIVE: a wave pushed into the enemy is PRESSURE (they must
      answer it, fewer contesters at the pit) = good; frozen near your own tower
      = zero map pressure while the objective spawns = bad. Exact inverse."""
    ws = wave.get("wave_state")
    if at_objective:
        return {"pushed_into_enemy": 0.80, "neutral": 0.50,
                "frozen_near_tower": 0.30, "under_tower": 0.25}.get(ws)
    return {"pushed_into_enemy": 0.30, "neutral": 0.5,
            "frozen_near_tower": 0.75, "under_tower": 0.8}.get(ws)


def _commit_fit(action_id, pi, it, objectives=None):
    """How well the action fits the SETUP — the player's real criterion.

    An aggressive commit needs setup (vision + support, not over-exposed); a
    defensive/ceding action is right when there is NO setup. The objective's
    value scales how much an aggressive commit toward it is worth, and makes
    ceding a contestable, valuable objective worse.

    Setup blend recalibrated 2026-06-11 against the player's explicit
    best-option marks (scripts/triage_commit_fit.py): the event-based info term
    systematically OVERESTIMATES safety (the player's own notes said "no
    tenemos visión" on states the term read as 0.7-0.9), so trusting it 0.6
    made the formula contradict the player's reviewed marks in 15/16 cases of
    fight_entry vs disengage. alpha 0.6 -> 0.3 measured (with death_v1's
    pre-fight visibility fix applied): agreement with marks 0.46 -> 0.58,
    Calibrar pairwise accuracy 0.57 -> 0.60 — both sources improve.
    """
    if action_id is None:
        return None
    if it is not None and pi is not None:
        setup = 0.3 * it + 0.7 * pi          # power carries more: see docstring
    elif it is not None:
        setup = it
    elif pi is not None:
        setup = pi
    else:
        return None
    ov = (objectives or {}).get("objective_value")
    if action_id in _AGGRESSIVE_ACTIONS:
        # high-value objective rewards committing even at medium setup; low value
        # demands cleaner setup. ov 0->x0.7, 0.5->x1.0, 1->x1.3
        fit = setup * (0.7 + 0.6 * ov) if isinstance(ov, (int, float)) else setup
        return _clamp01(fit)
    if action_id == "ignore_objective":
        base = 1 - setup                      # ceding is right when setup is low...
        if isinstance(ov, (int, float)):
            base -= 0.4 * ov * setup          # ...but worse if it was valuable AND you had setup
        return _clamp01(base)
    if action_id in _DEFENSIVE_ACTIONS:
        return _clamp01(1 - setup)
    if action_id in ("check_map", "vision_setup"):
        # gathering vision/info is good, and especially good when setup is low
        # (you inform the commit instead of going in blind)
        return _clamp01(0.6 + 0.4 * (1 - setup))
    return 0.5


def hef_score(state_features: Optional[dict], action_id: Optional[str] = None,
              weights: Optional[dict] = None) -> dict:
    sf = state_features or {}
    W = weights or _HEF_WEIGHTS          # learned weights override the defaults
    power = sf.get("power") or {}
    info = sf.get("info_risk") or {}
    wave = sf.get("wave_tempo") or {}

    pi = power_index(power)
    it = _info_term(info)
    terms = {}
    if pi is not None:
        terms["power"] = pi
    if it is not None:
        terms["info"] = it
    objectives = sf.get("objectives") or {}
    # An objective decision (objective_readiness sets time_to_obj_s=0 exactly)
    # reads the wave as PRESSURE, not as safety.
    at_objective = bool(objectives.get("next_major_obj")) and objectives.get("time_to_obj_s") == 0
    wt = _wave_term(wave, at_objective)
    if wt is not None:
        terms["wave"] = wt
    af = _commit_fit(action_id, pi, it, objectives)
    if af is not None:
        terms["action_fit"] = af

    if not terms:
        return {"score": None, "base": None, "terms": {}, "weights": {}, "penalties": {}}

    wsum = sum(W[k] for k in terms)
    base = sum(terms[k] * W[k] for k in terms) / wsum

    penalties = {}
    eu = info.get("enemies_unseen")
    if action_id == "fight_entry" and isinstance(eu, (int, float)) and eu >= 2:
        penalties["pelea_a_ciegas"] = 0.30                       # severe
    hp = power.get("entry_hp_pct")
    if isinstance(hp, (int, float)) and hp < 0.45:
        penalties["hp_bajo"] = round((0.45 - hp) * 0.6, 2)       # progressive
    jg = info.get("enemy_jg_unseen_s")
    if action_id in _AGGRESSIVE_ACTIONS and isinstance(jg, (int, float)) and jg >= 60:
        penalties["jungla_sin_ver"] = 0.15                        # progressive
    # The player's dominant failure: an aggressive commit with no setup (blind /
    # exposed, low info). Strong progressive penalty — his explicit request.
    if action_id in _AGGRESSIVE_ACTIONS and isinstance(it, (int, float)) and it < 0.40:
        penalties["commit_sin_setup"] = round(0.15 + 0.25 * (0.40 - it) / 0.40, 2)  # 0.15..0.40

    pen = sum(penalties.values())
    score = _clamp01(base - pen)
    return {
        "score": round(score, 2),
        "base": round(base, 2),
        "terms": {k: round(v, 2) for k, v in terms.items()},
        "weights": {k: round(W[k], 3) for k in terms},
        "penalties": penalties,
    }


def hef_label(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score < 0.30:
        return "decisión pobre"
    if score < 0.45:
        return "cuestionable"
    if score <= 0.60:
        return "aceptable"
    if score <= 0.78:
        return "buena"
    return "excelente"


# --------------------------------------------------------------------- action
def taken_index(options) -> Optional[int]:
    """Index of the option the player actually took (via the label marker).

    Accepts Option objects or plain dicts (parsed options_json) so callers that
    work on persisted rows (dashboard, backfill) share the same logic."""
    for i, o in enumerate(options):
        label = o.get("label") if isinstance(o, dict) else getattr(o, "label", "")
        if TAKEN_MARKER.search(label or ""):
            return i
    return None


def action_fingerprint(detector_id: str, action: Optional[dict], options) -> Optional[str]:
    """Stable identity of a decision's option set.

    user_best_option is an INDEX into the options, so a mark only stays
    meaningful while the option set keeps the same semantics. This fingerprint
    captures the semantic skeleton — the ordered action_ids plus which index is
    the taken one — and deliberately ignores label prose, which embeds volatile
    numbers (HP %, setup %, champion names) that change between re-analyses
    without changing what the options MEAN.

    Stored with the mark (decisions.user_feedback_fingerprint) and compared
    against the current incarnation: equal fingerprint = the stored index still
    points at the same choice. Plain readable string, not a hash, so mismatches
    can be debugged by eye.
    """
    if not action:
        return None
    avail = action.get("available_actions") or []
    ids = [str(a.get("action_id") or "?") for a in avail]
    if not ids:
        return None
    ti = taken_index(options or [])
    return f"v1|{detector_id}|{','.join(ids)}|t{ti if ti is not None else '?'}"


def build_action(detector_id: str, options, action_ids: list[str]) -> dict:
    """Standard `action` block.

    `action_ids` is parallel to `options`, mapping each option to a vocabulary
    action_id. The taken action is the action_id of the option flagged as taken.
    """
    ti = taken_index(options)
    available = [
        {"action_id": aid, "ev_model": round(float(o.ev_score), 2)}
        for o, aid in zip(options, action_ids)
    ]
    taken_action = action_ids[ti] if (ti is not None and ti < len(action_ids)) else None
    return {
        "detector_id": detector_id,
        "action_id": taken_action,
        "available_actions": available,
    }


# Standard SR turret coordinates (approx, public dataset) per team_id. Used
# only as a "distance to safety" proxy — static map data, not champion logic.
TOWERS: dict[int, list[tuple[int, int]]] = {
    100: [  # blue side (bottom-left base)
        (981, 10441), (1512, 6699), (1169, 4287),       # top: outer, inner, inhib
        (5846, 6396), (5048, 4812), (3651, 3696),       # mid: outer, inner, inhib
        (10504, 1029), (6919, 1483), (4281, 1253),      # bot: outer, inner, inhib
        (1748, 2270), (2177, 1807),                     # nexus turrets
    ],
    200: [  # red side (top-right base)
        (4318, 13875), (7943, 13411), (10481, 13650),   # top
        (8955, 8510), (9767, 10113), (11134, 11207),    # mid
        (13866, 4505), (13327, 8226), (13624, 10572),   # bot
        (13052, 12612), (12611, 13084),                 # nexus turrets
    ],
}

# Riot monsterType -> short objective vocab. ATAKHAN/HORDE -> None (out of vocab).
_OBJ_KIND = {"DRAGON": "dragon", "RIFTHERALD": "herald", "BARON_NASHOR": "baron"}


def obj_kind(monster_type: Optional[str]) -> Optional[str]:
    return _OBJ_KIND.get(monster_type or "")


# --------------------------------------------------------------------- helpers
def _enemies(ws: WorldState, our_team: int):
    return [p for p in ws.players.values() if p.team_id != our_team]


def _allies(ws: WorldState, our_team: int, us_id: int):
    return [p for p in ws.players.values() if p.team_id == our_team and p.participant_id != us_id]


def nearby_counts(ws: WorldState, us_id: int, our_team: int, radius: int = LOCAL_RADIUS_UNITS):
    us = ws.players[us_id]
    allies = sum(
        1 for p in _allies(ws, our_team, us_id)
        if (d := distance(us, p)) is not None and d <= radius
    )
    enemies = sum(
        1 for p in _enemies(ws, our_team)
        if (d := distance(us, p)) is not None and d <= radius
    )
    return allies, enemies


def local_power(ws: WorldState, us_id: int, our_team: int, radius: int = LOCAL_RADIUS_UNITS):
    """(local_level_diff, local_gold_diff) vs the mean of nearby enemies.

    Gold uses current_gold (spendable). Returns (None, None) if no local enemy.
    """
    us = ws.players[us_id]
    local_en = [p for p in _enemies(ws, our_team) if (distance(us, p) or 1e9) <= radius]
    level_diff = gold_diff = None
    if local_en and us.level is not None:
        lv = [p.level for p in local_en if p.level is not None]
        if lv:
            level_diff = round(us.level - sum(lv) / len(lv), 2)
    if local_en and us.current_gold is not None:
        cg = [p.current_gold for p in local_en if p.current_gold is not None]
        if cg:
            gold_diff = int(us.current_gold - sum(cg) / len(cg))
    return level_diff, gold_diff


def enemy_jungler_id(conn: sqlite3.Connection, match_id: str, our_team: int) -> Optional[int]:
    row = conn.execute(
        "SELECT participant_id FROM participants "
        "WHERE match_id = ? AND team_id <> ? AND team_position = 'JUNGLE' LIMIT 1",
        (match_id, our_team),
    ).fetchone()
    return row["participant_id"] if row else None


def enemy_jg_unseen_s(conn: sqlite3.Connection, match_id: str, our_team: int, at_ms: int) -> Optional[float]:
    jid = enemy_jungler_id(conn, match_id, our_team)
    if jid is None:
        return None
    last = last_event_appearance(conn, match_id, jid, at_ms)
    unseen = (at_ms - last) / 1000.0 if last else at_ms / 1000.0
    return round(unseen, 0)


THREAT_RADIUS_UNITS = 5500  # an unseen enemy within this of the decision point can
                            # plausibly rotate/flank; a farther laner sitting in its
                            # own lane cannot, so it is not a local threat.


def unseen_threats(
    conn: sqlite3.Connection, match_id: str, our_team: int, at_ms: int,
    us_x=None, us_y=None, threshold_s: float = UNSEEN_DANGEROUS_S, exclude_ids=(),
) -> list[dict]:
    """Enemies unseen >= threshold_s that are a LOCAL threat to a decision at
    (us_x, us_y). The enemy jungler always counts (it roams/ganks); any other
    enemy counts only if its last known position is within THREAT_RADIUS_UNITS
    (or we have no position to localize it). Without (us_x, us_y) every unseen
    enemy is returned (no localization possible). Returns dicts with
    {pid, champion, role, unseen_s, last_dist_u}."""
    jid = enemy_jungler_id(conn, match_id, our_team)
    out = []
    for p in conn.execute(
        "SELECT participant_id, champion_name, team_position FROM participants "
        "WHERE match_id = ? AND team_id <> ?", (match_id, our_team),
    ):
        pid = p["participant_id"]
        if pid in exclude_ids:
            continue
        last = last_event_appearance(conn, match_id, pid, at_ms)
        unseen = (at_ms - last) / 1000.0 if last else at_ms / 1000.0
        if unseen < threshold_s:
            continue
        last_dist = None
        if us_x is not None:
            fp = _frame_pos(conn, match_id, pid, at_ms)
            if fp:
                last_dist = math.hypot(fp[0] - us_x, fp[1] - us_y)
        if pid != jid and last_dist is not None and last_dist > THREAT_RADIUS_UNITS:
            continue  # far laner in its own lane: not a local threat
        out.append({
            "pid": pid, "champion": p["champion_name"], "role": p["team_position"],
            "unseen_s": round(unseen, 0),
            "last_dist_u": int(last_dist) if last_dist is not None else None,
        })
    return out


def enemies_unseen_count(
    conn: sqlite3.Connection, match_id: str, our_team: int, at_ms: int,
    threshold_s: float = UNSEEN_DANGEROUS_S, us_x=None, us_y=None,
) -> int:
    """Count enemies unseen >= threshold. With a decision position, counts only
    LOCAL threats (jungler + nearby); without one, the global count."""
    if us_x is not None:
        return len(unseen_threats(conn, match_id, our_team, at_ms, us_x, us_y, threshold_s))
    cnt = 0
    for p in conn.execute(
        "SELECT participant_id FROM participants WHERE match_id = ? AND team_id <> ?",
        (match_id, our_team),
    ):
        last = last_event_appearance(conn, match_id, p["participant_id"], at_ms)
        unseen = (at_ms - last) / 1000.0 if last else at_ms / 1000.0
        if unseen >= threshold_s:
            cnt += 1
    return cnt


def _frame_pos(conn: sqlite3.Connection, match_id: str, pid: int, ts_ms: int):
    row = conn.execute(
        "SELECT position_x, position_y FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, pid, ts_ms),
    ).fetchone()
    if row and row["position_x"] is not None:
        return (row["position_x"], row["position_y"])
    return None


def our_wards_nearby(
    conn: sqlite3.Connection, match_id: str, our_team: int,
    us_x: Optional[int], us_y: Optional[int], at_ms: int,
) -> Optional[int]:
    """Count our team's wards placed near us in the last WARD_LOOKBACK_MS.

    Ward positions are approximated from the placer's closest frame (Riot's
    WARD_PLACED carries no coords). wardType UNDEFINED is filtered (rule #6).
    """
    if us_x is None:
        return None
    team_of = {
        r["participant_id"]: r["team_id"]
        for r in conn.execute(
            "SELECT participant_id, team_id FROM participants WHERE match_id = ?", (match_id,)
        )
    }
    rows = conn.execute(
        "SELECT timestamp_ms, payload_json FROM timeline_events "
        "WHERE match_id = ? AND type = 'WARD_PLACED' AND timestamp_ms BETWEEN ? AND ?",
        (match_id, at_ms - WARD_LOOKBACK_MS, at_ms),
    ).fetchall()
    cnt = 0
    for r in rows:
        ev = json.loads(r["payload_json"])
        if ev.get("wardType") == "UNDEFINED":
            continue
        cid = ev.get("creatorId")
        if not cid or team_of.get(cid) != our_team:
            continue
        pos = _frame_pos(conn, match_id, cid, r["timestamp_ms"])
        if pos and math.hypot(pos[0] - us_x, pos[1] - us_y) <= WARD_NEARBY_UNITS:
            cnt += 1
    return cnt


def nearest_ally_tower_dist(x: Optional[int], y: Optional[int], our_team: int) -> Optional[int]:
    if x is None or y is None:
        return None
    towers = TOWERS.get(our_team)
    if not towers:
        return None
    return int(min(math.hypot(x - tx, y - ty) for tx, ty in towers))


def nearest_enemy_tower_dist(x: Optional[int], y: Optional[int], our_team: int) -> Optional[int]:
    if x is None or y is None:
        return None
    towers = TOWERS.get(200 if our_team == 100 else 100)
    if not towers:
        return None
    return int(min(math.hypot(x - tx, y - ty) for tx, ty in towers))


def _cs_at(conn: sqlite3.Connection, match_id: str, pid: int, ts_ms: int) -> Optional[int]:
    row = conn.execute(
        "SELECT minions_killed, jungle_minions_killed FROM timeline_frames "
        "WHERE match_id = ? AND participant_id = ? AND timestamp_ms <= ? "
        "ORDER BY timestamp_ms DESC LIMIT 1",
        (match_id, pid, ts_ms),
    ).fetchone()
    if not row:
        return None
    return (row["minions_killed"] or 0) + (row["jungle_minions_killed"] or 0)


_LANES = ("bot lane", "mid lane", "top lane")
WAVE_ALLY_MAX_DIST = 4500   # an ally laner this close defines "your" wave


def _player_wave(conn, match_id, player, our_team, at_ms, safety=None) -> Optional[str]:
    """Wave state from a player's position+farming, or None if they aren't
    actively farming a lane this minute."""
    if player.pos_x is None or map_lane(player.pos_x, player.pos_y) not in _LANES:
        return None
    cs_now = _cs_at(conn, match_id, player.participant_id, at_ms)
    cs_prev = _cs_at(conn, match_id, player.participant_id, at_ms - 90_000)
    if cs_now is None or cs_prev is None or (cs_now - cs_prev) < 2:
        return None
    if safety is None:
        da = nearest_ally_tower_dist(player.pos_x, player.pos_y, our_team)
        de = nearest_enemy_tower_dist(player.pos_x, player.pos_y, our_team)
        safety = (de / (da + de)) if (isinstance(da, (int, float)) and isinstance(de, (int, float)) and da + de > 0) else None
    if safety is None:
        return "neutral"
    return "frozen_near_tower" if safety >= 0.62 else ("pushed_into_enemy" if safety <= 0.40 else "neutral")


def infer_wave_state(conn, match_id, ws, us_id, our_team, at_ms, safety):
    """Rough wave state (proxy by position). Returns (state, source):
      'own'  → you're laning and farming, so your position IS the wave.
      'ally' → you're support/jungle/roaming; inherit the wave of the nearest
               farming ally laner (<= WAVE_ALLY_MAX_DIST), which is the wave
               relevant to your decision.
      (None, None) → no lane wave applies."""
    us = ws.players.get(us_id)
    if us is None:
        return None, None
    own = _player_wave(conn, match_id, us, our_team, at_ms, safety)
    if own is not None:
        return own, "own"
    best, best_d = None, None
    for pid, p in ws.players.items():
        if pid == us_id or p.team_id != our_team or p.pos_x is None:
            continue
        d = distance(us, p)
        if d is None or d > WAVE_ALLY_MAX_DIST:
            continue
        w = _player_wave(conn, match_id, p, our_team, at_ms)
        if w is not None and (best_d is None or d < best_d):
            best, best_d = w, d
    return (best, "ally") if best is not None else (None, None)


def cs_last_3min_diff(
    conn: sqlite3.Connection, match_id: str, us_id: int, our_team: int, at_ms: int
) -> Optional[int]:
    """Our CS gained in the last 3 min minus the same-role enemy's gain."""
    our_now = _cs_at(conn, match_id, us_id, at_ms)
    our_then = _cs_at(conn, match_id, us_id, at_ms - CS_WINDOW_MS)
    if our_now is None or our_then is None:
        return None
    our_gain = our_now - our_then
    pos_row = conn.execute(
        "SELECT team_position FROM participants WHERE match_id = ? AND participant_id = ?",
        (match_id, us_id),
    ).fetchone()
    role = pos_row["team_position"] if pos_row else None
    opp = conn.execute(
        "SELECT participant_id FROM participants "
        "WHERE match_id = ? AND team_id <> ? AND team_position = ? LIMIT 1",
        (match_id, our_team, role),
    ).fetchone() if role else None
    if not opp:
        return our_gain
    opp_now = _cs_at(conn, match_id, opp["participant_id"], at_ms)
    opp_then = _cs_at(conn, match_id, opp["participant_id"], at_ms - CS_WINDOW_MS)
    if opp_now is None or opp_then is None:
        return our_gain
    return our_gain - (opp_now - opp_then)


# --------------------------------------------------------- state_features dict
def build_state_features(
    conn: sqlite3.Connection,
    match_id: str,
    ws: WorldState,
    us_id: int,
    our_team: int,
    *,
    entry_hp_pct: Optional[float] = None,
    enemies_unseen: Optional[int] = None,
    wave_state: Optional[str] = None,
    cs_diff: Optional[int] = None,
    next_major_obj: Optional[str] = None,
    time_to_obj_s: Optional[float] = None,
    we_have_setup: Optional[bool] = None,
    objective_value: Optional[float] = None,
    include_wards: bool = True,
    include_cs: bool = True,
) -> dict:
    """Assemble the standardized state_features block from a WorldState.

    Detectors pass overrides for fields they already computed (entry_hp_pct,
    enemies_unseen, wave/objective context) to avoid recomputation.
    """
    us = ws.players[us_id]
    at_ms = ws.game_time_ms
    level_diff, gold_diff = local_power(ws, us_id, our_team)
    ally_n, enemy_n = nearby_counts(ws, us_id, our_team)
    if enemies_unseen is None:
        enemies_unseen = enemies_unseen_count(
            conn, match_id, our_team, at_ms, us_x=us.pos_x, us_y=us.pos_y)
    wards = (
        our_wards_nearby(conn, match_id, our_team, us.pos_x, us.pos_y, at_ms)
        if include_wards else None
    )
    if cs_diff is None and include_cs:
        cs_diff = cs_last_3min_diff(conn, match_id, us_id, our_team, at_ms)
    power = {
        "local_level_diff": level_diff,
        "local_gold_diff": gold_diff,
        "entry_hp_pct": round(entry_hp_pct, 3) if isinstance(entry_hp_pct, (int, float)) else None,
        "ally_count_nearby": ally_n,
        "enemy_count_nearby": enemy_n,
    }
    power["power_index"] = power_index(power)  # 0..1 relative-strength readout

    # Positional safety (τ-style) and local threat (influence-map heuristics).
    d_ally = nearest_ally_tower_dist(us.pos_x, us.pos_y, our_team)
    d_enemy = nearest_enemy_tower_dist(us.pos_x, us.pos_y, our_team)
    safety = None
    if isinstance(d_ally, (int, float)) and isinstance(d_enemy, (int, float)) and (d_ally + d_enemy) > 0:
        safety = round(d_enemy / (d_ally + d_enemy), 2)   # 1 = by our tower, 0 = at enemy tower
    threat = None
    if safety is not None:
        threat = round(_clamp(0.5 * min(enemy_n, 3) / 3 + 0.5 * (1 - safety), 0.0, 1.0), 2)
    elif enemy_n is not None:
        threat = round(_clamp(min(enemy_n, 3) / 3, 0.0, 1.0), 2)

    wave_source = "detector" if wave_state is not None else None
    if wave_state is None:                # infer if the detector didn't supply one
        wave_state, wave_source = infer_wave_state(conn, match_id, ws, us_id, our_team, at_ms, safety)

    return {
        "power": power,
        "info_risk": {
            "enemies_unseen": enemies_unseen,
            "enemy_jg_unseen_s": enemy_jg_unseen_s(conn, match_id, our_team, at_ms),
            "wards_nearby": wards,
            "distance_to_ally_tower_u": d_ally,
            "distance_to_enemy_tower_u": d_enemy,
            "safety_ratio": safety,
            "local_threat": threat,
        },
        "wave_tempo": {
            "wave_state": wave_state,
            "wave_source": wave_source,
            "cs_last_3min_diff": cs_diff,
        },
        "objectives": {
            "next_major_obj": next_major_obj,
            "time_to_obj_s": time_to_obj_s,
            "we_have_setup": we_have_setup,
            "objective_value": objective_value,
        },
    }
