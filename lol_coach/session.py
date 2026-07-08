"""Prioritized 'Sesion de hoy' review queue (#16).

Pure stdlib, no DB: the I/O layer (dashboard) feeds flat candidate dicts —
UNMARKED decisions from the player's own cohorts. The mix is DeepSeek's via the
external-designs contrast (R4), chosen over Perplexity's because it includes
positive reinforcement and an anti-monoculture slot: the dominant pattern gets
most of the budget, but never all of it (don't overtrain a single bias).
"""
from __future__ import annotations

from collections import Counter

from .progress import LEAD_POWER_INDEX, OVEREXT_CLASSES

SESSION_SIZE = 6
QUOTAS = (("dominant", 3), ("objective", 1), ("success", 1), ("other", 1))
SUCCESS_MIN_MARGIN = 0.2   # optimal beat the worst option by this = the call mattered
TEMPORAL_LAST = 5          # look at the last N dominant-pattern incidents
TEMPORAL_BIN_MIN = 3       # game-minute bin width for the clustering note
TEMPORAL_MIN_COUNT = 3     # incidents in one bin to claim a concentration

REASONS = {   # why this moment made today's session — shown as the card badge
    "dominant": "patrón dominante · peor EV gap primero",
    "objective": "objetivo de mayor costo",
    "success": "acierto difícil (refuerzo de consistencia)",
    "other": "variedad (no sobre-entrenar un solo sesgo)",
}


def classify(c: dict) -> str:
    """First matching bucket, priority dominant > objective > success > other."""
    pi = c.get("power_index")
    if (c.get("detector") == "death_v1"
            and c.get("action_class") in OVEREXT_CLASSES
            and isinstance(pi, (int, float)) and pi > LEAD_POWER_INDEX):
        return "dominant"
    gap = c.get("ev_gap")
    if (c.get("detector") == "objective_readiness_v1"
            and isinstance(gap, (int, float)) and gap > 0):
        return "objective"
    opt, worst = c.get("ev_optimal"), c.get("ev_min")
    if (isinstance(gap, (int, float)) and gap <= 0
            and isinstance(opt, (int, float)) and isinstance(worst, (int, float))
            and opt - worst >= SUCCESS_MIN_MARGIN):
        return "success"
    return "other"


def _key_gap_recent(c):
    return (-(c.get("ev_gap") or 0),
            -(c.get("game_start_ms") or 0), -(c.get("game_time_ms") or 0))


def _key_margin_recent(c):
    margin = (c.get("ev_optimal") or 0) - (c.get("ev_min") or 0)
    return (-margin, -(c.get("game_start_ms") or 0), -(c.get("game_time_ms") or 0))


def _key_recent(c):
    return (-(c.get("game_start_ms") or 0), -(c.get("game_time_ms") or 0))


_SORT = {"dominant": _key_gap_recent, "objective": _key_gap_recent,
         "success": _key_margin_recent, "other": _key_recent}


def build_session(cands: list, size: int = SESSION_SIZE, quotas=QUOTAS) -> list:
    """Quota fill + backfill. Returns the picked candidates (copies) with
    `bucket` and `reason` set. Backfill keeps the session full when a bucket
    lacks candidates, drawing leftovers in the same priority order."""
    buckets = {name: [] for name, _ in quotas}
    for c in cands:
        buckets[classify(c)].append(c)
    for name in buckets:
        buckets[name].sort(key=_SORT[name])
    picked, used = [], set()

    def _take(c, name):
        picked.append({**c, "bucket": name, "reason": REASONS[name]})
        used.add(c["id"])

    for name, quota in quotas:
        for c in buckets[name][:quota]:
            if c["id"] not in used:
                _take(c, name)
    if len(picked) < size:
        for name, _ in quotas:
            for c in buckets[name]:
                if len(picked) >= size:
                    break
                if c["id"] not in used:
                    _take(c, name)
    return picked[:size]


def temporal_note(minutes, last: int = TEMPORAL_LAST, bin_min: int = TEMPORAL_BIN_MIN,
                  min_count: int = TEMPORAL_MIN_COUNT):
    """Clustering of the dominant pattern's game-minute over the LAST incidents
    (chronological input; marked or not — the note describes the pattern, not
    the queue). '3 de tus últimos 5 overextends fueron en min 12-14' is worth
    more as an argument than any single card."""
    recent = [m for m in minutes if isinstance(m, (int, float))][-last:]
    if len(recent) < min_count:
        return None
    bins = Counter(int(m) // bin_min for m in recent)
    b, count = bins.most_common(1)[0]
    if count < min_count:
        return None
    return {"count": count, "last": len(recent),
            "min_from": b * bin_min, "min_to": b * bin_min + bin_min - 1}
