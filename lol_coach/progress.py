"""Behavioral-progress math for the player's dominant pattern (#10).

Pure stdlib, no DB: the I/O layer (dashboard) feeds per-match (deaths, incidents)
pairs in chronological order. Wilson intervals because a window holds 4-5
incidents over ~70 deaths — point estimates alone would celebrate noise
(audit 2026-07-01 + contraste R2). The criterion constants live HERE and only
here; the gathering layer imports them.
"""
from __future__ import annotations

import math

OVEREXT_CLASSES = ("engage_blind", "disengage_failed")
LEAD_POWER_INDEX = 0.55   # power_index above this at death = "with lead"
WINDOW = 12               # rolling window, games (backlog range 10-15)
GOAL_PROP = 0.05          # provisional goal: 1 incident per 20 deaths (until #17)
BLOCK = 25                # block size for the before/after comparison (range 20-30)


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes over n trials; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def rolling_series(per_match, window: int = WINDOW):
    """One point per window ending at each match (step 1). per_match is
    [(deaths, incidents)] in chronological order; empty result if the history
    is shorter than the window."""
    out = []
    for end in range(window, len(per_match) + 1):
        win = per_match[end - window:end]
        deaths = sum(d for d, _ in win)
        inc = sum(i for _, i in win)
        ci = wilson_interval(inc, deaths)
        out.append({
            "end": end, "games": window, "deaths": deaths, "incidents": inc,
            "rate_per_game": round(inc / window, 3),
            "deaths_per_game": round(deaths / window, 2),
            "prop": round(inc / deaths, 4) if deaths else None,
            "ci_low": round(ci[0], 4) if ci else None,
            "ci_high": round(ci[1], 4) if ci else None,
        })
    return out


def _block_stats(win):
    deaths = sum(d for d, _ in win)
    inc = sum(i for _, i in win)
    ci = wilson_interval(inc, deaths)
    return {"games": len(win), "deaths": deaths, "incidents": inc,
            "prop": round(inc / deaths, 4) if deaths else None,
            "ci_low": round(ci[0], 4) if ci else None,
            "ci_high": round(ci[1], 4) if ci else None}


def block_comparison(per_match, block: int = BLOCK):
    """Last `block` games vs the `block` before them, each with its own CI —
    the 'did you actually improve?' check (judge in blocks of 20-30, not per
    window). None until there are 2*block games."""
    if len(per_match) < 2 * block:
        return None
    return {"block": block,
            "prev": _block_stats(per_match[-2 * block:-block]),
            "last": _block_stats(per_match[-block:])}
