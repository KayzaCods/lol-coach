"""Loader for static League of Legends game data.

Reads the JSON files in data/game_data/ (populated by scripts/download_game_data.py)
and exposes helper functions that detectors can call without touching the files directly.

The module caches its data on first access (module-level singletons) so repeated
calls in a long analysis run are essentially free.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ── Path resolution ──────────────────────────────────────────────────────────
# Detectors can be run from any working directory, so we resolve relative to
# this file rather than cwd.
_GAME_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "game_data"

# ── Module-level caches ───────────────────────────────────────────────────────
_meraki: dict | None = None   # champion_name → {resource, stats, ...}
_items: dict | None = None    # item_id (str) → {name, gold, stats, tags, ...}
_runes: list | None = None    # list of rune path dicts
_champ_summary: dict | None = None  # champion_name → {partype, tags, stats, info}


# ── Internal loaders ─────────────────────────────────────────────────────────

def _load_meraki() -> dict:
    global _meraki
    if _meraki is not None:
        return _meraki
    path = _GAME_DATA_DIR / "champion_stats_meraki.json"
    if not path.exists():
        _meraki = {}
        return _meraki
    raw = json.loads(path.read_text(encoding="utf-8"))
    _meraki = raw.get("champions", {})
    return _meraki


def _load_champ_summary() -> dict:
    global _champ_summary
    if _champ_summary is not None:
        return _champ_summary
    path = _GAME_DATA_DIR / "champions_summary.json"
    if not path.exists():
        _champ_summary = {}
        return _champ_summary
    raw = json.loads(path.read_text(encoding="utf-8"))
    _champ_summary = raw.get("champions", {})
    return _champ_summary


def _load_items() -> dict:
    global _items
    if _items is not None:
        return _items
    path = _GAME_DATA_DIR / "items.json"
    if not path.exists():
        _items = {}
        return _items
    raw = json.loads(path.read_text(encoding="utf-8"))
    _items = raw.get("items", {})
    return _items


def _load_runes() -> list:
    global _runes
    if _runes is not None:
        return _runes
    path = _GAME_DATA_DIR / "runes.json"
    if not path.exists():
        _runes = []
        return _runes
    raw = json.loads(path.read_text(encoding="utf-8"))
    _runes = raw.get("rune_paths", [])
    return _runes


# ── Name normalization ────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Strip non-alphanumeric and lowercase for fuzzy matching.

    Examples:
        "Bel'Veth"  → "belveth"
        "Cho'Gath"  → "chogath"
        "Kai'Sa"    → "kaisa"
        "Kog'Maw"   → "kogmaw"
        "Rek'Sai"   → "reksai"
        "Kha'Zix"   → "khazix"
        "Wukong"    → "wukong"   (MonkeyKing in API)
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_meraki(champion_name: str) -> dict | None:
    data = _load_meraki()
    norm = _normalize(champion_name)
    for key, val in data.items():
        if _normalize(key) == norm:
            return val
    return None


def _find_summary(champion_name: str) -> dict | None:
    data = _load_champ_summary()
    norm = _normalize(champion_name)
    for key, val in data.items():
        if _normalize(key) == norm:
            return val
    return None


# ── Public API ────────────────────────────────────────────────────────────────

# Resource types that have a meaningful "mana/energy bar" we can analyze
MEANINGFUL_RESOURCES = {"Mana", "Energy"}
# Riot Live Client resourceType strings that indicate a real resource bar
LIVE_CLIENT_MEANINGFUL = {"MANA", "ENERGY"}


def resource_type(champion_name: str) -> str:
    """Return the Meraki resource type string for a champion.

    Returns "NONE" if unknown or truly resourceless.
    Meraki values: "MANA", "ENERGY", "NONE", "RAGE", "FURY",
                   "COURAGE", "SHIELD", "FEROCITY", "HEAT",
                   "GRIT", "CRIMSON_RUSH", "FLOW", etc.
    """
    entry = _find_meraki(champion_name)
    if entry:
        return (entry.get("resource") or "NONE").upper()
    # Fallback: check Data Dragon partype
    summary = _find_summary(champion_name)
    if summary:
        pt = summary.get("partype", "")
        if pt == "Mana":
            return "MANA"
        if pt == "Energy":
            return "ENERGY"
        if pt in ("", "None", "Blood Well", "Courage", "Fury", "Ferocity",
                  "Rage", "Shield", "Heat", "Grit", "Crimson Rush", "Flow"):
            return "NONE"
    return "MANA"   # safest default — avoids hiding mana for unknown champs


def has_meaningful_resource(champion_name: str) -> bool:
    """True if the champion has a Mana or Energy bar worth analyzing.

    Use this before computing mana% in any detector to avoid
    showing 0% mana for Garen/Bel'Veth/etc.
    """
    return resource_type(champion_name) in ("MANA", "ENERGY")


def base_hp(champion_name: str, level: int = 1) -> Optional[float]:
    """Estimated max HP at a given level (1-18) from Meraki data.

    Returns None if the champion is not found.
    """
    entry = _find_meraki(champion_name)
    if not entry:
        return None
    stats = entry.get("stats", {})
    hp = stats.get("health", {})
    base = hp.get("base")
    growth = hp.get("per_level") or 0.0
    if base is None:
        return None
    # Standard LoL formula: base + growth * (level - 1) * (0.7025 + 0.0175 * (level - 1))
    # Simplified linear approximation (within 3% for levels 1-18):
    lvl = max(1, min(18, level))
    return base + growth * (lvl - 1)


def champion_tags(champion_name: str) -> list[str]:
    """Data Dragon tags for a champion, e.g. ['Marksman'], ['Fighter', 'Tank']."""
    summary = _find_summary(champion_name)
    return summary.get("tags", []) if summary else []


def item_stats(item_id: str | int) -> dict:
    """Return {name, stats, gold, tags, from, into} for an item id."""
    items = _load_items()
    return items.get(str(item_id), {})


def game_data_available() -> bool:
    """True if the data/game_data/ directory exists with at least the Meraki file."""
    return (_GAME_DATA_DIR / "champion_stats_meraki.json").exists()


def patch_version() -> Optional[str]:
    """Client-facing patch (e.g. '26.13') from _index.json, or None if absent.
    Written by download_game_data.py; lets the dashboard show the live patch
    instead of a hardcoded string that drifts every two weeks."""
    path = _GAME_DATA_DIR / "_index.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("patch_version")
    except (OSError, ValueError):
        return None
