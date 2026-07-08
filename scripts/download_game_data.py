"""Download League of Legends game data from official sources and save as JSON.

Sources used:
  1. Riot Data Dragon (official, versioned) — champion stats, items, runes, summoner spells
  2. Community Dragon (community-maintained) — extended item/champion data, recommended builds
  3. Merakianalytics champion.json — champion base stats per level (for coaching math)

Usage:
    .venv\\Scripts\\python.exe scripts\\download_game_data.py

Output:
    data/game_data/
        version.txt              — current patch version
        champions_summary.json   — all champions: id, name, tags, stats, partype
        champions_full.json      — all champions full: abilities, tips, skins
        items.json               — all items: stats, cost, build path, tags
        runes.json               — all rune paths, keystones and minor runes
        summoner_spells.json     — all summoner spells
        champion_stats.json      — base stats per level for every champion (Meraki)
        patch_notes_summary.json — latest patch highlights (scraped from official site)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "game_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DDRAGON_BASE = "https://ddragon.leagueoflegends.com"
CDRAGON_BASE = "https://raw.communitydragon.org/latest"
MERAKI_BASE  = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "lol-coach/1.0 (personal coaching tool)"


def get(url: str, label: str, timeout: int = 60) -> dict | list | None:
    print(f"  GET {label}...", end=" ", flush=True)
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        print(f"OK ({len(r.content)//1024} KB)")
        return data
    except Exception as e:
        print(f"FAIL: {e}")
        return None


def save(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> saved {path.name} ({path.stat().st_size // 1024} KB)")


def main() -> None:
    # ── 1. Version ────────────────────────────────────────────────────────────
    print("\n[1] Fetching current patch version...")
    versions = get(f"{DDRAGON_BASE}/api/versions.json", "versions")
    if not versions:
        print("ERROR: could not get version. Aborting.")
        sys.exit(1)
    version = versions[0]
    (OUT_DIR / "version.txt").write_text(version, encoding="utf-8")
    # Client-facing patch (e.g. 26.11) derived from the DDragon major (16.11.1):
    # since the 2025 renumbering the client patch runs +10 ahead of DDragon.
    _v = version.split(".")
    patch_version = (f"{int(_v[0]) + 10}.{_v[1]}"
                     if len(_v) >= 2 and _v[0].isdigit() else version)
    print(f"  DDragon version: {version}  ->  client patch: {patch_version}")

    cdn = f"{DDRAGON_BASE}/cdn/{version}/data/en_US"

    # ── 2. Champion summary ───────────────────────────────────────────────────
    print("\n[2] Downloading champion summary...")
    champ_summary_raw = get(f"{cdn}/champion.json", "champion.json")
    if champ_summary_raw:
        # Flatten to list, add relevant fields
        champions = {}
        for key, c in champ_summary_raw["data"].items():
            champions[key] = {
                "id": c["id"],
                "name": c["name"],
                "title": c["title"],
                "blurb": c["blurb"],
                "tags": c["tags"],
                "partype": c["partype"],   # resource type: Mana, Energy, None, etc.
                "stats": c["stats"],       # base stats (hp, mp, armor, spellblock, etc.)
                "info": c["info"],         # attack/defense/magic/difficulty 0-10
            }
        save({"version": version, "champions": champions}, OUT_DIR / "champions_summary.json")

    # ── 3. Champion FULL (abilities + recommended items) ──────────────────────
    print("\n[3] Downloading champion full data (abilities, spells, tips)...")
    champ_full_raw = get(f"{cdn}/championFull.json", "championFull.json", timeout=120)
    if champ_full_raw:
        champions_full = {}
        for key, c in champ_full_raw["data"].items():
            spells = []
            for sp in c.get("spells", []):
                spells.append({
                    "id": sp["id"],
                    "name": sp["name"],
                    "description": sp["description"],
                    "tooltip": sp.get("tooltip", ""),
                    "maxrank": sp.get("maxrank"),
                    "cost": sp.get("cost"),
                    "costBurn": sp.get("costBurn"),
                    "cooldown": sp.get("cooldown"),
                    "cooldownBurn": sp.get("cooldownBurn"),
                    "range": sp.get("range"),
                    "rangeBurn": sp.get("rangeBurn"),
                })
            passive = c.get("passive", {})
            recommended = []
            for rec in c.get("recommended", []):
                recommended.append({
                    "map": rec.get("map"),
                    "mode": rec.get("mode"),
                    "type": rec.get("type"),
                    "blocks": [
                        {
                            "type": b.get("type"),
                            "items": [{"id": it["id"], "count": it.get("count", 1)} for it in b.get("items", [])]
                        }
                        for b in rec.get("blocks", [])
                    ]
                })
            champions_full[key] = {
                "id": c["id"],
                "name": c["name"],
                "lore": c.get("lore", ""),
                "allytips": c.get("allytips", []),
                "enemytips": c.get("enemytips", []),
                "partype": c.get("partype"),
                "stats": c.get("stats"),
                "passive": {
                    "name": passive.get("name"),
                    "description": passive.get("description"),
                },
                "spells": spells,
                "recommended": recommended,
            }
        save({"version": version, "champions": champions_full}, OUT_DIR / "champions_full.json")

    # ── 4. Items ──────────────────────────────────────────────────────────────
    print("\n[4] Downloading items...")
    items_raw = get(f"{cdn}/item.json", "item.json")
    if items_raw:
        items = {}
        for item_id, it in items_raw["data"].items():
            items[item_id] = {
                "name": it["name"],
                "description": it.get("description", ""),
                "plaintext": it.get("plaintext", ""),
                "gold": it.get("gold", {}),
                "tags": it.get("tags", []),
                "stats": it.get("stats", {}),
                "depth": it.get("depth"),
                "from": it.get("from", []),
                "into": it.get("into", []),
                "maps": it.get("maps", {}),
                "requiredChampion": it.get("requiredChampion"),
                "requiredAlly": it.get("requiredAlly"),
                "specialRecipe": it.get("specialRecipe"),
            }
        save({"version": version, "items": items}, OUT_DIR / "items.json")

    # ── 5. Runes ──────────────────────────────────────────────────────────────
    print("\n[5] Downloading runes...")
    runes_raw = get(f"{cdn}/runesReforged.json", "runesReforged.json")
    if runes_raw:
        save({"version": version, "rune_paths": runes_raw}, OUT_DIR / "runes.json")

    # ── 6. Summoner spells ────────────────────────────────────────────────────
    print("\n[6] Downloading summoner spells...")
    spells_raw = get(f"{cdn}/summoner.json", "summoner.json")
    if spells_raw:
        spells = {}
        for key, sp in spells_raw["data"].items():
            spells[key] = {
                "id": sp["id"],
                "name": sp["name"],
                "description": sp["description"],
                "tooltip": sp.get("tooltip", ""),
                "maxrank": sp.get("maxrank"),
                "cooldown": sp.get("cooldown"),
                "effect": sp.get("effect"),
                "modes": sp.get("modes", []),
            }
        save({"version": version, "summoner_spells": spells}, OUT_DIR / "summoner_spells.json")

    # ── 7. Meraki champion stats (base stats per level, scaling) ──────────────
    print("\n[7] Downloading Meraki champion stats (per-level scaling)...")
    meraki = get(f"{MERAKI_BASE}/champions.json", "meraki champions.json", timeout=120)
    if meraki:
        # Meraki has precise base stats + per-level growth for every champion
        meraki_stats = {}
        for champ_name, data in meraki.items():
            s = data.get("stats", {})
            meraki_stats[champ_name] = {
                "resource": data.get("resource"),   # "Mana", "Energy", "None", "Rage", etc.
                "attack_range": data.get("attackRange"),
                "adaptive_type": data.get("adaptiveType"),  # PHYSICAL or MAGICAL
                "stats": {
                    # base + per_level for each stat
                    k: {"base": v.get("flat"), "per_level": v.get("perLevel"), "percent": v.get("percent")}
                    for k, v in s.items()
                }
            }
        save({"source": "merakianalytics", "champions": meraki_stats}, OUT_DIR / "champion_stats_meraki.json")

    # ── 8. Community Dragon — item full data (includes recommended builds) ────
    print("\n[8] Downloading Community Dragon item data...")
    cdragon_items = get(
        f"{CDRAGON_BASE}/game/data/items.cdtb.bin.json",
        "cdragon items",
        timeout=120
    )
    if cdragon_items:
        # Filter to only the fields we care about
        items_cdragon = {}
        for item_id, it in cdragon_items.items():
            if not isinstance(it, dict):
                continue
            items_cdragon[item_id] = {
                "name": it.get("name"),
                "description": it.get("description"),
                "price": it.get("price"),
                "priceTotal": it.get("priceTotal"),
                "tags": it.get("categories", []),
                "from": it.get("requiredItems", []),
            }
        save({"source": "community_dragon", "items": items_cdragon}, OUT_DIR / "items_cdragon.json")

    # ── 9. Summary index ──────────────────────────────────────────────────────
    print("\n[9] Building index...")
    index = {
        "patch_version": patch_version,
        "ddragon_version": version,
        "version": version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {f.name: f.stat().st_size for f in sorted(OUT_DIR.iterdir()) if f.suffix in (".json", ".txt")},
        "sources": {
            "data_dragon": f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/",
            "community_dragon": "https://raw.communitydragon.org/latest/",
            "meraki": "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US/",
        },
        "champion_count": len(champions) if champ_summary_raw else 0,
        "item_count": len(items) if items_raw else 0,
    }
    save(index, OUT_DIR / "_index.json")

    print("\nDone! Game data saved to data/game_data/")
    print(f"  Files: {list(OUT_DIR.iterdir())}")


if __name__ == "__main__":
    main()
