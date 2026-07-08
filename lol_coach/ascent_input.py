"""Parse Ascent's per-recording input log (events.csv.gz) into significant events.

Ascent records raw peripheral input at high frequency:
    t_ms,event_type,input_name,value_a,value_b
with event_type in {mouse_delta, cursor_pos, mouse_button_down, mouse_button_up,
key_down, key_up, mouse_wheel}. `t_ms` is ms from recording start, which equals
game time (Ascent records the whole match, in_match_start_s = 0).

derive_events() collapses ~200k raw rows into the events worth storing:
  - click      : a mouse button press, positioned at the last known cursor_pos,
                 with hold duration (down->up).
  - key        : a key press, with hold duration (down->up).
  - wheel      : a mouse wheel tick (camera zoom).
  - hover_dwell: a stretch where the cursor stayed within a small radius for
                 >800ms (the hesitation signal). cursor_pos is only emitted when
                 the cursor moves, so a time gap between cursor_pos rows with no
                 meaningful displacement is a dwell.

Key identities are scancode-set-1 best-effort, limited to LoL-relevant keys we
are confident about; everything else is "sc_NN" rather than asserting a wrong
label (the dominant code 0x0E/14 in real data is ambiguous — confirm empirically
by pressing a known key and checking the log).
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

DWELL_MIN_MS = 800        # cursor resting longer than this = a dwell
DWELL_RADIUS_PX = 20      # within this displacement still counts as "resting"
EDGE_MARGIN_PX = 50       # ignore dwells at screen edges (edge-pan, not hesitation)

# Scancode set 1 (make codes) -> LoL-relevant key labels. Conservative on
# purpose: only keys we're confident about. Unknowns become "sc_NN".
SCANCODE_SET1 = {
    15: "Tab", 16: "Q", 17: "W", 18: "E", 19: "R",
    32: "D", 33: "F", 48: "B", 57: "Space", 29: "LCtrl", 56: "LAlt",
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
}
SPELL_KEYS = {"Q", "W", "E", "R"}
SUMMONER_KEYS = {"D", "F"}


def key_label(scancode: str | int | None) -> str | None:
    if scancode is None or scancode == "":
        return None
    try:
        sc = int(scancode)
    except (TypeError, ValueError):
        return str(scancode)
    return SCANCODE_SET1.get(sc, f"sc_{sc}")


def in_minimap(x: int, y: int, w: int = 1920, h: int = 1080) -> bool:
    """Approx LoL minimap box (bottom-right). Generous; exact size is configurable."""
    return x >= w - 330 and y >= h - 300


def near_edge(x: int, y: int, w: int = 1920, h: int = 1080, margin: int = EDGE_MARGIN_PX) -> bool:
    return x <= margin or y <= margin or x >= w - margin or y >= h - margin


def classify_click(button: str, x: int | None, y: int | None, w: int, h: int) -> str:
    if x is not None and y is not None and in_minimap(x, y, w, h):
        return "map_command"
    if button == "right":
        return "move"
    return "attack"  # left click: attack-move / skillshot / select (not distinguishable here)


def classify_key(label: str | None) -> str:
    if label in SPELL_KEYS:
        return "spell_cast"
    if label in SUMMONER_KEYS:
        return "summoner_spell"
    if label == "B":
        return "recall"
    return "key"


_BUTTON = {"1": "left", "2": "right"}


def derive_events(csv_gz_path: Path, width: int = 1920, height: int = 1080) -> list[dict]:
    """Return significant input events (sorted by game_time_ms) from a recording."""
    events: list[dict] = []
    cur_x = cur_y = None
    down_t: dict[str, int] = {}        # button input_name -> t of press
    down_pos: dict[str, tuple] = {}    # button input_name -> (x, y) at press
    key_down_t: dict[str, int] = {}    # scancode -> t of press

    # Dwell tracking over cursor_pos.
    anchor_x = anchor_y = anchor_t = last_t = None

    def flush_dwell():
        if anchor_t is None or last_t is None:
            return
        dur = last_t - anchor_t
        if dur >= DWELL_MIN_MS and not near_edge(anchor_x, anchor_y, width, height):
            events.append({
                "game_time_ms": anchor_t, "event_type": "hover_dwell",
                "button": None, "key": None,
                "screen_x": anchor_x, "screen_y": anchor_y,
                "duration_ms": dur, "classified_intent": "hesitation",
            })

    with gzip.open(csv_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                t = int(row["t_ms"])
            except (TypeError, ValueError):
                continue
            et = row["event_type"]
            name = row.get("input_name") or ""

            if et == "cursor_pos":
                x, y = _int(row["value_a"]), _int(row["value_b"])
                if x is None:
                    continue
                if anchor_t is None:
                    anchor_x, anchor_y, anchor_t = x, y, t
                elif abs(x - anchor_x) <= DWELL_RADIUS_PX and abs(y - anchor_y) <= DWELL_RADIUS_PX:
                    last_t = t  # still resting near anchor
                else:
                    flush_dwell()
                    anchor_x, anchor_y, anchor_t, last_t = x, y, t, t
                cur_x, cur_y = x, y

            elif et == "mouse_button_down":
                down_t[name] = t
                down_pos[name] = (cur_x, cur_y)

            elif et == "mouse_button_up":
                t0 = down_t.pop(name, None)
                px, py = down_pos.pop(name, (cur_x, cur_y))
                button = _BUTTON.get(name, name)
                events.append({
                    "game_time_ms": (t0 if t0 is not None else t),
                    "event_type": "click", "button": button, "key": None,
                    "screen_x": px, "screen_y": py,
                    "duration_ms": (t - t0) if t0 is not None else None,
                    "classified_intent": classify_click(button, px, py, width, height),
                })

            elif et == "key_down":
                key_down_t.setdefault(name, t)  # ignore auto-repeat re-presses

            elif et == "key_up":
                t0 = key_down_t.pop(name, None)
                label = key_label(name)
                events.append({
                    "game_time_ms": (t0 if t0 is not None else t),
                    "event_type": "key", "button": None, "key": label,
                    "screen_x": None, "screen_y": None,
                    "duration_ms": (t - t0) if t0 is not None else None,
                    "classified_intent": classify_key(label),
                })

            elif et == "mouse_wheel":
                events.append({
                    "game_time_ms": t, "event_type": "wheel", "button": None, "key": None,
                    "screen_x": cur_x, "screen_y": cur_y,
                    "duration_ms": None, "classified_intent": "camera",
                })

    flush_dwell()
    events.sort(key=lambda e: e["game_time_ms"])
    return events


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
