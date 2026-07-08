"""Wrapper around Riot's Live Client Data API.

The Live Client API is served by the running League of Legends client on
localhost:2999 over HTTPS with a self-signed cert. It exposes the current
game state once a match is in progress and is officially sanctioned by
Riot, so Vanguard does not object to polling it.

Reference: https://developer.riotgames.com/docs/lol#game-client-api
"""
from __future__ import annotations

import urllib3
import requests

# The Live Client uses a self-signed certificate. Disabling warnings is fine
# because we're only talking to 127.0.0.1.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"
DEFAULT_TIMEOUT = 1.5


class LiveClientUnavailable(Exception):
    """Raised when the Live Client endpoint is not responding (no game running)."""


def _get(path: str, timeout: float = DEFAULT_TIMEOUT) -> dict | list:
    try:
        r = requests.get(f"{BASE_URL}{path}", verify=False, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise LiveClientUnavailable(str(e)) from e
    if r.status_code != 200:
        raise LiveClientUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def all_game_data() -> dict:
    """Full snapshot: activePlayer, allPlayers, events, gameData."""
    return _get("/allgamedata")


def game_stats() -> dict:
    """Just the game-level stats: gameTime, gameMode, mapName, etc."""
    return _get("/gamestats")


def event_data() -> dict:
    """List of all events that have occurred this game (kills, dragons, etc.)."""
    return _get("/eventdata")


def is_game_running() -> bool:
    """True if a LoL match is currently in progress."""
    try:
        game_stats()
        return True
    except LiveClientUnavailable:
        return False
