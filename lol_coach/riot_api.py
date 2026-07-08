"""Thin wrapper around the Riot Games REST API.

Endpoints used (Phase 1):
- account-v1: resolve gameName#tagLine -> PUUID.
- match-v5: list of match IDs for a PUUID + match data + timeline.

Rate limits:
- Development key: 20 req/s and 100 req/2min.
- Personal key: higher, must be applied for.
We retry once on HTTP 429 honoring Retry-After.
"""
from __future__ import annotations

import time
import requests


class RiotAPIError(Exception):
    pass


class RiotAPI:
    def __init__(self, api_key: str, routing: str = "americas", platform: str = "la1"):
        self.api_key = api_key
        self.routing = routing
        self.platform = platform
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = api_key

    def _routing_host(self) -> str:
        return f"https://{self.routing}.api.riotgames.com"

    def _platform_host(self) -> str:
        return f"https://{self.platform}.api.riotgames.com"

    def _get(self, base: str, path: str, params: dict | None = None) -> dict | list:
        url = base + path
        for attempt in range(2):
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt == 0:
                wait = int(r.headers.get("Retry-After", "5"))
                time.sleep(min(wait, 30))
                continue
            raise RiotAPIError(f"HTTP {r.status_code} {url}: {r.text[:300]}")
        raise RiotAPIError(f"Rate limited after retry: {url}")

    def account_by_riot_id(self, game_name: str, tag_line: str) -> dict:
        return self._get(
            self._routing_host(),
            f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}",
        )

    def match_ids_by_puuid(
        self,
        puuid: str,
        count: int = 20,
        start: int = 0,
        queue: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        params: dict = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._get(
            self._routing_host(),
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
            params=params,
        )

    def match(self, match_id: str) -> dict:
        return self._get(self._routing_host(), f"/lol/match/v5/matches/{match_id}")

    def match_timeline(self, match_id: str) -> dict:
        return self._get(
            self._routing_host(), f"/lol/match/v5/matches/{match_id}/timeline"
        )
