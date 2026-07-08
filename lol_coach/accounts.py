"""Registry of Riot accounts and their resolved PUUIDs.

The recorder doesn't know or care which account is logged in — it just
polls the local Live Client. When ingesting, we use the
`active_player_name` (`GameName#TagLine`) captured in each session's
summary.json to pick the right PUUID for the Riot API call.

PUUIDs are cached to data/puuids.json so we don't re-resolve every run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .riot_api import RiotAPI


class AccountRegistry:
    def __init__(
        self,
        accounts: list[dict],
        cache_path: Path,
        api: RiotAPI,
    ):
        self.accounts = accounts  # [{"game_name": ..., "tag_line": ...}, ...]
        self.cache_path = Path(cache_path)
        self.api = api
        self._by_riot_id: dict[str, str] = {}
        self._load_cache()

    @staticmethod
    def riot_id_of(game_name: str, tag_line: str) -> str:
        return f"{game_name}#{tag_line}"

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                self._by_riot_id = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._by_riot_id = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._by_riot_id, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def resolve_all(self) -> dict[str, str]:
        """Ensure every configured account has its PUUID cached. Returns dict."""
        changed = False
        for acc in self.accounts:
            key = self.riot_id_of(acc["game_name"], acc["tag_line"])
            if key not in self._by_riot_id:
                resp = self.api.account_by_riot_id(acc["game_name"], acc["tag_line"])
                self._by_riot_id[key] = resp["puuid"]
                changed = True
        if changed:
            self._save_cache()
        return dict(self._by_riot_id)

    def puuid_for(self, riot_id: str | None) -> str | None:
        """Lookup PUUID by 'GameName#TagLine'. None if not configured."""
        if not riot_id:
            return None
        return self._by_riot_id.get(riot_id)

    def cohort_for(self, riot_id: str | None) -> str | None:
        """Lookup the cohort (elo tag) for a Riot ID. None if not set."""
        if not riot_id:
            return None
        for acc in self.accounts:
            if self.riot_id_of(acc["game_name"], acc["tag_line"]) == riot_id:
                return acc.get("cohort")
        return None

    def all_riot_ids(self) -> list[str]:
        return list(self._by_riot_id.keys())

    def all_puuids(self) -> list[str]:
        return list(self._by_riot_id.values())


def load_accounts(cfg: dict, api: RiotAPI, data_dir: Path) -> AccountRegistry:
    """Build the registry from config['riot']['accounts'] and resolve PUUIDs."""
    accounts = cfg["riot"].get("accounts")
    if not accounts:
        # Backward compat: old single-account format
        gn = cfg["riot"].get("game_name")
        tl = cfg["riot"].get("tag_line")
        if gn and tl:
            accounts = [{"game_name": gn, "tag_line": tl}]
        else:
            raise ValueError(
                "No accounts configured. Add [[riot.accounts]] blocks to config.toml."
            )
    reg = AccountRegistry(accounts, data_dir / "puuids.json", api)
    reg.resolve_all()
    return reg
