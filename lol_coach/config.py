"""Load config.toml. Local, gitignored, contains the Riot API key."""
from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"config.toml not found at {path}. "
            f"Copy config.example.toml to config.toml and fill it in."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)
