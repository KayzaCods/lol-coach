"""#11: payload de salud del pipeline para el banner de key muerta del dashboard.

Importar `dashboard` ejecuta load_config() en import-time (run-in-place), asi que
estos tests se saltan limpiamente en una maquina sin config.toml en vez de romper
la suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

try:
    import dashboard
except Exception:                      # sin config.toml (maquina ajena)
    dashboard = None

pytestmark = pytest.mark.skipif(
    dashboard is None, reason="requiere config.toml (dashboard carga config al importar)"
)

from conftest import ten_player_match


def test_dead_key_surfaces_with_days_since_ingest(conn, tmp_path):
    ten_player_match(conn)                       # ingested_at_utc = 2026-01-01
    status = tmp_path / "ingest_status.json"
    status.write_text(json.dumps({
        "last_run": "2026-07-01 15:31:32", "key_alive": False,
        "ingested": 0, "error": "api_key_expired",
    }), encoding="utf-8")

    res = dashboard.query_ingest_status(conn, status_path=status)

    assert res["key_alive"] is False
    assert res["error"] == "api_key_expired"
    assert res["last_ingested_utc"].startswith("2026-01-01")
    assert isinstance(res["days_since_ingest"], int) and res["days_since_ingest"] > 0


def test_missing_status_file_reports_unknown_not_crash(conn, tmp_path):
    ten_player_match(conn)
    res = dashboard.query_ingest_status(conn, status_path=tmp_path / "nope.json")
    assert res["key_alive"] is None              # desconocido, no False
    assert res["days_since_ingest"] is not None  # los dias salen de la DB igual


def test_alive_key_passes_through(conn, tmp_path):
    status = tmp_path / "ingest_status.json"
    status.write_text(json.dumps({"key_alive": True, "ingested": 2}), encoding="utf-8")
    res = dashboard.query_ingest_status(conn, status_path=status)
    assert res["key_alive"] is True
    assert res["days_since_ingest"] is None      # DB vacia: sin partidas ingestadas
