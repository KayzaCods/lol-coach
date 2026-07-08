"""#10: capa I/O del progreso conductual — _overextension_per_match cuenta bien
por partida y excluye la referencia. Mismo skip-guard que test_ingest_status
(importar dashboard carga config.toml en import-time)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

try:
    import dashboard
except Exception:
    dashboard = None

pytestmark = pytest.mark.skipif(
    dashboard is None, reason="requiere config.toml (dashboard carga config al importar)"
)

from conftest import insert_match

_NOW = "2026-01-01T00:00:00Z"


def _death(conn, match_id, t_ms, action_class, power_index):
    ctx = {"inferred_action_class": action_class,
           "state_features": {"power": {"power_index": power_index}}}
    conn.execute(
        "INSERT INTO decisions (match_id, detector_id, game_time_ms, context_json, "
        "options_json, argument, created_at_utc) VALUES (?,?,?,?,?,?,?)",
        (match_id, "death_v1", t_ms, json.dumps(ctx), "[]", "", _NOW),
    )


def test_counts_per_match_and_excludes_reference(conn):
    # Partida propia A: 3 muertes -> 1 incidente (engage_blind con lead; el
    # engage_worth con lead NO cuenta; el engage_blind sin lead tampoco).
    insert_match(conn, "A")
    conn.execute("UPDATE matches SET game_start_ms=1000 WHERE match_id='A'")
    _death(conn, "A", 100, "engage_blind", 0.70)
    _death(conn, "A", 200, "engage_worth", 0.90)
    _death(conn, "A", 300, "engage_blind", 0.30)
    # Partida propia B: sin muertes (cuenta como partida con 0/0).
    insert_match(conn, "B")
    conn.execute("UPDATE matches SET game_start_ms=2000 WHERE match_id='B'")
    # Partida challenger: se excluye entera aunque tenga incidentes.
    conn.execute(
        "INSERT INTO matches (match_id, game_creation_ms, game_start_ms, "
        "game_duration_s, queue_id, our_puuid, our_cohort, ingested_at_utc) "
        "VALUES ('REF', 0, 3000, 1800, 420, 'puuid_ref', 'challenger', ?)", (_NOW,))
    _death(conn, "REF", 100, "engage_blind", 0.90)
    conn.commit()

    pm = dashboard._overextension_per_match(conn, None)

    assert [m["match_id"] for m in pm] == ["A", "B"]   # cronologico, sin REF
    assert pm[0]["deaths"] == 3 and pm[0]["incidents"] == 1
    assert pm[1]["deaths"] == 0 and pm[1]["incidents"] == 0


def test_compute_progress_none_with_short_history(conn):
    insert_match(conn, "A")
    _death(conn, "A", 100, "engage_blind", 0.70)
    conn.commit()
    # 1 partida < WINDOW -> sin serie -> bloque None (la UI no pinta nada raro).
    assert dashboard.compute_progress(conn, None) is None


def test_reference_cohort_request_falls_back_to_own(conn):
    # Pedir cohort=challenger no debe producir progreso del jugador de referencia.
    insert_match(conn, "A")
    _death(conn, "A", 100, "engage_blind", 0.70)
    conn.execute(
        "INSERT INTO matches (match_id, game_creation_ms, game_start_ms, "
        "game_duration_s, queue_id, our_puuid, our_cohort, ingested_at_utc) "
        "VALUES ('REF', 0, 3000, 1800, 420, 'puuid_ref', 'challenger', ?)", (_NOW,))
    conn.commit()
    pm = dashboard._overextension_per_match(conn, "challenger")
    assert [m["match_id"] for m in pm] == ["A"]
