"""#16: capa I/O de la Sesion de hoy — query_session solo propone decisiones SIN
marcar de cohorts propios, excluyendo trades positivos ocultos. Mismo skip-guard
que test_ingest_status (dashboard carga config en import-time)."""
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
_OPTS = json.dumps([{"label": "Entrar (lo que hiciste)", "ev_score": 0.4},
                    {"label": "Retirarte", "ev_score": 0.7}])


def _dec(conn, match_id, t_ms, detector="death_v1", action_class="engage_blind",
         power_index=0.7, options=_OPTS, feedback=None, positive_trade=False):
    ctx = {"inferred_action_class": action_class,
           "state_features": {"power": {"power_index": power_index}}}
    if positive_trade:
        ctx["positive_trade"] = True
    conn.execute(
        "INSERT INTO decisions (match_id, detector_id, game_time_ms, context_json, "
        "options_json, argument, user_feedback, created_at_utc) VALUES (?,?,?,?,?,?,?,?)",
        (match_id, detector, t_ms, json.dumps(ctx), options, "", feedback, _NOW),
    )


def test_session_excludes_marked_reference_and_hidden(conn):
    insert_match(conn, "A")
    conn.execute("UPDATE matches SET game_start_ms=1000 WHERE match_id='A'")
    _dec(conn, "A", 100)                                  # dominante sin marcar: entra
    _dec(conn, "A", 200, feedback="agree")                # marcada: fuera
    _dec(conn, "A", 300, detector="trade_v1", action_class=None,
         power_index=None, positive_trade=True)           # trade ganado oculto: fuera
    conn.execute(
        "INSERT INTO matches (match_id, game_creation_ms, game_start_ms, "
        "game_duration_s, queue_id, our_puuid, our_cohort, ingested_at_utc) "
        "VALUES ('REF', 0, 3000, 1800, 420, 'p_ref', 'challenger', ?)", (_NOW,))
    _dec(conn, "REF", 100)                                # referencia: fuera
    conn.commit()

    res = dashboard.query_session(conn)

    ids = [(m["match"], m["game_time_ms"]) for m in res["moments"]]
    assert ids == [("A", 100)]
    assert res["moments"][0]["bucket"] == "dominant"
    assert res["moments"][0]["ev_gap"] == 0.3
    assert res["size"] == 6


def test_query_session_exclude_brings_only_new_moments(conn):
    # "Traer mas": los ids ya mostrados se excluyen para que la siguiente tanda
    # no repita momentos (ni los saltados sin marcar).
    insert_match(conn, "A")
    conn.execute("UPDATE matches SET game_start_ms=1000 WHERE match_id='A'")
    _dec(conn, "A", 100)
    _dec(conn, "A", 200)
    conn.commit()

    res1 = dashboard.query_session(conn)
    ids1 = {m["id"] for m in res1["moments"]}
    assert len(ids1) == 2

    res2 = dashboard.query_session(conn, exclude=ids1)
    assert res2["moments"] == []           # solo habia 2 candidatos y ya salieron


def test_session_buckets_and_temporal(conn):
    insert_match(conn, "A")
    conn.execute("UPDATE matches SET game_start_ms=1000 WHERE match_id='A'")
    # 3 dominantes concentradas en min 12-14 (nota temporal) + 1 objetivo con gap
    for t in (12 * 60_000, 13 * 60_000, 14 * 60_000):
        _dec(conn, "A", t)
    _dec(conn, "A", 20 * 60_000, detector="objective_readiness_v1", action_class=None,
         power_index=None)
    conn.commit()

    res = dashboard.query_session(conn)

    buckets = [m["bucket"] for m in res["moments"]]
    assert buckets.count("dominant") == 3
    assert buckets.count("objective") == 1
    assert res["temporal"] == {"count": 3, "last": 3, "min_from": 12, "min_to": 14}


def test_session_excludes_typed_marks(conn):
    insert_match(conn, "A")
    conn.execute("UPDATE matches SET game_start_ms=1000 WHERE match_id='A'")
    _dec(conn, "A", 100)                                  # sin tipo, sin marcar: entra
    _dec(conn, "A", 200)
    conn.execute("UPDATE decisions SET user_mark_type='wrong_moment' "
                 "WHERE game_time_ms=200")                # tipada: fuera
    conn.commit()
    res = dashboard.query_session(conn)
    times = [m["game_time_ms"] for m in res["moments"]]
    assert 100 in times
    assert 200 not in times
