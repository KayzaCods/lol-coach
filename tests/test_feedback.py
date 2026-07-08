"""#12: vocabulario de tipos de marca y el gate de entrenamiento del learner."""
from __future__ import annotations

import re

from lol_coach.feedback import MARK_TYPES, TRAINING_TYPES, trains_learner

_NOISE = re.compile(r"ejecu|mec[aá]nic|fall[eé]|combo|lag|oculta", re.I)


def test_mark_types_vocabulary():
    assert MARK_TYPES == ("decision", "execution", "mixed", "missing_context", "wrong_moment")
    assert TRAINING_TYPES == frozenset({"decision"})


def test_only_decision_type_trains():
    assert trains_learner("decision", "", _NOISE) is True
    for t in ("execution", "mixed", "missing_context", "wrong_moment"):
        assert trains_learner(t, "cualquier nota", _NOISE) is False


def test_untyped_falls_back_to_noise_regex():
    # marca vieja sin tipo (None o ""): usa el regex como antes
    assert trains_learner(None, "la decision fue mala", _NOISE) is True
    assert trains_learner(None, "falle el combo", _NOISE) is False
    assert trains_learner("", "buena nota", _NOISE) is True
    assert trains_learner(None, None, _NOISE) is True


def test_decisions_has_user_mark_type_column(tmp_path):
    from lol_coach import db as _db

    conn = _db.connect(tmp_path / "a.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    assert "user_mark_type" in cols
    conn.close()
    conn2 = _db.connect(tmp_path / "a.db")   # reconectar: la migración es idempotente
    cols2 = {r[1] for r in conn2.execute("PRAGMA table_info(decisions)")}
    assert "user_mark_type" in cols2


def _dec_obj():
    from lol_coach.decisions.base import Decision, Option
    return Decision(detector_id="death_v1", match_id="TEST", game_time_ms=60000,
                    moment="m", outcome="o", context={"action": {"action_id": 1}},
                    options=[Option("a", "c", 0.4)], argument="arg")


def test_persist_preserves_mark_type(conn):
    from lol_coach.analysis import persist_decisions
    from conftest import insert_match

    insert_match(conn, "TEST")
    persist_decisions(conn, "TEST", [_dec_obj()])
    conn.execute("UPDATE decisions SET user_mark_type='execution', "
                 "user_feedback_note='x' WHERE match_id='TEST'")
    conn.commit()
    persist_decisions(conn, "TEST", [_dec_obj()])   # re-analiza
    mt = conn.execute("SELECT user_mark_type FROM decisions "
                      "WHERE match_id='TEST'").fetchone()[0]
    assert mt == "execution"


def test_persist_preserves_mark_type_only_row(conn):
    # marca tipada SIN feedback/nota/best_option: aun asi debe preservarse
    from lol_coach.analysis import persist_decisions
    from conftest import insert_match

    insert_match(conn, "TEST")
    persist_decisions(conn, "TEST", [_dec_obj()])
    conn.execute("UPDATE decisions SET user_mark_type='wrong_moment' "
                 "WHERE match_id='TEST'")
    conn.commit()
    persist_decisions(conn, "TEST", [_dec_obj()])
    mt = conn.execute("SELECT user_mark_type FROM decisions "
                      "WHERE match_id='TEST'").fetchone()[0]
    assert mt == "wrong_moment"


import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
try:
    import dashboard as _dash
except Exception:
    _dash = None
import pytest as _pytest

_needs_dash = _pytest.mark.skipif(
    _dash is None, reason="requiere config.toml (dashboard carga config al importar)")


@_needs_dash
def test_set_feedback_mark_type_valid_and_invalid(conn):
    from conftest import insert_match

    insert_match(conn, "A")
    conn.execute(
        "INSERT INTO decisions (match_id, detector_id, game_time_ms, context_json, "
        "options_json, argument, created_at_utc) VALUES "
        "('A','death_v1',100,'{}','[]','arg','2026-01-01T00:00:00Z')")
    conn.commit()
    did = conn.execute("SELECT id FROM decisions").fetchone()[0]
    assert _dash.set_feedback(conn, did, {"mark_type": "decision"})["mark_type"] == "decision"
    assert _dash.set_feedback(conn, did, {"mark_type": "bogus"})["mark_type"] is None


def _dec_bo(conn, t, mark_type=None, note=""):
    conn.execute(
        "INSERT INTO decisions (match_id, detector_id, game_time_ms, context_json, "
        "options_json, argument, user_best_option, user_feedback_note, "
        "user_mark_type, created_at_utc) VALUES "
        "('A','death_v1',?,'{}','[]','arg',0,?,?,'2026-01-01T00:00:00Z')",
        (t, note, mark_type))


@_needs_dash
def test_learner_gate_by_mark_type(conn):
    from conftest import insert_match

    insert_match(conn, "A")
    _dec_bo(conn, 100, mark_type="execution")          # tipada no-decision -> excluida
    _dec_bo(conn, 200, mark_type="decision")           # tipada decision -> pasa el gate
    _dec_bo(conn, 300, mark_type=None, note="falle el combo")  # vieja con ruido -> excluida
    conn.commit()
    _pairs, stats = _dash._decision_feedback_pairs(conn)
    excluded = {e["id"] for e in stats["excluded_noisy"]}
    ids = {t: conn.execute("SELECT id FROM decisions WHERE game_time_ms=?",
                           (t,)).fetchone()[0] for t in (100, 200, 300)}
    assert ids[100] in excluded          # execution excluida
    assert ids[300] in excluded          # ruido legacy excluida
    assert ids[200] not in excluded      # decision no se excluye en el gate


def _dec_queue(conn, t, mark_type=None, note=""):
    # contexto minimo que produce un item en la cola de triaje: dos acciones
    # disponibles y best_option (1) != la tomada (0). _commit_fit puede devolver
    # None (sin state_features) y el item igual entra: solo cambia `agree`.
    ctx = ('{"action": {"action_id": "A", "available_actions": '
           '[{"action_id": "A"}, {"action_id": "B"}]}}')
    conn.execute(
        "INSERT INTO decisions (match_id, detector_id, game_time_ms, moment, "
        "context_json, options_json, argument, user_best_option, "
        "user_feedback_note, user_mark_type, created_at_utc) VALUES "
        "('A','death_v1',?,'m',?,'[]','arg',1,?,?,'2026-01-01T00:00:00Z')",
        (t, ctx, note, mark_type))


@_needs_dash
def test_triage_queue_gate_by_mark_type(conn):
    # Simetria con el learner: query_review_queue usa el mismo trains_learner,
    # asi que una marca 'execution' o ruido-legacy no debe aparecer en la cola.
    from conftest import insert_match

    insert_match(conn, "A")
    _dec_queue(conn, 100, mark_type="execution")           # excluida
    _dec_queue(conn, 200, mark_type="decision")            # incluida
    _dec_queue(conn, 300, mark_type=None, note="falle el combo")  # ruido legacy: excluida
    conn.commit()
    ids_in = {it["id"] for it in _dash.query_review_queue(conn)["items"]}
    ids = {t: conn.execute("SELECT id FROM decisions WHERE game_time_ms=?",
                           (t,)).fetchone()[0] for t in (100, 200, 300)}
    assert ids[200] in ids_in            # decision-typed aparece en la cola
    assert ids[100] not in ids_in        # execution excluida por el gate
    assert ids[300] not in ids_in        # ruido legacy excluida por el gate
