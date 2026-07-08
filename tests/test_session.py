"""Pure logic for the 'Sesión de hoy' prioritized queue (#16): bucket
classification, quota fill with dedupe/backfill, and the temporal clustering
note. No DB — the I/O layer feeds flat candidate dicts."""
from __future__ import annotations

from lol_coach.session import (
    QUOTAS,
    SESSION_SIZE,
    SUCCESS_MIN_MARGIN,
    build_session,
    classify,
    temporal_note,
)


def _c(cid, detector="death_v1", action_class=None, power_index=None,
       ev_taken=None, ev_optimal=None, ev_min=None, start=0, t=0):
    gap = (round(ev_optimal - ev_taken, 2)
           if ev_taken is not None and ev_optimal is not None else None)
    return {"id": cid, "detector": detector, "action_class": action_class,
            "power_index": power_index, "ev_taken": ev_taken,
            "ev_optimal": ev_optimal, "ev_min": ev_min, "ev_gap": gap,
            "game_start_ms": start, "game_time_ms": t}


# ------------------------------------------------------------------- classify
def test_dominant_needs_overext_class_and_lead():
    assert classify(_c(1, action_class="engage_blind", power_index=0.7,
                       ev_taken=0.4, ev_optimal=0.7, ev_min=0.4)) == "dominant"
    # sin lead -> no es el patron dominante
    assert classify(_c(2, action_class="engage_blind", power_index=0.4,
                       ev_taken=0.4, ev_optimal=0.7, ev_min=0.4)) == "other"
    # clase con leccion opuesta -> no dominante aunque haya lead
    assert classify(_c(3, action_class="engage_worth", power_index=0.9,
                       ev_taken=0.4, ev_optimal=0.7, ev_min=0.4)) == "other"


def test_objective_needs_positive_gap():
    assert classify(_c(4, detector="objective_readiness_v1",
                       ev_taken=0.4, ev_optimal=0.8, ev_min=0.3)) == "objective"
    # gap 0 con margen -> acierto (confirm-once tipico)
    assert classify(_c(5, detector="objective_readiness_v1",
                       ev_taken=0.8, ev_optimal=0.8, ev_min=0.3)) == "success"


def test_success_needs_margin():
    # hiciste lo optimo y la alternativa era mucho peor -> la decision importaba
    assert classify(_c(6, detector="trade_v1",
                       ev_taken=0.78, ev_optimal=0.78, ev_min=0.45)) == "success"
    # margen chico: acierto trivial, no refuerza nada
    assert classify(_c(7, detector="trade_v1",
                       ev_taken=0.78, ev_optimal=0.78,
                       ev_min=0.78 - SUCCESS_MIN_MARGIN + 0.05)) == "other"


# -------------------------------------------------------------- build_session
def _pool():
    doms = [_c(i, action_class="engage_blind", power_index=0.7,
               ev_taken=0.5 - g, ev_optimal=0.5, ev_min=0.1, start=i)
            for i, g in [(1, 0.4), (2, 0.3), (3, 0.5), (4, 0.2), (5, 0.1)]]
    objs = [_c(10, detector="objective_readiness_v1", ev_taken=0.5,
               ev_optimal=0.8, ev_min=0.3, start=10),
            _c(11, detector="objective_readiness_v1", ev_taken=0.3,
               ev_optimal=0.8, ev_min=0.3, start=11)]
    succ = [_c(20, detector="trade_v1", ev_taken=0.8, ev_optimal=0.8,
               ev_min=0.3, start=20),
            _c(21, detector="trade_v1", ev_taken=0.8, ev_optimal=0.8,
               ev_min=0.5, start=21)]
    other = [_c(30, detector="tempo_v1", ev_taken=0.5, ev_optimal=0.6,
                ev_min=0.5, start=30),
             _c(31, detector="tempo_v1", ev_taken=0.5, ev_optimal=0.6,
                ev_min=0.5, start=31)]
    return doms + objs + succ + other


def test_session_respects_quotas_and_ordering():
    picked = build_session(_pool())
    assert len(picked) == SESSION_SIZE
    by_bucket = {}
    for m in picked:
        by_bucket.setdefault(m["bucket"], []).append(m["id"])
    # 3 dominantes con peor gap primero: 0.5 (id 3), 0.4 (id 1), 0.3 (id 2)
    assert by_bucket["dominant"] == [3, 1, 2]
    assert by_bucket["objective"] == [11]      # gap 0.5 > 0.3
    assert by_bucket["success"] == [20]        # margen 0.5 > 0.3
    assert by_bucket["other"] == [31]          # el mas reciente
    assert all(m.get("reason") for m in picked)


def test_session_backfills_from_leftovers():
    # Solo dominantes: cuota 3 + backfill hasta agotar (5 disponibles)
    doms = [c for c in _pool() if classify(c) == "dominant"]
    picked = build_session(doms)
    assert len(picked) == 5
    assert all(m["bucket"] == "dominant" for m in picked)


def test_session_empty_pool():
    assert build_session([]) == []


# -------------------------------------------------------------- temporal_note
def test_temporal_note_detects_concentration():
    note = temporal_note([12, 13, 14, 20, 25])
    assert note == {"count": 3, "last": 5, "min_from": 12, "min_to": 14}


def test_temporal_note_none_when_spread():
    assert temporal_note([3, 9, 15, 21, 27]) is None


def test_temporal_note_none_with_too_few():
    assert temporal_note([12, 13]) is None


def test_temporal_note_only_last_five_count():
    # los 3 primeros (12-14) quedan fuera de la ventana de ultimos 5
    note = temporal_note([12, 13, 14, 20, 25, 27, 28, 29])
    assert note == {"count": 3, "last": 5, "min_from": 27, "min_to": 29}
