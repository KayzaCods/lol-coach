"""Pure-logic tests for trade_v1: drive _evaluate with a hand-built TradeFacts,
no timeline JSON / DB. The minute-scan I/O stays in analyze_trades /
_gather_trade_facts; test_trade.py keeps covering it end-to-end."""
from __future__ import annotations

from lol_coach.decisions.trade import TradeFacts, _evaluate
from lol_coach.decisions.features import taken_index


def _facts(**over) -> TradeFacts:
    base = dict(
        match_id="M", t_ms=540_000, m_start=9, m_end=9, positive=False,
        d_us=600, d_en=100, hp_pct=None, mana_pct=None, consequence=None,
        state_features={"power": {}, "info_risk": {}, "wave_tempo": {}, "objectives": {}},
    )
    base.update(over)
    return TradeFacts(**base)


def test_evaluate_lost_trade():
    d = _evaluate(_facts())
    assert d.detector_id == "trade_v1"
    assert d.context["positive_trade"] is False
    assert d.context["duo_damage_taken"] == 600
    assert "perdido" in d.moment.lower()
    assert taken_index(d.options) is not None


def test_evaluate_won_trade_is_positive():
    d = _evaluate(_facts(positive=True, d_us=100, d_en=600))
    assert d.context["positive_trade"] is True
    assert "ganado" in d.moment.lower()


def test_evaluate_lost_trade_appends_the_consequence():
    d = _evaluate(_facts(consequence="muerte de tu duo en los 90s siguientes"))
    assert d.context["consequence"] == "muerte de tu duo en los 90s siguientes"
    # the "what you did" option (index 1) carries the consequence, capitalized
    assert "Muerte de tu duo" in d.options[1].predicted_consequence
    assert "La consecuencia llego" in d.argument or "La consecuencia lleg" in d.argument


def test_evaluate_lost_trade_mana_dump_is_reasoned_by_mana_not_life():
    # Small life diff (< BAD_MIN_DIFF) but dumped mana: the reason is mana, not life.
    d = _evaluate(_facts(d_us=200, d_en=150, mana_pct=0.05))
    assert "sin man" in d.options[1].predicted_consequence.lower()
    assert "intercambio de vida" not in d.options[1].predicted_consequence.lower()


def test_evaluate_takes_no_conn():
    import inspect
    assert list(inspect.signature(_evaluate).parameters) == ["facts"]
