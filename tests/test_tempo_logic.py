"""Pure-logic tests for tempo_v1: drive _evaluate with a hand-built TempoFacts,
no SQLite (the I/O-dependent guards live in _gather_fight_facts; test_tempo.py
keeps covering them end-to-end)."""
from __future__ import annotations

from lol_coach.decisions.tempo import TempoFacts, _evaluate
from lol_coach.decisions.features import taken_index


def _facts(**over) -> TempoFacts:
    base = dict(
        match_id="M", fight_start_ms=510_000, duration_s=12.0,
        fight_x=7550, fight_y=7550, our_team=100, me_x=10000, me_y=4000,
        our_kills=0, our_deaths=2, our_champs=[], enemy_champs=["Elise"],
        distance_u=4314.0, travel_s=11.4, could_arrive=True,
        activity={"description": "muy poca actividad", "jungle_delta_in_minute": 0},
        objective=None, stay_action="push_wave",
        state_features={"power": {}, "info_risk": {}, "wave_tempo": {}, "objectives": {}},
    )
    base.update(over)
    return TempoFacts(**base)


def test_evaluate_lost_fight_we_could_reach():
    d = _evaluate(_facts())
    assert d.detector_id == "tempo_v1"
    assert d.context["could_arrive_in_time"] is True
    assert d.context["fight_outcome_for_us"] == "0-2"
    assert d.context["action"]["available_actions"][0]["action_id"] == "roam"
    assert taken_index(d.options) is not None


def test_evaluate_objective_at_stake_lifts_rotate():
    # A fight you missed with an objective falling to the enemy: rotating is worth
    # much more, and the rotate option names the objective.
    obj = {"taken_by": "enemigo", "subtype": "Dragon", "type": "DRAGON",
           "rel_to_fight_s": -3.0}
    d = _evaluate(_facts(objective=obj,
                         activity={"description": "actividad mixta (+10 CS)",
                                   "jungle_delta_in_minute": 0}))
    assert d.context["objective_at_stake"] == obj
    assert "Objetivo cercano" in d.options[0].predicted_consequence
    assert d.options[0].ev_score == 0.95  # 0.45 + could 0.20 + obj 0.20 + lost 0.10
    assert d.options[0].ev_score > d.options[1].ev_score


def test_evaluate_farming_through_a_lost_fight_lowers_stay():
    # Farming while a fight you could reach is lost -> staying is penalized.
    d = _evaluate(_facts(activity={"description": "farmeando lane (+30 CS)",
                                   "jungle_delta_in_minute": 0}))
    assert "farmeando" in d.options[1].predicted_consequence
    assert d.options[1].ev_score == 0.35  # 0.45 - 0.10 (farming while losing a fight)


def test_evaluate_dominant_win_without_you_makes_staying_fine():
    # Team won decisively without you: your absence didn't cost -> stay >= rotate.
    d = _evaluate(_facts(our_kills=3, our_deaths=0,
                         activity={"description": "actividad mixta (+10 CS)",
                                   "jungle_delta_in_minute": 0}))
    assert d.options[0].ev_score == 0.50  # rotate 0.45 + could 0.20 - dominant 0.15
    assert d.options[1].ev_score == 0.60  # stay 0.45 + dominant 0.15
    assert d.options[1].ev_score > d.options[0].ev_score


def test_evaluate_takes_no_conn():
    import inspect
    assert list(inspect.signature(_evaluate).parameters) == ["facts"]
