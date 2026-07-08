"""Pure-logic tests for death_v1: drive _evaluate with a hand-built DeathFacts,
no SQLite. This is the payoff of separating I/O from logic — the classification
branches are testable without synthesizing a whole match (test_death.py keeps
covering the end-to-end I/O path)."""
from __future__ import annotations

from lol_coach.decisions.death import DeathFacts, _evaluate
from lol_coach.decisions.features import taken_index


def _facts(**over) -> DeathFacts:
    """A DeathFacts with sane defaults (an engage_blind bot death); override per test."""
    base = dict(
        match_id="M", death_ms=600_000, death_pos=(10000, 4000),
        lane="bot lane", map_side="nuestro lado", killer="Elise",
        assisters=[], first_blood=False,
        our_state={"level": 11, "current_gold": 500, "total_gold": 8000, "cs": 100,
                   "kills_so_far": 1, "deaths_so_far": 2, "assists_so_far": 3},
        nearest_ally_champ=None, nearest_ally_dist=None, we_had_help=False,
        fighters_for=1, fighters_against=1,
        score_our_kills=2, score_enemy_kills=3,
        non_combatant_unseen=[], combatant_unseen=[],
        retreating=False, got_value=False, enemy_deaths=0, ally_deaths=0,
        ally_died_first=False, sacrifice_obj=None, top_damager=None,
        state_features={"power": {}, "info_risk": {}, "wave_tempo": {}, "objectives": {}},
        clip_path=None, clip_offset_s=None,
    )
    base.update(over)
    return DeathFacts(**base)


def test_evaluate_engage_blind_is_default_classification():
    d = _evaluate(_facts())
    assert d.detector_id == "death_v1"
    assert d.context["inferred_action_class"] == "engage_blind"
    assert taken_index(d.options) is not None


def test_evaluate_calculated_sacrifice_when_objective_bought():
    d = _evaluate(_facts(sacrifice_obj={"name": "la torre", "rel_s": -1.0}))
    assert d.context["inferred_action_class"] == "calculated_sacrifice"
    assert any("objetivo" in o.predicted_consequence.lower() or "torre" in o.label.lower()
               for o in d.options)


def test_evaluate_engage_worth_when_personal_value():
    d = _evaluate(_facts(got_value=True))
    assert d.context["inferred_action_class"] == "engage_worth"


def test_evaluate_engage_worth_when_team_won_the_local_fight():
    d = _evaluate(_facts(enemy_deaths=2, ally_deaths=0))
    assert d.context["inferred_action_class"] == "engage_worth"


def test_evaluate_support_lost_when_ally_fell_first():
    d = _evaluate(_facts(ally_died_first=True, ally_deaths=1))
    assert d.context["inferred_action_class"] == "support_lost"


def test_evaluate_trade_lost_on_two_sided_skirmish():
    d = _evaluate(_facts(ally_deaths=1))
    assert d.context["inferred_action_class"] == "trade_lost"


def test_evaluate_disengage_failed_when_retreating():
    d = _evaluate(_facts(retreating=True))
    assert d.context["inferred_action_class"] == "disengage_failed"


def test_evaluate_fog_gank_adds_niebla_line():
    d = _evaluate(_facts(combatant_unseen=[("Elise", 25.0)]))
    assert "desde la niebla" in d.argument
    assert d.context["combatant_enemies_unseen"] == [{"champion": "Elise", "unseen_s": 25}]


def test_evaluate_takes_no_conn():
    # Structural guard: _evaluate's only parameter is `facts` — it never touches a DB.
    import inspect
    assert list(inspect.signature(_evaluate).parameters) == ["facts"]
