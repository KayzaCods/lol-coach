"""Pure-logic tests for objective_readiness_v1: drive _evaluate with a hand-built
ObjectiveFacts (the context dict + branch flags), no SQLite. The must_defend /
dominating / normal branching is the logic; test_objective_readiness.py keeps
covering the I/O end-to-end."""
from __future__ import annotations

from lol_coach.decisions.objective_readiness import ObjectiveFacts, _evaluate
from lol_coach.decisions.features import taken_index


def _ctx(**over):
    base = dict(
        time_mmss="9:00", objective="Dragon Infernal", objective_type="DRAGON",
        taken_by="tu equipo", your_position_at_kill_frame={"x": 10000, "y": 4000},
        distance_to_pit_units=447, at_pit=True, too_far=False,
        your_state_at_kill={"hp_pct": None, "mana_pct": None, "level": 11,
                            "items_in_inventory": None, "current_gold": 500, "total_gold": 8000},
        recall_in_prep_window=None, team_wards_near_pit=1, your_wards_near_pit=0,
        enemy_wards_near_pit=0, team_sweeps_near_pit=0, your_sweeps_near_pit=0,
        vision_diff=1, your_wards_detail=[], your_role_is_support=False,
        can_contest=True, own_inhibitors_lost=0, team_gold_diff=0,
        objective_dominated=False, objective_uncontested=False, allies_near_pit=2,
        traded_for=None, state_features={"power": {}, "info_risk": {}, "wave_tempo": {}, "objectives": {}},
    )
    base.update(over)
    return base


def _facts(ctx=None, **over) -> ObjectiveFacts:
    base = dict(
        match_id="M", context=_ctx(**(ctx or {})), taken_by_us=True, has_resource=True,
        is_support=False, must_defend=False, presence_optional=False,
        name="Dragon Infernal", kill_ts=540_000, distance_u=447.0,
        hp_pct=None, mana_pct=None, items_count=None,
    )
    base.update(over)
    return ObjectiveFacts(**base)


def _taken_action_id(d):
    """The action_id of the option the player actually took (HEF's taken slot)."""
    ti = taken_index(d.options)
    return d.context["action"]["available_actions"][ti]["action_id"]


def test_evaluate_taken_with_setup_action_is_fight_entry():
    d = _evaluate(_facts())
    assert d.detector_id == "objective_readiness_v1"
    ti = taken_index(d.options)
    assert ti is not None
    assert d.context["action"]["available_actions"][ti]["action_id"] == "fight_entry"


def test_evaluate_must_defend_offers_cede():
    d = _evaluate(_facts(
        ctx={"taken_by": "enemigo", "at_pit": False, "can_contest": False},
        taken_by_us=False, must_defend=True))
    assert any("ceder" in o.label.lower() for o in d.options)
    ti = taken_index(d.options)
    assert d.context["action"]["available_actions"][ti]["action_id"] == "ignore_objective"


# --- dominating / presence_optional: we took it and our presence wasn't decisive ---

def test_evaluate_dominating_controlled_accompany_is_valid():
    # Big lead, we took it: hovering with vision wasn't critical -> accompanying is
    # valid (the taken option sits in the vision_setup slot), with a slim margin in
    # favor of spending the time elsewhere.
    d = _evaluate(_facts(ctx={"objective_dominated": True}, presence_optional=True))
    assert _taken_action_id(d) == "vision_setup"
    assert "controlado" in d.options[0].label.lower()
    assert d.options[0].ev_score > d.options[1].ev_score


def test_evaluate_dominating_too_far_credits_time_elsewhere():
    # Uncontested objective and you were far: converting the freed time into
    # farm/reset/pressure was the right use (ignore_objective slot).
    d = _evaluate(_facts(
        ctx={"objective_uncontested": True, "too_far": True, "at_pit": False},
        presence_optional=True))
    assert _taken_action_id(d) == "ignore_objective"
    assert "tiempo" in d.options[0].label.lower()
    assert d.options[0].ev_score > d.options[1].ev_score


# --- normal contest: we took it, but with thin setup ---

def test_evaluate_taken_with_low_setup_flags_the_risk():
    # We got it, but without vision/numbers: "what you did" is fight_entry, the
    # recommended is to build setup first (vision_setup), and its EV reflects the risk.
    d = _evaluate(_facts(ctx={
        "team_wards_near_pit": 0, "team_sweeps_near_pit": 0, "your_wards_near_pit": 0,
        "your_sweeps_near_pit": 0, "vision_diff": 0, "allies_near_pit": 0,
        "at_pit": False}))
    assert _taken_action_id(d) == "fight_entry"
    assert d.context["action"]["available_actions"][0]["action_id"] == "vision_setup"
    assert "conseguiste" in d.options[1].label.lower()


# --- normal contest: enemy took it while you were at the pit (contested and lost) ---

def test_evaluate_contest_lost_with_setup_is_the_right_call():
    # You were there WITH setup and it fell: no resultadismo -> the call was right,
    # the loss is execution/coinflip.
    d = _evaluate(_facts(ctx={"taken_by": "enemigo", "at_pit": True}, taken_by_us=False))
    assert _taken_action_id(d) == "fight_entry"
    assert "se perd" in d.options[0].label.lower()


def test_evaluate_contest_lost_without_setup_forcing_was_the_mistake():
    d = _evaluate(_facts(ctx={
        "taken_by": "enemigo", "at_pit": True, "team_wards_near_pit": 0,
        "vision_diff": 0, "allies_near_pit": 0}, taken_by_us=False))
    assert _taken_action_id(d) == "fight_entry"
    assert d.context["action"]["available_actions"][0]["action_id"] == "ignore_objective"
    assert "forzaste" in d.options[1].label.lower()


# --- normal contest: enemy took it and you weren't there ---

def test_evaluate_objective_for_objective_trade():
    d = _evaluate(_facts(
        ctx={"taken_by": "enemigo", "at_pit": False, "too_far": True,
             "traded_for": "las larvas"},
        taken_by_us=False))
    assert _taken_action_id(d) == "ignore_objective"
    assert "larvas" in d.options[0].label.lower()


def test_evaluate_not_there_low_setup_ceding_was_right():
    d = _evaluate(_facts(ctx={
        "taken_by": "enemigo", "at_pit": False, "too_far": True,
        "team_wards_near_pit": 0, "vision_diff": 0, "allies_near_pit": 0},
        taken_by_us=False))
    assert _taken_action_id(d) == "ignore_objective"
    assert d.options[0].ev_score > d.options[1].ev_score  # cede/trade recommended


def test_evaluate_not_there_team_level_cede_when_nobody_showed():
    # Partial setup but no ally played it -> collective failure, not yours.
    d = _evaluate(_facts(ctx={
        "taken_by": "enemigo", "at_pit": False, "too_far": True,
        "team_wards_near_pit": 1, "team_sweeps_near_pit": 1, "vision_diff": 1,
        "allies_near_pit": 0}, taken_by_us=False))
    assert _taken_action_id(d) == "ignore_objective"
    assert d.context["action"]["available_actions"][0]["action_id"] == "vision_setup"
    assert "equipo" in d.options[1].label.lower()


def test_evaluate_not_there_your_presence_was_the_gap():
    # Setup AND allies in zone, but you didn't show -> your presence was missing.
    d = _evaluate(_facts(ctx={
        "taken_by": "enemigo", "at_pit": False, "too_far": True,
        "team_wards_near_pit": 1, "team_sweeps_near_pit": 1, "vision_diff": 1,
        "allies_near_pit": 2}, taken_by_us=False))
    assert _taken_action_id(d) == "ignore_objective"
    assert d.context["action"]["available_actions"][0]["action_id"] == "fight_entry"
    assert "presentarte" in d.options[0].label.lower()


def test_evaluate_takes_no_conn():
    import inspect
    assert list(inspect.signature(_evaluate).parameters) == ["facts"]
