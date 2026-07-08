"""Characterization tests for the decision-evaluation criterion (features.py).

These pin the CURRENT behavior of the pure scoring functions — the HEF terms,
_commit_fit, the penalties, and the fingerprint/option helpers. They are the
safety net for the formula: a refactor (or an accidental tweak to a constant)
that changes a score now fails loudly instead of silently shifting every
decision's grade. Where Ciclo 10 made an intentional choice (alpha 0.3/0.7,
the wave inversion at objectives, dict-or-Option taken_index, the fingerprint
format) there is an explicit test asserting THAT choice, so reverting it breaks
a named test rather than a numeric one.
"""
from __future__ import annotations

import pytest

from lol_coach.decisions import features as F
from lol_coach.decisions.base import Option


# --------------------------------------------------------------- power_index
def test_power_index_none_and_empty_return_none():
    assert F.power_index(None) is None
    assert F.power_index({}) is None


def test_power_index_even_state_is_half():
    power = {
        "local_level_diff": 0, "local_gold_diff": 0, "entry_hp_pct": 0.5,
        "ally_count_nearby": 1, "enemy_count_nearby": 1,
    }
    assert F.power_index(power) == 0.5


def test_power_index_max_advantage_is_one():
    power = {
        "local_level_diff": 3, "local_gold_diff": 1500, "entry_hp_pct": 1.0,
        "ally_count_nearby": 2, "enemy_count_nearby": 0,
    }
    assert F.power_index(power) == 1.0


def test_power_index_max_disadvantage_is_zero():
    power = {
        "local_level_diff": -3, "local_gold_diff": -1500, "entry_hp_pct": 0.0,
        "ally_count_nearby": 0, "enemy_count_nearby": 2,
    }
    assert F.power_index(power) == 0.0


def test_power_index_headcount_only_shrinks_toward_half_not_blowout():
    # Only the numbers component present (2v0): a MISSING component counts as
    # neutral, so the score must shrink toward 0.5 — NOT read 1.0 the way a raw
    # head-count alone used to. (features.py _POWER_W docstring.)
    idx = F.power_index({"ally_count_nearby": 2, "enemy_count_nearby": 0})
    assert idx is not None
    assert 0.6 < idx < 0.7


@pytest.mark.parametrize("idx,label", [
    (0.20, "desventaja fuerte"),
    (0.40, "desventaja"),
    (0.50, "parejo"),
    (0.65, "ventaja"),
    (0.90, "ventaja fuerte"),
])
def test_power_label_buckets(idx, label):
    assert F.power_label(idx) == label


def test_power_label_none():
    assert F.power_label(None) is None


# ----------------------------------------------------------------- _info_term
def test_info_term_empty_is_none():
    assert F._info_term({}) is None


@pytest.mark.parametrize("unseen,expected", [(0, 1.0), (5, 0.0), (2, 0.6)])
def test_info_term_enemies_unseen_monotone(unseen, expected):
    assert F._info_term({"enemies_unseen": unseen}) == pytest.approx(expected)


def test_info_term_wards_floor_and_ceiling():
    assert F._info_term({"wards_nearby": 0}) == pytest.approx(0.4)
    assert F._info_term({"wards_nearby": 2}) == pytest.approx(1.0)


def test_info_term_averages_present_parts():
    # enemies_unseen=0 -> 1.0 ; wards_nearby=0 -> 0.4 ; mean = 0.7
    assert F._info_term({"enemies_unseen": 0, "wards_nearby": 0}) == pytest.approx(0.7)


# ----------------------------------------------------------------- _wave_term
def test_wave_term_unknown_state_is_none():
    assert F._wave_term({}) is None
    assert F._wave_term({"wave_state": "???"}) is None


def test_wave_term_inverts_between_deaths_and_objectives():
    # The exact-inverse property is an explicit design choice (player feedback):
    # a wave shoved into the enemy is EXPOSURE in a fight but PRESSURE at an
    # objective; frozen at your tower is SAFETY in a fight but zero pressure at
    # an objective.
    pushed = {"wave_state": "pushed_into_enemy"}
    frozen = {"wave_state": "frozen_near_tower"}
    assert F._wave_term(pushed, at_objective=False) == 0.30
    assert F._wave_term(pushed, at_objective=True) == 0.80
    assert F._wave_term(frozen, at_objective=False) == 0.75
    assert F._wave_term(frozen, at_objective=True) == 0.30


# ---------------------------------------------------------------- _commit_fit
def test_commit_fit_none_action_is_none():
    assert F._commit_fit(None, 0.5, 0.5) is None


def test_commit_fit_no_terms_is_none():
    assert F._commit_fit("fight_entry", None, None) is None


def test_commit_fit_setup_blend_is_03_info_07_power():
    # Ciclo 10 recalibration: setup = 0.3*info + 0.7*power (was 0.6/0.4).
    # High info + zero power must read as LOW fit for an aggressive commit.
    assert F._commit_fit("fight_entry", 0.0, 1.0) == pytest.approx(0.3)
    assert F._commit_fit("fight_entry", 1.0, 0.0) == pytest.approx(0.7)


def test_commit_fit_objective_value_scales_aggressive_commit():
    # setup held at 0.5 (pi=it=0.5): ov 0 -> x0.7, ov 1 -> x1.3
    assert F._commit_fit("fight_entry", 0.5, 0.5, {"objective_value": 0.0}) == pytest.approx(0.35)
    assert F._commit_fit("fight_entry", 0.5, 0.5, {"objective_value": 1.0}) == pytest.approx(0.65)


def test_commit_fit_defensive_is_inverse_of_setup():
    assert F._commit_fit("disengage", 1.0, 1.0) == pytest.approx(0.0)
    assert F._commit_fit("disengage", 0.0, 0.0) == pytest.approx(1.0)


def test_commit_fit_ignore_objective_worse_when_valuable_and_setup():
    # base = 1 - setup, minus 0.4*ov*setup. setup=0.5, ov=1 -> 0.5 - 0.2 = 0.3
    assert F._commit_fit("ignore_objective", 0.5, 0.5, {"objective_value": 1.0}) == pytest.approx(0.3)
    # with no value, just 1 - setup
    assert F._commit_fit("ignore_objective", 0.5, 0.5) == pytest.approx(0.5)


def test_commit_fit_vision_best_when_setup_low():
    assert F._commit_fit("vision_setup", 0.0, 0.0) == pytest.approx(1.0)
    assert F._commit_fit("vision_setup", 1.0, 1.0) == pytest.approx(0.6)


def test_commit_fit_unclassified_vocab_action_is_neutral():
    assert F._commit_fit("farm_jungle", 0.5, 0.5) == 0.5


# ------------------------------------------------------------------ hef_score
def test_hef_score_empty_state_is_null_score():
    out = F.hef_score({}, action_id="fight_entry")
    assert out["score"] is None
    assert out["terms"] == {}


def test_hef_score_power_only_passthrough():
    # Only a power term present, no action -> score == power_index, no penalties.
    sf = {"power": {"local_level_diff": 0, "local_gold_diff": 0, "entry_hp_pct": 0.5,
                    "ally_count_nearby": 1, "enemy_count_nearby": 1}}
    out = F.hef_score(sf, action_id=None)
    assert out["terms"] == {"power": 0.5}
    assert out["score"] == 0.5
    assert out["penalties"] == {}


def test_hef_score_blind_lowinfo_fight_entry_collapses_to_zero():
    # The player's dominant failure mode: aggressive commit, blind (5 unseen),
    # no setup. Both penalties fire and drive the score to 0.0.
    sf = {
        "power": {"local_level_diff": 0, "local_gold_diff": 0, "entry_hp_pct": 0.5,
                  "ally_count_nearby": 1, "enemy_count_nearby": 1},   # pi = 0.5
        "info_risk": {"enemies_unseen": 5},                            # it = 0.0
        "wave_tempo": {},
        "objectives": {},
    }
    out = F.hef_score(sf, action_id="fight_entry")
    assert out["base"] == pytest.approx(0.29, abs=0.01)
    assert "pelea_a_ciegas" in out["penalties"]
    assert out["penalties"]["pelea_a_ciegas"] == 0.30
    assert out["penalties"]["commit_sin_setup"] == pytest.approx(0.40)
    assert out["score"] == 0.0


def test_hef_score_hp_penalty_is_progressive():
    # entry_hp_pct 0.25 with a non-aggressive action isolates hp_bajo:
    # (0.45 - 0.25) * 0.6 = 0.12
    sf = {"power": {"entry_hp_pct": 0.25, "ally_count_nearby": 1, "enemy_count_nearby": 1,
                    "local_level_diff": 0, "local_gold_diff": 0}}
    out = F.hef_score(sf, action_id="disengage")
    assert out["penalties"]["hp_bajo"] == pytest.approx(0.12)
    assert "pelea_a_ciegas" not in out["penalties"]


def test_hef_score_objective_wave_reads_as_pressure():
    # At an objective (next_major_obj set AND time_to_obj_s == 0) a pushed wave
    # must be scored high (pressure), proving the at_objective branch is taken.
    sf = {
        "wave_tempo": {"wave_state": "pushed_into_enemy"},
        "objectives": {"next_major_obj": "dragon", "time_to_obj_s": 0},
    }
    out = F.hef_score(sf, action_id=None)
    assert out["terms"]["wave"] == 0.80


def test_hef_score_learned_weights_change_score_not_terms():
    sf = {"power": {"local_level_diff": 3, "local_gold_diff": 1500, "entry_hp_pct": 1.0,
                    "ally_count_nearby": 2, "enemy_count_nearby": 0},   # pi = 1.0
          "info_risk": {"enemies_unseen": 0}}                            # it = 1.0
    default = F.hef_score(sf, action_id=None)
    weighted = F.hef_score(sf, action_id=None, weights={"power": 1.0, "info": 0.0,
                                                        "action_fit": 0.0, "wave": 0.0})
    # Same underlying terms; only the weighting (hence base) differs.
    assert default["terms"] == weighted["terms"]
    assert weighted["weights"]["power"] == 1.0


@pytest.mark.parametrize("score,label", [
    (0.20, "decisión pobre"),
    (0.40, "cuestionable"),
    (0.55, "aceptable"),
    (0.70, "buena"),
    (0.90, "excelente"),
])
def test_hef_label_buckets(score, label):
    assert F.hef_label(score) == label


def test_hef_label_none():
    assert F.hef_label(None) is None


# ----------------------------------------------------------------- taken_index
def test_taken_index_with_option_objects():
    opts = [
        Option(label="Disengage", predicted_consequence="", ev_score=0.4),
        Option(label="Entraste (lo que hiciste)", predicted_consequence="", ev_score=0.3),
    ]
    assert F.taken_index(opts) == 1


def test_taken_index_with_plain_dicts():
    # The dashboard/backfill path passes parsed options_json (dicts).
    opts = [{"label": "estado real: x"}, {"label": "otra"}]
    assert F.taken_index(opts) == 0


def test_taken_index_no_marker_returns_none():
    assert F.taken_index([{"label": "a"}, {"label": "b"}]) is None


# ----------------------------------------------------------- action_fingerprint
def test_action_fingerprint_format_and_taken_index():
    action = {"available_actions": [{"action_id": "fight_entry"}, {"action_id": "disengage"}]}
    options = [{"label": "Entrar"}, {"label": "Retirarte (lo que hiciste)"}]
    assert F.action_fingerprint("death_v1", action, options) == "v1|death_v1|fight_entry,disengage|t1"


def test_action_fingerprint_unknown_taken_marked_with_question():
    action = {"available_actions": [{"action_id": "fight_entry"}, {"action_id": "disengage"}]}
    assert F.action_fingerprint("death_v1", action, [{"label": "a"}, {"label": "b"}]) == \
        "v1|death_v1|fight_entry,disengage|t?"


def test_action_fingerprint_none_when_no_action_or_actions():
    assert F.action_fingerprint("death_v1", None, []) is None
    assert F.action_fingerprint("death_v1", {"available_actions": []}, [{"label": "x"}]) is None


# ------------------------------------------------------------- tower distances
def test_nearest_tower_dist_none_inputs():
    assert F.nearest_ally_tower_dist(None, None, 100) is None
    assert F.nearest_ally_tower_dist(981, 10441, 999) is None  # unknown team


def test_nearest_ally_tower_dist_zero_at_a_tower():
    # (981, 10441) is a blue-side (team 100) tower coordinate.
    assert F.nearest_ally_tower_dist(981, 10441, 100) == 0


def test_nearest_enemy_tower_dist_uses_opposite_team():
    # For team 100 the enemy towers are team 200's; (4318, 13875) is one of them.
    assert F.nearest_enemy_tower_dist(4318, 13875, 100) == 0


# ------------------------------------------------------------------- obj_kind
@pytest.mark.parametrize("monster,kind", [
    ("DRAGON", "dragon"),
    ("RIFTHERALD", "herald"),
    ("BARON_NASHOR", "baron"),
    ("ATAKHAN", None),
    (None, None),
])
def test_obj_kind(monster, kind):
    assert F.obj_kind(monster) == kind
