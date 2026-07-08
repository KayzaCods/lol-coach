"""Integration tests for the objective-readiness detector (objective_readiness_v1).

Builds an epic-monster kill (a dragon taken by our team, at the pit, with vision
and allies in zone) and asserts the contract plus the detector's signature
behavior: when you contested with setup and it was YOUR objective, the taken
action is `fight_entry` ("lo correcto", confirm-once). Also guards the skips:
natural despawn and non-eval monsters produce no decision.
"""
from __future__ import annotations

from lol_coach.decisions import objective_readiness as O
from lol_coach.decisions.features import taken_index

from conftest import insert_event, ten_player_match

KILL_TS = 540_000                 # minute 9 — a conftest frame exists here
PIT = {"x": 9800, "y": 4400}      # dragon pit, blue-side river (near our ADC at 10000,4000)


def _dragon_taken_by_us_with_setup(conn, match_id):
    # Our support (5) wards near the pit in the prep window -> team vision.
    insert_event(conn, match_id, KILL_TS - 40_000, "WARD_PLACED", {
        "creatorId": 5, "wardType": "YELLOW_TRINKET",
    })
    # Our jungler (4) secures an Infernal drake at the pit.
    insert_event(conn, match_id, KILL_TS, "ELITE_MONSTER_KILL", {
        "killerId": 4, "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON",
        "position": PIT,
    })


def test_objective_taken_by_us_yields_one_decision(conn):
    ids = ten_player_match(conn)
    _dragon_taken_by_us_with_setup(conn, ids["match_id"])

    decisions = O.analyze_objective_readiness(conn, ids["match_id"])

    assert len(decisions) == 1
    d = decisions[0]
    assert d.detector_id == "objective_readiness_v1"
    assert d.game_time_ms == KILL_TS
    assert d.context["objective"] == "Dragón Infernal"
    assert d.context["taken_by"] == "tu equipo"
    assert d.context["at_pit"] is True


def test_contested_with_setup_taken_action_is_fight_entry(conn):
    ids = ten_player_match(conn)
    _dragon_taken_by_us_with_setup(conn, ids["match_id"])

    d = O.analyze_objective_readiness(conn, ids["match_id"])[0]

    # Exactly one taken option, and with good setup on our own objective the
    # taken action maps to fight_entry (we contested) — the confirm-once branch.
    ti = taken_index(d.options)
    assert ti is not None
    assert d.context["action"]["available_actions"][ti]["action_id"] == "fight_entry"
    assert d.context["can_contest"] is True
    assert set(d.context["state_features"]) == {"power", "info_risk", "wave_tempo", "objectives"}
    # The objective block marks this as an at-objective state (drives the wave term).
    assert d.context["state_features"]["objectives"]["time_to_obj_s"] == 0
    assert d.context["state_features"]["objectives"]["next_major_obj"] == "dragon"


def test_natural_despawn_is_skipped(conn):
    ids = ten_player_match(conn)
    # killerId 0 + non-team killerTeamId = nobody took it (e.g. Herald despawn).
    insert_event(conn, ids["match_id"], KILL_TS, "ELITE_MONSTER_KILL", {
        "killerId": 0, "killerTeamId": 300, "monsterType": "RIFTHERALD",
        "position": PIT,
    })

    assert O.analyze_objective_readiness(conn, ids["match_id"]) == []


def test_non_eval_monster_is_skipped(conn):
    ids = ten_player_match(conn)
    # Voidgrubs/HORDE are not in EVAL_OBJECTIVES.
    insert_event(conn, ids["match_id"], KILL_TS, "ELITE_MONSTER_KILL", {
        "killerId": 4, "monsterType": "HORDE", "position": PIT,
    })

    assert O.analyze_objective_readiness(conn, ids["match_id"]) == []


def test_spectator_dead_far_from_pit_is_skipped(conn):
    ids = ten_player_match(conn)
    # We (pid 1) died 40s before the dragon, far from the pit (top lane).
    insert_event(conn, ids["match_id"], KILL_TS - 40_000, "CHAMPION_KILL", {
        "victimId": 1, "position": {"x": 3000, "y": 12000},
    })
    _dragon_taken_by_us_with_setup(conn, ids["match_id"])

    # Dead when it fell and didn't die contesting -> no objective decision.
    assert O.analyze_objective_readiness(conn, ids["match_id"]) == []


def test_contest_death_at_pit_still_yields_decision(conn):
    ids = ten_player_match(conn)
    # We died 10s before the dragon, AT the pit (contesting it).
    insert_event(conn, ids["match_id"], KILL_TS - 10_000, "CHAMPION_KILL", {
        "victimId": 1, "position": {"x": 9800, "y": 4400},
    })
    _dragon_taken_by_us_with_setup(conn, ids["match_id"])

    # Dead at the kill but died contesting -> decision is kept.
    assert len(O.analyze_objective_readiness(conn, ids["match_id"])) == 1
