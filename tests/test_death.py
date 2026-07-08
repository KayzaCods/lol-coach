"""Integration tests for the death detector (death_v1).

These drive `analyze_deaths` end-to-end over a synthetic match and assert the
contract the rest of the pipeline depends on (one Decision per own-death, a
well-formed action block, exactly one taken option, a complete state_features),
plus the Ciclo 10 pre-fight visibility behavior: a killer who came out of fog
must surface in `combatant_enemies_unseen` and the "desde la niebla" argument
line — and must NOT when it was seen shortly before the commit.
"""
from __future__ import annotations

from lol_coach.decisions import death as D
from lol_coach.decisions.features import taken_index

from conftest import insert_event, ten_player_match, DEATH_MS


def _fog_gank_kill(conn, match_id):
    """A CHAMPION_KILL of our player by the enemy jungler (+ bot assist), with no
    prior visible events for either — i.e. they arrive from the fog of war."""
    insert_event(conn, match_id, DEATH_MS, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 1, "assistingParticipantIds": [6],
        "position": {"x": 10000, "y": 4000},
    })


def test_analyze_deaths_returns_one_decision_per_own_death(conn):
    ids = ten_player_match(conn)
    _fog_gank_kill(conn, ids["match_id"])

    decisions = D.analyze_deaths(conn, ids["match_id"])

    assert len(decisions) == 1
    d = decisions[0]
    assert d.detector_id == "death_v1"
    assert d.game_time_ms == DEATH_MS
    assert d.match_id == ids["match_id"]


def test_death_decision_has_wellformed_action_and_single_taken_option(conn):
    ids = ten_player_match(conn)
    _fog_gank_kill(conn, ids["match_id"])

    d = D.analyze_deaths(conn, ids["match_id"])[0]

    # Exactly one option is flagged as the one the player took.
    assert taken_index(d.options) is not None
    taken = [o for o in d.options if "hiciste" in o.label.lower()]
    assert len(taken) == 1

    action = d.context["action"]
    assert action["detector_id"] == "death_v1"
    assert action["action_id"] in {"fight_entry", "disengage"}
    assert len(action["available_actions"]) == len(d.options)


def test_death_decision_has_complete_state_features(conn):
    ids = ten_player_match(conn)
    _fog_gank_kill(conn, ids["match_id"])

    d = D.analyze_deaths(conn, ids["match_id"])[0]
    sf = d.context["state_features"]

    assert set(sf) == {"power", "info_risk", "wave_tempo", "objectives"}
    # The fog gank means 2 combatants were unseen at decision time, so the info
    # term's enemies_unseen must reflect that (drives the info penalty downstream).
    assert sf["info_risk"]["enemies_unseen"] == 2


def test_fog_gank_surfaces_combatant_unseen_and_niebla_line(conn):
    ids = ten_player_match(conn)
    _fog_gank_kill(conn, ids["match_id"])

    d = D.analyze_deaths(conn, ids["match_id"])[0]

    unseen = d.context["combatant_enemies_unseen"]
    champs = {e["champion"] for e in unseen}
    assert champs == {"Elise", "Caitlyn"}        # killer + assister, both from fog
    assert "desde la niebla" in d.argument


def test_seen_killer_is_not_counted_as_fog_gank(conn):
    ids = ten_player_match(conn)
    match_id = ids["match_id"]
    # Killer (8) and assister (6) are both involved in a visible kill 15s before
    # the death — within the pre-fight window — so they were NOT unseen at commit
    # time. (Placed far away so it is a different fight for the outcome counting.)
    insert_event(conn, match_id, DEATH_MS - 15_000, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 2, "assistingParticipantIds": [6],
        "position": {"x": 3000, "y": 12000},
    })
    _fog_gank_kill(conn, match_id)

    d = D.analyze_deaths(conn, match_id)[0]

    assert d.context["combatant_enemies_unseen"] == []
    assert "desde la niebla" not in d.argument


def test_kill_where_we_are_not_victim_yields_no_decision(conn):
    ids = ten_player_match(conn)
    # An enemy dies; we are not the victim -> the death detector ignores it.
    insert_event(conn, ids["match_id"], DEATH_MS, "CHAMPION_KILL", {
        "killerId": 1, "victimId": 8, "assistingParticipantIds": [],
        "position": {"x": 10000, "y": 4000},
    })

    assert D.analyze_deaths(conn, ids["match_id"]) == []
