"""Integration tests for the tempo detector (tempo_v1).

A "tempo" decision fires when the team had a fight you weren't part of, you were
NOT physically there, you weren't dead, and you COULD have reached it. These
tests build a 2-kill fight in mid while we sit in bot and assert that contract,
plus the two guards the player explicitly asked for: don't flag a fight we were
in, and don't flag one we couldn't have reached in time.
"""
from __future__ import annotations

from lol_coach.decisions import tempo as T
from lol_coach.decisions.features import taken_index

from conftest import insert_event, ten_player_match

FIGHT_START = 510_000        # 8:30 — inside the conftest frame window (480k/540k/600k)
FIGHT_END = 522_000          # 12s later: long enough that the rotation was reachable


def _fight_in_mid_without_us(conn, match_id, *, us_participates=False):
    """Enemy jungler (8) kills our top (2) and mid (3) in the mid river. We (1)
    are in bot the whole time. With us_participates, we get credited an assist so
    the detector should bail."""
    insert_event(conn, match_id, FIGHT_START, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 2,
        "assistingParticipantIds": [1] if us_participates else [],
        "position": {"x": 7500, "y": 7500},
    })
    insert_event(conn, match_id, FIGHT_END, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 3, "assistingParticipantIds": [],
        "position": {"x": 7600, "y": 7600},
    })


def test_fight_without_us_that_we_could_reach_is_flagged(conn):
    ids = ten_player_match(conn)
    _fight_in_mid_without_us(conn, ids["match_id"])

    decisions = T.analyze_tempo(conn, ids["match_id"])

    assert len(decisions) == 1
    d = decisions[0]
    assert d.detector_id == "tempo_v1"
    assert d.game_time_ms == FIGHT_START
    assert d.context["could_arrive_in_time"] is True
    assert d.context["fight_outcome_for_us"] == "0-2"     # both our players died


def test_tempo_decision_has_wellformed_action_and_state(conn):
    ids = ten_player_match(conn)
    _fight_in_mid_without_us(conn, ids["match_id"])

    d = T.analyze_tempo(conn, ids["match_id"])[0]

    assert taken_index(d.options) is not None
    action = d.context["action"]
    assert action["detector_id"] == "tempo_v1"
    assert action["available_actions"][0]["action_id"] == "roam"   # option 0 is "rotate"
    assert set(d.context["state_features"]) == {"power", "info_risk", "wave_tempo", "objectives"}


def test_fight_we_participated_in_is_not_a_tempo_problem(conn):
    ids = ten_player_match(conn)
    _fight_in_mid_without_us(conn, ids["match_id"], us_participates=True)

    assert T.analyze_tempo(conn, ids["match_id"]) == []


def test_unreachable_fight_is_not_flagged(conn):
    ids = ten_player_match(conn)
    # Two instantaneous kills (0s duration) far in the enemy top corner: the
    # travel time dwarfs the fight, so could_arrive is False -> no decision.
    insert_event(conn, ids["match_id"], FIGHT_START, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 2, "assistingParticipantIds": [],
        "position": {"x": 2000, "y": 13000},
    })
    insert_event(conn, ids["match_id"], FIGHT_START, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 3, "assistingParticipantIds": [],
        "position": {"x": 2100, "y": 13100},
    })

    assert T.analyze_tempo(conn, ids["match_id"]) == []


def test_single_kill_is_not_a_fight(conn):
    ids = ten_player_match(conn)
    insert_event(conn, ids["match_id"], FIGHT_START, "CHAMPION_KILL", {
        "killerId": 8, "victimId": 2, "assistingParticipantIds": [],
        "position": {"x": 7500, "y": 7500},
    })

    assert T.analyze_tempo(conn, ids["match_id"]) == []
