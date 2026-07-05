from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mundial.tournament_state import (
    TournamentMatch, TournamentState, approve_candidate, load_tournament_state,
    parse_fifa_payload, validate_state, write_state,
)


def match(match_id="M1", **changes):
    values = dict(
        match_id=match_id, phase="group", group="A", kickoff="2026-06-11T12:00:00Z",
        status="finished", team_a="Mexico", team_b="South Africa", score_90=(2, 1), winner="Mexico",
    )
    values.update(changes)
    return TournamentMatch(**values)


def state(*matches):
    return TournamentState("2026-06-12T00:00:00Z", "https://fifa.example", None, {m.match_id: m for m in matches})


def test_parse_fifa_payload_and_hash_round_trip(tmp_path: Path):
    parsed = parse_fifa_payload({"matches": [match().to_dict()]}, source_url="https://fifa.example")
    path = tmp_path / "state.json"
    write_state(parsed, path)
    loaded = load_tournament_state(path, required=True)
    assert loaded.matches["M1"].score_90 == (2, 1)
    assert json.loads(path.read_text())["hash"]


def test_future_result_and_wrong_winner_are_rejected():
    future = match(kickoff="2027-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="futuro"):
        validate_state(state(future), now=datetime(2026, 1, 1, tzinfo=timezone.utc), require_complete=False)
    with pytest.raises(ValueError, match="ganador"):
        validate_state(state(match(winner="South Africa")), require_complete=False)


def test_approved_result_cannot_change_or_regress():
    approved = state(match())
    changed = state(match(score_90=(3, 1)))
    with pytest.raises(ValueError, match="no puede retroceder ni cambiar"):
        validate_state(changed, approved, require_complete=False)


def test_dependency_and_eliminated_reentry_are_rejected():
    first = match("M73", phase="round_of_32", group=None, team_a="A", team_b="B", winner="A")
    next_match = match(
        "M89", phase="round_of_16", group=None, team_a="B", team_b="C",
        status="scheduled", score_90=None, winner=None, source_a="winner M73",
    )
    with pytest.raises(ValueError, match="dependencia|eliminado"):
        validate_state(state(first, next_match), require_complete=False)


def test_approval_keeps_backup_and_promotes(tmp_path: Path, monkeypatch):
    candidate_path, production_path = tmp_path / "candidate.json", tmp_path / "state.json"
    write_state(state(match()), candidate_path)
    monkeypatch.setattr("mundial.tournament_state.validate_state", lambda *args, **kwargs: None)
    promoted = approve_candidate(candidate_path, production_path)
    assert promoted.approved_at
    assert load_tournament_state(production_path, required=True).matches["M1"].winner == "Mexico"
