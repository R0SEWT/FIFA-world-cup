from mundial.config import load_groups
from mundial.prediction import DemoPredictor
from mundial.simulation import TournamentSimulator
from mundial.tournament_state import TournamentMatch, TournamentState


def partial_state(*matches):
    return TournamentState("2026-07-01T00:00:00Z", "https://fifa.example", None, {m.match_id: m for m in matches})


def test_group_result_is_fixed_and_reproducible():
    fixed = TournamentMatch(
        "M1", "group", "2026-06-11T00:00:00Z", "finished",
        "Mexico", "South Africa", "A", (7, 0), winner="Mexico",
    )
    simulator = TournamentSimulator(DemoPredictor())
    a = simulator.simulate(load_groups(), runs=3, seed=4, tournament_state=partial_state(fixed))
    b = simulator.simulate(load_groups(), runs=3, seed=4, tournament_state=partial_state(fixed))
    mexico_a = next(row for row in a.group_tables["A"] if row.team == "Mexico")
    assert mexico_a.expected_goals_for >= 7
    assert a.champion_probabilities == b.champion_probabilities


def test_finished_knockout_is_locked_and_loser_has_zero_probability():
    fixed = TournamentMatch(
        "M73", "round_of_32", "2026-06-30T00:00:00Z", "finished",
        "Mexico", "Canada", None, (1, 0), winner="Mexico",
    )
    result = TournamentSimulator(DemoPredictor()).simulate(
        load_groups(), overrides={"M73": "Canada"}, runs=5, seed=8,
        tournament_state=partial_state(fixed),
    )
    official = next(match for match in result.bracket if match.match_id == "M73")
    assert official.winner == "Mexico"
    assert official.official and not official.forced
    assert result.champion_probabilities["Canada"] == 0.0
