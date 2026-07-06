import pytest

from mundial.config import load_groups
from mundial.prediction import DemoPredictor
from mundial.simulation import TournamentSimulator, validate_groups


def test_official_groups_are_valid():
    validate_groups(load_groups())


def test_duplicate_team_is_rejected():
    groups = load_groups()
    groups["B"][0] = groups["A"][0]
    with pytest.raises(ValueError, match="unicas"):
        validate_groups(groups)


def test_complete_tournament_has_expected_structure():
    result = TournamentSimulator(DemoPredictor()).simulate(load_groups(), runs=8, seed=7)
    assert len(result.bracket) == 32
    assert result.bracket[-1].match_id == "M104"
    assert result.metadata["group_matches_per_run"] == 72
    assert result.metadata["knockout_matches_per_run"] == 32
    assert sum(result.champion_probabilities.values()) == pytest.approx(1.0)
    qualification_sum = sum(
        row.qualification_probability
        for table in result.group_tables.values()
        for row in table
    )
    assert qualification_sum == pytest.approx(32.0)


def test_knockout_matches_report_three_way_probabilities():
    result = TournamentSimulator(DemoPredictor()).simulate(load_groups(), runs=2, seed=13)
    for match in result.bracket:
        assert match.match_probability_a + match.draw_probability + match.match_probability_b == pytest.approx(1.0)
        assert 0.0 <= match.match_probability_a <= 1.0
        assert 0.0 <= match.draw_probability <= 1.0
        assert 0.0 <= match.match_probability_b <= 1.0


def test_final_override_is_respected_with_same_seed():
    simulator = TournamentSimulator(DemoPredictor())
    baseline = simulator.simulate(load_groups(), runs=1, seed=11)
    final = baseline.bracket[-1]
    forced = final.team_b if final.winner == final.team_a else final.team_a
    result = simulator.simulate(load_groups(), overrides={"M104": forced}, runs=1, seed=11)
    assert result.bracket[-1].winner == forced
    assert result.bracket[-1].forced


def test_demo_predictions_are_symmetric_and_have_score_matrix():
    predictor = DemoPredictor({"A": 1700, "B": 1500})
    forward = predictor.predict_match("A", "B")
    reverse = predictor.predict_match("B", "A")
    assert forward.score_probabilities is not None
    assert forward.prob_a == pytest.approx(reverse.prob_b)
    assert forward.prob_draw == pytest.approx(reverse.prob_draw)
    assert forward.expected_goals_a == pytest.approx(reverse.expected_goals_b)


def test_simulation_reports_monte_carlo_intervals():
    result = TournamentSimulator(DemoPredictor()).simulate(load_groups(), runs=100, seed=9)
    intervals = result.metadata["champion_confidence_intervals_95"]
    assert set(intervals) == set(result.champion_probabilities)
    assert all(0.0 <= lower <= upper <= 1.0 for lower, upper in intervals.values())
