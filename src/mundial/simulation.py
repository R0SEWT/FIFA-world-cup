"""Motor reproducible de grupos y eliminatorias para 48 selecciones."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Mapping

import numpy as np

from mundial.prediction import Predictor
from mundial.schemas import BracketMatch, GroupProjection, MatchPrediction, TournamentSimulation

if TYPE_CHECKING:
    from mundial.tournament_state import TournamentState

GROUPS = tuple("ABCDEFGHIJKL")
THIRD_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "A": ("C", "E", "F", "H", "I"),
    "B": ("E", "F", "G", "I", "J"),
    "D": ("B", "E", "F", "I", "J"),
    "E": ("A", "B", "C", "D", "F"),
    "G": ("A", "E", "H", "I", "J"),
    "I": ("C", "D", "F", "G", "H"),
    "K": ("D", "E", "I", "J", "L"),
    "L": ("E", "H", "I", "J", "K"),
}


@dataclass
class _Stats:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class _PlayedGame:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int


def validate_groups(groups: Mapping[str, list[str]]) -> None:
    """Valida la estructura que exige el formato 2026."""
    if set(groups) != set(GROUPS):
        raise ValueError("Deben existir exactamente los grupos A a L")
    teams = [team for group in GROUPS for team in groups[group]]
    if any(len(groups[group]) != 4 for group in GROUPS):
        raise ValueError("Cada grupo debe contener cuatro selecciones")
    if any(not team.strip() for team in teams):
        raise ValueError("Los nombres de las selecciones no pueden estar vacios")
    if len(set(teams)) != 48:
        raise ValueError("Las 48 selecciones deben ser unicas")


def _sample_score(prediction: MatchPrediction, rng: np.random.Generator) -> tuple[int, int]:
    if prediction.score_probabilities is None:
        raise ValueError("El predictor debe proporcionar una distribucion coherente de marcadores")
    matrix = np.asarray(prediction.score_probabilities, dtype=float)
    flat = int(rng.choice(matrix.size, p=matrix.ravel()))
    return tuple(int(value) for value in np.unravel_index(flat, matrix.shape))


def _apply_game(stats: dict[str, _Stats], game: _PlayedGame) -> None:
    a, b = stats[game.team_a], stats[game.team_b]
    a.goals_for += game.goals_a
    a.goals_against += game.goals_b
    b.goals_for += game.goals_b
    b.goals_against += game.goals_a
    if game.goals_a > game.goals_b:
        a.points += 3
    elif game.goals_a < game.goals_b:
        b.points += 3
    else:
        a.points += 1
        b.points += 1


def _head_to_head_key(team: str, tied: set[str], games: list[_PlayedGame]) -> tuple[int, int, int]:
    mini = {name: _Stats(name) for name in tied}
    for game in games:
        if game.team_a in tied and game.team_b in tied:
            _apply_game(mini, game)
    row = mini[team]
    return row.points, row.goal_difference, row.goals_for


def _rank_group(stats: dict[str, _Stats], games: list[_PlayedGame], rng: np.random.Generator) -> list[_Stats]:
    random_tiebreak = {team: float(rng.random()) for team in stats}
    preliminary = sorted(
        stats.values(),
        key=lambda row: (row.points, row.goal_difference, row.goals_for),
        reverse=True,
    )
    ranked: list[_Stats] = []
    cursor = 0
    while cursor < len(preliminary):
        base = preliminary[cursor]
        tied = [base]
        cursor += 1
        while cursor < len(preliminary):
            candidate = preliminary[cursor]
            if (candidate.points, candidate.goal_difference, candidate.goals_for) != (
                base.points,
                base.goal_difference,
                base.goals_for,
            ):
                break
            tied.append(candidate)
            cursor += 1
        tied_names = {row.team for row in tied}
        tied.sort(
            key=lambda row: (*_head_to_head_key(row.team, tied_names, games), random_tiebreak[row.team]),
            reverse=True,
        )
        ranked.extend(tied)
    return ranked


def _rank_thirds(thirds: list[tuple[str, _Stats]], rng: np.random.Generator) -> list[tuple[str, _Stats]]:
    random_tiebreak = {group: float(rng.random()) for group, _ in thirds}
    return sorted(
        thirds,
        key=lambda item: (
            item[1].points,
            item[1].goal_difference,
            item[1].goals_for,
            random_tiebreak[item[0]],
        ),
        reverse=True,
    )


def _allocate_thirds(best_thirds: list[tuple[str, _Stats]]) -> dict[str, str]:
    """Asigna terceros a ganadores respetando los cruces permitidos por FIFA."""
    third_by_group = {group: row.team for group, row in best_thirds}
    rank = {group: index for index, (group, _) in enumerate(best_thirds)}
    slots = sorted(THIRD_ELIGIBILITY, key=lambda slot: sum(g in third_by_group for g in THIRD_ELIGIBILITY[slot]))

    def search(index: int, used: set[str], allocation: dict[str, str]) -> dict[str, str] | None:
        if index == len(slots):
            return allocation.copy()
        slot = slots[index]
        candidates = sorted(
            (group for group in THIRD_ELIGIBILITY[slot] if group in third_by_group and group not in used),
            key=rank.get,
        )
        for group in candidates:
            used.add(group)
            allocation[slot] = third_by_group[group]
            result = search(index + 1, used, allocation)
            if result is not None:
                return result
            used.remove(group)
            allocation.pop(slot)
        return None

    result = search(0, set(), {})
    if result is None:
        raise RuntimeError("No se encontro una asignacion valida para los mejores terceros")
    return result


class TournamentSimulator:
    """Simula el torneo completo y agrega resultados de Monte Carlo."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._prediction_cache: dict[tuple[str, str, int | None], MatchPrediction] = {}
        self._used_pairs: set[tuple[str, str]] = set()

    def _predict(self, team_a: str, team_b: str, posterior_draw: int | None = None) -> MatchPrediction:
        self._used_pairs.add(tuple(sorted((team_a, team_b))))
        key = (team_a, team_b, posterior_draw)
        if key not in self._prediction_cache:
            try:
                self._prediction_cache[key] = self.predictor.predict_match(team_a, team_b, posterior_draw)
            except TypeError:
                self._prediction_cache[key] = self.predictor.predict_match(team_a, team_b)
        return self._prediction_cache[key]

    def _warm_prediction_cache(self, teams: list[str]) -> None:
        prime_method = getattr(self.predictor, "prime_matches", None)
        pairs = [(team_a, team_b) for team_a in teams for team_b in teams if team_a != team_b]
        if callable(prime_method):
            prime_method(pairs)
            return
        batch_method = getattr(self.predictor, "predict_matches", None)
        if not callable(batch_method):
            return
        missing = [pair for pair in pairs if (pair[0], pair[1], None) not in self._prediction_cache]
        if missing:
            for pair, prediction in zip(missing, batch_method(missing), strict=True):
                self._prediction_cache[(pair[0], pair[1], None)] = prediction

    def _play_group(
        self, teams: list[str], rng: np.random.Generator, posterior_draw: int | None,
        fixed_games: Mapping[frozenset[str], _PlayedGame] | None = None,
    ) -> tuple[list[_Stats], list[_PlayedGame]]:
        stats = {team: _Stats(team) for team in teams}
        games: list[_PlayedGame] = []
        for team_a, team_b in combinations(teams, 2):
            fixed = (fixed_games or {}).get(frozenset((team_a, team_b)))
            if fixed:
                if fixed.team_a == team_a:
                    game = fixed
                else:
                    game = _PlayedGame(team_a, team_b, fixed.goals_b, fixed.goals_a)
            else:
                goals_a, goals_b = _sample_score(self._predict(team_a, team_b, posterior_draw), rng)
                game = _PlayedGame(team_a, team_b, goals_a, goals_b)
            games.append(game)
            _apply_game(stats, game)
        return _rank_group(stats, games, rng), games

    def _play_knockout(
        self,
        match_id: str,
        round_name: str,
        team_a: str,
        team_b: str,
        overrides: Mapping[str, str],
        rng: np.random.Generator,
        posterior_draw: int | None,
        tournament_state: "TournamentState | None" = None,
    ) -> BracketMatch:
        official_match = tournament_state.matches.get(match_id) if tournament_state else None
        if official_match and official_match.team_a and official_match.team_b:
            if {team_a, team_b} != {official_match.team_a, official_match.team_b}:
                # Once FIFA publishes a pairing it is authoritative, including
                # tie-breaks that the simplified simulator cannot reconstruct.
                team_a, team_b = official_match.team_a, official_match.team_b
        official = bool(official_match and official_match.is_finished)
        if official:
            winner = str(official_match.winner)
            if winner == team_a:
                match_probability_a, draw_probability, match_probability_b = 1.0, 0.0, 0.0
            elif winner == team_b:
                match_probability_a, draw_probability, match_probability_b = 0.0, 0.0, 1.0
            else:
                match_probability_a, draw_probability, match_probability_b = 0.0, 1.0, 0.0
            return BracketMatch(
                match_id=match_id, round_name=round_name, team_a=team_a, team_b=team_b,
                winner=winner, probability_a=float(winner == team_a), probability_b=float(winner == team_b),
                match_probability_a=match_probability_a, draw_probability=draw_probability,
                match_probability_b=match_probability_b,
                forced=False, status="real", official=True,
            )
        prediction = self._predict(team_a, team_b, posterior_draw)
        decisive_total = prediction.prob_a + prediction.prob_b
        penalty_share_a = prediction.prob_a / decisive_total
        probability_a = prediction.prob_a + prediction.prob_draw * penalty_share_a
        probability_b = 1.0 - probability_a
        forced_team = overrides.get(match_id)
        forced = forced_team in (team_a, team_b)
        if forced:
            winner = str(forced_team)
        else:
            goals_a, goals_b = _sample_score(prediction, rng)
            if goals_a == goals_b:
                extra_a = int(rng.poisson(prediction.expected_goals_a / 3.0))
                extra_b = int(rng.poisson(prediction.expected_goals_b / 3.0))
                if extra_a == extra_b:
                    winner = str(rng.choice([team_a, team_b], p=[penalty_share_a, 1.0 - penalty_share_a]))
                else:
                    winner = team_a if extra_a > extra_b else team_b
            else:
                winner = team_a if goals_a > goals_b else team_b
        return BracketMatch(
            match_id=match_id,
            round_name=round_name,
            team_a=team_a,
            team_b=team_b,
            winner=winner,
            probability_a=probability_a,
            probability_b=probability_b,
            match_probability_a=prediction.prob_a,
            draw_probability=prediction.prob_draw,
            match_probability_b=prediction.prob_b,
            forced=forced,
            status="real" if official else "simulated",
            official=official,
        )

    def _knockout(
        self,
        ranked: dict[str, list[_Stats]],
        best_thirds: list[tuple[str, _Stats]],
        overrides: Mapping[str, str],
        rng: np.random.Generator,
        posterior_draw: int | None,
        tournament_state: "TournamentState | None" = None,
    ) -> tuple[str, list[BracketMatch]]:
        third_slot = _allocate_thirds(best_thirds)
        first = {group: ranked[group][0].team for group in GROUPS}
        second = {group: ranked[group][1].team for group in GROUPS}
        r32 = {
            "M73": (second["A"], second["B"]),
            "M74": (first["E"], third_slot["E"]),
            "M75": (first["F"], second["C"]),
            "M76": (first["C"], second["F"]),
            "M77": (first["I"], third_slot["I"]),
            "M78": (second["E"], second["I"]),
            "M79": (first["A"], third_slot["A"]),
            "M80": (first["L"], third_slot["L"]),
            "M81": (first["D"], third_slot["D"]),
            "M82": (first["G"], third_slot["G"]),
            "M83": (second["K"], second["L"]),
            "M84": (first["H"], second["J"]),
            "M85": (first["B"], third_slot["B"]),
            "M86": (first["J"], second["H"]),
            "M87": (first["K"], third_slot["K"]),
            "M88": (second["D"], second["G"]),
        }
        matches: list[BracketMatch] = []
        winners: dict[str, str] = {}
        losers: dict[str, str] = {}

        def play(match_id: str, round_name: str, team_a: str, team_b: str) -> None:
            match = self._play_knockout(
                match_id, round_name, team_a, team_b, overrides, rng, posterior_draw, tournament_state
            )
            matches.append(match)
            winners[match_id] = match.winner
            losers[match_id] = match.team_b if match.winner == match.team_a else match.team_a

        for match_id, teams in r32.items():
            play(match_id, "Ronda de 32", *teams)
        for match_id, source_a, source_b in (
            ("M89", "M74", "M77"), ("M90", "M73", "M75"),
            ("M91", "M76", "M78"), ("M92", "M79", "M80"),
            ("M93", "M83", "M84"), ("M94", "M81", "M82"),
            ("M95", "M86", "M88"), ("M96", "M85", "M87"),
        ):
            play(match_id, "Octavos", winners[source_a], winners[source_b])
        for match_id, source_a, source_b in (
            ("M97", "M89", "M90"), ("M98", "M93", "M94"),
            ("M99", "M91", "M92"), ("M100", "M95", "M96"),
        ):
            play(match_id, "Cuartos", winners[source_a], winners[source_b])
        play("M101", "Semifinal", winners["M97"], winners["M98"])
        play("M102", "Semifinal", winners["M99"], winners["M100"])
        play("M103", "Tercer puesto", losers["M101"], losers["M102"])
        play("M104", "Final", winners["M101"], winners["M102"])
        return winners["M104"], matches

    def simulate(
        self,
        groups: Mapping[str, list[str]],
        overrides: Mapping[str, str] | None = None,
        runs: int = 2_000,
        seed: int = 2026,
        tournament_state: "TournamentState | None" = None,
    ) -> TournamentSimulation:
        validate_groups(groups)
        if runs < 1:
            raise ValueError("runs debe ser al menos 1")
        overrides = overrides or {}
        self._used_pairs = set()
        rng = np.random.default_rng(seed)
        all_teams = [team for group in GROUPS for team in groups[group]]
        active_teams = set(all_teams)
        fixed_by_group: dict[str, dict[frozenset[str], _PlayedGame]] = {group: {} for group in GROUPS}
        if tournament_state:
            eliminated = tournament_state.eliminated_teams(groups)
            active_teams -= eliminated
            team_group = {team: group for group in GROUPS for team in groups[group]}
            for match in tournament_state.finished_matches.values():
                is_group_match = match.phase.strip().lower() in {"group", "group stage", "fase de grupos", "grupos"}
                if is_group_match and match.score_90 and match.team_a in team_group and match.team_b in team_group:
                    group = match.group or team_group[match.team_a]
                    if group == team_group[match.team_b]:
                        fixed_by_group[group][frozenset((match.team_a, match.team_b))] = _PlayedGame(
                            match.team_a, match.team_b, *match.score_90
                        )
        else:
            eliminated = set()
        if tournament_state is None:
            self._warm_prediction_cache(sorted(active_teams))
        posterior_draws = int(getattr(self.predictor, "posterior_draws", 0))
        accum = {
            group: {team: {"points": 0.0, "gf": 0.0, "ga": 0.0, "qualified": 0} for team in groups[group]}
            for group in GROUPS
        }
        champions = {team: 0 for team in all_teams}
        representative: list[BracketMatch] = []
        for run in range(runs):
            posterior_draw = int(rng.integers(posterior_draws)) if posterior_draws else None
            ranked: dict[str, list[_Stats]] = {}
            thirds: list[tuple[str, _Stats]] = []
            for group in GROUPS:
                table, _ = self._play_group(list(groups[group]), rng, posterior_draw, fixed_by_group[group])
                ranked[group] = table
                thirds.append((group, table[2]))
                for row in table:
                    target = accum[group][row.team]
                    target["points"] += row.points
                    target["gf"] += row.goals_for
                    target["ga"] += row.goals_against
            best_thirds = _rank_thirds(thirds, rng)[:8]
            qualified = {ranked[group][0].team for group in GROUPS}
            qualified.update(ranked[group][1].team for group in GROUPS)
            qualified.update(row.team for _, row in best_thirds)
            for group in GROUPS:
                for team in groups[group]:
                    accum[group][team]["qualified"] += int(team in qualified)
            champion, bracket = self._knockout(
                ranked, best_thirds, overrides, rng, posterior_draw, tournament_state
            )
            champions[champion] += 1
            if run == 0:
                representative = bracket
        group_tables: dict[str, list[GroupProjection]] = {}
        for group in GROUPS:
            projections = [
                GroupProjection(
                    team=team,
                    expected_points=values["points"] / runs,
                    expected_goals_for=values["gf"] / runs,
                    expected_goals_against=values["ga"] / runs,
                    qualification_probability=values["qualified"] / runs,
                )
                for team, values in accum[group].items()
            ]
            group_tables[group] = sorted(
                projections,
                key=lambda row: (row.expected_points, row.expected_goals_for - row.expected_goals_against),
                reverse=True,
            )
        champion_probabilities = {
            team: count / runs for team, count in sorted(champions.items(), key=lambda item: item[1], reverse=True)
        }
        champion_intervals = {}
        for team, count in champions.items():
            probability = count / runs
            z = 1.96
            denominator = 1.0 + z * z / runs
            center = (probability + z * z / (2.0 * runs)) / denominator
            half_width = z * np.sqrt(
                probability * (1.0 - probability) / runs + z * z / (4.0 * runs * runs)
            ) / denominator
            champion_intervals[team] = (
                max(0.0, float(center - half_width)),
                min(1.0, float(center + half_width)),
            )
        # ponytail: count unique pairs; market_weight is snapshot-based, same across posterior draws
        market_crossings_used = 0
        fallback_count = 0
        for (ta, tb, _), pred in self._prediction_cache.items():
            pair = tuple(sorted((ta, tb)))
            if pair in self._used_pairs:
                self._used_pairs.remove(pair)
                mw = getattr(pred, "market_weight", None)
                if mw is not None and mw > 0:
                    market_crossings_used += 1
                else:
                    fallback_count += 1
        return TournamentSimulation(
            group_tables=group_tables,
            bracket=representative,
            champion_probabilities=champion_probabilities,
            runs=runs,
            seed=seed,
            metadata={
                "group_matches_per_run": 72, "knockout_matches_per_run": 32,
                "posterior_draws": posterior_draws, "champion_confidence_intervals_95": champion_intervals,
                "market_crossings_used": market_crossings_used, "fallback_count": fallback_count,
                "official_state_hash": tournament_state.to_dict()["hash"] if tournament_state else None,
                "official_matches_fixed": len(tournament_state.finished_matches) if tournament_state else 0,
                "eliminated_teams": sorted(eliminated),
                "pending_matches_sampled_per_run": (
                    104 - len(tournament_state.finished_matches) if tournament_state else 104
                ),
            },
        )


def simulate_tournament(
    groups: Mapping[str, list[str]],
    predictor: Predictor,
    overrides: Mapping[str, str] | None = None,
    runs: int = 2_000,
    seed: int = 2026,
    tournament_state: "TournamentState | None" = None,
) -> TournamentSimulation:
    """Interfaz funcional solicitada por el dashboard."""
    return TournamentSimulator(predictor).simulate(groups, overrides, runs, seed, tournament_state)
