"""Contratos publicos y serializables del predictor y simulador."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatchPrediction:
    team_a: str
    team_b: str
    prob_a: float
    prob_draw: float
    prob_b: float
    expected_goals_a: float
    expected_goals_b: float
    likely_score: tuple[int, int]
    score_probabilities: tuple[tuple[float, ...], ...] | None = None
    # Optional Polymarket market blend fields (set only when market data is available)
    base_probabilities: tuple[float, float, float] | None = None
    market_probabilities: tuple[float, float, float] | None = None
    market_weight: float | None = None
    market_as_of: str | None = None
    market_slug: str | None = None

    def __post_init__(self) -> None:
        probabilities = (self.prob_a, self.prob_draw, self.prob_b)
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Las probabilidades deben estar entre 0 y 1")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Las probabilidades deben sumar 1")
        for field_name, triple in (("base_probabilities", self.base_probabilities), ("market_probabilities", self.market_probabilities)):
            if triple is not None:
                if len(triple) != 3 or any(v < 0.0 or v > 1.0 for v in triple):
                    raise ValueError(f"{field_name}: valores deben estar en [0,1]")
                if abs(sum(triple) - 1.0) > 1e-6:
                    raise ValueError(f"{field_name}: debe sumar 1")
        if self.market_weight is not None and not (0.0 <= self.market_weight <= 1.0):
            raise ValueError("market_weight debe estar en [0.0, 1.0]")
        if self.score_probabilities is not None:
            matrix = np.asarray(self.score_probabilities, dtype=float)
            if matrix.shape != (13, 13):
                raise ValueError("La matriz de marcadores debe tener forma 13x13")
            if not np.isfinite(matrix).all() or (matrix < 0.0).any():
                raise ValueError("La matriz de marcadores debe ser finita y no negativa")
            if abs(float(matrix.sum()) - 1.0) > 1e-6:
                raise ValueError("La matriz de marcadores debe sumar 1")
            derived = (
                float(np.tril(matrix, -1).sum()),
                float(np.trace(matrix)),
                float(np.triu(matrix, 1).sum()),
            )
            if max(abs(a - b) for a, b in zip(probabilities, derived, strict=True)) > 1e-6:
                raise ValueError("La matriz de marcadores no reproduce las probabilidades 1-X-2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_score_matrix(
        cls,
        team_a: str,
        team_b: str,
        score_matrix: np.ndarray,
    ) -> "MatchPrediction":
        """Construye todos los resumenes publicados desde una unica distribucion coherente."""
        matrix = np.asarray(score_matrix, dtype=float)
        if matrix.shape != (13, 13) or matrix.sum() <= 0:
            raise ValueError("Se requiere una matriz 13x13 con masa positiva")
        matrix = matrix / matrix.sum()
        goals = np.arange(13, dtype=float)
        best = np.unravel_index(int(matrix.argmax()), matrix.shape)
        return cls(
            team_a=team_a,
            team_b=team_b,
            prob_a=float(np.tril(matrix, -1).sum()),
            prob_draw=float(np.trace(matrix)),
            prob_b=float(np.triu(matrix, 1).sum()),
            expected_goals_a=float((matrix * goals[:, None]).sum()),
            expected_goals_b=float((matrix * goals[None, :]).sum()),
            likely_score=(int(best[0]), int(best[1])),
            score_probabilities=tuple(tuple(float(value) for value in row) for row in matrix),
        )


@dataclass(frozen=True)
class GroupProjection:
    team: str
    expected_points: float
    expected_goals_for: float
    expected_goals_against: float
    qualification_probability: float


@dataclass(frozen=True)
class BracketMatch:
    match_id: str
    round_name: str
    team_a: str
    team_b: str
    winner: str
    probability_a: float
    probability_b: float
    forced: bool = False
    status: str = "simulated"
    official: bool = False


@dataclass
class TournamentSimulation:
    group_tables: dict[str, list[GroupProjection]]
    bracket: list[BracketMatch]
    champion_probabilities: dict[str, float]
    runs: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
