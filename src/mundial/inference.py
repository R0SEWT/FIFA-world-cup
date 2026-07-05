"""Carga del modelo seleccionado y adaptacion al contrato de prediccion."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from mundial.config import ARTIFACTS_DIR
from mundial.data import STATIC_FEATURES
from mundial.schemas import MatchPrediction
from mundial.statistical import (
    ARTIFACT_VERSION,
    CalibrationPosterior,
    DixonColesPosterior,
    align_score_matrix,
    dixon_coles_matrix,
)


class KerasPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(artifacts_dir / "selected_model.keras")
        bundle = joblib.load(artifacts_dir / "inference_bundle.joblib")
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.team_states = bundle["team_states"]
        self.team_sequences = bundle["team_sequences"]
        self.model_type = bundle["model_type"]
        self.default_state = bundle["default_state"]
        self.h2h = bundle.get("h2h", {})
        manifest = json.loads((artifacts_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Version de artefactos incompatible: {manifest.get('version')}")
        self.calibration: CalibrationPosterior = joblib.load(artifacts_dir / "calibration_posterior.joblib")
        self.dixon_coles: DixonColesPosterior = joblib.load(artifacts_dir / "dixon_coles_posterior.joblib")
        self.posterior_draws = min(len(self.calibration.temperatures), len(self.dixon_coles.intercept))
        self._backbone_cache: dict[tuple[str, str], np.ndarray] = {}

    def _state(self, team: str) -> dict[str, float]:
        return self.team_states.get(team, self.default_state)

    def _raw_features(self, team_a: str, team_b: str) -> list[float]:
        a, b = self._state(team_a), self._state(team_b)
        first, second = sorted((team_a, team_b))
        pair = self.h2h.get((first, second), {"wins_first": 0.0, "draws": 0.0, "wins_second": 0.0, "gd_first": 0.0})
        a_is_first = team_a == first
        feature_map = {
            "rank_a": a["rank"], "rank_b": b["rank"], "rank_diff": b["rank"] - a["rank"],
            "goals_for_last10_a": a["goals_for_last10"], "goals_against_last10_a": a["goals_against_last10"],
            "goals_for_last10_b": b["goals_for_last10"], "goals_against_last10_b": b["goals_against_last10"],
            "h2h_wins_a": pair["wins_first"] if a_is_first else pair["wins_second"],
            "h2h_draws": pair["draws"],
            "h2h_wins_b": pair["wins_second"] if a_is_first else pair["wins_first"],
            "h2h_goal_difference": pair["gd_first"] if a_is_first else -pair["gd_first"],
            "players_imputed_a": a["players_imputed"], "players_imputed_b": b["players_imputed"], "neutral": 1.0,
        }
        for attribute in ("overall", "pace", "shooting", "defending", "physical"):
            feature_map[f"{attribute}_a"] = a[attribute]
            feature_map[f"{attribute}_b"] = b[attribute]
        return [feature_map[name] for name in STATIC_FEATURES]

    def _model_outputs(self, pairs: list[tuple[str, str]]):
        """Ejecuta el backbone; se separa para poder imponer simetria en cancha neutral."""
        raw = np.asarray([self._raw_features(team_a, team_b) for team_a, team_b in pairs], dtype=np.float32)
        static = self.scaler.transform(self.imputer.transform(raw)).astype(np.float32)
        if self.model_type.startswith("mlp"):
            inputs = static
        else:
            zero_sequence = np.zeros((10, 5), dtype=np.float32)
            inputs = {
                "static": static,
                "sequence_a": np.stack([self.team_sequences.get(team_a, zero_sequence) for team_a, _ in pairs]),
                "sequence_b": np.stack([self.team_sequences.get(team_b, zero_sequence) for _, team_b in pairs]),
            }
        return self.model.predict(inputs, verbose=0)

    def predict_matches(
        self,
        pairs: list[tuple[str, str]],
        posterior_draw: int | None = None,
    ) -> list[MatchPrediction]:
        """Predice muchos cruces en un unico batch para Monte Carlo."""
        if any(team_a == team_b for team_a, team_b in pairs):
            raise ValueError("Una seleccion no puede jugar contra si misma")
        if posterior_draw is not None and not 0 <= posterior_draw < self.posterior_draws:
            raise ValueError(f"posterior_draw debe estar entre 0 y {self.posterior_draws - 1}")
        self.prime_matches(pairs)
        symmetric_raw = np.stack([self._backbone_cache[pair] for pair in pairs])
        forward = self.calibration.calibrate(symmetric_raw, posterior_draw)
        reverse = self.calibration.calibrate(symmetric_raw[:, [2, 1, 0]], posterior_draw)[:, [2, 1, 0]]
        symmetric = (forward + reverse) / 2.0
        symmetric /= symmetric.sum(axis=1, keepdims=True)
        predictions: list[MatchPrediction] = []
        for index, (team_a, team_b) in enumerate(pairs):
            rate_a, rate_b, rho = self.dixon_coles.rates(team_a, team_b, posterior_draw)
            statistical, retained_mass = dixon_coles_matrix(rate_a, rate_b, rho)
            if retained_mass < 1.0 - 1e-6:
                raise ValueError(f"Masa truncada excesiva para {team_a} vs {team_b}: {1-retained_mass:.3g}")
            matrix = align_score_matrix(statistical, symmetric[index])
            predictions.append(MatchPrediction.from_score_matrix(team_a, team_b, matrix))
        return predictions

    def prime_matches(self, pairs: list[tuple[str, str]]) -> None:
        """Calcula una sola vez el backbone simetrizado para cualquier muestra posterior."""
        missing = [pair for pair in pairs if pair not in self._backbone_cache]
        if not missing:
            return
        result, _, _ = self._model_outputs(missing)
        reversed_pairs = [(team_b, team_a) for team_a, team_b in missing]
        reverse_result, _, _ = self._model_outputs(reversed_pairs)
        symmetric = (result + reverse_result[:, [2, 1, 0]]) / 2.0
        symmetric /= symmetric.sum(axis=1, keepdims=True)
        for pair, probabilities in zip(missing, symmetric, strict=True):
            self._backbone_cache[pair] = probabilities
            self._backbone_cache[(pair[1], pair[0])] = probabilities[[2, 1, 0]]

    def predict_match(self, team_a: str, team_b: str, posterior_draw: int | None = None) -> MatchPrediction:
        return self.predict_matches([(team_a, team_b)], posterior_draw=posterior_draw)[0]


def load_predictor(artifacts_dir: Path = ARTIFACTS_DIR):
    """Usa el modelo real cuando existe y deja un fallback explicito para demo."""
    from mundial.prediction import DemoPredictor

    required = (
        "selected_model.keras", "inference_bundle.joblib", "artifact_manifest.json",
        "calibration_posterior.joblib", "dixon_coles_posterior.joblib",
    )
    if all((artifacts_dir / name).exists() for name in required):
        try:
            keras_predictor = KerasPredictor(artifacts_dir)
            try:
                from mundial.market_blend import MarketBlendedPredictor
                from mundial.blend import load_blend_config
                blend_config = load_blend_config(artifacts_dir)
                snapshot_path = artifacts_dir / "polymarket_snapshot.json"
                if blend_config is not None and snapshot_path.exists():
                    predictor = MarketBlendedPredictor(keras_predictor, artifacts_dir)
                    alpha = blend_config.get("alpha", 0.0)
                    promoted = blend_config.get("promoted", False)
                    if promoted and alpha > 0:
                        mode = f"Modelo híbrido DL + Bayes + Polymarket (α={alpha:.2f})"
                    else:
                        mode = "Modelo híbrido DL + Bayes (Polymarket no promovido)"
                    return predictor, mode
            except Exception:
                pass  # polymarket optional
            return keras_predictor, "Modelo híbrido DL + Bayes entrenado"
        except (ValueError, KeyError, OSError) as error:
            return DemoPredictor(), f"Modo demostracion (artefactos incompatibles: {error})"
    return DemoPredictor(), "Modo demostracion (regenere los artefactos híbridos v2)"


def predict_match(team_a: str, team_b: str, artifacts_dir: Path = ARTIFACTS_DIR) -> MatchPrediction:
    """Interfaz funcional publica para predecir cualquier par de selecciones."""
    predictor, _ = load_predictor(artifacts_dir)
    return predictor.predict_match(team_a, team_b)
