"""Tests for mundial.blend (log_linear_pool, load_blend_config, run_blend_evaluation)
and mundial.market_blend (MarketBlendedPredictor)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mundial.blend import load_blend_config, log_linear_pool, run_blend_evaluation
from mundial.market_blend import MarketBlendedPredictor
from mundial.schemas import MatchPrediction

# ---------------------------------------------------------------------------
# Fake base predictors
# ---------------------------------------------------------------------------


class FakeBase:
    """DL-only stub — identity-on-diagonal matrix (all draws)."""

    posterior_draws = 10

    def predict_matches(self, pairs, posterior_draw=None):
        matrix = np.eye(13) / 13
        return [MatchPrediction.from_score_matrix(a, b, matrix) for a, b in pairs]

    def prime_matches(self, pairs):
        pass

    def predict_match(self, a, b, posterior_draw=None):
        return self.predict_matches([(a, b)], posterior_draw)[0]


class FakeBaseBalanced:
    """Stub with mass in all three outcome regions — required for align_score_matrix."""

    posterior_draws = 10

    def predict_matches(self, pairs, posterior_draw=None):
        # prob_a=0.4, prob_draw=0.3, prob_b=0.3
        matrix = np.zeros((13, 13))
        matrix[1, 0] = 0.2
        matrix[2, 0] = 0.1
        matrix[2, 1] = 0.1  # lower tri → A wins (0.4)
        matrix[0, 0] = 0.1
        matrix[1, 1] = 0.1
        matrix[2, 2] = 0.1  # diagonal → draws (0.3)
        matrix[0, 1] = 0.1
        matrix[0, 2] = 0.1
        matrix[1, 2] = 0.1  # upper tri → B wins (0.3)
        return [MatchPrediction.from_score_matrix(a, b, matrix) for a, b in pairs]

    def prime_matches(self, pairs):
        pass

    def predict_match(self, a, b, posterior_draw=None):
        return self.predict_matches([(a, b)], posterior_draw)[0]


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_BLEND_CONFIG = {
    "alpha": 0.2,
    "promoted": True,
    "promotion_reason": "both log_loss and brier improved",
    "n_calibration": 40,
    "n_evaluation": 25,
    "dataset_sha256": None,
    "generated_at": "2026-06-01T00:00:00Z",
    "calibration_date_range": ["2024-01-01", "2025-06-01"],
    "evaluation_date_range": ["2025-06-02", "2026-01-01"],
    "metrics": {},
}

_SNAPSHOT_MARKET = {
    "team_a": "Brazil",
    "team_b": "Argentina",
    "prob_a": 0.45,
    "prob_draw": 0.28,
    "prob_b": 0.27,
    "total_liquidity": 125_000.0,
    "spread": 0.02,
    "event_start": "2026-06-15T18:00:00Z",
    "captured_at": "2026-06-15T16:30:00Z",
    "slug": "brazil-vs-argentina",
    "condition_id": "0xabc",
}


# ---------------------------------------------------------------------------
# log_linear_pool
# ---------------------------------------------------------------------------


class TestLogLinearPool(unittest.TestCase):
    def test_alpha_zero(self):
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([0.4, 0.35, 0.25])
        result = log_linear_pool(p, m, 0.0)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=9)
        np.testing.assert_allclose(result, p / p.sum(), atol=1e-9)

    def test_alpha_one(self):
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([0.4, 0.35, 0.25])
        result = log_linear_pool(p, m, 1.0)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=9)
        np.testing.assert_allclose(result, m / m.sum(), atol=1e-9)

    def test_intermediate(self):
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([0.4, 0.35, 0.25])
        result = log_linear_pool(p, m, 0.3)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=9)
        self.assertFalse(np.allclose(result, p / p.sum()))
        self.assertFalse(np.allclose(result, m / m.sum()))

    def test_symmetry(self):
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([0.4, 0.35, 0.25])
        r1 = log_linear_pool(p, m, 0.3)
        r2 = log_linear_pool(p[::-1], m[::-1], 0.3)
        self.assertAlmostEqual(float(r2[0]), float(r1[2]), places=9)
        self.assertAlmostEqual(float(r2[1]), float(r1[1]), places=9)
        self.assertAlmostEqual(float(r2[2]), float(r1[0]), places=9)

    def test_mass_conservation(self):
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([0.4, 0.35, 0.25])
        rng = np.random.default_rng(42)
        for alpha in rng.uniform(0.0, 0.5, 20):
            result = log_linear_pool(p, m, float(alpha))
            self.assertAlmostEqual(float(result.sum()), 1.0, places=9)


# ---------------------------------------------------------------------------
# load_blend_config
# ---------------------------------------------------------------------------


class TestLoadBlendConfig(unittest.TestCase):
    def test_load_blend_config_missing(self):
        self.assertIsNone(load_blend_config(Path("/nonexistent")))

    def test_load_blend_config_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "market_blend.json").write_text(
                json.dumps(_BLEND_CONFIG), encoding="utf-8"
            )
            result = load_blend_config(Path(tmp))
        self.assertIsNotNone(result)
        self.assertEqual(result["alpha"], 0.2)
        self.assertTrue(result["promoted"])


# ---------------------------------------------------------------------------
# run_blend_evaluation — insufficient data path
# ---------------------------------------------------------------------------


class TestChronologicalSplit(unittest.TestCase):
    def test_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_blend_evaluation(
                FakeBase(),
                Path("/nonexistent/poly.parquet"),
                Path("/nonexistent/matches.parquet"),
                artifacts_dir=Path(tmp),
            )
        self.assertEqual(result["alpha"], 0.0)
        self.assertFalse(result["promoted"])
        self.assertIn("muestra insuficiente", result["promotion_reason"])


# ---------------------------------------------------------------------------
# MarketBlendedPredictor
# ---------------------------------------------------------------------------


class TestMarketBlendedPredictor(unittest.TestCase):
    def test_fallback(self):
        predictor = MarketBlendedPredictor(FakeBase(), artifacts_dir=Path("/nonexistent"))
        pred = predictor.predict_match("Brazil", "Argentina")

        self.assertIsInstance(pred, MatchPrediction)
        # DL-only: market_weight is 0.0 (set by _with_base_probs)
        self.assertEqual(pred.market_weight, 0.0)
        self.assertAlmostEqual(pred.prob_a + pred.prob_draw + pred.prob_b, 1.0, places=6)

    def test_with_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Snapshot is a flat list of MarketPrice dicts (matching load_snapshot expectation)
            (tmp_path / "polymarket_snapshot.json").write_text(
                json.dumps([_SNAPSHOT_MARKET]), encoding="utf-8"
            )
            (tmp_path / "market_blend.json").write_text(
                json.dumps(_BLEND_CONFIG), encoding="utf-8"
            )
            predictor = MarketBlendedPredictor(FakeBaseBalanced(), artifacts_dir=tmp_path)
            pred = predictor.predict_match("Brazil", "Argentina")

        self.assertIsInstance(pred, MatchPrediction)
        self.assertEqual(pred.market_weight, 0.2)
        self.assertEqual(pred.market_slug, "brazil-vs-argentina")
        self.assertAlmostEqual(pred.prob_a + pred.prob_draw + pred.prob_b, 1.0, places=6)
        if pred.score_probabilities is not None:
            matrix = np.asarray(pred.score_probabilities)
            self.assertAlmostEqual(float(matrix.sum()), 1.0, places=6)

    def test_monte_carlo_no_external_calls(self):
        predictor = MarketBlendedPredictor(FakeBase(), artifacts_dir=Path("/nonexistent"))
        pairs = [("Brazil", "Argentina")] * 100
        with patch("urllib.request.urlopen") as mock_urlopen:
            preds = predictor.predict_matches(pairs)
        mock_urlopen.assert_not_called()
        self.assertEqual(len(preds), 100)


if __name__ == "__main__":
    unittest.main()
