"""MarketBlendedPredictor: wraps KerasPredictor with Polymarket log-linear blend."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mundial.blend import load_blend_config, log_linear_pool
from mundial.config import ARTIFACTS_DIR
from mundial.polymarket import MarketPrice, load_snapshot
from mundial.schemas import MatchPrediction
from mundial.statistical import align_score_matrix

log = logging.getLogger(__name__)

_SNAPSHOT_NAME = "polymarket_snapshot.json"


class MarketBlendedPredictor:
    def __init__(self, base, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.base = base
        self.alpha = 0.0
        self._index: dict[tuple[str, str], MarketPrice] = {}

        snapshot_path = Path(artifacts_dir) / _SNAPSHOT_NAME
        try:
            markets = load_snapshot(snapshot_path)
        except (FileNotFoundError, OSError, Exception) as exc:
            log.warning("polymarket snapshot unavailable: %s", exc)
            return

        cfg = load_blend_config(artifacts_dir)
        if cfg is None:
            log.warning("market_blend.json missing; DL-only mode")
            return

        if not cfg.get("promoted", False) or float(cfg.get("alpha", 0.0)) == 0.0:
            log.info("market blend not promoted or alpha=0; DL-only mode")
            return

        self.alpha = float(cfg["alpha"])
        for mp in markets:
            self._index[(mp.team_a, mp.team_b)] = mp
            self._index[(mp.team_b, mp.team_a)] = mp

    @property
    def posterior_draws(self) -> int:
        return self.base.posterior_draws

    def prime_matches(self, pairs) -> None:
        self.base.prime_matches(pairs)

    def predict_match(self, team_a: str, team_b: str, posterior_draw=None) -> MatchPrediction:
        return self.predict_matches([(team_a, team_b)], posterior_draw)[0]

    def predict_matches(self, pairs, posterior_draw=None) -> list[MatchPrediction]:
        base_preds = self.base.predict_matches(pairs, posterior_draw)
        if self.alpha == 0.0:
            return base_preds

        results = []
        for pred, (team_a, team_b) in zip(base_preds, pairs):
            mp = self._index.get((team_a, team_b))
            if mp is None:
                results.append(pred)
                continue

            # Determine if we looked up in reversed order
            swapped = mp.team_a == team_b and mp.team_b == team_a
            if swapped:
                mkt_probs = np.array([mp.prob_b, mp.prob_draw, mp.prob_a])
            else:
                mkt_probs = np.array([mp.prob_a, mp.prob_draw, mp.prob_b])

            dl_probs = np.array([pred.prob_a, pred.prob_draw, pred.prob_b])
            blended = log_linear_pool(dl_probs, mkt_probs, self.alpha)

            new_matrix = align_score_matrix(np.array(pred.score_probabilities), blended)
            blended_pred = MatchPrediction.from_score_matrix(team_a, team_b, new_matrix)

            results.append(MatchPrediction(
                team_a=blended_pred.team_a,
                team_b=blended_pred.team_b,
                prob_a=blended_pred.prob_a,
                prob_draw=blended_pred.prob_draw,
                prob_b=blended_pred.prob_b,
                expected_goals_a=blended_pred.expected_goals_a,
                expected_goals_b=blended_pred.expected_goals_b,
                likely_score=blended_pred.likely_score,
                score_probabilities=blended_pred.score_probabilities,
                base_probabilities=(float(dl_probs[0]), float(dl_probs[1]), float(dl_probs[2])),
                market_probabilities=(float(mp.prob_a), float(mp.prob_draw), float(mp.prob_b)),
                market_weight=float(self.alpha),
                market_as_of=mp.captured_at,
                market_slug=mp.slug,
            ))
        return results
