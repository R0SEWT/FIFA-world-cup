"""MarketBlendedPredictor: wraps KerasPredictor with Polymarket log-linear blend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mundial.blend import log_linear_pool
from mundial.config import ARTIFACTS_DIR
from mundial.polymarket import MarketPrice, load_snapshot
from mundial.schemas import MatchPrediction
from mundial.statistical import align_score_matrix

log = logging.getLogger(__name__)

_SNAPSHOT_NAME = "polymarket_snapshot.json"


def _with_base_probs(pred: MatchPrediction) -> MatchPrediction:
    return MatchPrediction(
        team_a=pred.team_a,
        team_b=pred.team_b,
        prob_a=pred.prob_a,
        prob_draw=pred.prob_draw,
        prob_b=pred.prob_b,
        expected_goals_a=pred.expected_goals_a,
        expected_goals_b=pred.expected_goals_b,
        likely_score=pred.likely_score,
        score_probabilities=pred.score_probabilities,
        base_probabilities=(pred.prob_a, pred.prob_draw, pred.prob_b),
        market_weight=0.0,  # ponytail: DL-only within blended predictor
    )


class MarketBlendedPredictor:
    def __init__(self, base, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.base = base
        self.alpha = 0.0
        self._index: dict[tuple[str, str], MarketPrice] = {}

        try:
            from mundial.blend import load_blend_config
            cfg = load_blend_config(Path(artifacts_dir))
            snapshot_path = Path(artifacts_dir) / _SNAPSHOT_NAME
            markets = load_snapshot(snapshot_path) if snapshot_path.exists() else []
        except Exception as exc:
            log.warning("Polymarket init failed, using DL-only: %s", exc)
            self._index = {}
            self.alpha = 0.0
            return

        if cfg is None:
            log.warning("market_blend.json missing; DL-only mode")
            return

        if not cfg.get("promoted", False) or float(cfg.get("alpha", 0.0)) == 0.0:
            log.info("market blend not promoted or alpha=0; DL-only mode")
            return

        alpha = float(cfg["alpha"])
        if not 0.0 < alpha <= 0.5:
            log.warning("invalid promoted market alpha %.3f; DL-only mode", alpha)
            return
        self.alpha = alpha
        now = datetime.now(timezone.utc)
        for mp in sorted(markets, key=lambda item: item.event_start):
            try:
                kickoff = datetime.fromisoformat(mp.event_start.replace("Z", "+00:00"))
                captured = datetime.fromisoformat(mp.captured_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if captured >= kickoff or kickoff <= now:
                continue
            self._index[(mp.team_a, mp.team_b)] = mp
            self._index[(mp.team_b, mp.team_a)] = mp

    @property
    def market_count(self) -> int:
        return len(self._index) // 2

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
            return [_with_base_probs(p) for p in base_preds]

        results = []
        for pred, (team_a, team_b) in zip(base_preds, pairs):
            mp = self._index.get((team_a, team_b))
            if mp is None:
                results.append(_with_base_probs(pred))
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
                market_probabilities=(float(mkt_probs[0]), float(mkt_probs[1]), float(mkt_probs[2])),
                market_weight=float(self.alpha),
                market_as_of=mp.captured_at,
                market_slug=mp.slug,
            ))
        return results
