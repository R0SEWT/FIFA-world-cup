"""Market blend evaluation: log-linear pool of DL and Polymarket probabilities."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mundial.config import ARTIFACTS_DIR

log = logging.getLogger(__name__)


def log_linear_pool(p: np.ndarray, m: np.ndarray, alpha: float) -> np.ndarray:
    """Log-linear pool. p and m are shape (3,) or (N,3). Returns normalized probabilities."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, None)
    m = np.clip(np.asarray(m, dtype=float), 1e-6, None)
    q = p ** (1.0 - alpha) * m ** alpha
    return q / q.sum(axis=-1, keepdims=True)


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    picked = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(picked + 1e-15)))


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


def _macro_f1(probs: np.ndarray, outcomes: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    f1s = []
    for c in range(3):
        tp = float(np.sum((preds == c) & (outcomes == c)))
        fp = float(np.sum((preds == c) & (outcomes != c)))
        fn = float(np.sum((preds != c) & (outcomes == c)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == outcomes).astype(float)
    n = len(outcomes)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (confidence >= lo) & (confidence <= hi if idx == n_bins - 1 else confidence < hi)
        if not mask.any():
            continue
        ece += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean()) / n
    return float(ece)


def _all_metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "log_loss": _log_loss(probs, outcomes),
        "brier": _brier(probs, outcomes),
        "accuracy": _accuracy(probs, outcomes),
        "macro_f1": _macro_f1(probs, outcomes),
        "ece": _ece(probs, outcomes),
    }


def _metrics_with_ci(probs: np.ndarray, outcomes: np.ndarray, seed: int = 42, n_boot: int = 2000) -> dict:
    base = _all_metrics(probs, outcomes)
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    samples: dict[str, list[float]] = {k: [] for k in base}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, v in _all_metrics(probs[idx], outcomes[idx]).items():
            samples[k].append(v)
    base["ci_95"] = {k: [float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))] for k, vs in samples.items()}
    return base


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _join_data(poly: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Match polymarket records to historical results by teams and date (±1 day)."""
    poly = poly.copy()
    if not pd.api.types.is_datetime64_any_dtype(poly["event_start"]):
        poly["event_start"] = pd.to_datetime(poly["event_start"], utc=True)
    # Normalize to tz-naive date for comparison
    es = poly["event_start"]
    poly_date = (es.dt.tz_convert(None) if es.dt.tz is not None else es).dt.normalize()
    md = pd.to_datetime(matches["date"], utc=True)
    match_date = md.dt.tz_convert(None).dt.normalize()

    records = []
    for i, pm in enumerate(poly.itertuples(index=False)):
        ta, tb = pm.team_a, pm.team_b
        pdate = poly_date.iloc[i]
        subset = matches[(match_date - pdate).abs() <= pd.Timedelta(days=1)]

        match_row = None
        flipped = False
        for _, row in subset.iterrows():
            if row["home_team"] == ta and row["away_team"] == tb:
                match_row = row
                break
            if row["home_team"] == tb and row["away_team"] == ta:
                match_row = row
                flipped = True
                break

        if match_row is None:
            continue

        raw = int(match_row["result"])  # 0=home win, 1=draw, 2=away win
        # ponytail: flip map for when polymarket team_a is the away side
        outcome = {0: 2, 1: 1, 2: 0}[raw] if flipped else raw
        records.append({
            "event_start": pm.event_start,
            "team_a": ta, "team_b": tb,
            "outcome": outcome,
            "m_a": pm.prob_a, "m_draw": pm.prob_draw, "m_b": pm.prob_b,
        })
    return pd.DataFrame(records)


def _join_oof(joined: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Adjunta probabilidades OOF respetando orientación y fecha del encuentro."""
    records = []
    oof = oof.copy()
    oof["date"] = pd.to_datetime(oof["date"], utc=True).dt.normalize()
    for row in joined.itertuples(index=False):
        event_date = pd.Timestamp(row.event_start).tz_convert("UTC").normalize()
        candidates = oof[(oof["date"] - event_date).abs() <= pd.Timedelta(days=1)]
        match = candidates[(candidates["home_team"] == row.team_a) & (candidates["away_team"] == row.team_b)]
        flipped = False
        if match.empty:
            match = candidates[(candidates["home_team"] == row.team_b) & (candidates["away_team"] == row.team_a)]
            flipped = not match.empty
        if match.empty:
            continue
        source = match.iloc[0]
        payload = row._asdict()
        probs = [source["oof_prob_a"], source["oof_prob_draw"], source["oof_prob_b"]]
        if flipped:
            probs = [probs[2], probs[1], probs[0]]
        payload.update(dict(zip(["oof_prob_a", "oof_prob_draw", "oof_prob_b"], probs, strict=True)))
        records.append(payload)
    return pd.DataFrame(records)


def run_blend_evaluation(
    predictor,
    polymarket_path: Path,
    matches_path: Path,
    artifacts_dir: Path = ARTIFACTS_DIR,
    seed: int = 42,
    oof_path: Path | None = None,
) -> dict:
    """Runs full evaluation and writes market_blend.json. Returns the result dict."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    polymarket_path, matches_path = Path(polymarket_path), Path(matches_path)
    dataset_sha = _sha256_file(polymarket_path) if polymarket_path.exists() else None

    def _write_insufficient(reason: str) -> dict:
        result: dict = {
            "alpha": 0.0, "promoted": False, "promotion_reason": reason,
            "n_calibration": 0, "n_evaluation": 0,
            "dataset_sha256": dataset_sha, "generated_at": now_iso,
            "calibration_date_range": [None, None],
            "evaluation_date_range": [None, None],
            "metrics": {},
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "market_blend.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result

    if not polymarket_path.exists() or not matches_path.exists():
        log.warning("muestra insuficiente")
        return _write_insufficient("muestra insuficiente")

    poly = pd.read_parquet(polymarket_path)
    matches = pd.read_parquet(matches_path)
    joined = _join_data(poly, matches)

    if joined.empty:
        log.warning("muestra insuficiente")
        return _write_insufficient("muestra insuficiente")

    joined = joined.sort_values("event_start").reset_index(drop=True)

    # Never score historical rows with the production predictor: it may have
    # trained on their outcomes. The matched dataset must carry probabilities
    # produced by expanding-window models whose cutoff precedes each match.
    oof_columns = ["oof_prob_a", "oof_prob_draw", "oof_prob_b"]
    oof_path = Path(oof_path) if oof_path is not None else Path(artifacts_dir) / "market_oof.parquet"
    if not oof_path.exists():
        log.warning("faltan predicciones OOF temporales; se bloquea evaluación con fuga")
        return _write_insufficient("faltan predicciones OOF temporales sin fuga")
    oof = pd.read_parquet(oof_path)
    required_oof = ["home_team", "away_team", "date", "training_cutoff", *oof_columns]
    if not all(column in oof.columns for column in required_oof):
        return _write_insufficient("artefacto OOF inválido")
    if (pd.to_datetime(oof["training_cutoff"]) >= pd.to_datetime(oof["date"])).any():
        return _write_insufficient("artefacto OOF contiene fuga temporal")
    joined = _join_oof(joined, oof)
    if joined.empty or joined[oof_columns].isna().any().any():
        return _write_insufficient("predicciones OOF no emparejadas")
    dl_probs = joined[oof_columns].to_numpy(dtype=float)
    market_probs = joined[["m_a", "m_draw", "m_b"]].to_numpy(dtype=float)
    outcomes = joined["outcome"].to_numpy(dtype=int)

    n = len(joined)
    n_cal = int(n * 0.6)
    n_eval = n - n_cal

    if n_cal < 30 or n_eval < 20:
        log.warning("muestra insuficiente")
        return _write_insufficient("muestra insuficiente")

    cal_dl, eval_dl = dl_probs[:n_cal], dl_probs[n_cal:]
    cal_market, eval_market = market_probs[:n_cal], market_probs[n_cal:]
    cal_out, eval_out = outcomes[:n_cal], outcomes[n_cal:]
    cal_dates = joined["event_start"].iloc[:n_cal]
    eval_dates = joined["event_start"].iloc[n_cal:]

    # Alpha search on calibration set
    alphas = np.round(np.arange(0.0, 0.51, 0.01), 2)
    losses = [_log_loss(log_linear_pool(cal_dl, cal_market, float(a)), cal_out) for a in alphas]
    best_alpha = float(alphas[int(np.argmin(losses))])

    eval_blended = log_linear_pool(eval_dl, eval_market, best_alpha)
    dl_m = _metrics_with_ci(eval_dl, eval_out, seed=seed)
    market_m = _metrics_with_ci(eval_market, eval_out, seed=seed)
    blended_m = _metrics_with_ci(eval_blended, eval_out, seed=seed)

    promoted = blended_m["log_loss"] < dl_m["log_loss"] and blended_m["brier"] < dl_m["brier"]
    alpha_final = best_alpha if promoted else 0.0

    def _fmt(dt) -> str | None:
        return str(dt)[:10] if dt is not None else None

    result = {
        "alpha": alpha_final,
        "promoted": promoted,
        "promotion_reason": "both log_loss and brier improved" if promoted else "did not improve",
        "n_calibration": n_cal,
        "n_evaluation": n_eval,
        "dataset_sha256": dataset_sha,
        "generated_at": now_iso,
        "calibration_date_range": [_fmt(cal_dates.min()), _fmt(cal_dates.max())],
        "evaluation_date_range": [_fmt(eval_dates.min()), _fmt(eval_dates.max())],
        "metrics": {"dl": dl_m, "market": market_m, "blended": blended_m},
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "market_blend.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def load_blend_config(artifacts_dir: Path = ARTIFACTS_DIR) -> dict | None:
    """Loads market_blend.json. Returns None if file doesn't exist."""
    path = Path(artifacts_dir) / "market_blend.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
