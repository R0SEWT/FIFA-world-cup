#!/usr/bin/env python3
"""Genera predicciones DL OOF con cortes anuales anteriores a cada partido."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mundial.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR
from mundial.data import STATIC_FEATURES
from mundial.models import build_mlp, build_recurrent
from mundial.training import (
    _inputs, _swap_raw_static, _training_payload, load_training_data, set_seeds,
)


def _matched_indices(frame: pd.DataFrame, markets: pd.DataFrame) -> list[int]:
    dates = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    indices: set[int] = set()
    for market in markets.itertuples(index=False):
        event_date = pd.Timestamp(market.event_start).tz_convert("UTC").normalize()
        near = frame.index[(dates - event_date).abs() <= pd.Timedelta(days=1)]
        for index in near:
            row = frame.loc[index]
            if {row.home_team, row.away_team} == {market.team_a, market.team_b}:
                indices.add(int(index))
                break
    return sorted(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=RAW_DIR / "polymarket_moneyline.parquet")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "market_oof.parquet")
    parser.add_argument("--epochs", type=int, help="Sobrescribe las épocas del manifiesto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seeds(args.seed)
    frame, seq_a, seq_b = load_training_data(args.processed)
    markets = pd.read_parquet(args.markets)
    targets = _matched_indices(frame, markets)
    if not targets:
        raise RuntimeError("No se emparejaron partidos entre Polymarket y el dataset")

    manifest = json.loads((args.artifacts / "artifact_manifest.json").read_text(encoding="utf-8"))
    kind = str(manifest["selected_model"])
    epochs = int(args.epochs or manifest["production_epochs"])
    raw = frame[list(STATIC_FEATURES)].to_numpy(dtype=np.float32)
    swapped_raw = _swap_raw_static(raw)
    dates = pd.to_datetime(frame["date"])
    records = []

    for year in sorted({int(dates.iloc[index].year) for index in targets}):
        cutoff = pd.Timestamp(f"{year}-01-01")
        train_mask = (dates < cutoff).to_numpy()
        validation_indices = np.array([index for index in targets if dates.iloc[index].year == year], dtype=int)
        if not train_mask.any() or not len(validation_indices):
            continue
        imputer = SimpleImputer(strategy="median").fit(raw[train_mask])
        scaler = StandardScaler().fit(imputer.transform(raw[train_mask]))
        static = scaler.transform(imputer.transform(raw)).astype(np.float32)
        swapped = scaler.transform(imputer.transform(swapped_raw)).astype(np.float32)
        if kind == "mlp_adam":
            model = build_mlp(len(STATIC_FEATURES), "adam")
        elif kind == "mlp_sgd":
            model = build_mlp(len(STATIC_FEATURES), "sgd")
        elif kind in {"lstm", "gru"}:
            model = build_recurrent(len(STATIC_FEATURES), seq_a.shape[2], kind)
        else:
            raise ValueError(f"Modelo seleccionado desconocido: {kind}")
        train_inputs, train_targets = _training_payload(
            kind, static, swapped, seq_a, seq_b, frame, train_mask
        )
        model.fit(train_inputs, train_targets, epochs=epochs, batch_size=256, verbose=0)
        forward, _, _ = model.predict(
            _inputs(kind, static[validation_indices], seq_a[validation_indices], seq_b[validation_indices]), verbose=0
        )
        reverse, _, _ = model.predict(
            _inputs(kind, swapped[validation_indices], seq_b[validation_indices], seq_a[validation_indices]), verbose=0
        )
        probabilities = (forward + reverse[:, [2, 1, 0]]) / 2.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for index, probability in zip(validation_indices, probabilities, strict=True):
            row = frame.iloc[index]
            records.append({
                "home_team": row.home_team, "away_team": row.away_team, "date": row.date,
                "training_cutoff": cutoff - pd.Timedelta(days=1),
                "oof_prob_a": float(probability[0]), "oof_prob_draw": float(probability[1]),
                "oof_prob_b": float(probability[2]), "model_type": kind,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(f"Predicciones OOF: {len(records)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
