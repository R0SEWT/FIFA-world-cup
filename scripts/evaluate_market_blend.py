#!/usr/bin/env python3
"""Ajusta y evalúa el peso de Polymarket usando probabilidades OOF temporales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mundial.blend import run_blend_evaluation
from mundial.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR
from mundial.inference import KerasPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=RAW_DIR / "polymarket_moneyline.parquet")
    parser.add_argument("--matches", type=Path, default=PROCESSED_DIR / "matches.parquet")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--oof", type=Path, default=ARTIFACTS_DIR / "market_oof.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_blend_evaluation(
        KerasPredictor(args.artifacts), args.markets, args.matches,
        artifacts_dir=args.artifacts, seed=args.seed, oof_path=args.oof,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
