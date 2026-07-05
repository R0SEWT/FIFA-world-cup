from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mundial.config import RAW_DIR
from mundial.polymarket import MarketPrice, fetch_upcoming_markets


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical Polymarket moneyline data for past football matches and save as Parquet."
    )
    parser.add_argument("--output", default=str(RAW_DIR / "polymarket_moneyline.parquet"))
    parser.add_argument("--manifest", default=str(RAW_DIR / "polymarket_manifest.json"))
    args = parser.parse_args()

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    now = _now_utc()
    markets = fetch_upcoming_markets()
    # ponytail: fetch_upcoming_markets filters future markets; keep only past event_starts
    past: list[MarketPrice] = [
        m for m in markets
        if datetime.fromisoformat(m.event_start.replace("Z", "+00:00")) < now
    ]

    cols = ["team_a", "team_b", "prob_a", "prob_draw", "prob_b", "total_liquidity", "spread", "event_start", "captured_at", "slug", "condition_id"]
    df = pd.DataFrame([dataclasses.asdict(m) for m in past], columns=cols) if past else pd.DataFrame(columns=cols)
    df.to_parquet(output, index=False)

    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "sha256": sha,
        "rows": len(past),
        "parameters": {
            "min_liquidity": 10000.0,
            "max_spread": 0.05,
            "min_minutes_before_kickoff": 60,
        },
        "endpoints": [
            "https://gamma-api.polymarket.com/events",
            "https://clob.polymarket.com/prices-history",
        ],
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Markets fetched: {len(past)}")
    print(f"Output: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
