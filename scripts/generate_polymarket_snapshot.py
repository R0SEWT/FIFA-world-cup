from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from mundial.config import ARTIFACTS_DIR
from mundial.polymarket import MarketPrice, PolymarketClient, fetch_upcoming_markets


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch upcoming Polymarket markets and save as a JSON snapshot for offline use."
    )
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "polymarket_snapshot.json"))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    now = _now_utc()
    markets: list[MarketPrice] = fetch_upcoming_markets()

    snapshot = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "markets": [dataclasses.asdict(m) for m in markets],
    }
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"Markets saved: {len(markets)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
