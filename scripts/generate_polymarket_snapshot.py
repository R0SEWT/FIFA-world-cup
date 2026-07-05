from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from mundial.config import ARTIFACTS_DIR
from mundial.polymarket import MarketPrice, fetch_upcoming_markets, filter_pending_markets
from mundial.tournament_state import STATE_PATH, load_tournament_state


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch upcoming Polymarket markets and save as a JSON snapshot for offline use."
    )
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "polymarket_snapshot.json"))
    parser.add_argument("--tournament-state", default=str(STATE_PATH))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    now = _now_utc()
    markets: list[MarketPrice] = fetch_upcoming_markets()
    state_path = Path(args.tournament_state)
    state = load_tournament_state(state_path) if state_path.exists() else None
    if state is not None:
        markets = filter_pending_markets(markets, state)

    snapshot = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tournament_state_hash": state.to_dict()["hash"] if state else None,
        "markets": [dataclasses.asdict(m) for m in markets],
    }
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"Markets saved: {len(markets)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
