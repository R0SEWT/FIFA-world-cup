"""Polymarket Gamma + CLOB API client for pre-match moneyline prices."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mundial.config import load_aliases

_SOCCER_TAGS = {"soccer", "football", "world-cup", "fifa"}
_VS_SEPARATORS = (" vs ", " v ", " vs. ")
_TIMEOUT = 30


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_prices(raw: list[float]) -> list[float]:
    """Normalizes prices to sum to 1. Raises ValueError if any price <= 0."""
    if any(p <= 0 for p in raw):
        raise ValueError(f"All prices must be > 0, got {raw}")
    total = sum(raw)
    return [p / total for p in raw]


def _resolve(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name.strip(), name.strip())


def _extract_teams(title: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    for sep in _VS_SEPARATORS:
        if sep in title:
            parts = title.split(sep, 1)
            return _resolve(parts[0], aliases), _resolve(parts[1], aliases)
    return None


def _get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read())


@dataclass(frozen=True)
class MarketPrice:
    team_a: str
    team_b: str
    prob_a: float
    prob_draw: float
    prob_b: float
    total_liquidity: float
    spread: float
    event_start: str
    captured_at: str
    slug: str
    condition_id: str


@dataclass
class PolymarketClient:
    gamma_base: str = "https://gamma-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    min_liquidity: float = 10_000.0
    max_spread: float = 0.05
    min_minutes_before_kickoff: int = 60

    def _fetch_events(self) -> list[dict]:
        events: list[dict] = []
        limit = 100
        offset = 0
        base = f"{self.gamma_base}/events?tag_slug=soccer&limit={limit}"
        while True:
            page = _get_json(f"{base}&offset={offset}")
            if not page:
                break
            events.extend(page)
            if len(page) < limit:
                break
            offset += limit
        return events

    def _last_price_before(self, condition_id: str, cutoff: datetime) -> float | None:
        url = (
            f"{self.clob_base}/prices-history"
            f"?market={urllib.parse.quote(condition_id)}&interval=1m&fidelity=60"
        )
        data = _get_json(url)
        history = data.get("history", data) if isinstance(data, dict) else data
        cutoff_ts = cutoff.timestamp()
        # ponytail: linear scan; markets have O(hours) of 1-min ticks, fast enough
        best = None
        for point in history:
            if point["t"] <= cutoff_ts:
                best = float(point["p"])
        return best

    def _is_soccer_event(self, event: dict) -> bool:
        tags = [t.get("slug", "") for t in event.get("tags", [])]
        return any(s in _SOCCER_TAGS for s in tags)

    def _build_market_price(
        self,
        market: dict,
        event_start: datetime,
        aliases: dict[str, str],
        now: datetime,
    ) -> MarketPrice | None:
        if not market.get("active") or market.get("closed"):
            return None
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 3:
            return None

        cutoff = datetime.fromtimestamp(
            event_start.timestamp() - self.min_minutes_before_kickoff * 60,
            tz=timezone.utc,
        )
        if now >= cutoff:
            return None

        condition_id = market.get("conditionId", "")
        if not condition_id:
            return None

        tokens = market.get("tokens", outcomes)
        if len(tokens) != 3:
            return None

        try:
            raw_prices = [self._last_price_before(t.get("conditionId", condition_id), cutoff) for t in tokens]
        except Exception:
            return None

        if any(p is None or p <= 0 for p in raw_prices):
            return None

        raw_prices_f: list[float] = [float(p) for p in raw_prices]  # type: ignore[arg-type]
        spread = max(raw_prices_f) - min(raw_prices_f)
        if spread > self.max_spread:
            return None

        liquidity = float(market.get("volumeNum", market.get("volume", 0)) or 0)
        if liquidity < self.min_liquidity:
            return None

        title = market.get("groupItemTitle", market.get("question", ""))
        teams = _extract_teams(title, aliases)
        if teams is None:
            return None

        normed = normalize_prices(raw_prices_f)
        return MarketPrice(
            team_a=teams[0],
            team_b=teams[1],
            prob_a=normed[0],
            prob_draw=normed[1],
            prob_b=normed[2],
            total_liquidity=liquidity,
            spread=spread,
            event_start=_iso(event_start),
            captured_at=_iso(now),
            slug=market.get("slug", ""),
            condition_id=condition_id,
        )

    def fetch(self) -> list[MarketPrice]:
        aliases = load_aliases()
        now = _now_utc()
        results: list[MarketPrice] = []

        for event in self._fetch_events():
            if not self._is_soccer_event(event):
                continue
            start_raw = event.get("startDate") or event.get("startTime")
            if not start_raw:
                continue
            try:
                event_start = _parse_dt(str(start_raw))
            except ValueError:
                continue

            for market in event.get("markets", []):
                cat = market.get("category", "")
                desc = market.get("description", "").lower()
                is_moneyline = cat == "moneyline" or "90 minutes" in desc or "full time" in desc
                if not is_moneyline:
                    continue
                mp = self._build_market_price(market, event_start, aliases, now)
                if mp:
                    results.append(mp)
        return results


def fetch_upcoming_markets(client: PolymarketClient | None = None) -> list[MarketPrice]:
    """Returns validated MarketPrice list for upcoming international football matches."""
    return (client or PolymarketClient()).fetch()


def load_snapshot(snapshot_path: Path) -> list[MarketPrice]:
    """Loads a previously saved JSON snapshot. Never calls the internet."""
    raw = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    return [MarketPrice(**item) for item in raw]


if __name__ == "__main__":
    # ponytail: self-check verifies normalization and alias resolution without network
    r = normalize_prices([0.5, 0.3, 0.2])
    assert abs(sum(r) - 1.0) < 1e-9 and r == [0.5, 0.3, 0.2], r

    r2 = normalize_prices([0.4, 0.3, 0.2])
    assert abs(sum(r2) - 1.0) < 1e-9, r2

    aliases = load_aliases()
    assert aliases.get("IR Iran") == "Iran", aliases.get("IR Iran")
    assert _resolve("IR Iran", aliases) == "Iran"
    assert _extract_teams("Iran vs USA", {"USA": "United States"}) == ("Iran", "United States")

    try:
        normalize_prices([0.5, 0.0, 0.3])
        raise AssertionError("should raise")
    except ValueError:
        pass

    print("self-check OK")
