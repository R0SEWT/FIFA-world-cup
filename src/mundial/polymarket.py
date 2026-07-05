"""Cliente Gamma + CLOB para precios 1-X-2 prepartido de Polymarket."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from mundial.config import load_aliases

_TIMEOUT = 30
_WORLD_CUP_TAG_ID = 102232
_USER_AGENT = "mundial-2026-ai/0.1 (academic prediction project)"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_array(value: object) -> list:
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, list) else []
    return value if isinstance(value, list) else []


def normalize_prices(raw: list[float]) -> list[float]:
    """Normaliza tres precios positivos para obtener una distribución 1-X-2."""
    if len(raw) != 3 or any(price <= 0 for price in raw):
        raise ValueError(f"Se requieren tres precios positivos, se recibió {raw}")
    total = sum(raw)
    return [price / total for price in raw]


def _resolve(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name.strip(), name.strip())


def _extract_teams(title: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    normalized = title.replace(" vs. ", " vs ").replace(" v. ", " vs ").replace(" v ", " vs ")
    if " vs " not in normalized:
        return None
    first, second = normalized.split(" vs ", 1)
    second = second.split(" - ", 1)[0].strip()
    return _resolve(first, aliases), _resolve(second, aliases)


def _get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read())


@dataclass(frozen=True)
class MarketPrice:
    team_a: str
    team_b: str
    prob_a: float
    prob_draw: float
    prob_b: float
    total_liquidity: float
    spread: float | None
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
    tag_id: int = _WORLD_CUP_TAG_ID

    def _fetch_events(self, *, closed: bool) -> list[dict]:
        events: list[dict] = []
        limit, offset = 100, 0
        query = {"tag_id": self.tag_id, "closed": str(closed).lower(), "limit": limit}
        if not closed:
            query["active"] = "true"
        while True:
            query["offset"] = offset
            page = _get_json(f"{self.gamma_base}/events?{urllib.parse.urlencode(query)}")
            if not isinstance(page, list) or not page:
                break
            events.extend(item for item in page if isinstance(item, dict))
            if len(page) < limit:
                break
            offset += limit
        return events

    def _last_price_before(self, token_id: str, cutoff: datetime) -> float | None:
        query = urllib.parse.urlencode({
            "market": token_id,
            "startTs": int((cutoff - timedelta(days=14)).timestamp()),
            "endTs": int(cutoff.timestamp()),
            "fidelity": 60,
        })
        data = _get_json(f"{self.clob_base}/prices-history?{query}")
        history = data.get("history", []) if isinstance(data, dict) else []
        eligible = [point for point in history if float(point["t"]) <= cutoff.timestamp()]
        return float(max(eligible, key=lambda point: float(point["t"]))["p"]) if eligible else None

    @staticmethod
    def _event_start(event: dict) -> datetime | None:
        # startDate is the publication date in Gamma and must not be used as kickoff.
        candidates = [event.get("startTime"), event.get("eventStartTime")]
        candidates.extend(market.get("gameStartTime") for market in event.get("markets", []))
        for raw in candidates:
            if raw:
                try:
                    return _parse_dt(str(raw))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _moneyline_markets(event: dict) -> list[dict]:
        result = []
        for market in event.get("markets", []):
            market_type = market.get("sportsMarketType") or market.get("category")
            description = str(market.get("description", "")).lower()
            if market_type == "moneyline" and ("90 minutes" in description or "regular play" in description):
                result.append(market)
        return result

    @staticmethod
    def _yes_token(market: dict) -> str | None:
        outcomes = _json_array(market.get("outcomes"))
        token_ids = _json_array(market.get("clobTokenIds"))
        if len(outcomes) != len(token_ids):
            return None
        for outcome, token_id in zip(outcomes, token_ids, strict=True):
            if str(outcome).lower() == "yes":
                return str(token_id)
        return None

    @staticmethod
    def _current_yes_price(market: dict) -> float | None:
        bid, ask = market.get("bestBid"), market.get("bestAsk")
        if bid is not None and ask is not None and float(ask) >= float(bid):
            return (float(bid) + float(ask)) / 2.0
        outcomes = _json_array(market.get("outcomes"))
        prices = _json_array(market.get("outcomePrices"))
        for outcome, price in zip(outcomes, prices):
            if str(outcome).lower() == "yes":
                return float(price)
        return None

    @staticmethod
    def _market_role(market: dict, teams: tuple[str, str], aliases: dict[str, str]) -> int | None:
        label = str(market.get("groupItemTitle") or market.get("question") or "").strip()
        if "draw" in label.lower():
            return 1
        resolved = _resolve(label, aliases)
        if resolved == teams[0]:
            return 0
        if resolved == teams[1]:
            return 2
        question = str(market.get("question", "")).lower()
        if teams[0].lower() in question and " win" in question:
            return 0
        if teams[1].lower() in question and " win" in question:
            return 2
        return None

    def _build_event_price(
        self,
        event: dict,
        aliases: dict[str, str],
        captured_at: datetime,
        price_loader: Callable[[dict], float | None],
        *,
        historical: bool = False,
    ) -> MarketPrice | None:
        teams = _extract_teams(str(event.get("title", "")), aliases)
        event_start = self._event_start(event)
        if teams is None or event_start is None:
            return None
        markets = self._moneyline_markets(event)
        by_role: dict[int, dict] = {}
        for market in markets:
            role = self._market_role(market, teams, aliases)
            if role is not None:
                by_role[role] = market
        if set(by_role) != {0, 1, 2}:
            return None

        actual_spreads = [float(by_role[index].get("spread") or 0.0) for index in range(3)]
        # Gamma clears sports books at kickoff. Closed markets therefore no
        # longer expose their pre-match spread/liquidity; use pre-match price
        # history plus traded volume, and leave spread explicitly unavailable.
        if not historical and any(spread < 0 or spread > self.max_spread for spread in actual_spreads):
            return None
        quality_amount = sum(float(
            by_role[index].get("volumeNum" if historical else "liquidityNum") or 0.0
        ) for index in range(3))
        if quality_amount < self.min_liquidity:
            return None
        prices = [price_loader(by_role[index]) for index in range(3)]
        if any(price is None or float(price) <= 0 for price in prices):
            return None
        normalized = normalize_prices([float(price) for price in prices])  # type: ignore[arg-type]
        condition_ids = [str(by_role[index].get("conditionId", "")) for index in range(3)]
        return MarketPrice(
            team_a=teams[0], team_b=teams[1],
            prob_a=normalized[0], prob_draw=normalized[1], prob_b=normalized[2],
            total_liquidity=quality_amount, spread=None if historical else max(actual_spreads),
            event_start=_iso(event_start), captured_at=_iso(captured_at),
            slug=str(event.get("slug", "")), condition_id=",".join(condition_ids),
        )

    def fetch_upcoming(self) -> list[MarketPrice]:
        aliases, now = load_aliases(), _now_utc()
        results = []
        for event in self._fetch_events(closed=False):
            start = self._event_start(event)
            if start is None or start <= now:
                continue
            price = self._build_event_price(event, aliases, now, self._current_yes_price)
            if price is not None:
                results.append(price)
        return results

    def fetch_historical(self) -> list[MarketPrice]:
        aliases, now = load_aliases(), _now_utc()
        events = self._fetch_events(closed=True)

        def build(event: dict) -> MarketPrice | None:
            start = self._event_start(event)
            if start is None or start >= now:
                return None
            if len(self._moneyline_markets(event)) != 3 or _extract_teams(str(event.get("title", "")), aliases) is None:
                return None
            cutoff = start - timedelta(minutes=self.min_minutes_before_kickoff)

            def historical_price(market: dict) -> float | None:
                token_id = self._yes_token(market)
                return self._last_price_before(token_id, cutoff) if token_id else None

            return self._build_event_price(event, aliases, cutoff, historical_price, historical=True)

        # CLOB permits high read throughput; bounded concurrency keeps a full
        # tournament snapshot practical without approaching documented limits.
        with ThreadPoolExecutor(max_workers=12) as executor:
            prices = executor.map(build, events)
            return [price for price in prices if price is not None]


def fetch_upcoming_markets(client: PolymarketClient | None = None) -> list[MarketPrice]:
    return (client or PolymarketClient()).fetch_upcoming()


def fetch_historical_markets(client: PolymarketClient | None = None) -> list[MarketPrice]:
    return (client or PolymarketClient()).fetch_historical()


def load_snapshot(snapshot_path: Path) -> list[MarketPrice]:
    """Carga un snapshot local sin realizar llamadas externas."""
    raw = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    items = raw.get("markets", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Formato de snapshot Polymarket inválido")
    return [MarketPrice(**item) for item in items]
