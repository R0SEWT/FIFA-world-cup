"""Pruebas del cliente Polymarket contra el esquema real de Gamma."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from mundial.polymarket import (
    PolymarketClient,
    _extract_teams,
    fetch_historical_markets,
    fetch_upcoming_markets,
    load_snapshot,
    normalize_prices,
)


def _binary_market(label: str, yes: float, token: str, *, spread: float = 0.01, liquidity: float = 20_000):
    question = "Will Brazil vs. Norway end in a draw?" if label.startswith("Draw") else f"Will {label} win?"
    return {
        "active": True,
        "closed": False,
        "question": question,
        "groupItemTitle": label,
        "sportsMarketType": "moneyline",
        "description": "This market refers only to the outcome within the first 90 minutes of regular play.",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps([str(yes), str(1 - yes)]),
        "clobTokenIds": json.dumps([token, f"{token}-no"]),
        "bestBid": yes - 0.005,
        "bestAsk": yes + 0.005,
        "spread": spread,
        "liquidityNum": liquidity,
        "volumeNum": liquidity,
        "conditionId": f"condition-{token}",
        "gameStartTime": "2027-01-01T12:00:00Z",
    }


def _event(*, closed: bool = False):
    markets = [
        _binary_market("Brazil", 0.535, "bra"),
        _binary_market("Draw (Brazil vs. Norway)", 0.265, "draw"),
        _binary_market("Norway", 0.205, "nor"),
    ]
    for market in markets:
        market["closed"] = closed
        market["active"] = not closed
    return {
        "id": "654615",
        "slug": "fifwc-bra-nor-2026-07-05",
        "title": "Brazil vs. Norway",
        "startDate": "2026-07-01T10:03:40Z",  # publicación, no kickoff
        "startTime": "2027-01-01T12:00:00Z",
        "markets": markets,
    }


def test_normalize_prices():
    assert normalize_prices([0.535, 0.265, 0.205]) == pytest.approx([0.5323383, 0.2636816, 0.2039801])
    with pytest.raises(ValueError):
        normalize_prices([0.5, 0.5])
    with pytest.raises(ValueError):
        normalize_prices([0.5, 0.5, 0.0])


def test_extract_teams_and_aliases():
    assert _extract_teams("Brazil vs. Norway", {}) == ("Brazil", "Norway")
    assert _extract_teams("IR Iran vs USA", {"IR Iran": "Iran", "USA": "United States"}) == (
        "Iran", "United States"
    )


def test_real_gamma_shape_builds_one_1x2_quote(monkeypatch):
    client = PolymarketClient()
    monkeypatch.setattr("mundial.polymarket._now_utc", lambda: datetime(2026, 12, 31, tzinfo=timezone.utc))
    monkeypatch.setattr(client, "_fetch_events", lambda **_: [_event()])
    results = client.fetch_upcoming()
    assert len(results) == 1
    quote = results[0]
    assert (quote.team_a, quote.team_b) == ("Brazil", "Norway")
    assert quote.event_start == "2027-01-01T12:00:00Z"
    assert quote.prob_a + quote.prob_draw + quote.prob_b == pytest.approx(1.0)
    assert quote.spread == pytest.approx(0.01)
    assert quote.total_liquidity == pytest.approx(60_000)


def test_uses_actual_bid_ask_spread_not_probability_range(monkeypatch):
    client = PolymarketClient(max_spread=0.05)
    monkeypatch.setattr("mundial.polymarket._now_utc", lambda: datetime(2026, 12, 31, tzinfo=timezone.utc))
    monkeypatch.setattr(client, "_fetch_events", lambda **_: [_event()])
    quote = client.fetch_upcoming()[0]
    assert quote.prob_a - quote.prob_b > 0.05
    assert quote.spread <= 0.05


def test_rejects_wide_spread_and_low_liquidity(monkeypatch):
    client = PolymarketClient()
    monkeypatch.setattr("mundial.polymarket._now_utc", lambda: datetime(2026, 12, 31, tzinfo=timezone.utc))
    wide = _event()
    wide["markets"][1]["spread"] = 0.06
    monkeypatch.setattr(client, "_fetch_events", lambda **_: [wide])
    assert client.fetch_upcoming() == []

    low = _event()
    for market in low["markets"]:
        market["liquidityNum"] = 1_000
    monkeypatch.setattr(client, "_fetch_events", lambda **_: [low])
    assert client.fetch_upcoming() == []


def test_historical_uses_yes_tokens_at_one_hour_cutoff(monkeypatch):
    client = PolymarketClient()
    event = _event(closed=True)
    event["startTime"] = "2026-01-02T12:00:00Z"
    for market in event["markets"]:
        market["gameStartTime"] = event["startTime"]
    monkeypatch.setattr("mundial.polymarket._now_utc", lambda: datetime(2026, 1, 3, tzinfo=timezone.utc))
    monkeypatch.setattr(client, "_fetch_events", lambda **kwargs: [event] if kwargs["closed"] else [])
    seen = []

    def price(token, cutoff):
        seen.append((token, cutoff))
        return {"bra": 0.5, "draw": 0.3, "nor": 0.2}[token]

    monkeypatch.setattr(client, "_last_price_before", price)
    quote = client.fetch_historical()[0]
    assert [token for token, _ in seen] == ["bra", "draw", "nor"]
    assert all(cutoff.isoformat() == "2026-01-02T11:00:00+00:00" for _, cutoff in seen)
    assert quote.captured_at == "2026-01-02T11:00:00Z"


def test_public_helpers_delegate_to_correct_mode():
    client = PolymarketClient()
    with patch.object(client, "fetch_upcoming", return_value=[]) as upcoming:
        assert fetch_upcoming_markets(client) == []
        upcoming.assert_called_once()
    with patch.object(client, "fetch_historical", return_value=[]) as historical:
        assert fetch_historical_markets(client) == []
        historical.assert_called_once()


def test_snapshot_supports_envelope(tmp_path: Path):
    payload = {
        "generated_at": "2026-01-01T00:00:00Z",
        "markets": [{
            "team_a": "Brazil", "team_b": "Norway", "prob_a": 0.5,
            "prob_draw": 0.3, "prob_b": 0.2, "total_liquidity": 20_000,
            "spread": 0.01, "event_start": "2026-01-02T12:00:00Z",
            "captured_at": "2026-01-02T11:00:00Z", "slug": "fifwc-bra-nor",
            "condition_id": "a,d,b",
        }],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_snapshot(path)[0].slug == "fifwc-bra-nor"
