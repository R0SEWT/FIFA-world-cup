"""Tests for mundial.polymarket — normalize, alias resolution, filtering, snapshot load, API mock."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from mundial.polymarket import (
    MarketPrice,
    PolymarketClient,
    _extract_teams,
    fetch_upcoming_markets,
    load_snapshot,
    normalize_prices,
)
from mundial.config import load_aliases

# ---------------------------------------------------------------------------
# Shared fixtures (plain dicts, no pytest fixtures)
# ---------------------------------------------------------------------------

_MARKET = {
    "active": True,
    "closed": False,
    "category": "moneyline",
    "groupItemTitle": "Brazil vs Argentina",
    "conditionId": "0xabc",
    "outcomes": ["Brazil", "Draw", "Argentina"],
    "tokens": [
        {"token_id": "0xabc1"},
        {"token_id": "0xabc2"},
        {"token_id": "0xabc3"},
    ],
    "volumeNum": 50_000.0,
    "slug": "brazil-vs-argentina-ml",
}

_EVENT_START_FAR = datetime(2027, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW_EARLY = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
_ALIASES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# normalize_prices
# ---------------------------------------------------------------------------


class TestNormalizePrices(unittest.TestCase):
    def test_normalize_prices_sums_to_one(self):
        r = normalize_prices([0.4, 0.3, 0.2])
        self.assertAlmostEqual(sum(r), 1.0, places=9)

        r2 = normalize_prices([0.5, 0.3, 0.2])
        for a, b in zip(r2, [0.5, 0.3, 0.2]):
            self.assertAlmostEqual(a, b, places=9)

        r3 = normalize_prices([1.0, 1.0, 1.0])
        for v in r3:
            self.assertAlmostEqual(v, 1 / 3, places=9)

    def test_normalize_prices_clipping(self):
        with self.assertRaises(ValueError):
            normalize_prices([0.0, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Spread is computed on normalised prices (not raw)
# ---------------------------------------------------------------------------


class TestSpreadOnNormalized(unittest.TestCase):
    def test_spread_computed_on_normalized(self):
        # raw [0.6, 0.5, 0.5] → sum 1.6 → normalised [0.375, 0.3125, 0.3125]
        # spread = 0.375 − 0.3125 = 0.0625 > 0.05 → rejected
        client = PolymarketClient()
        with patch.object(client, "_last_price_before", side_effect=[0.6, 0.5, 0.5]):
            result = client._build_market_price(_MARKET, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNone(result)

    def test_spread_tight_accepted(self):
        # raw [0.34, 0.33, 0.33] → spread ≈ 0.01 ≤ 0.05 → accepted
        client = PolymarketClient()
        with patch.object(client, "_last_price_before", side_effect=[0.34, 0.33, 0.33]):
            result = client._build_market_price(_MARKET, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Alias resolution and team name stripping
# ---------------------------------------------------------------------------


class TestAliasResolution(unittest.TestCase):
    def test_alias_resolution(self):
        aliases = load_aliases()
        teams = _extract_teams("IR Iran vs Saudi Arabia", aliases)
        self.assertIsNotNone(teams)
        self.assertEqual(teams[0], "Iran")

        teams2 = _extract_teams("Cape Verde vs Senegal", aliases)
        self.assertIsNotNone(teams2)
        self.assertEqual(teams2[0], "Cabo Verde")

    def test_team_name_trailing_strip(self):
        teams = _extract_teams("Brazil vs Argentina - Moneyline", {})
        self.assertIsNotNone(teams)
        self.assertEqual(teams[1], "Argentina")


# ---------------------------------------------------------------------------
# Rejection: high spread, low liquidity, market already started
# ---------------------------------------------------------------------------


class TestRejection(unittest.TestCase):
    def _client_with(self, prices: list[float]) -> PolymarketClient:
        client = PolymarketClient()
        client._last_price_before = MagicMock(side_effect=prices)
        return client

    def test_rejection_high_spread(self):
        # spread = 0.9 − 0.05 = 0.85 > 0.05 → rejected
        client = self._client_with([0.9, 0.05, 0.05])
        result = client._build_market_price(_MARKET, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNone(result)

    def test_rejection_low_spread_accepted(self):
        client = self._client_with([0.34, 0.33, 0.33])
        result = client._build_market_price(_MARKET, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNotNone(result)

    def test_rejection_low_liquidity(self):
        low_liq = {**_MARKET, "volumeNum": 5_000.0}
        client = self._client_with([0.34, 0.33, 0.33])
        result = client._build_market_price(low_liq, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNone(result)

    def test_rejection_high_liquidity_accepted(self):
        client = self._client_with([0.34, 0.33, 0.33])
        result = client._build_market_price(_MARKET, _EVENT_START_FAR, _ALIASES, _NOW_EARLY)
        self.assertIsNotNone(result)

    def test_rejection_started_market(self):
        # now is 30 min before kickoff — cutoff is 60 min before → now ≥ cutoff → rejected
        event_start = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        now_late = datetime(2026, 8, 1, 11, 30, 0, tzinfo=timezone.utc)
        client = PolymarketClient()
        result = client._build_market_price(_MARKET, event_start, _ALIASES, now_late)
        self.assertIsNone(result)

    def test_upcoming_market_not_rejected(self):
        # now is 2 hours before kickoff → now < cutoff → not rejected by time check
        event_start = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        now_early = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        client = self._client_with([0.34, 0.33, 0.33])
        result = client._build_market_price(_MARKET, event_start, _ALIASES, now_early)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# load_snapshot — pure file read, no network
# ---------------------------------------------------------------------------


class TestLoadSnapshot(unittest.TestCase):
    def test_load_snapshot(self):
        snapshot_data = [
            {
                "team_a": "Brazil",
                "team_b": "Argentina",
                "prob_a": 0.45,
                "prob_draw": 0.28,
                "prob_b": 0.27,
                "total_liquidity": 125_000.0,
                "spread": 0.02,
                "event_start": "2026-06-15T18:00:00Z",
                "captured_at": "2026-06-15T16:30:00Z",
                "slug": "brazil-vs-argentina",
                "condition_id": "0xabc",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            path.write_text(json.dumps(snapshot_data), encoding="utf-8")
            result = load_snapshot(path)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MarketPrice)
        self.assertEqual(result[0].team_a, "Brazil")
        self.assertEqual(result[0].team_b, "Argentina")


# ---------------------------------------------------------------------------
# Integration: API client with mocked urllib transport
# ---------------------------------------------------------------------------


class TestApiClientWithMockTransport(unittest.TestCase):
    @staticmethod
    def _make_response(data: object):
        body = json.dumps(data).encode()

        class _FakeResp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return _FakeResp()

    def _fake_urlopen(self, url, timeout=30):
        url_str = str(url)
        if "gamma-api" in url_str:
            return self._make_response(self._events)
        # CLOB price history — route by token_id substring
        if "tok_g1" in url_str:
            return self._make_response({"history": [{"t": 1_000_000, "p": 0.34}]})
        if "tok_g" in url_str:
            return self._make_response({"history": [{"t": 1_000_000, "p": 0.33}]})
        if "tok_w1" in url_str:
            return self._make_response({"history": [{"t": 1_000_000, "p": 0.90}]})
        if "tok_w" in url_str:
            return self._make_response({"history": [{"t": 1_000_000, "p": 0.05}]})
        return self._make_response([])

    def setUp(self):
        good_market = {
            "active": True,
            "closed": False,
            "category": "moneyline",
            "groupItemTitle": "Brazil vs Argentina",
            "conditionId": "0xgood",
            "outcomes": ["Brazil", "Draw", "Argentina"],
            "tokens": [
                {"token_id": "tok_g1"},
                {"token_id": "tok_g2"},
                {"token_id": "tok_g3"},
            ],
            "volumeNum": 50_000.0,
            "slug": "brazil-vs-argentina-ml",
        }
        malformed_market = {
            # "active" key missing → skipped gracefully
            "category": "moneyline",
            "groupItemTitle": "France vs Italy",
            "conditionId": "0xbad",
            "outcomes": ["France", "Draw", "Italy"],
        }
        wide_spread_market = {
            "active": True,
            "closed": False,
            "category": "moneyline",
            "groupItemTitle": "Spain vs Germany",
            "conditionId": "0xwide",
            "outcomes": ["Spain", "Draw", "Germany"],
            "tokens": [
                {"token_id": "tok_w1"},
                {"token_id": "tok_w2"},
                {"token_id": "tok_w3"},
            ],
            "volumeNum": 50_000.0,
            "slug": "spain-vs-germany-ml",
        }
        self._events = [
            {
                "slug": "world-cup-soccer-2027",
                "tags": [{"slug": "soccer"}, {"slug": "world-cup"}],
                "startDate": "2027-01-01T12:00:00Z",
                "markets": [good_market, malformed_market, wide_spread_market],
            }
        ]

    def test_api_client_with_mock_transport(self):
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen):
            results = fetch_upcoming_markets()

        # Only the good market survives (malformed skipped, wide spread filtered)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].team_a, "Brazil")
        self.assertEqual(results[0].team_b, "Argentina")
        self.assertGreaterEqual(results[0].total_liquidity, 10_000.0)
        self.assertLessEqual(results[0].spread, 0.05)

    def test_no_results_on_empty_events(self):
        self._events = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen):
            results = fetch_upcoming_markets()
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
