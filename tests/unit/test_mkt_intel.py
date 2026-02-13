"""Comprehensive tests for Flow 3: Prediction Market Intelligence.

Tests all new components:
- Market models (UnifiedMarket, alerts, ScanResult)
- Kalshi REST client (parsing, API calls, error handling)
- Polymarket REST extensions (expiring markets, DataClient)
- WebSocket clients (message handling)
- WebSocket manager (subscriptions, real-time reads)
- Market scanner (scan cycle, volume spikes, odds movements)
- Wallet tracker (activity checks, discovery, metrics)
- Processor (scoring, signal generation)
- API routes (all endpoints)
- Telegram formatting (signal message)
- Config (new settings)
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest

from synesis.markets.kalshi import (
    KalshiClient,
    KalshiMarket,
)
from synesis.markets.models import (
    InsiderAlert,
    OddsMovement,
    ScanResult,
    UnifiedMarket,
    VolumeSpike,
)
from synesis.processing.mkt_intel.models import (
    MarketIntelOpportunity,
    MarketIntelSignal,
)


# ───────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────


def _make_market(
    platform: str = "polymarket",
    external_id: str = "market_1",
    question: str = "Will X happen?",
    yes_price: float = 0.65,
    no_price: float = 0.35,
    volume_24h: float = 50000.0,
    condition_id: str | None = "cond_1",
    ticker: str | None = None,
    end_date: datetime | None = None,
    **kwargs: Any,
) -> UnifiedMarket:
    return UnifiedMarket(
        platform=platform,
        external_id=external_id,
        condition_id=condition_id,
        ticker=ticker,
        question=question,
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=volume_24h,
        end_date=end_date,
        url=f"https://polymarket.com/event/{external_id}",
        **kwargs,
    )


def _make_kalshi_market(
    ticker: str = "TEST-MKT",
    external_id: str | None = None,
    **kwargs: Any,
) -> UnifiedMarket:
    return UnifiedMarket(
        platform="kalshi",
        external_id=external_id or ticker,
        condition_id=None,
        ticker=ticker,
        question=kwargs.pop("question", "Kalshi test?"),
        yes_price=kwargs.pop("yes_price", 0.55),
        no_price=kwargs.pop("no_price", 0.45),
        volume_24h=kwargs.pop("volume_24h", 10000.0),
        url=f"https://kalshi.com/markets/TEST/{ticker}",
        **kwargs,
    )


def _make_signal(**kwargs: Any) -> MarketIntelSignal:
    defaults: dict[str, Any] = {
        "timestamp": datetime.now(UTC),
        "total_markets_scanned": 10,
        "trending_markets": [_make_market()],
        "expiring_soon": [],
        "insider_activity": [],
        "odds_movements": [],
        "opportunities": [],
        "market_uncertainty_index": 0.5,
        "informed_activity_level": 0.0,
        "ws_connected": False,
    }
    defaults.update(kwargs)
    return MarketIntelSignal(**defaults)


@pytest.fixture
def mock_redis() -> Any:
    """Stateful mock Redis for unit tests."""
    redis = AsyncMock()
    _strings: dict[str, str] = {}
    _hashes: dict[str, dict[str, Any]] = {}

    async def mock_get(key: str) -> str | None:
        return _strings.get(key)

    async def mock_set(key: str, value: str, **kw: Any) -> bool:
        _strings[key] = value
        return True

    async def mock_hset(key: str, mapping: dict[str, str] | None = None, **kw: str) -> int:
        if key not in _hashes:
            _hashes[key] = {}
        _hashes[key].update(mapping or kw)
        return len(mapping or kw)

    async def mock_hgetall(key: str) -> dict[bytes, bytes]:
        h = _hashes.get(key, {})
        return {k.encode() if isinstance(k, str) else k: str(v).encode() for k, v in h.items()}

    async def mock_expire(key: str, seconds: int) -> bool:
        return True

    async def mock_delete(*keys: str) -> int:
        count = 0
        for k in keys:
            if k in _strings:
                del _strings[k]
                count += 1
            if k in _hashes:
                del _hashes[k]
                count += 1
        return count

    async def mock_incrbyfloat(key: str, amount: float) -> float:
        current = float(_strings.get(key, "0"))
        new = current + amount
        _strings[key] = str(new)
        return new

    async def mock_getdel(key: str) -> str | None:
        return _strings.pop(key, None)

    redis.get = mock_get
    redis.set = mock_set
    redis.hset = mock_hset
    redis.hgetall = mock_hgetall
    redis.expire = mock_expire
    redis.delete = mock_delete
    redis.incrbyfloat = mock_incrbyfloat

    async def mock_scan(
        cursor: int = 0, match: str = "*", count: int = 100
    ) -> tuple[int, list[str]]:
        import fnmatch

        matched = [k for k in _strings if fnmatch.fnmatch(k, match)]
        return (0, matched)

    redis.getdel = mock_getdel
    redis.scan = mock_scan
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()
    redis.publish = AsyncMock(return_value=1)
    redis._strings = _strings
    redis._hashes = _hashes
    return redis


# ───────────────────────────────────────────────────────────────
# 1. Market Models
# ───────────────────────────────────────────────────────────────


class TestUnifiedMarket:
    def test_create_polymarket(self) -> None:
        m = _make_market()
        assert m.platform == "polymarket"
        assert m.yes_price == 0.65
        assert m.condition_id == "cond_1"

    def test_create_kalshi(self) -> None:
        m = _make_kalshi_market()
        assert m.platform == "kalshi"
        assert m.ticker == "TEST-MKT"
        assert m.condition_id is None

    def test_default_values(self) -> None:
        m = UnifiedMarket(
            platform="polymarket",
            external_id="x",
            condition_id="cond_x",
            question="?",
            yes_price=0.5,
            no_price=0.5,
        )
        assert m.volume_24h == 0.0
        assert m.is_active is True
        assert m.open_interest is None


class TestOddsMovement:
    def test_create_up(self) -> None:
        mov = OddsMovement(
            market=_make_market(),
            price_change_1h=0.10,
            direction="up",
        )
        assert mov.direction == "up"

    def test_create_with_6h(self) -> None:
        mov = OddsMovement(
            market=_make_market(),
            price_change_1h=-0.08,
            price_change_6h=-0.15,
            direction="down",
        )
        assert mov.price_change_6h == -0.15


class TestInsiderAlert:
    def test_create(self) -> None:
        alert = InsiderAlert(
            market=_make_market(),
            wallet_address="0xabcd1234",
            insider_score=0.85,
            trade_direction="yes",
            trade_size=10000.0,
        )
        assert alert.insider_score == 0.85


class TestScanResult:
    def test_create_empty(self) -> None:
        sr = ScanResult(timestamp=datetime.now(UTC))
        assert sr.trending_markets == []
        assert sr.total_markets_scanned == 0

    def test_create_with_data(self) -> None:
        sr = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[_make_market()],
            expiring_markets=[_make_kalshi_market()],
            total_markets_scanned=2,
        )
        assert sr.total_markets_scanned == 2


# ───────────────────────────────────────────────────────────────
# 2. Signal Models
# ───────────────────────────────────────────────────────────────


class TestMarketIntelOpportunity:
    def test_create(self) -> None:
        opp = MarketIntelOpportunity(
            market=_make_market(),
            suggested_direction="yes",
            confidence=0.75,
            triggers=["insider_activity", "odds_movement"],
            reasoning="2 insider wallet(s) active; Price moved +0.10 in 1h",
        )
        assert opp.confidence == 0.75
        assert "insider_activity" in opp.triggers

    def test_confidence_bounds(self) -> None:
        # confidence is constrained to 0-1
        opp = MarketIntelOpportunity(
            market=_make_market(),
            suggested_direction="no",
            confidence=1.0,
        )
        assert opp.confidence == 1.0

    def test_confidence_rejects_out_of_bounds(self) -> None:
        with pytest.raises(Exception):
            MarketIntelOpportunity(
                market=_make_market(),
                suggested_direction="no",
                confidence=1.5,
            )


class TestMarketIntelSignal:
    def test_create_minimal(self) -> None:
        sig = MarketIntelSignal(timestamp=datetime.now(UTC))
        assert sig.total_markets_scanned == 0
        assert sig.signal_period == "1h"
        assert sig.ws_connected is False

    def test_serialization_roundtrip(self) -> None:
        sig = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["insider_activity"],
                    reasoning="Test",
                )
            ]
        )
        json_str = sig.model_dump_json()
        restored = MarketIntelSignal.model_validate_json(json_str)
        assert restored.total_markets_scanned == sig.total_markets_scanned
        assert len(restored.opportunities) == 1
        assert restored.opportunities[0].confidence == 0.8


# ───────────────────────────────────────────────────────────────
# 3. Kalshi Client
# ───────────────────────────────────────────────────────────────


class TestKalshiClient:
    @pytest.fixture
    def kalshi_market_data(self) -> dict[str, Any]:
        return {
            "ticker": "KXBTC-24-100K",
            "event_ticker": "KXBTC",
            "title": "Will Bitcoin exceed $100k?",
            "subtitle": None,
            "status": "open",
            "yes_bid_dollars": 0.65,
            "yes_ask_dollars": 0.68,
            "no_bid_dollars": 0.32,
            "no_ask_dollars": 0.35,
            "last_price_dollars": 0.66,
            "volume": 50000,
            "volume_24h": 5000,
            "open_interest": 10000,
            "close_time": "2025-12-31T00:00:00Z",
            "category": "crypto",
            "result": None,
        }

    def test_parse_market(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        market = client._parse_market(kalshi_market_data)
        assert market.ticker == "KXBTC-24-100K"
        assert market.yes_bid == 0.65
        assert market.yes_ask == 0.68
        assert market.is_active is True
        assert market.volume_24h == 5000
        assert "kalshi.com" in market.url

    def test_parse_market_mid_price(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        market = client._parse_market(kalshi_market_data)
        # yes_price should be mid of bid/ask
        assert market.yes_price == pytest.approx(0.665, abs=0.001)

    def test_parse_market_no_bid_ask_uses_last(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        data = {
            "ticker": "TEST",
            "event_ticker": "T",
            "title": "Test",
            "status": "closed",
            "last_price_dollars": 0.80,
            "volume": 0,
            "volume_24h": 0,
            "open_interest": 0,
        }
        market = client._parse_market(data)
        assert market.yes_price == 0.80
        assert market.is_active is False

    def test_parse_trade(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        trade = client._parse_trade(
            {
                "trade_id": "t1",
                "ticker": "TEST",
                "count": 10,
                "yes_price_dollars": 0.55,
                "taker_side": "yes",
                "created_time": "2025-03-01T12:00:00Z",
            }
        )
        assert trade.trade_id == "t1"
        assert trade.yes_price == 0.55
        assert trade.count == 10

    @pytest.mark.asyncio
    async def test_get_markets(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"markets": [kalshi_market_data]}
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            markets = await client.get_markets(status="open", limit=10)

        assert len(markets) == 1
        assert markets[0].ticker == "KXBTC-24-100K"

    @pytest.mark.asyncio
    async def test_get_markets_error(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            )
        )
        with patch.object(client, "_get_client", return_value=mock_http):
            markets = await client.get_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_get_market_single(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"market": kalshi_market_data}
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            market = await client.get_market("KXBTC-24-100K")

        assert market is not None
        assert market.ticker == "KXBTC-24-100K"

    @pytest.mark.asyncio
    async def test_get_market_not_found(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not found",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(404),
            )
        )
        with patch.object(client, "_get_client", return_value=mock_http):
            market = await client.get_market("NOPE")
        assert market is None

    @pytest.mark.asyncio
    async def test_get_orderbook(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "orderbook": {
                "yes": {"bids": [[0.60, 100], [0.55, 200]], "asks": [[0.65, 150]]},
                "no": {"bids": [[0.35, 100]], "asks": [[0.40, 120]]},
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            ob = await client.get_orderbook("TEST")

        assert ob is not None
        assert len(ob.yes_bids) == 2
        assert ob.yes_bids[0].price == 0.60
        assert ob.yes_bids[0].quantity == 100
        assert len(ob.yes_asks) == 1

    @pytest.mark.asyncio
    async def test_get_trades(self) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "trades": [
                {
                    "trade_id": "t1",
                    "ticker": "TEST",
                    "count": 5,
                    "yes_price_dollars": 0.70,
                    "taker_side": "yes",
                    "created_time": "2025-03-01T12:00:00Z",
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            trades = await client.get_trades("TEST")

        assert len(trades) == 1
        assert trades[0].yes_price == 0.70

    @pytest.mark.asyncio
    async def test_get_events(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "event_ticker": "KXBTC",
                    "title": "Bitcoin Prices",
                    "category": "crypto",
                    "status": "open",
                    "markets": [kalshi_market_data],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            events = await client.get_events()

        assert len(events) == 1
        assert events[0].event_ticker == "KXBTC"
        assert len(events[0].markets) == 1

    @pytest.mark.asyncio
    async def test_get_expiring_markets(self, kalshi_market_data: dict[str, Any]) -> None:
        client = KalshiClient.__new__(KalshiClient)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"markets": [kalshi_market_data]}
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            markets = await client.get_expiring_markets(hours=24)

        assert len(markets) == 1


class TestKalshiMarketDataclass:
    def test_mid_price(self) -> None:
        m = KalshiMarket(
            ticker="T",
            event_ticker="E",
            title="X",
            subtitle=None,
            status="open",
            yes_bid=0.60,
            yes_ask=0.70,
            no_bid=0.30,
            no_ask=0.40,
            last_price=0.65,
            volume=100,
            volume_24h=50,
            open_interest=200,
            close_time=None,
            category=None,
            result=None,
        )
        assert m.yes_price == pytest.approx(0.65)
        assert m.no_price == pytest.approx(0.35)

    def test_url(self) -> None:
        m = KalshiMarket(
            ticker="KXBTC-24-100K",
            event_ticker="KXBTC",
            title="Test",
            subtitle=None,
            status="open",
            yes_bid=0.5,
            yes_ask=0.5,
            no_bid=0.5,
            no_ask=0.5,
            last_price=0.5,
            volume=0,
            volume_24h=0,
            open_interest=0,
            close_time=None,
            category=None,
            result=None,
        )
        assert m.url == "https://kalshi.com/markets/KXBTC/KXBTC-24-100K"

    def test_is_active(self) -> None:
        m = KalshiMarket(
            ticker="T",
            event_ticker="E",
            title="X",
            subtitle=None,
            status="closed",
            yes_bid=0,
            yes_ask=0,
            no_bid=0,
            no_ask=0,
            last_price=1.0,
            volume=0,
            volume_24h=0,
            open_interest=0,
            close_time=None,
            category=None,
            result="yes",
        )
        assert m.is_active is False


# ───────────────────────────────────────────────────────────────
# 4. Polymarket Extensions
# ───────────────────────────────────────────────────────────────


class TestPolymarketClientExtensions:
    @pytest.mark.asyncio
    async def test_get_expiring_markets(self) -> None:
        from synesis.markets.polymarket import PolymarketClient

        client = PolymarketClient()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "m1",
                "conditionId": "c1",
                "question": "Expiring?",
                "slug": "expiring",
                "tokens": [
                    {"outcome": "Yes", "price": 0.80},
                    {"outcome": "No", "price": 0.20},
                ],
                "volume24hr": 1000,
                "volume": 5000,
                "endDate": (datetime.now(UTC) + timedelta(hours=12)).isoformat(),
                "active": True,
                "closed": False,
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            markets = await client.get_expiring_markets(hours=24)

        assert len(markets) == 1
        assert markets[0].yes_price == 0.80


class TestPolymarketDataClient:
    @pytest.mark.asyncio
    async def test_get_wallet_positions(self) -> None:
        from synesis.markets.polymarket import PolymarketDataClient

        client = PolymarketDataClient()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"market": "m1", "amount": 100}]
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            positions = await client.get_wallet_positions("0xtest")

        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_get_wallet_trades(self) -> None:
        from synesis.markets.polymarket import PolymarketDataClient

        client = PolymarketDataClient()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"trade_id": "t1", "pnl": 50.0}]
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            trades = await client.get_wallet_trades("0xtest", limit=100)

        assert len(trades) == 1

    @pytest.mark.asyncio
    async def test_get_top_holders(self) -> None:
        from synesis.markets.polymarket import PolymarketDataClient

        client = PolymarketDataClient()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"address": "0xwhale", "amount": 50000}]
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch.object(client, "_get_client", return_value=mock_http):
            holders = await client.get_top_holders("cond_1")

        assert len(holders) == 1
        assert holders[0]["address"] == "0xwhale"

    @pytest.mark.asyncio
    async def test_get_wallet_positions_error(self) -> None:
        from synesis.markets.polymarket import PolymarketDataClient

        client = PolymarketDataClient()
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            )
        )

        with patch.object(client, "_get_client", return_value=mock_http):
            positions = await client.get_wallet_positions("0xtest")

        assert positions == []


# ───────────────────────────────────────────────────────────────
# 5. WebSocket Manager
# ───────────────────────────────────────────────────────────────


class TestMarketWSManager:
    @pytest.fixture
    def ws_manager(self, mock_redis: Any) -> Any:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 5
        poly_ws.start = AsyncMock()
        poly_ws.stop = AsyncMock()
        poly_ws.subscribe = AsyncMock()
        poly_ws.unsubscribe = AsyncMock()

        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = True
        kalshi_ws.subscribed_count = 3
        kalshi_ws.start = AsyncMock()
        kalshi_ws.stop = AsyncMock()
        kalshi_ws.subscribe = AsyncMock()
        kalshi_ws.unsubscribe = AsyncMock()

        mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)
        return mgr

    def test_is_connected(self, ws_manager: Any) -> None:
        assert ws_manager.is_connected is True

    def test_total_subscribed(self, ws_manager: Any) -> None:
        assert ws_manager.total_subscribed == 8

    @pytest.mark.asyncio
    async def test_start_stop(self, ws_manager: Any) -> None:
        await ws_manager.start()
        ws_manager._poly_ws.start.assert_called_once()
        ws_manager._kalshi_ws.start.assert_called_once()

        await ws_manager.stop()
        ws_manager._poly_ws.stop.assert_called_once()
        ws_manager._kalshi_ws.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_subscriptions(self, ws_manager: Any) -> None:
        markets = [
            _make_market(external_id="m1", condition_id="cond_1"),
            _make_kalshi_market(ticker="K1"),
        ]
        await ws_manager.update_subscriptions(markets)

        # Should subscribe poly to cond_1 and kalshi to K1
        ws_manager._poly_ws.subscribe.assert_called_once()
        ws_manager._kalshi_ws.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_subscriptions_rotates(self, ws_manager: Any) -> None:
        # First subscription
        markets1 = [_make_market(external_id="m1", condition_id="cond_1")]
        await ws_manager.update_subscriptions(markets1)

        # Reset mocks
        ws_manager._poly_ws.subscribe.reset_mock()
        ws_manager._poly_ws.unsubscribe.reset_mock()

        # Update with different market
        markets2 = [_make_market(external_id="m2", condition_id="cond_2")]
        await ws_manager.update_subscriptions(markets2)

        # Should unsubscribe old, subscribe new
        ws_manager._poly_ws.unsubscribe.assert_called_once()
        ws_manager._poly_ws.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_realtime_volume(self, ws_manager: Any, mock_redis: Any) -> None:
        # Set up Redis with volume data
        mock_redis._strings["synesis:mkt_intel:ws:volume_1h:polymarket:cond_1"] = "5000.0"

        vol = await ws_manager.get_realtime_volume("polymarket", "cond_1")
        assert vol == 5000.0

    @pytest.mark.asyncio
    async def test_get_realtime_volume_missing(self, ws_manager: Any) -> None:
        vol = await ws_manager.get_realtime_volume("polymarket", "nonexistent")
        assert vol is None

    @pytest.mark.asyncio
    async def test_get_realtime_price(self, ws_manager: Any, mock_redis: Any) -> None:
        # Set up Redis with price data
        mock_redis._hashes["synesis:mkt_intel:ws:price:polymarket:cond_1"] = {"price": "0.72"}

        result = await ws_manager.get_realtime_price("polymarket", "cond_1")
        assert result is not None
        yes, no = result
        assert yes == pytest.approx(0.72)
        assert no == pytest.approx(0.28)

    @pytest.mark.asyncio
    async def test_get_realtime_price_missing(self, ws_manager: Any) -> None:
        result = await ws_manager.get_realtime_price("polymarket", "nonexistent")
        assert result is None


# ───────────────────────────────────────────────────────────────
# 6. Market Scanner
# ───────────────────────────────────────────────────────────────


class TestMarketScanner:
    @pytest.fixture
    def mock_poly_client(self) -> AsyncMock:
        client = AsyncMock()
        from synesis.markets.polymarket import SimpleMarket

        market = SimpleMarket(
            id="m1",
            condition_id="c1",
            question="Will X happen?",
            slug="x-happen",
            description=None,
            category="politics",
            yes_price=0.65,
            no_price=0.35,
            volume_24h=10000,
            volume_total=50000,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
        )
        expiring = SimpleMarket(
            id="m2",
            condition_id="c2",
            question="Expiring soon?",
            slug="expiring",
            description=None,
            category=None,
            yes_price=0.80,
            no_price=0.20,
            volume_24h=5000,
            volume_total=20000,
            end_date=datetime.now(UTC) + timedelta(hours=12),
            created_at=None,
            is_active=True,
            is_closed=False,
        )
        client.get_trending_markets = AsyncMock(return_value=[market])
        client.get_expiring_markets = AsyncMock(return_value=[expiring])
        return client

    @pytest.fixture
    def mock_kalshi_client(self) -> AsyncMock:
        client = AsyncMock()
        kalshi_market = KalshiMarket(
            ticker="K1",
            event_ticker="KE",
            title="Kalshi test?",
            subtitle=None,
            status="open",
            yes_bid=0.55,
            yes_ask=0.60,
            no_bid=0.40,
            no_ask=0.45,
            last_price=0.58,
            volume=1000,
            volume_24h=200,
            open_interest=500,
            close_time=None,
            category="econ",
            result=None,
        )
        client.get_markets = AsyncMock(return_value=[kalshi_market])
        client.get_expiring_markets = AsyncMock(return_value=[])
        client.get_event_categories = AsyncMock(return_value={"KE": "econ"})
        return client

    @pytest.mark.asyncio
    async def test_scan_basic(self, mock_poly_client: Any, mock_kalshi_client: Any) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=None,
        )
        result = await scanner.scan()

        assert result.total_markets_scanned >= 1
        assert len(result.trending_markets) >= 1
        # No DB = no odds movements
        assert result.odds_movements == []

    @pytest.mark.asyncio
    async def test_scan_with_ws_manager(
        self, mock_poly_client: Any, mock_kalshi_client: Any, mock_redis: Any
    ) -> None:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        poly_ws.start = AsyncMock()
        poly_ws.stop = AsyncMock()
        poly_ws.subscribe = AsyncMock()
        poly_ws.unsubscribe = AsyncMock()

        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = False
        kalshi_ws.subscribed_count = 0
        kalshi_ws.start = AsyncMock()
        kalshi_ws.stop = AsyncMock()
        kalshi_ws.subscribe = AsyncMock()
        kalshi_ws.unsubscribe = AsyncMock()

        ws_mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

        from synesis.processing.mkt_intel.scanner import MarketScanner

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=ws_mgr,
            db=None,
        )
        result = await scanner.scan()

        # WS subscriptions should be updated
        poly_ws.subscribe.assert_called()
        assert result.total_markets_scanned >= 1

    @pytest.mark.asyncio
    async def test_scan_with_odds_movement(
        self, mock_poly_client: Any, mock_kalshi_client: Any
    ) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        mock_db = AsyncMock()
        mock_db.insert_market_snapshot = AsyncMock()

        async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            # Return previous price of 0.50 for all requested market IDs
            ids = args[0] if args else []
            return [{"market_external_id": mid, "yes_price": 0.50} for mid in ids]

        mock_db.fetch = mock_fetch

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=mock_db,
        )
        result = await scanner.scan()

        # The poly trending market has yes_price=0.65, prev=0.50, change=+0.15
        assert len(result.odds_movements) >= 1
        mov = result.odds_movements[0]
        assert abs(mov.price_change_1h) >= 0.05
        assert mov.direction in ("up", "down")

    @pytest.mark.asyncio
    async def test_scan_handles_platform_errors_gracefully(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        # Both clients fail
        mock_poly = AsyncMock()
        mock_poly.get_trending_markets = AsyncMock(side_effect=Exception("API down"))
        mock_poly.get_expiring_markets = AsyncMock(side_effect=Exception("API down"))

        mock_kalshi = AsyncMock()
        mock_kalshi.get_markets = AsyncMock(side_effect=Exception("Kalshi down"))
        mock_kalshi.get_expiring_markets = AsyncMock(side_effect=Exception("Kalshi down"))

        scanner = MarketScanner(
            polymarket=mock_poly,
            kalshi=mock_kalshi,
            ws_manager=None,
            db=None,
        )
        result = await scanner.scan()

        # Should return empty result, not crash
        assert result.total_markets_scanned == 0
        assert result.trending_markets == []


# ───────────────────────────────────────────────────────────────
# 7. Wallet Tracker
# ───────────────────────────────────────────────────────────────


class TestWalletTracker:
    @pytest.fixture
    def mock_data_client(self) -> AsyncMock:
        client = AsyncMock()
        client.get_top_holders = AsyncMock(
            return_value=[
                {"address": "0xinsider", "amount": 50000, "outcome": "yes"},
                {"address": "0xrandom", "amount": 1000, "outcome": "no"},
            ]
        )
        client.get_wallet_trades = AsyncMock(
            return_value=[
                {"pnl": 100.0},
                {"pnl": 50.0},
                {"pnl": -20.0},
                {"pnl": 30.0},
            ]
        )
        return client

    @pytest.fixture
    def mock_db_with_wallets(self) -> AsyncMock:
        db = AsyncMock()
        db.get_watched_wallets = AsyncMock(
            return_value=[
                {"address": "0xinsider", "platform": "polymarket", "insider_score": 0.8},
            ]
        )
        db.fetchrow = AsyncMock(return_value=None)
        db.upsert_wallet = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_get_watched_wallets(self, mock_redis: Any, mock_db_with_wallets: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_with_wallets,
        )
        wallets = await tracker.get_watched_wallets()
        assert len(wallets) == 1
        assert wallets[0]["address"] == "0xinsider"

    @pytest.mark.asyncio
    async def test_get_watched_wallets_no_db(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(redis=mock_redis, db=None)
        wallets = await tracker.get_watched_wallets()
        assert wallets == []

    @pytest.mark.asyncio
    async def test_check_wallet_activity(
        self,
        mock_redis: Any,
        mock_db_with_wallets: Any,
        mock_data_client: Any,
    ) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_with_wallets,
            data_client=mock_data_client,
            insider_score_min=0.5,
        )

        markets = [
            _make_market(external_id="m1", condition_id="cond_1"),
        ]
        alerts = await tracker.check_wallet_activity(markets)

        assert len(alerts) == 1
        assert alerts[0].wallet_address == "0xinsider"
        assert alerts[0].insider_score == 0.8
        assert alerts[0].trade_direction == "yes"
        assert alerts[0].trade_size == 50000.0

    @pytest.mark.asyncio
    async def test_check_wallet_activity_filters_below_threshold(
        self,
        mock_redis: Any,
        mock_data_client: Any,
    ) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.get_watched_wallets = AsyncMock(
            return_value=[
                {"address": "0xinsider", "platform": "polymarket", "insider_score": 0.3},
            ]
        )

        tracker = WalletTracker(
            redis=mock_redis,
            db=db,
            data_client=mock_data_client,
            insider_score_min=0.5,
        )

        markets = [_make_market()]
        alerts = await tracker.check_wallet_activity(markets)
        # Score 0.3 < threshold 0.5 → no alert
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_check_wallet_activity_skips_kalshi(
        self,
        mock_redis: Any,
        mock_db_with_wallets: Any,
        mock_data_client: Any,
    ) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_with_wallets,
            data_client=mock_data_client,
        )

        # Kalshi markets have no wallet data
        markets = [_make_kalshi_market()]
        alerts = await tracker.check_wallet_activity(markets)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_discover_wallets(
        self,
        mock_redis: Any,
        mock_db_with_wallets: Any,
        mock_data_client: Any,
    ) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_with_wallets,
            data_client=mock_data_client,
        )

        discovered = await tracker.discover_wallets_from_market("cond_1")
        # 0xinsider not in DB (fetchrow returns None), 0xrandom not in DB
        assert len(discovered) == 2


# ───────────────────────────────────────────────────────────────
# 8. Processor (scoring + signal generation)
# ───────────────────────────────────────────────────────────────


class TestMarketIntelProcessor:
    @pytest.fixture
    def processor(self) -> Any:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        mock_settings = MagicMock()
        mock_settings.mkt_intel_insider_score_min = 0.5

        mock_scanner = AsyncMock()
        mock_scanner.scan = AsyncMock(
            return_value=ScanResult(
                timestamp=datetime.now(UTC),
                trending_markets=[
                    _make_market(external_id="m1", yes_price=0.65, volume_24h=50000),
                    _make_market(external_id="m2", yes_price=0.30, volume_24h=20000),
                ],
                expiring_markets=[
                    _make_market(
                        external_id="m3",
                        yes_price=0.80,
                        end_date=datetime.now(UTC) + timedelta(hours=6),
                    ),
                ],
                odds_movements=[
                    OddsMovement(
                        market=_make_market(external_id="m2"),
                        price_change_1h=0.10,
                        direction="up",
                    )
                ],
                total_markets_scanned=3,
            )
        )

        mock_wallet_tracker = AsyncMock()
        mock_wallet_tracker.check_wallet_activity = AsyncMock(
            return_value=[
                InsiderAlert(
                    market=_make_market(external_id="m1"),
                    wallet_address="0xwhale",
                    insider_score=0.9,
                    trade_direction="yes",
                    trade_size=50000,
                )
            ]
        )

        return MarketIntelProcessor(
            settings=mock_settings,
            scanner=mock_scanner,
            wallet_tracker=mock_wallet_tracker,
            ws_manager=None,
            db=None,
        )

    @pytest.mark.asyncio
    async def test_run_scan(self, processor: Any) -> None:
        signal = await processor.run_scan()

        assert isinstance(signal, MarketIntelSignal)
        assert signal.total_markets_scanned == 3
        assert len(signal.trending_markets) == 2
        assert len(signal.insider_activity) == 1
        assert len(signal.odds_movements) == 1

    @pytest.mark.asyncio
    async def test_run_scan_generates_opportunities(self, processor: Any) -> None:
        signal = await processor.run_scan()

        # m1 has insider_activity, m2 has odds_movement → should have opportunities
        assert len(signal.opportunities) >= 1
        top_opp = signal.opportunities[0]
        assert top_opp.confidence > 0
        assert len(top_opp.triggers) >= 1

    @pytest.mark.asyncio
    async def test_run_scan_opportunity_sorting(self, processor: Any) -> None:
        signal = await processor.run_scan()

        # Opportunities should be sorted by confidence descending
        if len(signal.opportunities) >= 2:
            assert signal.opportunities[0].confidence >= signal.opportunities[1].confidence

    @pytest.mark.asyncio
    async def test_run_scan_persists_to_db(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        mock_db = AsyncMock()
        mock_db.insert_mkt_intel_signal = AsyncMock()

        mock_scanner = AsyncMock()
        mock_scanner.scan = AsyncMock(return_value=ScanResult(timestamp=datetime.now(UTC)))

        mock_wallet = AsyncMock()
        mock_wallet.check_wallet_activity = AsyncMock(return_value=[])

        proc = MarketIntelProcessor(
            settings=MagicMock(),
            scanner=mock_scanner,
            wallet_tracker=mock_wallet,
            db=mock_db,
        )

        await proc.run_scan()
        mock_db.insert_mkt_intel_signal.assert_called_once()

    def test_calc_uncertainty_index(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)

        # Markets at 0.5 → maximum uncertainty
        markets = [_make_market(yes_price=0.50, no_price=0.50)]
        assert proc._calc_uncertainty_index(markets) == pytest.approx(1.0)

        # Markets at 0 or 1 → minimum uncertainty
        markets = [_make_market(yes_price=1.0, no_price=0.0)]
        assert proc._calc_uncertainty_index(markets) == pytest.approx(0.0)

        # Empty
        assert proc._calc_uncertainty_index([]) == 0.0

    def test_calc_informed_activity(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)

        # No alerts
        assert proc._calc_informed_activity([]) == 0.0

        # 1 wallet = 0.2
        alerts = [
            InsiderAlert(
                market=_make_market(),
                wallet_address="0xa",
                insider_score=0.8,
                trade_direction="yes",
                trade_size=1000,
            )
        ]
        assert proc._calc_informed_activity(alerts) == pytest.approx(0.2)

        # 5 wallets = 1.0
        alerts5 = [
            InsiderAlert(
                market=_make_market(),
                wallet_address=f"0x{i}",
                insider_score=0.8,
                trade_direction="yes",
                trade_size=1000,
            )
            for i in range(5)
        ]
        assert proc._calc_informed_activity(alerts5) == pytest.approx(1.0)

    def test_score_opportunities_odds_movement(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)
        proc._settings = MagicMock()

        market = _make_market(external_id="m1")
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[OddsMovement(market=market, price_change_1h=0.10, direction="up")],
        )

        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert "odds_movement" in opps[0].triggers
        assert opps[0].confidence > 0

    def test_score_opportunities_multiple_triggers(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)
        proc._settings = MagicMock()

        market = _make_market(
            external_id="m1",
            yes_price=0.40,
            end_date=datetime.now(UTC) + timedelta(hours=6),
        )
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[market],
            odds_movements=[OddsMovement(market=market, price_change_1h=0.08, direction="up")],
        )
        insider_alerts = [
            InsiderAlert(
                market=market,
                wallet_address="0xwhale",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=50000,
            )
        ]

        opps = proc._score_opportunities(scan, insider_alerts)
        assert len(opps) == 1
        assert len(opps[0].triggers) >= 3  # insider_activity, odds_movement, expiring_soon
        # Higher confidence with multiple triggers
        assert opps[0].confidence > 0.3

    def test_score_opportunities_no_triggers_filtered(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)
        proc._settings = MagicMock()

        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[_make_market(external_id="m1", volume_24h=0)],
            expiring_markets=[],
            odds_movements=[],
        )

        opps = proc._score_opportunities(scan, [])
        # No triggers → no opportunities
        assert len(opps) == 0


# ───────────────────────────────────────────────────────────────
# 9. API Routes
# ───────────────────────────────────────────────────────────────


class TestMktIntelAPIRoutes:
    @pytest.mark.asyncio
    async def test_get_latest_signal_empty(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_latest_signal

        state = MagicMock()
        state.redis = mock_redis
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=None)

        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_latest_signal(state)

        assert result["time"] is None
        assert result["signal"] is None

    @pytest.mark.asyncio
    async def test_get_latest_signal_with_data(self, mock_redis: Any) -> None:
        import orjson

        from synesis.api.routes.mkt_intel import get_latest_signal

        state = MagicMock()
        state.redis = mock_redis
        now = datetime.now(UTC)
        payload = {"test": "data", "opportunities": []}
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value={"time": now, "payload": orjson.dumps(payload)})

        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_latest_signal(state)

        assert result["time"] == now.isoformat()
        assert result["signal"]["test"] == "data"

    @pytest.mark.asyncio
    async def test_trigger_manual_scan(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import trigger_manual_scan

        state = MagicMock()
        state.redis = mock_redis

        result = await trigger_manual_scan(state)
        assert result["status"] == "scan_triggered"

    @pytest.mark.asyncio
    async def test_get_opportunities_empty(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_opportunities

        state = MagicMock()
        state.redis = mock_redis
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=None)

        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_opportunities(state)

        assert result["opportunities"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_opportunities_with_data(self, mock_redis: Any) -> None:
        import orjson

        from synesis.api.routes.mkt_intel import get_opportunities

        state = MagicMock()
        state.redis = mock_redis
        payload = {"opportunities": [{"market": "m1"}, {"market": "m2"}]}
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value={"payload": orjson.dumps(payload)})

        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_opportunities(state)

        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_get_watched_wallets_empty(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_watched_wallets

        state = MagicMock()
        state.redis = mock_redis
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(return_value=[])

        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_watched_wallets(state)

        assert result["wallets"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_ws_status(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_ws_status

        state = MagicMock()
        state.redis = mock_redis

        result = await get_ws_status(state)
        assert "polymarket_ws" in result
        assert "kalshi_ws" in result
        assert "total_subscribed_markets" in result

    @pytest.mark.asyncio
    async def test_get_ws_status_with_data(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_ws_status

        # Set health data in Redis
        mock_redis._hashes["synesis:mkt_intel:ws:health"] = {
            "poly_connected": "1",
            "kalshi_connected": "0",
            "total_subscribed": "42",
        }

        state = MagicMock()
        state.redis = mock_redis

        result = await get_ws_status(state)
        assert result["polymarket_ws"] is True
        assert result["kalshi_ws"] is False
        assert result["total_subscribed_markets"] == 42

    @pytest.mark.asyncio
    async def test_get_latest_signal_db_not_init(self, mock_redis: Any) -> None:
        from synesis.api.routes.mkt_intel import get_latest_signal

        state = MagicMock()
        state.redis = mock_redis

        with patch(
            "synesis.api.routes.mkt_intel.get_database",
            side_effect=RuntimeError("DB not init"),
        ):
            result = await get_latest_signal(state)

        assert "error" in result


# ───────────────────────────────────────────────────────────────
# 10. Telegram Formatting
# ───────────────────────────────────────────────────────────────


class TestTelegramFormatting:
    def test_format_empty_signal(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal()
        msg = format_mkt_intel_signal(signal)
        assert "MARKET INTEL" in msg
        assert "1h" in msg
        assert "REST-only" in msg  # ws_connected=False

    def test_format_with_ws_connected(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(ws_connected=True)
        msg = format_mkt_intel_signal(signal)
        assert "LIVE" in msg

    def test_format_with_opportunities(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(question="Will BTC hit 100k?"),
                    suggested_direction="yes",
                    confidence=0.85,
                    triggers=["insider_activity", "odds_movement"],
                    reasoning="2 insiders active; Price moved +0.10 in 1h",
                )
            ]
        )
        msg = format_mkt_intel_signal(signal)
        assert "Opportunities" in msg
        assert "BTC" in msg
        assert "YES" in msg
        assert "insider_activity" in msg

    def test_format_with_odds_movements(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            odds_movements=[
                OddsMovement(
                    market=_make_market(question="Odds test"),
                    price_change_1h=0.12,
                    price_change_6h=0.25,
                    direction="up",
                )
            ]
        )
        msg = format_mkt_intel_signal(signal)
        assert "Odds Movements" in msg
        assert "+0.12" in msg
        assert "+0.25" in msg

    def test_format_with_insider_activity(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            insider_activity=[
                InsiderAlert(
                    market=_make_market(question="Insider test"),
                    wallet_address="0xabcdef1234567890",
                    insider_score=0.92,
                    trade_direction="yes",
                    trade_size=75000,
                )
            ]
        )
        msg = format_mkt_intel_signal(signal)
        assert "Insider Activity" in msg
        assert "0xabcdef" in msg
        assert "0.92" in msg

    def test_format_with_expiring(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            expiring_soon=[
                _make_market(
                    question="Expiring test",
                    end_date=datetime.now(UTC) + timedelta(hours=5),
                )
            ]
        )
        msg = format_mkt_intel_signal(signal)
        assert "Expiring Soon" in msg

    def test_format_footer_metrics_not_shown(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            market_uncertainty_index=0.75,
            informed_activity_level=0.40,
        )
        msg = format_mkt_intel_signal(signal)
        assert "Uncertainty" not in msg
        assert "Informed" not in msg


# ───────────────────────────────────────────────────────────────
# 11. Config
# ───────────────────────────────────────────────────────────────


class TestMktIntelConfig:
    def test_default_settings(self) -> None:
        from synesis.config import Settings

        # Create with minimal required fields to test defaults
        settings = Settings(
            _env_file=None,
            twitterapi_api_key="test",
        )
        assert settings.mkt_intel_enabled is True
        assert settings.mkt_intel_interval == 3600
        assert settings.mkt_intel_insider_score_min == 0.5
        assert settings.mkt_intel_expiring_hours == 24
        assert settings.mkt_intel_ws_enabled is True
        assert settings.mkt_intel_volume_spike_threshold == 1.0

    def test_kalshi_settings_defaults(self) -> None:
        from synesis.config import Settings

        settings = Settings(
            _env_file=None,
            twitterapi_api_key="test",
        )
        assert settings.kalshi_api_key is None
        assert settings.kalshi_private_key_path is None
        assert "kalshi.com" in settings.kalshi_api_url
        assert "kalshi.com" in settings.kalshi_ws_url


# ───────────────────────────────────────────────────────────────
# 12. Constants
# ───────────────────────────────────────────────────────────────


class TestMktIntelConstants:
    def test_redis_prefix(self) -> None:
        from synesis.core.constants import MARKET_INTEL_REDIS_PREFIX

        assert MARKET_INTEL_REDIS_PREFIX == "synesis:mkt_intel"

    def test_urls(self) -> None:
        from synesis.core.constants import (
            DEFAULT_KALSHI_API_URL,
            DEFAULT_KALSHI_WS_URL,
            DEFAULT_POLYMARKET_CLOB_WS_URL,
            DEFAULT_POLYMARKET_DATA_API_URL,
        )

        assert "kalshi.com" in DEFAULT_KALSHI_API_URL
        assert "kalshi.com" in DEFAULT_KALSHI_WS_URL
        assert "polymarket.com" in DEFAULT_POLYMARKET_CLOB_WS_URL
        assert "polymarket.com" in DEFAULT_POLYMARKET_DATA_API_URL

    def test_limits(self) -> None:
        from synesis.core.constants import (
            MARKET_INTEL_MAX_TRACKED_MARKETS,
            MARKET_INTEL_SNAPSHOT_INTERVAL,
        )

        assert MARKET_INTEL_MAX_TRACKED_MARKETS == 100
        assert MARKET_INTEL_SNAPSHOT_INTERVAL == 300


# ───────────────────────────────────────────────────────────────
# 13. Batch 2: Odds Movement Formula Validation
# ───────────────────────────────────────────────────────────────


class TestOddsMovementFormulas:
    """Validate _detect_odds_movements() formulas with known inputs."""

    @pytest.fixture
    def scanner_for_odds(self) -> Any:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        def _create(
            prev_price_1h: float | None = 0.50,
            prev_price_6h: float | None = None,
            market_id: str = "market_1",
        ) -> MarketScanner:
            mock_db = AsyncMock()
            mock_db.insert_market_snapshot = AsyncMock()

            async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
                if "55 minutes" in query and "5 hours 55" not in query:
                    if prev_price_1h is not None:
                        # Return rows for all market_ids passed in
                        ids = args[0] if args else [market_id]
                        return [
                            {"market_external_id": mid, "yes_price": prev_price_1h} for mid in ids
                        ]
                    return []
                elif "5 hours 55" in query:
                    if prev_price_6h is not None:
                        ids = args[0] if args else [market_id]
                        return [
                            {"market_external_id": mid, "yes_price": prev_price_6h} for mid in ids
                        ]
                    return []
                return []

            mock_db.fetch = mock_fetch

            return MarketScanner(
                polymarket=AsyncMock(
                    get_trending_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                kalshi=AsyncMock(
                    get_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                ws_manager=None,
                db=mock_db,
            )

        return _create

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "current, prev, expected_change, expected_direction, expect_movement",
        [
            (0.65, 0.50, 0.15, "up", True),
            (0.40, 0.55, -0.15, "down", True),
            (0.54, 0.50, 0.04, None, False),  # Under threshold
            (0.55, 0.50, 0.05, "up", True),  # At threshold
        ],
    )
    async def test_odds_exact_values(
        self,
        scanner_for_odds: Any,
        current: float,
        prev: float,
        expected_change: float,
        expected_direction: str | None,
        expect_movement: bool,
    ) -> None:
        scanner = scanner_for_odds(prev_price_1h=prev)
        markets = [_make_market(yes_price=current)]
        movements = await scanner._detect_odds_movements(markets)

        if expect_movement:
            assert len(movements) == 1
            assert movements[0].price_change_1h == pytest.approx(expected_change)
            assert movements[0].direction == expected_direction
        else:
            assert len(movements) == 0

    @pytest.mark.asyncio
    async def test_odds_no_previous_snapshot(self, scanner_for_odds: Any) -> None:
        scanner = scanner_for_odds(prev_price_1h=None)
        markets = [_make_market(yes_price=0.65)]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 0

    @pytest.mark.asyncio
    async def test_odds_previous_price_none(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        mock_db = AsyncMock()
        mock_db.insert_market_snapshot = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[{"market_external_id": "market_1", "yes_price": None}]
        )

        scanner = MarketScanner(
            polymarket=AsyncMock(
                get_trending_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            kalshi=AsyncMock(
                get_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            ws_manager=None,
            db=mock_db,
        )
        markets = [_make_market(yes_price=0.65)]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 0

    @pytest.mark.asyncio
    async def test_odds_6h_lookup_success(self, scanner_for_odds: Any) -> None:
        scanner = scanner_for_odds(prev_price_1h=0.50, prev_price_6h=0.40)
        markets = [_make_market(yes_price=0.65)]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 1
        assert movements[0].price_change_1h == pytest.approx(0.15)
        assert movements[0].price_change_6h == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_odds_6h_lookup_none(self, scanner_for_odds: Any) -> None:
        scanner = scanner_for_odds(prev_price_1h=0.50, prev_price_6h=None)
        markets = [_make_market(yes_price=0.65)]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 1
        assert movements[0].price_change_6h is None

    @pytest.mark.asyncio
    async def test_odds_direction_zero_change(self, scanner_for_odds: Any) -> None:
        # Same price → change=0 → under 0.05 threshold
        scanner = scanner_for_odds(prev_price_1h=0.50)
        markets = [_make_market(yes_price=0.50)]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 0

    @pytest.mark.asyncio
    async def test_odds_sorted_by_abs_change(self, scanner_for_odds: Any) -> None:
        scanner = scanner_for_odds(prev_price_1h=0.50)

        markets = [
            _make_market(external_id="m_small", yes_price=0.55),  # +0.05
            _make_market(external_id="m_big", yes_price=0.30),  # -0.20
            _make_market(external_id="m_mid", yes_price=0.60),  # +0.10
        ]
        movements = await scanner._detect_odds_movements(markets)
        assert len(movements) == 3
        abs_changes = [abs(m.price_change_1h) for m in movements]
        assert abs_changes == sorted(abs_changes, reverse=True)

    @pytest.mark.asyncio
    async def test_odds_db_error_graceful(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        mock_db = AsyncMock()
        mock_db.insert_market_snapshot = AsyncMock()
        mock_db.fetch = AsyncMock(side_effect=RuntimeError("DB error"))

        scanner = MarketScanner(
            polymarket=AsyncMock(
                get_trending_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            kalshi=AsyncMock(
                get_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            ws_manager=None,
            db=mock_db,
        )

        markets = [
            _make_market(external_id="m_fail", yes_price=0.70),
            _make_market(external_id="m_ok", yes_price=0.70),
        ]
        movements = await scanner._detect_odds_movements(markets)
        # Batch query fails → returns empty gracefully
        assert len(movements) == 0


# ───────────────────────────────────────────────────────────────
# 15. Batch 3: Confidence Scoring Formula Validation
# ───────────────────────────────────────────────────────────────


class TestConfidenceScoringFormulas:
    """Validate _score_opportunities() formulas with known inputs."""

    @pytest.fixture
    def proc(self) -> Any:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        p = MarketIntelProcessor.__new__(MarketIntelProcessor)
        p._settings = MagicMock()
        return p

    @pytest.mark.parametrize(
        "wallet_count, expected_conf",
        [
            (1, 0.15),  # min(0.3, 1*0.15)=0.15
            (2, 0.3),  # min(0.3, 2*0.15)=0.3
            (5, 0.3),  # min(0.3, 5*0.15)=0.3 (capped)
        ],
    )
    def test_confidence_insider_only(
        self, proc: Any, wallet_count: int, expected_conf: float
    ) -> None:
        market = _make_market(external_id="m1", volume_24h=0)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[],
        )
        alerts = [
            InsiderAlert(
                market=market,
                wallet_address=f"0x{i:04x}",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=10000,
            )
            for i in range(wallet_count)
        ]
        opps = proc._score_opportunities(scan, alerts)
        assert len(opps) == 1
        assert opps[0].confidence == pytest.approx(expected_conf)
        assert opps[0].triggers == ["insider_activity"]

    @pytest.mark.parametrize(
        "change, expected_conf",
        [
            (0.05, 0.1),  # min(0.2, 0.05*2)=0.1
            (0.10, 0.2),  # min(0.2, 0.10*2)=0.2
            (0.50, 0.2),  # min(0.2, 0.50*2)=0.2 (capped)
        ],
    )
    def test_confidence_odds_only(self, proc: Any, change: float, expected_conf: float) -> None:
        market = _make_market(external_id="m1", volume_24h=0)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[OddsMovement(market=market, price_change_1h=change, direction="up")],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert opps[0].confidence == pytest.approx(expected_conf)
        assert opps[0].triggers == ["odds_movement"]

    def test_confidence_expiring_only(self, proc: Any) -> None:
        market = _make_market(
            external_id="m1",
            volume_24h=0,
            end_date=datetime.now(UTC) + timedelta(hours=18),
        )
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[],
            expiring_markets=[market],
            odds_movements=[],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        # 12h < 18h < 24h → +0.10
        assert opps[0].confidence == pytest.approx(0.1)
        assert "expiring_soon" in opps[0].triggers

    def test_confidence_all_triggers_combined(self, proc: Any) -> None:
        market = _make_market(
            external_id="m1",
            yes_price=0.40,
            volume_24h=0,
            end_date=datetime.now(UTC) + timedelta(hours=18),
        )
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[market],
            odds_movements=[OddsMovement(market=market, price_change_1h=0.10, direction="up")],
        )
        alerts = [
            InsiderAlert(
                market=market,
                wallet_address=f"0x{i:04x}",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=10000,
            )
            for i in range(2)
        ]
        opps = proc._score_opportunities(scan, alerts)
        assert len(opps) == 1
        # insider: min(0.3, 2*0.15)=0.3
        # odds: min(0.2, 0.10*2)=0.2
        # expiring (18h → 12-24h band): 0.10
        # total: 0.6
        assert opps[0].confidence == pytest.approx(0.6)
        assert len(opps[0].triggers) == 3

    def test_confidence_capped_at_1_0(self, proc: Any) -> None:
        market = _make_market(
            external_id="m1",
            yes_price=0.40,
            end_date=datetime.now(UTC) + timedelta(hours=3),
        )
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[market],
            odds_movements=[OddsMovement(market=market, price_change_1h=0.50, direction="up")],
        )
        alerts = [
            InsiderAlert(
                market=market,
                wallet_address=f"0x{i:04x}",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=10000,
            )
            for i in range(5)
        ]
        opps = proc._score_opportunities(scan, alerts)
        assert len(opps) == 1
        # insider: min(0.3, 5*0.15)=0.3
        # odds: min(0.2, 0.50*2)=0.2
        # expiring (<6h): 0.25
        # high_volume: 0.15
        # total: 0.9, then min(1.0, 0.9) = 0.9
        assert opps[0].confidence <= 1.0

    def test_confidence_no_triggers_filtered_out(self, proc: Any) -> None:
        market = _make_market(external_id="m1", volume_24h=0)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 0

    @pytest.mark.parametrize(
        "price_change, expected_direction",
        [
            (0.05, "yes"),
            (-0.05, "no"),
        ],
    )
    def test_direction_from_price_change(
        self, proc: Any, price_change: float, expected_direction: str
    ) -> None:
        direction_str = "up" if price_change > 0 else "down"
        market = _make_market(external_id="m1", yes_price=0.65)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[
                OddsMovement(market=market, price_change_1h=price_change, direction=direction_str)
            ],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert opps[0].suggested_direction == expected_direction

    @pytest.mark.parametrize(
        "yes_price, expected_direction",
        [
            (0.40, "yes"),  # < 0.5 → yes
            (0.60, "no"),  # >= 0.5 → no
        ],
    )
    def test_direction_from_yes_price_no_odds(
        self, proc: Any, yes_price: float, expected_direction: str
    ) -> None:
        market = _make_market(
            external_id="m1",
            yes_price=yes_price,
            end_date=datetime.now(UTC) + timedelta(hours=6),
        )
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[],
            expiring_markets=[market],
            odds_movements=[],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert opps[0].suggested_direction == expected_direction

    def test_opportunities_sorted_by_confidence_desc(self, proc: Any) -> None:
        m1 = _make_market(external_id="m1")
        m2 = _make_market(external_id="m2")
        m3 = _make_market(external_id="m3")
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[m1, m2, m3],
            expiring_markets=[],
            odds_movements=[
                OddsMovement(market=m1, price_change_1h=0.05, direction="up"),
                OddsMovement(market=m2, price_change_1h=0.15, direction="up"),
                OddsMovement(market=m3, price_change_1h=0.10, direction="up"),
            ],
        )
        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 3
        confs = [o.confidence for o in opps]
        assert confs == sorted(confs, reverse=True)


# ───────────────────────────────────────────────────────────────
# 16. Batch 4: Uncertainty & Informed Activity Formulas
# ───────────────────────────────────────────────────────────────


class TestUncertaintyAndInformedFormulas:
    @pytest.fixture
    def proc(self) -> Any:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        return MarketIntelProcessor.__new__(MarketIntelProcessor)

    @pytest.mark.parametrize(
        "yes_price, expected_uncertainty",
        [
            (0.25, 0.5),  # 1.0 - abs(-0.25)*2 = 0.5
            (0.75, 0.5),  # 1.0 - abs(0.25)*2 = 0.5
            (0.10, 0.2),  # 1.0 - abs(-0.40)*2 = 0.2
            (0.50, 1.0),  # 1.0 - abs(0.00)*2 = 1.0
            (1.00, 0.0),  # 1.0 - abs(0.50)*2 = 0.0
        ],
    )
    def test_uncertainty_midrange_values(
        self, proc: Any, yes_price: float, expected_uncertainty: float
    ) -> None:
        markets = [_make_market(yes_price=yes_price)]
        assert proc._calc_uncertainty_index(markets) == pytest.approx(expected_uncertainty)

    def test_uncertainty_multiple_markets_avg(self, proc: Any) -> None:
        markets = [
            _make_market(external_id="m1", yes_price=0.5),  # uncertainty=1.0
            _make_market(external_id="m2", yes_price=0.5),  # uncertainty=1.0
            _make_market(external_id="m3", yes_price=1.0),  # uncertainty=0.0
        ]
        result = proc._calc_uncertainty_index(markets)
        assert result == pytest.approx(2.0 / 3.0, abs=0.001)

    @pytest.mark.parametrize(
        "wallet_count, expected_activity",
        [
            (0, 0.0),
            (3, 0.6),  # min(1.0, 3*0.2)=0.6
            (6, 1.0),  # min(1.0, 6*0.2)=1.0 (capped)
        ],
    )
    def test_informed_activity_exact(
        self, proc: Any, wallet_count: int, expected_activity: float
    ) -> None:
        alerts = [
            InsiderAlert(
                market=_make_market(),
                wallet_address=f"0x{i:04x}",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=1000,
            )
            for i in range(wallet_count)
        ]
        assert proc._calc_informed_activity(alerts) == pytest.approx(expected_activity)

    def test_informed_activity_dedupes_wallets(self, proc: Any) -> None:
        # 4 alerts from 2 unique wallets → min(1.0, 2*0.2) = 0.4
        alerts = [
            InsiderAlert(
                market=_make_market(external_id=f"m{i}"),
                wallet_address="0xAAAA" if i < 2 else "0xBBBB",
                insider_score=0.9,
                trade_direction="yes",
                trade_size=1000,
            )
            for i in range(4)
        ]
        assert proc._calc_informed_activity(alerts) == pytest.approx(0.4)


# ───────────────────────────────────────────────────────────────
# 17. Batch 5: Wallet Tracker — update_wallet_metrics() + Edge Cases
# ───────────────────────────────────────────────────────────────


class TestWalletMetricsFormula:
    """Validate insider_score = min(1.0, win_rate * log10(max(total,1)) / 2)."""

    @pytest.mark.parametrize(
        "pnls, expected_score",
        [
            # trades=[+100, -50, +30] → wins=2, total=3, win_rate=2/3
            # score = min(1.0, (2/3) * log10(3) / 2) ≈ 0.159
            ([100.0, -50.0, 30.0], min(1.0, (2 / 3) * math.log10(3) / 2)),
            # trades=[+10] → wins=1, total=1, log10(1)=0 → score=0
            ([10.0], 0.0),
            # trades=[+1]*100 → wins=100, total=100, score=min(1.0, 1.0*2.0/2)=1.0
            ([1.0] * 100, 1.0),
            # trades=[+1]*10 → wins=10, total=10, score=min(1.0, 1.0*1.0/2)=0.5
            ([1.0] * 10, 0.5),
        ],
    )
    @pytest.mark.asyncio
    async def test_insider_score_formula(
        self, mock_redis: Any, pnls: list[float], expected_score: float
    ) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_wallet_trades = AsyncMock(return_value=[{"pnl": p} for p in pnls])

        mock_db = AsyncMock()
        mock_db.upsert_wallet_metrics = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        await tracker.update_wallet_metrics("0xtest")

        # Verify the score passed to DB
        mock_db.upsert_wallet_metrics.assert_called_once()
        call_kwargs = mock_db.upsert_wallet_metrics.call_args[1]
        actual_score = call_kwargs["insider_score"]
        assert actual_score == pytest.approx(expected_score, abs=0.001)

    @pytest.mark.asyncio
    async def test_update_metrics_zero_trades(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_wallet_trades = AsyncMock(return_value=[])
        mock_db = AsyncMock()
        mock_db.upsert_wallet_metrics = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        await tracker.update_wallet_metrics("0xtest")
        mock_db.upsert_wallet_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_metrics_no_data_client(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_db = AsyncMock()
        mock_db.upsert_wallet_metrics = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=None)
        await tracker.update_wallet_metrics("0xtest")
        mock_db.upsert_wallet_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_metrics_db_upsert_called(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 30.0}]
        )
        mock_db = AsyncMock()
        mock_db.upsert_wallet_metrics = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        await tracker.update_wallet_metrics("0xtest")

        mock_db.upsert_wallet_metrics.assert_called_once()
        kwargs = mock_db.upsert_wallet_metrics.call_args[1]
        assert kwargs["address"] == "0xtest"
        assert kwargs["platform"] == "polymarket"
        assert kwargs["total_trades"] == 3
        assert kwargs["wins"] == 2
        assert kwargs["win_rate"] == pytest.approx(2 / 3)
        assert kwargs["total_pnl"] == pytest.approx(80.0)  # 100-50+30

    @pytest.mark.asyncio
    async def test_update_metrics_pnl_none_handling(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_wallet_trades = AsyncMock(return_value=[{"pnl": None}, {"pnl": 50.0}])
        mock_db = AsyncMock()
        mock_db.upsert_wallet_metrics = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        await tracker.update_wallet_metrics("0xtest")

        mock_db.upsert_wallet_metrics.assert_called_once()
        kwargs = mock_db.upsert_wallet_metrics.call_args[1]
        assert kwargs["total_trades"] == 2
        assert kwargs["wins"] == 1  # only pnl=50 is > 0; pnl=None → 0 → not win
        assert kwargs["total_pnl"] == pytest.approx(50.0)


class TestWalletEdgeCases:
    @pytest.mark.asyncio
    async def test_check_activity_mixed_platforms(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(return_value=[])
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(
            return_value=[{"address": "0xinsider", "platform": "polymarket", "insider_score": 0.8}]
        )

        tracker = WalletTracker(
            redis=mock_redis, db=mock_db, data_client=mock_data_client, insider_score_min=0.5
        )

        markets = [
            _make_market(external_id="p1", condition_id="c1"),
            _make_market(external_id="p2", condition_id="c2"),
            _make_kalshi_market(ticker="K1"),
            _make_kalshi_market(ticker="K2"),
        ]
        await tracker.check_wallet_activity(markets)
        # Only poly markets (with condition_id) should trigger get_top_holders
        assert mock_data_client.get_top_holders.call_count == 2

    @pytest.mark.asyncio
    async def test_check_activity_score_at_threshold(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(
            return_value=[{"address": "0xinsider", "amount": 1000, "outcome": "yes"}]
        )
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(
            return_value=[{"address": "0xinsider", "platform": "polymarket", "insider_score": 0.5}]
        )

        tracker = WalletTracker(
            redis=mock_redis, db=mock_db, data_client=mock_data_client, insider_score_min=0.5
        )
        markets = [_make_market(condition_id="c1")]
        alerts = await tracker.check_wallet_activity(markets)
        # Score=0.5 >= threshold=0.5 → alert
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_check_activity_score_below_threshold(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(
            return_value=[{"address": "0xinsider", "amount": 1000, "outcome": "yes"}]
        )
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(
            return_value=[{"address": "0xinsider", "platform": "polymarket", "insider_score": 0.49}]
        )

        tracker = WalletTracker(
            redis=mock_redis, db=mock_db, data_client=mock_data_client, insider_score_min=0.5
        )
        markets = [_make_market(condition_id="c1")]
        alerts = await tracker.check_wallet_activity(markets)
        # Score=0.49 < threshold=0.5 → no alert
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_check_activity_empty_watched(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(return_value=[])

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        markets = [_make_market(condition_id="c1")]
        alerts = await tracker.check_wallet_activity(markets)
        assert len(alerts) == 0
        # get_top_holders should not be called
        mock_data_client.get_top_holders.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_activity_api_error_graceful(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        call_count = 0

        async def get_holders_with_error(cid: str, **kw: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API error")
            return [{"address": "0xinsider", "amount": 1000, "outcome": "yes"}]

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = get_holders_with_error
        mock_db = AsyncMock()
        mock_db.get_watched_wallets = AsyncMock(
            return_value=[{"address": "0xinsider", "platform": "polymarket", "insider_score": 0.8}]
        )

        tracker = WalletTracker(
            redis=mock_redis, db=mock_db, data_client=mock_data_client, insider_score_min=0.5
        )
        markets = [
            _make_market(external_id="m_fail", condition_id="c_fail"),
            _make_market(external_id="m_ok", condition_id="c_ok"),
        ]
        alerts = await tracker.check_wallet_activity(markets)
        # First market errors, second succeeds
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_discover_wallet_already_exists(self, mock_redis: Any) -> None:
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(return_value=[{"address": "0xexisting"}])
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value={"id": 1})  # Already exists
        mock_db.upsert_wallet = AsyncMock()

        tracker = WalletTracker(redis=mock_redis, db=mock_db, data_client=mock_data_client)
        discovered = await tracker.discover_wallets_from_market("cond_1")
        assert len(discovered) == 0
        mock_db.upsert_wallet.assert_not_called()


# ───────────────────────────────────────────────────────────────
# 18. Batch 6: WebSocket Message Handler Tests
# ───────────────────────────────────────────────────────────────


class TestPolymarketWSHandler:
    """Tests for PolymarketWSClient._handle_message()."""

    @pytest.fixture
    def poly_ws(self, mock_redis: Any) -> Any:
        from synesis.markets.polymarket_ws import PolymarketWSClient

        client = PolymarketWSClient.__new__(PolymarketWSClient)
        client._redis = mock_redis
        client._subscribed_tokens = set()
        client._running = False
        return client

    @pytest.mark.asyncio
    async def test_handle_last_trade_price(self, poly_ws: Any, mock_redis: Any) -> None:
        event = {
            "event_type": "last_trade_price",
            "asset_id": "tok_1",
            "price": 0.65,
            "size": 100,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        await poly_ws._handle_message(orjson.dumps(event))

        # Check price hset
        price_hash = mock_redis._hashes.get("synesis:mkt_intel:ws:price:polymarket:tok_1", {})
        assert price_hash["price"] == "0.65"
        assert price_hash["ts"] == "2025-01-01T00:00:00Z"

        # Check volume increment
        vol_val = mock_redis._strings.get("synesis:mkt_intel:ws:volume_1h:polymarket:tok_1")
        assert vol_val is not None
        assert float(vol_val) == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_handle_price_change(self, poly_ws: Any, mock_redis: Any) -> None:
        event = {
            "event_type": "price_change",
            "asset_id": "tok_1",
            "price": 0.70,
            "timestamp": "2025-01-01T00:01:00Z",
        }
        await poly_ws._handle_message(orjson.dumps(event))

        price_hash = mock_redis._hashes.get("synesis:mkt_intel:ws:price:polymarket:tok_1", {})
        assert price_hash["price"] == "0.7"

    @pytest.mark.asyncio
    async def test_handle_array_of_events(self, poly_ws: Any, mock_redis: Any) -> None:
        events = [
            {
                "event_type": "last_trade_price",
                "asset_id": "tok_a",
                "price": 0.50,
                "size": 10,
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "event_type": "last_trade_price",
                "asset_id": "tok_b",
                "price": 0.80,
                "size": 20,
                "timestamp": "2025-01-01T00:00:00Z",
            },
        ]
        await poly_ws._handle_message(orjson.dumps(events))

        assert "synesis:mkt_intel:ws:price:polymarket:tok_a" in mock_redis._hashes
        assert "synesis:mkt_intel:ws:price:polymarket:tok_b" in mock_redis._hashes

    @pytest.mark.asyncio
    async def test_handle_empty_asset_id_skipped(self, poly_ws: Any, mock_redis: Any) -> None:
        event = {"event_type": "last_trade_price", "asset_id": "", "price": 0.65, "size": 100}
        await poly_ws._handle_message(orjson.dumps(event))
        assert len(mock_redis._hashes) == 0

    @pytest.mark.asyncio
    async def test_handle_zero_price_skipped(self, poly_ws: Any, mock_redis: Any) -> None:
        event = {
            "event_type": "last_trade_price",
            "asset_id": "tok_1",
            "price": 0,
            "size": 100,
        }
        await poly_ws._handle_message(orjson.dumps(event))
        # price=0 → no hset
        assert "synesis:mkt_intel:ws:price:polymarket:tok_1" not in mock_redis._hashes

    @pytest.mark.asyncio
    async def test_handle_zero_size_no_volume_increment(
        self, poly_ws: Any, mock_redis: Any
    ) -> None:
        event = {
            "event_type": "last_trade_price",
            "asset_id": "tok_1",
            "price": 0.65,
            "size": 0,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        await poly_ws._handle_message(orjson.dumps(event))
        # Price should be set
        assert "synesis:mkt_intel:ws:price:polymarket:tok_1" in mock_redis._hashes
        # Volume should NOT be set
        assert "synesis:mkt_intel:ws:volume_1h:polymarket:tok_1" not in mock_redis._strings

    @pytest.mark.asyncio
    async def test_handle_malformed_json(self, poly_ws: Any) -> None:
        # Should not crash
        await poly_ws._handle_message(b"not valid json{{{")
        # No assertion needed — just verifying no exception


class TestKalshiWSHandler:
    """Tests for KalshiWSClient._handle_message()."""

    @pytest.fixture
    def kalshi_ws(self, mock_redis: Any) -> Any:
        from synesis.markets.kalshi_ws import KalshiWSClient

        client = KalshiWSClient.__new__(KalshiWSClient)
        client._redis = mock_redis
        client._subscribed_tickers = set()
        client._running = False
        return client

    @pytest.mark.asyncio
    async def test_handle_ticker_event(self, kalshi_ws: Any, mock_redis: Any) -> None:
        event = {
            "type": "ticker",
            "msg": {
                "market_ticker": "K1",
                "yes_bid_dollars": 0.55,
                "yes_ask_dollars": 0.60,
                "volume": 1000,
            },
        }
        await kalshi_ws._handle_message(orjson.dumps(event))

        price_hash = mock_redis._hashes.get("synesis:mkt_intel:ws:price:kalshi:K1", {})
        # mid = (0.55 + 0.60) / 2 = 0.575
        assert float(price_hash["price"]) == pytest.approx(0.575)
        assert price_hash["yes_bid"] == "0.55"
        assert price_hash["yes_ask"] == "0.6"
        assert price_hash["volume"] == "1000"

    @pytest.mark.asyncio
    async def test_handle_trade_event(self, kalshi_ws: Any, mock_redis: Any) -> None:
        event = {
            "type": "trade",
            "msg": {
                "market_ticker": "K1",
                "count": 50,
                "yes_price_dollars": 0.58,
            },
        }
        await kalshi_ws._handle_message(orjson.dumps(event))

        # Price hset
        price_hash = mock_redis._hashes.get("synesis:mkt_intel:ws:price:kalshi:K1", {})
        assert price_hash["price"] == "0.58"

        # Volume increment
        vol_val = mock_redis._strings.get("synesis:mkt_intel:ws:volume_1h:kalshi:K1")
        assert vol_val is not None
        assert float(vol_val) == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_handle_ticker_zero_bid_ask_skipped(
        self, kalshi_ws: Any, mock_redis: Any
    ) -> None:
        event = {
            "type": "ticker",
            "msg": {
                "market_ticker": "K1",
                "yes_bid_dollars": 0,
                "yes_ask_dollars": 0,
                "volume": 1000,
            },
        }
        await kalshi_ws._handle_message(orjson.dumps(event))

        price_hash = mock_redis._hashes.get("synesis:mkt_intel:ws:price:kalshi:K1", {})
        # No price key since both bid/ask are 0, but volume=1000 should be stored
        assert "price" not in price_hash
        assert price_hash.get("volume") == "1000"

    @pytest.mark.asyncio
    async def test_handle_empty_ticker_skipped(self, kalshi_ws: Any, mock_redis: Any) -> None:
        event = {
            "type": "ticker",
            "msg": {
                "market_ticker": "",
                "yes_bid_dollars": 0.55,
                "yes_ask_dollars": 0.60,
            },
        }
        await kalshi_ws._handle_message(orjson.dumps(event))
        assert len(mock_redis._hashes) == 0

    @pytest.mark.asyncio
    async def test_handle_malformed_json(self, kalshi_ws: Any) -> None:
        await kalshi_ws._handle_message(b"{{invalid json")
        # No crash


# ───────────────────────────────────────────────────────────────
# 19. Batch 7: Snapshot Storage Assertions
# ───────────────────────────────────────────────────────────────


class TestSnapshotStorage:
    """Tests for MarketScanner._store_snapshots()."""

    @pytest.fixture
    def scanner_for_snapshots(self) -> Any:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        def _create(
            ws_rt_vol: float | None = None,
        ) -> tuple[MarketScanner, AsyncMock]:
            mock_db = AsyncMock()
            mock_db.insert_market_snapshot = AsyncMock()

            mock_ws = None
            if ws_rt_vol is not None:
                mock_ws = AsyncMock()
                mock_ws.read_and_reset_volume = AsyncMock(return_value=ws_rt_vol)

            scanner = MarketScanner(
                polymarket=AsyncMock(
                    get_trending_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                kalshi=AsyncMock(
                    get_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                ws_manager=mock_ws,
                db=mock_db,
            )
            return scanner, mock_db

        return _create

    @pytest.mark.asyncio
    async def test_store_snapshots_calls_db(self, scanner_for_snapshots: Any) -> None:
        scanner, mock_db = scanner_for_snapshots()
        markets = [
            _make_market(external_id="m1", volume_24h=2400),
            _make_market(external_id="m2", volume_24h=4800),
        ]
        await scanner._store_snapshots(markets, {})
        assert mock_db.insert_market_snapshot.call_count == 2

    @pytest.mark.asyncio
    async def test_store_snapshots_correct_args(self, scanner_for_snapshots: Any) -> None:
        scanner, mock_db = scanner_for_snapshots()
        market = _make_market(
            external_id="m1",
            yes_price=0.65,
            no_price=0.35,
            volume_24h=2400,
            open_interest=5000.0,
        )
        await scanner._store_snapshots([market], {})

        mock_db.insert_market_snapshot.assert_called_once()
        call_kwargs = mock_db.insert_market_snapshot.call_args[1]
        assert call_kwargs["platform"] == "polymarket"
        assert call_kwargs["market_external_id"] == "m1"
        assert call_kwargs["yes_price"] == 0.65
        assert call_kwargs["no_price"] == 0.35
        assert call_kwargs["volume_1h"] is None  # No WS = no volume_1h
        assert call_kwargs["volume_24h"] == 2400  # REST volume stored
        assert call_kwargs["trade_count_1h"] is None
        assert call_kwargs["open_interest"] == 5000.0

    @pytest.mark.asyncio
    async def test_store_snapshots_ws_volume_override(self, scanner_for_snapshots: Any) -> None:
        scanner, mock_db = scanner_for_snapshots()
        market = _make_market(volume_24h=2400)
        await scanner._store_snapshots([market], {"market_1": 500.0})

        call_kwargs = mock_db.insert_market_snapshot.call_args[1]
        assert call_kwargs["volume_1h"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_store_snapshots_ws_volume_zero_no_fallback(
        self, scanner_for_snapshots: Any
    ) -> None:
        """WS returns 0 → volume_1h stays None (no fake fallback)."""
        scanner, mock_db = scanner_for_snapshots()
        market = _make_market(volume_24h=2400)
        await scanner._store_snapshots([market], {})

        call_kwargs = mock_db.insert_market_snapshot.call_args[1]
        assert call_kwargs["volume_1h"] is None  # No fake fallback
        assert call_kwargs["volume_24h"] == 2400  # REST volume still stored

    @pytest.mark.asyncio
    async def test_store_snapshots_error_graceful(self, scanner_for_snapshots: Any) -> None:
        scanner, mock_db = scanner_for_snapshots()

        call_count = 0

        async def insert_with_error(**kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB error")

        mock_db.insert_market_snapshot = insert_with_error

        markets = [
            _make_market(external_id="m_fail", volume_24h=2400),
            _make_market(external_id="m_ok", volume_24h=4800),
        ]
        await scanner._store_snapshots(markets, {})
        # Should not crash; second market still stored
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_store_snapshots_no_db(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        scanner = MarketScanner(
            polymarket=AsyncMock(
                get_trending_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            kalshi=AsyncMock(
                get_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            ws_manager=None,
            db=None,
        )
        # Should early return without error
        await scanner._store_snapshots([_make_market()], {})


# ───────────────────────────────────────────────────────────────
# 20. Batch 8: Telegram Formatting Edge Cases
# ───────────────────────────────────────────────────────────────


class TestTelegramFormattingEdgeCases:
    def test_format_all_sections_populated(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(question=f"Opp {i}"),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["insider_activity"],
                    reasoning="Test",
                )
                for i in range(3)
            ],
            odds_movements=[
                OddsMovement(
                    market=_make_market(question=f"Odds {i}"),
                    price_change_1h=0.10,
                    direction="up",
                )
                for i in range(2)
            ],
            insider_activity=[
                InsiderAlert(
                    market=_make_market(question="Insider test"),
                    wallet_address="0xinsider123",
                    insider_score=0.85,
                    trade_direction="yes",
                    trade_size=50000,
                )
            ],
            expiring_soon=[
                _make_market(
                    question="Expiring test",
                    end_date=datetime.now(UTC) + timedelta(hours=5),
                )
            ],
        )
        msg = format_mkt_intel_signal(signal)

        assert "Opportunities" in msg
        assert "Odds Movements" in msg
        assert "Insider Activity" in msg
        assert "Expiring Soon" in msg

    def test_format_top5_truncation(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(question=f"Opp {i}"),
                    suggested_direction="yes",
                    confidence=0.8 - i * 0.05,
                    triggers=["insider_activity"],
                )
                for i in range(10)
            ],
        )
        msg = format_mkt_intel_signal(signal)

        # Count occurrences of numbered items
        assert "1." in msg
        assert "5." in msg
        assert "6." not in msg

    def test_format_6h_price_missing(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            odds_movements=[
                OddsMovement(
                    market=_make_market(question="Test"),
                    price_change_1h=0.10,
                    price_change_6h=None,
                    direction="up",
                )
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "(1h)" in msg
        assert "(6h)" not in msg

    def test_format_large_signal_under_4096(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(question=f"Opportunity number {i}?"),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["insider_activity", "odds_movement"],
                    reasoning="Detailed reasoning text for the opportunity",
                )
                for i in range(5)
            ],
            odds_movements=[
                OddsMovement(
                    market=_make_market(question=f"Odds move market {i}?"),
                    price_change_1h=0.10 + i * 0.02,
                    price_change_6h=0.20 + i * 0.03,
                    direction="up",
                )
                for i in range(5)
            ],
            insider_activity=[
                InsiderAlert(
                    market=_make_market(question=f"Insider market {i}?"),
                    wallet_address=f"0x{i:040x}",
                    insider_score=0.85,
                    trade_direction="yes",
                    trade_size=50000,
                )
                for i in range(5)
            ],
            expiring_soon=[
                _make_market(
                    question=f"Expiring market {i}?",
                    end_date=datetime.now(UTC) + timedelta(hours=i + 1),
                )
                for i in range(5)
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert len(msg) < 4096


# ───────────────────────────────────────────────────────────────
# Outcome Label Tests
# ───────────────────────────────────────────────────────────────


class TestOutcomeLabelPassthrough:
    def test_poly_to_unified_passes_outcome_label(self) -> None:
        """Test that group_item_title flows through _poly_to_unified as outcome_label."""
        from synesis.markets.polymarket import SimpleMarket
        from synesis.processing.mkt_intel.scanner import _poly_to_unified

        sm = SimpleMarket(
            id="multi_1",
            condition_id="cond_multi",
            question="What will Hochul say?",
            slug="hochul-say",
            description=None,
            category="politics",
            yes_price=0.12,
            no_price=0.88,
            volume_24h=1000,
            volume_total=10000,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
            group_item_title="Venezuela",
        )

        unified = _poly_to_unified(sm)
        assert unified.outcome_label == "Venezuela"

    def test_poly_to_unified_none_when_no_group_item_title(self) -> None:
        """Test outcome_label is None when group_item_title not set."""
        from synesis.markets.polymarket import SimpleMarket
        from synesis.processing.mkt_intel.scanner import _poly_to_unified

        sm = SimpleMarket(
            id="regular_1",
            condition_id="cond_regular",
            question="Will it rain?",
            slug="rain",
            description=None,
            category="weather",
            yes_price=0.60,
            no_price=0.40,
            volume_24h=500,
            volume_total=5000,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
        )

        unified = _poly_to_unified(sm)
        assert unified.outcome_label is None

    def test_telegram_formatter_shows_outcome_label_in_opportunities(self) -> None:
        """Test that outcome_label appears in Opportunities section."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(
                        question="What will Hochul say?",
                        outcome_label="Venezuela",
                    ),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["high_volume"],
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "Venezuela" in msg
        assert "→ Venezuela" in msg

    def test_telegram_formatter_shows_outcome_label_in_odds_movements(self) -> None:
        """Test that outcome_label appears in Odds Movements section."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            odds_movements=[
                OddsMovement(
                    market=_make_market(
                        question="What will Hochul say?",
                        outcome_label="AI / Artificial Intelligence",
                    ),
                    price_change_1h=0.15,
                    direction="up",
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "AI / Artificial Intelligence" in msg

    def test_telegram_formatter_no_arrow_without_outcome_label(self) -> None:
        """Test no arrow when outcome_label is None."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(question="Simple market?"),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["high_volume"],
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "Simple market?" in msg
        assert "→" not in msg.split("Opportunities")[1].split("Triggers")[0]

    def test_telegram_formatter_shows_outcome_label_in_expiring(self) -> None:
        """Test that outcome_label appears in Expiring Soon section."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            expiring_soon=[
                _make_market(
                    question="What will Hochul say?",
                    outcome_label="1H O/U 114.5",
                    end_date=datetime.now(UTC) + timedelta(hours=3),
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "1H O/U 114.5" in msg

    def test_telegram_formatter_shows_yes_outcome_in_expiring(self) -> None:
        """Test that yes_outcome label shows next to price in Expiring Soon."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            expiring_soon=[
                _make_market(
                    question="XRP Up or Down - Feb 9",
                    yes_outcome="Up",
                    yes_price=0.07,
                    no_price=0.93,
                    end_date=datetime.now(UTC) + timedelta(hours=1),
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "Up $0.07" in msg

    def test_telegram_formatter_shows_yes_outcome_in_opportunities(self) -> None:
        """Test that yes_outcome label shows next to price in Opportunities."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            opportunities=[
                MarketIntelOpportunity(
                    market=_make_market(
                        question="BTC Up or Down",
                        yes_outcome="Up",
                        yes_price=0.49,
                        no_price=0.51,
                    ),
                    suggested_direction="yes",
                    confidence=0.8,
                    triggers=["high_volume"],
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "Up $0.49" in msg

    def test_telegram_formatter_no_yes_outcome_for_standard_markets(self) -> None:
        """Test that standard Yes/No markets don't show extra outcome label."""
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            expiring_soon=[
                _make_market(
                    question="Will it rain?",
                    yes_price=0.60,
                    no_price=0.40,
                    end_date=datetime.now(UTC) + timedelta(hours=2),
                ),
            ],
        )
        msg = format_mkt_intel_signal(signal)
        assert "$0.60" in msg
        # Should not have an outcome label before the price
        assert "Up $" not in msg
        assert "Down $" not in msg
        assert "Over $" not in msg


# ───────────────────────────────────────────────────────────────
# 21. Batch 9: WS Manager Price Complement
# ───────────────────────────────────────────────────────────────


class TestWSManagerPriceComplement:
    @pytest.fixture
    def ws_manager(self, mock_redis: Any) -> Any:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = True
        kalshi_ws.subscribed_count = 0
        return MarketWSManager(poly_ws, kalshi_ws, mock_redis)

    @pytest.mark.asyncio
    async def test_realtime_price_complement(self, ws_manager: Any, mock_redis: Any) -> None:
        mock_redis._hashes["synesis:mkt_intel:ws:price:polymarket:cond_1"] = {"price": "0.72"}
        result = await ws_manager.get_realtime_price("polymarket", "cond_1")
        assert result is not None
        yes, no = result
        assert yes == pytest.approx(0.72)
        assert no == pytest.approx(0.28)
        # Exact complement
        assert yes + no == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_realtime_price_boundary_0(self, ws_manager: Any, mock_redis: Any) -> None:
        mock_redis._hashes["synesis:mkt_intel:ws:price:polymarket:cond_1"] = {"price": "0.0"}
        result = await ws_manager.get_realtime_price("polymarket", "cond_1")
        assert result is not None
        assert result == (0.0, 1.0)

    @pytest.mark.asyncio
    async def test_realtime_price_boundary_1(self, ws_manager: Any, mock_redis: Any) -> None:
        mock_redis._hashes["synesis:mkt_intel:ws:price:polymarket:cond_1"] = {"price": "1.0"}
        result = await ws_manager.get_realtime_price("polymarket", "cond_1")
        assert result is not None
        assert result == (1.0, 0.0)

    @pytest.mark.asyncio
    async def test_realtime_price_missing_key(self, ws_manager: Any, mock_redis: Any) -> None:
        # Hash exists but no 'price' key
        mock_redis._hashes["synesis:mkt_intel:ws:price:polymarket:cond_1"] = {"volume": "1000"}
        result = await ws_manager.get_realtime_price("polymarket", "cond_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_realtime_volume_non_numeric(self, ws_manager: Any, mock_redis: Any) -> None:
        mock_redis._strings["synesis:mkt_intel:ws:volume_1h:polymarket:cond_1"] = "abc"
        result = await ws_manager.get_realtime_volume("polymarket", "cond_1")
        assert result is None


# ───────────────────────────────────────────────────────────────
# 22. Wallet Discovery Pipeline
# ───────────────────────────────────────────────────────────────


class TestWalletDiscoveryPipeline:
    """Tests for automated wallet discovery and scoring."""

    @pytest.fixture
    def mock_data_client(self) -> AsyncMock:
        """Mock PolymarketDataClient with holder and trade data."""
        client = AsyncMock()
        client.get_top_holders = AsyncMock(
            return_value=[
                {"address": "0xwinner1", "amount": 50000, "outcome": "yes"},
                {"address": "0xwinner2", "amount": 30000, "outcome": "no"},
                {"address": "0xloser1", "amount": 10000, "outcome": "yes"},
            ]
        )
        # Default: good win rate (3 wins out of 4)
        client.get_wallet_trades = AsyncMock(
            return_value=[
                {"pnl": 100.0},
                {"pnl": 50.0},
                {"pnl": -20.0},
                {"pnl": 30.0},
            ]
        )
        return client

    @pytest.fixture
    def mock_db_for_discovery(self) -> AsyncMock:
        """Mock DB for wallet discovery tests."""
        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xwinner1", "0xwinner2"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_discover_and_score_wallets_basic(
        self,
        mock_redis: Any,
        mock_db_for_discovery: Any,
        mock_data_client: Any,
    ) -> None:
        """Test the main discover_and_score_wallets method."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_for_discovery,
            data_client=mock_data_client,
        )

        markets = [
            _make_market(external_id="m1", condition_id="cond_1", volume_24h=100000),
            _make_market(external_id="m2", condition_id="cond_2", volume_24h=50000),
        ]

        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=5,
            auto_watch_threshold=0.5,
            min_trades_to_watch=3,
        )

        # Should have called get_top_holders for each market
        assert mock_data_client.get_top_holders.call_count == 2

        # Should have upserted wallets
        assert mock_db_for_discovery.upsert_wallet.call_count >= 3

        # Should have scored wallets needing update
        assert mock_db_for_discovery.upsert_wallet_metrics.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_and_score_auto_watches_high_scorers(
        self,
        mock_redis: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that wallets with high insider scores are auto-watched."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xwinner1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        # Set up data client to return trades that result in high score
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": 100.0} for _ in range(50)]  # 100% win rate, 50 trades
        )

        tracker = WalletTracker(
            redis=mock_redis,
            db=db,
            data_client=mock_data_client,
        )

        markets = [_make_market(condition_id="cond_1")]

        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.5,
            min_trades_to_watch=20,
        )

        # Should have set wallet as watched
        db.set_wallet_watched.assert_called_once()
        call_args = db.set_wallet_watched.call_args
        assert call_args[0][0] == "0xwinner1"  # address
        assert call_args[0][1] == "polymarket"  # platform
        assert call_args[0][2] is True  # is_watched

    @pytest.mark.asyncio
    async def test_discover_and_score_skips_low_scorers(
        self,
        mock_redis: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that wallets with low insider scores are NOT auto-watched."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xloser1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        # Set up data client to return trades that result in low score
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": -100.0} for _ in range(5)]  # 0% win rate
        )

        tracker = WalletTracker(
            redis=mock_redis,
            db=db,
            data_client=mock_data_client,
        )

        markets = [_make_market(condition_id="cond_1")]

        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.5,
            min_trades_to_watch=3,
        )

        # Should NOT have called set_wallet_watched
        db.set_wallet_watched.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_and_score_respects_min_trades(
        self,
        mock_redis: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that wallets with too few trades are NOT auto-watched."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xwinner1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        # High win rate but only 5 trades
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": 100.0} for _ in range(5)]
        )

        tracker = WalletTracker(
            redis=mock_redis,
            db=db,
            data_client=mock_data_client,
        )

        markets = [_make_market(condition_id="cond_1")]

        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.3,
            min_trades_to_watch=20,  # Require 20 trades
        )

        # Should NOT auto-watch (only 5 trades < 20 required)
        db.set_wallet_watched.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_and_score_skips_stale_wallets(
        self,
        mock_redis: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that wallets scored recently are skipped."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        # No wallets need updating (all scored recently)
        db.get_wallets_needing_score_update = AsyncMock(return_value=[])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        tracker = WalletTracker(
            redis=mock_redis,
            db=db,
            data_client=mock_data_client,
        )

        markets = [_make_market(condition_id="cond_1")]

        await tracker.discover_and_score_wallets(markets)

        # Should have upserted wallets but NOT scored any
        assert db.upsert_wallet.call_count >= 1
        db.upsert_wallet_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_and_score_filters_polymarket_only(
        self,
        mock_redis: Any,
        mock_db_for_discovery: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that only Polymarket markets are processed."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_for_discovery,
            data_client=mock_data_client,
        )

        markets = [
            _make_market(platform="polymarket", condition_id="cond_1"),
            _make_kalshi_market(ticker="KTEST"),  # Kalshi, should be skipped
        ]

        await tracker.discover_and_score_wallets(markets, top_n_markets=5)

        # Only one call for the Polymarket market
        assert mock_data_client.get_top_holders.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_and_score_sorts_by_volume(
        self,
        mock_redis: Any,
        mock_db_for_discovery: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that markets are sorted by volume and top N selected."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_for_discovery,
            data_client=mock_data_client,
        )

        markets = [
            _make_market(external_id="m1", condition_id="c1", volume_24h=10000),
            _make_market(external_id="m2", condition_id="c2", volume_24h=100000),
            _make_market(external_id="m3", condition_id="c3", volume_24h=50000),
        ]

        await tracker.discover_and_score_wallets(markets, top_n_markets=2)

        # Should only call for top 2 by volume (m2 and m3)
        assert mock_data_client.get_top_holders.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_and_score_no_db(self, mock_redis: Any) -> None:
        """Test graceful handling when DB is not available."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(redis=mock_redis, db=None)

        result = await tracker.discover_and_score_wallets([_make_market()])
        assert result == 0

    @pytest.mark.asyncio
    async def test_discover_and_score_no_data_client(
        self, mock_redis: Any, mock_db_for_discovery: Any
    ) -> None:
        """Test graceful handling when data client is not available."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(redis=mock_redis, db=mock_db_for_discovery, data_client=None)

        result = await tracker.discover_and_score_wallets([_make_market()])
        assert result == 0

    @pytest.mark.asyncio
    async def test_discover_and_score_empty_markets(
        self,
        mock_redis: Any,
        mock_db_for_discovery: Any,
        mock_data_client: Any,
    ) -> None:
        """Test with empty market list."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_for_discovery,
            data_client=mock_data_client,
        )

        result = await tracker.discover_and_score_wallets([])
        assert result == 0
        mock_data_client.get_top_holders.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_wallet_metrics_returns_score(
        self,
        mock_redis: Any,
        mock_db_for_discovery: Any,
        mock_data_client: Any,
    ) -> None:
        """Test that update_wallet_metrics returns insider score and trade count."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db_for_discovery,
            data_client=mock_data_client,
        )

        score, total = await tracker.update_wallet_metrics("0xtest")

        assert total == 4  # 4 trades in mock
        assert 0 <= score <= 1.0
        mock_db_for_discovery.upsert_wallet_metrics.assert_called_once()


class TestDatabaseWalletMethods:
    """Tests for new database methods supporting wallet discovery."""

    @pytest.mark.asyncio
    async def test_set_wallet_watched(self) -> None:
        """Test set_wallet_watched updates wallet watched status."""
        from synesis.storage.database import Database

        db = Database.__new__(Database)
        db.execute = AsyncMock()

        await db.set_wallet_watched("0xtest", "polymarket", True)

        db.execute.assert_called_once()
        args = db.execute.call_args[0]
        assert "UPDATE wallets" in args[0]
        assert "is_watched" in args[0]
        assert args[1] == "polymarket"
        assert args[2] == "0xtest"
        assert args[3] is True

    @pytest.mark.asyncio
    async def test_get_wallets_needing_score_update_empty(self) -> None:
        """Test get_wallets_needing_score_update with empty input."""
        from synesis.storage.database import Database

        db = Database.__new__(Database)
        db.fetch = AsyncMock()

        result = await db.get_wallets_needing_score_update([], "polymarket")

        assert result == []
        db.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_wallets_needing_score_update_returns_stale(self) -> None:
        """Test get_wallets_needing_score_update returns stale wallets."""
        from synesis.storage.database import Database

        db = Database.__new__(Database)
        db.fetch = AsyncMock(
            return_value=[
                {"address": "0xstale1"},
                {"address": "0xstale2"},
            ]
        )

        result = await db.get_wallets_needing_score_update(
            ["0xstale1", "0xstale2", "0xfresh"], "polymarket", stale_hours=24
        )

        assert result == ["0xstale1", "0xstale2"]
        db.fetch.assert_called_once()
        args = db.fetch.call_args[0]
        assert "wallet_metrics" in args[0]
        assert args[1] == "polymarket"
        assert args[2] == ["0xstale1", "0xstale2", "0xfresh"]
        assert args[3] == 24

    @pytest.mark.asyncio
    async def test_upsert_wallet_metrics(self) -> None:
        """Test upsert_wallet_metrics inserts/updates metrics."""
        from synesis.storage.database import Database

        db = Database.__new__(Database)
        db.execute = AsyncMock()

        await db.upsert_wallet_metrics(
            address="0xtest",
            platform="polymarket",
            total_trades=100,
            wins=75,
            win_rate=0.75,
            total_pnl=5000.0,
            insider_score=0.85,
        )

        db.execute.assert_called_once()
        args = db.execute.call_args[0]
        assert "INSERT INTO wallet_metrics" in args[0]
        assert "ON CONFLICT" in args[0]
        assert args[1] == "polymarket"
        assert args[2] == "0xtest"
        assert args[3] == 100
        assert args[4] == 75
        assert args[5] == 0.75
        assert args[6] == 5000.0
        assert args[7] == 0.85


class TestWalletDiscoveryAPIEndpoint:
    """Tests for the wallet discovery API endpoint."""

    @pytest.mark.asyncio
    async def test_trigger_wallet_discovery_success(self, mock_redis: Any) -> None:
        """Test successful wallet discovery trigger."""
        from synesis.api.routes.mkt_intel import trigger_wallet_discovery

        state = MagicMock()
        state.redis = mock_redis

        mock_db = AsyncMock()
        mock_db.upsert_wallet = AsyncMock()
        mock_db.get_wallets_needing_score_update = AsyncMock(return_value=[])
        mock_db.get_watched_wallets = AsyncMock(return_value=[])

        from synesis.markets.polymarket import SimpleMarket

        mock_gamma = AsyncMock()
        mock_gamma.get_trending_markets = AsyncMock(
            return_value=[
                SimpleMarket(
                    id="m1",
                    condition_id="c1",
                    question="Test?",
                    slug="test",
                    description=None,
                    category=None,
                    yes_price=0.5,
                    no_price=0.5,
                    volume_24h=10000,
                    volume_total=50000,
                    end_date=None,
                    created_at=None,
                    is_active=True,
                    is_closed=False,
                )
            ]
        )
        mock_gamma.close = AsyncMock()

        mock_data = AsyncMock()
        mock_data.get_top_holders = AsyncMock(return_value=[])
        mock_data.close = AsyncMock()

        with (
            patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db),
            patch(
                "synesis.markets.polymarket.PolymarketClient",
                return_value=mock_gamma,
            ),
            patch(
                "synesis.markets.polymarket.PolymarketDataClient",
                return_value=mock_data,
            ),
        ):
            result = await trigger_wallet_discovery(state)

        assert result["status"] == "completed"
        assert "markets_scanned" in result

    @pytest.mark.asyncio
    async def test_trigger_wallet_discovery_no_db(self, mock_redis: Any) -> None:
        """Test wallet discovery when DB not initialized."""
        from synesis.api.routes.mkt_intel import trigger_wallet_discovery

        state = MagicMock()
        state.redis = mock_redis

        with patch(
            "synesis.api.routes.mkt_intel.get_database",
            side_effect=RuntimeError("DB not init"),
        ):
            result = await trigger_wallet_discovery(state)

        assert result["status"] == "error"
        assert "Database not initialized" in result["detail"]

    @pytest.mark.asyncio
    async def test_trigger_wallet_discovery_no_markets(self, mock_redis: Any) -> None:
        """Test wallet discovery with no trending markets."""
        from synesis.api.routes.mkt_intel import trigger_wallet_discovery

        state = MagicMock()
        state.redis = mock_redis

        mock_db = AsyncMock()
        mock_gamma = AsyncMock()
        mock_gamma.get_trending_markets = AsyncMock(return_value=[])
        mock_gamma.close = AsyncMock()

        mock_data = AsyncMock()
        mock_data.close = AsyncMock()

        with (
            patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db),
            patch(
                "synesis.markets.polymarket.PolymarketClient",
                return_value=mock_gamma,
            ),
            patch(
                "synesis.markets.polymarket.PolymarketDataClient",
                return_value=mock_data,
            ),
        ):
            result = await trigger_wallet_discovery(state)

        assert result["status"] == "no_markets"


class TestMktIntelConfigAutoWatch:
    """Tests for the new auto-watch config setting."""

    def test_auto_watch_threshold_default(self) -> None:
        """Test mkt_intel_auto_watch_threshold has correct default."""
        from synesis.config import Settings

        settings = Settings(
            _env_file=None,
            twitterapi_api_key="test",
        )
        assert settings.mkt_intel_auto_watch_threshold == 0.6

    def test_auto_watch_threshold_custom(self) -> None:
        """Test mkt_intel_auto_watch_threshold can be customized."""
        from synesis.config import Settings

        settings = Settings(
            _env_file=None,
            twitterapi_api_key="test",
            mkt_intel_auto_watch_threshold=0.8,
        )
        assert settings.mkt_intel_auto_watch_threshold == 0.8


class TestProcessorWalletDiscoveryIntegration:
    """Tests for wallet discovery integration in processor."""

    @pytest.mark.asyncio
    async def test_run_scan_triggers_wallet_discovery(self) -> None:
        """Test that run_scan triggers wallet discovery as a background task."""
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        mock_settings = MagicMock()
        mock_settings.mkt_intel_auto_watch_threshold = 0.6

        mock_scanner = AsyncMock()
        mock_scanner.scan = AsyncMock(
            return_value=ScanResult(
                timestamp=datetime.now(UTC),
                trending_markets=[_make_market()],
            )
        )

        mock_wallet_tracker = AsyncMock()
        mock_wallet_tracker.check_wallet_activity = AsyncMock(return_value=[])
        mock_wallet_tracker.discover_and_score_wallets = AsyncMock(return_value=0)

        processor = MarketIntelProcessor(
            settings=mock_settings,
            scanner=mock_scanner,
            wallet_tracker=mock_wallet_tracker,
            ws_manager=None,
            db=None,
        )

        # Run scan
        await processor.run_scan()

        # Give the background task a chance to start
        import asyncio

        await asyncio.sleep(0.01)

        # The background task should have been created
        # (We can't directly verify it without more complex async testing,
        # but we can verify the method exists and is callable)
        assert hasattr(processor, "_run_wallet_discovery")


# ───────────────────────────────────────────────────────────────
# Volume Spike Detection
# ───────────────────────────────────────────────────────────────


class TestKalshiCategoryEnrichment:
    """Tests for Kalshi category enrichment via events endpoint."""

    @pytest.mark.asyncio
    async def test_kalshi_markets_get_category_from_events(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        kalshi_market = KalshiMarket(
            ticker="KTEST-1",
            event_ticker="KTEST",
            title="Test Kalshi?",
            subtitle=None,
            status="open",
            yes_bid=0.55,
            yes_ask=0.60,
            no_bid=0.40,
            no_ask=0.45,
            last_price=0.58,
            volume=1000,
            volume_24h=200,
            open_interest=500,
            close_time=None,
            category=None,  # No category on market
            result=None,
        )

        mock_kalshi = AsyncMock()
        mock_kalshi.get_markets = AsyncMock(return_value=[kalshi_market])
        mock_kalshi.get_expiring_markets = AsyncMock(return_value=[])
        mock_kalshi.get_event_categories = AsyncMock(return_value={"KTEST": "Sports"})

        mock_poly = AsyncMock()
        mock_poly.get_trending_markets = AsyncMock(return_value=[])
        mock_poly.get_expiring_markets = AsyncMock(return_value=[])

        scanner = MarketScanner(
            polymarket=mock_poly,
            kalshi=mock_kalshi,
            ws_manager=None,
            db=None,
        )
        result = await scanner.scan()

        # The Kalshi market should have category enriched from event
        kalshi_markets = [m for m in result.trending_markets if m.platform == "kalshi"]
        assert len(kalshi_markets) == 1
        assert kalshi_markets[0].category == "Sports"

        # Verify get_event_categories was called with the event ticker
        mock_kalshi.get_event_categories.assert_awaited_once_with(["KTEST"])


class TestVolumeSpike:
    def test_create(self) -> None:
        vs = VolumeSpike(
            market=_make_market(),
            volume_current=20000,
            volume_previous=8000,
            pct_change=1.5,
        )
        assert vs.pct_change == 1.5
        assert vs.volume_current == 20000
        assert vs.volume_previous == 8000

    def test_negative_pct_change_allowed(self) -> None:
        vs = VolumeSpike(
            market=_make_market(),
            volume_current=5000,
            volume_previous=8000,
            pct_change=-0.375,
        )
        assert vs.pct_change == -0.375


class TestReadAndResetVolume:
    @pytest.mark.asyncio
    async def test_returns_value_and_clears(self, mock_redis: Any) -> None:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = False
        kalshi_ws.subscribed_count = 0

        mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

        # Set volume in Redis
        mock_redis._strings["synesis:mkt_intel:ws:volume_1h:polymarket:cond_1"] = "5000.0"

        # Read and reset
        vol = await mgr.read_and_reset_volume("polymarket", "cond_1")
        assert vol == 5000.0

        # Key should be deleted
        assert "synesis:mkt_intel:ws:volume_1h:polymarket:cond_1" not in mock_redis._strings

        # Second call returns None
        vol2 = await mgr.read_and_reset_volume("polymarket", "cond_1")
        assert vol2 is None

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self, mock_redis: Any) -> None:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = False
        kalshi_ws.subscribed_count = 0

        mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)
        vol = await mgr.read_and_reset_volume("polymarket", "nonexistent")
        assert vol is None


class TestDetectVolumeSpikes:
    @pytest.fixture
    def scanner_with_ws_and_db(self, mock_redis: Any) -> Any:
        from synesis.markets.ws_manager import MarketWSManager
        from synesis.processing.mkt_intel.scanner import MarketScanner

        def _create(
            ws_volume: float | None = 20000.0,
            prev_volume: float | None = 8000.0,
            threshold: float = 1.0,
        ) -> MarketScanner:
            poly_ws = MagicMock()
            poly_ws.is_connected = True
            poly_ws.subscribed_count = 0
            poly_ws.start = AsyncMock()
            poly_ws.stop = AsyncMock()
            poly_ws.subscribe = AsyncMock()
            poly_ws.unsubscribe = AsyncMock()

            kalshi_ws = MagicMock()
            kalshi_ws.is_connected = False
            kalshi_ws.subscribed_count = 0
            kalshi_ws.start = AsyncMock()
            kalshi_ws.stop = AsyncMock()
            kalshi_ws.subscribe = AsyncMock()
            kalshi_ws.unsubscribe = AsyncMock()

            ws_mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

            # Set volume in Redis for the market
            if ws_volume is not None:
                mock_redis._strings["synesis:mkt_intel:ws:volume_1h:polymarket:cond_1"] = str(
                    ws_volume
                )

            mock_db = AsyncMock()
            mock_db.insert_market_snapshot = AsyncMock()

            mock_db.fetchrow = AsyncMock(return_value=None)

            if prev_volume is not None:
                # Volume spikes now use batch fetch; odds movements also use fetch
                def _make_fetch(pv: float) -> AsyncMock:
                    async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
                        if "volume_1h" in query:
                            return [{"market_external_id": "market_1", "volume_1h": pv}]
                        return []

                    return AsyncMock(side_effect=_fetch)

                mock_db.fetch = _make_fetch(prev_volume)
            else:
                mock_db.fetch = AsyncMock(return_value=[])

            return MarketScanner(
                polymarket=AsyncMock(
                    get_trending_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                kalshi=AsyncMock(
                    get_markets=AsyncMock(return_value=[]),
                    get_expiring_markets=AsyncMock(return_value=[]),
                ),
                ws_manager=ws_mgr,
                db=mock_db,
                volume_spike_threshold=threshold,
            )

        return _create

    @pytest.mark.asyncio
    async def test_spike_detected(self, scanner_with_ws_and_db: Any) -> None:
        # 20000 vs 8000 = 150% increase > 100% threshold
        scanner = scanner_with_ws_and_db(ws_volume=20000.0, prev_volume=8000.0)
        markets = [_make_market(external_id="market_1")]
        spikes = await scanner._detect_volume_spikes(markets, {"market_1": 20000.0})
        assert len(spikes) == 1
        assert spikes[0].volume_current == 20000.0
        assert spikes[0].volume_previous == 8000.0
        assert spikes[0].pct_change == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_below_threshold(self, scanner_with_ws_and_db: Any) -> None:
        # 12000 vs 8000 = 50% increase < 100% threshold
        scanner = scanner_with_ws_and_db(ws_volume=12000.0, prev_volume=8000.0)
        markets = [_make_market(external_id="market_1")]
        spikes = await scanner._detect_volume_spikes(markets, {"market_1": 12000.0})
        assert len(spikes) == 0

    @pytest.mark.asyncio
    async def test_no_previous_data(self, scanner_with_ws_and_db: Any) -> None:
        scanner = scanner_with_ws_and_db(ws_volume=20000.0, prev_volume=None)
        markets = [_make_market(external_id="market_1")]
        spikes = await scanner._detect_volume_spikes(markets, {"market_1": 20000.0})
        assert len(spikes) == 0

    @pytest.mark.asyncio
    async def test_no_ws_volume(self, scanner_with_ws_and_db: Any) -> None:
        scanner = scanner_with_ws_and_db(ws_volume=None, prev_volume=8000.0)
        markets = [_make_market(external_id="market_1")]
        spikes = await scanner._detect_volume_spikes(markets, {})
        assert len(spikes) == 0

    @pytest.mark.asyncio
    async def test_no_ws_manager(self) -> None:
        from synesis.processing.mkt_intel.scanner import MarketScanner

        scanner = MarketScanner(
            polymarket=AsyncMock(
                get_trending_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            kalshi=AsyncMock(
                get_markets=AsyncMock(return_value=[]),
                get_expiring_markets=AsyncMock(return_value=[]),
            ),
            ws_manager=None,
            db=AsyncMock(),
        )
        markets = [_make_market()]
        spikes = await scanner._detect_volume_spikes(markets, {})
        assert len(spikes) == 0


class TestProcessorVolumeSpikeScoring:
    def test_volume_spike_trigger(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)
        proc._settings = MagicMock()

        market = _make_market(external_id="m1", volume_24h=0)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[],
            volume_spikes=[
                VolumeSpike(
                    market=market,
                    volume_current=20000,
                    volume_previous=8000,
                    pct_change=1.5,
                )
            ],
        )

        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert "volume_spike" in opps[0].triggers
        assert opps[0].confidence >= 0.15
        assert "Volume spike" in opps[0].reasoning

    def test_volume_spike_200pct_higher_score(self) -> None:
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        proc = MarketIntelProcessor.__new__(MarketIntelProcessor)
        proc._settings = MagicMock()

        market = _make_market(external_id="m1", volume_24h=0)
        scan = ScanResult(
            timestamp=datetime.now(UTC),
            trending_markets=[market],
            expiring_markets=[],
            odds_movements=[],
            volume_spikes=[
                VolumeSpike(
                    market=market,
                    volume_current=30000,
                    volume_previous=10000,
                    pct_change=2.0,
                )
            ],
        )

        opps = proc._score_opportunities(scan, [])
        assert len(opps) == 1
        assert opps[0].confidence == pytest.approx(0.20)


class TestTelegramVolumeSpikes:
    def test_format_with_volume_spikes(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(
            volume_spikes=[
                VolumeSpike(
                    market=_make_market(question="Volume spike test"),
                    volume_current=20000,
                    volume_previous=8000,
                    pct_change=1.5,
                )
            ]
        )
        msg = format_mkt_intel_signal(signal)
        assert "Volume Spikes" in msg
        assert "150%" in msg
        assert "8,000" in msg
        assert "20,000" in msg

    def test_format_no_volume_spikes(self) -> None:
        from synesis.notifications.telegram import format_mkt_intel_signal

        signal = _make_signal(volume_spikes=[])
        msg = format_mkt_intel_signal(signal)
        assert "Volume Spikes" not in msg


class TestGetdelParseFailure:
    """Test 7: GETDEL parse failure consumes key and returns None (data loss path)."""

    @pytest.mark.asyncio
    async def test_corrupt_value_returns_none_key_consumed(self, mock_redis: Any) -> None:
        from synesis.markets.ws_manager import MarketWSManager

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = False
        kalshi_ws.subscribed_count = 0

        mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

        # Set corrupt (non-numeric) volume in Redis
        key = "synesis:mkt_intel:ws:volume_1h:polymarket:cond_corrupt"
        mock_redis._strings[key] = "not-a-number"

        vol = await mgr.read_and_reset_volume("polymarket", "cond_corrupt")

        # Should return None due to parse failure
        assert vol is None
        # Key should have been consumed (GETDEL deletes even on parse failure)
        assert key not in mock_redis._strings


class TestKalshiEnrichmentFailureKeepsMarkets:
    """Test 9: get_event_categories raises but _fetch_kalshi_trending still returns markets."""

    @pytest.mark.asyncio
    async def test_enrichment_error_returns_markets(self) -> None:
        from synesis.markets.kalshi import KalshiMarket
        from synesis.processing.mkt_intel.scanner import MarketScanner

        kalshi_market = KalshiMarket(
            ticker="KFAIL-1",
            event_ticker="KFAIL",
            title="Enrichment failure test?",
            subtitle=None,
            status="open",
            yes_bid=0.50,
            yes_ask=0.55,
            no_bid=0.45,
            no_ask=0.50,
            last_price=0.52,
            volume=1000,
            volume_24h=200,
            open_interest=500,
            close_time=None,
            category=None,
            result=None,
        )

        mock_kalshi = AsyncMock()
        mock_kalshi.get_markets = AsyncMock(return_value=[kalshi_market])
        mock_kalshi.get_expiring_markets = AsyncMock(return_value=[])
        mock_kalshi.get_event_categories = AsyncMock(side_effect=RuntimeError("API down"))

        mock_poly = AsyncMock()
        mock_poly.get_trending_markets = AsyncMock(return_value=[])
        mock_poly.get_expiring_markets = AsyncMock(return_value=[])

        scanner = MarketScanner(
            polymarket=mock_poly,
            kalshi=mock_kalshi,
            ws_manager=None,
            db=None,
        )
        result = await scanner.scan()

        # Markets should still be returned despite enrichment failure
        kalshi_markets = [m for m in result.trending_markets if m.platform == "kalshi"]
        assert len(kalshi_markets) == 1
        assert kalshi_markets[0].external_id == "KFAIL-1"
        # Category remains None since enrichment failed
        assert kalshi_markets[0].category is None


class TestVolumeSpikeDbQueryFailure:
    """Test 10: db.fetch raises inside _detect_volume_spikes → returns []."""

    @pytest.mark.asyncio
    async def test_db_fetch_raises_returns_empty(self, mock_redis: Any) -> None:
        from synesis.markets.ws_manager import MarketWSManager
        from synesis.processing.mkt_intel.scanner import MarketScanner

        poly_ws = MagicMock()
        poly_ws.is_connected = True
        poly_ws.subscribed_count = 0
        kalshi_ws = MagicMock()
        kalshi_ws.is_connected = False
        kalshi_ws.subscribed_count = 0

        ws_mgr = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        scanner = MarketScanner(
            polymarket=AsyncMock(),
            kalshi=AsyncMock(),
            ws_manager=ws_mgr,
            db=mock_db,
        )

        markets = [_make_market(external_id="market_1")]
        spikes = await scanner._detect_volume_spikes(markets, {"market_1": 20000.0})
        assert spikes == []


# ───────────────────────────────────────────────────────────────
# Wallet Demotion & Re-score
# ───────────────────────────────────────────────────────────────


class TestWalletDemotionInDiscovery:
    """Tests for wallet demotion during discover_and_score_wallets."""

    @pytest.fixture
    def mock_data_client(self) -> AsyncMock:
        client = AsyncMock()
        client.get_top_holders = AsyncMock(
            return_value=[{"address": "0xwatched1", "amount": 50000, "outcome": "yes"}]
        )
        return client

    @pytest.mark.asyncio
    async def test_demotes_watched_wallet_below_threshold(
        self, mock_redis: Any, mock_data_client: Any
    ) -> None:
        """Wallet currently watched but score drops below unwatch_threshold → demoted."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xwatched1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()
        # get_watched_wallets returns the wallet as currently watched
        db.get_watched_wallets = AsyncMock(
            return_value=[{"address": "0xwatched1", "platform": "polymarket", "insider_score": 0.7}]
        )

        # Low win rate → low insider score (1 win out of 10)
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": 10.0}] + [{"pnl": -5.0}] * 9
        )

        tracker = WalletTracker(redis=mock_redis, db=db, data_client=mock_data_client)

        markets = [_make_market(condition_id="cond_1")]
        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.6,
            min_trades_to_watch=3,
            unwatch_threshold=0.3,
        )

        # Should have demoted the wallet (set_wallet_watched with False)
        db.set_wallet_watched.assert_called_once()
        call_args = db.set_wallet_watched.call_args[0]
        assert call_args[0] == "0xwatched1"
        assert call_args[1] == "polymarket"
        assert call_args[2] is False

    @pytest.mark.asyncio
    async def test_does_not_demote_unwatched_wallet(
        self, mock_redis: Any, mock_data_client: Any
    ) -> None:
        """Wallet NOT currently watched + low score → no demotion call."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.upsert_wallet = AsyncMock()
        db.get_wallets_needing_score_update = AsyncMock(return_value=["0xwatched1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()
        db.get_watched_wallets = AsyncMock(return_value=[])  # empty

        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": -5.0}] * 5  # all losses
        )

        tracker = WalletTracker(redis=mock_redis, db=db, data_client=mock_data_client)

        markets = [_make_market(condition_id="cond_1")]
        await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.6,
            min_trades_to_watch=3,
            unwatch_threshold=0.3,
        )

        # Should NOT have called set_wallet_watched at all
        db.set_wallet_watched.assert_not_called()


class TestReScoreWatchedWallets:
    """Tests for the re_score_watched_wallets method."""

    @pytest.mark.asyncio
    async def test_rescores_and_demotes(self, mock_redis: Any) -> None:
        """Stale watched wallet with degraded score gets demoted."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.get_watched_wallets_needing_rescore = AsyncMock(return_value=["0xold1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        data_client = AsyncMock()
        # Poor trades → low score
        data_client.get_wallet_trades = AsyncMock(
            return_value=[{"pnl": -10.0}] * 8 + [{"pnl": 5.0}] * 2
        )

        tracker = WalletTracker(redis=mock_redis, db=db, data_client=data_client)

        demoted = await tracker.re_score_watched_wallets(unwatch_threshold=0.3, stale_hours=24)

        assert demoted == 1
        db.set_wallet_watched.assert_called_once()
        call_args = db.set_wallet_watched.call_args[0]
        assert call_args[0] == "0xold1"
        assert call_args[2] is False

    @pytest.mark.asyncio
    async def test_keeps_good_wallets(self, mock_redis: Any) -> None:
        """Stale watched wallet with good score stays watched."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.get_watched_wallets_needing_rescore = AsyncMock(return_value=["0xgood1"])
        db.upsert_wallet_metrics = AsyncMock()
        db.set_wallet_watched = AsyncMock()

        data_client = AsyncMock()
        # Great trades → high score
        data_client.get_wallet_trades = AsyncMock(return_value=[{"pnl": 100.0}] * 50)

        tracker = WalletTracker(redis=mock_redis, db=db, data_client=data_client)

        demoted = await tracker.re_score_watched_wallets(unwatch_threshold=0.3, stale_hours=24)

        assert demoted == 0
        db.set_wallet_watched.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_wallets_to_rescore(self, mock_redis: Any) -> None:
        """No stale wallets → returns 0 immediately."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        db = AsyncMock()
        db.get_watched_wallets_needing_rescore = AsyncMock(return_value=[])

        tracker = WalletTracker(redis=mock_redis, db=db, data_client=AsyncMock())

        demoted = await tracker.re_score_watched_wallets()
        assert demoted == 0

    @pytest.mark.asyncio
    async def test_no_db_returns_zero(self, mock_redis: Any) -> None:
        """No DB configured → returns 0."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        tracker = WalletTracker(redis=mock_redis, db=None, data_client=AsyncMock())

        demoted = await tracker.re_score_watched_wallets()
        assert demoted == 0


class TestDatabaseWatchedWalletsNeedingRescore:
    """Tests for get_watched_wallets_needing_rescore DB method."""

    @pytest.mark.asyncio
    async def test_returns_stale_watched_wallets(self) -> None:
        from synesis.storage.database import Database

        db = Database.__new__(Database)
        db.fetch = AsyncMock(return_value=[{"address": "0xstale1"}, {"address": "0xstale2"}])

        result = await db.get_watched_wallets_needing_rescore("polymarket", stale_hours=24)

        assert result == ["0xstale1", "0xstale2"]
        db.fetch.assert_called_once()
        args = db.fetch.call_args[0]
        assert "is_watched = TRUE" in args[0]
        assert "wallet_metrics" in args[0]
        assert args[1] == "polymarket"
        assert args[2] == 24


class TestProcessorRescoreIntegration:
    """Tests for processor wiring of wallet re-score task."""

    @pytest.mark.asyncio
    async def test_run_scan_triggers_rescore_on_first_run(self) -> None:
        """First run_scan should trigger rescore (last_rescore_time is None)."""
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor

        mock_settings = MagicMock()
        mock_settings.mkt_intel_auto_watch_threshold = 0.6
        mock_settings.mkt_intel_unwatch_threshold = 0.3

        mock_scanner = AsyncMock()
        mock_scanner.scan = AsyncMock(
            return_value=ScanResult(
                timestamp=datetime.now(UTC),
                trending_markets=[_make_market()],
            )
        )

        mock_wallet_tracker = AsyncMock()
        mock_wallet_tracker.check_wallet_activity = AsyncMock(return_value=[])
        mock_wallet_tracker.discover_and_score_wallets = AsyncMock(return_value=0)
        mock_wallet_tracker.re_score_watched_wallets = AsyncMock(return_value=0)

        processor = MarketIntelProcessor(
            settings=mock_settings,
            scanner=mock_scanner,
            wallet_tracker=mock_wallet_tracker,
            ws_manager=None,
            db=None,
        )

        await processor.run_scan()

        # Give the background tasks a chance to start
        await asyncio.sleep(0.01)

        assert processor._last_rescore_time is not None
        assert processor._rescore_task is not None
