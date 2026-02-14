"""End-to-end pipeline tests for Flow 3: Prediction Market Intelligence.

Tests the full pipeline with mocked external APIs but real internal wiring:
REST clients -> Scanner -> Wallet Tracker -> Processor -> Signal -> Telegram formatting -> API routes

These tests validate the complete flow works end-to-end without needing
real external services.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.markets.kalshi import KalshiMarket
from synesis.markets.polymarket import SimpleMarket
from synesis.processing.mkt_intel.models import MarketIntelSignal


def _poly_market(
    mid: str = "m1",
    question: str = "Will X happen?",
    yes_price: float = 0.65,
) -> SimpleMarket:
    return SimpleMarket(
        id=mid,
        condition_id=f"cond_{mid}",
        question=question,
        slug=f"slug-{mid}",
        description=None,
        category="politics",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        volume_24h=10000,
        volume_total=50000,
        end_date=None,
        created_at=None,
        is_active=True,
        is_closed=False,
        yes_token_id=f"token_{mid}_yes",
    )


def _kalshi_market(
    ticker: str = "K1",
    title: str = "Kalshi test?",
    yes_bid: float = 0.55,
    yes_ask: float = 0.60,
) -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker,
        event_ticker="KE",
        title=title,
        subtitle=None,
        status="open",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        last_price=(yes_bid + yes_ask) / 2,
        volume=5000,
        volume_24h=10000,
        open_interest=2000,
        close_time=datetime.now(UTC) + timedelta(days=7),
        category="economics",
        result=None,
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    _strings: dict[str, str] = {}
    _hashes: dict[str, dict[str, Any]] = {}
    _pubsub: list[tuple[str, str]] = []

    async def mock_get(key: str) -> str | None:
        return _strings.get(key)

    async def mock_set(key: str, value: str, **kw: Any) -> bool:
        _strings[key] = value
        return True

    async def mock_delete(*keys: str) -> int:
        count = 0
        for k in keys:
            if k in _strings:
                del _strings[k]
                count += 1
        return count

    async def mock_publish(channel: str, message: str) -> int:
        _pubsub.append((channel, message))
        return 1

    async def mock_hgetall(key: str) -> dict[bytes, bytes]:
        h = _hashes.get(key, {})
        return {k.encode(): str(v).encode() for k, v in h.items()}

    redis.get = mock_get
    redis.set = mock_set
    redis.delete = mock_delete
    redis.publish = mock_publish
    redis.hgetall = mock_hgetall
    redis.expire = AsyncMock(return_value=True)
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()
    redis._strings = _strings
    redis._hashes = _hashes
    redis._pubsub = _pubsub
    return redis


@pytest.fixture
def mock_poly_client() -> AsyncMock:
    client = AsyncMock()
    client.get_trending_markets = AsyncMock(
        return_value=[
            _poly_market("m1", "Will Fed cut rates?", 0.65),
            _poly_market("m2", "Will BTC hit 100k?", 0.40),
        ]
    )
    client.get_expiring_markets = AsyncMock(
        return_value=[
            SimpleMarket(
                id="m3",
                condition_id="cond_m3",
                question="Expiring market?",
                slug="expiring",
                description=None,
                category=None,
                yes_price=0.80,
                no_price=0.20,
                volume_24h=10000,
                volume_total=15000,
                end_date=datetime.now(UTC) + timedelta(hours=8),
                created_at=None,
                is_active=True,
                is_closed=False,
                yes_token_id="token_m3_yes",
            )
        ]
    )
    return client


@pytest.fixture
def mock_kalshi_client() -> AsyncMock:
    client = AsyncMock()
    client.get_markets = AsyncMock(
        return_value=[
            _kalshi_market("KFED-25-RATE", "Will Fed cut by March?", 0.50, 0.55),
        ]
    )
    client.get_expiring_markets = AsyncMock(return_value=[])
    client.get_event_categories = AsyncMock(return_value={"KE": "Economics"})
    return client


@pytest.fixture
def mock_poly_data_client() -> AsyncMock:
    client = AsyncMock()
    client.get_top_holders = AsyncMock(
        return_value=[
            {"address": "0xwhale_a", "amount": 100000, "outcome": "yes"},
            {"address": "0xwhale_b", "amount": 50000, "outcome": "no"},
            {"address": "0xrandom", "amount": 500, "outcome": "yes"},
        ]
    )
    client.get_wallet_trades = AsyncMock(
        return_value=[
            {"pnl": 100.0},
            {"pnl": 200.0},
            {"pnl": -50.0},
        ]
    )
    return client


def _mock_fetch_snapshots(query: str, *args: Any) -> list[dict[str, Any]]:
    """Return snapshot rows for all requested market IDs with price 0.50."""
    ids = args[0] if args else []
    return [{"market_external_id": mid, "yes_price": 0.50} for mid in ids]


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.insert_market_snapshot = AsyncMock()
    db.insert_mkt_intel_signal = AsyncMock()
    db.fetch = AsyncMock(side_effect=_mock_fetch_snapshots)
    db.fetchrow = AsyncMock(return_value={"yes_price": 0.50})
    db.get_watched_wallets = AsyncMock(
        return_value=[
            {
                "address": "0xwhale_a",
                "platform": "polymarket",
                "insider_score": 0.85,
                "win_rate": 0.72,
                "total_trades": 150,
            },
        ]
    )
    return db


class TestMktIntelE2EWithMocks:
    """Full pipeline test with mocked external APIs.

    Validates the complete flow from REST fetch to signal generation,
    including Telegram formatting and API route responses.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self,
        mock_poly_client: AsyncMock,
        mock_kalshi_client: AsyncMock,
        mock_poly_data_client: AsyncMock,
        mock_db: AsyncMock,
        mock_redis: AsyncMock,
    ) -> None:
        """Test complete pipeline: REST -> Scanner -> Wallets -> Processor -> Signal -> Telegram."""
        from synesis.notifications.telegram import format_mkt_intel_signal
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor
        from synesis.processing.mkt_intel.scanner import MarketScanner
        from synesis.processing.mkt_intel.wallets import WalletTracker

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=mock_db,
            expiring_hours=24,
        )
        wallet_tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_poly_data_client,
            insider_score_min=0.5,
        )
        processor = MarketIntelProcessor(
            settings=MagicMock(),
            scanner=scanner,
            wallet_tracker=wallet_tracker,
            ws_manager=None,
            db=mock_db,
        )

        signal = await processor.run_scan()

        # Verify signal structure
        assert isinstance(signal, MarketIntelSignal)
        assert signal.total_markets_scanned >= 3

        # Verify trending markets from both platforms
        platforms = {m.platform for m in signal.trending_markets}
        assert "polymarket" in platforms
        assert "kalshi" in platforms

        # Verify expiring markets
        assert len(signal.expiring_soon) >= 1

        # Verify odds movements
        assert len(signal.odds_movements) >= 1

        # Verify insider activity
        assert len(signal.insider_activity) >= 1
        assert signal.insider_activity[0].wallet_address == "0xwhale_a"

        # Verify opportunities generated
        assert len(signal.opportunities) >= 1
        assert signal.opportunities[0].confidence > 0
        assert len(signal.opportunities[0].triggers) >= 1

        # Verify aggregate metrics
        assert 0 <= signal.market_uncertainty_index <= 1.0
        assert signal.informed_activity_level > 0

        # Verify DB persistence
        mock_db.insert_mkt_intel_signal.assert_called_once()

        # Verify Telegram formatting
        msg = format_mkt_intel_signal(signal)
        assert "MARKET INTEL" in msg
        assert "Opportunities" in msg
        assert len(msg) < 4096

    @pytest.mark.asyncio
    async def test_pipeline_with_ws_manager(
        self,
        mock_poly_client: AsyncMock,
        mock_kalshi_client: AsyncMock,
        mock_poly_data_client: AsyncMock,
        mock_db: AsyncMock,
        mock_redis: AsyncMock,
    ) -> None:
        """Test pipeline with WebSocket manager providing real-time data."""
        from synesis.markets.ws_manager import MarketWSManager
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor
        from synesis.processing.mkt_intel.scanner import MarketScanner
        from synesis.processing.mkt_intel.wallets import WalletTracker

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

        ws_manager = MarketWSManager(poly_ws, kalshi_ws, mock_redis)

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=ws_manager,
            db=mock_db,
        )
        wallet_tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_poly_data_client,
        )
        processor = MarketIntelProcessor(
            settings=MagicMock(),
            scanner=scanner,
            wallet_tracker=wallet_tracker,
            ws_manager=ws_manager,
            db=mock_db,
        )

        signal = await processor.run_scan()

        assert signal.ws_connected is True
        assert signal.total_markets_scanned >= 1
        poly_ws.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_run_mkt_intel_loop_single_cycle(self, mock_redis: AsyncMock) -> None:
        """Test the main loop runs one cycle and handles shutdown."""
        from synesis.agent.__main__ import run_mkt_intel_loop

        mock_processor = AsyncMock()
        mock_signal = MarketIntelSignal(
            timestamp=datetime.now(UTC),
            total_markets_scanned=5,
        )
        mock_processor.run_scan = AsyncMock(return_value=mock_signal)

        shutdown_event = asyncio.Event()

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.3)
            shutdown_event.set()

        with (
            patch(
                "synesis.agent.__main__.format_mkt_intel_signal",
                return_value="test message",
            ),
            patch(
                "synesis.agent.__main__.send_long_telegram",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await asyncio.gather(
                run_mkt_intel_loop(
                    mock_processor,
                    mock_redis,
                    interval_seconds=1,
                    shutdown_event=shutdown_event,
                    startup_delay=0,
                ),
                trigger_shutdown(),
            )

        assert mock_processor.run_scan.call_count >= 1

    @pytest.mark.asyncio
    async def test_api_routes_work_with_signal_data(
        self, mock_db: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        """Test API routes return correct data from DB."""
        import orjson

        from synesis.api.routes.mkt_intel import (
            get_latest_signal,
            get_opportunities,
            get_watched_wallets,
            get_ws_status,
            trigger_manual_scan,
        )

        signal = MarketIntelSignal(
            timestamp=datetime.now(UTC),
            total_markets_scanned=10,
        )
        payload_json = orjson.dumps(signal.model_dump(mode="json"))

        mock_db.fetchrow = AsyncMock(
            return_value={"time": datetime.now(UTC), "payload": payload_json}
        )

        state = MagicMock()
        state.redis = mock_redis

        # Test /latest
        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_latest_signal(state)
        assert result["time"] is not None
        assert result["signal"]["total_markets_scanned"] == 10

        # Test /opportunities
        opp_payload = orjson.dumps({"opportunities": [{"id": 1}, {"id": 2}]})
        mock_db.fetchrow = AsyncMock(return_value={"payload": opp_payload})
        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_opportunities(state)
        assert result["count"] == 2

        # Test /run
        result = await trigger_manual_scan(state)
        assert result["status"] == "scan_triggered"

        # Test /wallets
        with patch("synesis.api.routes.mkt_intel.get_database", return_value=mock_db):
            result = await get_watched_wallets(state)
        assert "wallets" in result

        # Test /ws-status
        result = await get_ws_status(state)
        assert "polymarket_ws" in result

    @pytest.mark.asyncio
    async def test_signal_serialization_full_roundtrip(
        self,
        mock_poly_client: AsyncMock,
        mock_kalshi_client: AsyncMock,
        mock_poly_data_client: AsyncMock,
        mock_db: AsyncMock,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that a generated signal can serialize and deserialize correctly."""
        import orjson

        from synesis.processing.mkt_intel.processor import MarketIntelProcessor
        from synesis.processing.mkt_intel.scanner import MarketScanner
        from synesis.processing.mkt_intel.wallets import WalletTracker

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=mock_db,
        )
        wallet_tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_poly_data_client,
        )
        processor = MarketIntelProcessor(
            settings=MagicMock(),
            scanner=scanner,
            wallet_tracker=wallet_tracker,
            db=mock_db,
        )

        signal = await processor.run_scan()

        # Pydantic roundtrip
        json_str = signal.model_dump_json()
        restored = MarketIntelSignal.model_validate_json(json_str)
        assert restored.total_markets_scanned == signal.total_markets_scanned
        assert len(restored.opportunities) == len(signal.opportunities)

        # orjson roundtrip (used in DB storage)
        orjson_bytes = orjson.dumps(signal.model_dump(mode="json"))
        from_orjson = MarketIntelSignal.model_validate(orjson.loads(orjson_bytes))
        assert from_orjson.total_markets_scanned == signal.total_markets_scanned

    @pytest.mark.asyncio
    async def test_full_pipeline_with_wallet_discovery(
        self,
        mock_poly_client: AsyncMock,
        mock_kalshi_client: AsyncMock,
        mock_poly_data_client: AsyncMock,
        mock_redis: AsyncMock,
    ) -> None:
        """Test complete pipeline including wallet discovery background task."""
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor
        from synesis.processing.mkt_intel.scanner import MarketScanner
        from synesis.processing.mkt_intel.wallets import WalletTracker

        # Mock DB with full wallet discovery support
        mock_db = AsyncMock()
        mock_db.insert_market_snapshot = AsyncMock()
        mock_db.insert_mkt_intel_signal = AsyncMock()
        mock_db.fetch = AsyncMock(side_effect=_mock_fetch_snapshots)
        mock_db.fetchrow = AsyncMock(return_value={"yes_price": 0.50})
        mock_db.get_watched_wallets = AsyncMock(
            return_value=[
                {"address": "0xwhale_a", "platform": "polymarket", "insider_score": 0.85},
            ]
        )
        mock_db.upsert_wallet = AsyncMock()
        mock_db.get_wallets_needing_score_update = AsyncMock(
            return_value=["0xwhale_a", "0xwhale_b"]
        )
        mock_db.upsert_wallet_metrics = AsyncMock()
        mock_db.set_wallet_watched = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.mkt_intel_auto_watch_threshold = 0.5

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=mock_db,
        )
        wallet_tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_poly_data_client,
            insider_score_min=0.5,
        )
        processor = MarketIntelProcessor(
            settings=mock_settings,
            scanner=scanner,
            wallet_tracker=wallet_tracker,
            db=mock_db,
        )

        signal = await processor.run_scan()

        # Verify main signal generation works
        assert signal.total_markets_scanned >= 3
        assert len(signal.insider_activity) >= 1

        # Give background task time to start
        await asyncio.sleep(0.2)

        # Verify wallet discovery was triggered
        assert mock_poly_data_client.get_top_holders.call_count >= 1

    @pytest.mark.asyncio
    async def test_wallet_discovery_auto_watches_high_scorers(
        self,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that wallet discovery correctly auto-watches high scorers."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        # Setup mock data client returning high-scoring wallet trades
        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(
            return_value=[
                {"address": "0xpro_trader", "amount": 100000, "outcome": "yes"},
            ]
        )
        # 100% win rate with 50 positions = high insider score
        mock_data_client.get_wallet_positions = AsyncMock(
            return_value=[
                {
                    "cashPnl": 100.0,
                    "initialValue": 1000,
                    "currentValue": 1100,
                    "conditionId": f"c{i}",
                }
                for i in range(50)
            ]
        )
        mock_data_client.get_wallet_trades = AsyncMock(
            return_value=[{"side": "buy", "conditionId": f"c{i}"} for i in range(50)]
        )

        mock_db = AsyncMock()
        mock_db.upsert_wallet = AsyncMock()
        mock_db.get_wallets_needing_score_update = AsyncMock(return_value=["0xpro_trader"])
        mock_db.upsert_wallet_metrics = AsyncMock()
        mock_db.set_wallet_watched = AsyncMock()
        mock_db.get_wallet_first_seen = AsyncMock(return_value=None)
        mock_db.get_market_categories = AsyncMock(return_value={})

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_data_client,
            insider_score_min=0.5,
        )

        from synesis.markets.models import UnifiedMarket

        markets = [
            UnifiedMarket(
                platform="polymarket",
                external_id="m1",
                condition_id="cond_1",
                question="Test?",
                yes_price=0.5,
                no_price=0.5,
                volume_24h=100000,
            )
        ]

        newly_watched = await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=1,
            auto_watch_threshold=0.5,
            min_trades_to_watch=20,
        )

        # Should have auto-watched the high scorer
        assert newly_watched == 1
        mock_db.set_wallet_watched.assert_called_once()
        call_args = mock_db.set_wallet_watched.call_args[0]
        assert call_args[0] == "0xpro_trader"
        assert call_args[2] is True  # is_watched

    @pytest.mark.asyncio
    async def test_wallet_discovery_skips_recent_scores(
        self,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that wallet discovery skips wallets scored recently."""
        from synesis.processing.mkt_intel.wallets import WalletTracker

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = AsyncMock(
            return_value=[
                {"address": "0xalready_scored", "amount": 50000, "outcome": "yes"},
            ]
        )
        mock_data_client.get_wallet_trades = AsyncMock()

        mock_db = AsyncMock()
        mock_db.upsert_wallet = AsyncMock()
        # No wallets need scoring (all scored recently)
        mock_db.get_wallets_needing_score_update = AsyncMock(return_value=[])
        mock_db.upsert_wallet_metrics = AsyncMock()
        mock_db.set_wallet_watched = AsyncMock()

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_data_client,
        )

        from synesis.markets.models import UnifiedMarket

        markets = [
            UnifiedMarket(
                platform="polymarket",
                external_id="m1",
                condition_id="cond_1",
                question="Test?",
                yes_price=0.5,
                no_price=0.5,
            )
        ]

        await tracker.discover_and_score_wallets(markets)

        # Should have upserted wallets
        assert mock_db.upsert_wallet.call_count >= 1

        # Should NOT have called get_wallet_trades (no stale wallets)
        mock_data_client.get_wallet_trades.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_wallet_discovery_endpoint(
        self,
        mock_redis: AsyncMock,
    ) -> None:
        """Test the manual wallet discovery API endpoint."""
        from synesis.api.routes.mkt_intel import trigger_wallet_discovery

        mock_db = AsyncMock()
        mock_db.upsert_wallet = AsyncMock()
        mock_db.get_wallets_needing_score_update = AsyncMock(return_value=[])

        mock_gamma = AsyncMock()
        mock_gamma.get_trending_markets = AsyncMock(return_value=[_poly_market("m1", "Test?")])
        mock_gamma.close = AsyncMock()

        mock_data = AsyncMock()
        mock_data.get_top_holders = AsyncMock(
            return_value=[{"address": "0xtest", "amount": 1000, "outcome": "yes"}]
        )
        mock_data.close = AsyncMock()

        state = MagicMock()
        state.redis = mock_redis

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
        assert result["markets_scanned"] == 1
        assert "newly_watched" in result


class TestWalletDiscoveryRateLimiting:
    """Tests for rate limiting in wallet discovery."""

    @pytest.mark.asyncio
    async def test_discovery_rate_limits_api_calls(
        self,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that wallet discovery rate limits API calls."""
        import time

        from synesis.processing.mkt_intel.wallets import WalletTracker

        call_times: list[float] = []

        async def mock_get_top_holders(condition_id: str, **kw: Any) -> list[dict[str, Any]]:
            call_times.append(time.time())
            return [{"address": f"0x{condition_id}", "amount": 1000, "outcome": "yes"}]

        mock_data_client = AsyncMock()
        mock_data_client.get_top_holders = mock_get_top_holders
        mock_data_client.get_wallet_trades = AsyncMock(return_value=[{"pnl": 10.0}])

        mock_db = AsyncMock()
        mock_db.upsert_wallet = AsyncMock()
        mock_db.get_wallets_needing_score_update = AsyncMock(return_value=[])
        mock_db.upsert_wallet_metrics = AsyncMock()
        mock_db.set_wallet_watched = AsyncMock()

        tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=mock_data_client,
        )

        from synesis.markets.models import UnifiedMarket

        # Create 3 markets to trigger 3 API calls
        markets = [
            UnifiedMarket(
                platform="polymarket",
                external_id=f"m{i}",
                condition_id=f"c{i}",
                question=f"Test {i}?",
                yes_price=0.5,
                no_price=0.5,
                volume_24h=100000 - i * 1000,  # Different volumes
            )
            for i in range(3)
        ]

        await tracker.discover_and_score_wallets(markets, top_n_markets=3)

        # Should have made 3 calls with delays
        assert len(call_times) == 3

        # Check delays exist between calls (should be ~0.1s each)
        if len(call_times) >= 2:
            delay_1_2 = call_times[1] - call_times[0]
            delay_2_3 = call_times[2] - call_times[1]
            assert delay_1_2 >= 0.09  # 0.1s with some tolerance
            assert delay_2_3 >= 0.09


class TestProcessorBackgroundTaskIntegration:
    """Tests for processor background task integration."""

    @pytest.mark.asyncio
    async def test_processor_creates_background_task(
        self,
        mock_poly_client: AsyncMock,
        mock_kalshi_client: AsyncMock,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that processor creates a background task for wallet discovery."""
        from synesis.processing.mkt_intel.processor import MarketIntelProcessor
        from synesis.processing.mkt_intel.scanner import MarketScanner
        from synesis.processing.mkt_intel.wallets import WalletTracker

        discovery_called = asyncio.Event()

        async def mock_discover(*args: Any, **kwargs: Any) -> int:
            discovery_called.set()
            return 0

        mock_db = AsyncMock()
        mock_db.insert_market_snapshot = AsyncMock()
        mock_db.insert_mkt_intel_signal = AsyncMock()
        mock_db.fetch = AsyncMock(return_value=[])
        mock_db.fetchrow = AsyncMock(return_value=None)
        mock_db.get_watched_wallets = AsyncMock(return_value=[])

        mock_settings = MagicMock()
        mock_settings.mkt_intel_auto_watch_threshold = 0.6

        scanner = MarketScanner(
            polymarket=mock_poly_client,
            kalshi=mock_kalshi_client,
            ws_manager=None,
            db=mock_db,
        )
        wallet_tracker = WalletTracker(
            redis=mock_redis,
            db=mock_db,
            data_client=None,  # No data client = discovery returns early
        )
        # Override the discovery method
        wallet_tracker.discover_and_score_wallets = mock_discover

        processor = MarketIntelProcessor(
            settings=mock_settings,
            scanner=scanner,
            wallet_tracker=wallet_tracker,
            db=mock_db,
        )

        # Run scan
        await processor.run_scan()

        # Wait for background task
        try:
            await asyncio.wait_for(discovery_called.wait(), timeout=1.0)
            was_called = True
        except asyncio.TimeoutError:
            was_called = False

        assert was_called, "Wallet discovery background task was not triggered"
