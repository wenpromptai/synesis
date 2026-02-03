"""Tests for PostgreSQL database module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from synesis.storage.database import (
    Database,
    close_database,
    get_database,
    init_database,
)


class TestDatabase:
    """Tests for Database class."""

    def test_init(self) -> None:
        """Test database initialization."""
        db = Database(dsn="postgresql://user:pass@localhost/db")

        assert db._dsn == "postgresql://user:pass@localhost/db"
        assert db._min_size == 5
        assert db._max_size == 20
        assert db._pool is None

    def test_init_custom_pool_size(self) -> None:
        """Test database with custom pool size."""
        db = Database(dsn="postgresql://localhost/db", min_size=2, max_size=10)

        assert db._min_size == 2
        assert db._max_size == 10

    @pytest.mark.anyio
    async def test_connect_converts_dsn(self) -> None:
        """Test that connect converts SQLAlchemy DSN to asyncpg format."""
        db = Database(dsn="postgresql+asyncpg://user:pass@localhost/db")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool = AsyncMock()
            mock_create.return_value = mock_pool

            await db.connect()

            # Should have called with converted DSN
            call_args = mock_create.call_args
            assert "postgresql://user:pass@localhost/db" in str(call_args)

    @pytest.mark.anyio
    async def test_disconnect(self) -> None:
        """Test disconnecting from database."""
        db = Database(dsn="postgresql://localhost/db")
        mock_pool = AsyncMock()
        db._pool = mock_pool

        await db.disconnect()

        mock_pool.close.assert_called_once()

    @pytest.mark.anyio
    async def test_acquire_no_pool(self) -> None:
        """Test that acquire raises if not connected."""
        db = Database(dsn="postgresql://localhost/db")

        with pytest.raises(RuntimeError, match="not connected"):
            async with db.acquire():
                pass

    @pytest.mark.anyio
    async def test_execute(self) -> None:
        """Test executing a query."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.execute("INSERT INTO test VALUES ($1)", "value")

        assert result == "INSERT 1"
        mock_conn.execute.assert_called_once()

    @pytest.mark.anyio
    async def test_fetch(self) -> None:
        """Test fetching multiple rows."""
        db = Database(dsn="postgresql://localhost/db")

        mock_rows = [{"id": 1}, {"id": 2}]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.fetch("SELECT * FROM test")

        assert result == mock_rows

    @pytest.mark.anyio
    async def test_fetchrow(self) -> None:
        """Test fetching single row."""
        db = Database(dsn="postgresql://localhost/db")

        mock_row = {"id": 1, "name": "test"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.fetchrow("SELECT * FROM test WHERE id = $1", 1)

        assert result == mock_row

    @pytest.mark.anyio
    async def test_fetchval(self) -> None:
        """Test fetching single value."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.fetchval("SELECT COUNT(*) FROM test")

        assert result == 42


class TestDatabaseSignalOperations:
    """Tests for database signal operations."""

    @pytest.mark.anyio
    async def test_insert_signal(self) -> None:
        """Test inserting a NewsSignal."""
        from synesis.processing.news import (
            EventType,
            NewsSignal,
            LightClassification,
            SmartAnalysis,
            SourcePlatform,
            SourceType,
            Direction,
            ImpactLevel,
        )

        db = Database(dsn="postgresql://localhost/db")
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        extraction = LightClassification(
            event_type=EventType.macro,
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Federal Reserve",
            all_entities=["Federal Reserve", "Jerome Powell"],
        )
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sectors=["financials"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Bullish",
            thesis_confidence=0.8,
        )
        signal = NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            source_type=SourceType.news,
            raw_text="Fed cuts rates",
            external_id="123",
            extraction=extraction,
            analysis=analysis,
        )

        await db.insert_signal(signal)

        mock_conn.execute.assert_called_once()

    @pytest.mark.anyio
    async def test_insert_signal_with_prices(self) -> None:
        """Test inserting a NewsSignal with prices_at_signal."""
        from decimal import Decimal

        from synesis.processing.news import (
            Direction,
            EventType,
            ImpactLevel,
            LightClassification,
            NewsSignal,
            SmartAnalysis,
            SourcePlatform,
            SourceType,
        )

        db = Database(dsn="postgresql://localhost/db")
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        extraction = LightClassification(
            event_type=EventType.earnings,
            summary="AAPL beats earnings",
            confidence=0.9,
            primary_entity="Apple",
        )
        analysis = SmartAnalysis(
            tickers=["AAPL", "MSFT"],
            sectors=["technology"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Strong earnings",
            thesis_confidence=0.85,
        )
        signal = NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            source_type=SourceType.news,
            raw_text="AAPL beats",
            external_id="456",
            extraction=extraction,
            analysis=analysis,
        )

        prices = {"AAPL": Decimal("175.50"), "MSFT": Decimal("380.25")}
        await db.insert_signal(signal, prices_at_signal=prices)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Verify prices_at_signal is passed as JSON string
        prices_json = call_args[0][8]
        assert prices_json is not None
        assert "175.5" in prices_json
        assert "380.25" in prices_json

    @pytest.mark.anyio
    async def test_insert_raw_message(self) -> None:
        """Test inserting a raw message."""
        from synesis.processing.news import (
            SourcePlatform,
            SourceType,
            UnifiedMessage,
        )

        db = Database(dsn="postgresql://localhost/db")

        expected_id = uuid4()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=expected_id)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        message = UnifiedMessage(
            external_id="ext_123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

        result = await db.insert_raw_message(message)

        assert result == expected_id

    @pytest.mark.anyio
    async def test_insert_raw_message_duplicate(self) -> None:
        """Test inserting duplicate message returns existing ID."""
        from synesis.processing.news import (
            SourcePlatform,
            SourceType,
            UnifiedMessage,
        )

        db = Database(dsn="postgresql://localhost/db")

        existing_id = uuid4()
        mock_conn = AsyncMock()
        # First call returns None (conflict), second returns existing ID
        mock_conn.fetchval = AsyncMock(side_effect=[None, existing_id])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        message = UnifiedMessage(
            external_id="ext_123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

        result = await db.insert_raw_message(message)

        assert result == existing_id

    @pytest.mark.anyio
    async def test_insert_prediction(self) -> None:
        """Test inserting a prediction."""
        from synesis.processing.news import MarketEvaluation

        db = Database(dsn="postgresql://localhost/db")
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        evaluation = MarketEvaluation(
            market_id="mkt_123",
            market_question="Will something happen?",
            is_relevant=True,
            relevance_reasoning="Directly relevant",
            current_price=0.5,
            estimated_fair_price=0.7,
            edge=0.2,
            verdict="undervalued",
            confidence=0.8,
            reasoning="Strong evidence",
            recommended_side="yes",
        )

        await db.insert_prediction(evaluation, datetime.now(timezone.utc))

        mock_conn.execute.assert_called_once()


class TestGlobalDatabaseFunctions:
    """Tests for global database functions."""

    def test_get_database_not_initialized(self) -> None:
        """Test get_database raises if not initialized."""
        # Clear global
        import synesis.storage.database as db_module

        db_module._db = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_database()

    @pytest.mark.anyio
    async def test_init_database(self) -> None:
        """Test initializing global database."""
        import synesis.storage.database as db_module

        with patch.object(Database, "connect", new_callable=AsyncMock):
            db = await init_database("postgresql://localhost/test")

        assert db is not None
        assert db_module._db is db

        # Cleanup
        db_module._db = None

    @pytest.mark.anyio
    async def test_close_database(self) -> None:
        """Test closing global database."""
        import synesis.storage.database as db_module

        mock_db = MagicMock()
        mock_db.disconnect = AsyncMock()
        db_module._db = mock_db

        await close_database()

        mock_db.disconnect.assert_called_once()
        assert db_module._db is None


class TestDatabaseSentimentOperations:
    """Tests for sentiment database operations."""

    @pytest.mark.anyio
    async def test_insert_sentiment_signal(self) -> None:
        """Test inserting a SentimentSignal."""
        from synesis.processing.sentiment import SentimentSignal, TickerSentimentSummary

        db = Database(dsn="postgresql://localhost/db")
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        now = datetime.now(timezone.utc)
        signal = SentimentSignal(
            timestamp=now,
            period_start=now,
            period_end=now,
            watchlist=["AAPL", "TSLA"],
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    mention_count=50,
                    avg_sentiment=0.3,
                ),
            ],
            overall_sentiment="bullish",
            narrative_summary="Market is bullish on tech",
            total_posts_analyzed=100,
        )

        await db.insert_sentiment_signal(signal)

        mock_conn.execute.assert_called_once()
        # Verify the SQL contains expected table name
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO signals" in call_args[0][0]
        # Verify flow_id is 'sentiment'
        assert call_args[0][2] == "sentiment"
        # Verify watchlist is passed as tickers
        assert call_args[0][5] == ["AAPL", "TSLA"]

    @pytest.mark.anyio
    async def test_upsert_watchlist_ticker_new(self) -> None:
        """Test upserting a new ticker returns True."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        # xmax = 0 indicates new row, so is_new = True
        mock_conn.fetchval = AsyncMock(return_value=True)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        now = datetime.now(timezone.utc)
        result = await db.upsert_watchlist_ticker(
            ticker="AAPL",
            company_name="Apple Inc.",
            added_by="reddit",
            added_reason="High mention volume",
            expires_at=now,
        )

        assert result is True
        mock_conn.fetchval.assert_called_once()
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO watchlist" in call_args[0][0]
        assert call_args[0][1] == "AAPL"

    @pytest.mark.anyio
    async def test_upsert_watchlist_ticker_existing(self) -> None:
        """Test upserting existing ticker returns False."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        # xmax != 0 indicates update, so is_new = False
        mock_conn.fetchval = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        now = datetime.now(timezone.utc)
        result = await db.upsert_watchlist_ticker(
            ticker="AAPL",
            company_name="Apple Inc.",
            added_by="reddit",
            added_reason="Extended TTL",
            expires_at=now,
        )

        assert result is False

    @pytest.mark.anyio
    async def test_insert_sentiment_snapshot_all_params(self) -> None:
        """Test inserting sentiment snapshot with all parameters."""
        from decimal import Decimal

        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        now = datetime.now(timezone.utc)
        await db.insert_sentiment_snapshot(
            ticker="AAPL",
            snapshot_time=now,
            bullish_ratio=0.6,
            bearish_ratio=0.2,
            neutral_ratio=0.2,
            mention_count=50,
            dominant_emotion="bullish",
            sentiment_delta_6h=0.1,
            is_extreme_bullish=False,
            is_extreme_bearish=False,
            price_at_signal=Decimal("175.50"),
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO sentiment_snapshots" in call_args[0][0]
        # Verify price is converted to float
        assert call_args[0][11] == 175.50

    @pytest.mark.anyio
    async def test_insert_sentiment_snapshot_minimal_params(self) -> None:
        """Test inserting sentiment snapshot with minimal parameters."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        now = datetime.now(timezone.utc)
        await db.insert_sentiment_snapshot(
            ticker="TSLA",
            snapshot_time=now,
            bullish_ratio=0.33,
            bearish_ratio=0.33,
            neutral_ratio=0.34,
            mention_count=10,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Verify optional params are None
        assert call_args[0][6] is None  # dominant_emotion
        assert call_args[0][11] is None  # price_at_signal

    @pytest.mark.anyio
    async def test_deactivate_expired_watchlist(self) -> None:
        """Test deactivating expired watchlist tickers."""
        db = Database(dsn="postgresql://localhost/db")

        mock_rows = [{"ticker": "AAPL"}, {"ticker": "TSLA"}]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.deactivate_expired_watchlist()

        assert result == ["AAPL", "TSLA"]
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        assert "UPDATE watchlist" in call_args[0][0]
        assert "SET is_active = FALSE" in call_args[0][0]
        assert "expires_at < NOW()" in call_args[0][0]

    @pytest.mark.anyio
    async def test_deactivate_expired_watchlist_none_expired(self) -> None:
        """Test deactivating when no tickers are expired."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.deactivate_expired_watchlist()

        assert result == []

    @pytest.mark.anyio
    async def test_get_active_watchlist(self) -> None:
        """Test getting active watchlist tickers."""
        db = Database(dsn="postgresql://localhost/db")

        mock_rows = [{"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "TSLA"}]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.get_active_watchlist()

        assert result == ["AAPL", "MSFT", "TSLA"]
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        assert "SELECT ticker FROM watchlist" in call_args[0][0]
        assert "is_active = TRUE" in call_args[0][0]

    @pytest.mark.anyio
    async def test_get_active_watchlist_empty(self) -> None:
        """Test getting active watchlist when empty."""
        db = Database(dsn="postgresql://localhost/db")

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()
            )
        )
        db._pool = mock_pool

        result = await db.get_active_watchlist()

        assert result == []
