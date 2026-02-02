"""PostgreSQL database connection using raw asyncpg."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import asyncpg
import orjson

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np

    from synesis.processing.sentiment import SentimentSignal
    from synesis.processing.news import NewsSignal, MarketEvaluation, UnifiedMessage

logger = get_logger(__name__)

# Price outcome column mappings (used by multiple methods)
# Maps outcome_type to (column_name, SQL interval)
SIGNAL_PRICE_OUTCOME_COLUMNS = {
    "1h": ("prices_1h", "1 hour"),
    "6h": ("prices_6h", "6 hours"),
    "24h": ("prices_24h", "24 hours"),
}

SNAPSHOT_PRICE_OUTCOME_COLUMNS = {
    "1h": ("price_1h", "1 hour"),
    "6h": ("price_6h", "6 hours"),
    "24h": ("price_24h", "24 hours"),
}


class Database:
    """Async PostgreSQL database wrapper using asyncpg."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        # Convert SQLAlchemy-style DSN to asyncpg format
        dsn = self._dsn.replace("postgresql+asyncpg://", "postgresql://")

        async def init_connection(conn: asyncpg.Connection[asyncpg.Record]) -> None:
            """Initialize each connection with synesis schema search_path."""
            await conn.execute("SET search_path TO synesis, public")

        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=init_connection,
        )
        logger.info("Database pool created", min_size=self._min_size, max_size=self._max_size)

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection[asyncpg.Record]]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return status."""
        async with self.acquire() as conn:
            result = await conn.execute(query, *args)
            return cast(str, result)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            result = await conn.fetch(query, *args)
            return list(result)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    # -------------------------------------------------------------------------
    # Flow 1: Signal and Message Storage
    # -------------------------------------------------------------------------

    async def insert_signal(
        self,
        signal: "NewsSignal",
        prices_at_signal: dict[str, Decimal] | None = None,
    ) -> None:
        """Insert a NewsSignal into the signals hypertable.

        Args:
            signal: The signal to insert
            prices_at_signal: Optional dict of ticker prices at signal time
        """
        # Serialize signal to JSON for payload
        payload = signal.model_dump(mode="json")

        # Build tickers list from Stage 2 analysis (informed by research)
        tickers = None
        if signal.analysis and signal.analysis.tickers:
            tickers = signal.analysis.tickers

        # Build entities list from Stage 1 extraction
        entities = signal.extraction.all_entities if signal.extraction.all_entities else None

        # Build markets list from Stage 2 analysis market evaluations
        markets = None
        if signal.analysis and signal.analysis.market_evaluations:
            markets = [
                {
                    "market_id": eval.market_id,
                    "question": eval.market_question,
                    "verdict": eval.verdict,
                    "edge": eval.edge,
                    "is_relevant": eval.is_relevant,
                }
                for eval in signal.analysis.market_evaluations
                if eval.is_relevant
            ]

        # Get event type from extraction
        event_type = signal.extraction.event_type.value if signal.extraction else "other"

        # Convert prices to JSON-serializable format
        prices_json = None
        if prices_at_signal:
            prices_json = orjson.dumps({k: float(v) for k, v in prices_at_signal.items()}).decode(
                "utf-8"
            )

        query = """
            INSERT INTO signals (time, flow_id, signal_type, payload, markets, tickers, entities, prices_at_signal)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        await self.execute(
            query,
            signal.timestamp,
            "news",
            event_type,
            orjson.dumps(payload).decode("utf-8"),
            orjson.dumps(markets).decode("utf-8") if markets else None,
            tickers,
            entities,
            prices_json,
        )
        logger.debug(
            "Signal inserted",
            external_id=signal.external_id,
            event_type=event_type,
            tickers=tickers,
            entities=entities,
            markets_count=len(markets) if markets else 0,
            has_prices=prices_at_signal is not None,
        )

    async def insert_raw_message(
        self,
        message: "UnifiedMessage",
        embedding: "np.ndarray | None" = None,
        is_duplicate: bool = False,
        duplicate_of: UUID | None = None,
    ) -> UUID:
        """Insert a raw message into the raw_messages table.

        Args:
            message: The unified message to insert
            embedding: Optional embedding vector (256-dim)
            is_duplicate: Whether this is a duplicate message
            duplicate_of: UUID of the original message if duplicate

        Returns:
            UUID of the inserted message
        """
        # Convert embedding to PostgreSQL vector format if provided
        embedding_str = None
        if embedding is not None:
            # Format as PostgreSQL vector literal: [1.0, 2.0, 3.0, ...]
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        query = """
            INSERT INTO raw_messages (
                source_platform, source_account, source_type, external_id,
                raw_text, embedding, source_timestamp, is_duplicate, duplicate_of
            )
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9)
            ON CONFLICT (source_platform, external_id) DO NOTHING
            RETURNING id
        """
        result = await self.fetchval(
            query,
            message.source_platform.value,
            message.source_account,
            message.source_type.value,
            message.external_id,
            message.text,
            embedding_str,
            message.timestamp,
            is_duplicate,
            duplicate_of,
        )

        if result is None:
            # Message already exists, fetch existing ID
            existing: UUID | None = await self.fetchval(
                "SELECT id FROM raw_messages WHERE source_platform = $1 AND external_id = $2",
                message.source_platform.value,
                message.external_id,
            )
            if existing is None:
                raise RuntimeError(
                    f"Message insert failed for {message.external_id} "
                    "(no insert result and no existing record)"
                )
            logger.debug(
                "Message already exists",
                external_id=message.external_id,
                existing_id=str(existing),
            )
            return existing

        message_id = cast(UUID, result)
        logger.debug(
            "Raw message inserted",
            id=str(message_id),
            external_id=message.external_id,
            platform=message.source_platform.value,
        )
        return message_id

    # -------------------------------------------------------------------------
    # Flow 2: Sentiment Intelligence Storage
    # -------------------------------------------------------------------------

    async def insert_sentiment_signal(self, signal: "SentimentSignal") -> None:
        """Insert a SentimentSignal into the signals hypertable.

        Args:
            signal: The SentimentSignal to insert
        """
        # Serialize signal to JSON for payload
        payload = signal.model_dump(mode="json")

        query = """
            INSERT INTO signals (time, flow_id, signal_type, payload, tickers, entities)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        await self.execute(
            query,
            signal.timestamp,
            "sentiment",
            "sentiment",
            orjson.dumps(payload).decode("utf-8"),
            signal.watchlist,  # tickers array
            None,  # entities not applicable for Flow 2
        )
        logger.debug(
            "Sentiment signal inserted",
            timestamp=signal.timestamp.isoformat(),
            watchlist_size=len(signal.watchlist),
            posts_analyzed=signal.total_posts_analyzed,
        )

    async def upsert_watchlist_ticker(
        self,
        ticker: str,
        company_name: str | None,
        added_by: str,
        added_reason: str | None,
        expires_at: "datetime",
    ) -> bool:
        """Insert or update ticker in watchlist.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name if known
            added_by: Source that added the ticker ('reddit', 'news', 'manual')
            added_reason: Reason for adding
            expires_at: When the ticker should expire from watchlist

        Returns:
            True if ticker was newly added, False if updated
        """
        query = """
            INSERT INTO watchlist (ticker, company_name, added_by, added_reason, expires_at, is_active)
            VALUES ($1, $2, $3, $4, $5, TRUE)
            ON CONFLICT (ticker) DO UPDATE SET
                expires_at = GREATEST(watchlist.expires_at, EXCLUDED.expires_at),
                is_active = TRUE
            RETURNING (xmax = 0) AS is_new
        """
        result = await self.fetchval(
            query, ticker, company_name, added_by, added_reason, expires_at
        )
        is_new = result is True
        logger.debug(
            "Watchlist ticker upserted",
            ticker=ticker,
            is_new=is_new,
            expires_at=expires_at.isoformat(),
        )
        return is_new

    async def deactivate_expired_watchlist(self) -> list[str]:
        """Deactivate expired watchlist tickers.

        Returns:
            List of deactivated ticker symbols
        """
        query = """
            UPDATE watchlist
            SET is_active = FALSE
            WHERE is_active = TRUE AND expires_at < NOW()
            RETURNING ticker
        """
        rows = await self.fetch(query)
        removed = [row["ticker"] for row in rows]
        if removed:
            logger.info("Watchlist tickers deactivated", tickers=removed)
        return removed

    async def get_active_watchlist(self) -> list[str]:
        """Get all active watchlist tickers.

        Returns:
            List of active ticker symbols, sorted alphabetically
        """
        query = "SELECT ticker FROM watchlist WHERE is_active = TRUE ORDER BY ticker"
        rows = await self.fetch(query)
        return [row["ticker"] for row in rows]

    async def get_active_watchlist_with_metadata(self) -> list[asyncpg.Record]:
        """Get all active watchlist tickers with metadata.

        Returns:
            List of records with ticker, added_by, added_reason, added_at
        """
        query = """
            SELECT ticker, added_by, added_reason, added_at
            FROM watchlist
            WHERE is_active = TRUE
            ORDER BY ticker
        """
        return await self.fetch(query)

    async def insert_sentiment_snapshot(
        self,
        ticker: str,
        snapshot_time: "datetime",
        bullish_ratio: float,
        bearish_ratio: float,
        neutral_ratio: float,
        mention_count: int,
        dominant_emotion: str | None = None,
        sentiment_delta_6h: float | None = None,
        is_extreme_bullish: bool = False,
        is_extreme_bearish: bool = False,
        price_at_signal: Decimal | float | None = None,
    ) -> None:
        """Insert sentiment snapshot for a ticker.

        Args:
            ticker: Stock ticker symbol
            snapshot_time: Time of the snapshot
            bullish_ratio: Ratio of bullish mentions (0.0 to 1.0)
            bearish_ratio: Ratio of bearish mentions (0.0 to 1.0)
            neutral_ratio: Ratio of neutral mentions (0.0 to 1.0)
            mention_count: Number of mentions in the period
            dominant_emotion: Dominant emotion category
            sentiment_delta_6h: Change from previous 6h period
            is_extreme_bullish: True if >85% bullish
            is_extreme_bearish: True if >85% bearish
            price_at_signal: Stock price at snapshot time
        """
        query = """
            INSERT INTO sentiment_snapshots (
                ticker, snapshot_time, bullish_ratio, bearish_ratio, neutral_ratio,
                dominant_emotion, mention_count, sentiment_delta_6h,
                is_extreme_bullish, is_extreme_bearish, price_at_signal
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        await self.execute(
            query,
            ticker,
            snapshot_time,
            bullish_ratio,
            bearish_ratio,
            neutral_ratio,
            dominant_emotion,
            mention_count,
            sentiment_delta_6h,
            is_extreme_bullish,
            is_extreme_bearish,
            float(price_at_signal) if price_at_signal is not None else None,
        )
        logger.debug(
            "Sentiment snapshot inserted",
            ticker=ticker,
            snapshot_time=snapshot_time.isoformat(),
            mention_count=mention_count,
            has_price=price_at_signal is not None,
        )

    # -------------------------------------------------------------------------
    # Price Outcome Verification
    # -------------------------------------------------------------------------

    async def get_signals_pending_price_outcomes(
        self,
        outcome_type: str,
        limit: int = 100,
    ) -> list[asyncpg.Record]:
        """Get signals that need price outcome verification.

        Args:
            outcome_type: Which outcome to check ('1h', '6h', or '24h')
            limit: Maximum number of signals to return

        Returns:
            List of signal records with time, tickers, and prices_at_signal
        """
        if outcome_type not in SIGNAL_PRICE_OUTCOME_COLUMNS:
            raise ValueError(f"Invalid outcome_type: {outcome_type}")

        price_col, interval = SIGNAL_PRICE_OUTCOME_COLUMNS[outcome_type]

        query = f"""
            SELECT time, flow_id, tickers, prices_at_signal
            FROM signals
            WHERE prices_at_signal IS NOT NULL
              AND {price_col} IS NULL
              AND time < NOW() - INTERVAL '{interval}'
            ORDER BY time ASC
            LIMIT $1
        """
        return await self.fetch(query, limit)

    async def update_signal_price_outcome(
        self,
        signal_time: "datetime",
        flow_id: str,
        outcome_type: str,
        prices: dict[str, Decimal],
    ) -> None:
        """Update signal with price outcome.

        Args:
            signal_time: Signal timestamp (part of primary key)
            flow_id: Flow ID (part of primary key)
            outcome_type: Which outcome to update ('1h', '6h', or '24h')
            prices: Dict mapping ticker to price
        """
        if outcome_type not in SIGNAL_PRICE_OUTCOME_COLUMNS:
            raise ValueError(f"Invalid outcome_type: {outcome_type}")

        price_col, _ = SIGNAL_PRICE_OUTCOME_COLUMNS[outcome_type]
        prices_json = orjson.dumps({k: float(v) for k, v in prices.items()}).decode("utf-8")

        query = f"""
            UPDATE signals
            SET {price_col} = $1
            WHERE time = $2 AND flow_id = $3
        """
        await self.execute(query, prices_json, signal_time, flow_id)
        logger.debug(
            "Signal price outcome updated",
            signal_time=signal_time.isoformat(),
            outcome_type=outcome_type,
            tickers=list(prices.keys()),
        )

    async def get_sentiment_snapshots_pending_price_outcomes(
        self,
        outcome_type: str,
        limit: int = 100,
    ) -> list[asyncpg.Record]:
        """Get sentiment snapshots that need price outcome verification.

        Args:
            outcome_type: Which outcome to check ('1h', '6h', or '24h')
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot records with id, ticker, snapshot_time
        """
        if outcome_type not in SNAPSHOT_PRICE_OUTCOME_COLUMNS:
            raise ValueError(f"Invalid outcome_type: {outcome_type}")

        price_col, interval = SNAPSHOT_PRICE_OUTCOME_COLUMNS[outcome_type]

        query = f"""
            SELECT id, ticker, snapshot_time
            FROM sentiment_snapshots
            WHERE price_at_signal IS NOT NULL
              AND {price_col} IS NULL
              AND snapshot_time < NOW() - INTERVAL '{interval}'
            ORDER BY snapshot_time ASC
            LIMIT $1
        """
        return await self.fetch(query, limit)

    async def update_sentiment_snapshot_price_outcome(
        self,
        snapshot_id: UUID,
        outcome_type: str,
        price: Decimal | float,
    ) -> None:
        """Update sentiment snapshot with price outcome.

        Args:
            snapshot_id: Snapshot UUID
            outcome_type: Which outcome to update ('1h', '6h', or '24h')
            price: Price at outcome time
        """
        if outcome_type not in SNAPSHOT_PRICE_OUTCOME_COLUMNS:
            raise ValueError(f"Invalid outcome_type: {outcome_type}")

        price_col, _ = SNAPSHOT_PRICE_OUTCOME_COLUMNS[outcome_type]

        query = f"""
            UPDATE sentiment_snapshots
            SET {price_col} = $1
            WHERE id = $2
        """
        await self.execute(query, float(price), snapshot_id)
        logger.debug(
            "Sentiment snapshot price outcome updated",
            snapshot_id=str(snapshot_id),
            outcome_type=outcome_type,
            price=float(price),
        )

    # -------------------------------------------------------------------------
    # Flow 1: Signal and Prediction Storage (continued)
    # -------------------------------------------------------------------------

    async def insert_prediction(
        self,
        evaluation: "MarketEvaluation",
        timestamp: "datetime",
    ) -> None:
        """Insert a prediction to the predictions hypertable.

        Args:
            evaluation: The market evaluation from Stage 2B
            timestamp: The timestamp of the prediction
        """
        query = """
            INSERT INTO predictions (
                time, market_id, market_question, is_relevant, verdict,
                current_price, estimated_fair_price, edge, confidence,
                recommended_side, reasoning
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (time, market_id) DO UPDATE SET
                is_relevant = EXCLUDED.is_relevant,
                verdict = EXCLUDED.verdict,
                current_price = EXCLUDED.current_price,
                estimated_fair_price = EXCLUDED.estimated_fair_price,
                edge = EXCLUDED.edge,
                confidence = EXCLUDED.confidence,
                recommended_side = EXCLUDED.recommended_side,
                reasoning = EXCLUDED.reasoning
        """
        await self.execute(
            query,
            timestamp,
            evaluation.market_id,
            evaluation.market_question,
            evaluation.is_relevant,
            evaluation.verdict,
            evaluation.current_price,
            evaluation.estimated_fair_price,
            evaluation.edge,
            evaluation.confidence,
            evaluation.recommended_side,
            evaluation.reasoning,
        )
        logger.debug(
            "Prediction inserted",
            market_id=evaluation.market_id,
            verdict=evaluation.verdict,
            edge=evaluation.edge,
        )


# Global database instance (initialized in lifespan)
_db: Database | None = None


def get_database() -> Database:
    """Get the global database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


async def init_database(dsn: str) -> Database:
    """Initialize the global database instance."""
    global _db
    _db = Database(dsn)
    await _db.connect()
    return _db


async def close_database() -> None:
    """Close the global database instance."""
    global _db
    if _db:
        await _db.disconnect()
        _db = None
