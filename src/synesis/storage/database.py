"""PostgreSQL database connection using raw asyncpg."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import asyncpg
import orjson

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from synesis.processing.news import NewsSignal, MarketEvaluation, UnifiedMessage

logger = get_logger(__name__)


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
        logger.debug("Database pool created", min_size=self._min_size, max_size=self._max_size)

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.debug("Database pool closed")

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
    ) -> None:
        """Insert a NewsSignal into the signals hypertable.

        Args:
            signal: The signal to insert
        """
        # Serialize signal to JSON for payload
        payload = signal.model_dump(mode="json")

        # Build tickers list from Stage 2 analysis (informed by research)
        tickers = None
        if signal.analysis and signal.analysis.tickers:
            tickers = signal.analysis.tickers

        # Build entities list from Stage 1 extraction
        entities = signal.extraction.all_entities if signal.extraction.all_entities else None

        # Build topic arrays from Stage 1 extraction
        primary_topics = (
            [t.value for t in signal.extraction.primary_topics]
            if signal.extraction and signal.extraction.primary_topics
            else None
        )
        secondary_topics = (
            [t.value for t in signal.extraction.secondary_topics]
            if signal.extraction and signal.extraction.secondary_topics
            else None
        )

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

        # Get primary topic from extraction
        signal_type = (
            signal.extraction.primary_topics[0].value
            if signal.extraction and signal.extraction.primary_topics
            else "other"
        )

        query = """
            INSERT INTO signals (time, flow_id, signal_type, payload, markets, tickers, entities, primary_topics, secondary_topics)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        await self.execute(
            query,
            signal.timestamp,
            "news",
            signal_type,
            orjson.dumps(payload).decode("utf-8"),
            orjson.dumps(markets).decode("utf-8") if markets else None,
            tickers,
            entities,
            primary_topics,
            secondary_topics,
        )
        logger.debug(
            "Signal inserted",
            external_id=signal.external_id,
            signal_type=signal_type,
            tickers=tickers,
            entities=entities,
            primary_topics=primary_topics,
            secondary_topics=secondary_topics,
            markets_count=len(markets) if markets else 0,
        )

    async def insert_raw_message(
        self,
        message: "UnifiedMessage",
    ) -> UUID:
        """Insert a raw message into the raw_messages table.

        Args:
            message: The unified message to insert

        Returns:
            UUID of the inserted message
        """
        query = """
            INSERT INTO raw_messages (
                source_platform, source_account, external_id,
                raw_text, source_timestamp
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (source_platform, external_id) DO NOTHING
            RETURNING id
        """
        result = await self.fetchval(
            query,
            message.source_platform.value,
            message.source_account,
            message.external_id,
            message.text,
            message.timestamp,
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
    async def upsert_watchlist_ticker(
        self,
        ticker: str,
        added_by: str,
        added_reason: str | None,
        expires_at: "datetime",
    ) -> bool:
        """Insert or update ticker in watchlist.

        Args:
            ticker: Stock ticker symbol
            added_by: Source that added the ticker ('telegram', 'twitter', 'manual')
            added_reason: Reason for adding
            expires_at: When the ticker should expire from watchlist

        Returns:
            True if ticker was newly added, False if updated
        """
        query = """
            INSERT INTO watchlist (ticker, added_by, added_reason, expires_at, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            ON CONFLICT (ticker) DO UPDATE SET
                expires_at = GREATEST(watchlist.expires_at, EXCLUDED.expires_at),
                is_active = TRUE
            RETURNING (xmax = 0) AS is_new
        """
        result = await self.fetchval(query, ticker, added_by, added_reason, expires_at)
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
