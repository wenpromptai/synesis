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

    import numpy as np

    from synesis.processing.models import Flow1Signal, MarketEvaluation, UnifiedMessage

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

    async def insert_signal(self, signal: "Flow1Signal") -> None:
        """Insert a Flow1Signal into the signals hypertable.

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

        query = """
            INSERT INTO signals (time, flow_id, signal_type, payload, markets, tickers, entities)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        await self.execute(
            query,
            signal.timestamp,
            "flow1",
            event_type,
            orjson.dumps(payload).decode("utf-8"),
            orjson.dumps(markets).decode("utf-8") if markets else None,
            tickers,
            entities,
        )
        logger.debug(
            "Signal inserted",
            external_id=signal.external_id,
            event_type=event_type,
            tickers=tickers,
            entities=entities,
            markets_count=len(markets) if markets else 0,
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
            existing: UUID = await self.fetchval(
                "SELECT id FROM raw_messages WHERE source_platform = $1 AND external_id = $2",
                message.source_platform.value,
                message.external_id,
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
