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

    from synesis.processing.mkt_intel.models import MarketIntelSignal
    from synesis.processing.sentiment import SentimentSignal
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
            "news",
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
    ) -> UUID:
        """Insert a raw message into the raw_messages table.

        Args:
            message: The unified message to insert

        Returns:
            UUID of the inserted message
        """
        query = """
            INSERT INTO raw_messages (
                source_platform, source_account, source_type, external_id,
                raw_text, source_timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6)
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
        mention_count: int,
    ) -> None:
        """Insert sentiment snapshot for a ticker.

        Args:
            ticker: Stock ticker symbol
            snapshot_time: Time of the snapshot
            bullish_ratio: Ratio of bullish mentions (0.0 to 1.0)
            bearish_ratio: Ratio of bearish mentions (0.0 to 1.0)
            mention_count: Number of mentions in the period
        """
        query = """
            INSERT INTO sentiment_snapshots (
                ticker, snapshot_time, bullish_ratio, bearish_ratio,
                mention_count
            )
            VALUES ($1, $2, $3, $4, $5)
        """
        await self.execute(
            query,
            ticker,
            snapshot_time,
            bullish_ratio,
            bearish_ratio,
            mention_count,
        )
        logger.debug(
            "Sentiment snapshot inserted",
            ticker=ticker,
            snapshot_time=snapshot_time.isoformat(),
            mention_count=mention_count,
        )

    # -------------------------------------------------------------------------
    # Flow 3: Market Intelligence Storage
    # -------------------------------------------------------------------------

    async def insert_mkt_intel_signal(self, signal: "MarketIntelSignal") -> None:
        """Insert a MarketIntelSignal into the signals hypertable.

        Args:
            signal: The MarketIntelSignal to insert
        """
        payload = signal.model_dump(mode="json")
        query = """
            INSERT INTO signals (time, flow_id, signal_type, payload)
            VALUES ($1, $2, $3, $4)
        """
        await self.execute(
            query,
            signal.timestamp,
            "mkt_intel",
            "mkt_intel",
            orjson.dumps(payload).decode("utf-8"),
        )
        logger.debug(
            "Market intel signal inserted",
            timestamp=signal.timestamp.isoformat(),
            markets_scanned=signal.total_markets_scanned,
            opportunities=len(signal.opportunities),
        )

    async def insert_market_snapshot(
        self,
        time: "datetime",
        platform: str,
        market_external_id: str,
        question: str | None,
        category: str | None,
        yes_price: float | None,
        no_price: float | None,
        volume_1h: float | None,
        volume_24h: float | None,
        volume_total: float | None,
        trade_count_1h: int | None,
        open_interest: float | None,
    ) -> None:
        """Insert a market snapshot into the market_snapshots hypertable.

        Args:
            time: Snapshot timestamp
            platform: Platform name ('polymarket', 'kalshi')
            market_external_id: Platform's market ID
            question: Market question text
            category: Market category
            yes_price: Current YES price
            no_price: Current NO price
            volume_1h: Real WS-accumulated hourly volume (None if no WS data)
            volume_24h: 24h volume from REST API
            volume_total: All-time total volume
            trade_count_1h: Trade count in last hour
            open_interest: Open interest
        """
        query = """
            INSERT INTO market_snapshots (
                time, platform, market_external_id, question, category,
                yes_price, no_price, volume_1h, volume_24h, volume_total,
                trade_count_1h, open_interest
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (time, platform, market_external_id) DO UPDATE SET
                question = EXCLUDED.question,
                category = EXCLUDED.category,
                yes_price = EXCLUDED.yes_price,
                no_price = EXCLUDED.no_price,
                volume_1h = EXCLUDED.volume_1h,
                volume_24h = EXCLUDED.volume_24h,
                volume_total = EXCLUDED.volume_total,
                trade_count_1h = EXCLUDED.trade_count_1h,
                open_interest = EXCLUDED.open_interest
        """
        await self.execute(
            query,
            time,
            platform,
            market_external_id,
            question,
            category,
            yes_price,
            no_price,
            volume_1h,
            volume_24h,
            volume_total,
            trade_count_1h,
            open_interest,
        )

    async def upsert_wallet(self, address: str, platform: str) -> None:
        """Insert or update a wallet."""
        query = """
            INSERT INTO wallets (platform, address)
            VALUES ($1, $2)
            ON CONFLICT (platform, address) DO UPDATE SET
                last_active_at = NOW()
        """
        await self.execute(query, platform, address)

    async def get_watched_wallets(self, platform: str) -> list[asyncpg.Record]:
        """Get watched wallets with metrics.

        Args:
            platform: Platform to filter by ('polymarket')

        Returns:
            List of records with address, platform, insider_score,
            specialty_category, watch_reason
        """
        query = """
            SELECT w.address, w.platform, w.watch_reason,
                   wm.insider_score, wm.win_rate,
                   wm.total_trades, wm.specialty_category
            FROM wallets w
            LEFT JOIN wallet_metrics wm ON wm.wallet_id = w.id
            WHERE w.platform = $1 AND w.is_watched = TRUE
            ORDER BY wm.insider_score DESC NULLS LAST
        """
        return await self.fetch(query, platform)

    async def set_wallet_watched(
        self,
        address: str,
        platform: str,
        is_watched: bool,
        watch_reason: str | None = None,
    ) -> None:
        """Update wallet watched status.

        Args:
            address: Wallet address
            platform: Platform ('polymarket')
            is_watched: Whether to watch this wallet
            watch_reason: Why the wallet was watched ('score', 'high_conviction', or None)
        """
        query = """
            UPDATE wallets
            SET is_watched = $3, watch_reason = $4
            WHERE platform = $1 AND address = $2
        """
        await self.execute(
            query, platform, address, is_watched, watch_reason if is_watched else None
        )

    async def get_wallets_needing_score_update(
        self,
        addresses: list[str],
        platform: str,
        stale_hours: int = 24,
    ) -> list[str]:
        """Return addresses that haven't been scored in stale_hours.

        Args:
            addresses: List of wallet addresses to check
            platform: Platform ('polymarket')
            stale_hours: Hours after which a score is considered stale

        Returns:
            List of addresses that need scoring
        """
        if not addresses:
            return []

        query = """
            SELECT w.address
            FROM wallets w
            LEFT JOIN wallet_metrics wm ON wm.wallet_id = w.id
            WHERE w.platform = $1
              AND w.address = ANY($2)
              AND (
                  wm.updated_at IS NULL
                  OR wm.updated_at < NOW() - make_interval(hours => $3)
              )
        """
        rows = await self.fetch(query, platform, addresses, stale_hours)
        return [row["address"] for row in rows]

    async def get_watched_wallets_needing_rescore(
        self,
        platform: str,
        stale_hours: int = 24,
    ) -> list[str]:
        """Return watched wallet addresses whose scores are stale.

        Args:
            platform: Platform ('polymarket')
            stale_hours: Hours after which a score is considered stale

        Returns:
            List of addresses needing re-score
        """
        query = """
            SELECT w.address
            FROM wallets w
            LEFT JOIN wallet_metrics wm ON wm.wallet_id = w.id
            WHERE w.platform = $1
              AND w.is_watched = TRUE
              AND (
                  wm.updated_at IS NULL
                  OR wm.updated_at < NOW() - make_interval(hours => $2)
              )
        """
        rows = await self.fetch(query, platform, stale_hours)
        return [row["address"] for row in rows]

    async def upsert_wallet_metrics(
        self,
        address: str,
        platform: str,
        total_trades: int,
        wins: int,
        win_rate: float,
        total_pnl: float,
        insider_score: float,
        *,
        unique_markets: int = 0,
        avg_position_size: float = 0.0,
        wash_trade_ratio: float = 0.0,
        profitability_score: float = 0.0,
        focus_score: float = 0.0,
        sizing_score: float = 0.0,
        freshness_score: float = 0.0,
        wash_penalty: float = 0.0,
        specialty_category: str | None = None,
        specialty_win_rate: float | None = None,
    ) -> None:
        """Insert or update wallet metrics."""
        query = """
            INSERT INTO wallet_metrics (
                wallet_id, total_trades, wins, win_rate, total_pnl, insider_score,
                unique_markets, avg_position_size, wash_trade_ratio,
                profitability_score, focus_score, sizing_score, freshness_score, wash_penalty,
                specialty_category, specialty_win_rate, updated_at
            )
            SELECT w.id, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, NOW()
            FROM wallets w
            WHERE w.platform = $1 AND w.address = $2
            ON CONFLICT (wallet_id) DO UPDATE SET
                total_trades = EXCLUDED.total_trades,
                wins = EXCLUDED.wins,
                win_rate = EXCLUDED.win_rate,
                total_pnl = EXCLUDED.total_pnl,
                insider_score = EXCLUDED.insider_score,
                unique_markets = EXCLUDED.unique_markets,
                avg_position_size = EXCLUDED.avg_position_size,
                wash_trade_ratio = EXCLUDED.wash_trade_ratio,
                profitability_score = EXCLUDED.profitability_score,
                focus_score = EXCLUDED.focus_score,
                sizing_score = EXCLUDED.sizing_score,
                freshness_score = EXCLUDED.freshness_score,
                wash_penalty = EXCLUDED.wash_penalty,
                specialty_category = EXCLUDED.specialty_category,
                specialty_win_rate = EXCLUDED.specialty_win_rate,
                updated_at = NOW()
        """
        await self.execute(
            query,
            platform,
            address,
            total_trades,
            wins,
            win_rate,
            total_pnl,
            insider_score,
            unique_markets,
            avg_position_size,
            wash_trade_ratio,
            profitability_score,
            focus_score,
            sizing_score,
            freshness_score,
            wash_penalty,
            specialty_category,
            specialty_win_rate,
        )

    async def get_wallet_first_seen(self, address: str, platform: str) -> "datetime | None":
        """Get the first_seen_at timestamp for a wallet."""
        result = await self.fetchval(
            "SELECT first_seen_at FROM wallets WHERE platform = $1 AND address = $2",
            platform,
            address,
        )
        return result  # type: ignore[no-any-return]

    async def get_market_categories(self, market_ids: list[str]) -> dict[str, str | None]:
        """Get categories for markets from most recent snapshots."""
        if not market_ids:
            return {}
        rows = await self.fetch(
            """
            SELECT DISTINCT ON (market_external_id) market_external_id, category
            FROM market_snapshots
            WHERE market_external_id = ANY($1) AND category IS NOT NULL
            ORDER BY market_external_id, time DESC
            """,
            market_ids,
        )
        return {row["market_external_id"]: row["category"] for row in rows}

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
