"""PostgreSQL database connection using raw asyncpg."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import asyncpg

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from datetime import date, datetime

    from synesis.processing.events.models import CalendarEvent

logger = get_logger(__name__)


class Database:
    """Async PostgreSQL database wrapper using asyncpg."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        # Convert SQLAlchemy-style DSN to asyncpg format
        dsn = self._dsn.replace("postgresql+asyncpg://", "postgresql://")

        async def init_connection(conn: asyncpg.Connection) -> None:
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
    async def acquire(self) -> AsyncIterator[Any]:
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
            List of records with ticker, added_by, added_reason, added_at, expires_at
        """
        query = """
            SELECT ticker, added_by, added_reason, added_at, expires_at
            FROM watchlist
            WHERE is_active = TRUE
            ORDER BY ticker
        """
        return await self.fetch(query)

    async def remove_watchlist_ticker(self, ticker: str) -> bool:
        """Deactivate a specific ticker from watchlist.

        Returns:
            True if ticker was deactivated, False if not found/already inactive
        """
        query = """
            UPDATE watchlist
            SET is_active = FALSE
            WHERE ticker = $1 AND is_active = TRUE
            RETURNING ticker
        """
        result = await self.fetchval(query, ticker)
        return result is not None

    async def get_watchlist_metadata(self, ticker: str) -> asyncpg.Record | None:
        """Get metadata for a single watchlist ticker.

        Returns:
            Record with ticker, added_by, added_reason, added_at, expires_at
            or None if not found/inactive
        """
        query = """
            SELECT ticker, added_by, added_reason, added_at, expires_at
            FROM watchlist
            WHERE ticker = $1 AND is_active = TRUE
        """
        return await self.fetchrow(query, ticker)

    async def get_watchlist_stats(self) -> dict[str, int | dict[str, int]]:
        """Get watchlist statistics.

        Returns:
            Dict with total_tickers, sources breakdown, ttl_days
        """
        count_query = "SELECT COUNT(*) FROM watchlist WHERE is_active = TRUE"
        total = await self.fetchval(count_query) or 0

        sources_query = """
            SELECT added_by, COUNT(*) as cnt
            FROM watchlist
            WHERE is_active = TRUE
            GROUP BY added_by
        """
        rows = await self.fetch(sources_query)
        sources = {row["added_by"]: row["cnt"] for row in rows}

        return {
            "total_tickers": total,
            "sources": sources,
        }

    async def watchlist_contains(self, ticker: str) -> bool:
        """Check if a ticker is active in the watchlist."""
        query = "SELECT 1 FROM watchlist WHERE ticker = $1 AND is_active = TRUE"
        result = await self.fetchval(query, ticker)
        return result is not None

    # -------------------------------------------------------------------------
    # Event Radar: Calendar Events
    # -------------------------------------------------------------------------

    async def upsert_calendar_event(
        self,
        event: "CalendarEvent",
    ) -> int | None:
        """Insert a calendar event, skipping duplicates (dedup by title + event_date).

        Returns the event id on insert, None if already exists.
        """
        query = """
            INSERT INTO calendar_events (
                title, description, event_date, event_end_date, category,
                sector, region, tickers, source_urls, time_label
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (title, event_date) DO NOTHING
            RETURNING id
        """
        result = await self.fetchval(
            query,
            event.title,
            event.description,
            event.event_date,
            event.event_end_date,
            event.category,
            event.sector,
            event.region,
            event.tickers,
            event.source_urls,
            event.time_label,
        )
        return int(result) if result is not None else None

    async def get_upcoming_events(
        self,
        days: int = 7,
        *,
        region: list[str] | None = None,
        category: str | None = None,
        sector: str | None = None,
    ) -> list[asyncpg.Record]:
        """Get upcoming events within N days, with optional filters."""
        conditions = [
            "event_date >= CURRENT_DATE",
            "event_date <= CURRENT_DATE + make_interval(days => $1)",
        ]
        params: list[object] = [days]
        idx = 2

        if category:
            conditions.append(f"category = ${idx}")
            params.append(category)
            idx += 1

        if sector:
            conditions.append(f"sector = ${idx}")
            params.append(sector)
            idx += 1

        if region:
            conditions.append(f"region && ${idx}")
            params.append(region)
            idx += 1

        where = " AND ".join(conditions)
        query = f"""
            SELECT * FROM calendar_events
            WHERE {where}
            ORDER BY event_date
        """
        return await self.fetch(query, *params)

    async def get_events_by_date_range(
        self,
        start: "date",
        end: "date",
    ) -> list[asyncpg.Record]:
        """Get all events in a date range."""
        query = """
            SELECT * FROM calendar_events
            WHERE event_date >= $1 AND event_date <= $2
            ORDER BY event_date
        """
        return await self.fetch(query, start, end)

    async def get_event_by_id(self, event_id: int) -> asyncpg.Record | None:
        """Get a single event by ID."""
        return await self.fetchrow("SELECT * FROM calendar_events WHERE id = $1", event_id)

    async def get_events_discovered_since(
        self,
        since: "datetime",
    ) -> list[int]:
        """Get IDs of events discovered after a given timestamp."""
        query = """
            SELECT id FROM calendar_events
            WHERE discovered_at > $1
        """
        rows = await self.fetch(query, since)
        return [r["id"] for r in rows]

    async def get_last_fomc_meeting_date(self, before_date: "date") -> "date | None":
        """Get the most recent FOMC rate-decision date before a given date.

        Used to construct the minutes URL (minutes release date != meeting date).
        """
        query = """
            SELECT event_date FROM calendar_events
            WHERE category = 'fed'
              AND LOWER(title) NOT LIKE '%minute%'
              AND event_date < $1
            ORDER BY event_date DESC
            LIMIT 1
        """
        result = await self.fetchval(query, before_date)
        return cast("date | None", result)

    async def delete_past_events(self, before_date: "date") -> int:
        """Delete events older than a given date. Returns count deleted."""
        result = await self.execute(
            "DELETE FROM calendar_events WHERE event_date < $1", before_date
        )
        # result is like "DELETE 42"
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    # -------------------------------------------------------------------------
    # Raw Tweets: Twitter/X tweet storage
    # -------------------------------------------------------------------------

    async def store_raw_tweets(self, tweets: list[dict[str, Any]]) -> int:
        """Bulk insert raw tweets, deduplicating on primary key (account_username, tweet_id).

        Args:
            tweets: List of dicts with keys: tweet_id, account_username, tweet_text,
                    tweet_timestamp, tweet_url (optional)

        Returns:
            Number of new tweets inserted (excludes duplicates).
        """
        if not tweets:
            return 0

        query = """
            INSERT INTO raw_tweets (tweet_id, account_username, tweet_text, tweet_timestamp, tweet_url)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (account_username, tweet_id) DO NOTHING
        """
        inserted = 0
        async with self.acquire() as conn:
            async with conn.transaction():
                for tw in tweets:
                    try:
                        result = await conn.execute(
                            query,
                            tw["tweet_id"],
                            tw["account_username"],
                            tw["tweet_text"],
                            tw["tweet_timestamp"],
                            tw.get("tweet_url"),
                        )
                        if result and "INSERT 0 1" in result:
                            inserted += 1
                    except KeyError as exc:
                        logger.error(
                            "Malformed tweet dict, skipping",
                            missing_key=str(exc),
                        )

        logger.debug("Raw tweets stored", total=len(tweets), inserted=inserted)
        return inserted

    async def get_raw_tweets(self, since_hours: int = 24) -> list["asyncpg.Record"]:
        """Fetch recent raw tweets.

        Args:
            since_hours: Look back this many hours from now.

        Returns:
            List of raw tweet records ordered by tweet_timestamp DESC.
        """
        query = """
            SELECT tweet_id, account_username, tweet_text, tweet_timestamp, tweet_url, fetched_at
            FROM raw_tweets
            WHERE fetched_at >= NOW() - make_interval(hours => $1)
            ORDER BY tweet_timestamp DESC
        """
        return await self.fetch(query, since_hours)


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
