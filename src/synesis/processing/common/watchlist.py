"""Watchlist management for Flow 1 and Flow 2.

Manages a ticker watchlist in Redis with TTL-based expiration.
Tickers are added based on validated mentions (news or sentiment)
and auto-expire after the configured TTL period (default 7 days).

Redis Key Schema:
- synesis:watchlist:tickers - Set of active ticker symbols
- synesis:watchlist:ttl:{ticker} - String with TTL for auto-expire trigger
- synesis:watchlist:metadata:{ticker} - Hash with source, added_at, etc.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.storage.database import Database

logger = get_logger(__name__)

# Redis key prefixes
WATCHLIST_KEY = "synesis:watchlist:tickers"
TTL_KEY_PREFIX = "synesis:watchlist:ttl:"
METADATA_KEY_PREFIX = "synesis:watchlist:metadata:"

# Lua script for atomic check-and-delete (prevents race condition)
# Returns 1 if ticker was removed, 0 if TTL key still exists
CLEANUP_TICKER_LUA = """
if redis.call('exists', KEYS[1]) == 0 then
    redis.call('srem', KEYS[2], ARGV[1])
    redis.call('del', KEYS[3])
    return 1
else
    return 0
end
"""


@dataclass
class TickerMetadata:
    """Metadata for a watchlist ticker."""

    ticker: str
    source: str  # "twitter", "telegram", "reddit", etc.
    subreddit: str | None = None  # For Reddit source
    added_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    mention_count: int = 1

    def to_dict(self) -> dict[str, str]:
        """Convert to Redis hash-compatible dict (all string values)."""
        return {
            "ticker": self.ticker,
            "source": self.source,
            "subreddit": self.subreddit or "",
            "added_at": self.added_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "mention_count": str(self.mention_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> TickerMetadata:
        """Create from Redis hash dict."""
        return cls(
            ticker=data["ticker"],
            source=data["source"],
            subreddit=data.get("subreddit") or None,
            added_at=datetime.fromisoformat(data["added_at"]),
            last_seen_at=datetime.fromisoformat(data["last_seen_at"]),
            mention_count=int(data.get("mention_count", "1")),
        )


class WatchlistManager:
    """Manage ticker watchlist with TTL-based expiration.

    Uses Redis for fast access and optional PostgreSQL for durability:
    - Redis: Set of active tickers, TTL keys, metadata hashes
    - PostgreSQL: Persistent watchlist table with expiration

    When both are configured, writes go to both stores, reads prefer Redis.
    """

    def __init__(
        self,
        redis: Redis,
        db: Database | None = None,
        ttl_days: int = 7,
        on_ticker_added: Callable[[str], Awaitable[None]] | None = None,
        on_ticker_removed: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize watchlist manager.

        Args:
            redis: Redis client instance for fast access
            db: Optional PostgreSQL database for persistence
            ttl_days: Days before a ticker expires from watchlist
            on_ticker_added: Optional async callback invoked when a new ticker is added.
                            Receives the ticker symbol (uppercase) as argument.
            on_ticker_removed: Optional async callback invoked when a ticker is removed.
                              Receives the ticker symbol (uppercase) as argument.
        """
        self.redis = redis
        self.db = db
        self.ttl_days = ttl_days
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self._on_ticker_added = on_ticker_added
        self._on_ticker_removed = on_ticker_removed

    async def add_ticker(
        self,
        ticker: str,
        source: str,
        subreddit: str | None = None,
        company_name: str | None = None,
    ) -> bool:
        """Add ticker to watchlist or refresh TTL if exists.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            source: Source platform ("twitter", "telegram", "reddit", etc.)
            subreddit: Subreddit name if source is Reddit
            company_name: Full company name if known

        Returns:
            True if ticker was newly added, False if refreshed
        """
        ticker = ticker.upper()
        ttl_key = f"{TTL_KEY_PREFIX}{ticker}"
        metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"

        # Check if already exists
        is_new = not await self.redis.sismember(WATCHLIST_KEY, ticker)  # type: ignore[misc]

        # Add to set (idempotent)
        await self.redis.sadd(WATCHLIST_KEY, ticker)  # type: ignore[misc]

        # Set/refresh TTL key
        await self.redis.set(ttl_key, "1", ex=self.ttl_seconds)

        if is_new:
            # Create new metadata
            metadata = TickerMetadata(
                ticker=ticker,
                source=source,
                subreddit=subreddit,
            )
            await self.redis.hset(metadata_key, mapping=metadata.to_dict())  # type: ignore[misc]
            await self.redis.expire(metadata_key, self.ttl_seconds)

            logger.info(
                "Ticker added to watchlist",
                ticker=ticker,
                source=source,
                subreddit=subreddit,
                ttl_days=self.ttl_days,
            )
        else:
            # Update existing metadata
            await self.redis.hset(  # type: ignore[misc]
                metadata_key,
                mapping={
                    "last_seen_at": datetime.now(UTC).isoformat(),
                },
            )
            await self.redis.hincrby(metadata_key, "mention_count", 1)  # type: ignore[misc]
            await self.redis.expire(metadata_key, self.ttl_seconds)

            logger.debug(
                "Ticker watchlist TTL refreshed",
                ticker=ticker,
                source=source,
            )

        # Sync to PostgreSQL if available
        if self.db:
            try:
                expires_at = datetime.now(UTC) + timedelta(days=self.ttl_days)
                added_reason = (
                    f"Sentiment from r/{subreddit}" if subreddit else f"Signal from {source}"
                )
                await self.db.upsert_watchlist_ticker(
                    ticker=ticker,
                    company_name=company_name,
                    added_by=source,
                    added_reason=added_reason,
                    expires_at=expires_at,
                )
            except Exception as e:
                logger.warning(
                    "Failed to sync ticker to PostgreSQL",
                    ticker=ticker,
                    error=str(e),
                )

        # Invoke callback for new tickers (e.g., WebSocket subscription)
        if is_new and self._on_ticker_added:
            try:
                await self._on_ticker_added(ticker)
            except Exception as e:
                logger.warning(
                    "on_ticker_added callback failed",
                    ticker=ticker,
                    error=str(e),
                )

        return is_new

    async def remove_ticker(self, ticker: str) -> bool:
        """Remove ticker from watchlist.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker was removed, False if not found
        """
        ticker = ticker.upper()
        ttl_key = f"{TTL_KEY_PREFIX}{ticker}"
        metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"

        # Check if exists
        existed = await self.redis.sismember(WATCHLIST_KEY, ticker)  # type: ignore[misc]

        if existed:
            # Remove from set and delete keys
            await self.redis.srem(WATCHLIST_KEY, ticker)  # type: ignore[misc]
            await self.redis.delete(ttl_key, metadata_key)
            logger.info("Ticker removed from watchlist", ticker=ticker)

            # Invoke callback (e.g., WebSocket unsubscription)
            if self._on_ticker_removed:
                try:
                    await self._on_ticker_removed(ticker)
                except Exception as e:
                    logger.warning(
                        "on_ticker_removed callback failed",
                        ticker=ticker,
                        error=str(e),
                    )

        return bool(existed)

    async def get_all(self) -> list[str]:
        """Get all tickers in watchlist.

        Returns:
            List of ticker symbols
        """
        tickers = await self.redis.smembers(WATCHLIST_KEY)  # type: ignore[misc]
        # Redis returns bytes, decode to strings
        return sorted(t.decode() if isinstance(t, bytes) else t for t in tickers)

    async def get_metadata(self, ticker: str) -> TickerMetadata | None:
        """Get metadata for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TickerMetadata or None if not found
        """
        ticker = ticker.upper()
        metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"

        raw_data = await self.redis.hgetall(metadata_key)  # type: ignore[misc]
        if not raw_data:
            return None

        # Redis returns bytes, decode to strings
        data = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw_data.items()
        }
        return TickerMetadata.from_dict(data)

    async def get_all_with_metadata(self) -> list[TickerMetadata]:
        """Get all tickers with their metadata.

        Uses Redis pipeline for efficient batch fetching (O(1) round trips
        instead of O(n) for n tickers).

        Returns:
            List of TickerMetadata objects
        """
        tickers = await self.get_all()
        if not tickers:
            return []

        # Use pipeline for batch fetching metadata
        pipe = self.redis.pipeline()
        for ticker in tickers:
            metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"
            pipe.hgetall(metadata_key)

        metadata_dicts = await pipe.execute()

        results = []
        for ticker, raw_data in zip(tickers, metadata_dicts):
            if raw_data:
                # Decode bytes to strings if needed
                data = {
                    (k.decode() if isinstance(k, bytes) else k): (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in raw_data.items()
                }
                try:
                    results.append(TickerMetadata.from_dict(data))
                except Exception as e:
                    logger.warning(
                        "Failed to parse ticker metadata",
                        ticker=ticker,
                        error=str(e),
                    )

        return results

    async def cleanup_expired(self) -> list[str]:
        """Remove tickers whose TTL keys have expired.

        This should be called periodically to clean up tickers
        that haven't been seen within the TTL period.

        Uses atomic Lua script to prevent race condition where a ticker
        is re-added between checking TTL expiration and removal.

        Returns:
            List of removed ticker symbols
        """
        tickers = await self.get_all()
        removed = []

        # Register the Lua script once
        cleanup_script = self.redis.register_script(CLEANUP_TICKER_LUA)

        for ticker in tickers:
            ttl_key = f"{TTL_KEY_PREFIX}{ticker}"
            metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"

            # Atomic check-and-delete: only removes if TTL key doesn't exist
            result = await cleanup_script(
                keys=[ttl_key, WATCHLIST_KEY, metadata_key],
                args=[ticker],
            )

            if result == 1:
                removed.append(ticker)
                logger.info("Ticker expired and removed from watchlist", ticker=ticker)

                # Invoke callback (e.g., WebSocket unsubscription)
                if self._on_ticker_removed:
                    try:
                        await self._on_ticker_removed(ticker)
                    except Exception as e:
                        logger.warning(
                            "on_ticker_removed callback failed",
                            ticker=ticker,
                            error=str(e),
                        )

        # Also cleanup in PostgreSQL if available
        if self.db:
            try:
                db_removed = await self.db.deactivate_expired_watchlist()
                # Merge any additional tickers removed from DB
                for ticker in db_removed:
                    if ticker not in removed:
                        removed.append(ticker)
            except Exception as e:
                logger.warning(
                    "Failed to cleanup expired tickers in PostgreSQL",
                    error=str(e),
                )

        if removed:
            logger.info(
                "Watchlist cleanup complete",
                removed=removed,
                removed_count=len(removed),
            )

        return removed

    async def sync_from_db(self) -> int:
        """Load watchlist from PostgreSQL into Redis.

        This should be called on startup to restore the watchlist
        from the durable store. Restores both ticker membership and metadata.

        Returns:
            Number of tickers loaded
        """
        if not self.db:
            logger.debug("No database configured, skipping watchlist sync")
            return 0

        try:
            db_records = await self.db.get_active_watchlist_with_metadata()
            loaded = 0

            for record in db_records:
                ticker = record["ticker"]
                # Only add to Redis if not already present
                if not await self.redis.sismember(WATCHLIST_KEY, ticker):  # type: ignore[misc]
                    await self.redis.sadd(WATCHLIST_KEY, ticker)  # type: ignore[misc]

                    # Set TTL key
                    ttl_key = f"{TTL_KEY_PREFIX}{ticker}"
                    await self.redis.set(ttl_key, "1", ex=self.ttl_seconds)

                    # Restore metadata from database
                    # Extract subreddit from added_reason if present (e.g., "Sentiment from r/wallstreetbets")
                    added_reason = record["added_reason"] or ""
                    subreddit = None
                    if "r/" in added_reason:
                        import re

                        match = re.search(r"r/(\w+)", added_reason)
                        if match:
                            subreddit = match.group(1)

                    # Create metadata from DB fields
                    added_at = record["added_at"] or datetime.now(UTC)
                    metadata = TickerMetadata(
                        ticker=ticker,
                        source=record["added_by"] or "unknown",
                        subreddit=subreddit,
                        added_at=added_at,
                        last_seen_at=added_at,  # Best we can do on restore
                        mention_count=1,  # Reset on restore
                    )

                    # Store metadata in Redis
                    metadata_key = f"{METADATA_KEY_PREFIX}{ticker}"
                    await self.redis.hset(metadata_key, mapping=metadata.to_dict())  # type: ignore[misc]
                    await self.redis.expire(metadata_key, self.ttl_seconds)

                    loaded += 1

            if loaded:
                logger.info(
                    "Watchlist synced from PostgreSQL",
                    loaded=loaded,
                    total_in_db=len(db_records),
                )

            return loaded

        except Exception as e:
            logger.warning(
                "Failed to sync watchlist from PostgreSQL",
                error=str(e),
            )
            return 0

    async def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get watchlist statistics.

        Returns:
            Dict with count, sources breakdown, etc.
        """
        all_metadata = await self.get_all_with_metadata()

        sources: dict[str, int] = {}
        for meta in all_metadata:
            sources[meta.source] = sources.get(meta.source, 0) + 1

        return {
            "total_tickers": len(all_metadata),
            "sources": sources,
            "ttl_days": self.ttl_days,
        }

    async def contains(self, ticker: str) -> bool:
        """Check if ticker is in watchlist.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker is in watchlist
        """
        ticker = ticker.upper()
        result = await self.redis.sismember(WATCHLIST_KEY, ticker)  # type: ignore[misc]
        return bool(result)

    async def bulk_add(
        self,
        tickers: list[str],
        source: str,
        subreddit: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Add multiple tickers to watchlist.

        Args:
            tickers: List of ticker symbols
            source: Source platform
            subreddit: Subreddit name if source is Reddit

        Returns:
            Tuple of (newly_added, refreshed) ticker lists
        """
        newly_added = []
        refreshed = []

        for ticker in tickers:
            is_new = await self.add_ticker(ticker, source, subreddit)
            if is_new:
                newly_added.append(ticker)
            else:
                refreshed.append(ticker)

        return newly_added, refreshed
