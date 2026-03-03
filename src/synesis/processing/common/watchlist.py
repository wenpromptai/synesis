"""Watchlist management for Flow 1 and Flow 2.

Manages a ticker watchlist in PostgreSQL with TTL-based expiration.
Tickers are added based on validated mentions (news or sentiment)
and auto-expire after the configured TTL period (default 7 days).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)


class WatchlistManager:
    """Manage ticker watchlist with TTL-based expiration.

    Uses PostgreSQL as the sole store:
    - Upserts extend expiry via GREATEST(expires_at)
    - APScheduler deactivates expired tickers periodically
    """

    def __init__(self, db: Database, ttl_days: int = 7) -> None:
        self.db = db
        self.ttl_days = ttl_days

    async def add_ticker(self, ticker: str, source: str) -> bool:
        """Add ticker to watchlist or extend expiry if exists.

        Returns:
            True if ticker was newly added, False if refreshed
        """
        ticker = ticker.upper()
        expires_at = datetime.now(UTC) + timedelta(days=self.ttl_days)
        added_reason = f"Signal from {source}"

        is_new = await self.db.upsert_watchlist_ticker(
            ticker=ticker,
            added_by=source,
            added_reason=added_reason,
            expires_at=expires_at,
        )

        logger.debug(
            "Ticker added to watchlist" if is_new else "Ticker watchlist TTL refreshed",
            ticker=ticker,
            source=source,
            ttl_days=self.ttl_days,
        )
        return is_new

    async def remove_ticker(self, ticker: str) -> bool:
        """Remove ticker from watchlist.

        Returns:
            True if ticker was removed, False if not found
        """
        ticker = ticker.upper()
        removed = await self.db.remove_watchlist_ticker(ticker)
        if removed:
            logger.debug("Ticker removed from watchlist", ticker=ticker)
        return removed

    async def get_all(self) -> list[str]:
        """Get all active tickers in watchlist."""
        return await self.db.get_active_watchlist()

    async def get_metadata(self, ticker: str) -> dict[str, Any] | None:
        """Get metadata for a single ticker."""
        ticker = ticker.upper()
        record = await self.db.get_watchlist_metadata(ticker)
        return dict(record) if record is not None else None

    async def get_all_with_metadata(self) -> list[dict[str, Any]]:
        """Get all active tickers with metadata."""
        records = await self.db.get_active_watchlist_with_metadata()
        return [dict(r) for r in records]

    async def cleanup_expired(self) -> list[str]:
        """Deactivate tickers whose expiry has passed.

        Returns:
            List of deactivated ticker symbols
        """
        removed = await self.db.deactivate_expired_watchlist()
        if removed:
            logger.info(
                "Watchlist cleanup complete",
                removed=removed,
                removed_count=len(removed),
            )
        return removed

    async def get_stats(self) -> dict[str, Any]:
        """Get watchlist statistics."""
        return await self.db.get_watchlist_stats()

    async def contains(self, ticker: str) -> bool:
        """Check if ticker is in watchlist."""
        ticker = ticker.upper()
        return await self.db.watchlist_contains(ticker)

    async def bulk_add(
        self,
        tickers: list[str],
        source: str,
    ) -> tuple[list[str], list[str]]:
        """Add multiple tickers to watchlist.

        Returns:
            Tuple of (newly_added, refreshed) ticker lists
        """
        newly_added = []
        refreshed = []

        for ticker in tickers:
            is_new = await self.add_ticker(ticker, source)
            if is_new:
                newly_added.append(ticker)
            else:
                refreshed.append(ticker)

        return newly_added, refreshed
