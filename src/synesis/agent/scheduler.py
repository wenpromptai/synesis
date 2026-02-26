"""Centralized job scheduler for periodic processing flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.processing.common.watchlist import WatchlistManager
    from synesis.storage.database import Database

logger = get_logger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """Create a new scheduler instance."""
    return AsyncIOScheduler(timezone="UTC")


async def watchlist_cleanup_job(
    db: Database,
    watchlist: WatchlistManager | None,
) -> None:
    """Cleanup expired watchlist tickers."""
    try:
        expired_pg = await db.deactivate_expired_watchlist()
        if expired_pg:
            logger.info("Deactivated expired watchlist tickers (PostgreSQL)", tickers=expired_pg)
    except Exception:
        logger.exception("Watchlist cleanup job failed (PostgreSQL)")

    if watchlist:
        try:
            expired_redis = await watchlist.cleanup_expired()
            if expired_redis:
                logger.info("Removed expired watchlist tickers (Redis)", tickers=expired_redis)
        except Exception:
            logger.exception("Watchlist cleanup job failed (Redis)")
