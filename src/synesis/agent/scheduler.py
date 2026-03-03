"""Centralized job scheduler for periodic processing flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """Create a new scheduler instance."""
    return AsyncIOScheduler(timezone="UTC")


async def watchlist_cleanup_job(db: Database) -> None:
    """Deactivate expired watchlist tickers in PostgreSQL."""
    try:
        expired = await db.deactivate_expired_watchlist()
        if expired:
            logger.info("Deactivated expired watchlist tickers", tickers=expired)
    except Exception:
        logger.exception("Watchlist cleanup job failed")
