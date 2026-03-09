"""Centralized job scheduler for periodic processing flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred import FREDClient
    from synesis.providers.nasdaq import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
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


async def event_fetch_job(
    db: Database,
    redis: Redis,
    fred: FREDClient | None = None,
    nasdaq: NasdaqClient | None = None,
    sec_edgar: SECEdgarClient | None = None,
) -> None:
    """Fetch events from structured APIs (FRED, NASDAQ, FOMC, 13F)."""
    from synesis.processing.events.runner import run_structured_sources

    try:
        stored = await run_structured_sources(
            db, redis, fred=fred, nasdaq=nasdaq, sec_edgar=sec_edgar
        )
        logger.info("Event fetch job complete", stored=stored)
    except Exception:
        logger.exception("Event fetch job failed")


async def event_digest_job(
    db: Database,
    redis: Redis | None = None,
    sec_edgar: SECEdgarClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
) -> None:
    """Send daily Event Radar digest to Discord."""
    from synesis.processing.events.digest import send_event_digest

    try:
        sent = await send_event_digest(
            db, redis=redis, sec_edgar=sec_edgar, crawler=crawler, fred=fred
        )
        logger.info("Event digest job complete", sent=sent)
    except Exception:
        logger.exception("Event digest job failed")
