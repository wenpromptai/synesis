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


async def event_radar_job(
    db: Database,
    redis: Redis,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
    nasdaq: NasdaqClient | None = None,
    sec_edgar: SECEdgarClient | None = None,
) -> None:
    """Fetch events from structured APIs + crawl curated sources."""
    from synesis.processing.events.runner import run_full_discovery

    if crawler is None:
        logger.warning("Crawl4AI not available, skipping curated crawls")
    try:
        totals = await run_full_discovery(
            db,
            redis,
            crawler,
            fred=fred,
            nasdaq=nasdaq,
            sec_edgar=sec_edgar,
        )
        logger.info("Event radar job complete", **totals)
    except Exception:
        logger.exception("Event radar job failed")


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
        await send_event_digest(db, redis=redis, sec_edgar=sec_edgar, crawler=crawler, fred=fred)
    except Exception:
        logger.exception("Event digest job failed")
