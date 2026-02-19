"""Centralized job scheduler for periodic processing flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from synesis.core.logging import get_logger
from synesis.notifications.telegram import (
    format_mkt_intel_signal,
    format_sentiment_signal,
    format_watchlist_signal,
    send_long_telegram,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.processing.common.watchlist import WatchlistManager
    from synesis.processing.mkt_intel.processor import MarketIntelProcessor
    from synesis.processing.sentiment import SentimentProcessor
    from synesis.processing.watchlist.processor import WatchlistProcessor
    from synesis.storage.database import Database

logger = get_logger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """Create a new scheduler instance."""
    return AsyncIOScheduler(timezone="UTC")


async def sentiment_signal_job(
    processor: SentimentProcessor,
    redis: Redis,
) -> None:
    """Generate and publish a sentiment signal."""
    try:
        signal = await processor.generate_signal()
        await redis.publish("synesis:sentiment:signals", signal.model_dump_json())
        msg = format_sentiment_signal(signal)
        sent = await send_long_telegram(msg)
        if not sent:
            logger.error("Failed to send sentiment signal to Telegram")
        logger.info(
            "Sentiment signal generated",
            watchlist_size=len(signal.watchlist),
            posts_analyzed=signal.total_posts_analyzed,
            overall_sentiment=signal.overall_sentiment,
        )
    except Exception:
        logger.exception("Sentiment signal job failed")


async def mkt_intel_job(
    processor: MarketIntelProcessor,
    redis: Redis,
) -> None:
    """Run market intelligence scan and publish signal."""
    try:
        signal = await processor.run_scan()
        await redis.publish("synesis:mkt_intel:signals", signal.model_dump_json())
        msg = format_mkt_intel_signal(signal)
        sent = await send_long_telegram(msg)
        if not sent:
            logger.error("Failed to send mkt_intel signal to Telegram")
        logger.info(
            "Mkt_intel signal generated",
            markets_scanned=signal.total_markets_scanned,
            opportunities=len(signal.opportunities),
        )
    except Exception:
        logger.exception("Market intel job failed")


async def watchlist_intel_job(
    processor: WatchlistProcessor,
    redis: Redis,
) -> None:
    """Run watchlist analysis and publish signal."""
    try:
        signal = await processor.run_analysis()
        await redis.publish("synesis:watchlist_intel:signals", signal.model_dump_json())
        msg = format_watchlist_signal(signal)
        sent = await send_long_telegram(msg)
        if not sent:
            logger.error("Failed to send watchlist intel signal to Telegram")
        logger.info(
            "Watchlist intel signal generated",
            tickers_analyzed=signal.tickers_analyzed,
            alerts=len(signal.alerts),
        )
    except Exception:
        logger.exception("Watchlist intel job failed")


async def watchlist_cleanup_job(
    db: Database,
    watchlist: WatchlistManager | None,
) -> None:
    """Cleanup expired watchlist tickers."""
    try:
        expired_pg = await db.deactivate_expired_watchlist()
        if expired_pg:
            logger.info("Deactivated expired watchlist tickers (PostgreSQL)", tickers=expired_pg)
        if watchlist:
            expired_redis = await watchlist.cleanup_expired()
            if expired_redis:
                logger.info("Removed expired watchlist tickers (Redis)", tickers=expired_redis)
    except Exception:
        logger.exception("Watchlist cleanup job failed")
