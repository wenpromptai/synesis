"""Agent lifecycle module used by the FastAPI server.

Provides `agent_lifespan()` — an async context manager that starts and stops
the full processing pipeline (APScheduler periodic jobs). The FastAPI app calls
this from its own lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from synesis.providers.finnhub.prices import FinnhubPriceProvider

from redis.asyncio import Redis

from synesis.agent.scheduler import (
    create_scheduler,
    event_digest_job,
    event_fetch_job,
    market_movers_job,
    refresh_tickers_job,
    watchlist_cleanup_job,
)
from synesis.config import Settings
from synesis.core.logging import get_logger
from synesis.processing.events.digest import send_event_digest
from synesis.processing.events.runner import run_structured_sources
from synesis.processing.market.job import market_movers_job as _market_movers_fn
from synesis.processing.twitter.job import twitter_agent_job
from synesis.providers.finnhub.prices import close_price_service, init_price_service
from synesis.providers.fred import FREDClient
from synesis.providers.nasdaq import NasdaqClient
from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.storage.database import Database, close_database, init_database
from synesis.storage.redis import close_redis, init_redis

logger = get_logger(__name__)


@dataclass
class AgentState:
    """Holds references to all running agent resources."""

    redis: Redis
    db: Database | None
    settings: Settings
    db_enabled: bool
    scheduler: AsyncIOScheduler | None = None
    trigger_fns: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def agent_lifespan(
    settings: Settings,
    shutdown_event: asyncio.Event,
) -> AsyncIterator[AgentState]:
    """Async context manager that starts/stops the entire agent pipeline.

    Yields an AgentState with references to all running resources.
    On exit, gracefully shuts down everything.
    """
    redis: Redis | None = None
    db: Database | None = None
    db_initialized = False
    price_service: FinnhubPriceProvider | None = None
    scheduler: AsyncIOScheduler | None = None

    try:
        logger.debug("Connecting to Redis")
        redis = await init_redis(settings.redis_url)
        logger.debug("Redis connected")

        if settings.database_url:
            try:
                logger.debug("Connecting to PostgreSQL")
                db = await init_database(settings.database_url)
                db_initialized = True
                logger.debug("PostgreSQL connected")
            except Exception as e:
                logger.warning(
                    "PostgreSQL connection failed, continuing without database storage",
                    error=str(e),
                )
        else:
            logger.info("No DATABASE_URL configured, skipping database storage")

        if settings.finnhub_api_key:
            price_service = await init_price_service(
                settings.finnhub_api_key.get_secret_value(), redis
            )
            await price_service.start()
            logger.debug("PriceService initialized with WebSocket")
        else:
            logger.info("No FINNHUB_API_KEY configured, price tracking disabled")

        # Set up APScheduler for periodic jobs
        scheduler = create_scheduler()

        if db:
            scheduler.add_job(
                watchlist_cleanup_job,
                IntervalTrigger(minutes=5),
                args=[db],
                id="watchlist_cleanup",
                max_instances=1,
            )

        if settings.twitterapi_api_key and settings.twitter_accounts:
            scheduler.add_job(
                twitter_agent_job,
                CronTrigger(hour=10, minute=0, timezone="America/New_York"),
                args=[db],
                id="twitter_agent",
                max_instances=1,
            )
            logger.info(
                "Twitter agent digest scheduled",
                accounts=settings.twitter_accounts,
                schedule="10:00 ET (America/New_York, DST-aware)",
            )

        fred_client = None
        nasdaq_client = None
        sec_edgar_client = None
        if db:
            nasdaq_client = NasdaqClient(redis=redis)
            sec_edgar_client = SECEdgarClient(redis=redis)

            if settings.fred_api_key:
                fred_client = FREDClient(redis=redis)

            if settings.event_radar_enabled:
                scheduler.add_job(
                    event_fetch_job,
                    CronTrigger(hour=18, minute=0, timezone="America/New_York"),
                    args=[db, redis, fred_client, nasdaq_client, sec_edgar_client],
                    id="event_fetch",
                    max_instances=1,
                )
                scheduler.add_job(
                    event_digest_job,
                    CronTrigger(hour=19, minute=0, timezone="America/New_York"),
                    args=[db, redis],
                    id="event_digest",
                    max_instances=1,
                )
                logger.info(
                    "Event Radar scheduled",
                    fetch="6pm ET daily",
                    digest="7pm ET daily",
                )
            else:
                logger.info("Event Radar disabled — skipping schedule")

        if settings.market_movers_enabled:
            scheduler.add_job(
                market_movers_job,
                CronTrigger(hour=10, minute=30, timezone="America/New_York"),
                args=[redis],
                id="market_movers",
                max_instances=1,
            )
            logger.info("Market movers scheduled", schedule="10:30am ET daily")
        else:
            logger.info("Market movers disabled — skipping schedule")

        scheduler.add_job(
            refresh_tickers_job,
            CronTrigger(day_of_week="mon", hour=6, minute=0, timezone="UTC"),
            id="refresh_tickers",
            max_instances=1,
        )
        logger.info("Ticker refresh scheduled", schedule="Monday 6am UTC weekly")

        scheduler.start()

        logger.info(
            "Agent ready",
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_model_smart=settings.llm_model_smart,
            db_enabled=db_initialized,
        )

        trigger_fns: dict[str, Any] = {}
        if settings.twitterapi_api_key and settings.twitter_accounts:

            async def _trigger_twitter_agent() -> None:
                await twitter_agent_job(db)

            trigger_fns["twitter_agent"] = _trigger_twitter_agent

        if db:

            async def _trigger_event_discover() -> int:
                return await run_structured_sources(
                    db,
                    redis,
                    fred=fred_client,
                    nasdaq=nasdaq_client,
                    sec_edgar=sec_edgar_client,
                )

            trigger_fns["event_discover"] = _trigger_event_discover

        if settings.market_movers_enabled:

            async def _trigger_market_movers() -> None:
                await _market_movers_fn(redis)

            trigger_fns["market_movers"] = _trigger_market_movers

        if db:

            async def _trigger_event_digest() -> bool:
                return await send_event_digest(db, redis=redis)

            trigger_fns["event_digest"] = _trigger_event_digest

        yield AgentState(
            redis=redis,
            db=db,
            settings=settings,
            db_enabled=db_initialized,
            scheduler=scheduler,
            trigger_fns=trigger_fns,
        )

    finally:
        logger.info("Shutting down agent...")

        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.debug("Scheduler stopped")

        if price_service:
            try:
                await close_price_service()
                logger.debug("PriceService closed")
            except Exception:
                logger.error("Error closing PriceService", exc_info=True)

        if db_initialized:
            try:
                await close_database()
                logger.debug("PostgreSQL disconnected")
            except Exception:
                logger.error("Error closing database", exc_info=True)

        if redis:
            try:
                await close_redis()
                logger.debug("Redis disconnected")
            except Exception:
                logger.error("Error closing Redis", exc_info=True)

        logger.info("Agent shutdown complete")
