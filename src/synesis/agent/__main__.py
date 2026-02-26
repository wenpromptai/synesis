"""Agent lifecycle module used by the FastAPI server.

Provides `agent_lifespan()` — an async context manager that starts and stops
the full ingestion + processing pipeline (Telegram, PydanticAI agent workers).
The FastAPI app calls this from its own lifespan.

Configuration (set in .env):
    - TELEGRAM_API_ID, TELEGRAM_API_HASH: For Telegram
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from synesis.providers.finnhub.prices import FinnhubPriceProvider

from redis.asyncio import Redis

from synesis.agent.pydantic_runner import run_pydantic_agent
from synesis.agent.scheduler import (
    create_scheduler,
    watchlist_cleanup_job,
)
from synesis.config import Settings
from synesis.core.logging import get_logger
from synesis.ingestion.telegram import TelegramListener, TelegramMessage
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.news import SourcePlatform, UnifiedMessage
from synesis.providers.finnhub.prices import close_price_service, init_price_service
from synesis.storage.database import Database, close_database, init_database
from synesis.storage.redis import close_redis, init_redis

logger = get_logger(__name__)

# Redis queue key
INCOMING_QUEUE = "synesis:queue:incoming"


@dataclass
class AgentState:
    """Holds references to all running agent resources."""

    redis: Redis
    db: Database | None
    settings: Settings
    agent_task: asyncio.Task[None] | None
    telegram_enabled: bool
    db_enabled: bool
    scheduler: AsyncIOScheduler | None = None
    trigger_fns: dict[str, Any] = field(default_factory=dict)
    _background_tasks: list[asyncio.Task[None]] = field(default_factory=list)


async def push_to_queue(redis: Redis, message: UnifiedMessage) -> bool:
    """Push a unified message to the Redis queue for processing.

    Returns:
        True if queued successfully, False on failure.
    """
    try:
        await redis.rpush(INCOMING_QUEUE, message.model_dump_json())  # type: ignore[misc]
        logger.debug(
            "Message queued",
            message_id=message.external_id,
            source=message.source_account,
        )
        return True
    except Exception:
        logger.error(
            "Failed to queue message — MESSAGE DROPPED",
            message_id=message.external_id,
            exc_info=True,
        )
        return False


def create_telegram_to_queue_callback(
    redis: Redis,
) -> Callable[[TelegramMessage], Coroutine[Any, Any, None]]:
    """Create callback that pushes Telegram messages to Redis queue."""

    async def callback(telegram_msg: TelegramMessage) -> None:
        message = UnifiedMessage(
            external_id=str(telegram_msg.message_id),
            source_platform=SourcePlatform.telegram,
            source_account=telegram_msg.channel_name,
            text=telegram_msg.text,
            timestamp=telegram_msg.timestamp,
            raw=telegram_msg.raw,
        )

        success = await push_to_queue(redis, message)
        if success:
            logger.debug(
                "Telegram message received and queued",
                message_id=telegram_msg.message_id,
                channel=telegram_msg.channel_name,
                text_preview=telegram_msg.text[:80] if telegram_msg.text else "",
            )

    return callback


@asynccontextmanager
async def agent_lifespan(
    settings: Settings,
    shutdown_event: asyncio.Event,
) -> AsyncIterator[AgentState]:
    """Async context manager that starts/stops the entire agent pipeline.

    Yields an AgentState with references to all running resources.
    On exit, gracefully shuts down everything.
    """
    # Track resources for cleanup
    redis: Redis | None = None
    db: Database | None = None
    db_initialized = False
    price_service: FinnhubPriceProvider | None = None
    telegram_listener: TelegramListener | None = None
    agent_task: asyncio.Task[None] | None = None
    scheduler: AsyncIOScheduler | None = None

    try:
        # 1. Connect to Redis (also sets the module-level global for get_redis())
        logger.debug("Connecting to Redis")
        redis = await init_redis(settings.redis_url)
        logger.debug("Redis connected")

        # 2. Connect to PostgreSQL (if configured)
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

        # 2b. Initialize PriceService with WebSocket (if Finnhub configured)
        if settings.finnhub_api_key:
            price_service = await init_price_service(
                settings.finnhub_api_key.get_secret_value(), redis
            )
            await price_service.start()
            logger.debug("PriceService initialized with WebSocket")
        else:
            logger.info("No FINNHUB_API_KEY configured, price tracking disabled")

        # 2c. Initialize shared WatchlistManager
        watchlist = WatchlistManager(
            redis,
            db=db,
        )

        # Sync from DB
        if db:
            await watchlist.sync_from_db()

        # 3. Start Telegram listener (if configured)
        if settings.telegram_api_id and settings.telegram_api_hash:
            telegram_listener = TelegramListener(
                api_id=settings.telegram_api_id,
                api_hash=settings.telegram_api_hash.get_secret_value(),
                session_name=settings.telegram_session_name,
                channels=settings.telegram_channels,
            )
            telegram_listener.on_message(create_telegram_to_queue_callback(redis))
            await telegram_listener.start()
            logger.info(
                "Telegram listener started",
                channels=settings.telegram_channels,
            )
        else:
            logger.warning("Telegram credentials not set, Telegram listener disabled")

        # Warn if Telegram notification config is incomplete
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.warning(
                "Telegram notification config incomplete — signals will NOT be sent to Telegram. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

        # Check we have at least one source
        if telegram_listener is None:
            logger.error(
                "No data sources configured! Set TELEGRAM_API_ID/TELEGRAM_API_HASH in .env"
            )
            sys.exit(1)

        # 4. Set up APScheduler for periodic jobs
        scheduler = create_scheduler()

        # Watchlist cleanup
        if db:
            scheduler.add_job(
                watchlist_cleanup_job,
                IntervalTrigger(minutes=5),
                args=[db, watchlist],
                id="watchlist_cleanup",
                max_instances=1,
            )

        scheduler.start()

        # 5. Start agent processing loop
        def agent_exception_handler(task: asyncio.Task[None]) -> None:
            """Surface agent task exceptions immediately."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.exception("Agent task crashed!", error=str(exc), exc_info=exc)

        logger.debug("Starting PydanticAI agent")
        agent_task = asyncio.create_task(
            run_pydantic_agent(
                watchlist=watchlist,
                redis=redis,
                db=db,
            )
        )
        agent_task.add_done_callback(agent_exception_handler)

        # Give agent task time to initialize and surface any early errors
        await asyncio.sleep(0.1)
        if agent_task.done():
            exc = agent_task.exception()
            if exc:
                logger.error("Agent failed during startup", error=str(exc))
                raise exc

        # Log status
        logger.info(
            "Agent ready",
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            telegram_enabled=telegram_listener is not None,
            db_enabled=db_initialized,
            queue=INCOMING_QUEUE,
        )

        yield AgentState(
            redis=redis,
            db=db,
            settings=settings,
            agent_task=agent_task,
            telegram_enabled=telegram_listener is not None,
            db_enabled=db_initialized,
            scheduler=scheduler,
        )

    finally:
        # Graceful shutdown
        logger.info("Shutting down agent...")

        # Stop scheduler (all periodic jobs)
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.debug("Scheduler stopped")

        if agent_task:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.error("Agent task raised during shutdown", exc_info=True)

        if telegram_listener:
            try:
                await telegram_listener.stop()
                logger.debug("Telegram listener stopped")
            except Exception:
                logger.error("Error stopping Telegram listener", exc_info=True)

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
