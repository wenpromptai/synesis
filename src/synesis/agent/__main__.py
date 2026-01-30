"""Entry point for running the agent as a module.

This is a complete end-to-end system that:
1. Starts ingestion (Twitter WebSocket stream and/or Telegram listener)
2. Pushes incoming messages to Redis queue
3. Agent processes messages from the queue using PydanticAI

Usage:
    uv run -m synesis.agent

Configuration:
    Set in .env:
    - TWITTERAPI_API_KEY: For Twitter stream
    - TELEGRAM_API_ID, TELEGRAM_API_HASH: For Telegram
"""

import asyncio
import signal
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from redis.asyncio import Redis

from synesis.config import Settings, get_settings
from synesis.core.logging import get_logger, setup_logging
from synesis.ingestion.telegram import TelegramListener, TelegramMessage
from synesis.ingestion.twitterapi import Tweet, TwitterStreamClient
from synesis.processing.models import SourcePlatform, SourceType, UnifiedMessage
from synesis.storage.database import close_database, init_database

logger = get_logger(__name__)

# Redis queue key
INCOMING_QUEUE = "synesis:queue:incoming"


async def push_to_queue(redis: Redis, message: UnifiedMessage) -> None:
    """Push a unified message to the Redis queue for processing."""
    await redis.rpush(INCOMING_QUEUE, message.model_dump_json())  # type: ignore[misc]
    logger.debug(
        "Message queued",
        message_id=message.external_id,
        source=message.source_account,
    )


def create_tweet_to_queue_callback(
    redis: Redis, settings: Settings
) -> Callable[[Tweet], Coroutine[Any, Any, None]]:
    """Create callback that pushes tweets to Redis queue."""

    async def callback(tweet: Tweet) -> None:
        source_type_str = settings.get_twitter_source_type(tweet.username)
        source_type = SourceType.news if source_type_str == "news" else SourceType.analysis

        message = UnifiedMessage(
            external_id=tweet.tweet_id,
            source_platform=SourcePlatform.twitter,
            source_account=f"@{tweet.username}",
            text=tweet.text,
            timestamp=tweet.timestamp,
            source_type=source_type,
            raw=tweet.raw,
        )

        await push_to_queue(redis, message)
        logger.info(
            "Tweet received and queued",
            tweet_id=tweet.tweet_id,
            username=tweet.username,
            text_preview=tweet.text[:80] if tweet.text else "",
        )

    return callback


def create_telegram_to_queue_callback(
    redis: Redis, settings: Settings
) -> Callable[[TelegramMessage], Coroutine[Any, Any, None]]:
    """Create callback that pushes Telegram messages to Redis queue."""

    async def callback(telegram_msg: TelegramMessage) -> None:
        source_type_str = settings.get_telegram_source_type(telegram_msg.channel_name)
        source_type = SourceType.news if source_type_str == "news" else SourceType.analysis

        message = UnifiedMessage(
            external_id=str(telegram_msg.message_id),
            source_platform=SourcePlatform.telegram,
            source_account=telegram_msg.channel_name,
            text=telegram_msg.text,
            timestamp=telegram_msg.timestamp,
            source_type=source_type,
            raw=telegram_msg.raw,
        )

        await push_to_queue(redis, message)
        logger.info(
            "Telegram message received and queued",
            message_id=telegram_msg.message_id,
            channel=telegram_msg.channel_name,
            text_preview=telegram_msg.text[:80] if telegram_msg.text else "",
        )

    return callback


async def run_unified_agent() -> None:
    """Run the complete agent system: ingestion + processing.

    This function:
    1. Connects to Redis
    2. Starts ingestion (Twitter and/or Telegram)
    3. Ingestion callbacks push messages to Redis queue
    4. Agent loop (pydantic or claude_sdk) processes from queue
    """
    settings = get_settings()
    setup_logging(settings)

    logger.info(
        "Starting Synesis Agent",
        llm_provider=settings.llm_provider,
    )

    # Track resources for cleanup
    redis: Redis | None = None
    db_initialized = False
    twitter_client: TwitterStreamClient | None = None
    telegram_listener: TelegramListener | None = None
    agent_task: asyncio.Task[None] | None = None
    shutdown_event = asyncio.Event()

    def handle_shutdown(signum: int, frame: object) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received", signal=signum)
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        # 1. Connect to Redis
        logger.info("Connecting to Redis")
        redis = Redis.from_url(settings.redis_url)
        await redis.ping()  # type: ignore[misc]
        logger.info("Redis connected")

        # 2. Connect to PostgreSQL (if configured)
        if settings.database_url:
            try:
                logger.info("Connecting to PostgreSQL")
                await init_database(settings.database_url)
                db_initialized = True
                logger.info("PostgreSQL connected")
            except Exception as e:
                logger.warning(
                    "PostgreSQL connection failed, continuing without database storage",
                    error=str(e),
                )
        else:
            logger.info("No DATABASE_URL configured, skipping database storage")

        # 3. Start Twitter stream (if configured)
        if settings.twitterapi_api_key:
            twitter_client = TwitterStreamClient(
                api_key=settings.twitterapi_api_key.get_secret_value(),
            )
            twitter_client.on_tweet(create_tweet_to_queue_callback(redis, settings))
            await twitter_client.start()
            logger.info(
                "Twitter stream started",
                news_accounts=settings.twitter_news_accounts,
                analysis_accounts=settings.twitter_analysis_accounts,
            )
        else:
            logger.warning("TWITTERAPI_API_KEY not set, Twitter stream disabled")

        # 4. Start Telegram listener (if configured)
        if settings.telegram_api_id and settings.telegram_api_hash:
            telegram_listener = TelegramListener(
                api_id=settings.telegram_api_id,
                api_hash=settings.telegram_api_hash.get_secret_value(),
                session_name=settings.telegram_session_name,
                channels=settings.telegram_channels,
            )
            telegram_listener.on_message(create_telegram_to_queue_callback(redis, settings))
            await telegram_listener.start()
            logger.info(
                "Telegram listener started",
                channels=settings.telegram_channels,
            )
        else:
            logger.warning("Telegram credentials not set, Telegram listener disabled")

        # Check we have at least one source
        if not twitter_client and not telegram_listener:
            logger.error(
                "No data sources configured! Set TWITTERAPI_API_KEY and/or "
                "TELEGRAM_API_ID/TELEGRAM_API_HASH in .env"
            )
            sys.exit(1)

        # 5. Start agent processing loop
        def agent_exception_handler(task: asyncio.Task[None]) -> None:
            """Surface agent task exceptions immediately."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.exception("Agent task crashed!", error=str(exc), exc_info=exc)

        from synesis.agent.pydantic_runner import run_pydantic_agent

        logger.info("Starting PydanticAI agent")
        agent_task = asyncio.create_task(run_pydantic_agent())
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
            twitter_enabled=twitter_client is not None,
            telegram_enabled=telegram_listener is not None,
            db_enabled=db_initialized,
            queue=INCOMING_QUEUE,
        )

        # 6. Wait for shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.exception("Fatal error in agent", error=str(e))
        raise

    finally:
        # Graceful shutdown
        logger.info("Shutting down agent...")

        if agent_task:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

        if twitter_client:
            await twitter_client.stop()
            logger.info("Twitter stream stopped")

        if telegram_listener:
            await telegram_listener.stop()
            logger.info("Telegram listener stopped")

        if db_initialized:
            await close_database()
            logger.info("PostgreSQL disconnected")

        if redis:
            await redis.close()
            logger.info("Redis disconnected")

        logger.info("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(run_unified_agent())
