"""PydanticAI agent runner - two-stage controlled flow implementation.

This runner uses the two-stage news processing architecture:
- Stage 1: Entity extraction (fast, no judgment calls)
- Stage 2: Smart analysis (all informed judgments with research context)

Key characteristics:
- You control the flow (not agent-driven)
- Predictable execution order
- All judgment calls happen in Stage 2 with research context
- Easy to debug

Usage:
    uv run synesis
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import asyncpg
from redis.asyncio import Redis

from synesis.config import get_settings
from synesis.core.constants import MIN_THESIS_CONFIDENCE_FOR_ALERT
from synesis.core.logging import get_logger
from synesis.core.processor import NewsProcessor, ProcessingResult
from synesis.notifications.telegram import (
    format_condensed_signal,
    format_stage1_signal,
    send_long_telegram,
)
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.news import (
    NewsSignal,
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.providers.factory import create_ticker_provider
from synesis.storage.database import Database, get_database

logger = get_logger(__name__)

# Redis queue keys
INCOMING_QUEUE = "synesis:queue:incoming"
SIGNAL_CHANNEL = "synesis:signals"


async def store_signal(signal: NewsSignal, redis: Redis) -> None:
    """Publish a signal to Redis for real-time subscribers.

    Args:
        signal: The NewsSignal to publish
        redis: Redis client
    """
    try:
        signal_json = signal.model_dump_json()
    except (TypeError, ValueError) as e:
        logger.error(
            "Failed to serialize signal",
            error=str(e),
            message_id=signal.external_id,
        )
        return

    try:
        await redis.publish(SIGNAL_CHANNEL, signal_json)
    except Exception:
        logger.error(
            "Failed to publish signal to Redis — real-time subscribers will not see this signal",
            message_id=signal.external_id,
            exc_info=True,
        )
        return

    logger.debug("Signal published", message_id=signal.external_id)


async def emit_signal_to_db(
    message: UnifiedMessage,
    extraction: LightClassification,
    analysis: SmartAnalysis | None,
) -> None:
    """Store signal to signals table (DB only, no Telegram).

    Args:
        message: Original message
        extraction: Stage 1 entity extraction
        analysis: Stage 2 smart analysis (None for low/normal urgency signals)
    """
    try:
        db = get_database()
    except RuntimeError:
        logger.debug("Database not available, skipping signal storage")
        return

    try:
        signal = NewsSignal(
            timestamp=message.timestamp,
            source_platform=message.source_platform,
            source_account=message.source_account,
            raw_text=message.text,
            external_id=message.external_id,
            news_category=extraction.news_category,
            extraction=extraction,
            analysis=analysis,
        )
        await db.insert_signal(signal)
        logger.debug(
            "Signal stored to database",
            message_id=message.external_id,
        )
    except asyncpg.UniqueViolationError:
        logger.debug("Signal already exists", message_id=message.external_id)
    except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError) as e:
        logger.warning(
            "Database connection error — signal not stored",
            error=str(e),
            message_id=message.external_id,
        )
    except Exception:
        logger.exception("Failed to store signal — DATA LOSS", message_id=message.external_id)


async def emit_prediction_to_db(evaluation: MarketEvaluation, message: UnifiedMessage) -> None:
    """Store prediction to predictions table (DB only, no Telegram).

    Args:
        evaluation: Market evaluation from Stage 2
        message: Original message for context
    """
    try:
        db = get_database()
    except RuntimeError:
        logger.debug("Database not available, skipping prediction storage")
        return

    try:
        await db.insert_prediction(evaluation, message.timestamp)
        logger.debug(
            "Prediction stored to database",
            market_id=evaluation.market_id,
            verdict=evaluation.verdict,
        )
    except Exception:
        logger.exception(
            "Failed to store prediction — DATA LOSS",
            market_id=evaluation.market_id,
        )


async def emit_stage1_telegram(
    message: UnifiedMessage,
    extraction: LightClassification,
) -> None:
    """Send Stage 1 first-pass Telegram notification.

    Sent immediately after Gate 1 for high/critical urgency signals.

    Args:
        message: Original message
        extraction: Stage 1 entity extraction
    """
    telegram_msg = format_stage1_signal(message=message, extraction=extraction)
    sent = await send_long_telegram(telegram_msg)
    if not sent:
        logger.error(
            "Failed to send Stage 1 signal to Telegram",
            message_id=message.external_id,
        )
    else:
        logger.debug(
            "Stage 1 signal sent to Telegram",
            message_id=message.external_id,
            urgency=extraction.urgency.value,
        )


async def emit_combined_telegram(
    message: UnifiedMessage,
    analysis: SmartAnalysis,
) -> None:
    """Send ONE condensed Telegram message with signal + top polymarket.

    Args:
        message: Original message
        analysis: Stage 2 smart analysis
    """
    # Format condensed message (single message, ~900-2000 chars)
    telegram_msg = format_condensed_signal(
        message=message,
        analysis=analysis,
    )

    sent = await send_long_telegram(telegram_msg)
    if not sent:
        logger.error(
            "Failed to send news signal to Telegram",
            message_id=message.external_id,
        )
    else:
        logger.debug(
            "Combined signal sent to Telegram",
            message_id=message.external_id,
            has_edge=analysis.has_tradable_edge,
            markets_evaluated=len(analysis.market_evaluations),
        )


async def emit_raw_message_to_db(message: UnifiedMessage) -> None:
    """Store raw message to raw_messages table.

    Args:
        message: The raw message to store
    """
    try:
        db = get_database()
    except RuntimeError:
        logger.debug("Database not initialized, skipping raw message storage")
        return

    try:
        await db.insert_raw_message(message)
        logger.debug("Raw message stored to DB", message_id=message.external_id)
    except Exception:
        logger.exception("Failed to store raw message", message_id=message.external_id)


async def emit_signal(
    result: ProcessingResult,
    redis: Redis,
) -> None:
    """Emit a processing result as a signal.

    Converts the ProcessingResult to a NewsSignal and:
    - Stores raw message to raw_messages table (DB)
    - Publishes signal to Redis for real-time subscribers
    - Stores signal to signals table (DB)
    - Stores each prediction to predictions table (DB)
    - Sends Stage 2 Telegram (condensed signal) if confidence gate passes
      (Stage 1 Telegram is sent earlier via on_stage1_complete callback in process_message)

    Args:
        result: The processing result to emit
        redis: Redis client
    """
    # 0. Store raw message to DB (always, even for skipped)
    await emit_raw_message_to_db(result.message)

    signal = result.to_signal()
    if signal is None:
        logger.debug("No signal to emit (skipped or no extraction)")
        return

    # Store to Redis for real-time subscribers
    await store_signal(signal, redis)

    if result.extraction is None:
        return

    urgency = result.extraction.urgency
    is_high_urgency = urgency in (UrgencyLevel.high, UrgencyLevel.critical)

    if not is_high_urgency:
        # Low/normal urgency: store extraction-only signal to DB (no Stage 2 ran)
        await emit_signal_to_db(result.message, result.extraction, None)
        return

    # Stage 1 Telegram is sent immediately via on_stage1_complete callback
    # in process_worker(), so we skip it here.

    if result.analysis is None:
        # Stage 2 failed — store extraction-only and log for visibility.
        logger.error(
            "Stage 2 analysis unavailable for high-urgency signal",
            message_id=result.message.external_id,
            urgency=urgency.value,
        )
        await emit_signal_to_db(result.message, result.extraction, None)
        return

    # 1. Store full signal to DB (extraction + analysis, signals table)
    await emit_signal_to_db(result.message, result.extraction, result.analysis)

    # 2. Store each prediction to DB (predictions table)
    for evaluation in result.analysis.market_evaluations:
        await emit_prediction_to_db(evaluation, result.message)

    # 3. Send Stage 2 Telegram (add story) if confidence gate passes
    if result.analysis.thesis_confidence < MIN_THESIS_CONFIDENCE_FOR_ALERT:
        logger.info(
            "Skipping Stage 2 Telegram alert (low confidence)",
            message_id=result.message.external_id,
            thesis_confidence=f"{result.analysis.thesis_confidence:.0%}",
        )
        return

    await emit_combined_telegram(result.message, result.analysis)


async def process_worker(
    worker_id: int,
    queue: asyncio.Queue[UnifiedMessage],
    processor: NewsProcessor,
    redis: Redis,
    shutdown: asyncio.Event,
) -> None:
    """Worker that processes messages from the queue.

    Args:
        worker_id: Unique identifier for this worker
        queue: Queue of messages to process
        processor: Initialized NewsProcessor
        redis: Redis client for signal emission
        shutdown: Event to signal graceful shutdown
    """
    log = logger.bind(worker_id=worker_id)
    log.debug("Worker started")

    while not shutdown.is_set():
        try:
            # Use timeout to check shutdown periodically
            message = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        try:

            async def _on_stage1(msg: UnifiedMessage, ext: LightClassification) -> None:
                await emit_stage1_telegram(msg, ext)

            processing_result = await processor.process_message(
                message, on_stage1_complete=_on_stage1
            )
            await emit_signal(processing_result, redis)

            # Log summary
            if processing_result.has_edge:
                best = processing_result.best_opportunity
                if best:
                    edge_str = f"{best.edge:.2%}" if best.edge else "N/A"
                    log.info(
                        "OPPORTUNITY FOUND",
                        market_id=best.market_id,
                        question=best.market_question[:50],
                        edge=edge_str,
                        side=best.recommended_side,
                    )

            log.debug(
                "Worker processed message",
                message_id=message.external_id,
                processing_time_ms=f"{processing_result.processing_time_ms:.1f}",
            )
        except Exception as e:
            log.exception("Worker error processing message", error=str(e))
        finally:
            queue.task_done()

    log.debug("Worker stopped")


async def run_pydantic_agent(
    watchlist: WatchlistManager | None = None,
    redis: Redis | None = None,
    db: Database | None = None,
) -> None:
    """Run the PydanticAI agent with concurrent worker processing.

    Args:
        watchlist: Optional shared WatchlistManager (created in __main__.py)

    Architecture:
    - Redis consumer: Pulls messages from Redis queue (runs in main coroutine)
    - Work queue: Internal asyncio.Queue for distribution
    - Worker pool: N concurrent workers process messages in parallel

    Uses simple asyncio.create_task() pattern (like arq library) which is more
    robust than TaskGroup - individual task failures don't cancel all tasks.
    """
    settings = get_settings()

    logger.debug(
        "Starting PydanticAI agent",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_model_smart=settings.llm_model_smart,
        num_workers=settings.processing_workers,
    )

    # Track ownership of resources (only close what we create)
    own_redis = redis is None
    own_db = db is None

    # Use shared Redis (passed from __main__.py)
    # If not provided (standalone mode), create local instance
    if redis is None:
        redis = Redis.from_url(settings.redis_url)
        logger.debug("Created local Redis connection (standalone mode)")

    # Help mypy narrow the type (redis is guaranteed non-None after above block)
    assert redis is not None

    shutdown = asyncio.Event()
    work_queue: asyncio.Queue[UnifiedMessage] = asyncio.Queue(
        maxsize=settings.processing_queue_size
    )
    workers: list[asyncio.Task[None]] = []
    # Standalone providers (initialized in try block)
    ticker_provider = None
    processor: NewsProcessor | None = None

    try:
        await redis.ping()  # type: ignore[misc]
        logger.debug("Connected to Redis", url=settings.redis_url.split("@")[-1])

        # Use shared Database (passed from __main__.py)
        # If not provided (standalone mode), create local instance
        if own_db:
            try:
                db = Database(settings.database_url)
                await db.connect()
                logger.debug("Database connected (standalone mode)")
            except Exception as e:
                logger.warning(
                    "Database not available, WebSocket watchlist will be empty", error=str(e)
                )

        # Initialize Finnhub ticker provider for ticker verification
        try:
            ticker_provider = await create_ticker_provider(redis)
            logger.debug("Ticker provider initialized")
        except ValueError as e:
            logger.warning(
                "Ticker provider configuration error — ticker verification disabled",
                error=str(e),
            )
        except Exception:
            logger.error("Ticker provider initialization failed", exc_info=True)

        # Use shared watchlist (passed from __main__.py)
        # If not provided (standalone mode), create local instance
        if watchlist is None:
            watchlist = WatchlistManager(redis, db=db)
            if db:
                await watchlist.sync_from_db()
            logger.debug("Created local WatchlistManager (standalone mode)")

        processor = NewsProcessor(
            redis,
            ticker_provider=ticker_provider,
            watchlist=watchlist,
        )
        await processor.initialize()

        # Start workers with simple create_task (NOT TaskGroup)
        workers = [
            asyncio.create_task(
                process_worker(i, work_queue, processor, redis, shutdown),
                name=f"worker-{i}",
            )
            for i in range(settings.processing_workers)
        ]

        logger.info(
            "Agent ready, started workers",
            queue=INCOMING_QUEUE,
            num_workers=settings.processing_workers,
        )

        # Main consumer loop (runs in main coroutine, not a task)
        while not shutdown.is_set():
            try:
                result = await redis.blpop([INCOMING_QUEUE], timeout=5)  # type: ignore[misc]

                if result is None:
                    continue

                _, data = result
                data_str = data.decode() if isinstance(data, bytes) else data

                try:
                    message = UnifiedMessage.model_validate_json(data_str)
                    await work_queue.put(message)
                    logger.debug(
                        "Message enqueued",
                        message_id=message.external_id,
                        queue_size=work_queue.qsize(),
                    )
                except Exception as e:
                    logger.error("Failed to parse message", error=str(e), data=data_str[:200])

            except asyncio.CancelledError:
                logger.info("Agent cancelled, initiating shutdown")
                shutdown.set()
                break
            except Exception as e:
                logger.exception("Redis consumer error", error=str(e))
                await asyncio.sleep(1)

    finally:
        # Signal shutdown and wait for queue to drain
        shutdown.set()

        if not work_queue.empty():
            logger.info("Waiting for queue to drain", remaining=work_queue.qsize())
            try:
                await asyncio.wait_for(work_queue.join(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Queue drain timeout, some messages may be lost")

        # Cancel workers gracefully
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        if processor:
            try:
                await processor.close()
            except Exception:
                logger.error("Error closing NewsProcessor", exc_info=True)
        if ticker_provider:
            try:
                await ticker_provider.close()
            except Exception:
                logger.error("Error closing ticker provider", exc_info=True)
        # Only close resources we created (not passed from __main__.py)
        if own_db and db:
            try:
                await db.disconnect()
            except Exception:
                logger.error("Error disconnecting database", exc_info=True)
        if own_redis:
            try:
                await redis.close()
            except Exception:
                logger.error("Error closing Redis", exc_info=True)
        logger.info("Agent stopped")


async def enqueue_test_message(redis_client: Redis | None = None) -> None:
    """Enqueue a test message for development.

    Args:
        redis_client: Optional Redis client (creates one if not provided)
    """
    settings = get_settings()

    own_redis = redis_client is None
    redis: Redis = redis_client if redis_client is not None else Redis.from_url(settings.redis_url)

    try:
        # Create a test message
        test_message = UnifiedMessage(
            external_id=f"test_{datetime.now(timezone.utc).timestamp():.0f}",
            source_platform=SourcePlatform.telegram,
            source_account="test_account",
            text="*FED CUTS RATES BY 25BPS, AS EXPECTED - Fed funds rate now at 4.00-4.25%",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )

        # Enqueue
        await redis.lpush(INCOMING_QUEUE, test_message.model_dump_json())  # type: ignore[misc]
        logger.info("Test message enqueued", message_id=test_message.external_id)

    finally:
        if own_redis:
            await redis.close()


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Enqueue a test message
        asyncio.run(enqueue_test_message())
    else:
        # Run the agent
        asyncio.run(run_pydantic_agent())
