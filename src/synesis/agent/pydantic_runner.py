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
    AGENT_MODE=pydantic python -m synesis.agent
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from redis.asyncio import Redis

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.core.processor import NewsProcessor, ProcessingResult
from synesis.notifications.telegram import (
    format_condensed_signal,
    send_telegram,
)
from synesis.processing.models import (
    Flow1Signal,
    ImpactLevel,
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.storage.database import get_database

logger = get_logger(__name__)

# Redis queue keys
INCOMING_QUEUE = "synesis:queue:incoming"
SIGNAL_CHANNEL = "synesis:signals"

# Output directory for signals
SIGNALS_DIR = Path("shared/output")


async def store_signal(signal: Flow1Signal, redis: Redis) -> None:
    """Store a signal and notify subscribers.

    Writes to both:
    - JSONL file in data/signals/
    - Redis pub/sub channel for real-time notifications

    Args:
        signal: The Flow1Signal to store
        redis: Redis client
    """
    # Ensure output directory exists
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    # Write to daily JSONL file
    date_str = signal.timestamp.strftime("%Y-%m-%d")
    output_file = SIGNALS_DIR / f"signals_{date_str}.jsonl"

    signal_json = signal.model_dump_json()

    # Use thread pool to avoid blocking event loop
    def _write_sync() -> None:
        with open(output_file, "a") as f:
            f.write(signal_json + "\n")

    await asyncio.to_thread(_write_sync)

    # Publish to Redis for real-time subscribers
    await redis.publish(SIGNAL_CHANNEL, signal_json)

    logger.info(
        "Signal stored",
        file=str(output_file),
        message_id=signal.external_id,
        has_opportunities=len(signal.opportunities) > 0,
    )


async def emit_signal_to_db(
    message: UnifiedMessage,
    extraction: LightClassification,
    analysis: SmartAnalysis,
) -> None:
    """Store signal to signals table (DB only, no Telegram).

    Args:
        message: Original message
        extraction: Stage 1 entity extraction
        analysis: Stage 2 smart analysis
    """
    try:
        db = get_database()
        signal = Flow1Signal(
            timestamp=message.timestamp,
            source_platform=message.source_platform,
            source_account=message.source_account,
            source_type=message.source_type,
            raw_text=message.text,
            external_id=message.external_id,
            news_category=extraction.news_category,
            extraction=extraction,
            analysis=analysis,
            classification=extraction,  # Legacy field
            watchlist_tickers=analysis.tickers,
            watchlist_sectors=analysis.sectors,
        )
        await db.insert_signal(signal)
        logger.debug("Signal stored to database", message_id=message.external_id)
    except RuntimeError:
        logger.debug("Database not available, skipping signal storage")
    except Exception as e:
        logger.error(
            "Failed to store signal - DATA LOSS", error=str(e), message_id=message.external_id
        )


async def emit_prediction_to_db(evaluation: MarketEvaluation, message: UnifiedMessage) -> None:
    """Store prediction to predictions table (DB only, no Telegram).

    Args:
        evaluation: Market evaluation from Stage 2
        message: Original message for context
    """
    try:
        db = get_database()
        await db.insert_prediction(evaluation, message.timestamp)
        logger.debug(
            "Prediction stored to database",
            market_id=evaluation.market_id,
            verdict=evaluation.verdict,
        )
    except RuntimeError:
        logger.debug("Database not available, skipping prediction storage")
    except Exception as e:
        logger.error(
            "Failed to store prediction - DATA LOSS",
            error=str(e),
            market_id=evaluation.market_id,
        )


async def emit_combined_telegram(
    message: UnifiedMessage,
    extraction: LightClassification,
    analysis: SmartAnalysis,
) -> None:
    """Send ONE condensed Telegram message with signal + top polymarket.

    Args:
        message: Original message
        extraction: Stage 1 entity extraction (unused, kept for API compatibility)
        analysis: Stage 2 smart analysis
    """
    # Format condensed message (single message, ~900-2000 chars)
    telegram_msg = format_condensed_signal(
        message=message,
        analysis=analysis,
    )

    # Single message, no splitting needed
    await send_telegram(telegram_msg)
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
        await db.insert_raw_message(message)
        logger.debug("Raw message stored to DB", message_id=message.external_id)
    except RuntimeError:
        # Database not initialized - skip DB storage
        logger.debug("Database not initialized, skipping raw message storage")
    except Exception as e:
        logger.error("Failed to store raw message", error=str(e))


async def emit_signal(result: ProcessingResult, redis: Redis) -> None:
    """Emit a processing result as a signal.

    Converts the ProcessingResult to a Flow1Signal and:
    - Stores raw message to raw_messages table (DB)
    - Stores it in JSONL files + Redis (legacy)
    - Stores signal to signals table (DB)
    - Stores each prediction to predictions table (DB)
    - Sends ONE combined Telegram message (signal + best polymarket edge)

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

    # Legacy: Store to JSONL and Redis
    await store_signal(signal, redis)

    if result.analysis is None or result.extraction is None:
        return

    # 1. Store signal to DB (signals table)
    await emit_signal_to_db(result.message, result.extraction, result.analysis)

    # 2. Store each prediction to DB (predictions table)
    for evaluation in result.analysis.market_evaluations:
        await emit_prediction_to_db(evaluation, result.message)

    # 3. Send ONE combined Telegram message (signal + best polymarket edge if any)
    # Skip if urgency is low OR impact is low (noise filtering)
    if (
        result.extraction.urgency == UrgencyLevel.low
        or result.analysis.predicted_impact == ImpactLevel.low
    ):
        logger.debug(
            "Skipping Telegram: low urgency OR low impact",
            message_id=result.message.external_id,
            urgency=result.extraction.urgency.value,
            impact=result.analysis.predicted_impact.value,
        )
        return

    await emit_combined_telegram(result.message, result.extraction, result.analysis)


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
            processing_result = await processor.process_message(message)
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


async def run_pydantic_agent() -> None:
    """Run the PydanticAI agent with concurrent worker processing.

    Architecture:
    - Redis consumer: Pulls messages from Redis queue (runs in main coroutine)
    - Work queue: Internal asyncio.Queue for distribution
    - Worker pool: N concurrent workers process messages in parallel

    Uses simple asyncio.create_task() pattern (like arq library) which is more
    robust than TaskGroup - individual task failures don't cancel all tasks.
    """
    settings = get_settings()

    logger.info(
        "Starting PydanticAI agent",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_model_smart=settings.llm_model_smart,
        num_workers=settings.processing_workers,
    )

    redis = Redis.from_url(settings.redis_url)
    shutdown = asyncio.Event()
    work_queue: asyncio.Queue[UnifiedMessage] = asyncio.Queue(
        maxsize=settings.processing_queue_size
    )
    workers: list[asyncio.Task[None]] = []

    try:
        await redis.ping()
        logger.info("Connected to Redis", url=settings.redis_url.split("@")[-1])

        processor = NewsProcessor(redis)
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
                result = await redis.blpop(INCOMING_QUEUE, timeout=5)

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

        await processor.close()
        await redis.close()
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
            source_platform=SourcePlatform.twitter,
            source_account="test_account",
            text="*FED CUTS RATES BY 25BPS, AS EXPECTED - Fed funds rate now at 4.00-4.25%",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
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
