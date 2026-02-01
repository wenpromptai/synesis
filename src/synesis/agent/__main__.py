"""Entry point for running the agent as a module.

This is a complete end-to-end system that:
1. Starts ingestion (Twitter WebSocket stream, Telegram listener, Reddit RSS)
2. Pushes incoming messages to Redis queue (Flow 1)
3. Processes Reddit posts through sentiment pipeline (Flow 2)
4. Agent processes messages from the queue using PydanticAI

Usage:
    uv run -m synesis.agent

Configuration:
    Set in .env:
    - TWITTERAPI_API_KEY: For Twitter stream
    - TELEGRAM_API_ID, TELEGRAM_API_HASH: For Telegram
    - REDDIT_SUBREDDITS: For Reddit RSS (default: wallstreetbets,stocks,options)
"""

import asyncio
import signal
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from redis.asyncio import Redis

from synesis.config import Settings, get_settings
from synesis.core.logging import get_logger, setup_logging
from synesis.ingestion.prices import PriceService
from synesis.ingestion.reddit import RedditPost, RedditRSSClient
from synesis.ingestion.telegram import TelegramListener, TelegramMessage
from synesis.ingestion.twitterapi import Tweet, TwitterStreamClient
from synesis.notifications.telegram import format_sentiment_signal, send_telegram
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.sentiment import SentimentProcessor
from synesis.processing.news import SourcePlatform, SourceType, UnifiedMessage
from synesis.storage.database import Database, close_database, init_database

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


def create_reddit_to_sentiment_callback(
    sentiment_processor: SentimentProcessor,
) -> Callable[[RedditPost], Coroutine[Any, Any, None]]:
    """Create callback that processes Reddit posts through sentiment pipeline.

    Unlike Twitter/Telegram which go to the main queue for news processing,
    Reddit posts go directly to the sentiment pipeline.
    """

    async def callback(post: RedditPost) -> None:
        # Process through Gate 1 (lexicon analysis) and buffer
        result = await sentiment_processor.process_post(post)

        logger.info(
            "Reddit post processed (sentiment)",
            post_id=post.post_id,
            subreddit=post.subreddit,
            sentiment=f"{result.compound:.2f}",
            tickers=result.tickers_mentioned[:5],  # Log first 5
            text_preview=post.title[:60],
        )

    return callback


async def run_sentiment_signal_loop(
    sentiment_processor: SentimentProcessor,
    redis: Redis,
    interval_seconds: int,
    shutdown_event: asyncio.Event,
) -> None:
    """Run the sentiment signal generation loop.

    Generates sentiment signals at the configured interval (default 6 hours).
    Also publishes signals to Redis pub/sub for real-time subscribers.
    """
    signal_channel = "synesis:sentiment:signals"

    while not shutdown_event.is_set():
        try:
            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=interval_seconds,
                )
                # Shutdown event was set
                break
            except TimeoutError:
                # Interval elapsed, generate signal
                pass

            logger.info("Generating sentiment signal")
            signal = await sentiment_processor.generate_signal()

            # Publish signal to Redis for subscribers
            signal_json = signal.model_dump_json()
            await redis.publish(signal_channel, signal_json)

            # Send to Telegram
            telegram_message = format_sentiment_signal(signal)
            await send_telegram(telegram_message)

            logger.info(
                "Sentiment signal generated and published",
                watchlist_size=len(signal.watchlist),
                posts_analyzed=signal.total_posts_analyzed,
                overall_sentiment=signal.overall_sentiment,
            )

        except Exception as e:
            logger.exception("Sentiment signal generation failed", error=str(e))
            # Continue running, don't crash the loop


async def run_price_outcome_loop(
    db: Database,
    price_service: PriceService,
    shutdown_event: asyncio.Event,
    watchlist: WatchlistManager | None = None,
    check_interval: int = 300,  # 5 minutes
) -> None:
    """Fill in price outcomes (1h, 6h, 24h) for signals and sentiment snapshots.

    Also handles watchlist expiration cleanup for both PostgreSQL and Redis.

    Uses batched price fetching for efficiency:
    1. Collect all unique tickers across pending records
    2. Single batch fetch from cache/REST
    3. Update all records from the batch result

    Args:
        db: Database instance
        price_service: PriceService for fetching current prices
        shutdown_event: Event to signal shutdown
        watchlist: Optional WatchlistManager for Redis cleanup
        check_interval: Seconds between checks (default 5 minutes)
    """
    outcome_types = ["1h", "6h", "24h"]

    while not shutdown_event.is_set():
        try:
            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=check_interval)
                break
            except TimeoutError:
                pass

            # Cleanup expired watchlist tickers in PostgreSQL
            expired_pg = await db.deactivate_expired_watchlist()
            if expired_pg:
                logger.info(
                    "Deactivated expired watchlist tickers (PostgreSQL)", tickers=expired_pg
                )

            # Cleanup expired watchlist tickers in Redis (if watchlist provided)
            if watchlist:
                expired_redis = await watchlist.cleanup_expired()
                if expired_redis:
                    logger.info("Removed expired watchlist tickers (Redis)", tickers=expired_redis)

            for outcome_type in outcome_types:
                # ─── SIGNALS (multiple tickers per row) ───
                signals = await db.get_signals_pending_price_outcomes(outcome_type, limit=100)
                if signals:
                    # Collect ALL unique tickers across ALL pending signals
                    signal_tickers: set[str] = set()
                    for sig in signals:
                        if sig["tickers"]:
                            signal_tickers.update(sig["tickers"])

                    # One batch fetch
                    if signal_tickers:
                        all_signal_prices = await price_service.get_prices(
                            list(signal_tickers), fallback_to_rest=True
                        )

                        # Update each signal with its subset
                        for sig in signals:
                            if sig["tickers"]:
                                prices = {
                                    t: all_signal_prices[t.upper()]
                                    for t in sig["tickers"]
                                    if t.upper() in all_signal_prices
                                }
                                if prices:
                                    await db.update_signal_price_outcome(
                                        sig["time"], sig["flow_id"], outcome_type, prices
                                    )

                    logger.info(
                        "Updated signals for price outcome",
                        outcome_type=outcome_type,
                        count=len(signals),
                    )

                # ─── SENTIMENT SNAPSHOTS (one ticker per row) ───
                snapshots = await db.get_sentiment_snapshots_pending_price_outcomes(
                    outcome_type, limit=100
                )
                if snapshots:
                    # Collect unique tickers
                    snapshot_tickers = list({s["ticker"] for s in snapshots})

                    # One batch fetch
                    all_snapshot_prices = await price_service.get_prices(
                        snapshot_tickers, fallback_to_rest=True
                    )

                    # Update each snapshot
                    for snapshot in snapshots:
                        ticker = snapshot["ticker"].upper()
                        if ticker in all_snapshot_prices:
                            await db.update_sentiment_snapshot_price_outcome(
                                snapshot["id"], outcome_type, all_snapshot_prices[ticker]
                            )

                    logger.info(
                        "Updated snapshots for price outcome",
                        outcome_type=outcome_type,
                        count=len(snapshots),
                    )

        except Exception as e:
            logger.exception("Price outcome update failed", error=str(e))
            # Continue running, don't crash the loop


async def run_unified_agent() -> None:
    """Run the complete agent system: ingestion + processing.

    This function:
    1. Connects to Redis
    2. Starts ingestion (Twitter, Telegram, Reddit RSS)
    3. Ingestion callbacks push messages to Redis queue (Flow 1)
    4. Reddit posts go through Flow 2 sentiment pipeline
    5. Agent loop (pydantic) processes from queue
    """
    settings = get_settings()
    setup_logging(settings)

    logger.info(
        "Starting Synesis Agent",
        llm_provider=settings.llm_provider,
    )

    # Track resources for cleanup
    redis: Redis | None = None
    db: Database | None = None
    db_initialized = False
    price_service: PriceService | None = None
    twitter_client: TwitterStreamClient | None = None
    telegram_listener: TelegramListener | None = None
    reddit_client: RedditRSSClient | None = None
    sentiment_processor: SentimentProcessor | None = None
    agent_task: asyncio.Task[None] | None = None
    sentiment_signal_task: asyncio.Task[None] | None = None
    price_outcome_task: asyncio.Task[None] | None = None
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
                db = await init_database(settings.database_url)
                db_initialized = True
                logger.info("PostgreSQL connected")
            except Exception as e:
                logger.warning(
                    "PostgreSQL connection failed, continuing without database storage",
                    error=str(e),
                )
        else:
            logger.info("No DATABASE_URL configured, skipping database storage")

        # 2b. Initialize PriceService with WebSocket (if Finnhub configured)
        if settings.finnhub_api_key:
            price_service = PriceService(settings.finnhub_api_key.get_secret_value(), redis)
            await price_service.start_websocket()
            logger.info("PriceService initialized with WebSocket")
        else:
            logger.info("No FINNHUB_API_KEY configured, price tracking disabled")

        # 2c. Initialize shared WatchlistManager with WebSocket callbacks
        on_ticker_added = None
        on_ticker_removed = None
        if price_service:

            async def subscribe_ticker(ticker: str) -> None:
                await price_service.subscribe([ticker])

            async def unsubscribe_ticker(ticker: str) -> None:
                await price_service.unsubscribe([ticker])

            on_ticker_added = subscribe_ticker
            on_ticker_removed = unsubscribe_ticker

        watchlist = WatchlistManager(
            redis,
            db=db,
            on_ticker_added=on_ticker_added,
            on_ticker_removed=on_ticker_removed,
        )

        # Sync from DB and subscribe existing tickers
        if db:
            await watchlist.sync_from_db()
            if price_service:
                tickers = await watchlist.get_all()
                if tickers:
                    await price_service.subscribe(tickers)
                    logger.info("Subscribed to initial watchlist", count=len(tickers))

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

        # 5. Start Reddit RSS client + sentiment processor (if configured)
        if settings.reddit_subreddits:
            # Initialize sentiment processor (with shared watchlist)
            sentiment_processor = SentimentProcessor(
                settings, redis, db=db, price_service=price_service, watchlist=watchlist
            )

            # Initialize Reddit RSS client
            reddit_client = RedditRSSClient(
                subreddits=settings.reddit_subreddits,
                poll_interval=settings.reddit_poll_interval,
            )
            reddit_client.on_post(create_reddit_to_sentiment_callback(sentiment_processor))
            await reddit_client.start()

            logger.info(
                "Reddit RSS + sentiment started",
                subreddits=settings.reddit_subreddits,
                poll_interval_hours=settings.reddit_poll_interval // 3600,
            )

            # Start sentiment signal generation loop (same interval as polling)
            sentiment_signal_task = asyncio.create_task(
                run_sentiment_signal_loop(
                    sentiment_processor,
                    redis,
                    interval_seconds=settings.reddit_poll_interval,
                    shutdown_event=shutdown_event,
                )
            )
        else:
            logger.info("No Reddit subreddits configured, sentiment disabled")

        # Check we have at least one source (news or sentiment)
        has_news_source = twitter_client is not None or telegram_listener is not None
        has_sentiment_source = reddit_client is not None
        if not has_news_source and not has_sentiment_source:
            logger.error(
                "No data sources configured! Set TWITTERAPI_API_KEY, "
                "TELEGRAM_API_ID/TELEGRAM_API_HASH, or REDDIT_SUBREDDITS in .env"
            )
            sys.exit(1)

        # 5b. Start price outcome tracking loop (if DB and PriceService available)
        if db and price_service:
            price_outcome_task = asyncio.create_task(
                run_price_outcome_loop(db, price_service, shutdown_event, watchlist=watchlist)
            )
            logger.info("Price outcome tracking loop started")

        # 6. Start agent processing loop
        def agent_exception_handler(task: asyncio.Task[None]) -> None:
            """Surface agent task exceptions immediately."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.exception("Agent task crashed!", error=str(exc), exc_info=exc)

        from synesis.agent.pydantic_runner import run_pydantic_agent

        logger.info("Starting PydanticAI agent")
        agent_task = asyncio.create_task(
            run_pydantic_agent(
                price_service=price_service,
                finnhub_api_key=settings.finnhub_api_key,
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
            twitter_enabled=twitter_client is not None,
            telegram_enabled=telegram_listener is not None,
            reddit_enabled=reddit_client is not None,
            sentiment_enabled=sentiment_processor is not None,
            price_tracking_enabled=price_outcome_task is not None,
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

        if sentiment_signal_task:
            sentiment_signal_task.cancel()
            try:
                await sentiment_signal_task
            except asyncio.CancelledError:
                pass
            logger.info("Sentiment signal loop stopped")

        if price_outcome_task:
            price_outcome_task.cancel()
            try:
                await price_outcome_task
            except asyncio.CancelledError:
                pass
            logger.info("Price outcome loop stopped")

        if twitter_client:
            await twitter_client.stop()
            logger.info("Twitter stream stopped")

        if telegram_listener:
            await telegram_listener.stop()
            logger.info("Telegram listener stopped")

        if reddit_client:
            await reddit_client.stop()
            logger.info("Reddit RSS client stopped")

        if sentiment_processor:
            await sentiment_processor.close()
            logger.info("Sentiment processor closed")

        if price_service:
            await price_service.close()
            logger.info("PriceService closed")

        if db_initialized:
            await close_database()
            logger.info("PostgreSQL disconnected")

        if redis:
            await redis.close()
            logger.info("Redis disconnected")

        logger.info("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(run_unified_agent())
