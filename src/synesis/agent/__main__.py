"""Agent lifecycle module used by the FastAPI server.

Provides `agent_lifespan()` — an async context manager that starts and stops
the full ingestion + processing pipeline (Twitter, Telegram, Reddit, sentiment,
PydanticAI agent workers). The FastAPI app calls this from its own lifespan.

Configuration (set in .env):
    - TWITTERAPI_API_KEY: For Twitter stream
    - TELEGRAM_API_ID, TELEGRAM_API_HASH: For Telegram
    - REDDIT_SUBREDDITS: For Reddit RSS (default: wallstreetbets,stocks,options)
"""

from __future__ import annotations

import asyncio
import sys
import time as _time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synesis.processing.mkt_intel.processor import MarketIntelProcessor
    from synesis.processing.watchlist.processor import WatchlistProcessor
    from synesis.providers.finnhub.prices import FinnhubPriceProvider

from redis.asyncio import Redis

from synesis.config import Settings
from synesis.core.logging import get_logger
from synesis.providers.factory import create_ticker_provider
from synesis.providers.finnhub.prices import close_price_service, init_price_service
from synesis.ingestion.reddit import RedditPost, RedditRSSClient
from synesis.ingestion.telegram import TelegramListener, TelegramMessage
from synesis.ingestion.twitterapi import Tweet, TwitterStreamClient
from synesis.notifications.telegram import (
    format_arb_alert,
    format_mkt_intel_signal,
    format_sentiment_signal,
    format_watchlist_signal,
    send_long_telegram,
)
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.sentiment import SentimentProcessor
from synesis.processing.news import SourcePlatform, SourceType, UnifiedMessage
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
    twitter_enabled: bool
    telegram_enabled: bool
    reddit_enabled: bool
    sentiment_enabled: bool
    db_enabled: bool
    _background_tasks: list[asyncio.Task[None]] = field(default_factory=list)


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
        logger.debug(
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
        logger.debug(
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

        logger.debug(
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
    startup_delay: int = 60,  # Wait 60s on startup for Reddit to fetch posts
) -> None:
    """Run the sentiment signal generation loop.

    On startup, waits briefly for Reddit to fetch initial posts, then generates
    the first signal. After that, generates signals at the configured interval.
    Also publishes signals to Redis pub/sub.
    """
    signal_channel = "synesis:sentiment:signals"
    first_run = True

    while not shutdown_event.is_set():
        try:
            # Determine wait time: short delay on first run, full interval after
            wait_time = startup_delay if first_run else interval_seconds

            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=wait_time,
                )
                # Shutdown event was set
                break
            except TimeoutError:
                # Wait elapsed, generate signal
                pass

            first_run = False
            logger.debug("Generating sentiment signal")
            signal = await sentiment_processor.generate_signal()

            # Publish signal to Redis for subscribers
            signal_json = signal.model_dump_json()
            await redis.publish(signal_channel, signal_json)

            # Send to Telegram
            telegram_message = format_sentiment_signal(signal)
            sent = await send_long_telegram(telegram_message)
            if not sent:
                logger.error(
                    "Failed to send sentiment signal to Telegram",
                    watchlist_size=len(signal.watchlist),
                    message_length=len(telegram_message),
                )

            logger.info(
                "Sentiment signal generated and published",
                watchlist_size=len(signal.watchlist),
                posts_analyzed=signal.total_posts_analyzed,
                overall_sentiment=signal.overall_sentiment,
            )

        except Exception as e:
            logger.exception("Sentiment signal generation failed", error=str(e))
            # Continue running, don't crash the loop


async def run_watchlist_cleanup_loop(
    db: Database,
    shutdown_event: asyncio.Event,
    watchlist: WatchlistManager | None = None,
    check_interval: int = 300,  # 5 minutes
) -> None:
    """Periodically cleanup expired watchlist tickers.

    Handles watchlist expiration cleanup for both PostgreSQL and Redis.

    Args:
        db: Database instance
        shutdown_event: Event to signal shutdown
        watchlist: Optional WatchlistManager for Redis cleanup
        check_interval: Seconds between checks (default 5 minutes)
    """
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

        except Exception as e:
            logger.exception("Watchlist cleanup failed", error=str(e))
            # Continue running, don't crash the loop


async def run_mkt_intel_loop(
    processor: "MarketIntelProcessor",
    redis: Redis,
    interval_seconds: int,
    shutdown_event: asyncio.Event,
    startup_delay: int = 30,
) -> None:
    """Run the market intelligence signal generation loop.

    Scans markets every interval, generates signals, sends Telegram notifications.
    Also publishes signals to Redis pub/sub.
    """
    from synesis.core.constants import MARKET_INTEL_REDIS_PREFIX

    signal_channel = "synesis:mkt_intel:signals"
    trigger_key = f"{MARKET_INTEL_REDIS_PREFIX}:trigger_scan"
    first_run = True

    while not shutdown_event.is_set():
        try:
            wait_time = startup_delay if first_run else interval_seconds

            # Wait for interval, shutdown, or manual trigger
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                break
            except TimeoutError:
                pass

            # Check for manual trigger
            triggered = await redis.get(trigger_key)
            if triggered:
                await redis.delete(trigger_key)
                logger.info("Manual mkt_intel scan triggered")

            first_run = False
            logger.debug("Running mkt_intel scan")
            signal = await processor.run_scan()

            # Publish to Redis
            signal_json = signal.model_dump_json()
            await redis.publish(signal_channel, signal_json)

            # Send Telegram
            telegram_message = format_mkt_intel_signal(signal)
            sent = await send_long_telegram(telegram_message)
            if not sent:
                logger.error(
                    "Failed to send mkt_intel signal to Telegram",
                    opportunities=len(signal.opportunities),
                )

            logger.info(
                "Mkt_intel signal generated and published",
                markets_scanned=signal.total_markets_scanned,
                opportunities=len(signal.opportunities),
            )

        except Exception as e:
            logger.exception("Mkt_intel signal generation failed", error=str(e))


async def run_watchlist_intel_loop(
    processor: "WatchlistProcessor",
    redis: Redis,
    interval_seconds: int,
    shutdown_event: asyncio.Event,
    startup_delay: int = 120,
) -> None:
    """Run the watchlist intelligence analysis loop.

    Periodically sweeps watchlist tickers with fundamental providers.
    """
    from synesis.core.constants import WATCHLIST_INTEL_REDIS_PREFIX

    signal_channel = "synesis:watchlist_intel:signals"
    trigger_key = f"{WATCHLIST_INTEL_REDIS_PREFIX}:trigger"
    first_run = True

    while not shutdown_event.is_set():
        try:
            wait_time = startup_delay if first_run else interval_seconds

            # Wait for interval, shutdown, or manual trigger
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                break
            except TimeoutError:
                pass

            # Check for manual trigger
            triggered = await redis.get(trigger_key)
            if triggered:
                await redis.delete(trigger_key)
                logger.info("Manual watchlist intel scan triggered")

            first_run = False
            logger.debug("Running watchlist intel scan")
            signal = await processor.run_analysis()

            # Publish to Redis
            signal_json = signal.model_dump_json()
            await redis.publish(signal_channel, signal_json)

            # Send Telegram
            telegram_message = format_watchlist_signal(signal)
            sent = await send_long_telegram(telegram_message)
            if not sent:
                logger.error(
                    "Failed to send watchlist intel signal to Telegram",
                    tickers=signal.tickers_analyzed,
                )

            logger.info(
                "Watchlist intel signal generated and published",
                tickers_analyzed=signal.tickers_analyzed,
                alerts=len(signal.alerts),
            )

        except Exception as e:
            logger.exception("Watchlist intel generation failed", error=str(e))


async def run_arb_monitor(
    redis: Redis,
    scanner: Any,
    ws_manager: Any,
    shutdown_event: asyncio.Event,
    cooldown_minutes: int = 10,
) -> None:
    """Monitor Redis price updates and alert on cross-platform arb opportunities.

    Subscribes to price update channel and checks matched pairs for gaps.
    Polymarket WS publishes ``polymarket:{token_id}:{price}`` where token_id
    is a CLOB token ID (YES side).  We build a lookup dict from token IDs so
    incoming messages can be matched to the correct market pair.
    """
    from synesis.core.constants import CROSS_PLATFORM_ARB_MIN_GAP, PRICE_UPDATE_CHANNEL
    from synesis.markets.models import CrossPlatformArb

    pubsub = redis.pubsub()
    await pubsub.subscribe(PRICE_UPDATE_CHANNEL)
    cooldowns: dict[str, float] = {}  # pair_key → last_alert_time

    # Cached lookup dicts — rebuilt only when cross_platform_matches changes
    cached_version: int = -1
    poly_token_to_pair: dict[str, tuple[Any, Any]] = {}
    kalshi_ticker_to_pair: dict[str, tuple[Any, Any]] = {}

    logger.debug("Arb monitor started, listening for price updates")

    try:
        while not shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                    timeout=5.0,
                )
            except TimeoutError:
                # Prune stale cooldowns periodically (on every timeout)
                cutoff = _time.time() - cooldown_minutes * 60
                stale = [k for k, v in cooldowns.items() if v < cutoff]
                for k in stale:
                    del cooldowns[k]
                continue

            if message is None or message["type"] != "message":
                continue

            try:
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                parts = data.split(":", 2)
                if len(parts) != 3:
                    logger.debug("Malformed price update", data=data)
                    continue
                platform, market_id, price_str = parts
                new_price = float(price_str)

                # Rebuild lookup dicts only when matched pairs change
                if scanner.matches_version != cached_version:
                    cached_version = scanner.matches_version
                    matches = scanner.cross_platform_matches
                    poly_token_to_pair = {}
                    kalshi_ticker_to_pair = {}
                    for poly_mkt, kalshi_mkt in matches:
                        if poly_mkt.yes_token_id:
                            poly_token_to_pair[poly_mkt.yes_token_id] = (poly_mkt, kalshi_mkt)
                        kalshi_id = kalshi_mkt.ticker or kalshi_mkt.external_id
                        if kalshi_id:
                            kalshi_ticker_to_pair[kalshi_id] = (poly_mkt, kalshi_mkt)

                # Match the incoming message to a cross-platform pair
                pair: tuple[Any, Any] | None = None
                counterpart_platform: str | None = None
                counterpart_ws_key: str | None = None

                if platform == "polymarket":
                    pair = poly_token_to_pair.get(market_id)
                    if pair:
                        counterpart_platform = "kalshi"
                        counterpart_ws_key = pair[1].ticker or pair[1].external_id
                elif platform == "kalshi":
                    pair = kalshi_ticker_to_pair.get(market_id)
                    if pair:
                        counterpart_platform = "polymarket"
                        # WS manager stores Polymarket prices by token ID
                        counterpart_ws_key = pair[0].yes_token_id

                if not pair or not counterpart_platform or not counterpart_ws_key:
                    continue

                poly_mkt, kalshi_mkt = pair

                # Get other platform's price from WS manager
                if not ws_manager:
                    continue
                other_price_data = await ws_manager.get_realtime_price(
                    counterpart_platform, counterpart_ws_key
                )
                if not other_price_data:
                    continue
                other_price = other_price_data[0]

                gap = abs(new_price - other_price)
                if gap < CROSS_PLATFORM_ARB_MIN_GAP:
                    continue

                poly_condition_id = poly_mkt.condition_id or poly_mkt.external_id
                kalshi_ticker = kalshi_mkt.ticker or kalshi_mkt.external_id
                pair_key = f"{poly_condition_id}:{kalshi_ticker}"
                now = _time.time()
                last_alert = cooldowns.get(pair_key, 0)
                if now - last_alert < cooldown_minutes * 60:
                    continue

                # Build arb and send alert
                if platform == "polymarket":
                    poly_price = new_price
                    kalshi_price = other_price
                else:
                    poly_price = other_price
                    kalshi_price = new_price

                buy_platform = "polymarket" if poly_price < kalshi_price else "kalshi"
                arb = CrossPlatformArb(
                    polymarket=poly_mkt,
                    kalshi=kalshi_mkt,
                    price_gap=gap,
                    suggested_buy_platform=buy_platform,
                    suggested_side="yes",
                    match_similarity=0.0,
                )
                alert_msg = format_arb_alert(arb)
                try:
                    await send_long_telegram(alert_msg)
                    cooldowns[pair_key] = now
                    logger.info(
                        "Arb alert sent",
                        gap=f"${gap:.2f}",
                        poly=f"${poly_price:.2f}",
                        kalshi=f"${kalshi_price:.2f}",
                    )
                except Exception as e:
                    logger.error("Arb alert send failed", error=str(e), gap=f"${gap:.2f}")

            except Exception as e:
                logger.warning("Arb monitor message processing error", error=str(e))

    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe(PRICE_UPDATE_CHANNEL)
        await pubsub.close()
        logger.debug("Arb monitor stopped")


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
    twitter_client: TwitterStreamClient | None = None
    telegram_listener: TelegramListener | None = None
    reddit_client: RedditRSSClient | None = None
    sentiment_processor: SentimentProcessor | None = None
    agent_task: asyncio.Task[None] | None = None
    sentiment_signal_task: asyncio.Task[None] | None = None
    watchlist_cleanup_task: asyncio.Task[None] | None = None
    mkt_intel_task: asyncio.Task[None] | None = None
    mkt_intel_ws_manager = None
    mkt_intel_kalshi_client = None
    mkt_intel_poly_client = None
    mkt_intel_data_client = None
    arb_monitor_task: asyncio.Task[None] | None = None
    watchlist_intel_task: asyncio.Task[None] | None = None
    ticker_provider = None
    watchlist_intel_factset = None
    watchlist_intel_sec_edgar = None
    watchlist_intel_nasdaq = None

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
            # Initialize ticker provider (FactSet or Finnhub based on config)
            try:
                ticker_provider = await create_ticker_provider(redis)
            except Exception as e:
                logger.error("Ticker provider not available, sentiment will use LLM-only validation", error=str(e))

            # Initialize sentiment processor (with shared watchlist)
            sentiment_processor = SentimentProcessor(
                settings,
                redis,
                db=db,
                watchlist=watchlist,
                ticker_provider=ticker_provider,
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

            # Start sentiment signal generation loop
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

        # 5c. Start Market Intelligence (Flow 3) if enabled
        if settings.mkt_intel_enabled:
            from synesis.markets.kalshi import KalshiClient
            from synesis.markets.polymarket import (
                PolymarketClient as PolyClient,
                PolymarketDataClient,
            )
            from synesis.processing.mkt_intel.processor import MarketIntelProcessor as MktIntelProc
            from synesis.processing.mkt_intel.scanner import MarketScanner
            from synesis.processing.mkt_intel.wallets import WalletTracker

            # Create REST clients
            kalshi_client = KalshiClient()
            poly_client = PolyClient()
            poly_data_client = PolymarketDataClient()
            mkt_intel_kalshi_client = kalshi_client
            mkt_intel_poly_client = poly_client
            mkt_intel_data_client = poly_data_client

            # Create WebSocket clients (if enabled)
            ws_manager = None
            if settings.mkt_intel_ws_enabled and redis:
                from synesis.markets.kalshi_ws import KalshiWSClient
                from synesis.markets.polymarket_ws import PolymarketWSClient
                from synesis.markets.ws_manager import MarketWSManager

                poly_ws = PolymarketWSClient(redis)
                kalshi_ws = KalshiWSClient(redis)
                ws_manager = MarketWSManager(poly_ws, kalshi_ws, redis)
                mkt_intel_ws_manager = ws_manager
                await ws_manager.start()
                logger.debug("Market intelligence WebSocket clients started")

            # Create scanner
            scanner = MarketScanner(
                polymarket=poly_client,
                kalshi=kalshi_client,
                ws_manager=ws_manager,
                db=db,
                expiring_hours=settings.mkt_intel_expiring_hours,
                volume_spike_threshold=settings.mkt_intel_volume_spike_threshold,
            )

            # Create wallet tracker
            wallet_tracker = WalletTracker(
                redis=redis,
                db=db,
                data_client=poly_data_client,
                insider_score_min=settings.mkt_intel_insider_score_min,
            )

            # Create processor
            mkt_intel_processor = MktIntelProc(
                settings=settings,
                scanner=scanner,
                wallet_tracker=wallet_tracker,
                ws_manager=ws_manager,
                db=db,
            )

            # Start 15-min aggregation loop
            mkt_intel_task = asyncio.create_task(
                run_mkt_intel_loop(
                    mkt_intel_processor,
                    redis,
                    settings.mkt_intel_interval,
                    shutdown_event,
                )
            )

            # Start real-time arb monitor (if WebSocket enabled)
            if ws_manager:
                arb_monitor_task = asyncio.create_task(
                    run_arb_monitor(
                        redis=redis,
                        scanner=scanner,
                        ws_manager=ws_manager,
                        shutdown_event=shutdown_event,
                    )
                )
                logger.debug("Real-time arb monitor started")

            logger.info(
                "Market intelligence started",
                interval=settings.mkt_intel_interval,
                ws_enabled=settings.mkt_intel_ws_enabled,
            )
        else:
            logger.info("Market intelligence disabled (mkt_intel_enabled=False)")

        # 5d. Start Watchlist Intelligence (Flow 4) if enabled
        if settings.watchlist_intel_enabled and redis:
            from synesis.processing.watchlist.processor import (
                WatchlistProcessor as WatchlistProc,
            )

            # Initialize fundamental providers for watchlist only
            try:
                from synesis.providers.factset.client import FactSetClient as FSClient
                from synesis.providers.factset.provider import (
                    FactSetProvider as FSProvider,
                )

                fs_client = FSClient()
                watchlist_intel_factset = FSProvider(client=fs_client)
                logger.debug("Watchlist intel: FactSet provider initialized")
            except Exception as e:
                logger.warning("Watchlist intel: FactSet not available", error=str(e))

            try:
                from synesis.providers.nasdaq.client import NasdaqClient as NQClient
                from synesis.providers.sec_edgar.client import (
                    SECEdgarClient as SECClient,
                )

                watchlist_intel_sec_edgar = SECClient(redis=redis)
                watchlist_intel_nasdaq = NQClient(redis=redis)
                logger.debug("Watchlist intel: SEC EDGAR + NASDAQ initialized")
            except Exception as e:
                logger.warning("Watchlist intel: SEC EDGAR/NASDAQ not available", error=str(e))

            watchlist_intel_processor = WatchlistProc(
                watchlist=watchlist,
                factset=watchlist_intel_factset,
                sec_edgar=watchlist_intel_sec_edgar,
                nasdaq=watchlist_intel_nasdaq,
                db=db,
                ticker_provider=ticker_provider,
            )

            watchlist_intel_task = asyncio.create_task(
                run_watchlist_intel_loop(
                    watchlist_intel_processor,
                    redis,
                    settings.watchlist_intel_interval,
                    shutdown_event,
                )
            )
            logger.info(
                "Watchlist intelligence started",
                interval=settings.watchlist_intel_interval,
            )
        else:
            logger.info("Watchlist intelligence disabled (watchlist_intel_enabled=False)")

        # Warn if Telegram notification config is incomplete
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.warning(
                "Telegram notification config incomplete — signals will NOT be sent to Telegram. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

        # Check we have at least one source (news or sentiment)
        has_news_source = twitter_client is not None or telegram_listener is not None
        has_sentiment_source = reddit_client is not None
        if not has_news_source and not has_sentiment_source:
            logger.error(
                "No data sources configured! Set TWITTERAPI_API_KEY, "
                "TELEGRAM_API_ID/TELEGRAM_API_HASH, or REDDIT_SUBREDDITS in .env"
            )
            sys.exit(1)

        # 5b. Start watchlist cleanup loop (if DB available)
        if db:
            watchlist_cleanup_task = asyncio.create_task(
                run_watchlist_cleanup_loop(db, shutdown_event, watchlist=watchlist)
            )
            logger.debug("Watchlist cleanup loop started")

        # 6. Start agent processing loop
        def agent_exception_handler(task: asyncio.Task[None]) -> None:
            """Surface agent task exceptions immediately."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.exception("Agent task crashed!", error=str(exc), exc_info=exc)

        from synesis.agent.pydantic_runner import run_pydantic_agent

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
            twitter_enabled=twitter_client is not None,
            telegram_enabled=telegram_listener is not None,
            reddit_enabled=reddit_client is not None,
            sentiment_enabled=sentiment_processor is not None,
            mkt_intel_enabled=mkt_intel_task is not None,
            watchlist_intel_enabled=watchlist_intel_task is not None,
            watchlist_cleanup_enabled=watchlist_cleanup_task is not None,
            db_enabled=db_initialized,
            queue=INCOMING_QUEUE,
        )

        # Build background tasks list for the state
        background_tasks: list[asyncio.Task[None]] = []
        if sentiment_signal_task:
            background_tasks.append(sentiment_signal_task)
        if watchlist_cleanup_task:
            background_tasks.append(watchlist_cleanup_task)
        if mkt_intel_task:
            background_tasks.append(mkt_intel_task)
        if arb_monitor_task:
            background_tasks.append(arb_monitor_task)
        if watchlist_intel_task:
            background_tasks.append(watchlist_intel_task)

        yield AgentState(
            redis=redis,
            db=db,
            settings=settings,
            agent_task=agent_task,
            twitter_enabled=twitter_client is not None,
            telegram_enabled=telegram_listener is not None,
            reddit_enabled=reddit_client is not None,
            sentiment_enabled=sentiment_processor is not None,
            db_enabled=db_initialized,
            _background_tasks=background_tasks,
        )

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
            logger.debug("Sentiment signal loop stopped")

        if watchlist_cleanup_task:
            watchlist_cleanup_task.cancel()
            try:
                await watchlist_cleanup_task
            except asyncio.CancelledError:
                pass
            logger.debug("Watchlist cleanup loop stopped")

        if arb_monitor_task:
            arb_monitor_task.cancel()
            try:
                await arb_monitor_task
            except asyncio.CancelledError:
                pass
            logger.debug("Arb monitor stopped")

        if mkt_intel_task:
            mkt_intel_task.cancel()
            try:
                await mkt_intel_task
            except asyncio.CancelledError:
                pass
            logger.debug("Mkt_intel signal loop stopped")

        if watchlist_intel_task:
            watchlist_intel_task.cancel()
            try:
                await watchlist_intel_task
            except asyncio.CancelledError:
                pass
            logger.debug("Watchlist intel loop stopped")

        for wl_client, wl_name in [
            (watchlist_intel_factset, "Watchlist FactSet"),
            (watchlist_intel_sec_edgar, "Watchlist SEC EDGAR"),
            (watchlist_intel_nasdaq, "Watchlist NASDAQ"),
        ]:
            if wl_client:
                try:
                    await wl_client.close()
                except Exception as e:
                    logger.error(f"Failed to close {wl_name} client", error=str(e))

        if mkt_intel_ws_manager:
            try:
                await mkt_intel_ws_manager.stop()
            except Exception as e:
                logger.error("Failed to stop WS manager", error=str(e))
            logger.debug("Market intelligence WebSocket clients stopped")

        for client, name in [
            (mkt_intel_kalshi_client, "Kalshi"),
            (mkt_intel_poly_client, "Polymarket"),
            (mkt_intel_data_client, "Polymarket Data"),
        ]:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Failed to close {name} client", error=str(e))

        if twitter_client:
            await twitter_client.stop()
            logger.debug("Twitter stream stopped")

        if telegram_listener:
            await telegram_listener.stop()
            logger.debug("Telegram listener stopped")

        if reddit_client:
            await reddit_client.stop()
            logger.debug("Reddit RSS client stopped")

        if sentiment_processor:
            await sentiment_processor.close()
            logger.debug("Sentiment processor closed")

        if price_service:
            await close_price_service()
            logger.debug("PriceService closed")

        if db_initialized:
            await close_database()
            logger.debug("PostgreSQL disconnected")

        if redis:
            await close_redis()
            logger.debug("Redis disconnected")

        logger.info("Agent shutdown complete")
