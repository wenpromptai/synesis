#!/usr/bin/env python3
"""Test script for sentiment end-to-end: Reddit RSS -> Sentiment -> Telegram.

This script:
1. Fetches posts from Reddit RSS feeds
2. Processes them through the sentiment pipeline (Gate 1 + Gate 2)
3. Generates a sentiment signal
4. Sends the formatted signal to Telegram

Usage:
    uv run python scripts/test_sentiment_telegram.py

Requirements:
    - Redis running (docker compose up -d)
    - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    - LLM provider configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)
"""

import asyncio

from redis.asyncio import Redis

from synesis.config import get_settings
from synesis.core.logging import get_logger, setup_logging
from synesis.ingestion.reddit import RedditRSSClient
from synesis.notifications.telegram import format_sentiment_signal, send_telegram
from synesis.processing.sentiment import SentimentProcessor

logger = get_logger(__name__)


async def main() -> None:
    """Run sentiment test: Reddit -> Sentiment -> Telegram."""
    settings = get_settings()
    setup_logging(settings)

    logger.info("Starting sentiment test")

    # Check Telegram is configured
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        return

    # Use default subreddits if not configured
    subreddits = settings.reddit_subreddits or ["wallstreetbets", "stocks", "options"]
    logger.info("Target subreddits", subreddits=subreddits)

    redis: Redis | None = None
    reddit_client: RedditRSSClient | None = None

    try:
        # 1. Connect to Redis
        logger.info("Connecting to Redis")
        redis = Redis.from_url(settings.redis_url)
        await redis.ping()  # type: ignore[misc]
        logger.info("Redis connected")

        # 2. Initialize sentiment processor
        sentiment_processor = SentimentProcessor(settings, redis)
        logger.info("Sentiment processor initialized")

        # 3. Fetch Reddit posts via RSS
        reddit_client = RedditRSSClient(subreddits=subreddits)
        logger.info("Fetching Reddit posts...")

        all_posts = []
        for subreddit in subreddits:
            posts = await reddit_client.fetch_subreddit(subreddit)
            all_posts.extend(posts)
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            await asyncio.sleep(1.0)  # Be polite to Reddit

        logger.info(f"Total posts fetched: {len(all_posts)}")

        if not all_posts:
            logger.warning("No posts fetched, cannot generate signal")
            return

        # 4. Process posts through sentiment pipeline (Gate 1 + Gate 2)
        logger.info("Processing posts through sentiment pipeline...")

        # Process each post through Gate 1 (buffers them)
        for post in all_posts:
            await sentiment_processor.process_post(post)

        # 5. Generate signal (runs Gate 2 on buffered posts)
        logger.info("Generating sentiment signal...")
        signal = await sentiment_processor.generate_signal()

        logger.info(
            "Signal generated",
            overall_sentiment=signal.overall_sentiment,
            tickers=len(signal.ticker_sentiments),
            watchlist=len(signal.watchlist),
            posts_analyzed=signal.total_posts_analyzed,
        )

        # 6. Format and send to Telegram
        logger.info("Formatting and sending to Telegram...")
        message = format_sentiment_signal(signal)

        # Print to console first
        print("\n" + "=" * 60)
        print("TELEGRAM MESSAGE PREVIEW")
        print("=" * 60)
        print(message)
        print("=" * 60)
        print(f"Message length: {len(message)} characters")
        print("=" * 60 + "\n")

        # Send to Telegram
        success = await send_telegram(message)

        if success:
            logger.info("Telegram message sent successfully!")
        else:
            logger.error("Failed to send Telegram message")

        # Cleanup
        await sentiment_processor.close()

    except Exception as e:
        logger.exception("Sentiment test failed", error=str(e))
        raise

    finally:
        if reddit_client:
            await reddit_client.stop()
        if redis:
            await redis.close()
            logger.info("Redis disconnected")


if __name__ == "__main__":
    asyncio.run(main())
