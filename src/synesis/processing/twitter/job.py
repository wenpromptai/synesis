"""Daily Twitter/X data collection job.

Fetches tweets from configured accounts and persists to raw_tweets table.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.ingestion.twitterapi import Tweet, TwitterClient

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)

MAX_CONCURRENT_FETCHES = 5


async def twitter_agent_job(
    db: Database | None = None,
) -> None:
    """Daily Twitter data collection job.

    Fetches tweets from all configured accounts and stores them
    in the raw_tweets table. Deduplication is handled by the DB
    composite primary key (account_username, tweet_id).
    """
    settings = get_settings()

    if not settings.twitterapi_api_key or not settings.twitter_accounts:
        logger.warning("Twitter job skipped: no API key or accounts configured")
        return

    if not db:
        logger.warning("Twitter job skipped: no database configured")
        return

    client = TwitterClient(
        api_key=settings.twitterapi_api_key.get_secret_value(),
        base_url=settings.twitter_api_base_url,
        accounts=settings.twitter_accounts,
    )

    # Fetch tweets from all accounts concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    fetch_errors = 0

    async def fetch_account(username: str) -> list[Tweet]:
        nonlocal fetch_errors
        async with semaphore:
            try:
                tweets, _ = await client.get_user_tweets(username)
                logger.debug(
                    "Fetched tweets for account",
                    username=username,
                    total=len(tweets),
                )
                return tweets
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError):
                fetch_errors += 1
                logger.exception("Failed to fetch tweets", username=username)
                return []

    results = await asyncio.gather(*[fetch_account(acc) for acc in settings.twitter_accounts])
    all_tweets: list[Tweet] = [t for batch in results for t in batch]

    accounts_total = len(settings.twitter_accounts)
    if fetch_errors == accounts_total:
        logger.error(
            "All account fetches failed, skipping storage",
            accounts_total=accounts_total,
        )
        return

    if not all_tweets:
        logger.info("No tweets found, skipping storage")
        return

    # Store all tweets to raw_tweets table (DB deduplicates via composite PK)
    raw_tweet_rows = [
        {
            "tweet_id": t.tweet_id,
            "account_username": t.username,
            "tweet_text": t.text,
            "tweet_timestamp": t.timestamp,
            "tweet_url": t.raw.get("url"),
        }
        for t in all_tweets
    ]

    try:
        inserted = await db.store_raw_tweets(raw_tweet_rows)
        logger.info(
            "Twitter data collection complete",
            total_fetched=len(all_tweets),
            new_stored=inserted,
            accounts_with_tweets=sum(1 for batch in results if batch),
        )
    except Exception:
        logger.exception("Failed to store raw tweets")
        raise
