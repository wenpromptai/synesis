"""Daily Twitter agent digest job.

Fetches tweets from configured accounts, runs LLM analysis,
auto-adds mentioned tickers to watchlist, and posts digest to Discord.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.ingestion.twitterapi import Tweet, TwitterClient
from synesis.notifications.discord import format_twitter_agent_embeds, send_discord
from synesis.processing.twitter.analyzer import TwitterAgentAnalyzer

if TYPE_CHECKING:
    from synesis.processing.common.watchlist import WatchlistManager
    from synesis.providers.base import TickerProvider
    from synesis.providers.yfinance.client import YFinanceClient

logger = get_logger(__name__)

# Fetch up to 20 tweets per account (one page), filter by 24h window
TWEET_AGE_HOURS = 24
MAX_CONCURRENT_FETCHES = 5


async def twitter_agent_job(
    watchlist: WatchlistManager | None = None,
    yfinance: YFinanceClient | None = None,
    ticker_provider: TickerProvider | None = None,
) -> None:
    """Daily Twitter agent digest job."""
    settings = get_settings()

    if not settings.twitterapi_api_key or not settings.twitter_accounts:
        logger.warning("Twitter agent job skipped: no API key or accounts configured")
        return

    client = TwitterClient(
        api_key=settings.twitterapi_api_key.get_secret_value(),
        base_url=settings.twitter_api_base_url,
        accounts=settings.twitter_accounts,
    )

    # Fetch tweets from all accounts concurrently
    cutoff = datetime.now(UTC) - timedelta(hours=TWEET_AGE_HOURS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    fetch_errors = 0

    async def fetch_account(username: str) -> list[Tweet]:
        nonlocal fetch_errors
        async with semaphore:
            try:
                tweets, _ = await client.get_user_tweets(username)
                recent = [t for t in tweets if t.timestamp >= cutoff]
                logger.debug(
                    "Fetched tweets for account",
                    username=username,
                    total=len(tweets),
                    recent=len(recent),
                )
                return recent
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError):
                fetch_errors += 1
                logger.exception("Failed to fetch tweets", username=username)
                return []

    results = await asyncio.gather(*[fetch_account(acc) for acc in settings.twitter_accounts])
    all_tweets: list[Tweet] = [t for batch in results for t in batch]

    accounts_total = len(settings.twitter_accounts)
    if fetch_errors == accounts_total:
        logger.error(
            "All account fetches failed, skipping digest",
            accounts_total=accounts_total,
        )
        return

    if not all_tweets:
        logger.info("No tweets found in last 24hrs, skipping digest")
        return

    logger.info(
        "Twitter agent job: tweets collected",
        total_tweets=len(all_tweets),
        accounts_with_tweets=sum(1 for batch in results if batch),
    )

    # Run LLM analysis
    analyzer = TwitterAgentAnalyzer()
    analysis = await analyzer.analyze_tweets(
        all_tweets,
        yfinance=yfinance,
        ticker_provider=ticker_provider,
    )

    if not analysis:
        logger.warning("Twitter agent analysis returned no results")
        return

    # Auto-add mentioned tickers to watchlist with meaningful reason
    if watchlist and analysis.themes:
        for theme in analysis.themes:
            for tm in theme.tickers:
                try:
                    sources = ", ".join(f"@{s.lstrip('@')}" for s in theme.sources)
                    reason = f"{theme.title}: {tm.reasoning}"[:200]
                    await watchlist.add_ticker(
                        tm.ticker,
                        source=sources,
                        added_reason=reason,
                    )
                except Exception:
                    logger.exception(
                        "Failed to add ticker to watchlist",
                        ticker=tm.ticker,
                    )

    # Send to Discord — one message per theme (mirrors Stage 2 appearance)
    if settings.discord_twitter_webhook_url:
        messages = format_twitter_agent_embeds(analysis)
        sent_ok = 0
        for i, embeds in enumerate(messages):
            ok = await send_discord(
                embeds, webhook_url_override=settings.discord_twitter_webhook_url
            )
            if ok:
                sent_ok += 1
            else:
                logger.warning("Discord message send failed", message_index=i)
            # Small delay between messages to respect Discord rate limits
            if i < len(messages) - 1:
                await asyncio.sleep(0.5)
        logger.info(
            "Twitter agent digest sent to Discord",
            messages_sent=sent_ok,
            messages_total=len(messages),
            themes=len(analysis.themes),
        )
    else:
        logger.warning("No DISCORD_TWITTER_WEBHOOK_URL configured, skipping Discord notification")
