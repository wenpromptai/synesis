"""Test ticker extraction with updated prompts against live DB data.

Usage:
    uv run python scripts/test_ticker_extraction.py

Runs SocialSentimentAnalyst and NewsAnalyst against last 24h data
and reports extracted tickers without running the full pipeline.
"""

import asyncio
from datetime import UTC, datetime

from synesis.config import get_settings
from synesis.processing.intelligence.specialists.news.agent import (
    NewsDeps,
    _format_messages,
    _gather_messages,
    analyze_news,
)
from synesis.processing.intelligence.specialists.social_sentiment.agent import (
    SocialSentimentDeps,
    _format_tweets_by_account,
    _gather_tweets,
    analyze_social_sentiment,
)
from synesis.storage.database import Database


async def main() -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()

    today = datetime.now(UTC).date()

    # ── Preview data ──────────────────────────────────────────────
    tweets = await _gather_tweets(db)
    messages = await _gather_messages(db)

    print(f"\n{'=' * 60}")
    print("DATA SUMMARY (last 24h)")
    print(f"{'=' * 60}")
    print(f"Tweets:   {len(tweets)}")
    print(f"Messages: {len(messages)} (impact >= 20)")

    if tweets:
        formatted_tweets = _format_tweets_by_account(tweets)
        print("\n--- Formatted tweets preview (first 2000 chars) ---")
        print(formatted_tweets[:2000])
        print("...")

    if messages:
        formatted_msgs = _format_messages(messages)
        print("\n--- Formatted messages preview (first 2000 chars) ---")
        print(formatted_msgs[:2000])
        print("...")

    # ── Run SocialSentimentAnalyst ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RUNNING SocialSentimentAnalyst...")
    print(f"{'=' * 60}")

    social_deps = SocialSentimentDeps(db=db, current_date=today)
    social_result = await analyze_social_sentiment(social_deps)

    social_tickers = [m.ticker for m in social_result.ticker_mentions]
    print(f"\nTickers extracted: {len(social_tickers)}")
    print(f"Tickers: {sorted(social_tickers)}")
    print(f"Macro themes: {len(social_result.macro_themes)}")
    for theme in social_result.macro_themes:
        print(f"  - {theme}")
    print(f"Summary: {social_result.summary}")

    # ── Run NewsAnalyst ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RUNNING NewsAnalyst...")
    print(f"{'=' * 60}")

    news_deps = NewsDeps(db=db, current_date=today)
    news_result = await analyze_news(news_deps)

    news_tickers: set[str] = set()
    for cluster in news_result.story_clusters:
        for mention in cluster.tickers:
            news_tickers.add(mention.ticker)

    print(f"\nTickers extracted: {len(news_tickers)}")
    print(f"Tickers: {sorted(news_tickers)}")
    print(f"Story clusters: {len(news_result.story_clusters)}")
    for cluster in news_result.story_clusters:
        cluster_tickers = [t.ticker for t in cluster.tickers]
        print(f"  - [{cluster.event_type}] {cluster.headline} → {cluster_tickers}")
    print(f"Macro themes: {len(news_result.macro_themes)}")
    print(f"Summary: {news_result.summary}")

    # ── Combined ──────────────────────────────────────────────────
    all_tickers = sorted(set(social_tickers) | news_tickers)
    print(f"\n{'=' * 60}")
    print(f"COMBINED: {len(all_tickers)} unique tickers")
    print(f"{'=' * 60}")
    print(f"From social only: {sorted(set(social_tickers) - news_tickers)}")
    print(f"From news only:   {sorted(news_tickers - set(social_tickers))}")
    print(f"From both:        {sorted(set(social_tickers) & news_tickers)}")
    print(f"Total:            {all_tickers}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
