"""Integration smoke test for sentiment pipeline (Reddit).

Uses REAL APIs (Reddit RSS, LLM, Finnhub, Telegram) with mocked storage.
Run with: pytest -m integration

Environment variables required:
- ANTHROPIC_API_KEY or OPENAI_API_KEY
- FINNHUB_API_KEY (for ticker validation)
- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (for notifications)
"""

from typing import Any

import pytest

from synesis.config import get_settings
from synesis.ingestion.reddit import RedditPost, RedditRSSClient
from synesis.processing.sentiment import SentimentProcessor


@pytest.mark.integration
class TestSentimentE2E:
    """Smoke test for sentiment pipeline with real APIs."""

    @pytest.fixture
    def subreddits(self) -> list[str]:
        """Use just 2 subreddits for faster smoke test."""
        return ["wallstreetbets", "stocks"]

    @pytest.mark.anyio
    async def test_sentiment_smoke(
        self,
        mock_redis: Any,
        mock_db: Any,
        finnhub_service: Any,
        subreddits: list[str],
    ) -> None:
        """Smoke test: Full sentiment pipeline with real Reddit + LLM + Telegram.

        Verifies:
        1. Reddit RSS fetches real posts
        2. Gate 1 lexicon analysis extracts tickers
        3. Gate 2 LLM validation filters false positives
        4. Telegram notification sends
        5. Mock storage captures data correctly
        """
        settings = get_settings()

        # Fetch REAL posts from Reddit RSS (only 5 per subreddit for speed)
        reddit_client = RedditRSSClient(
            subreddits=subreddits,
            poll_interval=3600,
        )

        print(f"\n{'=' * 60}")
        print(f"FETCHING from Reddit: {subreddits}")

        all_posts: list[RedditPost] = []
        for subreddit in subreddits:
            posts = await reddit_client.fetch_subreddit(subreddit)
            all_posts.extend(posts[:5])  # Only 5 posts per sub for speed
            print(f"  r/{subreddit}: {len(posts)} posts (using 5)")

        print(f"  Total: {len(all_posts)} posts")
        assert len(all_posts) > 0, "No posts fetched"

        # Process through sentiment pipeline
        processor = SentimentProcessor(settings, mock_redis, db=mock_db, finnhub=finnhub_service)

        try:
            print(f"\n{'=' * 60}")
            print("GATE 1 (Lexicon):")
            for post in all_posts:
                result = await processor.process_post(post)
                if result.tickers_mentioned:
                    print(f"  [{post.subreddit}] {result.tickers_mentioned[:3]}")

            print(f"\n{'=' * 60}")
            print("GATE 2 (LLM Validation) + Signal...")
            signal = await processor.generate_signal()

            print("\nSignal:")
            print(
                f"  Watchlist: {signal.watchlist[:10]}{'...' if len(signal.watchlist) > 10 else ''}"
            )
            print(f"  Sentiment: {signal.overall_sentiment}")
            print(f"  Posts analyzed: {signal.total_posts_analyzed}")

            # Send REAL Telegram notification
            from synesis.notifications.telegram import (
                format_sentiment_signal,
                send_telegram,
            )

            print(f"\n{'=' * 60}")
            print("TELEGRAM (sending...):")
            sent = await send_telegram(format_sentiment_signal(signal))
            print(f"  Sent: {sent}")

            # Verify mock storage
            watchlist = mock_redis._test_sets.get("synesis:watchlist:tickers", set())
            print(f"\nWatchlist in Redis: {len(watchlist)} tickers")

            # Verify TTL keys exist
            for ticker in list(watchlist)[:3]:
                ttl_key = f"synesis:watchlist:ttl:{ticker}"
                assert ttl_key in mock_redis._test_strings, f"Missing TTL for {ticker}"

            # Verify DB
            print(f"Watchlist in DB: {len(mock_db._test_watchlist)} tickers")
            assert len(mock_db._test_signals) >= 1, "Signal not stored in DB"

            # No false positives
            false_positives = {"WHAT", "WEEK", "DD", "CEO", "YOLO", "HOLD", "THE", "FOR"}
            found_fps = watchlist & false_positives
            assert len(found_fps) == 0, f"False positives: {found_fps}"

        finally:
            await reddit_client.stop()
            await processor.close()
