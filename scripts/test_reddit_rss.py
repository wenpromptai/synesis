#!/usr/bin/env python3
"""Test script for Reddit RSS client.

Usage:
    uv run python scripts/test_reddit_rss.py
    uv run python scripts/test_reddit_rss.py --subreddit stocks
    uv run python scripts/test_reddit_rss.py --subreddit wallstreetbets --verbose
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synesis.ingestion.reddit import RedditPost, RedditRSSClient
from synesis.processing.sentiment.analyzer import SentimentAnalyzer


async def test_single_subreddit(subreddit: str, verbose: bool = False) -> list[RedditPost]:
    """Test fetching from a single subreddit."""
    print(f"\n{'=' * 60}")
    print(f"Testing r/{subreddit}")
    print("=" * 60)

    client = RedditRSSClient(subreddits=[subreddit])

    try:
        posts = await client.fetch_subreddit(subreddit)
        print(f"Fetched {len(posts)} posts from r/{subreddit}")

        if not posts:
            print("  No posts found!")
            return []

        for i, post in enumerate(posts[:10], 1):  # Show first 10
            print(f"\n  [{i}] {post.title[:80]}...")
            print(f"      ID: {post.post_id}")
            print(f"      Author: u/{post.author}")
            print(f"      Time: {post.timestamp}")
            print(f"      URL: {post.url}")

            if verbose and post.content:
                content_preview = post.content[:200].replace("\n", " ")
                print(f"      Content: {content_preview}...")

        return posts

    finally:
        await client.stop()


async def test_callback_flow(subreddits: list[str]) -> None:
    """Test the callback flow (simulating how the agent uses it)."""
    print(f"\n{'=' * 60}")
    print("Testing callback flow")
    print("=" * 60)

    received_posts: list[RedditPost] = []

    async def on_new_post(post: RedditPost) -> None:
        """Callback for new posts."""
        received_posts.append(post)
        print(f"  [CALLBACK] New post from r/{post.subreddit}: {post.title[:50]}...")

    client = RedditRSSClient(
        subreddits=subreddits,
        poll_interval=5,  # Short interval for testing
    )
    client.on_post(on_new_post)

    try:
        # Do initial poll (populates seen_ids, no callbacks)
        posts = await client.poll_all_subreddits()
        print(f"Initial poll: {len(posts)} posts seen")

        # Clear seen_ids to simulate finding new posts
        client._seen_ids.clear()

        # Second poll should trigger callbacks
        posts = await client.poll_all_subreddits()
        print(f"Second poll: {len(posts)} new posts")
        print(f"Callbacks received: {len(received_posts)}")

    finally:
        await client.stop()


async def test_sentiment_analysis(subreddit: str = "wallstreetbets") -> None:
    """Test sentiment analysis on Reddit posts."""
    print(f"\n{'=' * 60}")
    print(f"SENTIMENT ANALYSIS: r/{subreddit}")
    print("=" * 60)

    client = RedditRSSClient(subreddits=[subreddit])
    analyzer = SentimentAnalyzer()

    try:
        posts = await client.fetch_subreddit(subreddit)
        print(f"Fetched {len(posts)} posts\n")

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        all_tickers: dict[str, list[float]] = {}

        for i, post in enumerate(posts[:15], 1):
            result = await analyzer.analyze(post.full_text)

            # Categorize
            if result.is_bullish:
                bullish_count += 1
                sentiment_label = "BULLISH"
            elif result.is_bearish:
                bearish_count += 1
                sentiment_label = "BEARISH"
            else:
                neutral_count += 1
                sentiment_label = "NEUTRAL"

            # Collect ticker sentiments
            for ticker in result.tickers_mentioned:
                if ticker not in all_tickers:
                    all_tickers[ticker] = []
                all_tickers[ticker].append(result.compound)

            print(f"[{i}] {post.title[:60]}...")
            print(
                f"    Sentiment: {sentiment_label} ({result.compound:+.3f}) | Strength: {result.strength}"
            )
            print(f"    Tickers: {', '.join(result.tickers_mentioned) or 'none'}")
            if result.tickers_mentioned:
                print(f"    Confidence: {result.confidence:.2f}")
            print()

        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total = bullish_count + bearish_count + neutral_count
        print(f"Bullish: {bullish_count}/{total} ({bullish_count / total * 100:.0f}%)")
        print(f"Bearish: {bearish_count}/{total} ({bearish_count / total * 100:.0f}%)")
        print(f"Neutral: {neutral_count}/{total} ({neutral_count / total * 100:.0f}%)")

        if all_tickers:
            print(f"\nTickers mentioned ({len(all_tickers)} unique):")
            # Sort by mention count
            sorted_tickers = sorted(all_tickers.items(), key=lambda x: len(x[1]), reverse=True)
            for ticker, sentiments in sorted_tickers[:10]:
                avg_sentiment = sum(sentiments) / len(sentiments)
                direction = (
                    "bullish"
                    if avg_sentiment > 0.05
                    else "bearish"
                    if avg_sentiment < -0.05
                    else "neutral"
                )
                print(
                    f"  ${ticker}: {len(sentiments)} mentions, avg sentiment {avg_sentiment:+.3f} ({direction})"
                )

    finally:
        await client.stop()


async def test_wallstreetbets_heavy(verbose: bool = False) -> None:
    """Heavy testing on wallstreetbets subreddit."""
    print(f"\n{'=' * 60}")
    print("HEAVY TEST: r/wallstreetbets")
    print("=" * 60)

    client = RedditRSSClient(subreddits=["wallstreetbets"])

    try:
        # Test 1: Basic fetch
        print("\n[Test 1] Basic fetch...")
        posts = await client.fetch_subreddit("wallstreetbets")
        print(f"  Fetched {len(posts)} posts")

        if not posts:
            print("  WARNING: No posts returned! Check if Reddit is blocking.")
            return

        # Test 2: Verify post structure
        print("\n[Test 2] Verifying post structure...")
        for i, post in enumerate(posts[:5], 1):
            errors = []
            if not post.post_id:
                errors.append("missing post_id")
            if not post.title:
                errors.append("missing title")
            if not post.url:
                errors.append("missing url")
            if not post.timestamp:
                errors.append("missing timestamp")

            status = "OK" if not errors else f"ERRORS: {', '.join(errors)}"
            print(f"  Post {i}: {status}")

            if verbose:
                print(f"    Title: {post.title[:60]}...")
                print(f"    ID: {post.post_id}")
                print(f"    Author: {post.author}")
                print(f"    Has content: {bool(post.content)}")

        # Test 3: Check for typical WSB content patterns
        print("\n[Test 3] Checking for typical WSB patterns...")
        tickers_found = 0
        dd_posts = 0
        yolo_posts = 0

        for post in posts:
            full_text = post.full_text.upper()
            # Check for stock tickers ($XXX or just XXX patterns)
            if any(
                ticker in full_text
                for ticker in ["$GME", "$AMC", "$SPY", "$TSLA", "$NVDA", "$AAPL"]
            ):
                tickers_found += 1
            # Check for DD (Due Diligence) posts
            if "DD" in full_text or "DUE DILIGENCE" in full_text:
                dd_posts += 1
            # Check for YOLO posts
            if "YOLO" in full_text:
                yolo_posts += 1

        print(f"  Posts mentioning major tickers: {tickers_found}/{len(posts)}")
        print(f"  DD posts: {dd_posts}")
        print(f"  YOLO posts: {yolo_posts}")

        # Test 4: Test deduplication
        print("\n[Test 4] Testing deduplication...")
        first_fetch = await client.poll_all_subreddits()
        print(f"  First poll: {len(first_fetch)} new posts")

        second_fetch = await client.poll_all_subreddits()
        print(f"  Second poll: {len(second_fetch)} new posts (should be 0 or few)")

        # Test 5: full_text property
        print("\n[Test 5] Testing full_text property...")
        for post in posts[:3]:
            full_text = post.full_text
            print(f"  Post: {post.title[:40]}...")
            print(f"    full_text length: {len(full_text)} chars")
            print(f"    Has content beyond title: {len(full_text) > len(post.title)}")

        print("\n" + "=" * 60)
        print("HEAVY TEST COMPLETE")
        print("=" * 60)

    finally:
        await client.stop()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Reddit RSS client")
    parser.add_argument(
        "--subreddit",
        "-s",
        default="wallstreetbets",
        help="Subreddit to test (default: wallstreetbets)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including post content",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Run heavy tests on wallstreetbets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all default subreddits",
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Run sentiment analysis on posts",
    )

    args = parser.parse_args()

    print("Reddit RSS Client Test Suite")
    print("============================")

    if args.sentiment:
        await test_sentiment_analysis(args.subreddit)
    elif args.heavy:
        await test_wallstreetbets_heavy(args.verbose)
    elif args.all:
        subreddits = ["wallstreetbets", "stocks", "options"]
        for sub in subreddits:
            await test_single_subreddit(sub, args.verbose)
        await test_callback_flow(subreddits)
    else:
        await test_single_subreddit(args.subreddit, args.verbose)
        await test_callback_flow([args.subreddit])

    print("\n\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
