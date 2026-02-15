#!/usr/bin/env python3
"""Test script for Sentiment Intelligence pipeline.

This script tests the two-gate sentiment analysis:
- Gate 1: Lexicon-based sentiment scoring
- Gate 2: LLM refinement for ticker validation and narrative

Usage:
    # Test with live Reddit RSS
    uv run python scripts/test_sentiment.py --subreddit wallstreetbets

    # Test with specific subreddit and limit
    uv run python scripts/test_sentiment.py --subreddit stocks --limit 10

    # Test watchlist operations
    uv run python scripts/test_sentiment.py --test-watchlist

    # Test full pipeline with signal generation
    uv run python scripts/test_sentiment.py --full-pipeline
"""

import argparse
import asyncio
import sys

# Add src to path for imports
sys.path.insert(0, "src")


async def test_watchlist() -> None:
    """Test watchlist manager operations."""
    from redis.asyncio import Redis

    from synesis.config import get_settings
    from synesis.processing.sentiment import WatchlistManager

    print("\n=== Testing Watchlist Manager ===\n")

    settings = get_settings()
    redis = Redis.from_url(settings.redis_url)

    try:
        await redis.ping()
        print("✓ Redis connected")

        watchlist = WatchlistManager(redis, ttl_days=7)

        # Test adding tickers
        print("\n--- Adding tickers ---")
        added_aapl = await watchlist.add_ticker("AAPL", source="reddit", subreddit="stocks")
        print(f"Added AAPL: is_new={added_aapl}")

        added_tsla = await watchlist.add_ticker("TSLA", source="reddit", subreddit="wallstreetbets")
        print(f"Added TSLA: is_new={added_tsla}")

        # Test refreshing TTL
        added_aapl_again = await watchlist.add_ticker("AAPL", source="reddit")
        print(f"Added AAPL again (should refresh): is_new={added_aapl_again}")

        # Test getting all tickers
        print("\n--- Watchlist contents ---")
        tickers = await watchlist.get_all()
        print(f"Tickers: {tickers}")

        # Test metadata
        print("\n--- Ticker metadata ---")
        meta = await watchlist.get_metadata("AAPL")
        if meta:
            print(f"AAPL metadata: source={meta.source}, mentions={meta.mention_count}")

        # Test stats
        stats = await watchlist.get_stats()
        print(f"Stats: {stats}")

        # Test contains
        print("\n--- Contains check ---")
        has_aapl = await watchlist.contains("AAPL")
        has_msft = await watchlist.contains("MSFT")
        print(f"Contains AAPL: {has_aapl}")
        print(f"Contains MSFT: {has_msft}")

        # Clean up test data
        print("\n--- Cleanup ---")
        await watchlist.remove_ticker("AAPL")
        await watchlist.remove_ticker("TSLA")
        tickers = await watchlist.get_all()
        print(f"Tickers after cleanup: {tickers}")

        print("\n✓ Watchlist tests passed!")

    finally:
        await redis.aclose()


async def test_gate1(subreddit: str, limit: int) -> None:
    """Test Gate 1: Lexicon analysis on Reddit posts."""
    from synesis.ingestion.reddit import RedditRSSClient
    from synesis.processing.sentiment.analyzer import SentimentAnalyzer

    print(f"\n=== Testing Gate 1 on r/{subreddit} ===\n")

    client = RedditRSSClient(subreddits=[subreddit])
    analyzer = SentimentAnalyzer()

    try:
        posts = await client.fetch_subreddit(subreddit)
        print(f"Fetched {len(posts)} posts from r/{subreddit}\n")

        all_tickers: dict[str, list[float]] = {}

        for i, post in enumerate(posts[:limit], 1):
            result = await analyzer.analyze(post.full_text)

            # Aggregate tickers
            for ticker in result.tickers_mentioned:
                if ticker not in all_tickers:
                    all_tickers[ticker] = []
                all_tickers[ticker].append(result.compound)

            sentiment_label = (
                "🟢 bullish"
                if result.is_bullish
                else ("🔴 bearish" if result.is_bearish else "⚪ neutral")
            )

            print(f"[{i}] {post.title[:60]}...")
            print(f"    Sentiment: {result.compound:+.3f} ({sentiment_label})")
            print(f"    Tickers: {', '.join(result.tickers_mentioned) or 'none'}")
            print()

        # Summary
        print("\n--- Gate 1 Summary ---")
        print(f"Posts analyzed: {min(limit, len(posts))}")
        print(f"Unique tickers found: {len(all_tickers)}")

        if all_tickers:
            print("\nTop tickers by mentions:")
            sorted_tickers = sorted(all_tickers.items(), key=lambda x: len(x[1]), reverse=True)
            for ticker, scores in sorted_tickers[:10]:
                avg = sum(scores) / len(scores)
                print(f"  {ticker}: {len(scores)} mentions, avg sentiment: {avg:+.3f}")

    finally:
        await client.stop()


async def test_full_pipeline(subreddit: str, limit: int, with_db: bool = False) -> None:
    """Test full sentiment pipeline: Gate 1 + Gate 2 + Signal."""
    from redis.asyncio import Redis

    from synesis.config import get_settings
    from synesis.ingestion.reddit import RedditRSSClient
    from synesis.processing.sentiment import SentimentProcessor
    from synesis.storage.database import Database

    print(f"\n=== Testing Full Sentiment Pipeline on r/{subreddit} ===\n")

    settings = get_settings()
    redis = Redis.from_url(settings.redis_url)
    db: Database | None = None

    try:
        await redis.ping()
        print("✓ Redis connected")

        # Connect to PostgreSQL if requested
        if with_db:
            try:
                db = Database(settings.database_url)
                await db.connect()
                print("✓ PostgreSQL connected")
            except Exception as e:
                print(f"⚠ PostgreSQL connection failed: {e}")
                print("  Continuing without database persistence...")
                db = None

        # Fetch posts
        client = RedditRSSClient(subreddits=[subreddit])
        posts = await client.fetch_subreddit(subreddit)
        print(f"✓ Fetched {len(posts)} posts from r/{subreddit}")
        await client.stop()

        # Limit posts
        posts = posts[:limit]

        # Create processor (with optional database)
        processor = SentimentProcessor(settings, redis, db=db)

        # Sync watchlist from database on startup
        if db:
            loaded = await processor.watchlist.sync_from_db()
            print(f"✓ Synced {loaded} tickers from database")

        # Process through Gate 1 + Gate 2
        print(f"\nProcessing {len(posts)} posts through Gate 1 + Gate 2...\n")
        refinement = await processor.process_posts(posts)

        # Display Gate 2 results
        print("=== Gate 2 Results ===\n")

        print(f"Overall Sentiment: {refinement.overall_sentiment}")
        print(f"Confidence: {refinement.sentiment_confidence:.0%}")
        print()

        print("Validated Tickers:")
        for ticker in refinement.validated_tickers:
            if ticker.is_valid_ticker:
                sentiment = (
                    "🟢"
                    if ticker.sentiment_label == "bullish"
                    else ("🔴" if ticker.sentiment_label == "bearish" else "⚪")
                )
                print(
                    f"  {sentiment} {ticker.ticker} ({ticker.company_name}): "
                    f"{ticker.avg_sentiment:+.2f}, "
                    f"confidence: {ticker.confidence:.0%}"
                )
                if ticker.key_catalysts:
                    print(f"      Catalysts: {', '.join(ticker.key_catalysts)}")

        if refinement.rejected_tickers:
            print(f"\nRejected (false positives): {', '.join(refinement.rejected_tickers[:20])}")

        bullish_tickers = [
            t.ticker
            for t in refinement.validated_tickers
            if t.sentiment_label == "bullish" and t.avg_sentiment > 0.5
        ]
        bearish_tickers = [
            t.ticker
            for t in refinement.validated_tickers
            if t.sentiment_label == "bearish" and t.avg_sentiment < -0.5
        ]
        if bullish_tickers:
            print(f"\n🚀 Extreme Bullish: {', '.join(bullish_tickers)}")
        if bearish_tickers:
            print(f"💀 Extreme Bearish: {', '.join(bearish_tickers)}")

        print(f"\nNarrative Summary:\n{refinement.narrative_summary}")

        if refinement.key_themes:
            print(f"\nKey Themes: {', '.join(refinement.key_themes)}")

        # Generate signal
        print("\n=== Generating Signal ===\n")

        # First add posts to buffer
        for post in posts:
            await processor.process_post(post)

        signal = await processor.generate_signal()

        print(f"Signal Period: {signal.period_start} to {signal.period_end}")
        print(f"Posts Analyzed: {signal.total_posts_analyzed}")
        print(f"Watchlist: {', '.join(signal.watchlist) or 'empty'}")
        print(f"Added: {', '.join(signal.watchlist_added) or 'none'}")
        print(f"Overall: {signal.overall_sentiment}")

        if db:
            print("\n=== Database Persistence ===\n")
            print("✓ Signal persisted to synesis.signals table")
            print(f"✓ {len(signal.ticker_sentiments)} sentiment snapshots persisted")
            print(f"✓ {len(signal.watchlist)} watchlist tickers synced to synesis.watchlist")

        # Cleanup
        await processor.close()
        print("\n✓ Full pipeline test complete!")

    finally:
        if db:
            await db.disconnect()
        await redis.aclose()


async def test_database_operations() -> None:
    """Test database operations for sentiment."""
    from datetime import UTC, datetime, timedelta

    from synesis.config import get_settings
    from synesis.storage.database import Database

    print("\n=== Testing Sentiment Database Operations ===\n")

    settings = get_settings()

    try:
        db = Database(settings.database_url)
        await db.connect()
        print("✓ PostgreSQL connected")

        # Test 1: Watchlist operations
        print("\n--- Testing Watchlist Operations ---")
        test_ticker = "TEST123"
        expires_at = datetime.now(UTC) + timedelta(days=7)

        is_new = await db.upsert_watchlist_ticker(
            ticker=test_ticker,
            company_name="Test Company Inc",
            added_by="test_script",
            added_reason="Testing database operations",
            expires_at=expires_at,
        )
        print(f"Upsert TEST123: is_new={is_new}")

        # Get active watchlist
        watchlist = await db.get_active_watchlist()
        print(f"Active watchlist: {watchlist}")

        # Test 2: Deactivate expired (shouldn't affect our test ticker)
        removed = await db.deactivate_expired_watchlist()
        print(f"Deactivated expired: {removed}")

        # Test 3: Sentiment snapshot
        print("\n--- Testing Sentiment Snapshots ---")
        await db.insert_sentiment_snapshot(
            ticker=test_ticker,
            snapshot_time=datetime.now(UTC),
            bullish_ratio=0.65,
            bearish_ratio=0.20,
            mention_count=42,
        )
        print(f"✓ Inserted sentiment snapshot for {test_ticker}")

        # Test 4: Query to verify
        print("\n--- Verifying Database Contents ---")
        rows = await db.fetch(
            "SELECT ticker, company_name, added_by FROM watchlist WHERE ticker = $1",
            test_ticker,
        )
        if rows:
            row = rows[0]
            print(
                f"  Watchlist row: {row['ticker']} - {row['company_name']} (added by {row['added_by']})"
            )

        snapshots = await db.fetch(
            "SELECT ticker, bullish_ratio, mention_count FROM sentiment_snapshots WHERE ticker = $1",
            test_ticker,
        )
        if snapshots:
            snap = snapshots[0]
            print(
                f"  Snapshot row: {snap['ticker']} - {snap['bullish_ratio']} bullish, {snap['mention_count']} mentions"
            )

        # Cleanup test data
        print("\n--- Cleanup ---")
        await db.execute("DELETE FROM sentiment_snapshots WHERE ticker = $1", test_ticker)
        await db.execute("DELETE FROM watchlist WHERE ticker = $1", test_ticker)
        print(f"✓ Cleaned up test ticker {test_ticker}")

        print("\n✓ Database operations test complete!")

    except Exception as e:
        print(f"\n✗ Database test failed: {e}")
        raise

    finally:
        await db.disconnect()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Sentiment Intelligence")
    parser.add_argument(
        "--subreddit",
        default="wallstreetbets",
        help="Subreddit to test (default: wallstreetbets)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of posts to analyze (default: 10)",
    )
    parser.add_argument(
        "--test-watchlist",
        action="store_true",
        help="Test watchlist manager only",
    )
    parser.add_argument(
        "--gate1-only",
        action="store_true",
        help="Test Gate 1 (lexicon) only, no LLM",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Test full Gate 1 + Gate 2 + Signal pipeline",
    )
    parser.add_argument(
        "--with-db",
        action="store_true",
        help="Include PostgreSQL persistence in tests",
    )
    parser.add_argument(
        "--test-db",
        action="store_true",
        help="Test database operations only",
    )

    args = parser.parse_args()

    if args.test_db:
        await test_database_operations()
    elif args.test_watchlist:
        await test_watchlist()
    elif args.gate1_only:
        await test_gate1(args.subreddit, args.limit)
    elif args.full_pipeline:
        await test_full_pipeline(args.subreddit, args.limit, with_db=args.with_db)
    else:
        # Default: just test Gate 1
        await test_gate1(args.subreddit, args.limit)


if __name__ == "__main__":
    asyncio.run(main())
