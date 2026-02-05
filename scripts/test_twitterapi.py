#!/usr/bin/env python3
"""Test script to verify Twitter client is working."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from synesis.config import get_settings
from synesis.core.logging import setup_logging
from synesis.ingestion.twitterapi import Tweet, TwitterClient

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "test"
OUTPUT_FILE = OUTPUT_DIR / "twitter_tweets.jsonl"


async def on_tweet(tweet: Tweet) -> None:
    """Handle incoming tweets."""
    # Print to console
    print(f"\n{'=' * 60}")
    print(f"User: @{tweet.username}")
    print(f"Time: {tweet.timestamp}")
    print(f"Tweet ID: {tweet.tweet_id}")
    print(f"Text: {tweet.text[:500]}..." if len(tweet.text) > 500 else f"Text: {tweet.text}")
    print(f"{'=' * 60}\n")

    # Write to JSONL
    record = {
        "timestamp": tweet.timestamp.isoformat(),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "tweet_id": tweet.tweet_id,
        "user_id": tweet.user_id,
        "username": tweet.username,
        "text": tweet.text,
        "raw": tweet.raw,
    }

    with open(OUTPUT_FILE, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")

    print(f"Saved to {OUTPUT_FILE}")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    if not settings.twitterapi_api_key:
        print("Error: TWITTERAPI_API_KEY must be set in .env")
        return

    if not settings.twitter_accounts:
        print("Error: TWITTER_ACCOUNTS must be set in .env")
        return

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting Twitter client...")
    print(f"Base URL: {settings.twitter_api_base_url}")
    print(f"Accounts: {settings.twitter_accounts}")
    print(f"Output: {OUTPUT_FILE}")

    client = TwitterClient(
        api_key=settings.twitterapi_api_key.get_secret_value(),
        base_url=settings.twitter_api_base_url,
        accounts=settings.twitter_accounts,
        poll_interval=60.0,
    )

    client.on_tweet(on_tweet)

    print("\nFetching initial tweets...")

    # Do one immediate poll to get initial state
    async for tweet in client.poll_accounts():
        await on_tweet(tweet)

    print("\nPolling for new tweets... (Ctrl+C to stop)")
    print(f"Poll interval: {client.poll_interval}s\n")

    try:
        await client.start()
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
