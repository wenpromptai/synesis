#!/usr/bin/env python3
"""Test script to verify Twitter Stream client is working.

Prerequisites:
1. Subscribe to twitterapi.io Stream plan (Starter $29/mo for 6 accounts)
2. Configure filter rules at https://twitterapi.io/tweet-filter-rules
3. Set TWITTERAPI_API_KEY in .env
"""

import asyncio
from pathlib import Path

import orjson

from synesis.config import get_settings
from synesis.core.logging import setup_logging
from synesis.ingestion.twitterapi import Tweet, TwitterStreamClient

OUTPUT_DIR = Path(__file__).parent.parent / "shared" / "output"
OUTPUT_FILE = OUTPUT_DIR / "twitter_stream.jsonl"


async def on_tweet(tweet: Tweet) -> None:
    """Handle incoming tweets from stream."""
    print(f"\n{'=' * 60}")
    print(f"User: @{tweet.username}")
    print(f"Time: {tweet.timestamp}")
    print(f"Tweet ID: {tweet.tweet_id}")
    print(f"Text: {tweet.text[:500]}..." if len(tweet.text) > 500 else f"Text: {tweet.text}")
    print(f"{'=' * 60}\n")

    # Write to JSONL
    record = {
        "timestamp": tweet.timestamp.isoformat(),
        "username": tweet.username,
        "text": tweet.text,
        "views": tweet.raw.get("viewCount"),
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

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting Twitter Stream client...")
    print("Note: Filter rules must be configured at twitterapi.io/tweet-filter-rules")
    print(f"Output: {OUTPUT_FILE}")

    client = TwitterStreamClient(
        api_key=settings.twitterapi_api_key.get_secret_value(),
    )

    client.on_tweet(on_tweet)

    print("\nConnecting to WebSocket stream... (Ctrl+C to stop)\n")

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
