#!/usr/bin/env python3
"""Test script to verify Telegram listener is working."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from synesis.config import get_settings
from synesis.core.logging import setup_logging
from synesis.ingestion.telegram import TelegramListener, TelegramMessage

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "test"
OUTPUT_FILE = OUTPUT_DIR / "telegram_messages.jsonl"
SESSION_DIR = Path(__file__).parent.parent / "shared" / "sessions"


async def on_message(msg: TelegramMessage) -> None:
    """Handle incoming messages."""
    # Print to console
    print(f"\n{'=' * 60}")
    print(f"Channel: {msg.channel_name}")
    print(f"Time: {msg.timestamp}")
    print(f"Message ID: {msg.message_id}")
    print(f"Text: {msg.text[:500]}..." if len(msg.text) > 500 else f"Text: {msg.text}")
    print(f"{'=' * 60}\n")

    # Write to JSONL - store raw with received_at timestamp
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "text": msg.text,
        **msg.raw,
    }

    with open(OUTPUT_FILE, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")

    print(f"Saved to {OUTPUT_FILE}")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        return

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    session_path = str(SESSION_DIR / settings.telegram_session_name)

    print("Starting Telegram listener...")
    print(f"API ID: {settings.telegram_api_id}")
    print(f"Channels: {settings.telegram_channels or 'All (no filter)'}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Session: {session_path}")

    listener = TelegramListener(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash.get_secret_value(),
        session_name=session_path,
        channels=settings.telegram_channels,
    )

    listener.on_message(on_message)

    await listener.start()

    # First time: you'll be prompted to enter your phone number and code
    print("\nListening for messages... (Ctrl+C to stop)")
    print("If this is your first run, you'll be prompted to authenticate.\n")

    try:
        await listener.run_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await listener.stop()


if __name__ == "__main__":
    asyncio.run(main())
