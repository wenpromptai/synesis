#!/usr/bin/env python3
"""Test script to verify Telegram listener is working."""

import asyncio
from datetime import datetime
from pathlib import Path

import httpx
import orjson

from synesis.config import Settings, get_settings
from synesis.core.logging import setup_logging
from synesis.ingestion.telegram import TelegramListener, TelegramMessage

OUTPUT_DIR = Path(__file__).parent.parent / "shared" / "output"
OUTPUT_FILE = OUTPUT_DIR / "telegram_messages.jsonl"
SESSION_DIR = Path(__file__).parent.parent / "shared" / "sessions"

# Module-level state for webhook
http_client: httpx.AsyncClient | None = None
settings: Settings | None = None


async def on_message(msg: TelegramMessage) -> None:
    """Handle incoming messages."""
    # Print to console
    print(f"\n{'='*60}")
    print(f"Channel: {msg.channel_name}")
    print(f"Time: {msg.timestamp}")
    print(f"Message ID: {msg.message_id}")
    print(f"Text: {msg.text[:500]}..." if len(msg.text) > 500 else f"Text: {msg.text}")
    print(f"{'='*60}\n")

    # Write to JSONL - store raw with received_at timestamp
    record = {
        "received_at": datetime.utcnow().isoformat(),
        "text": msg.text,
        **msg.raw,
    }

    with open(OUTPUT_FILE, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")

    print(f"Saved to {OUTPUT_FILE}")

    # Send to n8n webhook if configured
    if http_client and settings and settings.n8n_webhook_url:
        try:
            await http_client.post(settings.n8n_webhook_url, json=record)
            print(f"Sent to webhook: {settings.n8n_webhook_url}")
        except Exception as e:
            print(f"Webhook error: {e}")


async def main() -> None:
    global settings, http_client

    settings = get_settings()
    setup_logging(settings)

    # Initialize HTTP client for webhook
    http_client = httpx.AsyncClient(timeout=10.0)

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        return

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    session_path = str(SESSION_DIR / settings.telegram_session_name)

    print(f"Starting Telegram listener...")
    print(f"API ID: {settings.telegram_api_id}")
    print(f"Channels: {settings.telegram_channels or 'All (no filter)'}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Session: {session_path}")
    print(f"Webhook: {settings.n8n_webhook_url or 'Not configured'}")

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
        await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
