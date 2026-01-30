#!/usr/bin/env python3
"""Get your Telegram chat ID.

Usage:
    1. Send any message to your bot in Telegram first
    2. Run: uv run python scripts/get_telegram_chat_id.py
"""

import httpx
import os


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        # Try loading from .env
        try:
            from dotenv import load_dotenv

            load_dotenv()
            token = os.getenv("TELEGRAM_BOT_TOKEN")
        except ImportError:
            pass

    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        print("Set it in .env or export TELEGRAM_BOT_TOKEN=your_token")
        return

    url = f"https://api.telegram.org/bot{token}/getUpdates"

    print("Fetching updates from bot...")
    response = httpx.get(url)
    data = response.json()

    if not data.get("ok"):
        print(f"Error: {data.get('description')}")
        return

    results = data.get("result", [])

    if not results:
        print("\nNo messages found!")
        print("Make sure you:")
        print("  1. Open Telegram and find your bot")
        print("  2. Send ANY message to the bot (like /start or 'hello')")
        print("  3. Run this script again")
        return

    print("\nFound chats:")
    seen = set()
    for update in results:
        msg = update.get("message") or update.get("channel_post") or {}
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        name = chat.get("first_name") or chat.get("title") or chat.get("username") or "Unknown"

        if chat_id and chat_id not in seen:
            seen.add(chat_id)
            print(f"  Chat ID: {chat_id}")
            print(f"    Type: {chat_type}")
            print(f"    Name: {name}")
            print()

    if seen:
        # Get the most recent one
        latest = results[-1]
        msg = latest.get("message") or latest.get("channel_post") or {}
        chat_id = msg.get("chat", {}).get("id")
        print("Add this to your .env:")
        print(f"TELEGRAM_CHAT_ID={chat_id}")


if __name__ == "__main__":
    main()
