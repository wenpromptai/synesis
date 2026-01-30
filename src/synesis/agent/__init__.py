"""Agent module for news processing using PydanticAI.

This module provides a complete end-to-end system:
1. Ingestion from Twitter WebSocket stream and/or Telegram
2. Messages pushed to Redis queue
3. Agent processes messages using PydanticAI

Usage:
    uv run -m synesis.agent

Or in code:
    from synesis.agent import run_agent
    await run_agent()

Configuration (set in .env):
    - TWITTERAPI_API_KEY: For Twitter stream
    - TELEGRAM_API_ID, TELEGRAM_API_HASH: For Telegram
"""

from synesis.agent.__main__ import run_unified_agent

# Alias for backwards compatibility
run_agent = run_unified_agent

__all__ = ["run_agent", "run_unified_agent"]
