"""Provider factory for creating financial data providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.base import TickerProvider
from synesis.providers.finnhub import FinnhubTickerProvider

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def create_ticker_provider(
    redis: Redis,
    api_key: str | None = None,
) -> TickerProvider:
    """Create a FinnhubTickerProvider for ticker verification.

    Args:
        redis: Redis client for caching
        api_key: Optional API key override

    Raises:
        ValueError: If FINNHUB_API_KEY is not configured
    """
    settings = get_settings()
    key = api_key or (
        settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
    )
    if not key:
        raise ValueError("FINNHUB_API_KEY required for ticker verification")

    logger.debug("Creating FinnhubTickerProvider")
    return FinnhubTickerProvider(api_key=key, redis=redis)
