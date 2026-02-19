"""Provider factory for creating financial data providers.

This module provides a factory function that creates a ticker provider based on
configuration settings. This allows swapping providers without changing
consumer code.

Usage:
    from synesis.providers import create_ticker_provider

    # Create provider based on settings
    ticker_provider = await create_ticker_provider(redis)

    # Use the provider
    is_valid, ticker_region, company = await ticker_provider.verify_ticker("AAPL")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.base import (
    TickerProvider,
    WatchlistDataProvider,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def create_ticker_provider(
    redis: Redis,
    api_key: str | None = None,
) -> TickerProvider:
    """Create a ticker provider based on settings.

    Args:
        redis: Redis client for caching
        api_key: Optional API key override (used for finnhub)

    Returns:
        TickerProvider implementation based on settings.ticker_provider

    Raises:
        ValueError: If the configured provider is not supported or API key is missing
    """
    settings = get_settings()
    provider_type = settings.ticker_provider

    if provider_type == "factset":
        from synesis.providers.factset import FactSetTickerProvider
        from synesis.providers.factset.client import get_factset_client

        logger.debug("Creating FactSetTickerProvider")
        return FactSetTickerProvider(client=get_factset_client(), redis=redis)

    if provider_type == "finnhub":
        from synesis.providers.finnhub import FinnhubTickerProvider

        key = api_key or (
            settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
        )
        if not key:
            raise ValueError("Finnhub API key required for finnhub ticker provider")

        logger.debug("Creating FinnhubTickerProvider")
        return FinnhubTickerProvider(api_key=key, redis=redis)

    raise ValueError(f"Unsupported ticker provider: {provider_type}")


async def create_fundamentals_provider(redis: Redis) -> WatchlistDataProvider | None:
    """Create a fundamentals provider for the watchlist processor.

    Args:
        redis: Redis client for caching

    Returns:
        WatchlistDataProvider implementation based on settings.fundamentals_provider,
        or None if provider is "none".

    Raises:
        ValueError: If the configured provider is not supported or API key is missing
    """
    settings = get_settings()
    provider_type = settings.fundamentals_provider

    if provider_type == "none":
        return None

    if provider_type == "factset":
        from synesis.providers.factset.client import FactSetClient
        from synesis.providers.factset.provider import FactSetProvider, FactSetWatchlistAdapter

        logger.debug("Creating FactSetWatchlistAdapter")
        return FactSetWatchlistAdapter(FactSetProvider(client=FactSetClient()))

    if provider_type == "finnhub":
        from synesis.providers.finnhub.fundamentals import FinnhubWatchlistAdapter

        key = settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
        if not key:
            raise ValueError("Finnhub API key required for finnhub fundamentals provider")

        logger.debug("Creating FinnhubWatchlistAdapter")
        return FinnhubWatchlistAdapter(api_key=key, redis=redis)

    raise ValueError(f"Unsupported fundamentals provider: {provider_type}")
