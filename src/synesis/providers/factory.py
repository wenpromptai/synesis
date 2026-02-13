"""Provider factory for creating financial data providers.

This module provides factory functions that create providers based on
configuration settings. This allows swapping providers without changing
consumer code.

Usage:
    from synesis.providers import create_price_provider, create_ticker_provider

    # Create providers based on settings
    price_provider = await create_price_provider(redis)
    ticker_provider = await create_ticker_provider(redis)

    # Use the providers
    price = await price_provider.get_price("AAPL")
    is_valid, company = await ticker_provider.verify_ticker("AAPL")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.base import (
    PriceProvider,
    TickerProvider,
)
from synesis.providers.finnhub import (
    FinnhubPriceProvider,
    FinnhubTickerProvider,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def create_price_provider(
    redis: Redis,
    api_key: str | None = None,
) -> PriceProvider:
    """Create a price provider based on settings.

    Args:
        redis: Redis client for caching
        api_key: Optional API key override (uses settings if not provided)

    Returns:
        PriceProvider implementation based on settings.price_provider

    Raises:
        ValueError: If the configured provider is not supported or API key is missing
    """
    settings = get_settings()
    provider_type = settings.price_provider

    if provider_type == "finnhub":
        key = api_key or (
            settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
        )
        if not key:
            raise ValueError("Finnhub API key required for finnhub price provider")

        logger.debug("Creating FinnhubPriceProvider")
        return FinnhubPriceProvider(api_key=key, redis=redis)

    raise ValueError(f"Unsupported price provider: {provider_type}")


async def create_ticker_provider(
    redis: Redis,
    api_key: str | None = None,
) -> TickerProvider:
    """Create a ticker provider based on settings.

    Args:
        redis: Redis client for caching
        api_key: Optional API key override (uses settings if not provided)

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
        key = api_key or (
            settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
        )
        if not key:
            raise ValueError("Finnhub API key required for finnhub ticker provider")

        logger.debug("Creating FinnhubTickerProvider")
        return FinnhubTickerProvider(api_key=key, redis=redis)

    raise ValueError(f"Unsupported ticker provider: {provider_type}")
