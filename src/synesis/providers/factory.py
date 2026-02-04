"""Provider factory for creating financial data providers.

This module provides factory functions that create providers based on
configuration settings. This allows swapping providers without changing
consumer code.

Usage:
    from synesis.providers import get_price_provider, get_ticker_provider

    # Create providers based on settings
    price_provider = await create_price_provider(redis)
    ticker_provider = await create_ticker_provider(redis)

    # Use the providers
    price = await price_provider.get_price("AAPL")
    is_valid, company = await ticker_provider.verify_ticker("AAPL")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.base import (
    FundamentalsProvider,
    PriceProvider,
    TickerProvider,
)
from synesis.providers.finnhub import (
    FinnhubFundamentalsProvider,
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

    # Future: Add other providers here
    # elif provider_type == "polygon":
    #     return PolygonPriceProvider(...)
    # elif provider_type == "yahoo":
    #     return YahooPriceProvider(...)

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


async def create_fundamentals_provider(
    redis: Redis,
    api_key: str | None = None,
) -> FundamentalsProvider | None:
    """Create a fundamentals provider based on settings.

    Args:
        redis: Redis client for caching
        api_key: Optional API key override (uses settings if not provided)

    Returns:
        FundamentalsProvider implementation based on settings.fundamentals_provider,
        or None if fundamentals_provider is set to "none"

    Raises:
        ValueError: If the configured provider is not supported or API key is missing
    """
    settings = get_settings()
    provider_type = settings.fundamentals_provider

    if provider_type == "none":
        logger.debug("Fundamentals provider disabled")
        return None

    if provider_type == "finnhub":
        key = api_key or (
            settings.finnhub_api_key.get_secret_value() if settings.finnhub_api_key else None
        )
        if not key:
            raise ValueError("Finnhub API key required for finnhub fundamentals provider")

        logger.debug("Creating FinnhubFundamentalsProvider")
        return FinnhubFundamentalsProvider(api_key=key, redis=redis)

    # Future: Add other providers here
    # elif provider_type == "polygon":
    #     return PolygonFundamentalsProvider(...)

    raise ValueError(f"Unsupported fundamentals provider: {provider_type}")


# =============================================================================
# Combined Service (Backwards Compatibility)
# =============================================================================


class FinnhubService:
    """Combined Finnhub service that implements all provider protocols.

    This class combines TickerProvider and FundamentalsProvider functionality
    in a single service for backwards compatibility with existing code that
    uses FinnhubService directly.

    New code should prefer using the individual providers via the factory
    functions.

    Usage:
        service = FinnhubService(api_key="your_key", redis=redis_client)
        is_valid, company = await service.verify_ticker("AAPL")
        financials = await service.get_basic_financials("AAPL")
        await service.close()
    """

    def __init__(self, api_key: str, redis: Redis) -> None:
        """Initialize FinnhubService.

        Args:
            api_key: Finnhub API key
            redis: Redis client for caching
        """
        self._ticker = FinnhubTickerProvider(api_key=api_key, redis=redis)
        self._fundamentals = FinnhubFundamentalsProvider(api_key=api_key, redis=redis)

    # ─────────────────────────────────────────────────────────────
    # TickerProvider delegation
    # ─────────────────────────────────────────────────────────────

    async def verify_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Verify if a ticker symbol exists on a major exchange."""
        return await self._ticker.verify_ticker(ticker)

    async def search_symbol(self, query: str) -> list[dict[str, str]]:
        """Search for stock symbols matching a query."""
        return await self._ticker.search_symbol(query)

    # ─────────────────────────────────────────────────────────────
    # FundamentalsProvider delegation
    # ─────────────────────────────────────────────────────────────

    async def get_basic_financials(self, ticker: str) -> dict[str, Any] | None:
        """Get key financial metrics for a stock."""
        return await self._fundamentals.get_basic_financials(ticker)

    async def get_insider_transactions(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent insider transactions."""
        return await self._fundamentals.get_insider_transactions(ticker, limit)

    async def get_insider_sentiment(self, ticker: str) -> dict[str, Any] | None:
        """Get aggregate insider sentiment (MSPR score)."""
        return await self._fundamentals.get_insider_sentiment(ticker)

    async def get_sec_filings(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent SEC filings."""
        return await self._fundamentals.get_sec_filings(ticker, limit)

    async def get_eps_surprises(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical EPS surprises."""
        return await self._fundamentals.get_eps_surprises(ticker, limit)

    async def get_earnings_calendar(self, ticker: str) -> dict[str, Any] | None:
        """Get next earnings date."""
        return await self._fundamentals.get_earnings_calendar(ticker)

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        await self._ticker.close()
        await self._fundamentals.close()
        logger.debug("FinnhubService closed")
