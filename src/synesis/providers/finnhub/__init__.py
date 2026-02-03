"""Finnhub provider implementations.

This module exports the Finnhub implementations of the provider protocols:
- FinnhubPriceProvider: Real-time price data with WebSocket + REST + Redis
- FinnhubTickerProvider: Ticker validation and symbol search
- FinnhubFundamentalsProvider: Company fundamental data

Also exports backwards-compatible aliases and utility functions.
"""

from synesis.providers.finnhub.fundamentals import FinnhubFundamentalsProvider
from synesis.providers.finnhub.prices import (
    FinnhubPriceProvider,
    PriceService,  # Backwards compat alias
    RateLimiter,
    close_price_service,
    get_price_service,
    get_rate_limiter,
    init_price_service,
)
from synesis.providers.finnhub.ticker import FinnhubTickerProvider

__all__ = [
    # Price provider
    "FinnhubPriceProvider",
    "PriceService",  # Backwards compat
    "RateLimiter",
    "get_price_service",
    "init_price_service",
    "close_price_service",
    "get_rate_limiter",
    # Ticker provider
    "FinnhubTickerProvider",
    # Fundamentals provider
    "FinnhubFundamentalsProvider",
]
