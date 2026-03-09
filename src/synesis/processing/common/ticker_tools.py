"""Shared ticker verification tools for LLM agents.

This module provides a unified ticker verification function used by
the Flow 1 SmartAnalyzer agent and the Twitter agent.

Uses FinnhubTickerProvider for primary verification, with automatic
SearXNG fallback when a ticker is not found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.processing.common.web_search import format_search_results, search_ticker_analysis

if TYPE_CHECKING:
    from synesis.providers.base import TickerProvider

logger = get_logger(__name__)


async def verify_ticker(
    ticker: str,
    ticker_provider: "TickerProvider | None",
) -> str:
    """Verify if a ticker symbol exists using a ticker provider.

    Use this tool to validate tickers BEFORE including them in your analysis.
    If the ticker is not found via Finnhub, automatically falls back to a
    SearXNG search to look it up (free, no quota cost).

    Args:
        ticker: The ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")
        ticker_provider: TickerProvider instance (e.g., FinnhubTickerProvider) or None

    Returns:
        Verification result string — VERIFIED with company name, NOT FOUND with
        SearXNG search results, or an error message.
    """
    ticker = ticker.upper()

    if not ticker_provider:
        return await _searxng_fallback(ticker)

    try:
        is_valid, company_name = await ticker_provider.verify_ticker(ticker)
        if is_valid:
            return f"VERIFIED: '{ticker}'. Company: {company_name}"
        return await _searxng_fallback(ticker)
    except (ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError, OSError) as e:
        logger.warning("Ticker verification failed", ticker=ticker, error=str(e))
        return await _searxng_fallback(ticker)


async def _searxng_fallback(ticker: str) -> str:
    """Fall back to SearXNG to look up an unverified ticker."""
    try:
        results = await search_ticker_analysis(ticker)
        if results:
            return (
                f"NOT FOUND via Finnhub. SearXNG results for '{ticker}':\n"
                + format_search_results(results)
            )
        return (
            f"NOT FOUND: '{ticker}' not found via Finnhub or SearXNG. May be invalid or delisted."
        )
    except Exception as e:
        logger.warning("SearXNG fallback failed", ticker=ticker, error=str(e))
        return f"NOT FOUND: '{ticker}' not found. Verification unavailable."
