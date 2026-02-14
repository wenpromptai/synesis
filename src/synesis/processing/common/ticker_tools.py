"""Shared ticker verification tools for LLM agents.

This module provides a unified ticker verification function that can be used
by both Flow 1 (SmartAnalyzer) and Flow 2 (SentimentRefiner) agents.

Uses FactSet TickerProvider for ticker verification (global coverage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.providers.base import TickerProvider

logger = get_logger(__name__)


async def verify_ticker(
    ticker: str,
    ticker_provider: "TickerProvider | None",
) -> str:
    """Verify if a ticker symbol exists using a ticker provider.

    Use this tool to validate tickers BEFORE including them in your analysis.

    Args:
        ticker: The ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")
        ticker_provider: TickerProvider instance (e.g., FactSetTickerProvider) or None

    Returns:
        Verification result string describing whether the ticker is valid,
        including the full ticker_region and company name if found.
    """
    ticker = ticker.upper()

    if not ticker_provider:
        return f"Ticker provider unavailable. Use web search to verify '{ticker}' instead."

    try:
        is_valid, ticker_region, company_name = await ticker_provider.verify_ticker(ticker)
        if is_valid:
            return f"VERIFIED: '{ticker}' → {ticker_region}. Company: {company_name}"
        return (
            f"NOT FOUND: '{ticker}' not found in major exchanges. "
            f"Could be invalid or delisted. Use web search to verify."
        )
    except (ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError, OSError) as e:
        logger.warning("Ticker verification failed", ticker=ticker, error=str(e))
        return f"Ticker verification error: {e}. Use web search to verify '{ticker}' instead."
