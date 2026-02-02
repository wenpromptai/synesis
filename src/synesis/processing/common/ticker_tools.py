"""Shared ticker verification tools for LLM agents.

This module provides a unified ticker verification function that can be used
by both Flow 1 (SmartAnalyzer) and Flow 2 (SentimentRefiner) agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.ingestion.finnhub import FinnhubService

logger = get_logger(__name__)


async def verify_ticker_finnhub(
    ticker: str,
    finnhub: "FinnhubService | None",
) -> str:
    """Verify if a US ticker symbol exists using Finnhub.

    IMPORTANT: This tool only verifies US tickers (NYSE, NASDAQ, etc.).
    For non-US tickers, use web search instead.

    Use this tool to validate US tickers BEFORE including them in your analysis.

    Args:
        ticker: The US ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")
        finnhub: FinnhubService instance or None

    Returns:
        Verification result string describing whether the ticker is valid
    """
    ticker = ticker.upper()

    if not finnhub:
        return f"Finnhub unavailable. Use web search to verify '{ticker}' instead."

    try:
        is_valid, company_name = await finnhub.verify_ticker(ticker)
        if is_valid:
            return f"VERIFIED: '{ticker}' is a valid US ticker. Company: {company_name}"
        return (
            f"NOT FOUND: '{ticker}' not found in US exchanges. "
            f"Could be non-US ticker, invalid, or delisted. Use web search to verify."
        )
    except Exception as e:
        logger.warning("Finnhub verification failed", ticker=ticker, error=str(e))
        return f"Finnhub error: {e}. Use web search to verify '{ticker}' instead."
