"""Shared ticker verification tools for LLM agents.

Verifies tickers against the static data/us_tickers.json file
(refreshed weekly by APScheduler). Falls back to SearXNG search
when a ticker is not found in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

from synesis.core.logging import get_logger
from synesis.processing.common.web_search import format_search_results, search_ticker_analysis

logger = get_logger(__name__)

_TICKERS_FILE = Path(__file__).resolve().parents[4] / "data" / "us_tickers.json"

# In-memory cache of the ticker file (loaded once on first call)
_ticker_cache: dict[str, str] | None = None


def _load_tickers() -> dict[str, str]:
    """Load the static ticker file into memory (singleton)."""
    global _ticker_cache
    if _ticker_cache is not None:
        return _ticker_cache

    if not _TICKERS_FILE.exists():
        logger.warning("Ticker file not found", path=str(_TICKERS_FILE))
        _ticker_cache = {}
        return _ticker_cache

    with open(_TICKERS_FILE) as f:
        _ticker_cache = json.load(f)

    logger.info("Ticker cache loaded", tickers=len(_ticker_cache))
    return _ticker_cache


async def verify_ticker(ticker: str, **_kwargs: object) -> str:
    """Verify if a ticker symbol exists against data/us_tickers.json.

    Falls back to SearXNG search when not found in the file.

    Args:
        ticker: The ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")

    Returns:
        Verification result string — VERIFIED with company name, NOT FOUND with
        SearXNG search results, or an error message.
    """
    ticker = ticker.upper()
    tickers = _load_tickers()

    if ticker in tickers:
        return f"VERIFIED: '{ticker}'. Company: {tickers[ticker]}"

    return await _searxng_fallback(ticker)


async def _searxng_fallback(ticker: str) -> str:
    """Fall back to SearXNG to look up an unverified ticker."""
    try:
        results = await search_ticker_analysis(ticker)
        if results:
            return (
                f"NOT FOUND in US ticker list. SearXNG results for '{ticker}':\n"
                + format_search_results(results)
            )
        return (
            f"NOT FOUND: '{ticker}' not in US ticker list or SearXNG. May be invalid or delisted."
        )
    except Exception as e:
        logger.warning("SearXNG fallback failed", ticker=ticker, error=str(e))
        return f"NOT FOUND: '{ticker}' not found. Verification unavailable."
