"""Shared ticker verification tools for LLM agents.

Verifies tickers against the static data/us_tickers.json file
(refreshed weekly by APScheduler). Falls back to yfinance Search
when a ticker is not found in the file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yfinance as yf

from synesis.core.logging import get_logger

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
        data: dict[str, str] = json.load(f)
    _ticker_cache = data

    logger.info("Ticker cache loaded", tickers=len(_ticker_cache))
    return _ticker_cache


async def verify_ticker(ticker: str, **_kwargs: object) -> str:
    """Verify if a ticker symbol exists against data/us_tickers.json.

    Falls back to yfinance Search when not found in the file.

    Args:
        ticker: The ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")

    Returns:
        Verification result string — VERIFIED with company name, NOT FOUND with
        yfinance search results, or an error message.
    """
    ticker = ticker.upper()
    tickers = _load_tickers()

    if ticker in tickers:
        return f"VERIFIED: '{ticker}'. Company: {tickers[ticker]}"

    return await _yfinance_fallback(ticker)


def _yfinance_search(query: str) -> list[dict[str, str]]:
    """Search yfinance for a ticker or company name (synchronous)."""
    try:
        s = yf.Search(query)
        return [
            {
                "symbol": q.get("symbol", ""),
                "name": q.get("shortname", ""),
                "exchange": q.get("exchange", ""),
            }
            for q in (s.quotes or [])[:5]
        ]
    except Exception as e:
        logger.warning("yfinance search failed", query=query, error=str(e))
        return []


async def _yfinance_fallback(ticker: str) -> str:
    """Fall back to yfinance Search to look up an unverified ticker."""
    results = await asyncio.to_thread(_yfinance_search, ticker)
    if results:
        matches = "\n".join(f"  {r['symbol']} — {r['name']} ({r['exchange']})" for r in results)
        return f"NOT FOUND in US ticker list. yfinance matches for '{ticker}':\n{matches}"
    return f"NOT FOUND: '{ticker}' not in US ticker list or yfinance. May be invalid or delisted."
