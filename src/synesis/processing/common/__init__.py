"""Shared utilities for processing flows.

Cross-flow utilities:
- LLM provider factory (PydanticAI model creation)
- Web search utility (market impact research)
- Watchlist management (unified ticker tracking)
"""

from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker_finnhub
from synesis.processing.common.watchlist import TickerMetadata, WatchlistManager
from synesis.processing.common.web_search import (
    Recency,
    SearchProvidersExhaustedError,
    format_search_results,
    search_market_impact,
)

__all__ = [
    "Recency",
    "SearchProvidersExhaustedError",
    "TickerMetadata",
    "WatchlistManager",
    "create_model",
    "format_search_results",
    "search_market_impact",
    "verify_ticker_finnhub",
]
