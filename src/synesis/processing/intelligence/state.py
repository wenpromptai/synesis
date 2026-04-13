"""LangGraph state definition for the intelligence pipeline.

Convention: State must be serializable (for checkpointing). Provider clients
(DB, Redis, httpx) are passed via closure in graph.py, NOT through state.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class ScanState(TypedDict):
    """State for the daily scan pipeline.

    Social/news signal discovery → MacroStrategist regime + ticker screening.
    No Send fan-outs, so no reducers needed.
    """

    current_date: str

    # Layer 1 outputs
    social_analysis: dict[str, Any]
    news_analysis: dict[str, Any]

    # Ticker extraction
    target_tickers: list[str]
    l1_tickers: list[str]

    # MacroStrategist output
    watchlist_context: dict[str, Any]
    macro_view: dict[str, Any]

    # Final output
    brief: dict[str, Any]


class AnalyzeState(TypedDict):
    """State for the on-demand ticker analysis pipeline.

    Takes tickers as input → company/price analysis → debate → trader.
    Uses Annotated[list, add] reducers for parallel Send fan-out fields.
    """

    current_date: str
    target_tickers: list[str]

    # Ticker research (single writer — runs parallel with L2, no reducer)
    ticker_research: dict[str, Any]

    # Layer 2 outputs (parallel Send writers)
    company_analyses: Annotated[list[dict[str, Any]], add]
    price_analyses: Annotated[list[dict[str, Any]], add]

    # Debate outputs (parallel Send writers)
    bull_analyses: Annotated[list[dict[str, Any]], add]
    bear_analyses: Annotated[list[dict[str, Any]], add]

    # Trader outputs (parallel Send writers in per_ticker mode)
    trade_ideas: Annotated[list[dict[str, Any]], add]
    portfolio_note: str

    # Final output
    brief: dict[str, Any]
