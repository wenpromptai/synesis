"""LangGraph state definition for the intelligence pipeline.

Convention: State must be serializable (for checkpointing). Provider clients
(DB, Redis, httpx) are passed via closure in graph.py, NOT through state.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class IntelligenceState(TypedDict):
    """Shared state for the daily intelligence pipeline.

    Keys without reducers are overwritten by the last writer.
    Keys with Annotated[list, add] append from each writer (safe for parallel nodes).
    """

    # Input (set at invocation)
    current_date: str

    # Tier 1 outputs (one writer each, no reducer needed)
    social_analysis: dict[str, Any]
    news_analysis: dict[str, Any]

    # Ticker extraction (deterministic)
    target_tickers: list[str]

    # Tier 2 outputs (multiple parallel writers via Send, needs reducer)
    company_analyses: Annotated[list[dict[str, Any]], add]

    # Final output
    brief: dict[str, Any]
