"""Debate subgraph — multi-round bull/bear debate loop per ticker.

Compiled once and invoked per ticker from the main intelligence graph.
Uses LangGraph's cyclic conditional edges for the debate loop:

    START → bull_debate → bear_debate → route
                                          ├─ round < max_rounds → bull_debate
                                          └─ round >= max_rounds → END
"""

from __future__ import annotations

from datetime import date
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from synesis.core.logging import get_logger
from synesis.processing.intelligence.debate.bear import research_bear
from synesis.processing.intelligence.debate.bull import research_bull

logger = get_logger(__name__)


class DebateState(TypedDict):
    """State for the per-ticker debate subgraph."""

    # Context (input, read-only during debate)
    ticker: str
    current_date: str
    social_analysis: dict[str, Any]
    news_analysis: dict[str, Any]
    company_analyses: list[dict[str, Any]]
    price_analyses: list[dict[str, Any]]

    # Debate tracking
    debate_history: Annotated[list[dict[str, Any]], add]
    round: int
    max_rounds: int


async def _bull_debate_node(state: DebateState) -> dict[str, Any]:
    """Run BullResearcher with debate history context."""
    current = date.fromisoformat(state["current_date"])
    result = await research_bull(dict(state), current, debate_history=state["debate_history"])
    return {"debate_history": [result.model_dump(mode="json")]}


async def _bear_debate_node(state: DebateState) -> dict[str, Any]:
    """Run BearResearcher with debate history, then increment round."""
    current = date.fromisoformat(state["current_date"])
    result = await research_bear(dict(state), current, debate_history=state["debate_history"])
    return {
        "debate_history": [result.model_dump(mode="json")],
        "round": state["round"] + 1,
    }


def _should_continue(state: DebateState) -> str:
    """Continue debate if more rounds remain."""
    if state["round"] >= state["max_rounds"]:
        return END
    return "bull_debate"


def build_debate_subgraph() -> Any:
    """Build and compile the debate subgraph. Call once at startup."""
    graph = StateGraph(DebateState)

    graph.add_node("bull_debate", _bull_debate_node)
    graph.add_node("bear_debate", _bear_debate_node)

    graph.add_edge(START, "bull_debate")
    graph.add_edge("bull_debate", "bear_debate")
    graph.add_conditional_edges("bear_debate", _should_continue, ["bull_debate", END])

    return graph.compile()
