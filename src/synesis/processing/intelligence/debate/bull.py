"""BullResearcher — builds the strongest evidence-based case FOR investing.

Receives per-ticker analyst context (social, news, company, price) via Send
fan-out and argues why the ticker is a compelling opportunity. One call per
ticker. Does NOT score — that is deferred to the Trader (Phase 3D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model, web_search_config
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.context import (
    format_company_context_for_ticker,
    format_debate_history,
    format_news_context_for_ticker,
    format_price_context_for_ticker,
    format_social_context_for_ticker,
)
from synesis.processing.intelligence.models import TickerDebate

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 2


_SEARCH_DESC = (
    "find recent catalysts, analyst upgrades, product launches, deal flow, "
    "guidance revisions, or gaps the analysts missed"
)

SYSTEM_PROMPT = """\
You are a senior Bull Researcher at a multi-strategy hedge fund. Your job is \
to build the strongest possible evidence-based case for BUYING this ticker.

Today's date: {current_date}

## Context

Below you will find research from upstream analysts covering company \
fundamentals, price/technicals, social sentiment, and news for this specific \
ticker. Not all sections may be present — work with whatever data is provided.

Synthesize ALL available data to form your bull thesis. Look for where \
multiple sources confirm each other — that's where conviction comes from.

## Your Role

You are an ADVOCATE — argue for buying, not for holding or watching. \
The Trader downstream will weigh both sides and decide. Your job is to make \
the strongest case you can, grounded in the data.

If the data is mixed or even bearish, find the contrarian angle. \
Acknowledge the top risks but explain why the opportunity outweighs them.

{debate_instructions}

## Output

- `argument`: 3-5 paragraphs. Lead with your thesis, then build the case \
using specifics from the analyst data. Cite actual numbers — don't be vague.
- `key_evidence`: 3-6 bullet points of your strongest supporting data points.

## Tools
{search_docs}\
- `web_read(url)` — read full article content (~4000 chars). Unlimited.

## Rules
- Cite specific data from the reports. Do not fabricate numbers.
- Be specific — no generic arguments.
"""


@dataclass
class BullResearcherDeps:
    """Dependencies for BullResearcher."""

    current_date: date
    web_search_calls: int = field(default=0, init=False)


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[BullResearcherDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for supporting context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[BullResearcherDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Public API ───────────────────────────────────────────────────


_DEBATE_INSTRUCTIONS = """\
## Debate Context

You are in a multi-round debate. The bear researcher's previous arguments are \
shown below. You MUST directly counter their strongest points — do not ignore \
them. Explain why their risks are overstated, mispriced, or outweighed by \
the opportunity. Build on your previous arguments but sharpen them in light \
of the bear's critique."""

_NO_DEBATE_INSTRUCTIONS = ""


async def research_bull(
    state: dict[str, Any],
    current_date: date,
    debate_history: list[dict[str, Any]] | None = None,
) -> TickerDebate:
    """Run the BullResearcher for a single ticker."""
    ticker = state["ticker"]
    history = debate_history or []
    round_num = len([h for h in history if h.get("role") == "bull"]) + 1
    logger.info("Starting BullResearcher", ticker=ticker, round=round_num)

    deps = BullResearcherDeps(current_date=current_date)

    # Build prompt from per-ticker context (no macro — deferred to Trader)
    prompt_parts = [
        f"## Ticker: {ticker}",
        format_social_context_for_ticker(state, ticker),
        format_news_context_for_ticker(state, ticker),
        format_company_context_for_ticker(state, ticker),
        format_price_context_for_ticker(state, ticker),
    ]
    if history:
        prompt_parts.append(format_debate_history(history))
    user_prompt = "\n\n".join(prompt_parts)

    debate_instructions = _DEBATE_INSTRUCTIONS if history else _NO_DEBATE_INSTRUCTIONS
    search = web_search_config(_WEB_SEARCH_CAP, _SEARCH_DESC)
    tools: list[Any] = [_tool_web_read]
    if not search.native:
        tools.append(_tool_web_search)

    agent: Agent[BullResearcherDeps, TickerDebate] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=BullResearcherDeps,
        output_type=TickerDebate,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=current_date,
            debate_instructions=debate_instructions,
            search_docs=search.prompt_docs,
        ),
        tools=tools,
        builtin_tools=search.builtin_tools,
    )

    result = await agent.run(user_prompt, deps=deps)
    output = result.output

    logger.info("BullResearcher complete", ticker=ticker, round=round_num)

    # Ensure consistent metadata
    updates: dict[str, Any] = {}
    if output.role != "bull":
        updates["role"] = "bull"
    if output.ticker != ticker:
        updates["ticker"] = ticker
    if output.analysis_date != current_date:
        updates["analysis_date"] = current_date
    if output.round != round_num:
        updates["round"] = round_num
    if updates:
        output = output.model_copy(update=updates)

    return output
