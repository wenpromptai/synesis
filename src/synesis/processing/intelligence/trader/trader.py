"""Trader — the sole decision maker in the intelligence pipeline.

Receives macro regime + debate arguments and produces TradeIdea outputs.
Per-ticker mode receives full debate history; portfolio mode receives
compressed summaries (last round only). Supports both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.context import (
    format_debate_history,
    format_debate_summary_for_ticker,
    format_macro_context,
)
from synesis.processing.intelligence.models import TraderOutput

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 3

_SYSTEM_PROMPT = """\
You are the senior Trader at a multi-strategy hedge fund. You are the SOLE \
decision maker — every other agent in this pipeline gathered data or argued \
a case. You decide.

Today's date: {current_date}

## Context

Below you will find the macro regime assessment plus the bull/bear debate \
arguments for {scope_description}. The researchers have already synthesized \
all analyst data (fundamentals, technicals, sentiment, news) into their \
arguments with specific figures — work from their analysis.

## Your Job

1. **Read both sides of the debate.** Who made the stronger case? Where did \
the evidence actually point?
2. **Make a decisive call.** If you're not convinced either way, simply \
don't produce a TradeIdea for that ticker.
3. **Write the trade_structure.** This is the most important field — it is \
what we act on. Be specific and complete: "buy 100 shares NVDA", \
"bull call spread NVDA 150/160 June exp", "sell NVDA May 150 puts", \
"equity L/S: long NVDA / short AMD 2:1 ratio". Consider the macro regime \
and current IV when choosing between shares and options.
4. **Name the catalyst and timeframe.** What triggers the move, and when?

{mode_instructions}

## Tools
- `web_search(query, recency)` — search for anything that informs your \
trading decision: current prices, recent earnings, breaking news, options \
flow, analyst targets, or verify claims from the debate. Budget: \
{web_search_cap} calls.
- `web_read(url)` — read full article content (~4000 chars). Unlimited.

## Rules
- Ground your decision in the debate evidence. Do not fabricate data.
- If one side of the debate is missing (error), note this and decide \
with what you have — or skip if insufficient.
- Be specific with trade structure — no vague "consider options".
"""

_PORTFOLIO_INSTRUCTIONS = """\
## Portfolio Mode
You are reviewing ALL tickers together. Consider:
- Cross-ticker correlation (are multiple ideas in the same sector/theme?)
- Concentration risk (too much exposure to one factor?)
- Capital allocation (which ideas deserve the largest position?)
- **Pair / relative value trades**: If one ticker has a strong bull case and \
another has a strong bear case, you can create a single TradeIdea with \
tickers=["NVDA", "AMD"] and trade_structure describing the combined position \
(e.g., "equity L/S: long NVDA / short AMD").
Add a portfolio_note with cross-ticker observations."""

_PER_TICKER_INSTRUCTIONS = ""


@dataclass
class TraderDeps:
    """Dependencies for Trader."""

    current_date: date
    web_search_calls: int = field(default=0, init=False)


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[TraderDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for verification context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[TraderDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Context Formatters ───────────────────────────────────────────


def _format_debate_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format full debate history for a single ticker."""
    bull_analyses = state.get("bull_analyses", [])
    bear_analyses = state.get("bear_analyses", [])

    history = sorted(
        [
            item
            for item in bull_analyses + bear_analyses
            if not item.get("error") and item.get("ticker") == ticker
        ],
        key=lambda x: (x.get("round", 0), x.get("role") == "bear"),
    )

    if not history:
        return "## Debate\n[No debate available — analysis failed or missing]"

    return format_debate_history(history)


def _build_per_ticker_prompt(state: dict[str, Any], ticker: str) -> str:
    """Build the user prompt for per-ticker mode."""
    parts = [
        f"## Ticker: {ticker}",
        format_macro_context(state),
        _format_debate_for_ticker(state, ticker),
    ]
    return "\n\n".join(parts)


def _build_portfolio_prompt(state: dict[str, Any], tickers: list[str]) -> str:
    """Build the user prompt for portfolio mode.

    Uses compressed debate summaries (last round's argument + key_evidence
    per side, earlier rounds discarded) to keep context manageable.
    """
    parts = [format_macro_context(state)]
    for ticker in tickers:
        parts.append(f"---\n\n## Ticker: {ticker}")
        parts.append(format_debate_summary_for_ticker(state, ticker))
    return "\n\n".join(parts)


# ── Public API ───────────────────────────────────────────────────


async def analyze_trade_per_ticker(
    state: dict[str, Any],
    current_date: date,
) -> TraderOutput:
    """Run Trader for a single ticker (per_ticker mode).

    Receives macro context + full debate history for one ticker.
    Called once per ticker via Send fan-out.
    """
    ticker = state["ticker"]
    logger.info("Starting Trader (per_ticker)", ticker=ticker)

    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_per_ticker_prompt(state, ticker)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_SYSTEM_PROMPT.format(
            current_date=current_date,
            web_search_cap=_WEB_SEARCH_CAP,
            mode_instructions=_PER_TICKER_INSTRUCTIONS,
            scope_description=f"ticker {ticker}",
        ),
        tools=[_tool_web_search, _tool_web_read],
    )

    result = await agent.run(user_prompt, deps=deps)
    output = result.output

    logger.info(
        "Trader complete (per_ticker)",
        ticker=ticker,
        ideas=len(output.trade_ideas),
    )

    for idea in output.trade_ideas:
        if idea.analysis_date != current_date:
            idea.analysis_date = current_date

    return output


async def analyze_trade_portfolio(
    state: dict[str, Any],
    current_date: date,
    tickers: list[str],
) -> TraderOutput:
    """Run Trader for all tickers at once (portfolio mode).

    Receives macro context + compressed debate summaries for all tickers
    in one call. Can produce pair/relative value trades with multi-ticker
    TradeIdeas.
    """
    logger.info("Starting Trader (portfolio)", tickers=tickers)

    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_portfolio_prompt(state, tickers)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_SYSTEM_PROMPT.format(
            current_date=current_date,
            web_search_cap=_WEB_SEARCH_CAP,
            mode_instructions=_PORTFOLIO_INSTRUCTIONS,
            scope_description=f"tickers {', '.join(tickers)}",
        ),
        tools=[_tool_web_search, _tool_web_read],
    )

    result = await agent.run(user_prompt, deps=deps)
    output = result.output

    logger.info(
        "Trader complete (portfolio)",
        ideas=len(output.trade_ideas),
    )

    for idea in output.trade_ideas:
        if idea.analysis_date != current_date:
            idea.analysis_date = current_date

    return output
