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
from synesis.processing.common.llm import create_model, web_search_config
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.context import (
    format_debate_history,
    format_debate_summary_for_ticker,
)
from synesis.processing.intelligence.models import TraderOutput

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 5


_SEARCH_DESC = (
    "look up current prices, recent earnings, breaking news, options flow, "
    "analyst targets, or verify claims from the debate"
)

_SYSTEM_PROMPT = """\
You are the senior Trader at a multi-strategy hedge fund. You are the SOLE \
decision maker — every other agent in this pipeline gathered data or argued \
a case. You decide.

Today's date: {current_date}

## Context

Below you will find the macro regime assessment plus the bull/bear debate \
arguments for {scope_description}. The researchers have already synthesized \
all analyst data (fundamentals, technicals, sentiment, news) into their \
arguments with specific figures — work from their analysis. Each side has \
stated their variant vs consensus, a catalyst, and what would prove them wrong.

## Your Job

1. **Read both sides of the debate.** Who made the stronger case? Whose \
variant perception is better supported by evidence? Which side's catalyst \
is more imminent and more likely?
2. **Make a decisive call.** If you're not convinced either way, simply \
don't produce a TradeIdea for that ticker. Having no position IS a position.
3. **Write the trade_structure.** Equity positions ONLY — this is what we \
act on. Formats: "long NVDA" or "short AMD". One ticker per trade idea. \
No options strategies, no pair trades.
4. **Set entry, target, and stop.** Use current price as entry. Target = \
your conviction case price. Stop = where the thesis is wrong (the level \
that invalidates the variant). Calculate risk_reward_ratio as \
(target - entry) / (entry - stop) for longs, or \
(entry - target) / (stop - entry) for shorts. \
**Reject any trade with R/R below 2:1 for longs or 3:1 for shorts.**
5. **Assign conviction_tier:**
   - Tier 1 (highest): Multiple independent signals confirm thesis, downside \
bounded, hard catalyst within 30 days. Maps to 5-8% position.
   - Tier 2: Strong directional thesis, 1-2 uncertainties remain, catalyst \
within 60 days. Maps to 2-5% position.
   - Tier 3: Interesting setup but critical unknowns, optionality only. \
Maps to 0.5-2% position.
6. **Write expression_note.** Based on IV/RV data from the debate, note \
whether options could enhance this trade (e.g., "IV at low end of range vs \
realized — calls are cheap for leveraged exposure" or "IV elevated — \
consider selling put spreads for defined-risk entry"). Do NOT construct \
specific strikes or expiries — just flag the vol regime and direction.
7. **Name the catalyst and timeframe.** What triggers the move, and when?

{mode_instructions}

## Tools
{search_docs}\
- `web_read(url)` — read full article content (~4000 chars). Unlimited.

## Rules
- Ground your decision in the debate evidence. Do not fabricate data.
- If one side of the debate is missing (error), note this and decide \
with what you have — or skip if insufficient.
- Every TradeIdea MUST have entry_price, target_price, stop_price, and \
conviction_tier. Incomplete ideas are useless.
- Write downside_scenario: what happens if you're wrong? Be specific.
"""

_PORTFOLIO_INSTRUCTIONS = """\
## Portfolio Mode
You are reviewing ALL tickers together. Consider:
- Cross-ticker correlation (are multiple ideas in the same sector/theme?)
- Concentration risk (too much exposure to one factor?)
- Capital allocation (which ideas deserve the largest position? Use conviction tiers.)
- If you see a natural long/short pair (e.g., long semis via AVGO, short \
software via CRM), create separate TradeIdeas for each leg and note the \
pairing in portfolio_note.
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
        _format_debate_for_ticker(state, ticker),
    ]
    return "\n\n".join(parts)


def _build_portfolio_prompt(state: dict[str, Any], tickers: list[str]) -> str:
    """Build the user prompt for portfolio mode.

    Uses compressed debate summaries (last round's argument + key_evidence
    per side, earlier rounds discarded) to keep context manageable.
    """
    parts: list[str] = []
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

    search = web_search_config(_WEB_SEARCH_CAP, _SEARCH_DESC)
    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_per_ticker_prompt(state, ticker)
    tools: list[Any] = [_tool_web_read]
    if not search.native:
        tools.append(_tool_web_search)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_SYSTEM_PROMPT.format(
            current_date=current_date,
            mode_instructions=_PER_TICKER_INSTRUCTIONS,
            scope_description=f"ticker {ticker}",
            search_docs=search.prompt_docs,
        ),
        tools=tools,
        builtin_tools=search.builtin_tools,
    )

    result = await agent.run(user_prompt, deps=deps)
    usage = result.usage()
    logger.info(
        "Trader API response (per_ticker)",
        ticker=ticker,
        finish_reason=result.response.finish_reason,
        model=result.response.model_name,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
        tool_calls=usage.tool_calls,
    )
    output = result.output

    # Log trade ideas detail
    if output.trade_ideas:
        for idea in output.trade_ideas:
            logger.info(
                "Trader trade idea (per_ticker)",
                ticker=ticker,
                idea_tickers=idea.tickers,
                structure=idea.trade_structure,
                timeframe=idea.timeframe,
                conviction_tier=idea.conviction_tier,
                risk_reward=idea.risk_reward_ratio,
                entry=idea.entry_price,
                target=idea.target_price,
                stop=idea.stop_price,
            )
    else:
        logger.warning(
            "Trader produced 0 ideas (per_ticker)",
            ticker=ticker,
            web_searches_used=deps.web_search_calls,
        )

    logger.info(
        "Trader complete (per_ticker)",
        ticker=ticker,
        ideas=len(output.trade_ideas),
        web_searches_used=deps.web_search_calls,
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
    in one call. Produces one TradeIdea per ticker with portfolio-level
    awareness (correlation, concentration, capital allocation).
    """
    logger.info("Starting Trader (portfolio)", tickers=tickers)

    search = web_search_config(_WEB_SEARCH_CAP, _SEARCH_DESC)
    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_portfolio_prompt(state, tickers)
    tools_list: list[Any] = [_tool_web_read]
    if not search.native:
        tools_list.append(_tool_web_search)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_SYSTEM_PROMPT.format(
            current_date=current_date,
            mode_instructions=_PORTFOLIO_INSTRUCTIONS,
            scope_description=f"tickers {', '.join(tickers)}",
            search_docs=search.prompt_docs,
        ),
        tools=tools_list,
        builtin_tools=search.builtin_tools,
    )

    result = await agent.run(user_prompt, deps=deps)
    usage = result.usage()
    logger.info(
        "Trader API response (portfolio)",
        tickers=tickers,
        finish_reason=result.response.finish_reason,
        model=result.response.model_name,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
        tool_calls=usage.tool_calls,
    )
    output = result.output

    # Log trade ideas detail
    if output.trade_ideas:
        for idea in output.trade_ideas:
            logger.info(
                "Trader trade idea (portfolio)",
                idea_tickers=idea.tickers,
                structure=idea.trade_structure,
                timeframe=idea.timeframe,
                conviction_tier=idea.conviction_tier,
                risk_reward=idea.risk_reward_ratio,
                entry=idea.entry_price,
                target=idea.target_price,
                stop=idea.stop_price,
            )
    else:
        logger.warning(
            "Trader produced 0 ideas (portfolio)",
            tickers=tickers,
            web_searches_used=deps.web_search_calls,
        )

    if output.portfolio_note:
        logger.info(
            "Trader portfolio note",
            note_length=len(output.portfolio_note),
            note_preview=output.portfolio_note[:200],
        )

    logger.info(
        "Trader complete (portfolio)",
        ideas=len(output.trade_ideas),
        web_searches_used=deps.web_search_calls,
        has_portfolio_note=bool(output.portfolio_note),
    )

    for idea in output.trade_ideas:
        if idea.analysis_date != current_date:
            idea.analysis_date = current_date

    return output
