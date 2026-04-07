"""EquityStrategist — the sole decision maker in the pipeline.

Reads all upstream analyst outputs (social, news, company, price) and
the macro regime, then produces ranked trade ideas with sentiment scores.
Analysts provide information — this agent provides judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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
from synesis.processing.intelligence.models import EquityIdeas

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 2


@dataclass
class EquityStrategistDeps:
    """Dependencies for EquityStrategist."""

    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = 0


# ── Context Formatting ───────────────────────────────────────────


def _format_macro_context(state: dict[str, Any]) -> str:
    """Format MacroView for the LLM prompt."""
    macro = state.get("macro_view", {})
    if not macro:
        return "## Macro Context\nNo macro assessment available."

    lines = ["## Macro Context"]
    lines.append(
        f"- **Regime**: {macro.get('regime', '?')} "
        f"(sentiment: {macro.get('sentiment_score', 0):+.1f})"
    )
    for driver in macro.get("key_drivers", []):
        lines.append(f"- Driver: {driver}")
    for tilt in macro.get("sector_tilts", []):
        lines.append(
            f"- Sector tilt: {tilt.get('sector', '?')} "
            f"({tilt.get('sentiment_score', 0):+.1f}) — {tilt.get('reasoning', '')}"
        )
    for risk in macro.get("risks", []):
        lines.append(f"- Risk: {risk}")
    return "\n".join(lines)


def _format_social_context(state: dict[str, Any]) -> str:
    """Format social sentiment analysis for the LLM prompt."""
    social = state.get("social_analysis", {})
    if not social:
        return "## Social Sentiment\nNo social analysis available."

    lines = ["## Social Sentiment"]
    lines.append(f"Summary: {social.get('summary', 'N/A')}")

    for mention in social.get("ticker_mentions", []):
        accounts = ", ".join(mention.get("source_accounts", []))
        lines.append(
            f"- **{mention.get('ticker', '?')}**: {mention.get('context', '')} [from: {accounts}]"
        )

    for theme in social.get("macro_themes", []):
        lines.append(f"- Theme: {theme.get('theme', '?')} — {theme.get('context', '')}")

    return "\n".join(lines)


def _format_news_context(state: dict[str, Any]) -> str:
    """Format news analysis for the LLM prompt."""
    news = state.get("news_analysis", {})
    if not news:
        return "## News\nNo news analysis available."

    lines = ["## News"]
    lines.append(f"Summary: {news.get('summary', 'N/A')}")

    for cluster in news.get("story_clusters", []):
        urgency = cluster.get("urgency", "normal")
        event_type = cluster.get("event_type", "other")
        lines.append(f"\n### {cluster.get('headline', '?')} [{event_type}, urgency={urgency}]")
        for fact in cluster.get("key_facts", []):
            lines.append(f"- {fact}")
        for ticker in cluster.get("tickers", []):
            lines.append(f"- Ticker: **{ticker.get('ticker', '?')}** — {ticker.get('context', '')}")

    for theme in news.get("macro_themes", []):
        lines.append(f"- Theme: {theme.get('theme', '?')} — {theme.get('context', '')}")

    return "\n".join(lines)


def _format_company_context(state: dict[str, Any]) -> str:
    """Format company analyses for the LLM prompt."""
    companies = state.get("company_analyses", [])
    valid = [c for c in companies if not c.get("error")]
    if not valid:
        return "## Company Analyses\nNo company analyses available."

    lines = ["## Company Analyses"]
    for c in valid:
        ticker = c.get("ticker", "?")
        lines.append(f"\n### {ticker} ({c.get('company_name', '')})")

        # Key fundamentals
        health = c.get("financial_health", {})
        if health:
            parts = []
            if health.get("piotroski_f") is not None:
                parts.append(f"Piotroski={health['piotroski_f']}")
            if health.get("roe") is not None:
                parts.append(f"ROE={health['roe']:.1%}")
            if health.get("revenue_growth") is not None:
                parts.append(f"RevGrowth={health['revenue_growth']:.1%}")
            if health.get("market_cap") is not None:
                parts.append(f"MCap=${health['market_cap'] / 1e9:.1f}B")
            if parts:
                lines.append(f"Fundamentals: {', '.join(parts)}")

        # Insider activity
        insider = c.get("insider_signal", {})
        if insider:
            mspr = insider.get("mspr")
            if mspr is not None:
                lines.append(
                    f"Insiders: MSPR={mspr:+.2f}, "
                    f"buys={insider.get('buy_count', 0)}, sells={insider.get('sell_count', 0)}, "
                    f"cluster={'YES' if insider.get('cluster_detected') else 'no'}"
                )
            if insider.get("csuite_activity"):
                lines.append(f"C-suite: {insider['csuite_activity']}")

        # Red flags
        for rf in c.get("red_flags", []):
            lines.append(
                f"⚠️ [{rf.get('severity', '?')}] {rf.get('flag', '')}: {rf.get('evidence', '')}"
            )

        # Qualitative
        if c.get("primary_thesis"):
            lines.append(f"Thesis: {c['primary_thesis']}")
        if c.get("business_summary"):
            lines.append(f"Business: {c['business_summary'][:200]}...")
        risks = c.get("key_risks", [])
        if risks:
            lines.append(f"Risks: {', '.join(risks[:3])}")

    return "\n".join(lines)


def _format_price_context(state: dict[str, Any]) -> str:
    """Format price analyses for the LLM prompt."""
    prices = state.get("price_analyses", [])
    valid = [p for p in prices if not p.get("error")]
    if not valid:
        return "## Price Analysis\nNo price analysis available."

    lines = ["## Price Analysis"]
    for p in valid:
        ticker = p.get("ticker", "?")
        lines.append(f"\n### {ticker}")
        if p.get("spot_price"):
            change = p.get("change_1d_pct")
            change_str = f" ({change:+.1f}%)" if change is not None else ""
            lines.append(f"Price: ${p['spot_price']:.2f}{change_str}")
        if p.get("technical_narrative"):
            lines.append(f"Technical: {p['technical_narrative']}")
        if p.get("options_narrative"):
            lines.append(f"Options: {p['options_narrative']}")
        if p.get("notable_setups"):
            lines.append("Notable setups: " + "; ".join(p["notable_setups"]))
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a senior equity strategist. You are the SOLE DECISION MAKER in this pipeline.
All upstream analysts have gathered information for you — social sentiment, news events,
company fundamentals, and price/technical data. YOUR job is to synthesize everything
and produce actionable trade ideas.

Today's date: {current_date}

## Your Job

Read all the context provided and produce **TradeIdeas** for the most promising opportunities.

For each idea:
- `sentiment_score`: -1.0 (strong short) to 1.0 (strong long). This is YOUR call.
  ±0.8-1.0 = high conviction, ±0.4-0.7 = moderate, ±0.1-0.3 = low.
- `thesis`: 2-3 sentence investment case. Be specific — cite the evidence from analysts.
- `structure`: How to express the trade (buy shares, buy calls, put spread, etc.)
- `catalyst`: What triggers the move and when.
- `timeframe`: Expected holding period.
- `key_risk`: Single biggest risk to this trade.

## Guidelines
- You are the judge. The analysts provided facts — you decide what to do with them.
- Consider macro alignment: a long tech trade during risk_off needs extra justification.
- Cross-reference: does the company analysis support what social/news are saying?
- Don't just list every ticker — filter to ideas with clear, differentiated theses.
- A single well-reasoned idea is worth more than five generic ones.

## Tools
- `web_search(query, recency)` — Search the web for context. Budget: {web_search_cap} calls.
  Use specific queries for maximum value.
- `web_read(url)` — Read a web page for full article content (~4000 chars). Unlimited calls.
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[EquityStrategistDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for equity context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[EquityStrategistDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Public API ───────────────────────────────────────────────────


async def analyze_equity(
    state: dict[str, Any],
    deps: EquityStrategistDeps,
) -> EquityIdeas:
    """Run the EquityStrategist on pipeline state."""
    logger.info("Starting EquityStrategist")

    # Build prompt from all upstream context
    macro_context = _format_macro_context(state)
    social_context = _format_social_context(state)
    news_context = _format_news_context(state)
    company_context = _format_company_context(state)
    price_context = _format_price_context(state)

    user_prompt = "\n\n".join(
        [
            macro_context,
            social_context,
            news_context,
            company_context,
            price_context,
        ]
    )

    # Construct agent
    agent: Agent[EquityStrategistDeps, EquityIdeas] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=EquityStrategistDeps,
        output_type=EquityIdeas,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            web_search_cap=_WEB_SEARCH_CAP,
        ),
        tools=[_tool_web_search, _tool_web_read],
    )

    try:
        result = await agent.run(user_prompt, deps=deps)
        output: EquityIdeas = result.output
    except Exception:
        logger.exception("EquityStrategist LLM call failed")
        return EquityIdeas(trade_ideas=[], analysis_date=deps.current_date)

    logger.info("EquityStrategist complete", trade_ideas=len(output.trade_ideas))

    if output.analysis_date != deps.current_date:
        output = output.model_copy(update={"analysis_date": deps.current_date})

    return output
