"""Market brief LLM analysis — synthesizes market data, news, events, and social signals."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.market.models import ContextGaps, MarketBriefAnalysis

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a senior market strategist producing the morning market brief for a
professional trading desk. It is 10:30 AM ET — the market has been open for
one hour.

You will receive:
1. MARKET SNAPSHOT — live benchmarks, sectors, VIX, and top movers
2. EVENTS CONTEXT — event diary entries from year-to-date (earnings, macro,
   fed decisions, filings) with outcomes
3. TWITTER INTEL — today's social media digest from financial accounts
4. NEWS SIGNALS — signals from the last 24 hours (Telegram/news analysis)
5. WEB SEARCH — targeted web search results for movers/themes not explained by internal context

Your job is to produce an actionable morning rundown:
- Explain WHY sectors and top movers are moving (connect to news, events,
  social signals)
- Identify patterns: is this a sector rotation? risk-off? earnings-driven?
- Cross-reference: if Tech is up and a big AI earnings beat was yesterday,
  connect those dots
- Be specific with tickers and numbers
- Be opinionated — give expert analysis, not just summaries

Today's date is {today}.
"""


def _format_events_context(events: list[dict[str, Any]]) -> str:
    """Format diary event entries for the LLM prompt."""
    if not events:
        return "No event diary entries available."

    lines: list[str] = []
    for entry in events[:30]:  # Cap at 30 most recent
        payload = entry.get("payload", {})
        entry_date = entry.get("entry_date", "")

        # YesterdayBriefAnalysis format
        headline = payload.get("headline", "")
        synthesis = payload.get("synthesis", "")
        themes = payload.get("themes", [])
        top_movers = payload.get("top_movers", [])

        lines.append(f"[{entry_date}] {headline}")
        if synthesis:
            lines.append(f"  Synthesis: {synthesis[:500]}")
        if top_movers:
            lines.append(f"  Movers: {', '.join(top_movers[:10])}")
        for theme in themes[:3]:
            title = theme.get("title", "")
            sentiment = theme.get("sentiment", "")
            tickers = theme.get("tickers", [])
            lines.append(f"  - {title} ({sentiment}) [{', '.join(tickers[:5])}]")
        lines.append("")

    return "\n".join(lines)


def _format_twitter_context(twitter_entries: list[dict[str, Any]]) -> str:
    """Format today's Twitter digest for the LLM prompt."""
    if not twitter_entries:
        return "No Twitter digest available for today."

    lines: list[str] = []
    for entry in twitter_entries:
        payload = entry.get("payload", {})
        themes = payload.get("themes", [])

        for theme in themes:
            title = theme.get("title", "")
            sentiment = theme.get("sentiment", "")
            summary = theme.get("summary", "")
            tickers = theme.get("tickers", [])
            ticker_syms = [t.get("ticker", "") if isinstance(t, dict) else t for t in tickers]
            sources = theme.get("sources", [])

            lines.append(f"- {title} ({sentiment})")
            if summary:
                lines.append(f"  {summary[:300]}")
            if ticker_syms:
                lines.append(f"  Tickers: {', '.join(ticker_syms[:8])}")
            if sources:
                lines.append(f"  Sources: {', '.join(sources[:5])}")

    return "\n".join(lines) if lines else "No Twitter themes found."


def _format_signals_context(signals: list[dict[str, Any]]) -> str:
    """Format recent news signals for the LLM prompt."""
    if not signals:
        return "No recent news signals."

    lines: list[str] = []
    for sig in signals[:20]:  # Cap at 20 most recent
        payload = sig.get("payload") or {}
        tickers = sig.get("tickers") or []
        topics = sig.get("primary_topics") or []
        timestamp = sig.get("time", "")

        # Extract key fields from signal payload
        extraction = payload.get("extraction") or {}
        headline = extraction.get("headline", "")
        summary = extraction.get("summary", "")

        analysis = payload.get("analysis") or {}
        market_impact = analysis.get("market_impact_summary", "")

        parts = [f"[{timestamp}] {headline}"]
        if summary:
            parts.append(f"  {summary[:200]}")
        if market_impact:
            parts.append(f"  Impact: {market_impact[:200]}")
        if tickers:
            parts.append(f"  Tickers: {', '.join(tickers[:8])}")
        if topics:
            parts.append(f"  Topics: {', '.join(topics[:5])}")
        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def _format_search_context(search_results: list[dict[str, Any]]) -> str:
    """Format web search results for the LLM prompt."""
    if not search_results:
        return "No web search results available."

    lines: list[str] = []
    for r in search_results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"- {title}: {snippet[:500]}")
        else:
            lines.append(f"- {title}")

    return "\n".join(lines)


_GAP_SYSTEM_PROMPT = """\
You are a financial research assistant. You will receive today's market snapshot
and context gathered from internal sources (event diary, Twitter digest, news signals).

Your job: identify which top movers or market themes are NOT adequately explained
by the existing context. For each gap, produce a targeted web search query that
would fill it.

Rules:
- Only output queries for movers/themes that truly lack explanation in the context
- If a mover's move is clearly explained by an event, earnings, or news signal
  already in the context, do NOT search for it
- Queries should be specific (include ticker, event type, date) not generic
- Return an EMPTY gaps list if the context is already sufficient
- Maximum 5 queries — prioritize the largest unexplained moves

Today's date is {today}.
"""


async def identify_context_gaps(
    market_data_text: str,
    events_context: list[dict[str, Any]],
    twitter_context: list[dict[str, Any]],
    signals_context: list[dict[str, Any]],
) -> ContextGaps:
    """Fast LLM call to identify what the gathered context doesn't explain.

    Returns targeted search queries for unexplained movers/themes, or empty
    gaps list if context is sufficient.
    """
    agent: Agent[None, ContextGaps] = Agent(
        model=create_model(),  # fast/cheap model
        system_prompt=_GAP_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=ContextGaps,
        retries=1,
    )

    events_text = _format_events_context(events_context)
    twitter_text = _format_twitter_context(twitter_context)
    signals_text = _format_signals_context(signals_context)

    prompt = f"""## MARKET SNAPSHOT
{market_data_text}

## EVENTS CONTEXT (YTD diary)
{events_text}

## TWITTER INTEL (today)
{twitter_text}

## NEWS SIGNALS (last 24hrs)
{signals_text}

Identify which top movers or themes are not explained by the context above."""

    try:
        result = await agent.run(prompt)
        gaps = result.output
        logger.info(
            "Context gap analysis complete",
            gaps_found=len(gaps.gaps),
            queries=[g.query for g in gaps.gaps],
        )
        return gaps
    except Exception:
        logger.warning("Context gap identification failed, skipping web search", exc_info=True)
        return ContextGaps(gaps=[])


async def analyze_market_brief(
    market_data_text: str,
    events_context: list[dict[str, Any]],
    twitter_context: list[dict[str, Any]],
    signals_context: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
) -> MarketBriefAnalysis | None:
    """Run LLM analysis on market data with full context.

    Returns MarketBriefAnalysis or None on failure.
    """
    agent: Agent[None, MarketBriefAnalysis] = Agent(
        model=create_model(smart=True),
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=MarketBriefAnalysis,
        retries=1,
    )

    events_text = _format_events_context(events_context)
    twitter_text = _format_twitter_context(twitter_context)
    signals_text = _format_signals_context(signals_context)
    search_text = _format_search_context(search_results)

    prompt = f"""## MARKET SNAPSHOT
{market_data_text}

## EVENTS CONTEXT (YTD diary)
{events_text}

## TWITTER INTEL (today)
{twitter_text}

## NEWS SIGNALS (last 24hrs)
{signals_text}

## WEB SEARCH CONTEXT
{search_text}

Produce the morning market brief analysis."""

    try:
        result = await agent.run(prompt)
        analysis = result.output
        logger.info(
            "Market brief analysis complete",
            drivers=len(analysis.key_drivers),
            movers=len(analysis.mover_insights),
        )
        return analysis
    except Exception:
        logger.exception("Market brief LLM analysis failed")
        return None
