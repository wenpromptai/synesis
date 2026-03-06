"""General events/surprise sub-analyzer for the yesterday brief pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import SubAnalysis

logger = get_logger(__name__)

EVENTS_SYSTEM_PROMPT = """\
You are a senior strategist reviewing yesterday's non-earnings, non-macro events
for a professional trading desk.

Analyze the calendar events and surprise developments. Focus on:
- AI model releases: ANY new model from ANY lab — benchmark leaps, capability jumps,
  pricing disruption, competitive positioning, infra demand implications
- Major announcements: billion-dollar investments (e.g. NVIDIA investing in LITE/COHR),
  strategic partnerships, joint ventures, new business initiatives, supply deals
- New technology/products: chip launches, platform releases, breakthrough tech demos
- Regulatory actions: scope, enforcement trend, affected companies
- Conferences: key announcements, guidance changes, strategic pivots
- Corporate actions: M&A implications, buyback signals, management changes
- Surprise events: unscheduled developments that moved markets

Guidelines:
- Group into 1-4 themes (e.g., "AI Product Cycle", "Energy Supply Shock")
- Calendar events use source="calendar", surprises use source="surprise"
- Use appropriate category (tech, corporate, sector, regulatory)
- key_takeaways: 2-3 bullets summarizing the most important conclusions
- tickers_affected: all tickers mentioned across themes
- Be opinionated — what are the trading implications

Today's date is {today}. You are analyzing events from yesterday.
"""


def _format_calendar_events(events: list[dict[str, Any]]) -> str:
    """Format non-earnings calendar events for the LLM prompt."""
    lines = ['## CALENDAR EVENTS (tag themes driven by these as source="calendar")']
    for ev in events:
        title = ev.get("title", "")
        desc = ev.get("description", "") or ""
        tickers = ev.get("tickers", [])
        ticker_str = ", ".join(f"${t}" for t in tickers[:8]) if tickers else ""
        category = ev.get("category", "other")

        entry = f"- {title} (category={category}"
        if ticker_str:
            entry += f", tickers={ticker_str}"
        entry += f")\n  {desc[:300]}"

        outcome = ev.get("outcome", "")
        if outcome:
            entry += f"\n  OUTCOME: {outcome}"
        lines.append(entry)

    return "\n".join(lines)


def _format_surprise_events(surprises: list[dict[str, Any]]) -> str:
    """Format surprise events for the LLM prompt."""
    lines = ['## SURPRISE EVENTS (tag themes driven by these as source="surprise")']
    for ev in surprises:
        title = ev.get("title", "")
        snippet = ev.get("snippet", "") or ""
        url = ev.get("url", "")
        entry = f"- {title}\n  {snippet[:300]}"
        if url:
            entry += f"\n  Source: {url}"
        lines.append(entry)

    return "\n".join(lines)


async def analyze_events(
    calendar_events: list[dict[str, Any]],
    surprise_events: list[dict[str, Any]],
    market_data: str = "",
) -> SubAnalysis | None:
    """Analyze non-earnings calendar + surprise events and return a SubAnalysis."""
    if not calendar_events and not surprise_events:
        return None

    agent: Agent[None, SubAnalysis] = Agent(
        model=create_model(smart=True),
        system_prompt=EVENTS_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=SubAnalysis,
        retries=1,
    )

    sections: list[str] = []
    if market_data:
        sections.append(market_data)
    if calendar_events:
        sections.append(_format_calendar_events(calendar_events))
    if surprise_events:
        sections.append(_format_surprise_events(surprise_events))

    prompt = "Analyze yesterday's events:\n\n" + "\n\n".join(sections)

    try:
        result = await agent.run(prompt)
        logger.info(
            "Events sub-analysis complete",
            themes=len(result.output.themes),
            tickers=len(result.output.tickers_affected),
        )
        return result.output
    except Exception:
        logger.exception("Events sub-analysis failed")
        return None
