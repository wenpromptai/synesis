"""Macro sub-analyzer for the yesterday brief pipeline.

Handles economic_data (CPI, GDP, NFP, PPI, PCE) and fed
(FOMC decisions, speeches, minutes) events as a single macro group.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import SubAnalysis

logger = get_logger(__name__)

MACRO_CATEGORIES = {"economic_data", "fed"}

MACRO_SYSTEM_PROMPT = """\
You are a senior macro strategist reviewing yesterday's economic data releases
and central bank developments for a professional trading desk.

Analyze the macro events and produce themed analysis. Focus on:
- Economic data: actual vs expected, revision direction, trend context,
  implication for Fed policy path
- Fed statements: rate action, tone (hawkish/dovish), forward guidance,
  dot plot shifts, QT pace, key language changes from prior statement
- FOMC minutes: debate highlights, dissent signals, balance of risks discussion,
  any hints about upcoming policy shifts not in the statement
- Cross-read: how does today's data change the probability distribution of
  upcoming Fed meetings?
- Duration, rates, and FX implications
- Equity sector rotation implications (growth vs value, rate-sensitive sectors)

Guidelines:
- Group into 1-3 themes (e.g., "Hawkish Fed Surprise", "Disinflation Stalls")
- Use category="macro" or "fed" as appropriate
- Use source="calendar" for all themes (these are scheduled releases)
- key_takeaways: 2-3 bullets summarizing the most important conclusions
- tickers_affected: all tickers mentioned across themes
- Be opinionated — what are the trading implications

Today's date is {today}. You are analyzing macro events from yesterday.
"""


def _format_macro_events(events: list[dict[str, Any]]) -> str:
    """Format macro calendar events for the LLM prompt."""
    lines = ['## MACRO EVENTS (tag themes driven by these as source="calendar")']
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


async def analyze_macro(
    events: list[dict[str, Any]],
    market_data: str = "",
) -> SubAnalysis | None:
    """Analyze economic_data + fed events and return a SubAnalysis."""
    if not events:
        return None

    agent: Agent[None, SubAnalysis] = Agent(
        model=create_model(smart=True),
        system_prompt=MACRO_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=SubAnalysis,
        retries=1,
    )

    formatted = _format_macro_events(events)
    prompt = f"Analyze yesterday's macro events:\n\n{formatted}"
    if market_data:
        prompt = f"{market_data}\n\n{prompt}"

    try:
        result = await agent.run(prompt)
        logger.info(
            "Macro sub-analysis complete",
            themes=len(result.output.themes),
            tickers=len(result.output.tickers_affected),
        )
        return result.output
    except Exception:
        logger.exception("Macro sub-analysis failed")
        return None
