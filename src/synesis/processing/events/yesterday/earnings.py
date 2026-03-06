"""Earnings sub-analyzer for the yesterday brief pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import SubAnalysis

logger = get_logger(__name__)

EARNINGS_SYSTEM_PROMPT = """\
You are a senior equity analyst reviewing yesterday's earnings results for a
professional trading desk.

Analyze the earnings events and produce themed analysis. Focus on:
- EPS beat/miss magnitude and quality (one-time items vs organic)
- Revenue trends and guidance revisions (raised/lowered/maintained)
- Sector-wide patterns (are semis all beating? are retailers all missing?)
- Price reaction vs expectation (did a beat still sell off?)
- Forward guidance impact on sector multiples

Guidelines:
- Group related earnings into 1-3 themes (e.g., "AI Infrastructure Beats",
  "Consumer Weakness Deepens")
- Each theme needs a clear sentiment call
- key_takeaways: 2-3 bullets summarizing the most important conclusions
- tickers_affected: all tickers mentioned across themes
- Use category="earnings" and source="calendar" for all themes
- Be opinionated — what does this mean for the sector, not just the company

Today's date is {today}. You are analyzing earnings from yesterday.
"""


def _format_earnings_events(events: list[dict[str, Any]]) -> str:
    """Format earnings events into structured text for the LLM prompt."""
    lines: list[str] = []
    for ev in events:
        title = ev.get("title", "")
        desc = ev.get("description", "") or ""
        tickers = ev.get("tickers", [])
        ticker_str = ", ".join(f"${t}" for t in tickers[:8]) if tickers else ""
        outcome = ev.get("outcome", "")

        entry = f"- {title}"
        if ticker_str:
            entry += f" (tickers={ticker_str})"
        entry += f"\n  {desc[:300]}"
        if outcome:
            entry += f"\n  OUTCOME: {outcome}"
        lines.append(entry)

    return "\n".join(lines)


async def analyze_earnings(
    events: list[dict[str, Any]],
    market_data: str = "",
) -> SubAnalysis | None:
    """Analyze earnings events and return a SubAnalysis."""
    if not events:
        return None

    agent: Agent[None, SubAnalysis] = Agent(
        model=create_model(smart=True),
        system_prompt=EARNINGS_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=SubAnalysis,
        retries=1,
    )

    formatted = _format_earnings_events(events)
    prompt = f"Analyze these earnings results:\n\n{formatted}"
    if market_data:
        prompt = f"{market_data}\n\n{prompt}"

    try:
        result = await agent.run(prompt)
        logger.info(
            "Earnings sub-analysis complete",
            themes=len(result.output.themes),
            tickers=len(result.output.tickers_affected),
        )
        return result.output
    except Exception:
        logger.exception("Earnings sub-analysis failed")
        return None
