"""Consolidator for yesterday brief sub-analyses into final YesterdayBriefAnalysis."""

from __future__ import annotations

from datetime import date

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import SubAnalysis, YesterdayBriefAnalysis

logger = get_logger(__name__)

CONSOLIDATION_SYSTEM_PROMPT = """\
You are a senior market strategist producing the final end-of-day review for a
professional trading desk. You will receive pre-analyzed themes and takeaways
from specialist analysts covering earnings, macro/events, and 13F filings.

Your job is to:
1. Merge and re-rank the themes — combine related themes across categories if
   they tell a unified story (e.g., "hot CPI + hawkish Fed" = one rates theme)
2. Produce a compelling headline summarizing yesterday's session
3. Fill market_snapshot using the MARKET DATA provided
   - Data fields (equities, rates_fx, commodities, volatility, sector_performance)
     must be terse: "SPY -0.8%, QQQ -1.2%" — NO commentary in data fields.
     All interpretation belongs in the summary field.
4. Write synthesis connecting the dots across all themes
5. Generate 2-5 specific, opinionated actionables with direction, tickers,
   rationale, and timeframe
6. Identify 2-3 key risks to monitor
7. List 5-10 top movers from across all themes

Guidelines:
- Preserve the specialist analysts' theme quality — refine, don't rewrite
- Cross-reference across categories for deeper insight
- Be opinionated — give expert analysis, not just summaries
- source: preserve each theme's original source tag (calendar/analysis)

Today's date is {today}. You are analyzing events from yesterday.
"""


def _format_sub_analyses(sub_analyses: list[SubAnalysis]) -> str:
    """Format sub-analyses into structured text for the consolidation prompt."""
    sections: list[str] = []

    for i, sa in enumerate(sub_analyses, 1):
        lines = [f"## SUB-ANALYSIS {i}"]
        lines.append(f"Key takeaways: {'; '.join(sa.key_takeaways)}")
        lines.append(f"Tickers affected: {', '.join(sa.tickers_affected)}")
        lines.append("")

        for theme in sa.themes:
            lines.append(f"### Theme: {theme.title}")
            lines.append(f"  Category: {theme.category}")
            lines.append(f"  Sentiment: {theme.sentiment}")
            lines.append(f"  Source: {theme.source}")
            lines.append(f"  Outcome: {theme.outcome}")
            lines.append(f"  Analysis: {theme.analysis}")
            if theme.key_events:
                lines.append(f"  Key events: {'; '.join(theme.key_events)}")
            if theme.tickers:
                lines.append(f"  Tickers: {', '.join(theme.tickers)}")
            if theme.market_reaction:
                lines.append(f"  Market reaction: {theme.market_reaction}")
            lines.append("")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


async def consolidate(
    sub_analyses: list[SubAnalysis],
    market_data: str = "",
) -> YesterdayBriefAnalysis | None:
    """Consolidate sub-analyses into a final YesterdayBriefAnalysis."""
    if not sub_analyses:
        return None

    agent: Agent[None, YesterdayBriefAnalysis] = Agent(
        model=create_model(tier="vsmart"),
        system_prompt=CONSOLIDATION_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=YesterdayBriefAnalysis,
        retries=1,
    )

    formatted = _format_sub_analyses(sub_analyses)
    prompt = "Consolidate these specialist analyses into a final end-of-day review:\n\n"
    if market_data:
        prompt += f"{market_data}\n\n"
    prompt += formatted

    try:
        result = await agent.run(prompt)
        analysis = result.output
        logger.info(
            "Yesterday brief consolidation complete",
            themes=len(analysis.themes),
            actionables=len(analysis.actionables),
            top_movers=len(analysis.top_movers),
        )
        return analysis
    except Exception:
        logger.exception("Yesterday brief consolidation failed")
        return None
