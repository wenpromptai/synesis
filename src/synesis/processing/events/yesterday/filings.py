"""SEC filings sub-analyzer for the yesterday brief pipeline.

Handles 13F hedge fund filings.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import SubAnalysis

logger = get_logger(__name__)

FILINGS_SYSTEM_PROMPT = """\
You are a senior hedge fund analyst reviewing yesterday's 13F filings
for a professional trading desk.

You will receive 13F hedge fund quarterly position disclosures (new/exited/changed positions).

Analyze the data and produce themed analysis. Focus on:
- Major position changes, cross-fund convergence, inferred theses, contrarian signals

Guidelines:
- Group into 1-4 themes (e.g., "Smart Money Piling Into AI Infra",
  "Hedge Funds Rotating Out of Mega-Cap Tech")
- Use category="13f_filing" for all themes
- Use source="calendar" for all themes
- key_takeaways: 2-3 bullets summarizing the most important conclusions
- tickers_affected: map issuer names (e.g. "APPLE INC") to real ticker symbols
  (e.g. "AAPL"). Use your knowledge to identify the correct ticker for each issuer.
- Be opinionated — what does the positioning tell us

Today's date is {today}. You are analyzing SEC filings from yesterday.
"""


def _format_filing_briefs(filing_briefs: list[dict[str, Any]]) -> str:
    """Format 13F filing data for the LLM prompt."""
    lines: list[str] = ["## 13F FILINGS"]
    for fb in filing_briefs:
        fund = fb.get("fund_name", "Unknown")
        lines.append(f"\n### {fund}")
        if fb.get("new_positions"):
            names = [h["name_of_issuer"] for h in fb["new_positions"][:10]]
            lines.append(f"  New positions: {', '.join(names)}")
        if fb.get("exited_positions"):
            names = [h["name_of_issuer"] for h in fb["exited_positions"][:10]]
            lines.append(f"  Exited positions: {', '.join(names)}")
        if fb.get("increased"):
            names = [
                f"{h['name_of_issuer']} ({h['change_pct']:+.0f}%)" for h in fb["increased"][:10]
            ]
            lines.append(f"  Top increases: {', '.join(names)}")
        if fb.get("decreased"):
            names = [
                f"{h['name_of_issuer']} ({h['change_pct']:+.0f}%)" for h in fb["decreased"][:10]
            ]
            lines.append(f"  Top decreases: {', '.join(names)}")
        if fb.get("total_value_current"):
            lines.append(f"  Portfolio value: ${fb['total_value_current'] / 1000:.1f}M")

    return "\n".join(lines)


async def analyze_filings(
    filing_briefs: list[dict[str, Any]],
    market_data: str = "",
) -> SubAnalysis | None:
    """Analyze 13F filings."""
    if not filing_briefs:
        return None

    agent: Agent[None, SubAnalysis] = Agent(
        model=create_model(smart=True),
        system_prompt=FILINGS_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=SubAnalysis,
        retries=1,
    )

    sections: list[str] = []
    if market_data:
        sections.append(market_data)
    sections.append(_format_filing_briefs(filing_briefs))

    prompt = "Analyze yesterday's 13F filings:\n\n" + "\n\n".join(sections)

    try:
        result = await agent.run(prompt)
        logger.info(
            "Filings sub-analysis complete",
            themes=len(result.output.themes),
            tickers=len(result.output.tickers_affected),
        )
        return result.output
    except Exception:
        logger.exception("Filings sub-analysis failed")
        return None
