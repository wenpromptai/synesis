"""Yesterday brief orchestrator — splits events by type, runs sub-analyzers in parallel."""

from __future__ import annotations

import asyncio
from typing import Any

from synesis.core.logging import get_logger
from synesis.processing.events.models import SubAnalysis, YesterdayBriefAnalysis
from synesis.processing.events.yesterday.consolidator import consolidate
from synesis.processing.events.yesterday.earnings import analyze_earnings
from synesis.processing.events.yesterday.events import analyze_events
from synesis.processing.events.yesterday.filings import analyze_filings
from synesis.processing.events.yesterday.macro import MACRO_CATEGORIES, analyze_macro

logger = get_logger(__name__)


async def synthesize_yesterday_brief(
    yesterday_events: list[dict[str, Any]],
    surprise_events: list[dict[str, Any]],
    filing_briefs: list[dict[str, Any]],
    market_data: str = "",
) -> YesterdayBriefAnalysis | None:
    """Synthesize yesterday's events into an expert backward-looking analysis.

    Splits events by type, runs focused sub-analyzers in parallel (Sonnet tier),
    then consolidates via a final LLM call (Opus tier).

    Returns YesterdayBriefAnalysis or None on failure.
    """
    if not yesterday_events and not surprise_events and not filing_briefs:
        return None

    # Split calendar events into earnings, macro, and other
    earnings_events = [e for e in yesterday_events if e.get("category") == "earnings"]
    macro_events = [e for e in yesterday_events if e.get("category") in MACRO_CATEGORIES]
    other_events = [
        e
        for e in yesterday_events
        if e.get("category") not in {"earnings", "13f_filing"}
        and e.get("category") not in MACRO_CATEGORIES
    ]

    # Run sub-analyzers in parallel (skip if no data)
    tasks: list[asyncio.Task[SubAnalysis | None]] = []

    if earnings_events:
        tasks.append(asyncio.create_task(analyze_earnings(earnings_events, market_data)))
    if macro_events:
        tasks.append(asyncio.create_task(analyze_macro(macro_events, market_data)))
    if other_events or surprise_events:
        tasks.append(
            asyncio.create_task(analyze_events(other_events, surprise_events, market_data))
        )
    if filing_briefs:
        tasks.append(asyncio.create_task(analyze_filings(filing_briefs, market_data)))

    if not tasks:
        return None

    results = await asyncio.gather(*tasks)
    sub_analyses = [r for r in results if r is not None]

    if not sub_analyses:
        logger.warning("All sub-analyzers returned None")
        return None

    logger.info("Sub-analyses complete", count=len(sub_analyses))

    # Consolidate into final analysis
    return await consolidate(sub_analyses, market_data)
