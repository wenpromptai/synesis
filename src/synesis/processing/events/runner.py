"""Event Radar pipeline runner — orchestrates crawling, extraction, dedup, and storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.processing.events.crawler import (
    crawl_curated_sources,
    fetch_13f_events,
    fetch_fred_macro_events,
    fetch_nasdaq_earnings_events,
)
from synesis.processing.events.dedup import deduplicate_and_store
from synesis.processing.events.extractor import extract_events_from_markdown
from synesis.processing.events.models import CalendarEvent

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred import FREDClient
    from synesis.providers.nasdaq import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


async def run_structured_sources(
    db: Database,
    redis: Redis,
    fred: FREDClient | None = None,
    nasdaq: NasdaqClient | None = None,
    sec_edgar: SECEdgarClient | None = None,
) -> int:
    """Fetch events from structured APIs (FRED, NASDAQ earnings, SEC 13F).

    Returns total events stored.
    """
    all_events: list[CalendarEvent] = []

    if fred:
        try:
            events = await fetch_fred_macro_events(fred)
            all_events.extend(events)
        except Exception:
            logger.exception("FRED macro events failed")

    if nasdaq:
        try:
            events = await fetch_nasdaq_earnings_events(nasdaq)
            all_events.extend(events)
        except Exception:
            logger.exception("NASDAQ earnings events failed")

    if sec_edgar:
        try:
            events = await fetch_13f_events(sec_edgar, redis)
            all_events.extend(events)
        except Exception:
            logger.exception("SEC 13F events failed")

    if not all_events:
        logger.info("No structured events found")
        return 0

    return await deduplicate_and_store(all_events, db)


async def run_curated_sources(
    db: Database,
    redis: Redis,
    crawler: Crawl4AICrawlerProvider,
    force: bool = False,
) -> int:
    """Crawl curated sources, extract events with LLM, store.

    Returns total events stored.
    """
    crawled = await crawl_curated_sources(crawler, redis, force=force)
    if not crawled:
        logger.info("No curated sources crawled or all empty")
        return 0

    all_events: list[CalendarEvent] = []
    for source_config, result in crawled:
        try:
            events = await extract_events_from_markdown(
                markdown=result.markdown,
                source_url=source_config["url"],
                source_name=source_config.get("name", ""),
                default_region=source_config.get("region", "US"),
                default_tickers=source_config.get("tickers"),
            )
            all_events.extend(events)
        except Exception:
            logger.exception(
                "Event extraction failed for source",
                source=source_config.get("name"),
            )

    if not all_events:
        logger.info("No events extracted from curated sources")
        return 0

    return await deduplicate_and_store(all_events, db)


async def run_full_discovery(
    db: Database,
    redis: Redis,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
    nasdaq: NasdaqClient | None = None,
    sec_edgar: SECEdgarClient | None = None,
) -> dict[str, int]:
    """Run complete event discovery pipeline (structured + curated).

    Returns dict with counts per source type.
    """
    logger.info("Starting full event discovery pipeline")

    structured = await run_structured_sources(
        db, redis, fred=fred, nasdaq=nasdaq, sec_edgar=sec_edgar
    )
    curated = 0
    if crawler:
        try:
            curated = await run_curated_sources(db, redis, crawler, force=True)
        except Exception:
            logger.exception("Curated sources failed, structured results preserved")

    totals = {
        "structured": structured,
        "curated": curated,
        "total": structured + curated,
    }
    logger.info("Event discovery complete", **totals)
    return totals
