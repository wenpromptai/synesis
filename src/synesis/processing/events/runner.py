"""Event Radar pipeline runner — orchestrates structured API fetching and storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.processing.events.fetchers import (
    fetch_13f_events,
    fetch_fomc_events,
    fetch_fred_macro_events,
    fetch_nasdaq_earnings_events,
)
from synesis.processing.events.models import CalendarEvent

if TYPE_CHECKING:
    from redis.asyncio import Redis

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
    """Fetch events from structured APIs (FRED, NASDAQ earnings, FOMC, SEC 13F).

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

    try:
        events = await fetch_fomc_events()
        all_events.extend(events)
    except Exception:
        logger.exception("FOMC calendar fetch failed")

    if not all_events:
        logger.info("No structured events found")
        return 0

    stored = 0
    failed = 0
    for event in all_events:
        try:
            result = await db.upsert_calendar_event(event)
            if result is not None:
                stored += 1
        except Exception:
            failed += 1
            logger.exception(
                "Failed to upsert calendar event",
                event_title=event.title,
                event_date=str(event.event_date),
                event_category=event.category,
            )
    if failed:
        logger.error(
            "Some events failed to store", failed=failed, stored=stored, total=len(all_events)
        )
    logger.info("Events stored", total=len(all_events), new=stored)
    return stored
