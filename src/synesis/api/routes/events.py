"""Event Radar API endpoints."""

from __future__ import annotations

import asyncio
from datetime import date

from fastapi import APIRouter, HTTPException, Query
from starlette.requests import Request

from synesis.api.utils import create_tracked_task
from synesis.core.dependencies import AgentStateDep, DbDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter
from synesis.processing.events.models import CalendarEvent, CalendarEventRow

router = APIRouter()

logger = get_logger(__name__)

# Hold references to background tasks so they aren't GC'd
_background_tasks: set[asyncio.Task[object]] = set()


@router.get("/upcoming")
@limiter.limit("30/minute")
async def get_upcoming_events(
    request: Request,
    db: DbDep,
    days: int = Query(default=7, ge=1, le=90),
    region: str | None = Query(default=None, description="Comma-separated: US,JP,SG,HK,global"),
    category: str | None = Query(default=None),
    sector: str | None = Query(default=None),
) -> list[CalendarEventRow]:
    """Upcoming calendar events within the next N days.

    Pulls from the Event Radar table — events with `event_date` between today
    and `today + days`. The Event Radar fetcher populates this with earnings
    reports, FOMC meetings/minutes, FRED economic releases (CPI, PPI, NFP, GDP),
    13F filing windows, and conferences.

    **Query params:**
    - `days` (int, 1–90, default 7): lookahead window.
    - `region` (str, optional): comma-separated codes from
      `US` | `JP` | `SG` | `HK` | `global`. Matches if any region overlaps.
    - `category` (str, optional): one of
      `earnings` | `economic_data` | `fed` | `13f_filing` | `conference`
      | `release` | `regulatory` | `other`.
    - `sector` (str, optional): one of
      `ai` | `semiconductors` | `ai_infrastructure` | `power` | `energy` | `precious_metals`.

    **Returns:** `list[CalendarEventRow]` — each event has `id`, `title`,
    `description`, `event_date`, `event_end_date` (multi-day events),
    `category`, `sector`, `region`, `tickers`, `source_urls`, `time_label`
    ("AH"/"PM"/"DM"), `discovered_at`, `updated_at`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/events/upcoming?days=14&region=US&category=earnings'
    # → list of upcoming earnings reports (NVDA, AAPL...) for the next 14 days
    ```
    """
    region_list = [r.strip() for r in region.split(",")] if region else None
    rows = await db.get_upcoming_events(
        days,
        region=region_list,
        category=category,
        sector=sector,
    )
    return [_row_to_model(r) for r in rows]


@router.get("/calendar")
@limiter.limit("30/minute")
async def get_calendar(
    request: Request,
    db: DbDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date = Query(..., description="End date (YYYY-MM-DD)"),
) -> list[CalendarEventRow]:
    """All Event Radar events between two dates.

    Same row shape as `/upcoming` but takes an absolute date range instead of
    a lookahead window. Useful for pulling, e.g., "everything in May" — earnings
    reports, FOMC dates, CPI/NFP releases, etc.

    **Query params:**
    - `start` (YYYY-MM-DD, required): inclusive start.
    - `end` (YYYY-MM-DD, required): inclusive end.

    The window cannot exceed 90 days.

    **Returns:** `list[CalendarEventRow]` — see `/upcoming` for field details.

    **Errors:**
    - `400` if `end - start > 90 days`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/events/calendar?start=2026-05-01&end=2026-05-31'
    ```
    """
    if (end - start).days > 90:
        raise HTTPException(status_code=400, detail="Date range cannot exceed 90 days")
    rows = await db.get_events_by_date_range(start, end)
    return [_row_to_model(r) for r in rows]


@router.get("/{event_id}")
@limiter.limit("30/minute")
async def get_event(
    request: Request,
    db: DbDep,
    event_id: int,
) -> CalendarEventRow:
    """Fetch a single event by its DB primary key.

    **Path params:**
    - `event_id` (int): row id from the events table.

    **Returns:** `CalendarEventRow`.

    **Errors:**
    - `404` if event not found.

    **Example:** `curl http://localhost:7337/api/v1/events/12345`
    """
    row = await db.get_event_by_id(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _row_to_model(row)


@router.post("/discover")
@limiter.limit("5/minute")
async def trigger_discovery(
    request: Request,
    state: AgentStateDep,
) -> dict[str, str]:
    """Manually trigger the Event Radar discovery pipeline.

    Runs the full structured-fetch job (FRED release dates, NASDAQ earnings,
    SEC filings, central-bank schedules) and upserts results into the events
    table. Equivalent to the 6pm ET cron job, runs in background.

    **Inputs:** none.

    **Returns:**
    - `status` (str): `"triggered"`.
    - `message` (str): confirmation.

    **Errors:**
    - `503` if database isn't initialized.

    **Example:** `curl -X POST http://localhost:7337/api/v1/events/discover`
    """
    trigger = state.trigger_fns.get("event_discover")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Event Radar not configured (requires database)",
        )

    def _on_done(t: asyncio.Task[object]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Event discovery background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        else:
            logger.info("Event discovery completed", events_stored=t.result())

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Event discovery pipeline started in background"}


@router.post("/digest")
@limiter.limit("5/minute")
async def trigger_digest(
    request: Request,
    state: AgentStateDep,
) -> dict[str, str]:
    """Manually trigger the "What's Coming" event digest.

    Same job as the 7pm ET cron — pulls upcoming events, formats a Discord
    embed grouping by date/category, and posts to the brief webhook. Runs
    in background.

    **Inputs:** none.

    **Returns:**
    - `status` (str): `"triggered"`.
    - `message` (str): confirmation.

    **Errors:**
    - `503` if database isn't initialized.

    **Example:** `curl -X POST http://localhost:7337/api/v1/events/digest`
    """
    trigger = state.trigger_fns.get("event_digest")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Event digest not configured (requires database)",
        )

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Event digest background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Event digest started in background"}


@router.post("")
@limiter.limit("5/minute")
async def add_event(
    request: Request,
    db: DbDep,
    event: CalendarEvent,
) -> dict[str, str | int | None]:
    """Insert or upsert a manually-curated calendar event.

    Useful for ad-hoc additions (analyst days, conferences) the structured
    fetch doesn't pick up automatically. Uniqueness is enforced server-side
    by `(source, source_id)` — a duplicate (source, source_id) updates rather
    than re-inserts.

    **Body (JSON):** a `CalendarEvent` —
    - `title` (str, required).
    - `event_date` (YYYY-MM-DD, required): the calendar date.
    - `event_end_date` (YYYY-MM-DD, optional): for multi-day events.
    - `category` (required): `earnings` | `economic_data` | `fed` | `13f_filing` |
      `conference` | `release` | `regulatory` | `other`.
    - `region` (list[str], required): one or more of `US` | `JP` | `SG` | `HK` | `global`.
    - `sector` (str, optional): `ai` | `semiconductors` | `ai_infrastructure` |
      `power` | `energy` | `precious_metals`.
    - `tickers` (list[str], default `[]`): related ticker symbols.
    - `source_urls` (list[str], default `[]`): source URLs for this event.
    - `description` (str, optional): free-text description.
    - `time_label` (str, optional): `AH` (after-hours) | `PM` (pre-market) | `DM` (during-market).

    **Returns:**
    - `status` (str): `"created"`.
    - `id` (int | null): row id of the upserted event.

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/events \\
      -H "Content-Type: application/json" \\
      -d '{
        "title":"NVDA GTC Keynote",
        "event_date":"2026-06-09",
        "category":"conference",
        "region":["US"],
        "tickers":["NVDA"],
        "source_urls":["https://www.nvidia.com/gtc"]
      }'
    ```
    """
    event_id = await db.upsert_calendar_event(event)
    return {"status": "created", "id": event_id}


def _row_to_model(row: object) -> CalendarEventRow:
    """Convert an asyncpg.Record to CalendarEventRow."""
    # asyncpg.Record supports dict-like access
    r = dict(row)  # type: ignore[call-overload]
    return CalendarEventRow.model_validate(r)
