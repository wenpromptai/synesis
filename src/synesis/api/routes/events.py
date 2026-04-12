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
    """Get upcoming events within N days."""
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
    """Get all events in a date range."""
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
    """Get a single event by ID."""
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
    """Manually trigger the full event discovery pipeline."""
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
    """Manually trigger the event digest (What's Coming)."""
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
    """Manually add a calendar event."""
    event_id = await db.upsert_calendar_event(event)
    return {"status": "created", "id": event_id}


def _row_to_model(row: object) -> CalendarEventRow:
    """Convert an asyncpg.Record to CalendarEventRow."""
    # asyncpg.Record supports dict-like access
    r = dict(row)  # type: ignore[call-overload]
    return CalendarEventRow.model_validate(r)
