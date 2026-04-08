"""Intelligence pipeline endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from synesis.api.utils import create_tracked_task
from synesis.core.dependencies import AgentStateDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

router = APIRouter()

logger = get_logger(__name__)

# Hold references to background tasks so they aren't GC'd
_background_tasks: set[asyncio.Task[None]] = set()


@router.post("/trigger")
@limiter.limit("2/minute")
async def trigger_intelligence_brief(request: Request, state: AgentStateDep) -> dict[str, str]:
    """Manually trigger the daily intelligence brief."""
    trigger = state.trigger_fns.get("intelligence_brief")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Intelligence pipeline not configured (requires database + providers)",
        )

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Intelligence brief background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=exc,
            )

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Intelligence brief started in background"}
