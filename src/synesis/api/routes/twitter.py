"""Twitter agent endpoints."""

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


@router.post("/analyze")
@limiter.limit("5/minute")
async def trigger_twitter_agent(request: Request, state: AgentStateDep) -> dict[str, str]:
    """Manually trigger the daily Twitter agent.

    Fires the same job the scheduler runs at 10:00 ET. Pulls recent tweets from
    accounts in `TWITTER_ACCOUNTS`, runs them through the LLM digest, and posts
    to the Twitter Discord webhook. Runs in the background — returns immediately.

    **Inputs:** none.

    **Returns:**
    - `status` (str): `"triggered"` on success.
    - `message` (str): human-readable confirmation.

    **Errors:**
    - `503` if `TWITTERAPI_API_KEY` is missing or `TWITTER_ACCOUNTS` is empty
      (the trigger isn't registered at startup in that case).

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/twitter/analyze
    # {"status":"triggered","message":"Twitter data collection job started in background"}
    ```
    """
    trigger = state.trigger_fns.get("twitter_agent")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Twitter agent not configured (missing TWITTERAPI_API_KEY or TWITTER_ACCOUNTS)",
        )

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Twitter agent background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Twitter data collection job started in background"}
