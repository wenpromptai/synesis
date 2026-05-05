"""System status and config endpoints."""

from fastapi import APIRouter
from starlette.requests import Request

from synesis.config import get_settings
from synesis.core.dependencies import AgentStateDep
from synesis.core.rate_limit import limiter

router = APIRouter()


@router.get("/status")
@limiter.limit("60/minute")
async def system_status(request: Request, state: AgentStateDep) -> dict[str, object]:
    """Live agent runtime status.

    **Inputs:** none (query path only).

    **Returns:**
    - `db_enabled` (bool): True if a database connection is active.
    - `scheduler_running` (bool): True if the APScheduler is running.

    **Example response:**
    ```json
    {"db_enabled": true, "scheduler_running": true}
    ```
    """
    return {
        "db_enabled": state.db_enabled,
        "scheduler_running": state.scheduler is not None and state.scheduler.running,
    }


@router.get("/config")
@limiter.limit("60/minute")
async def system_config(request: Request) -> dict[str, object]:
    """Non-secret runtime config snapshot.

    **Inputs:** none.

    **Returns:**
    - `env` (str): "development" | "staging" | "production".
    - `llm_provider` (str): "anthropic" | "openai".

    **Example response:**
    ```json
    {"env": "development", "llm_provider": "openai"}
    ```
    """
    settings = get_settings()
    return {
        "env": settings.env,
        "llm_provider": settings.llm_provider,
    }
