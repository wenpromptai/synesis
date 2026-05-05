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

    Reports whether Telegram ingestion is wired up and whether the background
    agent task (queue worker pool) is currently running.

    **Inputs:** none (query path only).

    **Returns:**
    - `telegram` (bool): True if Telegram credentials are configured.
    - `agent_running` (bool): True if the queue worker task exists and has not finished.

    **Example response:**
    ```json
    {"telegram": true, "agent_running": true}
    ```
    """
    return {
        "telegram": state.telegram_enabled,
        "agent_running": state.agent_task is not None and not state.agent_task.done(),
    }


@router.get("/config")
@limiter.limit("60/minute")
async def system_config(request: Request) -> dict[str, object]:
    """Non-secret runtime config snapshot.

    Returns environment + LLM provider + a derived flag for whether Telegram
    credentials are configured. Secrets and API keys are never returned.

    **Inputs:** none.

    **Returns:**
    - `env` (str): "development" | "staging" | "production".
    - `llm_provider` (str): "anthropic" | "openai".
    - `telegram_enabled` (bool): True if `TELEGRAM_API_ID` is set.

    **Example response:**
    ```json
    {"env": "development", "llm_provider": "openai", "telegram_enabled": true}
    ```
    """
    settings = get_settings()
    return {
        "env": settings.env,
        "llm_provider": settings.llm_provider,
        "telegram_enabled": bool(settings.telegram_api_id),
    }
