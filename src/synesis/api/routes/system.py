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
    return {
        "telegram": state.telegram_enabled,
        "agent_running": state.agent_task is not None and not state.agent_task.done(),
    }


@router.get("/config")
@limiter.limit("60/minute")
async def system_config(request: Request) -> dict[str, object]:
    settings = get_settings()
    return {
        "env": settings.env,
        "llm_provider": settings.llm_provider,
        "telegram_enabled": bool(settings.telegram_api_id),
    }
