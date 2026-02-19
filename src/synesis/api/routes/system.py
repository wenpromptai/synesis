"""System status and config endpoints."""

from fastapi import APIRouter

from synesis.config import get_settings
from synesis.core.dependencies import AgentStateDep

router = APIRouter()


@router.get("/status")
async def system_status(state: AgentStateDep) -> dict[str, object]:
    return {
        "telegram": state.telegram_enabled,
        "reddit": state.reddit_enabled,
        "agent_running": state.agent_task is not None and not state.agent_task.done(),
    }


@router.get("/config")
async def system_config() -> dict[str, object]:
    settings = get_settings()
    return {
        "env": settings.env,
        "llm_provider": settings.llm_provider,
        "telegram_enabled": bool(settings.telegram_api_id),
        "reddit_enabled": bool(settings.reddit_subreddits),
    }
