"""FastAPI dependencies for dependency injection."""

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis

from synesis.agent import AgentState
from synesis.config import Settings, get_settings
from synesis.providers.factset.client import get_factset_client
from synesis.providers.factset.provider import FactSetProvider
from synesis.storage.database import Database, get_database
from synesis.storage.redis import get_redis

# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_db() -> Database:
    """Get database dependency."""
    return get_database()


async def get_agent_state(request: Request) -> AgentState:
    """Get AgentState from app.state (set during lifespan)."""
    return request.app.state.agent  # type: ignore[no-any-return]


def get_factset_provider() -> FactSetProvider:
    """Get FactSet provider with singleton client."""
    return FactSetProvider(client=get_factset_client())


# Annotated dependencies for use in route handlers
DbDep = Annotated[Database, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]
AgentStateDep = Annotated[AgentState, Depends(get_agent_state)]
FactSetProviderDep = Annotated[FactSetProvider, Depends(get_factset_provider)]
