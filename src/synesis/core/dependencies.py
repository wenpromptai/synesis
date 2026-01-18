"""FastAPI dependencies for dependency injection."""

from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from synesis.config import Settings, get_settings
from synesis.core.events import EventBus
from synesis.storage.database import Database, get_database
from synesis.storage.redis import get_redis

# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_db() -> Database:
    """Get database dependency."""
    return get_database()


async def get_event_bus(
    redis: Annotated[Redis, Depends(get_redis)],
) -> EventBus:
    """Get event bus dependency."""
    return EventBus(redis)


# Annotated dependencies for use in route handlers
DbDep = Annotated[Database, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]
EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
