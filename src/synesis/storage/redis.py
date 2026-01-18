"""Redis client connection."""

from redis.asyncio import Redis

from synesis.core.logging import get_logger

logger = get_logger(__name__)

# Global Redis instance (initialized in lifespan)
_redis: Redis | None = None


def get_redis() -> Redis:
    """Get the global Redis instance."""
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


async def init_redis(redis_url: str) -> Redis:
    """Initialize the global Redis instance."""
    global _redis
    _redis = Redis.from_url(redis_url, decode_responses=False)
    # Test connection
    pong = _redis.ping()
    if hasattr(pong, "__await__"):
        await pong
    logger.info("Redis connected")
    return _redis


async def close_redis() -> None:
    """Close the global Redis instance."""
    global _redis
    if _redis:
        await _redis.aclose()
        logger.info("Redis disconnected")
        _redis = None
