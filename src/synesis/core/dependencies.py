"""FastAPI dependencies for dependency injection."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from redis.asyncio import Redis

from synesis.agent import AgentState
from synesis.config import Settings, get_settings
from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
from synesis.providers.finnhub.prices import FinnhubPriceProvider, get_price_service
from synesis.providers.nasdaq import NasdaqClient
from synesis.providers.sec_edgar import SECEdgarClient
from synesis.storage.database import Database, get_database
from synesis.storage.redis import get_redis

# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Module-level singletons (initialised lazily on first use)
_sec_edgar_client: SECEdgarClient | None = None
_nasdaq_client: NasdaqClient | None = None
_crawler: Crawl4AICrawlerProvider | None = None


def get_db() -> Database:
    """Get database dependency."""
    return get_database()


async def get_agent_state(request: Request) -> AgentState:
    """Get AgentState from app.state (set during lifespan)."""
    return request.app.state.agent  # type: ignore[no-any-return]


def get_price_provider() -> FinnhubPriceProvider:
    """Get the global Finnhub price service."""
    try:
        return get_price_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503, detail="Price service not available (no FINNHUB_API_KEY)"
        )


async def get_sec_edgar_client(redis: Redis = Depends(get_redis)) -> SECEdgarClient:
    """Get or create singleton SEC EDGAR client."""
    global _sec_edgar_client
    if _sec_edgar_client is None:
        _sec_edgar_client = SECEdgarClient(redis=redis)
    return _sec_edgar_client


async def get_nasdaq_client(redis: Redis = Depends(get_redis)) -> NasdaqClient:
    """Get or create singleton NASDAQ client."""
    global _nasdaq_client
    if _nasdaq_client is None:
        _nasdaq_client = NasdaqClient(redis=redis)
    return _nasdaq_client


def get_crawler() -> Crawl4AICrawlerProvider:
    """Get or create singleton Crawl4AI crawler."""
    global _crawler
    if _crawler is None:
        _crawler = Crawl4AICrawlerProvider()
    return _crawler


# Annotated dependencies for use in route handlers
DbDep = Annotated[Database, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]
AgentStateDep = Annotated[AgentState, Depends(get_agent_state)]
PriceServiceDep = Annotated[FinnhubPriceProvider, Depends(get_price_provider)]
SECEdgarClientDep = Annotated[SECEdgarClient, Depends(get_sec_edgar_client)]
NasdaqClientDep = Annotated[NasdaqClient, Depends(get_nasdaq_client)]
Crawl4AICrawlerDep = Annotated[Crawl4AICrawlerProvider, Depends(get_crawler)]
