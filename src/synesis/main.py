"""FastAPI application entry point."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from synesis.agent import agent_lifespan
from synesis.api import api_router
from synesis.config import get_settings
from synesis.core.dependencies import AgentStateDep
from synesis.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — starts the agent pipeline alongside the HTTP server."""
    settings = get_settings()
    setup_logging(settings)
    shutdown_event = asyncio.Event()

    async with agent_lifespan(settings, shutdown_event) as state:
        app.state.agent = state
        app.state.shutdown_event = shutdown_event
        logger.info("Synesis ready", env=settings.env)
        yield
        # FastAPI is shutting down — signal the agent
        shutdown_event.set()


app = FastAPI(
    title="Synesis",
    description="Real-time financial news analysis and prediction market trading",
    version="0.1.0",
    lifespan=lifespan,
)

# Infrastructure (no prefix, not versioned)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check — always ok if process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(state: AgentStateDep) -> dict[str, str]:
    """Readiness check — verifies infrastructure is connected."""
    checks: dict[str, str] = {}
    try:
        await state.redis.ping()  # type: ignore[misc]
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"
    checks["db"] = "ok" if state.db else "disabled"
    checks["agent"] = "ok" if state.agent_task and not state.agent_task.done() else "error"
    status = "ready" if all(v != "error" for v in checks.values()) else "not_ready"
    return {"status": status, **checks}


# Domain API
app.include_router(api_router, prefix="/api/v1")
