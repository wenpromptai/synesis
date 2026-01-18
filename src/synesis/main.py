"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from synesis.config import get_settings
from synesis.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings = get_settings()
    setup_logging(settings)
    logger.info("Starting Synesis", env=settings.env)

    # TODO: Start Telegram listener here as background task

    yield

    logger.info("Shutting down Synesis")


app = FastAPI(
    title="Synesis",
    description="Real-time financial news analysis and prediction market trading",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness check."""
    # TODO: Check DB and Redis connections
    return {"status": "ready"}
