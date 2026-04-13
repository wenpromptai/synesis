"""Intelligence pipeline endpoints."""

import asyncio
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request

from synesis.api.utils import create_tracked_task
from synesis.core.dependencies import (
    AgentStateDep,
    Crawl4AICrawlerDep,
    MassiveClientDep,
    SECEdgarClientDep,
    YFinanceClientDep,
)
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

router = APIRouter()

logger = get_logger(__name__)

# Hold references to background tasks so they aren't GC'd
_background_tasks: set[asyncio.Task[None]] = set()


_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,14}$")


class AnalyzeRequest(BaseModel):
    """Request body for the ticker analysis endpoint."""

    tickers: list[str] = Field(min_length=1, max_length=10)

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(t.upper().strip() for t in v))  # dedup, preserve order
        for t in cleaned:
            if not _TICKER_RE.match(t):
                raise ValueError(f"Invalid ticker: {t!r}")
        return cleaned


@router.post("/trigger")
@limiter.limit("2/minute")
async def trigger_scan_brief(request: Request, state: AgentStateDep) -> dict[str, str]:
    """Manually trigger the daily scan pipeline (macro + watchlist to Discord)."""
    trigger = state.trigger_fns.get("scan_brief")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Scan pipeline not configured (requires database + providers)",
        )

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Scan brief background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=exc,
            )

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Scan brief started in background"}


@router.post("/analyze")
@limiter.limit("2/minute")
async def analyze_tickers(
    request: Request,
    body: AnalyzeRequest,
    sec_edgar: SECEdgarClientDep,
    yfinance: YFinanceClientDep,
    massive: MassiveClientDep,
    crawler: Crawl4AICrawlerDep,
) -> dict[str, Any]:
    """Run deep analysis pipeline for specific tickers.

    Runs company/price analysis, bull/bear debate, and trader for each ticker.
    Saves brief to KG. Returns full analysis results.

    May take several minutes for multiple tickers.
    """
    from synesis.config import get_settings
    from synesis.processing.intelligence.job import run_ticker_analysis

    settings = get_settings()
    twitter_key = (
        settings.twitterapi_api_key.get_secret_value() if settings.twitterapi_api_key else None
    )

    try:
        result = await run_ticker_analysis(
            tickers=body.tickers,
            sec_edgar=sec_edgar,
            yfinance=yfinance,
            massive=massive,
            crawler=crawler,
            twitter_api_key=twitter_key,
        )
    except Exception:
        logger.exception("Ticker analysis failed", tickers=body.tickers)
        raise HTTPException(status_code=500, detail="Ticker analysis failed")
    if not result.get("date"):
        raise HTTPException(status_code=500, detail="Analysis produced no results")
    return result
