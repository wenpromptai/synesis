"""Watchlist CRUD endpoints.

Note: Adding a ticker via API adds it to PostgreSQL but does NOT auto-subscribe
to Finnhub WebSocket — that only happens on the agent's WatchlistManager instance.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from starlette.requests import Request

from synesis.core.dependencies import DbDep
from synesis.core.rate_limit import limiter
from synesis.processing.common.watchlist import WatchlistManager

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TickerMetadataResponse(BaseModel):
    ticker: str
    source: str
    added_at: datetime
    expires_at: datetime


class AddTickerRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol (e.g. AAPL)")
    source: str = Field(default="api", description="Source identifier")


class AddTickerResponse(BaseModel):
    ticker: str
    is_new: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_manager(db: DbDep) -> WatchlistManager:
    return WatchlistManager(db)


def _record_to_response(r: dict[str, Any]) -> TickerMetadataResponse:
    return TickerMetadataResponse(
        ticker=r["ticker"],
        source=r.get("added_by", "unknown"),
        added_at=r["added_at"],
        expires_at=r["expires_at"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[str])
@limiter.limit("60/minute")
async def list_tickers(request: Request, db: DbDep) -> list[str]:
    return await _get_manager(db).get_all()


@router.get("/stats")
@limiter.limit("60/minute")
async def watchlist_stats(request: Request, db: DbDep) -> dict[str, Any]:
    return await _get_manager(db).get_stats()


@router.get("/detailed", response_model=list[TickerMetadataResponse])
@limiter.limit("60/minute")
async def list_tickers_detailed(request: Request, db: DbDep) -> list[TickerMetadataResponse]:
    metadata_list = await _get_manager(db).get_all_with_metadata()
    return [_record_to_response(r) for r in metadata_list]


@router.get("/{ticker}", response_model=TickerMetadataResponse)
@limiter.limit("60/minute")
async def get_ticker(request: Request, ticker: str, db: DbDep) -> TickerMetadataResponse:
    record = await _get_manager(db).get_metadata(ticker)
    if not record:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return _record_to_response(record)


@router.post("/", response_model=AddTickerResponse, status_code=201)
@limiter.limit("10/minute")
async def add_ticker(request: Request, body: AddTickerRequest, db: DbDep) -> AddTickerResponse:
    is_new = await _get_manager(db).add_ticker(
        ticker=body.ticker,
        source=body.source,
    )
    return AddTickerResponse(ticker=body.ticker.upper(), is_new=is_new)


@router.delete("/{ticker}", status_code=204)
@limiter.limit("10/minute")
async def remove_ticker(request: Request, ticker: str, db: DbDep) -> Response:
    removed = await _get_manager(db).remove_ticker(ticker)
    if not removed:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return Response(status_code=204)


@router.post("/cleanup", response_model=list[str])
@limiter.limit("10/minute")
async def cleanup_expired(request: Request, db: DbDep) -> list[str]:
    return await _get_manager(db).cleanup_expired()
