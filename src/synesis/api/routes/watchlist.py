"""Watchlist CRUD endpoints.

Note: Adding a ticker via API adds it to Redis/PG but does NOT auto-subscribe
to Finnhub WebSocket — that only happens on the agent's WatchlistManager instance.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from starlette.requests import Request

from synesis.core.dependencies import DbDep, RedisDep
from synesis.core.rate_limit import limiter
from synesis.processing.common.watchlist import TickerMetadata, WatchlistManager

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models (TickerMetadata is a dataclass, so wrap for serialization)
# ---------------------------------------------------------------------------


class TickerMetadataResponse(BaseModel):
    ticker: str
    source: str
    added_at: datetime
    last_seen_at: datetime
    mention_count: int = 1


class AddTickerRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol (e.g. AAPL)")
    source: str = Field(default="api", description="Source identifier")


class AddTickerResponse(BaseModel):
    ticker: str
    is_new: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_manager(redis: RedisDep, db: DbDep) -> WatchlistManager:
    return WatchlistManager(redis=redis, db=db)


def _metadata_to_response(m: TickerMetadata) -> TickerMetadataResponse:
    return TickerMetadataResponse(
        ticker=m.ticker,
        source=m.source,
        added_at=m.added_at,
        last_seen_at=m.last_seen_at,
        mention_count=m.mention_count,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[str])
@limiter.limit("60/minute")
async def list_tickers(request: Request, redis: RedisDep, db: DbDep) -> list[str]:
    return await _get_manager(redis, db).get_all()


@router.get("/stats")
@limiter.limit("60/minute")
async def watchlist_stats(
    request: Request, redis: RedisDep, db: DbDep
) -> dict[str, int | dict[str, int]]:
    return await _get_manager(redis, db).get_stats()


@router.get("/detailed", response_model=list[TickerMetadataResponse])
@limiter.limit("60/minute")
async def list_tickers_detailed(
    request: Request, redis: RedisDep, db: DbDep
) -> list[TickerMetadataResponse]:
    metadata_list = await _get_manager(redis, db).get_all_with_metadata()
    return [_metadata_to_response(m) for m in metadata_list]


@router.get("/{ticker}", response_model=TickerMetadataResponse)
@limiter.limit("60/minute")
async def get_ticker(
    request: Request, ticker: str, redis: RedisDep, db: DbDep
) -> TickerMetadataResponse:
    m = await _get_manager(redis, db).get_metadata(ticker)
    if not m:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return _metadata_to_response(m)


@router.post("/", response_model=AddTickerResponse, status_code=201)
@limiter.limit("10/minute")
async def add_ticker(
    request: Request, body: AddTickerRequest, redis: RedisDep, db: DbDep
) -> AddTickerResponse:
    is_new = await _get_manager(redis, db).add_ticker(
        ticker=body.ticker,
        source=body.source,
    )
    return AddTickerResponse(ticker=body.ticker.upper(), is_new=is_new)


@router.delete("/{ticker}", status_code=204)
@limiter.limit("10/minute")
async def remove_ticker(request: Request, ticker: str, redis: RedisDep, db: DbDep) -> Response:
    removed = await _get_manager(redis, db).remove_ticker(ticker)
    if not removed:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return Response(status_code=204)


@router.post("/cleanup", response_model=list[str])
@limiter.limit("10/minute")
async def cleanup_expired(request: Request, redis: RedisDep, db: DbDep) -> list[str]:
    return await _get_manager(redis, db).cleanup_expired()
