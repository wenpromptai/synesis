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
    """List all active watchlist tickers as plain symbols.

    **Inputs:** none.

    **Returns:** `list[str]` — uppercase tickers, e.g. `["NVDA", "AAPL", "MSFT"]`.

    **Example:** `curl http://localhost:7337/api/v1/watchlist/`
    """
    return await _get_manager(db).get_all()


@router.get("/stats")
@limiter.limit("60/minute")
async def watchlist_stats(request: Request, db: DbDep) -> dict[str, Any]:
    """Aggregate stats about the watchlist.

    **Inputs:** none.

    **Returns (object):**
    - `total_tickers` (int): count of active rows.
    - `sources` (dict[str, int]): counts grouped by who/what added each ticker
      (e.g. `{"api": 5, "telegram": 3, "intelligence": 8}`).

    **Example:** `curl http://localhost:7337/api/v1/watchlist/stats`
    """
    return await _get_manager(db).get_stats()


@router.get("/detailed", response_model=list[TickerMetadataResponse])
@limiter.limit("60/minute")
async def list_tickers_detailed(request: Request, db: DbDep) -> list[TickerMetadataResponse]:
    """List watchlist tickers with full metadata.

    **Inputs:** none.

    **Returns:** `list[TickerMetadataResponse]` — each item has `ticker`,
    `source` (who added it), `added_at`, `expires_at`.

    **Example response item:**
    ```json
    {"ticker": "NVDA", "source": "api",
     "added_at": "2026-04-30T12:00:00Z", "expires_at": "2026-05-30T12:00:00Z"}
    ```
    """
    metadata_list = await _get_manager(db).get_all_with_metadata()
    return [_record_to_response(r) for r in metadata_list]


@router.get("/{ticker}", response_model=TickerMetadataResponse)
@limiter.limit("60/minute")
async def get_ticker(request: Request, ticker: str, db: DbDep) -> TickerMetadataResponse:
    """Lookup metadata for a single watchlist ticker.

    **Path params:**
    - `ticker` (str): symbol, case-insensitive.

    **Returns:** `TickerMetadataResponse` (`ticker`, `source`, `added_at`, `expires_at`).

    **Errors:**
    - `404` if ticker isn't on the watchlist.

    **Example:** `curl http://localhost:7337/api/v1/watchlist/NVDA`
    """
    record = await _get_manager(db).get_metadata(ticker)
    if not record:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return _record_to_response(record)


@router.post("/", response_model=AddTickerResponse, status_code=201)
@limiter.limit("10/minute")
async def add_ticker(request: Request, body: AddTickerRequest, db: DbDep) -> AddTickerResponse:
    """Add a ticker to the watchlist (or refresh its TTL if already present).

    Note: Adding via API persists to Postgres only — it does NOT auto-subscribe
    the Finnhub WebSocket to live quotes. That subscription is established
    when the agent starts up.

    **Body (JSON):**
    - `ticker` (str): symbol (e.g. `"AAPL"`).
    - `source` (str, default `"api"`): who/what added it; surfaces in `/stats`.

    **Returns (201):**
    - `ticker` (str): uppercase symbol.
    - `is_new` (bool): True if added fresh, False if it was already there.

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/watchlist/ \\
      -H "Content-Type: application/json" \\
      -d '{"ticker":"NVDA","source":"manual"}'
    # {"ticker":"NVDA","is_new":true}
    ```
    """
    is_new = await _get_manager(db).add_ticker(
        ticker=body.ticker,
        source=body.source,
    )
    return AddTickerResponse(ticker=body.ticker.upper(), is_new=is_new)


@router.delete("/{ticker}", status_code=204)
@limiter.limit("10/minute")
async def remove_ticker(request: Request, ticker: str, db: DbDep) -> Response:
    """Remove a ticker from the watchlist.

    **Path params:**
    - `ticker` (str): symbol, case-insensitive.

    **Returns:** `204 No Content` on success.

    **Errors:**
    - `404` if ticker isn't on the watchlist.

    **Example:** `curl -X DELETE http://localhost:7337/api/v1/watchlist/NVDA`
    """
    removed = await _get_manager(db).remove_ticker(ticker)
    if not removed:
        raise HTTPException(404, detail=f"Ticker '{ticker.upper()}' not on watchlist")
    return Response(status_code=204)


@router.post("/cleanup", response_model=list[str])
@limiter.limit("10/minute")
async def cleanup_expired(request: Request, db: DbDep) -> list[str]:
    """Manually purge tickers whose `expires_at` is in the past.

    The agent runs this on a 5-minute interval; this endpoint is for ad-hoc
    use. Returns the symbols that were removed.

    **Inputs:** none.

    **Returns:** `list[str]` — uppercase tickers that were expired and removed.

    **Example:** `curl -X POST http://localhost:7337/api/v1/watchlist/cleanup`
    """
    return await _get_manager(db).cleanup_expired()
