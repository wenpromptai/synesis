"""Finnhub API endpoints (prices, ticker verification, symbol search)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.requests import Request

from synesis.core.constants import FINNHUB_WS_MAX_SYMBOLS
from synesis.core.dependencies import PriceServiceDep, TickerProviderDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter
from synesis.providers.finnhub import QuoteData

logger = get_logger(__name__)

router = APIRouter()


class TickerListRequest(BaseModel):
    """Request body for subscribe/unsubscribe."""

    tickers: list[str]


def _quote_to_response(quote: QuoteData) -> dict[str, Any]:
    """Convert QuoteData to a JSON-serialisable dict."""
    return asdict(quote)


@router.get("/subscriptions")
@limiter.limit("120/minute")
async def get_subscriptions(request: Request, price_service: PriceServiceDep) -> dict[str, Any]:
    """List tickers currently subscribed on the Finnhub WebSocket.

    **Inputs:** none.

    **Returns:**
    - `subscribed_tickers` (list[str]): sorted symbols.
    - `count` (int).
    - `ws_connected` (bool): True if the WS client object exists.
    - `max_symbols` (int): Finnhub free-tier cap (`FINNHUB_WS_MAX_SYMBOLS`).

    **Example:** `curl http://localhost:7337/api/v1/fh/subscriptions`
    """
    return {
        "subscribed_tickers": sorted(price_service._subscribed_tickers),
        "count": len(price_service._subscribed_tickers),
        "ws_connected": price_service._ws is not None,
        "max_symbols": FINNHUB_WS_MAX_SYMBOLS,
    }


@router.get("")
@limiter.limit("60/minute")
async def get_batch_prices(
    request: Request,
    tickers: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Batch quote fetch via Finnhub REST API.

    **Query params:**
    - `tickers` (str, required): comma-separated symbols, case-insensitive.

    **Returns:**
    - `quotes` (dict[str, QuoteData]): keyed by uppercase ticker.
      Each `QuoteData`: `current`, `change`, `percent_change`, `high`, `low`,
      `open`, `previous_close`, `timestamp`.
    - `found` (int): tickers successfully resolved.
    - `missing` (list[str]): symbols that returned no data.

    **Errors:**
    - `400` if no tickers were provided.

    **Example:** `curl 'http://localhost:7337/api/v1/fh?tickers=AAPL,TSLA,NVDA'`
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")

    quotes = await price_service.get_quotes(ticker_list)
    return {
        "quotes": {k: _quote_to_response(v) for k, v in quotes.items()},
        "found": len(quotes),
        "missing": sorted(set(ticker_list) - set(quotes.keys())),
    }


@router.post("/subscribe")
@limiter.limit("60/minute")
async def subscribe_tickers(
    request: Request,
    body: TickerListRequest,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Subscribe symbols to the Finnhub real-time price WebSocket.

    Once subscribed, live trade prices land in the in-memory cache and are
    readable via `/fh/ws/prices` and `/fh/ws/prices/{ticker}`.

    **Body (JSON):**
    - `tickers` (list[str]): symbols to subscribe (case-insensitive).

    **Returns:**
    - `subscribed` (list[str]): the requested symbols, uppercased.
    - `total` (int): total subscriptions after this call.

    **Errors:**
    - `400` if `tickers` is empty or if adding them would exceed
      `FINNHUB_WS_MAX_SYMBOLS` (Finnhub free-tier cap).

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/fh/subscribe \\
      -H "Content-Type: application/json" \\
      -d '{"tickers":["NVDA","AAPL"]}'
    ```
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    upper_tickers = [t.upper() for t in body.tickers]

    # Check capacity
    current = len(price_service._subscribed_tickers)
    new_tickers = [t for t in upper_tickers if t not in price_service._subscribed_tickers]
    if current + len(new_tickers) > FINNHUB_WS_MAX_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Would exceed max symbol limit ({FINNHUB_WS_MAX_SYMBOLS}). "
                f"Currently subscribed: {current}, new: {len(new_tickers)}"
            ),
        )

    await price_service.subscribe(upper_tickers)
    return {
        "subscribed": upper_tickers,
        "total": len(price_service._subscribed_tickers),
    }


@router.post("/unsubscribe")
@limiter.limit("60/minute")
async def unsubscribe_tickers(
    request: Request,
    body: TickerListRequest,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Unsubscribe symbols from the Finnhub price WebSocket.

    **Body (JSON):**
    - `tickers` (list[str]): symbols to drop.

    **Returns:**
    - `unsubscribed` (list[str]): uppercase symbols requested.
    - `total` (int): subscriptions remaining after this call.

    **Errors:**
    - `400` if `tickers` is empty.

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/fh/unsubscribe \\
      -H "Content-Type: application/json" \\
      -d '{"tickers":["NVDA"]}'
    ```
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    upper_tickers = [t.upper() for t in body.tickers]
    await price_service.unsubscribe(upper_tickers)
    return {
        "unsubscribed": upper_tickers,
        "total": len(price_service._subscribed_tickers),
    }


# ─── WebSocket cached prices ────────────────────────────────


@router.get("/ws/prices")
@limiter.limit("120/minute")
async def get_ws_batch_prices(
    request: Request,
    tickers: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Read cached real-time prices from the Finnhub WebSocket stream.

    Cheaper than `/fh?tickers=...` (which hits the REST API). Tickers must
    have been subscribed first via `POST /fh/subscribe`.

    **Query params:**
    - `tickers` (str, required): comma-separated symbols.

    **Returns:**
    - `prices` (dict[str, float]): keyed by uppercase ticker.
    - `found` (int).
    - `missing` (list[str]): tickers without a cached price (likely unsubscribed).

    **Errors:**
    - `400` if no tickers provided.

    **Example:** `curl 'http://localhost:7337/api/v1/fh/ws/prices?tickers=AAPL,TSLA'`
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")

    prices = await price_service.get_cached_prices(ticker_list)
    return {
        "prices": prices,
        "found": len(prices),
        "missing": sorted(set(ticker_list) - set(prices.keys())),
    }


@router.get("/ws/prices/{ticker}")
@limiter.limit("120/minute")
async def get_ws_single_price(
    request: Request,
    ticker: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Cached real-time price for a single ticker from the WS stream.

    **Path params:**
    - `ticker` (str): symbol; must be currently subscribed.

    **Returns:**
    - `ticker` (str), `price` (float).

    **Errors:**
    - `404` if no cached price exists (ticker not subscribed or no trades yet).

    **Example:** `curl http://localhost:7337/api/v1/fh/ws/prices/NVDA`
    """
    price = await price_service.get_cached_price(ticker.upper())
    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"No WebSocket price for {ticker.upper()}. Is it subscribed?",
        )
    return {"ticker": ticker.upper(), "price": price}


# ─── Ticker verification & search ──────────────────────────


@router.get("/ticker/verify/{ticker}")
@limiter.limit("120/minute")
async def verify_ticker(
    request: Request,
    ticker: str,
    ticker_provider: TickerProviderDep,
) -> dict[str, Any]:
    """Check whether a ticker exists on a major US exchange.

    **Path params:**
    - `ticker` (str): symbol, case-insensitive.

    **Returns:**
    - `valid` (bool): True if the symbol is recognized.
    - `company_name` (str | null): name when valid, null otherwise.

    **Example:** `curl http://localhost:7337/api/v1/fh/ticker/verify/NVDA`
    """
    valid, company_name = await ticker_provider.verify_ticker(ticker)
    return {
        "valid": valid,
        "company_name": company_name,
    }


@router.get("/ticker/search")
@limiter.limit("120/minute")
async def search_ticker(
    request: Request,
    q: str,
    ticker_provider: TickerProviderDep,
) -> dict[str, Any]:
    """Search for stock symbols matching a free-text query (Finnhub).

    **Query params:**
    - `q` (str, required): company name or ticker fragment, e.g. `nvidia` or `nvd`.

    **Returns:**
    - `results` (list): each `{symbol, description, type}`.
    - `count` (int).

    **Errors:**
    - `400` if `q` is missing or whitespace-only.

    **Example:** `curl 'http://localhost:7337/api/v1/fh/ticker/search?q=nvidia'`
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    results = await ticker_provider.search_symbol(q)
    return {"results": results, "count": len(results)}


# ─── REST API quotes ────────────────────────────────────────
# /{ticker} MUST be last — path param would shadow fixed paths
@router.get("/{ticker}")
@limiter.limit("60/minute")
async def get_single_price(
    request: Request,
    ticker: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Single-ticker quote from the Finnhub REST API.

    Note: this route uses a path param so it MUST be declared last in the file —
    otherwise it shadows fixed paths like `/subscriptions`.

    **Path params:**
    - `ticker` (str): symbol, case-insensitive.

    **Returns:** flat object with `ticker`, `current`, `change`, `percent_change`,
    `high`, `low`, `open`, `previous_close`, `timestamp`.

    **Errors:**
    - `404` if Finnhub returns no data for the symbol.
    - upstream Finnhub status code on HTTP error (4xx/5xx).
    - `502` on transport / unexpected errors.

    **Example:** `curl http://localhost:7337/api/v1/fh/NVDA`
    """
    try:
        quote = await price_service.get_quote(ticker.upper())
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Finnhub API error for {ticker.upper()}: {e.response.status_code}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch quote for {ticker.upper()}: {e}"
        ) from e
    return {"ticker": ticker.upper(), **_quote_to_response(quote)}
