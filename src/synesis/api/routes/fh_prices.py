"""Finnhub real-time price API endpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from synesis.core.constants import FINNHUB_WS_MAX_SYMBOLS
from synesis.core.dependencies import PriceServiceDep
from synesis.core.logging import get_logger
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
async def get_subscriptions(price_service: PriceServiceDep) -> dict[str, Any]:
    """List subscribed tickers and WebSocket status."""
    return {
        "subscribed_tickers": sorted(price_service._subscribed_tickers),
        "count": len(price_service._subscribed_tickers),
        "ws_connected": price_service._ws is not None,
        "max_symbols": FINNHUB_WS_MAX_SYMBOLS,
    }


@router.get("")
async def get_batch_prices(
    tickers: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Get full quote data for multiple tickers (comma-separated query param).

    Example: GET /fh_prices?tickers=AAPL,TSLA,NVDA
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
async def subscribe_tickers(
    body: TickerListRequest,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Subscribe to real-time price updates for tickers."""
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
async def unsubscribe_tickers(
    body: TickerListRequest,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Unsubscribe from real-time price updates."""
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
async def get_ws_batch_prices(
    tickers: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Get cached prices from WebSocket stream (comma-separated query param).

    These are real-time trade prices cached by the WebSocket connection.
    Tickers must be subscribed first via POST /subscribe.

    Example: GET /fh_prices/ws/prices?tickers=AAPL,TSLA
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
async def get_ws_single_price(
    ticker: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Get cached price for a single ticker from WebSocket stream.

    The ticker must be subscribed first via POST /subscribe.
    """
    price = await price_service.get_cached_price(ticker.upper())
    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"No WebSocket price for {ticker.upper()}. Is it subscribed?",
        )
    return {"ticker": ticker.upper(), "price": price}


# ─── REST API quotes ────────────────────────────────────────
# /{ticker} MUST be last — path param would shadow fixed paths
@router.get("/{ticker}")
async def get_single_price(
    ticker: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Get full quote data for a single ticker from Finnhub REST API."""
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
