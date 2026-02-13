"""Finnhub real-time price API endpoints."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from synesis.core.constants import FINNHUB_WS_MAX_SYMBOLS
from synesis.core.dependencies import PriceServiceDep
from synesis.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TickerListRequest(BaseModel):
    """Request body for subscribe/unsubscribe."""

    tickers: list[str]


def _decimal_to_float(prices: dict[str, Decimal]) -> dict[str, float]:
    """Convert Decimal prices to float for JSON serialisation."""
    return {k: float(v) for k, v in prices.items()}


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
    """Get prices for multiple tickers (comma-separated query param).

    Example: GET /fh_prices?tickers=AAPL,TSLA,NVDA
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")

    prices = await price_service.get_prices(ticker_list, fallback_to_rest=True)
    return {
        "prices": _decimal_to_float(prices),
        "found": len(prices),
        "missing": sorted(set(ticker_list) - set(prices.keys())),
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


# /{ticker} MUST be last — path param would shadow /subscriptions otherwise
@router.get("/{ticker}")
async def get_single_price(
    ticker: str,
    price_service: PriceServiceDep,
) -> dict[str, Any]:
    """Get the current price for a single ticker (cache + REST fallback)."""
    price = await price_service.get_price(ticker.upper())
    if price is None:
        raise HTTPException(status_code=404, detail=f"No price available for {ticker.upper()}")
    return {"ticker": ticker.upper(), "price": float(price)}
