"""Models for the market brief pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from synesis.providers.yfinance.models import MarketMovers


class TickerChange(BaseModel):
    """A single ticker with price change data."""

    ticker: str
    name: str | None = None
    label: str | None = None
    last: float | None = None
    prev_close: float | None = None
    change_pct: float | None = None


class MarketBriefData(BaseModel):
    """Raw market data for the brief (not LLM output)."""

    equities: list[TickerChange]
    rates_fx: list[TickerChange]
    commodities: list[TickerChange]
    volatility: TickerChange | None = None
    sectors: list[TickerChange]
    movers: MarketMovers
    fetched_at: datetime
