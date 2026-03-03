"""Pydantic models for yfinance data."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class EquityQuote(BaseModel):
    """Snapshot quote for an equity, ETF, or index."""

    ticker: str
    name: str | None = None
    currency: str | None = None
    exchange: str | None = None
    last: float | None = None
    prev_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None
    market_cap: float | None = None
    avg_50d: float | None = None
    avg_200d: float | None = None


class OHLCBar(BaseModel):
    """A single OHLCV bar."""

    date: date | datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None


class FXRate(BaseModel):
    """FX spot rate."""

    pair: str
    rate: float | None = None
    bid: float | None = None
    ask: float | None = None


class OptionsGreeks(BaseModel):
    """Black-Scholes Greeks for an options contract."""

    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    implied_volatility: float | None = None


class OptionsContract(BaseModel):
    """A single options contract."""

    contract_symbol: str
    strike: float
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    in_the_money: bool | None = None
    greeks: OptionsGreeks | None = None


class OptionsChain(BaseModel):
    """Full options chain for a given expiration."""

    ticker: str
    expiration: str
    calls: list[OptionsContract]
    puts: list[OptionsContract]
