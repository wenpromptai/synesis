"""Pydantic models for yfinance data."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


def _clean_float(value: Any) -> float | None:
    """Convert a value to float, returning None for NaN, Inf, None, or non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clean_int(value: Any) -> int | None:
    """Convert a value to int, returning None for NaN, Inf, None, or non-numeric."""
    f = _clean_float(value)
    if f is None:
        return None
    return int(f)


class MarketMover(BaseModel):
    """A single stock entry from gainers/losers/most-actives screener."""

    ticker: str
    name: str | None = None
    price: float | None = None
    change_pct: float | None = None
    change_abs: float | None = None
    volume: int | None = None
    avg_volume_3m: int | None = None
    volume_ratio: float | None = None
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None


class MarketMovers(BaseModel):
    """Top market movers snapshot from the yfinance screener."""

    gainers: list[MarketMover]
    losers: list[MarketMover]
    most_actives: list[MarketMover]
    fetched_at: datetime


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


class OptionsSnapshot(BaseModel):
    """Pre-computed options snapshot with realized vol and ATM chain."""

    ticker: str
    spot: float | None = None
    realized_vol_30d: float | None = None
    expiration: str
    days_to_expiry: int
    calls: list[OptionsContract]
    puts: list[OptionsContract]


class CompanyFundamentals(BaseModel):
    """Full fundamental snapshot for a company, parsed from yfinance .info dict."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    employees: int | None = None
    business_summary: str | None = None

    # Market / size
    market_cap: float | None = None
    beta: float | None = None

    # Income statement
    total_revenue: float | None = None
    ebitda: float | None = None

    # Balance sheet
    total_cash: float | None = None
    total_debt: float | None = None
    free_cash_flow: float | None = None

    # Liquidity & leverage
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None

    # Profitability
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None

    # Valuation multiples
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None

    # Earnings per share
    forward_eps: float | None = None
    trailing_eps: float | None = None

    # Short interest
    shares_short: int | None = None
    short_percent_of_float: float | None = None

    # Analyst targets
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    analyst_count: int | None = None

    # Ownership
    held_percent_insiders: float | None = None
    held_percent_institutions: float | None = None

    @classmethod
    def from_yfinance_info(cls, ticker: str, info: dict[str, Any]) -> CompanyFundamentals:
        """Build a CompanyFundamentals from a yfinance Ticker.info dict."""
        return cls(
            ticker=ticker,
            name=info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            employees=_clean_int(info.get("fullTimeEmployees")),
            business_summary=info.get("longBusinessSummary"),
            market_cap=_clean_float(info.get("marketCap")),
            beta=_clean_float(info.get("beta")),
            total_revenue=_clean_float(info.get("totalRevenue")),
            ebitda=_clean_float(info.get("ebitda")),
            total_cash=_clean_float(info.get("totalCash")),
            total_debt=_clean_float(info.get("totalDebt")),
            free_cash_flow=_clean_float(info.get("freeCashflow")),
            current_ratio=_clean_float(info.get("currentRatio")),
            quick_ratio=_clean_float(info.get("quickRatio")),
            debt_to_equity=_clean_float(info.get("debtToEquity")),
            roe=_clean_float(info.get("returnOnEquity")),
            roa=_clean_float(info.get("returnOnAssets")),
            gross_margin=_clean_float(info.get("grossMargins")),
            operating_margin=_clean_float(info.get("operatingMargins")),
            profit_margin=_clean_float(info.get("profitMargins")),
            revenue_growth=_clean_float(info.get("revenueGrowth")),
            price_to_book=_clean_float(info.get("priceToBook")),
            price_to_sales=_clean_float(info.get("priceToSalesTrailing12Months")),
            ev_to_ebitda=_clean_float(info.get("enterpriseToEbitda")),
            ev_to_revenue=_clean_float(info.get("enterpriseToRevenue")),
            forward_eps=_clean_float(info.get("forwardEps")),
            trailing_eps=_clean_float(info.get("trailingEps")),
            shares_short=_clean_int(info.get("sharesShort")),
            short_percent_of_float=_clean_float(info.get("shortPercentOfFloat")),
            analyst_target_mean=_clean_float(info.get("targetMeanPrice")),
            analyst_target_high=_clean_float(info.get("targetHighPrice")),
            analyst_target_low=_clean_float(info.get("targetLowPrice")),
            analyst_count=_clean_int(info.get("numberOfAnalystOpinions")),
            held_percent_insiders=_clean_float(info.get("heldPercentInsiders")),
            held_percent_institutions=_clean_float(info.get("heldPercentInstitutions")),
        )
