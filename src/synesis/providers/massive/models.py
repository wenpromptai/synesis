"""Pydantic models for Massive.com market data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# --- Aggregates ---


class Bar(BaseModel):
    """A single OHLCV bar."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    timestamp: int  # Unix milliseconds
    transactions: int | None = None


class BarsResponse(BaseModel):
    """Collection of OHLCV bars for a ticker."""

    ticker: str
    bars: list[Bar]
    adjusted: bool = True


class DailySummary(BaseModel):
    """Open/close daily summary with extended hours."""

    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    after_hours: float | None = None
    pre_market: float | None = None


# --- Reference ---


class TickerInfo(BaseModel):
    """Basic ticker metadata from search/list."""

    ticker: str
    name: str
    market: str = ""
    locale: str = ""
    type: str = ""
    active: bool = True
    primary_exchange: str | None = None
    currency: str | None = None
    cik: str | None = None
    composite_figi: str | None = None


class TickerOverview(TickerInfo):
    """Full ticker details including fundamentals."""

    market_cap: float | None = None
    description: str | None = None
    homepage_url: str | None = None
    total_employees: int | None = None
    list_date: str | None = None
    sic_code: str | None = None
    sic_description: str | None = None
    weighted_shares_outstanding: float | None = None
    phone_number: str | None = None
    address: dict[str, str] | None = None
    branding: dict[str, str] | None = None
    ticker_root: str | None = None
    ticker_suffix: str | None = None
    share_class_figi: str | None = None
    share_class_shares_outstanding: float | None = None
    round_lot: int | None = None
    delisted_utc: str | None = None


class TickerEvent(BaseModel):
    """A historical event for a ticker (e.g. ticker change)."""

    type: str
    date: str
    ticker_change: dict[str, str] | None = None


# --- Fundamentals & Corporate Actions ---


class FinancialResult(BaseModel):
    """A single financial filing from the legacy vX endpoint."""

    tickers: list[str] = []
    start_date: str = ""
    end_date: str = ""
    filing_date: str | None = None
    fiscal_period: str | None = None
    fiscal_year: str | None = None
    timeframe: str | None = None
    financials: dict[str, Any] = {}


class Dividend(BaseModel):
    """A historical cash dividend distribution."""

    ticker: str
    ex_dividend_date: str
    pay_date: str | None = None
    record_date: str | None = None
    declaration_date: str | None = None
    cash_amount: float
    currency: str = "USD"
    frequency: int | None = None
    distribution_type: str | None = None


class Split(BaseModel):
    """A stock split event."""

    ticker: str
    execution_date: str
    split_from: float
    split_to: float
    adjustment_type: str | None = None


class ShortInterest(BaseModel):
    """Bi-monthly FINRA short interest data."""

    ticker: str
    settlement_date: str
    short_interest: int
    avg_daily_volume: int | None = None
    days_to_cover: float | None = None


class ShortVolume(BaseModel):
    """Daily short sale volume."""

    ticker: str
    date: str
    short_volume: float
    total_volume: float
    exempt_volume: float | None = None


# --- News ---


class NewsInsight(BaseModel):
    """Per-ticker sentiment from a news article."""

    ticker: str
    sentiment: str | None = None
    sentiment_reasoning: str | None = None


class NewsArticle(BaseModel):
    """A news article with ticker sentiment analysis."""

    id: str
    title: str
    published_utc: str
    article_url: str
    tickers: list[str] = []
    description: str | None = None
    keywords: list[str] = []
    insights: list[NewsInsight] = []
    author: str | None = None
    image_url: str | None = None


# --- Technical Indicators ---


class IndicatorValue(BaseModel):
    """A single SMA/EMA/RSI data point."""

    timestamp: int
    value: float


class MACDValue(BaseModel):
    """A single MACD data point with signal and histogram."""

    timestamp: int
    value: float
    signal: float
    histogram: float


# --- Market Operations ---


class MarketStatus(BaseModel):
    """Current market open/closed status."""

    market: str
    early_hours: bool
    after_hours: bool
    server_time: str
    exchanges: dict[str, str]
    currencies: dict[str, str]


class MarketHoliday(BaseModel):
    """An upcoming market holiday."""

    date: str
    exchange: str
    name: str
    status: str


# --- Options ---


class OptionsContractRef(BaseModel):
    """An options contract listing (reference data, no pricing)."""

    ticker: str
    underlying_ticker: str
    contract_type: str
    exercise_style: str = ""
    expiration_date: str
    strike_price: float
    shares_per_contract: int = 100
    primary_exchange: str | None = None
    cfi: str | None = None
