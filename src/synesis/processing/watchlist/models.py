"""Data models for Flow 4: Watchlist Intelligence.

Signal and report models for the watchlist fundamental analysis flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TickerIntelligence(BaseModel):
    """Raw data gathered for one ticker from all providers."""

    ticker: str
    company_name: str | None = None

    # Timestamps for recency weighting
    data_as_of: datetime | None = None
    fundamentals_period: str | None = None  # e.g., "LTM ending 2025-12-31"
    price_as_of: str | None = None  # e.g., "2026-02-18"

    # FactSet
    market_cap: float | None = None
    fundamentals: dict[str, Any] | None = None
    price_change_1d: float | None = None
    price_change_1m: float | None = None

    # SEC EDGAR
    recent_filings: list[dict[str, Any]] = Field(default_factory=list)
    insider_sentiment: dict[str, Any] | None = None
    recent_insider_txns: list[dict[str, Any]] = Field(default_factory=list)
    eps_history: list[dict[str, Any]] = Field(default_factory=list)

    # Nasdaq
    next_earnings: dict[str, Any] | None = None

    # Web search (analyst ratings, news)
    recent_news: list[dict[str, Any]] = Field(default_factory=list)


class CatalystAlert(BaseModel):
    """Something actionable detected for a ticker."""

    ticker: str
    alert_type: Literal[
        "earnings_imminent",
        "insider_buying",
        "insider_selling",
        "new_sec_filing",
        "valuation_extreme",
    ]
    severity: Literal["high", "medium", "low"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class TickerReport(BaseModel):
    """LLM-synthesized report for one ticker."""

    ticker: str
    company_name: str = ""
    fundamental_summary: str = ""
    catalyst_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    overall_outlook: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class WatchlistSignal(BaseModel):
    """Output of one watchlist analysis cycle."""

    timestamp: datetime
    tickers_analyzed: int = 0
    ticker_reports: list[TickerReport] = Field(default_factory=list)
    alerts: list[CatalystAlert] = Field(default_factory=list)
    summary: str = ""
