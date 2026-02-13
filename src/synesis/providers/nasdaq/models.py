"""Pydantic models for NASDAQ earnings data."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class EarningsEvent(BaseModel):
    """A single earnings calendar event."""

    ticker: str
    company_name: str
    earnings_date: date
    time: str  # "pre-market", "after-hours", "during-market"
    eps_forecast: float | None = None
    num_estimates: int = 0
    market_cap: float | None = None
    fiscal_quarter: str = ""  # "Dec/2025"
