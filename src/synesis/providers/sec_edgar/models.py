"""Pydantic models for SEC EDGAR data."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class SECFiling(BaseModel):
    """A single SEC filing."""

    ticker: str
    form: str  # "8-K", "10-K", "10-Q", "4", etc.
    filed_date: date
    accepted_datetime: datetime
    accession_number: str
    primary_document: str
    items: str  # e.g., "2.02" for earnings results
    url: str  # Full URL to filing document


class EarningsRelease(BaseModel):
    """An 8-K earnings press release (Item 2.02) with optional full content."""

    ticker: str
    filed_date: date
    accepted_datetime: datetime
    accession_number: str
    url: str
    items: str
    content: str | None = None  # Markdown press release (from Crawl4AI)


class InsiderTransaction(BaseModel):
    """A single insider transaction parsed from Form 4 XML."""

    ticker: str
    owner_name: str
    owner_relationship: str  # "Director", "Officer", "10% Owner"
    transaction_date: date
    transaction_code: str  # P=purchase, S=sale, M=exercise
    shares: float
    price_per_share: float | None
    shares_after: float
    acquired_or_disposed: str  # "A" or "D"
    filing_date: date
    filing_url: str


class Holding13F(BaseModel):
    """A single holding from a 13F-HR InfoTable."""

    name_of_issuer: str
    title_of_class: str
    cusip: str
    value_thousands: int
    shares: int
    investment_discretion: str


class Filing13F(BaseModel):
    """Parsed 13F-HR filing with holdings."""

    cik: str
    fund_name: str
    filed_date: date
    report_date: date
    accession_number: str
    url: str
    holdings: list[Holding13F]
    total_value_thousands: int
