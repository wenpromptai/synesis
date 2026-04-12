"""Pydantic models for Event Radar pipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

EventCategory = Literal[
    "earnings",
    "economic_data",
    "fed",
    "13f_filing",
    "conference",
    "release",
    "regulatory",
    "other",
]

EventSector = Literal[
    "ai", "semiconductors", "ai_infrastructure", "power", "energy", "precious_metals"
]

EventRegion = Literal["US", "JP", "SG", "HK", "global"]


class CalendarEvent(BaseModel):
    """A market-relevant event extracted from any source."""

    title: str
    description: str | None = None
    event_date: date
    event_end_date: date | None = None
    category: EventCategory
    sector: EventSector | None = None
    region: list[EventRegion]
    tickers: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    time_label: str | None = None  # "AH" | "PM" | "DM"


class CalendarEventRow(BaseModel):
    """A calendar event as stored in the database (includes DB-generated fields)."""

    id: int
    title: str
    description: str | None = None
    event_date: date
    event_end_date: date | None = None
    category: EventCategory
    sector: EventSector | None = None
    region: list[str]
    tickers: list[str]
    source_urls: list[str]
    time_label: str | None = None
    discovered_at: datetime
    updated_at: datetime
