"""Models for the market brief pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

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


class ContextGap(BaseModel):
    """A gap in context that needs a web search to fill."""

    query: str = Field(
        description="Targeted web search query (e.g. 'NVDA earnings Q1 2026 results')"
    )
    reason: str = Field(
        description="Why this search is needed — what mover or theme is unexplained"
    )


class ContextGaps(BaseModel):
    """LLM-identified gaps in context that require web search."""

    gaps: list[ContextGap] = Field(
        default_factory=list,
        description="0-5 targeted search queries for movers/themes not explained by existing context. "
        "Empty list if context is sufficient.",
    )


class MoverInsight(BaseModel):
    """Why a specific stock is moving."""

    ticker: str
    move: str = Field(description="Price move e.g. '+5.2%' or '-3.1%'")
    explanation: str = Field(description="1-2 sentence explanation of why it moved")


class MarketBriefAnalysis(BaseModel):
    """LLM analysis synthesizing market data with news, events, and social signals."""

    headline: str = Field(description="One-liner capturing today's market story")
    summary: str = Field(
        description="2-3 paragraph narrative connecting market moves to underlying drivers"
    )
    key_drivers: list[str] = Field(
        description="3-5 main factors driving today's market (e.g. 'Fed hawkish rhetoric', 'AI earnings beat')"
    )
    mover_insights: list[MoverInsight] = Field(
        description="Explanations for top 5-10 notable movers"
    )
    sector_rotation: str = Field(description="1-2 sentences on sector flows and rotation patterns")
    outlook: str = Field(description="What to watch for the rest of the session and coming days")
