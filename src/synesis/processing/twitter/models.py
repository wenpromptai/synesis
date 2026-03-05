"""Pydantic models for Twitter agent daily digest."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TickerMention(BaseModel):
    """A ticker mentioned within a theme, with per-ticker direction."""

    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    reasoning: str
    current_price: float | None = None
    price_context: str | None = None
    trade_idea: str | None = None
    time_horizon: Literal["intraday", "days", "weeks", "months"] | None = None
    conviction: Literal["high", "medium", "low"] | None = None


class Theme(BaseModel):
    """A synthesized investment theme extracted from multiple tweets."""

    title: str
    summary: str
    category: Literal["macro", "sector", "earnings", "geopolitical", "trade_idea", "technical"]
    sources: list[str]
    tickers: list[TickerMention]
    risk_factors: list[str]
    verified: bool
    verification_notes: str
    conviction: Literal["high", "medium", "low"]
    research_notes: str | None = None


class TwitterAgentAnalysis(BaseModel):
    """Full daily digest output from the Twitter agent analyzer."""

    market_overview: str
    themes: list[Theme]
    raw_tweet_count: int = 0
