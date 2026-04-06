"""Output models for the intelligence pipeline.

Shared across all specialist agents and the LangGraph pipeline (Phase 3).

Convention: All agents use sentiment_score (-1.0 to 1.0) for directional signals.
- -1.0 = max bearish / strong sell
- 0.0 = neutral
- +1.0 = max bullish / strong buy
The magnitude of the score reflects conviction.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Shared Models (used by multiple agents)
# =============================================================================


class TickerMention(BaseModel):
    """A ticker surfaced by any analyst with sentiment and context."""

    ticker: str
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Sentiment: -1.0 (max bearish) to 1.0 (max bullish)",
    )
    context: str = ""
    source_accounts: list[str] = Field(default_factory=list)


class MacroTheme(BaseModel):
    """A non-ticker trading theme (e.g. risk-off, sector rotation)."""

    theme: str
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Broad market impact: -1.0 (strongly bearish for SPY) to 1.0 (strongly bullish). Asset-specific implications left to strategists.",
    )
    context: str = ""
    source_accounts: list[str] = Field(default_factory=list)


# =============================================================================
# Company Analyst Output
# =============================================================================


class FinancialHealthScore(BaseModel):
    """Mix of yfinance pre-computed ratios + quarterly trends."""

    # yfinance ratios
    market_cap: float | None = None
    beta: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    free_cash_flow: float | None = None
    ebitda: float | None = None
    total_cash: float | None = None
    total_debt: float | None = None
    short_percent_of_float: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    forward_eps: float | None = None

    # Computed scores
    piotroski_f: int | None = Field(default=None, ge=0, le=9)
    beneish_m: float | None = None

    # Quarterly trends
    quarterly_eps_trend: list[dict[str, Any]] = Field(default_factory=list)
    quarterly_revenue_trend: list[dict[str, Any]] = Field(default_factory=list)
    latest_filing_period: str = ""


class InsiderSignal(BaseModel):
    """Aggregated insider transaction analysis from SEC EDGAR."""

    mspr: float | None = None
    buy_count: int = 0
    sell_count: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    cluster_detected: bool = False
    csuite_activity: str = ""
    form144_count: int = 0
    notable_transactions: list[str] = Field(default_factory=list)
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Insider sentiment: -1.0 (heavy selling) to 1.0 (heavy buying)",
    )


class RedFlag(BaseModel):
    """Individual red flag detection."""

    category: Literal["financial", "governance", "disclosure"]
    flag: str
    severity: Literal["critical", "warning", "watch"]
    evidence: str


class CompanyAnalysis(BaseModel):
    """Final CompanyAnalyst output per ticker."""

    ticker: str
    company_name: str
    sector: str = ""
    industry: str = ""
    analysis_date: date
    latest_annual_filing: str = ""

    # Quantitative (deterministic)
    financial_health: FinancialHealthScore
    insider_signal: InsiderSignal
    red_flags: list[RedFlag] = Field(default_factory=list)

    # Qualitative (LLM from 10-K/10-Q prose)
    business_summary: str = ""
    earnings_quality: str = ""
    risk_assessment: str = ""
    geographic_exposure: str = ""
    key_customers_suppliers: str = ""

    # Cross-referenced insights
    insider_vs_financials: str = ""
    disclosure_consistency: str = ""

    # Synthesis
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Overall signal: -1.0 (strong sell) to 1.0 (strong buy)",
    )
    primary_thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)


# =============================================================================
# Social Sentiment Analyst Output
# =============================================================================


class SocialSentimentAnalysis(BaseModel):
    """Output of SocialSentimentAnalyst — feeds extract_tickers + strategists."""

    ticker_mentions: list[TickerMention] = Field(default_factory=list)
    macro_themes: list[MacroTheme] = Field(default_factory=list)
    summary: str = ""
    analysis_date: date


# =============================================================================
# News Analyst Output
# =============================================================================


class NewsEventType(str, Enum):
    """Classification of a news event."""

    earnings = "earnings"
    mna = "m&a"
    regulatory = "regulatory"
    macro = "macro"
    geopolitical = "geopolitical"
    management = "management"
    legal = "legal"
    product = "product"
    financing = "financing"
    other = "other"


class NewsStoryCluster(BaseModel):
    """A group of related news messages about the same event."""

    headline: str
    event_type: NewsEventType
    message_count: int = 1
    tickers: list[TickerMention] = Field(default_factory=list)
    urgency: Literal["low", "normal", "high", "critical"] = "normal"
    key_facts: list[str] = Field(default_factory=list)


class NewsAnalysis(BaseModel):
    """Output of NewsAnalyst — groups news into story clusters with ticker signals."""

    story_clusters: list[NewsStoryCluster] = Field(default_factory=list)
    macro_themes: list[MacroTheme] = Field(default_factory=list)
    summary: str = ""
    analysis_date: date
    messages_analyzed: int = 0
