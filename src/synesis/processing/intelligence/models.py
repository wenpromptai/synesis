"""Output models for the intelligence pipeline.

Shared across all specialist agents and the LangGraph pipeline (Phase 3).

Design principle: Analysts are INFORMATION GATHERERS. They extract, summarize,
and structure key facts. They do NOT assign sentiment scores or trading signals.
BullResearcher and BearResearcher debate opposing cases per ticker — neither
scores. MacroView has sentiment_score because regime direction is inherently
directional. The Trader agent is the sole decision maker.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Shared Models (used by multiple agents)
# =============================================================================


class TickerMention(BaseModel):
    """A ticker surfaced by any analyst with context."""

    ticker: str
    context: str = ""
    source_accounts: list[str] = Field(default_factory=list)


class MacroTheme(BaseModel):
    """A non-ticker trading theme (e.g. risk-off, sector rotation)."""

    theme: str
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


class RedFlag(BaseModel):
    """Individual red flag detection."""

    category: Literal["financial", "governance", "disclosure"]
    flag: str
    severity: Literal["critical", "warning", "watch"]
    evidence: str


class CompanyAnalysis(BaseModel):
    """CompanyAnalyst output — structured information, no scoring."""

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

    # Qualitative (LLM from 10-K/10-Q/8-K prose)
    business_summary: str = ""
    earnings_quality: str = ""
    risk_assessment: str = ""
    geographic_exposure: str = ""
    key_customers_suppliers: str = ""
    growth_catalysts: str = ""
    competitive_position: str = ""

    # Cross-referenced insights
    insider_vs_financials: str = ""
    disclosure_consistency: str = ""

    # Key findings (no scoring — deferred to Trader)
    primary_thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)


# =============================================================================
# Social Sentiment Analyst Output
# =============================================================================


class SocialSentimentAnalysis(BaseModel):
    """Output of SocialSentimentAnalyst — feeds extract_tickers + downstream analysts + debate."""

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
    """Output of NewsAnalyst — groups news into story clusters."""

    story_clusters: list[NewsStoryCluster] = Field(default_factory=list)
    macro_themes: list[MacroTheme] = Field(default_factory=list)
    summary: str = ""
    analysis_date: date
    messages_analyzed: int = 0


# =============================================================================
# Strategist Output (Layer 2)
# =============================================================================


class SectorTilt(BaseModel):
    """A sector/asset class tilt from the macro regime."""

    sector: str
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Tilt: -1.0 (strongly underweight) to 1.0 (strongly overweight)",
    )
    reasoning: str = ""


class MacroView(BaseModel):
    """MacroStrategist output — regime assessment + sector tilts."""

    regime: Literal["risk_on", "risk_off", "transitioning", "uncertain"]
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Broad market outlook: -1.0 (strongly bearish) to 1.0 (strongly bullish)",
    )
    key_drivers: list[str] = Field(default_factory=list)
    sector_tilts: list[SectorTilt] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    analysis_date: date


# =============================================================================
# Debate Output (Layer 3 — per-ticker via Send)
# =============================================================================


class TickerDebate(BaseModel):
    """Output of a BullResearcher or BearResearcher for a single ticker."""

    role: Literal["bull", "bear"]
    ticker: str = Field(min_length=1)
    argument: str = ""
    key_evidence: list[str] = Field(default_factory=list)
    analysis_date: date
    round: int = Field(default=1, ge=1)


# =============================================================================
# Trader Output (Phase 3D)
# =============================================================================


class TradeIdea(BaseModel):
    """A trade recommendation from the Trader.

    Single-ticker ideas have one ticker. Pair/relative value trades have 2+
    (only produced in portfolio mode). trade_structure is the primary field —
    it describes exactly what to do and is the cue for execution.
    """

    tickers: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    trade_structure: str = Field(
        min_length=1,
        description="The specific trade: 'buy 100 shares NVDA', 'bull call spread NVDA 150/160 June', "
        "'equity L/S: long NVDA / short AMD', etc.",
    )
    thesis: str = ""
    catalyst: str = ""
    timeframe: str = ""
    key_risk: str = ""
    analysis_date: date


class TraderOutput(BaseModel):
    """Full Trader output — one or more TradeIdeas."""

    trade_ideas: list[TradeIdea] = Field(default_factory=list)
    portfolio_note: str = ""
    analysis_date: date


# =============================================================================
# Price Analyst Output
# =============================================================================


class PriceAnalysis(BaseModel):
    """PriceAnalyst output per ticker — information only, no scoring."""

    ticker: str
    analysis_date: date
    spot_price: float | None = None
    change_1d_pct: float | None = None

    # Technical indicators (from pandas-ta on yfinance bars)
    ema_8: float | None = None
    ema_21: float | None = None
    ema_cross: str = ""
    adx: float | None = None
    rsi_14: float | None = None
    macd_histogram: float | None = None
    macd_signal_cross: str = ""
    atr_percent: float | None = None
    bb_width_percentile: float | None = None
    bb_percent_b: float | None = None
    price_zscore: float | None = None
    volume_ratio: float | None = None
    obv_trend: str = ""
    nearest_support: float | None = None
    nearest_resistance: float | None = None

    # Options metrics (IV self-computed from Massive EOD, RV from yfinance)
    atm_iv: float | None = None
    realized_vol_30d: float | None = None
    iv_rv_spread: float | None = None
    put_call_volume_ratio: float | None = None
    atm_skew_ratio: float | None = None
    days_to_expiry: int | None = None

    # Pattern flags (deterministic)
    notable_setups: list[str] = Field(default_factory=list)

    # LLM interpretation (no scoring)
    technical_narrative: str = ""
    options_narrative: str = ""
