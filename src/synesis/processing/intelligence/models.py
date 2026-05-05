"""Output models for the intelligence pipeline.

Shared across all specialist agents and the LangGraph pipeline.

Design principle: Analysts are INFORMATION GATHERERS. They extract, summarize,
and structure key facts. They do NOT assign sentiment scores or trading signals.
BullResearcher and BearResearcher debate opposing cases per ticker — neither
scores. The Trader agent is the sole decision maker.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Ticker Research Output (analyze graph — pre-gathering via web + Twitter)
# =============================================================================


class TickerResearchItem(BaseModel):
    """Per-ticker social + news research context for debate agents."""

    ticker: str
    social_highlights: list[str] = Field(default_factory=list)
    news_highlights: list[str] = Field(default_factory=list)
    key_narratives: str = ""
    sentiment_lean: str = ""  # bullish, bearish, mixed, neutral
    verified_claims: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)


class TickerResearchAnalysis(BaseModel):
    """Output from the TickerResearchAnalyst."""

    research: list[TickerResearchItem] = Field(default_factory=list)
    summary: str = ""
    analysis_date: date


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


class AnalystConsensus(BaseModel):
    """Aggregated analyst ratings from yfinance."""

    consensus_period: str = ""  # e.g. "2026-04"
    buy_count: int = 0
    hold_count: int = 0
    sell_count: int = 0
    price_target_mean: float | None = None
    price_target_median: float | None = None
    price_target_high: float | None = None
    price_target_low: float | None = None
    current_price: float | None = None
    recent_actions: list[str] = Field(default_factory=list)


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
    analyst_consensus: AnalystConsensus
    red_flags: list[RedFlag] = Field(default_factory=list)

    # Qualitative (LLM from 10-K/10-Q/8-K prose)
    business_summary: str = ""
    earnings_quality: str = ""
    risk_assessment: str = ""
    geographic_exposure: str = ""
    key_customers_suppliers: str = ""
    forward_outlook: str = ""
    competitive_position: str = ""

    # Cross-referenced insights
    insider_vs_financials: str = ""
    disclosure_consistency: str = ""

    # Key findings (no scoring — deferred to Trader)
    primary_thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)


# =============================================================================
# Debate Output (Layer 3 — per-ticker via Send)
# =============================================================================


class TickerDebate(BaseModel):
    """Output of a BullResearcher or BearResearcher for a single ticker."""

    role: Literal["bull", "bear"]
    ticker: str = Field(min_length=1)
    argument: str = ""
    key_evidence: list[str] = Field(default_factory=list)

    # Variant perception — anchored against consensus
    variant_vs_consensus: str = ""
    estimated_upside_downside: str = ""
    catalyst: str = ""
    catalyst_timeline: str = ""
    what_would_change_my_mind: str = ""

    analysis_date: date
    round: int = Field(default=1, ge=1)


# =============================================================================
# Trader Output (Phase 3D)
# =============================================================================


class TradeIdea(BaseModel):
    """A trade recommendation from the Trader.

    One TradeIdea per ticker. Equity positions only — no options strategy
    construction. trade_structure describes the equity position;
    expression_note provides vol context for optional enhancement.
    """

    tickers: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, max_length=1)
    trade_structure: str = Field(
        min_length=1,
        description="Equity position: 'long NVDA' or 'short AMD'.",
    )
    thesis: str = ""
    catalyst: str = ""
    timeframe: str = ""
    key_risk: str = ""

    # R/R framework
    entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    risk_reward_ratio: float | None = None

    # Conviction
    conviction_tier: Literal[1, 2, 3] | None = None
    conviction_rationale: str = ""
    downside_scenario: str = ""

    # Vol context (replaces options strategy construction)
    expression_note: str = ""

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
