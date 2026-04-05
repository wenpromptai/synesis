"""Output models for the intelligence pipeline.

Shared across all specialist agents and the LangGraph pipeline (Phase 3).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

SignalDirection = Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]


class FinancialHealthScore(BaseModel):
    """Mix of yfinance pre-computed ratios + XBRL multi-quarter trends."""

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

    # Computed scores (from XBRL multi-quarter data)
    piotroski_f: int | None = Field(default=None, ge=0, le=9)
    beneish_m: float | None = None

    # XBRL quarterly trends
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
    signal: SignalDirection = "neutral"


class RedFlag(BaseModel):
    """Individual red flag detection."""

    category: Literal["financial", "governance", "disclosure"]
    flag: str
    severity: Literal["critical", "warning", "watch"]
    evidence: str


class CompanyAnalysis(BaseModel):
    """Final USCompanyAnalyst output per ticker."""

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
    overall_signal: SignalDirection = "neutral"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)
