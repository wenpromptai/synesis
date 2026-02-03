"""Pydantic models for FactSet data responses."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class FactSetPrice(BaseModel):
    """Daily price data from FactSet Global Prices."""

    fsym_id: str = Field(..., description="FactSet security ID (e.g., 'K7TPSX-R')")
    price_date: date = Field(..., description="Price date")
    open: float | None = Field(None, description="Opening price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float = Field(..., description="Closing price")
    volume: float | None = Field(None, description="Trading volume")
    is_adjusted: bool = Field(False, description="Whether prices are split/dividend adjusted")

    # Pre-calculated returns from FactSet
    one_day_pct: float | None = Field(None, description="1-day return %")
    wtd_pct: float | None = Field(None, description="Week-to-date return %")
    mtd_pct: float | None = Field(None, description="Month-to-date return %")
    qtd_pct: float | None = Field(None, description="Quarter-to-date return %")
    ytd_pct: float | None = Field(None, description="Year-to-date return %")
    one_mth_pct: float | None = Field(None, description="1-month return %")
    three_mth_pct: float | None = Field(None, description="3-month return %")
    six_mth_pct: float | None = Field(None, description="6-month return %")
    one_yr_pct: float | None = Field(None, description="1-year return %")
    two_yr_pct: float | None = Field(None, description="2-year return %")
    three_yr_pct: float | None = Field(None, description="3-year return %")
    five_yr_pct: float | None = Field(None, description="5-year return %")
    ten_yr_pct: float | None = Field(None, description="10-year return %")


class FactSetSecurity(BaseModel):
    """Security master data from FactSet."""

    fsym_id: str = Field(..., description="FactSet security ID (e.g., 'K7TPSX-R')")
    fsym_security_id: str | None = Field(None, description="Security-level ID for shares lookups")
    ticker: str | None = Field(None, description="Ticker with region (e.g., 'NVDA-US')")
    name: str = Field(..., description="Company/security name")
    exchange_code: str = Field(..., description="Exchange code (e.g., 'NAS')")
    security_type: str = Field(..., description="Security type (e.g., 'SHARE')")
    currency: str = Field(..., description="Trading currency (e.g., 'USD')")
    country: str | None = Field(None, description="Country of incorporation")
    sector: str | None = Field(None, description="GICS sector name")
    industry: str | None = Field(None, description="Industry name")


class FactSetFundamentals(BaseModel):
    """Fundamental data from FactSet Fundamentals."""

    fsym_id: str = Field(..., description="FactSet security ID")
    period_end: date = Field(..., description="Fiscal period end date")
    fiscal_year: int | None = Field(None, description="Fiscal year")
    period_type: str = Field(..., description="Period type: 'annual', 'quarterly', 'ltm'")

    # Valuation metrics
    eps_diluted: float | None = Field(None, description="Diluted EPS")
    bps: float | None = Field(None, description="Book value per share")
    dps: float | None = Field(None, description="Dividends per share")

    # Profitability
    roe: float | None = Field(None, description="Return on equity")
    roa: float | None = Field(None, description="Return on assets")
    net_margin: float | None = Field(None, description="Net profit margin")
    gross_margin: float | None = Field(None, description="Gross profit margin")
    operating_margin: float | None = Field(None, description="Operating margin")

    # Leverage
    debt_to_equity: float | None = Field(None, description="Debt to equity ratio")
    debt_to_assets: float | None = Field(None, description="Debt to assets ratio")

    # Valuation (when price data is available)
    ev_to_ebitda: float | None = Field(None, description="EV/EBITDA ratio")
    price_to_book: float | None = Field(None, description="Price to book ratio")
    price_to_sales: float | None = Field(None, description="Price to sales ratio")


class FactSetCorporateAction(BaseModel):
    """Corporate action event from FactSet."""

    fsym_id: str = Field(..., description="FactSet security ID")
    event_type: str = Field(..., description="Event type: 'dividend', 'split', 'rights', 'bonus'")
    event_code: str | None = Field(None, description="Raw event code (DVC, FSP, etc.)")
    effective_date: date = Field(..., description="Effective date of the action")
    record_date: date | None = Field(None, description="Record date for eligibility")
    pay_date: date | None = Field(None, description="Payment date (for dividends)")
    ex_date: date | None = Field(None, description="Ex-dividend/split date")

    # Dividend-specific
    dividend_amount: float | None = Field(None, description="Dividend amount per share")
    dividend_currency: str | None = Field(None, description="Dividend currency")
    dividend_type: str | None = Field(None, description="Dividend type (regular, special, etc.)")

    # Split-specific
    split_factor: float | None = Field(None, description="Split factor (e.g., 4.0 for 4:1 split)")
    split_from: float | None = Field(None, description="Split from ratio")
    split_to: float | None = Field(None, description="Split to ratio")


class FactSetSharesOutstanding(BaseModel):
    """Shares outstanding data from FactSet."""

    fsym_id: str = Field(..., description="FactSet security ID")
    fsym_security_id: str | None = Field(None, description="Security-level ID")
    report_date: date = Field(..., description="Report date for shares data")
    shares_outstanding: float = Field(
        ..., description="Adjusted shares outstanding (actual count, not millions)"
    )
    shares_outstanding_raw: float | None = Field(
        None, description="Raw shares in millions as stored in database"
    )
    adr_ratio: float | None = Field(None, description="ADR ratio if applicable")
    has_adr: bool = Field(False, description="Whether security has an ADR")
