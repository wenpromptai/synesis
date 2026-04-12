"""Pydantic models for yfinance data."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


def _clean_float(value: Any) -> float | None:
    """Convert a value to float, returning None for NaN, Inf, None, or non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clean_int(value: Any) -> int | None:
    """Convert a value to int, returning None for NaN, Inf, None, or non-numeric."""
    f = _clean_float(value)
    if f is None:
        return None
    return int(f)


class MarketMover(BaseModel):
    """A single stock entry from gainers/losers/most-actives screener."""

    ticker: str
    name: str | None = None
    price: float | None = None
    change_pct: float | None = None
    change_abs: float | None = None
    volume: int | None = None
    avg_volume_3m: int | None = None
    volume_ratio: float | None = None
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None


class MarketMovers(BaseModel):
    """Top market movers snapshot from the yfinance screener."""

    gainers: list[MarketMover]
    losers: list[MarketMover]
    most_actives: list[MarketMover]
    fetched_at: datetime


class EquityQuote(BaseModel):
    """Snapshot quote for an equity, ETF, or index."""

    ticker: str
    name: str | None = None
    currency: str | None = None
    exchange: str | None = None
    last: float | None = None
    prev_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None
    market_cap: float | None = None
    avg_50d: float | None = None
    avg_200d: float | None = None


class OHLCBar(BaseModel):
    """A single OHLCV bar."""

    date: date | datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None


class FXRate(BaseModel):
    """FX spot rate."""

    pair: str
    rate: float | None = None
    bid: float | None = None
    ask: float | None = None


class OptionsGreeks(BaseModel):
    """Black-Scholes Greeks for an options contract."""

    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    implied_volatility: float | None = None


class OptionsContract(BaseModel):
    """A single options contract."""

    contract_symbol: str
    strike: float
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    in_the_money: bool | None = None
    greeks: OptionsGreeks | None = None


class OptionsChain(BaseModel):
    """Full options chain for a given expiration."""

    ticker: str
    expiration: str
    calls: list[OptionsContract]
    puts: list[OptionsContract]


class OptionsSnapshot(BaseModel):
    """Pre-computed options snapshot with realized vol and ATM chain."""

    ticker: str
    spot: float | None = None
    realized_vol_30d: float | None = None
    expiration: str
    days_to_expiry: int
    calls: list[OptionsContract]
    puts: list[OptionsContract]


class RecommendationTrend(BaseModel):
    """Monthly aggregated analyst recommendation counts."""

    period: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0


class UpgradeDowngrade(BaseModel):
    """Individual analyst firm grade change with price target."""

    date: datetime
    firm: str
    to_grade: str
    from_grade: str
    action: str
    price_target_action: str | None = None
    current_price_target: float | None = None
    prior_price_target: float | None = None


class AnalystPriceTargets(BaseModel):
    """Consensus analyst price target summary."""

    current: float | None = None
    high: float | None = None
    low: float | None = None
    mean: float | None = None
    median: float | None = None


class AnalystRatings(BaseModel):
    """Complete analyst ratings snapshot for a ticker."""

    ticker: str
    recommendations: list[RecommendationTrend] = Field(default_factory=list)
    upgrades_downgrades: list[UpgradeDowngrade] = Field(default_factory=list)
    price_targets: AnalystPriceTargets | None = None


class CompanyFundamentals(BaseModel):
    """Full fundamental snapshot for a company, parsed from yfinance .info dict."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    employees: int | None = None
    business_summary: str | None = None

    # Market / size
    market_cap: float | None = None
    beta: float | None = None

    # Income statement
    total_revenue: float | None = None
    ebitda: float | None = None

    # Balance sheet
    total_cash: float | None = None
    total_debt: float | None = None
    free_cash_flow: float | None = None

    # Liquidity & leverage
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None

    # Profitability
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None

    # Valuation multiples
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None

    # Earnings per share
    forward_eps: float | None = None
    trailing_eps: float | None = None

    # Short interest
    shares_short: int | None = None
    short_percent_of_float: float | None = None

    # Analyst targets
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    analyst_count: int | None = None

    # Ownership
    held_percent_insiders: float | None = None
    held_percent_institutions: float | None = None

    @classmethod
    def from_yfinance_info(cls, ticker: str, info: dict[str, Any]) -> CompanyFundamentals:
        """Build a CompanyFundamentals from a yfinance Ticker.info dict."""
        return cls(
            ticker=ticker,
            name=info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            employees=_clean_int(info.get("fullTimeEmployees")),
            business_summary=info.get("longBusinessSummary"),
            market_cap=_clean_float(info.get("marketCap")),
            beta=_clean_float(info.get("beta")),
            total_revenue=_clean_float(info.get("totalRevenue")),
            ebitda=_clean_float(info.get("ebitda")),
            total_cash=_clean_float(info.get("totalCash")),
            total_debt=_clean_float(info.get("totalDebt")),
            free_cash_flow=_clean_float(info.get("freeCashflow")),
            current_ratio=_clean_float(info.get("currentRatio")),
            quick_ratio=_clean_float(info.get("quickRatio")),
            debt_to_equity=_clean_float(info.get("debtToEquity")),
            roe=_clean_float(info.get("returnOnEquity")),
            roa=_clean_float(info.get("returnOnAssets")),
            gross_margin=_clean_float(info.get("grossMargins")),
            operating_margin=_clean_float(info.get("operatingMargins")),
            profit_margin=_clean_float(info.get("profitMargins")),
            revenue_growth=_clean_float(info.get("revenueGrowth")),
            price_to_book=_clean_float(info.get("priceToBook")),
            price_to_sales=_clean_float(info.get("priceToSalesTrailing12Months")),
            ev_to_ebitda=_clean_float(info.get("enterpriseToEbitda")),
            ev_to_revenue=_clean_float(info.get("enterpriseToRevenue")),
            forward_eps=_clean_float(info.get("forwardEps")),
            trailing_eps=_clean_float(info.get("trailingEps")),
            shares_short=_clean_int(info.get("sharesShort")),
            short_percent_of_float=_clean_float(info.get("shortPercentOfFloat")),
            analyst_target_mean=_clean_float(info.get("targetMeanPrice")),
            analyst_target_high=_clean_float(info.get("targetHighPrice")),
            analyst_target_low=_clean_float(info.get("targetLowPrice")),
            analyst_count=_clean_int(info.get("numberOfAnalystOpinions")),
            held_percent_insiders=_clean_float(info.get("heldPercentInsiders")),
            held_percent_institutions=_clean_float(info.get("heldPercentInstitutions")),
        )


class QuarterlyIncomeStatement(BaseModel):
    """Quarterly income statement from yfinance."""

    period: date
    total_revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_expense: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    ebitda: float | None = None
    ebit: float | None = None
    basic_eps: float | None = None
    diluted_eps: float | None = None
    research_and_development: float | None = None
    selling_general_and_administration: float | None = None
    interest_expense: float | None = None
    interest_income: float | None = None
    tax_provision: float | None = None
    pretax_income: float | None = None
    reconciled_depreciation: float | None = None
    basic_average_shares: float | None = None
    diluted_average_shares: float | None = None


class QuarterlyBalanceSheet(BaseModel):
    """Quarterly balance sheet from yfinance."""

    period: date
    total_assets: float | None = None
    current_assets: float | None = None
    cash_and_cash_equivalents: float | None = None
    receivables: float | None = None
    inventory: float | None = None
    prepaid_assets: float | None = None
    net_ppe: float | None = None
    goodwill_and_intangibles: float | None = None
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    accounts_payable: float | None = None
    current_debt: float | None = None
    long_term_debt: float | None = None
    total_debt: float | None = None
    stockholders_equity: float | None = None
    retained_earnings: float | None = None
    common_stock_equity: float | None = None
    ordinary_shares_number: float | None = None
    working_capital: float | None = None
    net_tangible_assets: float | None = None
    invested_capital: float | None = None
    net_debt: float | None = None
    capital_lease_obligations: float | None = None


class QuarterlyCashFlow(BaseModel):
    """Quarterly cash flow statement from yfinance."""

    period: date
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    depreciation_and_amortization: float | None = None
    stock_based_compensation: float | None = None
    change_in_working_capital: float | None = None
    change_in_inventory: float | None = None
    change_in_receivables: float | None = None
    change_in_payables: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    net_common_stock_issuance: float | None = None
    net_long_term_debt_issuance: float | None = None
    issuance_of_capital_stock: float | None = None
    repayment_of_debt: float | None = None
    interest_paid: float | None = None


class QuarterlyFinancials(BaseModel):
    """Complete quarterly financial statements from yfinance.

    Contains up to 5 quarters of income, balance sheet, and cash flow data.
    Updates same-day when earnings are released (vs XBRL which lags until 10-K/10-Q filing).
    """

    ticker: str
    income: list[QuarterlyIncomeStatement] = Field(default_factory=list)
    balance_sheet: list[QuarterlyBalanceSheet] = Field(default_factory=list)
    cash_flow: list[QuarterlyCashFlow] = Field(default_factory=list)

    @classmethod
    def from_yfinance(
        cls,
        ticker: str,
        financials: Any,
        balance_sheet: Any,
        cashflow: Any,
    ) -> QuarterlyFinancials:
        """Build from yfinance Ticker quarterly dataframes."""

        def _val(df: Any, row: str, col: Any) -> float | None:
            if df is None or df.empty or row not in df.index:
                return None
            return _clean_float(df.loc[row].get(col))

        income_stmts: list[QuarterlyIncomeStatement] = []
        if financials is not None and not financials.empty:
            for col in financials.columns:
                income_stmts.append(
                    QuarterlyIncomeStatement(
                        period=col.date() if hasattr(col, "date") else col,
                        total_revenue=_val(financials, "Total Revenue", col),
                        cost_of_revenue=_val(financials, "Cost Of Revenue", col),
                        gross_profit=_val(financials, "Gross Profit", col),
                        operating_expense=_val(financials, "Operating Expense", col),
                        operating_income=_val(financials, "Operating Income", col),
                        net_income=_val(financials, "Net Income", col),
                        ebitda=_val(financials, "EBITDA", col),
                        ebit=_val(financials, "EBIT", col),
                        basic_eps=_val(financials, "Basic EPS", col),
                        diluted_eps=_val(financials, "Diluted EPS", col),
                        research_and_development=_val(financials, "Research And Development", col),
                        selling_general_and_administration=_val(
                            financials, "Selling General And Administration", col
                        ),
                        interest_expense=_val(financials, "Interest Expense", col),
                        interest_income=_val(financials, "Interest Income", col),
                        tax_provision=_val(financials, "Tax Provision", col),
                        pretax_income=_val(financials, "Pretax Income", col),
                        reconciled_depreciation=_val(financials, "Reconciled Depreciation", col),
                        basic_average_shares=_val(financials, "Basic Average Shares", col),
                        diluted_average_shares=_val(financials, "Diluted Average Shares", col),
                    )
                )

        bs_stmts: list[QuarterlyBalanceSheet] = []
        if balance_sheet is not None and not balance_sheet.empty:
            for col in balance_sheet.columns:
                bs_stmts.append(
                    QuarterlyBalanceSheet(
                        period=col.date() if hasattr(col, "date") else col,
                        total_assets=_val(balance_sheet, "Total Assets", col),
                        current_assets=_val(balance_sheet, "Current Assets", col),
                        cash_and_cash_equivalents=_val(
                            balance_sheet, "Cash And Cash Equivalents", col
                        ),
                        receivables=_val(balance_sheet, "Receivables", col),
                        inventory=_val(balance_sheet, "Inventory", col),
                        prepaid_assets=_val(balance_sheet, "Prepaid Assets", col),
                        net_ppe=_val(balance_sheet, "Net PPE", col),
                        goodwill_and_intangibles=_val(
                            balance_sheet, "Goodwill And Other Intangible Assets", col
                        ),
                        total_liabilities=_val(
                            balance_sheet, "Total Liabilities Net Minority Interest", col
                        ),
                        current_liabilities=_val(balance_sheet, "Current Liabilities", col),
                        accounts_payable=_val(balance_sheet, "Accounts Payable", col),
                        current_debt=_val(balance_sheet, "Current Debt", col),
                        long_term_debt=_val(balance_sheet, "Long Term Debt", col),
                        total_debt=_val(balance_sheet, "Total Debt", col),
                        stockholders_equity=_val(balance_sheet, "Stockholders Equity", col),
                        retained_earnings=_val(balance_sheet, "Retained Earnings", col),
                        common_stock_equity=_val(balance_sheet, "Common Stock Equity", col),
                        ordinary_shares_number=_val(balance_sheet, "Ordinary Shares Number", col),
                        working_capital=_val(balance_sheet, "Working Capital", col),
                        net_tangible_assets=_val(balance_sheet, "Net Tangible Assets", col),
                        invested_capital=_val(balance_sheet, "Invested Capital", col),
                        net_debt=_val(balance_sheet, "Net Debt", col),
                        capital_lease_obligations=_val(
                            balance_sheet, "Capital Lease Obligations", col
                        ),
                    )
                )

        cf_stmts: list[QuarterlyCashFlow] = []
        if cashflow is not None and not cashflow.empty:
            for col in cashflow.columns:
                cf_stmts.append(
                    QuarterlyCashFlow(
                        period=col.date() if hasattr(col, "date") else col,
                        operating_cash_flow=_val(cashflow, "Operating Cash Flow", col),
                        capital_expenditure=_val(cashflow, "Capital Expenditure", col),
                        free_cash_flow=_val(cashflow, "Free Cash Flow", col),
                        depreciation_and_amortization=_val(
                            cashflow, "Depreciation And Amortization", col
                        ),
                        stock_based_compensation=_val(cashflow, "Stock Based Compensation", col),
                        change_in_working_capital=_val(cashflow, "Change In Working Capital", col),
                        change_in_inventory=_val(cashflow, "Change In Inventory", col),
                        change_in_receivables=_val(cashflow, "Change In Receivables", col),
                        change_in_payables=_val(
                            cashflow, "Change In Payables And Accrued Expense", col
                        ),
                        investing_cash_flow=_val(cashflow, "Investing Cash Flow", col),
                        financing_cash_flow=_val(cashflow, "Financing Cash Flow", col),
                        net_common_stock_issuance=_val(cashflow, "Net Common Stock Issuance", col),
                        net_long_term_debt_issuance=_val(
                            cashflow, "Net Long Term Debt Issuance", col
                        ),
                        issuance_of_capital_stock=_val(cashflow, "Issuance Of Capital Stock", col),
                        repayment_of_debt=_val(cashflow, "Repayment Of Debt", col),
                        interest_paid=_val(cashflow, "Interest Paid Supplemental Data", col),
                    )
                )

        return cls(
            ticker=ticker.upper(),
            income=income_stmts,
            balance_sheet=bs_stmts,
            cash_flow=cf_stmts,
        )
