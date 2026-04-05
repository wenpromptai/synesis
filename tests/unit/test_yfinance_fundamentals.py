"""Tests for CompanyFundamentals model in yfinance provider."""

from __future__ import annotations


import pytest

from synesis.providers.yfinance.models import CompanyFundamentals

# ---------------------------------------------------------------------------
# Sample yfinance .info dict (AXTI-like)
# ---------------------------------------------------------------------------

AXTI_INFO = {
    "symbol": "AXTI",
    "shortName": "AXT Inc",
    "sector": "Technology",
    "industry": "Semiconductor Equipment & Materials",
    "fullTimeEmployees": 2300,
    "longBusinessSummary": "AXT, Inc. designs and develops compound and single element semiconductors.",
    "marketCap": 230000000,
    "beta": 1.45,
    "totalRevenue": 120000000,
    "ebitda": 15000000,
    "totalCash": 40000000,
    "totalDebt": 25000000,
    "freeCashflow": 8000000,
    "currentRatio": 3.2,
    "quickRatio": 2.1,
    "debtToEquity": 18.5,
    "returnOnEquity": 0.08,
    "returnOnAssets": 0.04,
    "grossMargins": 0.32,
    "operatingMargins": 0.07,
    "profitMargins": 0.05,
    "revenueGrowth": 0.12,
    "priceToBook": 1.8,
    "priceToSalesTrailing12Months": 1.9,
    "enterpriseToEbitda": 14.2,
    "enterpriseToRevenue": 1.7,
    "forwardEps": 0.35,
    "trailingEps": 0.28,
    "sharesShort": 5000000,
    "shortPercentOfFloat": 0.06,
    "targetMeanPrice": 8.5,
    "targetHighPrice": 12.0,
    "targetLowPrice": 5.0,
    "numberOfAnalystOpinions": 4,
    "heldPercentInsiders": 0.12,
    "heldPercentInstitutions": 0.55,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_company_fundamentals_from_yfinance_info() -> None:
    """Parse a real AXTI-like info dict into CompanyFundamentals."""
    fund = CompanyFundamentals.from_yfinance_info("AXTI", AXTI_INFO)

    assert fund.ticker == "AXTI"
    assert fund.name == "AXT Inc"
    assert fund.sector == "Technology"
    assert fund.industry == "Semiconductor Equipment & Materials"
    assert fund.employees == 2300
    assert (
        fund.business_summary
        == "AXT, Inc. designs and develops compound and single element semiconductors."
    )

    assert fund.market_cap == pytest.approx(230000000.0)
    assert fund.beta == pytest.approx(1.45)
    assert fund.total_revenue == pytest.approx(120000000.0)
    assert fund.ebitda == pytest.approx(15000000.0)
    assert fund.total_cash == pytest.approx(40000000.0)
    assert fund.total_debt == pytest.approx(25000000.0)
    assert fund.free_cash_flow == pytest.approx(8000000.0)

    assert fund.current_ratio == pytest.approx(3.2)
    assert fund.quick_ratio == pytest.approx(2.1)
    assert fund.debt_to_equity == pytest.approx(18.5)
    assert fund.roe == pytest.approx(0.08)
    assert fund.roa == pytest.approx(0.04)
    assert fund.gross_margin == pytest.approx(0.32)
    assert fund.operating_margin == pytest.approx(0.07)
    assert fund.profit_margin == pytest.approx(0.05)
    assert fund.revenue_growth == pytest.approx(0.12)

    assert fund.price_to_book == pytest.approx(1.8)
    assert fund.price_to_sales == pytest.approx(1.9)
    assert fund.ev_to_ebitda == pytest.approx(14.2)
    assert fund.ev_to_revenue == pytest.approx(1.7)

    assert fund.forward_eps == pytest.approx(0.35)
    assert fund.trailing_eps == pytest.approx(0.28)
    assert fund.shares_short == 5000000
    assert fund.short_percent_of_float == pytest.approx(0.06)

    assert fund.analyst_target_mean == pytest.approx(8.5)
    assert fund.analyst_target_high == pytest.approx(12.0)
    assert fund.analyst_target_low == pytest.approx(5.0)
    assert fund.analyst_count == 4

    assert fund.held_percent_insiders == pytest.approx(0.12)
    assert fund.held_percent_institutions == pytest.approx(0.55)


def test_company_fundamentals_handles_missing_fields() -> None:
    """Empty dict should produce a model with all optional fields as None."""
    fund = CompanyFundamentals.from_yfinance_info("EMPTY", {})

    assert fund.ticker == "EMPTY"
    assert fund.name is None
    assert fund.sector is None
    assert fund.industry is None
    assert fund.employees is None
    assert fund.business_summary is None
    assert fund.market_cap is None
    assert fund.beta is None
    assert fund.total_revenue is None
    assert fund.ebitda is None
    assert fund.total_cash is None
    assert fund.total_debt is None
    assert fund.free_cash_flow is None
    assert fund.current_ratio is None
    assert fund.quick_ratio is None
    assert fund.debt_to_equity is None
    assert fund.roe is None
    assert fund.roa is None
    assert fund.gross_margin is None
    assert fund.operating_margin is None
    assert fund.profit_margin is None
    assert fund.revenue_growth is None
    assert fund.price_to_book is None
    assert fund.price_to_sales is None
    assert fund.ev_to_ebitda is None
    assert fund.ev_to_revenue is None
    assert fund.forward_eps is None
    assert fund.trailing_eps is None
    assert fund.shares_short is None
    assert fund.short_percent_of_float is None
    assert fund.analyst_target_mean is None
    assert fund.analyst_target_high is None
    assert fund.analyst_target_low is None
    assert fund.analyst_count is None
    assert fund.held_percent_insiders is None
    assert fund.held_percent_institutions is None


def test_company_fundamentals_handles_nan() -> None:
    """NaN, Inf, -Inf, and non-numeric values should all become None."""
    noisy_info = {
        "shortName": "Noisy Corp",
        "beta": float("nan"),
        "marketCap": float("inf"),
        "totalRevenue": float("-inf"),
        "currentRatio": "N/A",
        "debtToEquity": None,
        "fullTimeEmployees": float("nan"),
        "numberOfAnalystOpinions": "unknown",
    }
    fund = CompanyFundamentals.from_yfinance_info("NOISY", noisy_info)

    assert fund.ticker == "NOISY"
    assert fund.name == "Noisy Corp"
    assert fund.beta is None
    assert fund.market_cap is None
    assert fund.total_revenue is None
    assert fund.current_ratio is None
    assert fund.debt_to_equity is None
    assert fund.employees is None
    assert fund.analyst_count is None
