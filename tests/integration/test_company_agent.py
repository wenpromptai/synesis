"""Integration tests for CompanyAnalyst against real AAOI data.

Verifies all three phases of the pipeline produce correct, non-empty results
using live SEC EDGAR + yfinance data. The LLM synthesis phase (Phase 3) is
tested separately since it costs money and takes ~30-40 seconds.

Run with: uv run pytest tests/integration/test_company_agent.py -v -m integration
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
)
from synesis.processing.intelligence.specialists.company.agent import (
    CompanyDeps,
    _build_financial_health,
    _build_insider_signal,
    _gather_edgar_filings,
    _gather_edgar_insiders,
    _gather_yfinance,
    analyze_company,
)
from synesis.processing.intelligence.specialists.company.scoring import (
    detect_insider_cluster,
    detect_red_flags,
)

TICKER = "AAOI"


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
async def real_deps(real_redis):
    """Real provider clients for integration testing."""
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient

    edgar = SECEdgarClient(redis=real_redis)
    yfinance = YFinanceClient(redis=real_redis)

    deps = CompanyDeps(
        sec_edgar=edgar,
        yfinance=yfinance,
        crawler=None,
        current_date=date.today(),
    )
    yield deps
    await edgar.close()


# ── Phase 1: Data Gathering ──────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_yfinance(real_deps):
    """yfinance returns fundamentals + quote with key fields populated."""
    data = await _gather_yfinance(real_deps.yfinance, TICKER)

    fund = data["fundamentals"]
    assert fund.ticker == TICKER
    assert fund.name is not None
    assert fund.sector == "Technology"
    assert fund.market_cap is not None and fund.market_cap > 0
    assert fund.roe is not None
    assert fund.current_ratio is not None

    quote = data["quote"]
    assert quote.ticker == TICKER
    assert quote.last is not None and quote.last > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_yfinance_quarterly(real_deps):
    """yfinance returns quarterly income, balance sheet, and cash flow."""
    data = await _gather_yfinance(real_deps.yfinance, TICKER)
    quarterly = data["quarterly"]

    assert len(quarterly.income) >= 3, "Should have at least 3 quarters of income data"
    assert len(quarterly.balance_sheet) >= 3, "Should have at least 3 quarters of balance sheet"
    assert len(quarterly.cash_flow) >= 3, "Should have at least 3 quarters of cash flow"

    # Verify income statement fields
    inc = quarterly.income[0]
    assert inc.total_revenue is not None and inc.total_revenue > 0
    assert inc.net_income is not None
    assert inc.basic_eps is not None
    assert inc.gross_profit is not None

    # Verify balance sheet fields
    bs = quarterly.balance_sheet[0]
    assert bs.total_assets is not None and bs.total_assets > 0
    assert bs.total_liabilities is not None
    assert bs.stockholders_equity is not None
    assert bs.ordinary_shares_number is not None

    # Verify cash flow fields
    cf = quarterly.cash_flow[0]
    assert cf.operating_cash_flow is not None
    assert cf.free_cash_flow is not None
    assert cf.capital_expenditure is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_edgar_insiders(real_deps):
    """EDGAR returns insider transactions and sentiment for AAOI."""
    data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    txns = data["transactions"]
    assert len(txns) > 0, "AAOI should have insider transactions"

    t = txns[0]
    assert t.owner_name
    assert t.transaction_code in ("P", "S")
    assert t.shares > 0
    assert t.transaction_date is not None

    sentiment = data["sentiment"]
    assert sentiment is not None
    assert "mspr" in sentiment
    assert "sell_count" in sentiment

    assert isinstance(data["late_filings"], list)
    assert isinstance(data["form144"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_edgar_filings(real_deps):
    """EDGAR fetches 10-K and 10-Q filing content."""
    data = await _gather_edgar_filings(real_deps.sec_edgar, TICKER, None)

    assert data["filing_10k"] is not None, "AAOI should have a 10-K"
    assert data["filing_10k"].form == "10-K"
    assert data["content_10k"] is not None
    assert len(data["content_10k"]) > 10000, "10-K content should be substantial"

    assert data["filing_10q"] is not None, "AAOI should have a 10-Q"
    assert data["filing_10q"].form == "10-Q"
    assert data["content_10q"] is not None
    assert len(data["content_10q"]) > 5000, "10-Q content should be substantial"


# ── Phase 2: Deterministic Scoring ───────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_financial_health(real_deps):
    """FinancialHealthScore is populated from yfinance data."""
    yf_data = await _gather_yfinance(real_deps.yfinance, TICKER)

    health = _build_financial_health(yf_data)

    assert isinstance(health, FinancialHealthScore)
    assert health.market_cap is not None and health.market_cap > 0
    assert health.roe is not None
    assert health.gross_margin is not None
    assert len(health.quarterly_eps_trend) >= 2
    assert len(health.quarterly_revenue_trend) >= 2
    assert health.latest_filing_period != ""
    assert health.piotroski_f is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_insider_signal(real_deps):
    """InsiderSignal correctly aggregates AAOI's insider selling."""
    insider_data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    signal = _build_insider_signal(insider_data)

    assert isinstance(signal, InsiderSignal)
    assert signal.sell_count > 0
    assert signal.total_sell_value > 0
    assert signal.mspr is not None and signal.mspr < 0
    assert signal.mspr < 0  # bearish (heavy selling)
    assert len(signal.notable_transactions) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insider_cluster_detection(real_deps):
    """Cluster detection finds AAOI's recent insider selling cluster."""
    insider_data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in insider_data["transactions"]
        if t.owner_name and t.transaction_code == "S"
    ]

    assert detect_insider_cluster(txn_dicts) is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_red_flag_detection(real_deps):
    """Red flags are detected from real AAOI data."""
    insider_data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in insider_data["transactions"]
        if t.owner_name
    ]

    cutoff = date.today() - timedelta(days=2 * 365)
    late_dicts = [
        {"form_type": lf.form_type, "filed_date": str(lf.filed_date)}
        for lf in insider_data.get("late_filings", [])
        if lf.filed_date >= cutoff
    ]

    flags = detect_red_flags(late_dicts, txn_dicts, {})

    flag_names = [f.flag for f in flags]
    assert "insider_selling_cluster" in flag_names


# ── Phase 3: Full Pipeline (includes LLM call) ──────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline(real_deps):
    """Full three-phase pipeline produces a valid CompanyAnalysis.

    This test calls the LLM (OpenAI 5.2) and costs ~$0.10-0.30.
    It takes ~30-60 seconds to complete.
    """
    result = await analyze_company(TICKER, real_deps)

    assert isinstance(result, CompanyAnalysis)

    # Identity
    assert result.ticker == TICKER
    assert result.company_name
    assert result.sector == "Technology"
    assert result.analysis_date == date.today()
    assert "10-K" in result.latest_annual_filing

    # Quantitative (Phase 2)
    assert result.financial_health.market_cap is not None
    assert result.financial_health.market_cap > 0
    assert result.financial_health.piotroski_f is not None
    assert len(result.financial_health.quarterly_eps_trend) >= 2

    assert result.insider_signal.sell_count > 0
    assert result.insider_signal.mspr is not None and result.insider_signal.mspr < 0  # bearish
    assert result.insider_signal.cluster_detected is True

    assert len(result.red_flags) > 0
    assert any(rf.flag == "insider_selling_cluster" for rf in result.red_flags)

    # Qualitative (Phase 3 — LLM output)
    assert len(result.business_summary) > 50
    assert len(result.earnings_quality) > 50
    assert len(result.risk_assessment) > 50
    assert len(result.geographic_exposure) > 20
    assert len(result.key_customers_suppliers) > 20
    assert len(result.insider_vs_financials) > 50
    assert len(result.disclosure_consistency) > 20

    # Synthesis
    assert len(result.primary_thesis) > 20
    assert len(result.key_risks) >= 1
    assert len(result.monitoring_triggers) >= 1
