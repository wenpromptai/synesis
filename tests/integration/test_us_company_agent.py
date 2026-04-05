"""Integration tests for USCompanyAnalyst against real AAOI data.

Verifies all three phases of the pipeline produce correct, non-empty results
using live SEC EDGAR + yfinance data. The LLM synthesis phase (Phase 3) is
tested separately since it costs money and takes ~30-40 seconds.

Run with: uv run pytest tests/integration/test_us_company_agent.py -v -m integration
"""

from __future__ import annotations

from datetime import date

import pytest

from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
)
from synesis.processing.intelligence.specialists.us_company.agent import (
    USCompanyDeps,
    _build_financial_health,
    _build_insider_signal,
    _gather_edgar_filings,
    _gather_edgar_insiders,
    _gather_edgar_xbrl,
    _gather_yfinance,
    analyze_company,
)
from synesis.processing.intelligence.specialists.us_company.scoring import (
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

    deps = USCompanyDeps(
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
async def test_gather_edgar_xbrl(real_deps):
    """EDGAR XBRL returns multi-quarter EPS and revenue trends."""
    data = await _gather_edgar_xbrl(real_deps.sec_edgar, TICKER)

    assert len(data["eps_history"]) >= 2, "Should have at least 2 quarters of EPS"
    assert len(data["revenue_history"]) >= 2, "Should have at least 2 quarters of revenue"

    # Verify EPS entry structure
    eps_entry = data["eps_history"][0]
    assert "period" in eps_entry
    assert "actual" in eps_entry
    assert "form" in eps_entry

    # Verify XBRL facts are populated
    facts = data["facts"]
    assert facts is not None
    assert len(facts.facts) > 0
    concept_names = {f.concept for f in facts.facts}
    assert "NetIncomeLoss" in concept_names or "Assets" in concept_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_edgar_insiders(real_deps):
    """EDGAR returns insider transactions and sentiment for AAOI."""
    data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    # AAOI has active insider selling
    txns = data["transactions"]
    assert len(txns) > 0, "AAOI should have insider transactions"

    # Verify transaction structure
    t = txns[0]
    assert t.owner_name
    assert t.transaction_code in ("P", "S")
    assert t.shares > 0
    assert t.transaction_date is not None

    # Sentiment should exist
    sentiment = data["sentiment"]
    assert sentiment is not None
    assert "mspr" in sentiment
    assert "sell_count" in sentiment

    # Late filings
    assert isinstance(data["late_filings"], list)

    # Form 144
    assert isinstance(data["form144"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_edgar_filings(real_deps):
    """EDGAR fetches 10-K and 10-Q filing content."""
    data = await _gather_edgar_filings(real_deps.sec_edgar, TICKER, None)

    # 10-K should exist
    assert data["filing_10k"] is not None, "AAOI should have a 10-K"
    assert data["filing_10k"].form == "10-K"
    assert data["content_10k"] is not None
    assert len(data["content_10k"]) > 10000, "10-K content should be substantial"

    # 10-Q should exist
    assert data["filing_10q"] is not None, "AAOI should have a 10-Q"
    assert data["filing_10q"].form == "10-Q"
    assert data["content_10q"] is not None
    assert len(data["content_10q"]) > 5000, "10-Q content should be substantial"


# ── Phase 2: Deterministic Scoring ───────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_financial_health(real_deps):
    """FinancialHealthScore is populated from real data."""
    yf_data = await _gather_yfinance(real_deps.yfinance, TICKER)
    xbrl_data = await _gather_edgar_xbrl(real_deps.sec_edgar, TICKER)

    health = _build_financial_health(yf_data, xbrl_data)

    assert isinstance(health, FinancialHealthScore)
    # yfinance fields
    assert health.market_cap is not None and health.market_cap > 0
    assert health.roe is not None
    assert health.gross_margin is not None
    # XBRL trends
    assert len(health.quarterly_eps_trend) >= 2
    assert len(health.quarterly_revenue_trend) >= 2
    assert health.latest_filing_period != ""
    # Piotroski should compute (at least partial)
    assert health.piotroski_f is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_insider_signal(real_deps):
    """InsiderSignal correctly aggregates AAOI's insider selling."""
    insider_data = await _gather_edgar_insiders(real_deps.sec_edgar, TICKER)

    signal = _build_insider_signal(insider_data)

    assert isinstance(signal, InsiderSignal)
    # AAOI has active selling (MSPR = -1.0)
    assert signal.sell_count > 0
    assert signal.total_sell_value > 0
    assert signal.mspr is not None and signal.mspr < 0
    assert signal.signal in ("sell", "strong_sell")
    # Notable transactions should be populated
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

    # AAOI has 3+ insiders selling within 14 days
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

    # Filter late filings to recent (last 3 years)
    cutoff = date.today().replace(year=date.today().year - 3)
    late_dicts = [
        {"form_type": lf.form_type, "filed_date": str(lf.filed_date)}
        for lf in insider_data.get("late_filings", [])
        if lf.filed_date >= cutoff
    ]

    flags = detect_red_flags(late_dicts, txn_dicts, {})

    # Should detect insider selling cluster at minimum
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
    assert result.insider_signal.signal in ("sell", "strong_sell")
    assert result.insider_signal.cluster_detected is True

    assert len(result.red_flags) > 0
    assert any(rf.flag == "insider_selling_cluster" for rf in result.red_flags)

    # Qualitative (Phase 3 — LLM output)
    assert len(result.business_summary) > 50, "business_summary should be substantive"
    assert len(result.earnings_quality) > 50, "earnings_quality should be substantive"
    assert len(result.risk_assessment) > 50, "risk_assessment should be substantive"
    assert len(result.geographic_exposure) > 20, "geographic_exposure should be substantive"
    assert len(result.key_customers_suppliers) > 20, "key_customers_suppliers should be substantive"
    assert len(result.insider_vs_financials) > 50, "insider_vs_financials should be substantive"
    assert len(result.disclosure_consistency) > 20, "disclosure_consistency should be substantive"

    # Synthesis
    assert result.overall_signal in ("strong_buy", "buy", "neutral", "sell", "strong_sell")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.primary_thesis) > 20
    assert len(result.key_risks) >= 1
    assert len(result.monitoring_triggers) >= 1
