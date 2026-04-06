"""CompanyAnalyst — comprehensive company analysis via SEC EDGAR + yfinance.

Three-phase pipeline per ticker:
1. Data gathering (no LLM): yfinance fundamentals + quarterly financials, EDGAR insiders/filings
2. Deterministic scoring (no LLM): Piotroski, insider clusters, red flags
3. LLM synthesis: PydanticAI agent interprets scores + filing prose → CompanyAnalysis
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
    RedFlag,
    SignalDirection,
)
from synesis.processing.intelligence.specialists.company.scoring import (
    compute_piotroski_f,
    detect_insider_cluster,
    detect_red_flags,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.providers.yfinance.models import QuarterlyFinancials

logger = get_logger(__name__)


@dataclass
class CompanyDeps:
    """Dependencies for CompanyAnalyst."""

    sec_edgar: SECEdgarClient
    yfinance: YFinanceClient
    crawler: Crawl4AICrawlerProvider | None = None
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())


# ── Phase 1: Data Gathering ──────────────────────────────────────


async def _gather_yfinance(yf: YFinanceClient, ticker: str) -> dict[str, Any]:
    """Fetch yfinance fundamentals, quote, and quarterly financials."""
    fundamentals = await yf.get_fundamentals(ticker)
    quote = await yf.get_quote(ticker)
    quarterly = await yf.get_quarterly_financials(ticker)
    return {"fundamentals": fundamentals, "quote": quote, "quarterly": quarterly}


async def _gather_edgar_insiders(edgar: SECEdgarClient, ticker: str) -> dict[str, Any]:
    """Fetch insider transactions, sentiment, Form 144, late filings."""
    transactions = await edgar.get_insider_transactions(ticker, limit=20, codes=["P", "S"])
    sentiment = await edgar.get_insider_sentiment(ticker)
    form144 = await edgar.get_form144_filings(ticker, limit=10)
    late_filings = await edgar.get_late_filing_alerts(ticker)
    return {
        "transactions": transactions,
        "sentiment": sentiment,
        "form144": form144,
        "late_filings": late_filings,
    }


async def _gather_edgar_filings(
    edgar: SECEdgarClient,
    ticker: str,
    crawler: Crawl4AICrawlerProvider | None,
) -> dict[str, Any]:
    """Fetch latest 10-K and 10-Q filing content."""
    filings_10k = await edgar.get_filings(ticker, form_types=["10-K"], limit=1)
    filings_10q = await edgar.get_filings(ticker, form_types=["10-Q"], limit=1)

    content_10k = None
    content_10q = None
    filing_10k_meta = None
    filing_10q_meta = None

    if filings_10k:
        filing_10k_meta = filings_10k[0]
        content_10k = await edgar.get_filing_content(filings_10k[0].url, crawler=crawler)

    if filings_10q:
        filing_10q_meta = filings_10q[0]
        content_10q = await edgar.get_filing_content(filings_10q[0].url, crawler=crawler)

    return {
        "content_10k": content_10k,
        "content_10q": content_10q,
        "filing_10k": filing_10k_meta,
        "filing_10q": filing_10q_meta,
    }


# ── Phase 2: Deterministic Scoring ──────────────────────────────


def _safe_div(a: float | None, b: float | None) -> float | None:
    """Safe division returning None if either operand is None or divisor is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _build_financial_health(yf_data: dict[str, Any]) -> FinancialHealthScore:
    """Build FinancialHealthScore from yfinance fundamentals + quarterly data."""
    fundamentals = yf_data["fundamentals"]
    quarterly: QuarterlyFinancials = yf_data["quarterly"]

    # Extract current and previous quarter for Piotroski scoring
    inc = quarterly.income
    bs = quarterly.balance_sheet
    cf = quarterly.cash_flow

    cur_inc = inc[0] if inc else None
    prev_inc = inc[1] if len(inc) > 1 else None
    cur_bs = bs[0] if bs else None
    prev_bs = bs[1] if len(bs) > 1 else None
    cur_cf = cf[0] if cf else None

    # Compute Piotroski from quarterly data
    cur_assets = cur_bs.total_assets if cur_bs else None
    prev_assets = prev_bs.total_assets if prev_bs else None
    cur_revenue = cur_inc.total_revenue if cur_inc else None
    prev_revenue = prev_inc.total_revenue if prev_inc else None

    piotroski = compute_piotroski_f(
        net_income_current=cur_inc.net_income if cur_inc else None,
        operating_cf_current=cur_cf.operating_cash_flow if cur_cf else None,
        roa_current=_safe_div(cur_inc.net_income if cur_inc else None, cur_assets),
        roa_previous=_safe_div(prev_inc.net_income if prev_inc else None, prev_assets),
        long_term_debt_current=cur_bs.long_term_debt if cur_bs else None,
        long_term_debt_previous=prev_bs.long_term_debt if prev_bs else None,
        current_ratio_current=(
            _safe_div(cur_bs.current_assets, cur_bs.current_liabilities) if cur_bs else None
        ),
        current_ratio_previous=(
            _safe_div(prev_bs.current_assets, prev_bs.current_liabilities) if prev_bs else None
        ),
        shares_outstanding_current=cur_bs.ordinary_shares_number if cur_bs else None,
        shares_outstanding_previous=prev_bs.ordinary_shares_number if prev_bs else None,
        gross_margin_current=_safe_div(cur_inc.gross_profit if cur_inc else None, cur_revenue),
        gross_margin_previous=_safe_div(prev_inc.gross_profit if prev_inc else None, prev_revenue),
        asset_turnover_current=_safe_div(cur_revenue, cur_assets),
        asset_turnover_previous=_safe_div(prev_revenue, prev_assets),
    )

    # Build quarterly trends for LLM context
    eps_trend = [
        {"period": str(q.period), "actual": q.basic_eps, "revenue": q.total_revenue}
        for q in inc
        if q.basic_eps is not None
    ]
    revenue_trend = [
        {"period": str(q.period), "actual": q.total_revenue} for q in inc if q.total_revenue
    ]

    latest_period = str(inc[0].period) if inc else ""

    return FinancialHealthScore(
        market_cap=fundamentals.market_cap,
        beta=fundamentals.beta,
        current_ratio=fundamentals.current_ratio,
        quick_ratio=fundamentals.quick_ratio,
        debt_to_equity=fundamentals.debt_to_equity,
        roe=fundamentals.roe,
        roa=fundamentals.roa,
        gross_margin=fundamentals.gross_margin,
        operating_margin=fundamentals.operating_margin,
        profit_margin=fundamentals.profit_margin,
        revenue_growth=fundamentals.revenue_growth,
        free_cash_flow=fundamentals.free_cash_flow,
        ebitda=fundamentals.ebitda,
        total_cash=fundamentals.total_cash,
        total_debt=fundamentals.total_debt,
        short_percent_of_float=fundamentals.short_percent_of_float,
        price_to_book=fundamentals.price_to_book,
        ev_to_ebitda=fundamentals.ev_to_ebitda,
        forward_eps=fundamentals.forward_eps,
        piotroski_f=piotroski,
        beneish_m=None,  # Needs 2yr comparative data — future enhancement
        quarterly_eps_trend=eps_trend,
        quarterly_revenue_trend=revenue_trend,
        latest_filing_period=latest_period,
    )


def _build_insider_signal(insider_data: dict[str, Any]) -> InsiderSignal:
    """Build InsiderSignal from EDGAR insider data."""
    transactions = insider_data["transactions"]
    sentiment = insider_data.get("sentiment") or {}
    form144 = insider_data.get("form144", [])

    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in transactions
        if t.owner_name
    ]

    sells = [t for t in txn_dicts if t["transaction_code"] == "S"]
    buys = [t for t in txn_dicts if t["transaction_code"] == "P"]
    cluster = detect_insider_cluster(sells) or detect_insider_cluster(buys)

    # C-suite activity summary
    csuite_keywords = {"CEO", "CFO", "COO", "CTO", "President"}
    csuite_txns = [
        t for t in transactions if any(kw in (t.owner_relationship or "") for kw in csuite_keywords)
    ]
    csuite_summary = ""
    if csuite_txns:
        parts = []
        for t in csuite_txns[:3]:
            action = "bought" if t.transaction_code == "P" else "sold"
            price_str = f" @ ${t.price_per_share:.2f}" if t.price_per_share else ""
            parts.append(
                f"{t.owner_name} ({t.owner_relationship}) {action} "
                f"{int(t.shares):,} shares{price_str} on {t.transaction_date}"
            )
        csuite_summary = "; ".join(parts)

    # Notable transactions (top by dollar value)
    notable = []
    for t in sorted(
        transactions,
        key=lambda x: (x.shares or 0) * (x.price_per_share or 0),
        reverse=True,
    )[:5]:
        action = "bought" if t.transaction_code == "P" else "sold"
        price_str = f" @ ${t.price_per_share:.2f}" if t.price_per_share else ""
        notable.append(
            f"{t.owner_name} ({t.owner_relationship}) {action} "
            f"{int(t.shares):,} shares{price_str} on {t.transaction_date}"
        )

    # Signal from MSPR
    mspr = sentiment.get("mspr")
    if mspr is not None:
        if mspr >= 0.5:
            signal: SignalDirection = "strong_buy"
        elif mspr >= 0.1:
            signal = "buy"
        elif mspr <= -0.5:
            signal = "strong_sell"
        elif mspr <= -0.1:
            signal = "sell"
        else:
            signal = "neutral"
    else:
        signal = "neutral"

    return InsiderSignal(
        mspr=mspr,
        buy_count=sentiment.get("buy_count", 0),
        sell_count=sentiment.get("sell_count", 0),
        total_buy_value=sentiment.get("total_buy_value", 0.0),
        total_sell_value=sentiment.get("total_sell_value", 0.0),
        cluster_detected=cluster,
        csuite_activity=csuite_summary,
        form144_count=len(form144),
        notable_transactions=notable,
        signal=signal,
    )


# ── Phase 3: LLM Synthesis ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior equity research analyst specializing in fundamental analysis of US public companies.
You combine quantitative rigor with qualitative judgment from SEC filings.

Today's date: {current_date}

## Your Analytical Framework

1. EARNINGS QUALITY (most important)
   - Cash flow from operations vs reported net income (accrual quality)
   - Revenue recognition patterns: is growth real or pulled forward?
   - Non-GAAP vs GAAP divergence: are adjustments growing?

2. FINANCIAL STRENGTH
   - Pre-computed scores are provided. Interpret them in context.
   - Piotroski F-Score: 7-9 strong, 4-6 moderate, 0-3 weak
   - Beneish M-Score: >-1.78 suggests possible earnings manipulation (if available)

3. INSIDER CROSS-REFERENCING (your unique edge)
   - Compare insider buying/selling patterns against financial trends
   - C-suite buying during price weakness = strong conviction signal
   - C-suite selling during growth claims = potential red flag
   - Cluster buying/selling (3+ insiders within 14 days) = high-signal event

4. RED FLAG DETECTION
   - Late filings (NT 10-K / NT 10-Q)
   - Cash flow divergence (profit but no cash)
   - Insider selling clusters before negative announcements

5. FILING PROSE ANALYSIS
   - Extract geographic exposure, customer/supplier concentration
   - Assess competitive moat and business quality from 10-K descriptions
   - Compare MD&A tone against actual financial numbers
   - Newer filings should be weighted MORE heavily than older ones

## Rules
- NEVER fabricate financial data. If unavailable, say "not available"
- Always cite which filing period (Q1 2025, FY 2024) data comes from
- Cross-reference at least 2 data points before making claims
- If insider activity contradicts financial trends, flag this prominently
- Confidence calibration:
  90-100%: Overwhelming evidence, multiple confirming signals
  70-89%: Strong evidence with minor uncertainties
  50-69%: Mixed signals, reasonable arguments both ways
  <50%: Insufficient data or highly conflicting signals
"""


class _LLMAnalysisOutput(BaseModel):
    """Fields the LLM fills in. Merged with pre-computed data to form CompanyAnalysis."""

    business_summary: str
    earnings_quality: str
    risk_assessment: str
    geographic_exposure: str
    key_customers_suppliers: str
    insider_vs_financials: str
    disclosure_consistency: str
    overall_signal: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    primary_thesis: str
    key_risks: list[str]
    monitoring_triggers: list[str]


def _build_user_prompt(
    ticker: str,
    yf_data: dict[str, Any],
    financial_health: FinancialHealthScore,
    insider_signal: InsiderSignal,
    red_flags: list[RedFlag],
    filing_data: dict[str, Any],
) -> str:
    """Build the user prompt with all gathered data."""
    fundamentals = yf_data["fundamentals"]
    quote = yf_data["quote"]
    quarterly: QuarterlyFinancials = yf_data["quarterly"]

    sections: list[str] = []

    sections.append(f"# Analysis Request: {ticker}")
    if fundamentals.name:
        sections.append(f"**Company:** {fundamentals.name}")
    if fundamentals.sector or fundamentals.industry:
        sections.append(
            f"**Sector:** {fundamentals.sector or 'N/A'} / {fundamentals.industry or 'N/A'}"
        )
    if quote.last:
        sections.append(f"**Current Price:** ${quote.last:.2f}")

    sections.append("\n## Pre-Computed Financial Health")
    sections.append(financial_health.model_dump_json(indent=2))

    # Include full quarterly financials for LLM to analyze trends
    sections.append("\n## Quarterly Financial Statements")
    sections.append(quarterly.model_dump_json(indent=2))

    sections.append("\n## Insider Activity")
    sections.append(insider_signal.model_dump_json(indent=2))

    if red_flags:
        sections.append("\n## Detected Red Flags")
        for rf in red_flags:
            sections.append(
                f"- **[{rf.severity.upper()}] {rf.flag}** ({rf.category}): {rf.evidence}"
            )
    else:
        sections.append("\n## Detected Red Flags\nNone detected.")

    if filing_data.get("content_10k"):
        meta = filing_data.get("filing_10k")
        label = (
            f"10-K (filed {meta.filed_date}, period ending {meta.report_date})" if meta else "10-K"
        )
        sections.append(f"\n## Latest Annual Filing: {label}")
        content = filing_data["content_10k"]
        if len(content) > 300000:
            content = content[:300000] + "\n\n[... truncated for length ...]"
        sections.append(content)

    if filing_data.get("content_10q"):
        meta = filing_data.get("filing_10q")
        label = (
            f"10-Q (filed {meta.filed_date}, period ending {meta.report_date})" if meta else "10-Q"
        )
        sections.append(f"\n## Latest Quarterly Filing: {label}")
        content = filing_data["content_10q"]
        if len(content) > 200000:
            content = content[:200000] + "\n\n[... truncated for length ...]"
        sections.append(content)

    sections.append(
        "\n## Instructions\n"
        "Analyze this company and produce a complete analysis. "
        "Fill in ALL qualitative fields (business_summary, earnings_quality, "
        "risk_assessment, geographic_exposure, key_customers_suppliers, "
        "insider_vs_financials, disclosure_consistency) plus the synthesis "
        "fields (overall_signal, confidence, primary_thesis, key_risks, "
        "monitoring_triggers). Use the filing prose and quarterly financials "
        "for qualitative insights and cross-reference against the quantitative data."
    )

    return "\n".join(sections)


# ── Public API ───────────────────────────────────────────────────


async def analyze_company(
    ticker: str,
    deps: CompanyDeps,
) -> CompanyAnalysis:
    """Run the full three-phase analysis pipeline for a single ticker."""
    ticker = ticker.upper()
    logger.info("Starting CompanyAnalyst", ticker=ticker)

    # Phase 1: Gather data concurrently (return_exceptions so one failure
    # doesn't discard successful results from the other providers)
    results = await asyncio.gather(
        _gather_yfinance(deps.yfinance, ticker),
        _gather_edgar_insiders(deps.sec_edgar, ticker),
        _gather_edgar_filings(deps.sec_edgar, ticker, deps.crawler),
        return_exceptions=True,
    )

    # yfinance is required — drives Phase 2 scoring
    if isinstance(results[0], BaseException):
        logger.error("yfinance fetch failed — cannot proceed", ticker=ticker, error=str(results[0]))
        raise results[0]
    yf_data = results[0]

    # EDGAR sources degrade gracefully
    _empty_insider: dict[str, Any] = {
        "transactions": [],
        "sentiment": None,
        "form144": [],
        "late_filings": [],
    }
    _empty_filing: dict[str, Any] = {
        "content_10k": None,
        "content_10q": None,
        "filing_10k": None,
        "filing_10q": None,
    }

    if isinstance(results[1], BaseException):
        logger.warning(
            "EDGAR insider fetch failed — proceeding without insider data",
            ticker=ticker,
            error=str(results[1]),
        )
        insider_data = _empty_insider
    else:
        insider_data = results[1]

    if isinstance(results[2], BaseException):
        logger.warning(
            "EDGAR filing fetch failed — proceeding without filing prose",
            ticker=ticker,
            error=str(results[2]),
        )
        filing_data = _empty_filing
    else:
        filing_data = results[2]

    logger.info("Phase 1 complete: data gathered", ticker=ticker)

    # Phase 2: Deterministic scoring
    financial_health = _build_financial_health(yf_data)
    insider_signal = _build_insider_signal(insider_data)

    # Only flag late filings from the last 2 years as red flags
    cutoff = deps.current_date - timedelta(days=2 * 365)
    late_filing_dicts = [
        {"form_type": lf.form_type, "filed_date": str(lf.filed_date)}
        for lf in insider_data.get("late_filings", [])
        if lf.filed_date >= cutoff
    ]
    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in insider_data.get("transactions", [])
    ]

    # Red flag: cash flow divergence from quarterly data
    financial_dict: dict[str, Any] = {}
    quarterly = yf_data["quarterly"]
    if quarterly.income and quarterly.cash_flow:
        ni = quarterly.income[0].net_income
        opcf = quarterly.cash_flow[0].operating_cash_flow
        if ni is not None:
            financial_dict["net_income"] = ni
        if opcf is not None:
            financial_dict["operating_cf"] = opcf

    red_flags = detect_red_flags(late_filing_dicts, txn_dicts, financial_dict)

    logger.info(
        "Phase 2 complete: scoring done",
        ticker=ticker,
        piotroski=financial_health.piotroski_f,
        insider_signal=insider_signal.signal,
        red_flags=len(red_flags),
    )

    # Phase 3: LLM synthesis
    agent = Agent(
        model=create_model(tier="vsmart"),
        output_type=_LLMAnalysisOutput,
        system_prompt=SYSTEM_PROMPT.format(current_date=deps.current_date),
    )

    user_prompt = _build_user_prompt(
        ticker, yf_data, financial_health, insider_signal, red_flags, filing_data
    )

    # Assemble metadata before LLM call (needed for both success and fallback)
    fundamentals = yf_data["fundamentals"]
    company_info = await deps.sec_edgar.get_company_info(ticker)
    company_name = company_info.name if company_info else (fundamentals.name or ticker)

    filing_10k = filing_data.get("filing_10k")
    latest_filing = ""
    if filing_10k:
        year = filing_10k.report_date.year if filing_10k.report_date else "?"
        latest_filing = f"10-K FY{year}, filed {filing_10k.filed_date}"

    try:
        result = await agent.run(user_prompt)
        llm_output = result.output
    except Exception:
        logger.exception("LLM synthesis failed", ticker=ticker)
        return CompanyAnalysis(
            ticker=ticker,
            company_name=company_name,
            sector=fundamentals.sector or "",
            industry=fundamentals.industry or "",
            analysis_date=deps.current_date,
            latest_annual_filing=latest_filing,
            financial_health=financial_health,
            insider_signal=insider_signal,
            red_flags=red_flags,
            business_summary="[LLM synthesis failed — qualitative analysis unavailable]",
        )

    logger.info("Phase 3 complete: LLM synthesis done", ticker=ticker)

    return CompanyAnalysis(
        ticker=ticker,
        company_name=company_name,
        sector=fundamentals.sector or "",
        industry=fundamentals.industry or "",
        analysis_date=deps.current_date,
        latest_annual_filing=latest_filing,
        financial_health=financial_health,
        insider_signal=insider_signal,
        red_flags=red_flags,
        business_summary=llm_output.business_summary,
        earnings_quality=llm_output.earnings_quality,
        risk_assessment=llm_output.risk_assessment,
        geographic_exposure=llm_output.geographic_exposure,
        key_customers_suppliers=llm_output.key_customers_suppliers,
        insider_vs_financials=llm_output.insider_vs_financials,
        disclosure_consistency=llm_output.disclosure_consistency,
        overall_signal=llm_output.overall_signal,
        confidence=llm_output.confidence,
        primary_thesis=llm_output.primary_thesis,
        key_risks=llm_output.key_risks,
        monitoring_triggers=llm_output.monitoring_triggers,
    )
