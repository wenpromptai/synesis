"""CompanyAnalyst — comprehensive company analysis via SEC EDGAR + yfinance.

Three-phase pipeline per ticker:
1. Data gathering (no LLM): yfinance fundamentals + quarterly financials, EDGAR insiders/filings
2. Deterministic scoring (no LLM): Piotroski, insider clusters, red flags
3. LLM synthesis: PydanticAI agent interprets scores + filing prose → CompanyAnalysis
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
    RedFlag,
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
    """Fetch yfinance fundamentals and quarterly financials."""
    fundamentals = await yf.get_fundamentals(ticker)
    quarterly = await yf.get_quarterly_financials(ticker)
    return {"fundamentals": fundamentals, "quarterly": quarterly}


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
    """Fetch latest annual/quarterly filings and recent material 8-K/6-K filings.

    Tries 10-K/10-Q first (domestic issuers). If neither is found, falls back
    to 20-F/6-K (foreign private issuers like ADRs, Israeli/Dutch companies).
    """
    filings_10k = await edgar.get_filings(ticker, form_types=["10-K"], limit=1)
    filings_10q = await edgar.get_filings(ticker, form_types=["10-Q"], limit=1)

    is_foreign = not filings_10k and not filings_10q

    # Foreign private issuer fallback: 20-F (annual) and 6-K (interim)
    if is_foreign:
        logger.info("No 10-K/10-Q found — trying 20-F/6-K (foreign issuer)", ticker=ticker)
        filings_10k = await edgar.get_filings(ticker, form_types=["20-F"], limit=1)
        filings_10q = await edgar.get_filings(ticker, form_types=["6-K"], limit=1)

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

    if not filing_10k_meta and not filing_10q_meta:
        logger.warning("No annual or quarterly filings found", ticker=ticker)

    # Fetch recent material 8-K events (last 5, material items only)
    material_8k_items = ["1.01", "1.02", "2.01", "2.02", "5.02", "7.01", "8.01"]
    try:
        events_8k = await edgar.get_8k_events(
            ticker, items=material_8k_items, limit=5, crawler=crawler
        )
    except Exception:
        logger.warning(
            "8-K event fetch failed — proceeding without 8-K data", ticker=ticker, exc_info=True
        )
        events_8k = []

    return {
        "content_10k": content_10k,
        "content_10q": content_10q,
        "filing_10k": filing_10k_meta,
        "filing_10q": filing_10q_meta,
        "is_foreign_issuer": is_foreign,
        "events_8k": events_8k,
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

    mspr = sentiment.get("mspr")

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
    )


# ── Phase 3: LLM Synthesis ──────────────────────────────────────

SYSTEM_PROMPT = """\
You extract qualitative intelligence from SEC filings, financial data, and insider \
activity — the kind of information NOT available from financial data APIs. Your \
output feeds directly into bull/bear researchers who will form the investment \
thesis — your job is to give them clean, structured, fact-rich context to work \
with. Always preserve specifics: revenue percentages, customer names, filing \
dates, guidance figures, and regulatory details.

Today's date: {current_date}

## What You're Extracting (in priority order)

1. **GROWTH CATALYSTS & FORWARD NARRATIVE** (highest value)
   - Product/technology pipeline and transition timeline (e.g., 800G→1.6T→CPO)
   - AI demand signals, TAM expansion, management revenue/margin projections
   - Forward guidance from earnings 8-K or MD&A
   - New markets, sovereign AI, physical AI, or other emerging revenue streams
   - Capex plans and manufacturing capacity ramp

2. **CUSTOMER & SUPPLIER CONCENTRATION** (critical risk signal)
   - Extract exact percentages: "Customer A = 22% of revenue"
   - Name customers when disclosed (common in small/mid-cap filings)
   - Supply chain single-source dependencies (e.g., sole foundry for advanced nodes)
   - Note if concentration is increasing or decreasing vs prior period

3. **COMPETITIVE POSITION & MOAT**
   - What protects this business? (ecosystem lock-in, IP, scale, vertical integration)
   - Who does management name as competitive threats?
   - Market share context if disclosed
   - Technology differentiation vs commoditized products

4. **GEOGRAPHIC EXPOSURE & REGULATORY RISKS**
   - Revenue breakdown by region with dollar amounts and percentages
   - Caveats that modify geographic data (e.g., billing location vs end-customer)
   - Export controls, tariffs, sanctions — specific to this company
   - Active litigation, regulatory proceedings (EU DMA, DOJ antitrust, etc.)

5. **FINANCIAL HEALTH & EARNINGS QUALITY**
   - Pre-computed scores are provided (Piotroski F-Score: 7-9 strong, 4-6 moderate, 0-3 weak)
   - Cash flow from operations vs reported net income — is profit real?
   - Non-GAAP vs GAAP divergence trends
   - Quarter-over-quarter improvement or deterioration

6. **INSIDER ACTIVITY**
   - Compare insider buying/selling against financial trends and management narrative
   - C-suite buying during weakness = management confidence signal
   - C-suite selling during growth claims = potential red flag
   - Cluster activity (3+ insiders within 14 days) = coordinated action

7. **MATERIAL 8-K EVENTS**
   - Material agreements, M&A, earnings results, officer changes, Reg FD disclosures
   - What happened, when, and the key details (dollar amounts, parties, dates)

## Guidelines
- NEVER fabricate data. If unavailable, say "not available".
- Always cite which filing period (Q1 2025, FY 2024) or 8-K date data comes from.
- Cross-reference at least 2 data points before making claims.
- If insider activity contradicts financial trends or management narrative, flag prominently.
- Focus on what the FILINGS reveal that financial data APIs do NOT provide.
"""


class _LLMAnalysisOutput(BaseModel):
    """Fields the LLM fills in. Merged with pre-computed data to form CompanyAnalysis."""

    business_summary: str
    earnings_quality: str
    risk_assessment: str
    geographic_exposure: str
    key_customers_suppliers: str
    growth_catalysts: str
    competitive_position: str
    insider_vs_financials: str
    disclosure_consistency: str
    primary_thesis: str
    key_risks: list[str]
    monitoring_triggers: list[str]


def _extract_filing_sections(content: str, form_type: str) -> str:
    """Extract high-value sections from SEC filing text.

    10-K: Item 1A (Risk Factors, 40K), Item 7 (MD&A, 40K), Item 1 (Business, 20K)
    10-Q: Part I Item 2 (MD&A, 40K), Part II Item 1A (Risk Factors, 20K)
    20-F: Item 3.D (Risk Factors, 40K), Item 5 (Operating/Financial Review, 40K),
          Item 4 (Company Information, 20K)
    6-K: No standard sections — uses raw truncation fallback.

    All matching sections are extracted and concatenated.
    Falls back to first 50K chars if no section headers found.
    """
    if form_type == "10-K":
        section_patterns = [
            (r"(?i)(?:ITEM\s*1A[\.\s\-—:]+RISK\s*FACTORS)", "Risk Factors", 40_000),
            (r"(?i)(?:ITEM\s*7[\.\s\-—:]+MANAGEMENT.S\s*DISCUSSION)", "MD&A", 40_000),
            (r"(?i)(?:ITEM\s*1[\.\s\-—:]+BUSINESS(?!\s*COMBINATION))", "Business", 20_000),
        ]
    elif form_type == "20-F":
        section_patterns = [
            (r"(?i)(?:ITEM\s*3\.?\s*D[\.\s\-—:]+RISK\s*FACTORS)", "Risk Factors", 40_000),
            (
                r"(?i)(?:ITEM\s*5[\.\s\-—:]+OPERATING\s*AND\s*FINANCIAL\s*REVIEW)",
                "Operating & Financial Review",
                40_000,
            ),
            (
                r"(?i)(?:ITEM\s*4[\.\s\-—:]+INFORMATION\s*ON\s*THE\s*COMPANY)",
                "Company Information",
                20_000,
            ),
        ]
    else:  # 10-Q, 6-K, other
        section_patterns = [
            (
                r"(?i)(?:(?:PART\s*I\s*,?\s*)?ITEM\s*2[\.\s\-—:]+MANAGEMENT.S\s*DISCUSSION)",
                "MD&A",
                40_000,
            ),
            (
                r"(?i)(?:(?:PART\s*II\s*,?\s*)?ITEM\s*1A[\.\s\-—:]+RISK\s*FACTORS)",
                "Risk Factors",
                20_000,
            ),
        ]

    # Pattern to detect start of any ITEM section (used to find section boundaries)
    item_boundary = re.compile(r"(?i)\n\s*(?:PART\s*[IV]+\s*[\.,]?\s*)?ITEM\s*\d+[A-Z]?[\.\s\-—:]")

    extracted_parts: list[str] = []
    for pattern, label, max_chars in section_patterns:
        match = re.search(pattern, content)
        if not match:
            continue
        start = match.start()
        # Find next ITEM header after this section's content starts
        rest = content[match.end() :]
        end_match = item_boundary.search(rest)
        if end_match:
            section_text = content[start : match.end() + end_match.start()]
        else:
            section_text = content[start:]

        if len(section_text) > max_chars:
            section_text = section_text[:max_chars] + "\n[... section truncated ...]"
        extracted_parts.append(f"### {label}\n{section_text}")

    if not extracted_parts:
        logger.info(
            "No ITEM headers found in filing — using raw truncation fallback",
            form_type=form_type,
            content_len=len(content),
        )
        truncated = content[:50_000]
        if len(content) > 50_000:
            truncated += "\n[... truncated for length ...]"
        return truncated

    return "\n\n".join(extracted_parts)


def _format_quarterly_summary(quarterly: "QuarterlyFinancials") -> str:
    """Format a compact quarterly summary instead of full JSON dump.

    Shows key metrics per quarter: Revenue, EPS, Operating Income, FCF.
    """
    lines: list[str] = []

    for i, inc in enumerate(quarterly.income[:4]):
        period = str(inc.period)
        parts = [f"**{period}**"]

        if inc.total_revenue is not None:
            parts.append(f"Rev=${inc.total_revenue / 1e9:.2f}B")
        if inc.basic_eps is not None:
            parts.append(f"EPS=${inc.basic_eps:.2f}")
        if inc.operating_income is not None:
            parts.append(f"OpInc=${inc.operating_income / 1e9:.2f}B")
        if inc.net_income is not None:
            parts.append(f"NI=${inc.net_income / 1e9:.2f}B")
        if inc.gross_profit is not None and inc.total_revenue:
            parts.append(f"GM={inc.gross_profit / inc.total_revenue:.1%}")

        # Match cash flow by quarter index
        if i < len(quarterly.cash_flow):
            cf = quarterly.cash_flow[i]
            if cf.free_cash_flow is not None:
                parts.append(f"FCF=${cf.free_cash_flow / 1e9:.2f}B")
            if cf.operating_cash_flow is not None:
                parts.append(f"OpCF=${cf.operating_cash_flow / 1e9:.2f}B")
            if cf.stock_based_compensation is not None:
                parts.append(f"SBC=${cf.stock_based_compensation / 1e9:.2f}B")

        lines.append("- " + ", ".join(parts))

    return "\n".join(lines) if lines else "No quarterly data available."


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
    quarterly: QuarterlyFinancials = yf_data["quarterly"]

    sections: list[str] = []

    sections.append(f"# Analysis Request: {ticker}")
    if fundamentals.name:
        sections.append(f"**Company:** {fundamentals.name}")
    if fundamentals.sector or fundamentals.industry:
        sections.append(
            f"**Sector:** {fundamentals.sector or 'N/A'} / {fundamentals.industry or 'N/A'}"
        )

    sections.append("\n## Pre-Computed Financial Health")
    sections.append(financial_health.model_dump_json(indent=2))

    sections.append("\n## Quarterly Financial Summary (last 4 quarters)")
    sections.append(_format_quarterly_summary(quarterly))

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

    is_foreign = filing_data.get("is_foreign_issuer", False)
    annual_form = "20-F" if is_foreign else "10-K"
    quarterly_form = "6-K" if is_foreign else "10-Q"

    if filing_data.get("content_10k"):
        meta = filing_data.get("filing_10k")
        label = (
            f"{annual_form} (filed {meta.filed_date}, period ending {meta.report_date})"
            if meta
            else annual_form
        )
        sections.append(f"\n## Latest Annual Filing: {label}")
        sections.append(_extract_filing_sections(filing_data["content_10k"], annual_form))

    if filing_data.get("content_10q"):
        meta = filing_data.get("filing_10q")
        label = (
            f"{quarterly_form} (filed {meta.filed_date}, period ending {meta.report_date})"
            if meta
            else quarterly_form
        )
        sections.append(f"\n## Latest Quarterly Filing: {label}")
        sections.append(_extract_filing_sections(filing_data["content_10q"], quarterly_form))

    # Material 8-K events (most time-sensitive filings)
    events_8k = filing_data.get("events_8k", [])
    if events_8k:
        sections.append("\n## Recent Material 8-K Events")
        for event in events_8k:
            items_str = ", ".join(event.item_descriptions) if event.item_descriptions else "Other"
            sections.append(f"\n### 8-K ({event.filed_date}) — {items_str}")
            if event.content:
                # Truncate 8-K content to keep context manageable
                content = event.content
                if len(content) > 10_000:
                    content = content[:10_000] + "\n[... truncated ...]"
                sections.append(content)

    sections.append(
        "\n## Instructions\n"
        "Produce a thorough company analysis. For each field:\n"
        "- **business_summary**: What they do, market position, key products/services\n"
        "- **earnings_quality**: Cash conversion, revenue recognition, GAAP vs non-GAAP\n"
        "- **risk_assessment**: Key risks from filings (regulatory, geopolitical, operational)\n"
        "- **geographic_exposure**: Revenue by region with % and caveats\n"
        "- **key_customers_suppliers**: Concentration % (e.g., 'Customer A = 22% of revenue'), "
        "supply chain dependencies\n"
        "- **growth_catalysts**: Product pipeline, AI/tech demand signals, forward guidance, "
        "TAM expansion, management projections, upcoming catalysts\n"
        "- **competitive_position**: Moat (ecosystem, IP, scale), competitive threats, "
        "market share context\n"
        "- **insider_vs_financials**: Do insider actions align with management narrative?\n"
        "- **disclosure_consistency**: Does MD&A match reality? Any red flags in tone?\n"
        "- **primary_thesis**: One-paragraph key finding for an investment analyst\n"
        "- **key_risks**: Top 3 risks\n"
        "- **monitoring_triggers**: What to watch next (earnings, regulatory deadlines, "
        "product launches, order announcements)"
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
        logger.error("yfinance fetch failed — cannot proceed", ticker=ticker, exc_info=results[0])
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
        "is_foreign_issuer": False,
        "events_8k": [],
    }

    if isinstance(results[1], BaseException):
        logger.warning(
            "EDGAR insider fetch failed — proceeding without insider data",
            ticker=ticker,
            exc_info=results[1],
        )
        insider_data = _empty_insider
    else:
        insider_data = results[1]

    if isinstance(results[2], BaseException):
        logger.warning(
            "EDGAR filing fetch failed — proceeding without filing prose",
            ticker=ticker,
            exc_info=results[2],
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
        insider_mspr=insider_signal.mspr,
        red_flags=len(red_flags),
    )

    # Phase 3: LLM synthesis
    agent = Agent(
        model=create_model(smart=True),
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
    is_foreign = filing_data.get("is_foreign_issuer", False)
    latest_filing = ""
    if filing_10k:
        form_label = "20-F" if is_foreign else "10-K"
        year = filing_10k.report_date.year if filing_10k.report_date else "?"
        latest_filing = f"{form_label} FY{year}, filed {filing_10k.filed_date}"

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
        growth_catalysts=llm_output.growth_catalysts,
        competitive_position=llm_output.competitive_position,
        insider_vs_financials=llm_output.insider_vs_financials,
        disclosure_consistency=llm_output.disclosure_consistency,
        primary_thesis=llm_output.primary_thesis,
        key_risks=llm_output.key_risks,
        monitoring_triggers=llm_output.monitoring_triggers,
    )
