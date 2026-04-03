"""SEC EDGAR API endpoints.

Provides access to SEC filings, insider transactions, XBRL financial data,
13F institutional holdings, and ownership/governance filings.
All data sourced from the free SEC EDGAR API (no key required).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from starlette.requests import Request

from synesis.core.dependencies import Crawl4AICrawlerDep, SECEdgarClientDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Company Info
# ─────────────────────────────────────────────────────────────


@router.get("/company")
@limiter.limit("60/minute")
async def get_company_info(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
) -> dict[str, Any]:
    """Get company metadata from SEC EDGAR.

    Returns SIC industry code, filer category, fiscal year end, exchanges,
    state of incorporation, former names, and other metadata. All extracted
    from the SEC submissions response (no additional API calls).

    Examples:
        GET /company?ticker=AAPL
        GET /company?ticker=TSLA
    """
    info = await client.get_company_info(ticker)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No SEC data for {ticker.upper()}")
    result: dict[str, Any] = info.model_dump(mode="json")
    return result


# ─────────────────────────────────────────────────────────────
# Filings
# ─────────────────────────────────────────────────────────────


@router.get("/filings")
@limiter.limit("60/minute")
async def get_filings(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    forms: str | None = Query(None, description="Comma-separated form types (e.g., 8-K,10-K)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get recent SEC filings for a ticker.

    Supports all SEC form types: 10-K (annual), 10-Q (quarterly), 8-K (current events),
    4 (insider), 13F-HR (institutional), S-1 (IPO), DEF 14A (proxy), etc.

    Examples:
        GET /filings?ticker=AAPL&limit=5
        GET /filings?ticker=AAPL&forms=8-K,10-K&limit=3
        GET /filings?ticker=TSLA&forms=10-Q
    """
    form_types = [f.strip() for f in forms.split(",") if f.strip()] if forms else None
    filings = await client.get_filings(ticker, form_types=form_types, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


# ─────────────────────────────────────────────────────────────
# Insiders
# ─────────────────────────────────────────────────────────────


@router.get("/insiders")
@limiter.limit("60/minute")
async def get_insider_transactions(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(10, ge=1, le=50),
    codes: str | None = Query(
        "P,S",
        description=(
            "Comma-separated Form 4 transaction codes to include. "
            "Default 'P,S' returns only open-market purchases and sales (conviction signals). "
            "Pass 'all' to include RSU grants (A), tax withholdings (F), option exercises (M), etc."
        ),
    ),
) -> dict[str, Any]:
    """Get insider transactions from Form 3/4/5 filings.

    Parses Form 4 XML to extract who bought/sold, how many shares, at what price.
    By default returns only open-market purchases (P) and sales (S) — the discretionary
    signals reflecting actual insider conviction. Pass codes=all for everything.

    Transaction codes: P=purchase, S=sale, A=RSU grant, F=tax withholding,
    M=option exercise, G=gift, C=conversion, W=warrant exercise.

    Examples:
        GET /insiders?ticker=AAPL&codes=P,S&limit=5  (conviction signals only)
        GET /insiders?ticker=AAPL&codes=all&limit=10  (all transactions)
        GET /insiders?ticker=TSLA&codes=S&limit=20    (sells only)
    """
    code_filter: list[str] | None = None
    if codes and codes.strip().lower() != "all":
        code_filter = [c.strip().upper() for c in codes.split(",") if c.strip()]

    transactions = await client.get_insider_transactions(ticker, limit=limit, codes=code_filter)
    return {
        "ticker": ticker.upper(),
        "transactions": [t.model_dump(mode="json") for t in transactions],
        "count": len(transactions),
        "codes_filter": code_filter,
    }


@router.get("/insiders/sells")
@limiter.limit("60/minute")
async def get_insider_sells(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    min_value: float = Query(0, ge=0, description="Minimum transaction value"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get open-market insider sells above a value threshold.

    Only returns discretionary open-market sales (code S).
    Excludes tax-withholding sales (F) which are mechanical.
    Value = shares * price_per_share.

    Examples:
        GET /insiders/sells?ticker=AAPL&min_value=100000  (sells > $100K)
        GET /insiders/sells?ticker=NVDA&min_value=1000000  (sells > $1M)
    """
    transactions = await client.get_insider_transactions(ticker, limit=50, codes=["S"])
    sells = [t for t in transactions if (t.price_per_share or 0) * t.shares >= min_value]
    return {
        "ticker": ticker.upper(),
        "sells": [t.model_dump(mode="json") for t in sells[:limit]],
        "count": len(sells[:limit]),
        "codes_filter": ["S"],
    }


@router.get("/derivatives")
@limiter.limit("60/minute")
async def get_derivative_transactions(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get derivative insider transactions (options, warrants, RSUs) from Form 4.

    Returns the derivative table from Form 4 XML — includes security title
    (e.g., "Stock Option", "Restricted Stock Unit"), exercise price, expiration
    date, and underlying share count.

    Examples:
        GET /derivatives?ticker=AAPL&limit=5
        GET /derivatives?ticker=MSFT&limit=10
    """
    transactions = await client.get_derivative_transactions(ticker, limit=limit)
    return {
        "ticker": ticker.upper(),
        "transactions": [t.model_dump(mode="json") for t in transactions],
        "count": len(transactions),
    }


@router.get("/sentiment")
@limiter.limit("60/minute")
async def get_insider_sentiment(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
) -> dict[str, Any]:
    """Get computed insider sentiment from Form 4 data.

    Returns MSPR (Modified Smoothed Profit Ratio) — a ratio from -1 to +1
    where positive = net buying, negative = net selling. Also includes raw
    buy/sell counts and dollar values.

    Examples:
        GET /sentiment?ticker=AAPL
        GET /sentiment?ticker=TSLA
    """
    sentiment = await client.get_insider_sentiment(ticker)
    if sentiment is None:
        raise HTTPException(status_code=404, detail=f"No insider data for {ticker.upper()}")
    return sentiment


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────


@router.get("/search")
@limiter.limit("60/minute")
async def search_filings(
    request: Request,
    client: SECEdgarClientDep,
    query: str = Query(..., min_length=1, description="Search query"),
    forms: str | None = Query(None, description="Comma-separated form types"),
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0, le=10000, description="Pagination offset"),
) -> dict[str, Any]:
    """Full-text search across all SEC filings (EFTS).

    Searches filing content since 2001. Supports Boolean operators,
    exact phrases ("debt restructuring"), wildcards (acqui*), and exclusions (-term).
    Use offset for pagination (max 10000).

    Examples:
        GET /search?query=artificial+intelligence&forms=10-K&limit=5
        GET /search?query="supply+chain"&date_from=2024-01-01
        GET /search?query=restructuring&forms=8-K&limit=10&offset=10  (page 2)
    """
    form_list = [f.strip() for f in forms.split(",") if f.strip()] if forms else None
    search_result = await client.search_filings(
        query=query,
        forms=form_list,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return {
        "query": query,
        "results": search_result["results"],
        "count": len(search_result["results"]),
        "total_hits": search_result["total_hits"],
        "offset": search_result["offset"],
    }


# ─────────────────────────────────────────────────────────────
# Earnings
# ─────────────────────────────────────────────────────────────


@router.get("/earnings")
@limiter.limit("10/minute")
async def get_earnings_releases(
    request: Request,
    client: SECEdgarClientDep,
    crawler: Crawl4AICrawlerDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Get earnings press releases (8-K Item 2.02) with full content.

    Fetches 8-K filings filtered to Item 2.02 (Results of Operations),
    then extracts the press release content via Crawl4AI for each filing.

    Examples:
        GET /earnings?ticker=AAPL&limit=2
        GET /earnings?ticker=MSFT&limit=5
    """
    releases = await client.get_earnings_releases(ticker, limit=limit, crawler=crawler)
    return {
        "ticker": ticker.upper(),
        "releases": [r.model_dump(mode="json") for r in releases],
        "count": len(releases),
    }


@router.get("/earnings/latest")
@limiter.limit("10/minute")
async def get_latest_earnings_release(
    request: Request,
    client: SECEdgarClientDep,
    crawler: Crawl4AICrawlerDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
) -> dict[str, Any]:
    """Get the most recent earnings press release with full content.

    Shorthand for /earnings?limit=1 — returns the single most recent
    8-K Item 2.02 filing with its full text content.

    Examples:
        GET /earnings/latest?ticker=AAPL
        GET /earnings/latest?ticker=GOOG
    """
    releases = await client.get_earnings_releases(ticker, limit=1, crawler=crawler)
    if not releases:
        raise HTTPException(
            status_code=404,
            detail=f"No earnings releases found for {ticker.upper()}",
        )
    return releases[0].model_dump(mode="json")


# ─────────────────────────────────────────────────────────────
# 8-K Events
# ─────────────────────────────────────────────────────────────


@router.get("/events")
@limiter.limit("60/minute")
async def get_8k_events(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    items: str | None = Query(
        None,
        description=(
            "Comma-separated 8-K item codes to filter (e.g., '1.01,2.01,5.02'). "
            "Omit to return all 8-K filings."
        ),
    ),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get 8-K filings with item filtering and human-readable descriptions.

    Each 8-K filing reports one or more material events. This endpoint parses
    the item codes and provides human-readable descriptions for each.

    Key item codes:
        1.01 = Material Agreement    2.01 = Acquisition/Disposition
        2.02 = Earnings Release      2.05 = Layoffs/Restructuring
        5.02 = Officer Changes       7.01 = Regulation FD Disclosure

    Examples:
        GET /events?ticker=AAPL&limit=5                          (all 8-K events)
        GET /events?ticker=AAPL&items=1.01,2.01,5.02&limit=10   (M&A + officer changes)
        GET /events?ticker=TSLA&items=2.02                       (earnings only)
    """
    item_filter = [i.strip() for i in items.split(",") if i.strip()] if items else None
    events = await client.get_8k_events(ticker, items=item_filter, limit=limit)
    return {
        "ticker": ticker.upper(),
        "events": [e.model_dump(mode="json") for e in events],
        "count": len(events),
        "items_filter": item_filter,
    }


# ─────────────────────────────────────────────────────────────
# XBRL / Financial Data
# ─────────────────────────────────────────────────────────────


@router.get("/facts")
@limiter.limit("30/minute")
async def get_company_facts(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    concepts: str | None = Query(
        None,
        description="Comma-separated XBRL concept names to filter (e.g., 'NetIncomeLoss,Revenues')",
    ),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Get all XBRL financial facts for a company in one call.

    Returns every XBRL concept the company has ever reported — income statement,
    balance sheet, cash flow, per-share data, and more. Returns deduplicated
    "best fit" values (frame-bearing entries only).

    Use the `concepts` parameter to filter to specific XBRL tags.
    Concept aliases are expanded automatically (e.g., "Revenues" also includes
    "RevenueFromContractWithCustomerExcludingAssessedTax" and vice versa).

    Examples:
        GET /facts?ticker=AAPL                                        (everything)
        GET /facts?ticker=AAPL&concepts=NetIncomeLoss,Assets&limit=5  (filtered)
        GET /facts?ticker=MSFT&concepts=Revenues                      (revenue history)
    """
    concept_list = [c.strip() for c in concepts.split(",") if c.strip()] if concepts else None
    facts = await client.get_company_facts(ticker, concepts=concept_list, limit=limit)
    if facts is None:
        raise HTTPException(status_code=404, detail=f"No XBRL data for {ticker.upper()}")
    return facts.model_dump(mode="json")


@router.get("/frames/{taxonomy}/{tag}/{unit}/{period}")
@limiter.limit("30/minute")
async def get_xbrl_frame(
    request: Request,
    client: SECEdgarClientDep,
    taxonomy: str,
    tag: str,
    unit: str,
    period: str,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Get cross-company XBRL data for a single metric and period.

    Returns one value per company for the specified concept, unit, and period.
    Sorted by absolute value descending — useful for screening and sector comparisons.

    Period format: CY2024 (annual), CY2024Q3 (quarterly), CY2024Q3I (instantaneous/balance sheet)
    Taxonomy: us-gaap, dei, ifrs-full, srt

    Examples:
        GET /frames/us-gaap/Revenues/USD/CY2024Q3?limit=10      (top revenue companies)
        GET /frames/us-gaap/NetIncomeLoss/USD/CY2024Q3?limit=20  (top earners)
        GET /frames/us-gaap/Assets/USD/CY2024Q3I?limit=10        (largest companies by assets)
    """
    frame = await client.get_xbrl_frame(taxonomy, tag, unit, period, limit=limit)
    if frame is None:
        raise HTTPException(
            status_code=404,
            detail=f"No XBRL frame for {taxonomy}/{tag}/{unit}/{period}",
        )
    return frame.model_dump(mode="json")


# ─────────────────────────────────────────────────────────────
# Ownership / Governance
# ─────────────────────────────────────────────────────────────


@router.get("/activists")
@limiter.limit("60/minute")
async def get_activist_filings(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Get Schedule 13D/13G filings (activist/passive investor >5% ownership).

    Required when an investor acquires >5% beneficial ownership.
    SC 13D = activist intent (plans to influence), SC 13G = passive investment.
    Amendments (/A) indicate ownership changes.

    Examples:
        GET /activists?ticker=AAPL&limit=5
        GET /activists?ticker=DIS&limit=10  (Disney — frequent activist targets)
    """
    filings = await client.get_activist_filings(ticker, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


@router.get("/form144")
@limiter.limit("60/minute")
async def get_form144_filings(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get Form 144 pre-sale notices (restricted stock sale intent).

    Insiders must file Form 144 before selling restricted or control securities.
    These filings provide an early warning signal of upcoming insider sales —
    the actual sale may follow within 90 days.

    Examples:
        GET /form144?ticker=AAPL&limit=5
        GET /form144?ticker=TSLA&limit=10
    """
    filings = await client.get_form144_filings(ticker, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


@router.get("/late-filings")
@limiter.limit("60/minute")
async def get_late_filing_alerts(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
) -> dict[str, Any]:
    """Get late filing notifications (NT 10-K / NT 10-Q).

    Companies file NT (Notification of Late Filing) when they cannot meet
    the filing deadline for their 10-K or 10-Q. Often a red flag indicating
    accounting issues, restatements, internal control problems, or auditor disputes.

    Examples:
        GET /late-filings?ticker=AAPL  (AAPL had 2 back in 2006 — options backdating)
        GET /late-filings?ticker=SMCI  (Super Micro — recent late filing issues)
    """
    alerts = await client.get_late_filing_alerts(ticker)
    return {
        "ticker": ticker.upper(),
        "alerts": [a.model_dump(mode="json") for a in alerts],
        "count": len(alerts),
    }


@router.get("/ipos")
@limiter.limit("30/minute")
async def get_ipo_filings(
    request: Request,
    client: SECEdgarClientDep,
    query: str | None = Query(None, description="Search query (company name or keyword)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get S-1 IPO registration filings via full-text search.

    Pre-IPO companies don't have tickers, so this uses SEC's EFTS search engine
    to find S-1 and S-1/A registration statements. S-1/A amendments often indicate
    the IPO is progressing (pricing updates, underwriter changes).

    Examples:
        GET /ipos?query=technology&limit=5  (tech IPOs)
        GET /ipos?query=biotech&limit=10    (biotech IPOs)
        GET /ipos?limit=10                  (recent S-1 filings)
    """
    filings = await client.get_ipo_filings(query=query, limit=limit)
    return {
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
        "query": query,
    }


@router.get("/proxy")
@limiter.limit("30/minute")
async def get_proxy_filings(
    request: Request,
    client: SECEdgarClientDep,
    crawler: Crawl4AICrawlerDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Get DEF 14A proxy statements with optional full content.

    Proxy statements contain executive compensation details, shareholder proposals,
    board composition, and governance information. Filed annually before the AGM.
    Content extracted via Crawl4AI (HTML → clean markdown).

    Examples:
        GET /proxy?ticker=AAPL&limit=2
        GET /proxy?ticker=TSLA&limit=3
    """
    filings = await client.get_proxy_filings(ticker, limit=limit, crawler=crawler)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


# ─────────────────────────────────────────────────────────────
# 13F Holdings
# ─────────────────────────────────────────────────────────────


@router.get("/13f")
@limiter.limit("30/minute")
async def get_13f_holdings(
    request: Request,
    client: SECEdgarClientDep,
    cik: str = Query(..., description="CIK of the fund (e.g., 1067983 for Berkshire)"),
    fund_name: str | None = Query(None, description="Fund name for labeling"),
) -> dict[str, Any]:
    """Get 13F-HR quarterly holdings for a fund by CIK.

    Institutional managers with >$100M AUM must file 13F-HR quarterly,
    disclosing all equity holdings. Returns parsed holdings sorted by value.
    Use CIK (not ticker) since funds may not have tickers.

    Well-known CIKs: Berkshire=1067983, Bridgewater=1350694, Renaissance=1037389

    Examples:
        GET /13f?cik=1067983&fund_name=Berkshire+Hathaway
        GET /13f?cik=1350694&fund_name=Bridgewater
    """
    filings = await client.get_13f_filings(cik, limit=1)
    if not filings:
        raise HTTPException(status_code=404, detail=f"No 13F filings found for CIK {cik}")

    holdings = await client.get_13f_holdings(filings[0], cik)
    if holdings is None:
        raise HTTPException(status_code=404, detail=f"Could not parse 13F holdings for CIK {cik}")

    if fund_name:
        holdings.fund_name = fund_name

    return holdings.model_dump(mode="json")


@router.get("/13f/compare")
@limiter.limit("10/minute")
async def compare_13f_quarters(
    request: Request,
    client: SECEdgarClientDep,
    cik: str = Query(..., description="CIK of the fund"),
    fund_name: str = Query(..., description="Fund name for labeling"),
) -> dict[str, Any]:
    """Compare two most recent 13F filings to find position changes.

    Diffs the two most recent 13F-HR filings to identify new positions,
    exited positions, and significant increases/decreases (>10% change).
    Useful for tracking what top funds are buying and selling.

    Examples:
        GET /13f/compare?cik=1067983&fund_name=Berkshire+Hathaway
        GET /13f/compare?cik=1037389&fund_name=Renaissance+Technologies
    """
    result = await client.compare_13f_quarters(cik, fund_name)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough 13F data for CIK {cik} to compare quarters",
        )
    return result


# ─────────────────────────────────────────────────────────────
# Tender Offers / M&A
# ─────────────────────────────────────────────────────────────


@router.get("/tender-offers")
@limiter.limit("60/minute")
async def get_tender_offers(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Get tender offer filings (SC TO-T, SC TO-I, SC 14D-9).

    SC TO-T = third-party tender offer (acquisition attempt),
    SC TO-I = issuer tender offer (buyback),
    SC 14D-9 = target's recommendation on a tender offer.

    Examples:
        GET /tender-offers?ticker=ATVI&limit=5  (Activision during MSFT acquisition)
        GET /tender-offers?ticker=VMW&limit=5
    """
    filings = await client.get_tender_offers(ticker, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


# ─────────────────────────────────────────────────────────────
# Foreign Issuers
# ─────────────────────────────────────────────────────────────


@router.get("/foreign")
@limiter.limit("60/minute")
async def get_foreign_filings(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get foreign issuer filings (20-F, 6-K, 40-F).

    20-F = annual report (equivalent to 10-K for US companies),
    6-K = current report (equivalent to 8-K),
    40-F = Canadian issuer annual report.
    Use for ADRs like TSM, BABA, ASML, NVO, etc.

    Examples:
        GET /foreign?ticker=TSM&limit=5
        GET /foreign?ticker=BABA&limit=10
    """
    filings = await client.get_foreign_filings(ticker, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


# ─────────────────────────────────────────────────────────────
# Effectiveness Notices (IPO ready to price)
# ─────────────────────────────────────────────────────────────


@router.get("/effectiveness")
@limiter.limit("30/minute")
async def get_effectiveness_notices(
    request: Request,
    client: SECEdgarClientDep,
    query: str | None = Query(None, description="Search query (company name or keyword)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get EFFECT notices — S-1 registrations declared effective.

    When SEC declares an S-1 effective, the company can begin selling shares.
    This is the final step before an IPO prices. Pairs with /ipos (S-1 filings)
    to track the full IPO pipeline.

    Examples:
        GET /effectiveness?limit=10          (recent effectiveness notices)
        GET /effectiveness?query=biotech     (biotech IPOs going effective)
    """
    notices = await client.get_effectiveness_notices(query=query, limit=limit)
    return {
        "filings": [n.model_dump(mode="json") for n in notices],
        "count": len(notices),
        "query": query,
    }


# ─────────────────────────────────────────────────────────────
# Filing Feed (real-time)
# ─────────────────────────────────────────────────────────────


@router.get("/feed")
@limiter.limit("30/minute")
async def get_filing_feed(
    request: Request,
    client: SECEdgarClientDep,
    form_type: str | None = Query(None, description="Filter by form type (e.g., 8-K, 4, SC 13D)"),
    cik: str | None = Query(None, description="Filter by CIK number"),
    count: int = Query(40, ge=1, le=100),
) -> dict[str, Any]:
    """Get latest SEC filings from the EDGAR Atom feed (near-real-time).

    Updated as filings are submitted. Useful for monitoring new filings
    without polling the submissions API. Filterable by form type and CIK.

    Examples:
        GET /feed?form_type=8-K&count=20          (latest 8-K filings)
        GET /feed?form_type=4&count=10             (latest insider filings)
        GET /feed?cik=0000320193&count=10          (latest AAPL filings)
        GET /feed?count=40                         (all recent filings)
    """
    entries = await client.get_filing_feed(form_type=form_type, cik=cik, count=count)
    return {
        "entries": [e.model_dump(mode="json") for e in entries],
        "count": len(entries),
        "form_type": form_type,
        "cik": cik,
    }
