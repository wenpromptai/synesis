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
    """Company metadata from SEC EDGAR.

    All metadata extracted from the SEC submissions response (single upstream call).

    **Query params:**
    - `ticker` (str, required): symbol, case-insensitive.

    **Returns:** an object with `ticker`, `cik`, `name`, `entity_type`,
    `sic`, `sic_description`, `category` (e.g. "Large accelerated filer"),
    `fiscal_year_end`, `state_of_incorporation`, `exchanges` (list),
    `tickers` (list), `ein`, `former_names` (list), `phone`, `website`.

    **Errors:**
    - `404` if no SEC data exists for that ticker.

    **Examples:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/company?ticker=AAPL'
    curl 'http://localhost:7337/api/v1/sec_edgar/company?ticker=TSLA'
    ```
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
    """Recent SEC filings for a ticker.

    Supports every SEC form type — 10-K (annual), 10-Q (quarterly),
    8-K (current events), 4 (insider), 13F-HR (institutional), S-1 (IPO),
    DEF 14A (proxy), etc.

    **Query params:**
    - `ticker` (str, required).
    - `forms` (str, optional): comma-separated form types (e.g. `8-K,10-K`).
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, form, filed_date, accepted_datetime,
      accession_number, primary_document, items, url, report_date}`.

    **Examples:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/filings?ticker=AAPL&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/filings?ticker=AAPL&forms=8-K,10-K&limit=3'
    ```
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
    """Insider transactions from Form 3/4/5 filings.

    Parses Form 4 XML to extract who bought/sold, how many shares, at what price.
    Defaults to open-market activity only (`P,S`) since those are the discretionary
    conviction signals; pass `codes=all` to include grants, tax withholdings, etc.

    Transaction codes: `P`=purchase, `S`=sale, `A`=RSU grant, `F`=tax withholding,
    `M`=option exercise, `G`=gift, `C`=conversion, `W`=warrant exercise.

    **Query params:**
    - `ticker` (str, required).
    - `limit` (int, 1–50, default 10).
    - `codes` (str, default `P,S`): comma-separated codes, or the literal `all`.

    **Returns:**
    - `ticker` (str), `count` (int), `codes_filter` (list[str] | null),
    - `transactions` (list): each `{ticker, owner_name, owner_relationship,
      owner_country, transaction_date, transaction_code, shares, price_per_share,
      shares_after, acquired_or_disposed, filing_date, filing_url,
      transaction_type_label, is_open_market}`.

    **Examples:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/insiders?ticker=AAPL&codes=P,S&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/insiders?ticker=TSLA&codes=S&limit=20'
    ```
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
    """Open-market insider sells above a dollar threshold.

    Filters Form 4 data to discretionary open-market sales (code `S`) only.
    Tax-withholding sells (code `F`) are excluded as they are mechanical.
    Transaction value = `shares * price_per_share`.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `min_value` (float, ≥0, default 0): minimum dollar value to include.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int), `codes_filter` (`["S"]`),
    - `sells` (list): each `{ticker, owner_name, owner_relationship,
      owner_country, transaction_date, transaction_code, shares, price_per_share,
      shares_after, acquired_or_disposed, filing_date, filing_url,
      transaction_type_label, is_open_market}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/insiders/sells?ticker=AAPL&min_value=100000&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/insiders/sells?ticker=NVDA&min_value=1000000'
    ```
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
    """Derivative insider transactions (options, warrants, RSUs) from Form 4.

    Parses the derivative table of Form 4 XML filings — includes security title
    (e.g., "Stock Option", "RSU"), exercise price, expiration date, and
    underlying share count.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `transactions` (list): each `{ticker, owner_name, owner_relationship,
      transaction_date, transaction_code, security_title, exercise_price,
      underlying_shares, expiration_date, acquired_or_disposed,
      filing_date, filing_url, transaction_type_label, is_open_market}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/derivatives?ticker=AAPL&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/derivatives?ticker=MSFT&limit=10'
    ```
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
    """Computed insider sentiment from Form 4 data.

    Aggregates all Form 4 transactions to produce a sentiment score.
    MSPR (Modified Smoothed Profit Ratio) ranges from -1 to +1,
    where positive = net buying, negative = net selling.

    **Query params:**
    - `ticker` (str, required): symbol.

    **Returns:** object with `ticker`, `mspr` (float, -1 to +1),
    `change` (int, net buy minus sell count), `buy_count`, `sell_count`,
    `total_buy_value`, `total_sell_value`.

    **Errors:**
    - `404` if no Form 4 data found for ticker.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/sentiment?ticker=AAPL'
    curl 'http://localhost:7337/api/v1/sec_edgar/sentiment?ticker=TSLA'
    ```
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
    exact phrases (`"debt restructuring"`), wildcards (`acqui*`), and
    exclusions (`-term`). Use `offset` for pagination (max 10,000).

    **Query params:**
    - `query` (str, required): search text.
    - `forms` (str, optional): comma-separated form types (e.g. `10-K,8-K`).
    - `date_from` (YYYY-MM-DD, optional): inclusive start date.
    - `date_to` (YYYY-MM-DD, optional): inclusive end date.
    - `limit` (int, 1–50, default 10).
    - `offset` (int, 0–10000, default 0): pagination offset.

    **Returns:**
    - `query` (str): echoed.
    - `results` (list): each result has filing metadata from EFTS.
    - `count` (int): items in this page.
    - `total_hits` (int): total matching filings.
    - `offset` (int): echoed.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/search?query=artificial+intelligence&forms=10-K&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/search?query=restructuring&forms=8-K&limit=10&offset=10'
    ```
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
    """Earnings press releases (8-K Item 2.02) with full text content.

    Fetches 8-K filings filtered to Item 2.02 (Results of Operations and
    Financial Condition), then extracts the press release markdown via
    Crawl4AI for each. Slower than other endpoints due to page crawling.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–20, default 5).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `releases` (list): each `{ticker, filed_date, accepted_datetime,
      accession_number, url, items, content}` — `content` is the
      press release as markdown (null if crawl failed).

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/earnings?ticker=AAPL&limit=2'
    curl 'http://localhost:7337/api/v1/sec_edgar/earnings?ticker=MSFT&limit=5'
    ```
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
    """Most recent earnings press release (8-K Item 2.02) with full text.

    Shorthand for `/earnings?limit=1`. Returns the single most recent
    filing with its crawled markdown content.

    **Query params:**
    - `ticker` (str, required): symbol.

    **Returns:** one `EarningsRelease` object — `{ticker, filed_date,
    accepted_datetime, accession_number, url, items, content}`.

    **Errors:**
    - `404` if no earnings releases found.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/earnings/latest?ticker=AAPL'
    curl 'http://localhost:7337/api/v1/sec_edgar/earnings/latest?ticker=GOOG'
    ```
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
    """8-K material event filings with parsed item codes and descriptions.

    Each 8-K reports one or more material events. This endpoint parses the
    item codes and maps them to human-readable descriptions.

    Key item codes: `1.01`=Material Agreement, `2.01`=Acquisition/Disposition,
    `2.02`=Earnings Release, `2.05`=Layoffs/Restructuring, `5.02`=Officer Changes,
    `7.01`=Regulation FD Disclosure.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `items` (str, optional): comma-separated item codes to filter.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int), `items_filter` (list[str] | null),
    - `events` (list): each `{ticker, filed_date, accepted_datetime,
      accession_number, url, items (list[str]), item_descriptions (list[str]),
      content}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/events?ticker=AAPL&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/events?ticker=AAPL&items=1.01,2.01,5.02&limit=10'
    ```
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
    """All XBRL financial facts for a company.

    Returns every XBRL concept the company has filed — income statement,
    balance sheet, cash flow, per-share data. Deduplicated to frame-bearing
    entries (best-fit values per period). Concept aliases are expanded
    automatically (e.g., `Revenues` also matches
    `RevenueFromContractWithCustomerExcludingAssessedTax`).

    **Query params:**
    - `ticker` (str, required): symbol.
    - `concepts` (str, optional): comma-separated XBRL tag names to filter.
    - `limit` (int, 1–100, default 20): max facts returned per concept.

    **Returns:** `CompanyFacts` — `{ticker, cik, entity_name, concept_count,
    facts (list)}`. Each fact: `{concept, label, unit, period_end, value,
    form, frame, filed}`.

    **Errors:**
    - `404` if no XBRL data for ticker.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/facts?ticker=AAPL'
    curl 'http://localhost:7337/api/v1/sec_edgar/facts?ticker=AAPL&concepts=NetIncomeLoss,Assets&limit=5'
    ```
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
    """Cross-company XBRL data for a single metric and reporting period.

    Returns one value per company, sorted by absolute value descending —
    useful for sector screening and relative comparisons.

    **Path params:**
    - `taxonomy`: `us-gaap` | `dei` | `ifrs-full` | `srt`.
    - `tag`: XBRL concept name, e.g. `Revenues`, `NetIncomeLoss`, `Assets`.
    - `unit`: unit of measure, e.g. `USD`, `USD/shares`, `pure`.
    - `period`: `CY2024` (annual) | `CY2024Q3` (quarterly) |
      `CY2024Q3I` (instantaneous, for balance sheet items).

    **Query params:**
    - `limit` (int, 1–500, default 100).

    **Returns:** `XBRLFrame` — `{taxonomy, tag, unit, period, entry_count,
    entries (list)}`. Each entry: `{cik, entity_name, value, accession_number, end}`.

    **Errors:**
    - `404` if no data for that combination.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/frames/us-gaap/Revenues/USD/CY2024Q3?limit=10'
    curl 'http://localhost:7337/api/v1/sec_edgar/frames/us-gaap/Assets/USD/CY2024Q3I?limit=10'
    ```
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
    """Schedule 13D/13G filings — investors with >5% beneficial ownership.

    Filed when any investor crosses 5% ownership. SC 13D = activist intent
    (plans to influence management), SC 13G = passive investment. `/A`
    amendments indicate ownership changes.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–20, default 5).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, form_type, filed_date,
      accession_number, url, is_amendment}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/activists?ticker=AAPL&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/activists?ticker=DIS&limit=10'
    ```
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
    """Form 144 pre-sale notices for restricted stock.

    Insiders must file Form 144 before selling restricted or control securities.
    An early warning signal — the actual sale may follow within 90 days.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, filed_date, accession_number, url}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/form144?ticker=AAPL&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/form144?ticker=TSLA&limit=10'
    ```
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
    """Late filing notifications (NT 10-K / NT 10-Q).

    Companies file NT forms when they cannot meet the 10-K or 10-Q deadline.
    A red flag for accounting issues, restatements, internal control problems,
    or auditor disputes.

    **Query params:**
    - `ticker` (str, required): symbol.

    **Returns:**
    - `ticker` (str), `count` (int),
    - `alerts` (list): each `{ticker, form_type, filed_date,
      accession_number, url, original_form}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/late-filings?ticker=SMCI'
    curl 'http://localhost:7337/api/v1/sec_edgar/late-filings?ticker=AAPL'
    ```
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
    """S-1 IPO registration filings via full-text search.

    Pre-IPO companies don't have tickers, so this searches SEC's EFTS for
    S-1 and S-1/A registration statements. S-1/A amendments indicate
    the IPO is progressing (pricing updates, underwriter changes).

    **Query params:**
    - `query` (str, optional): company name or keyword to filter results.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `filings` (list), `count` (int), `query` (str | null).
      Each filing: `{entity_name, form_type, filed_date, accession_number,
      url, is_amendment}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/ipos?query=technology&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/ipos?limit=10'
    ```
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
    """DEF 14A proxy statements with full text content.

    Proxy statements cover executive compensation, shareholder proposals,
    board composition, and governance. Filed annually before the AGM.
    Full text is extracted via Crawl4AI (HTML → markdown).

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–20, default 5).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, filed_date, accession_number,
      url, content}` — `content` is markdown (null if crawl failed).

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/proxy?ticker=AAPL&limit=2'
    curl 'http://localhost:7337/api/v1/sec_edgar/proxy?ticker=TSLA&limit=3'
    ```
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
    """13F-HR quarterly equity holdings for a fund by CIK.

    Institutional managers with >$100M AUM must file 13F-HR quarterly,
    disclosing all equity holdings. Returns the most recent filing's parsed
    holdings sorted by value. Use CIK (not ticker) since funds lack tickers.

    Well-known CIKs: Berkshire=1067983, Bridgewater=1350694, Renaissance=1037389.

    **Query params:**
    - `cik` (str, required): numeric CIK of the fund.
    - `fund_name` (str, optional): label to apply — overrides the name from the filing.

    **Returns:** `Filing13F` — `{cik, fund_name, filed_date, report_date,
    accession_number, url, total_value_thousands, holdings (list)}`.
    Each holding: `{name_of_issuer, title_of_class, cusip, value_thousands,
    shares, investment_discretion}`.

    **Errors:**
    - `404` if no 13F filings found for CIK.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/13f?cik=1067983&fund_name=Berkshire+Hathaway'
    curl 'http://localhost:7337/api/v1/sec_edgar/13f?cik=1350694&fund_name=Bridgewater'
    ```
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
    """Diff two most recent 13F filings to find position changes.

    Compares the two most recent 13F-HR filings to identify new positions,
    exited positions, and significant size changes (>10% move in share count).

    **Query params:**
    - `cik` (str, required): fund CIK.
    - `fund_name` (str, required): label for display.

    **Returns:** object with `fund_name`, `cik`, `current_report_date`,
    `previous_report_date`, `total_value_current`, `total_value_previous`,
    `new_positions` (list), `exited_positions` (list), `increased` (list),
    `decreased` (list). Each item is a holding dict with an added `change_pct`
    and `prev_shares` on changed positions.

    **Errors:**
    - `404` if fewer than two 13F filings exist for the CIK.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/13f/compare?cik=1067983&fund_name=Berkshire+Hathaway'
    curl 'http://localhost:7337/api/v1/sec_edgar/13f/compare?cik=1037389&fund_name=Renaissance+Technologies'
    ```
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
    """Tender offer filings (SC TO-T, SC TO-I, SC 14D-9).

    SC TO-T = third-party tender offer (acquisition attempt),
    SC TO-I = issuer tender offer (share buyback via tender),
    SC 14D-9 = target board's recommendation on a tender offer.

    **Query params:**
    - `ticker` (str, required): symbol.
    - `limit` (int, 1–20, default 5).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, form_type, filed_date,
      accession_number, url, is_amendment}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/tender-offers?ticker=ATVI&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/tender-offers?ticker=VMW&limit=5'
    ```
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
    """Foreign issuer filings (20-F, 6-K, 40-F).

    For ADRs and foreign-listed US securities. 20-F = annual report
    (equivalent to 10-K), 6-K = current report (equivalent to 8-K),
    40-F = Canadian issuer annual report.

    **Query params:**
    - `ticker` (str, required): symbol (e.g. `TSM`, `BABA`, `ASML`).
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `ticker` (str), `count` (int),
    - `filings` (list): each `{ticker, form_type, filed_date,
      accession_number, url, report_date}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/foreign?ticker=TSM&limit=5'
    curl 'http://localhost:7337/api/v1/sec_edgar/foreign?ticker=BABA&limit=10'
    ```
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
    """EFFECT notices — S-1 registrations declared effective (IPO ready to price).

    When the SEC declares an S-1 effective, the company can begin selling
    shares. This is the final step before an IPO prices. Pairs with `/ipos`
    to track the full IPO pipeline.

    **Query params:**
    - `query` (str, optional): company name or keyword to filter.
    - `limit` (int, 1–50, default 10).

    **Returns:**
    - `filings` (list), `count` (int), `query` (str | null).
      Each notice: `{entity_name, form_type, filed_date, accession_number, url}`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/effectiveness?limit=10'
    curl 'http://localhost:7337/api/v1/sec_edgar/effectiveness?query=biotech'
    ```
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
    """Latest SEC filings from the EDGAR Atom feed (near-real-time).

    Updated as filings are submitted. Useful for monitoring new filings
    without polling the submissions API. Filterable by form type and CIK.

    **Query params:**
    - `form_type` (str, optional): e.g. `8-K`, `4`, `SC 13D`.
    - `cik` (str, optional): numeric CIK to filter to one company.
    - `count` (int, 1–100, default 40).

    **Returns:**
    - `entries` (list), `count` (int), `form_type` (str | null), `cik` (str | null).
      Each entry: `{title, link, summary, updated, category}` — `category`
      is the form type (e.g. `8-K`).

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/sec_edgar/feed?form_type=8-K&count=20'
    curl 'http://localhost:7337/api/v1/sec_edgar/feed?form_type=4&count=10'
    curl 'http://localhost:7337/api/v1/sec_edgar/feed?cik=0000320193&count=10'
    ```
    """
    entries = await client.get_filing_feed(form_type=form_type, cik=cik, count=count)
    return {
        "entries": [e.model_dump(mode="json") for e in entries],
        "count": len(entries),
        "form_type": form_type,
        "cik": cik,
    }
