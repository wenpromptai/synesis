"""SEC EDGAR API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from starlette.requests import Request

from synesis.core.dependencies import Crawl4AICrawlerDep, SECEdgarClientDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter()


@router.get("/filings")
@limiter.limit("60/minute")
async def get_filings(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    forms: str | None = Query(None, description="Comma-separated form types (e.g., 8-K,10-K)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Get recent SEC filings for a ticker."""
    form_types = [f.strip() for f in forms.split(",") if f.strip()] if forms else None
    filings = await client.get_filings(ticker, form_types=form_types, limit=limit)
    return {
        "ticker": ticker.upper(),
        "filings": [f.model_dump(mode="json") for f in filings],
        "count": len(filings),
    }


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
    """Get recent insider transactions from Form 4 filings.

    By default returns only open-market purchases (P) and sales (S) —
    the discretionary signals that reflect actual insider conviction.
    RSU awards (A), tax-withholding sales (F), and option exercises (M)
    are excluded by default as they are programmatic/mechanical with no
    informational content about the insider's view of the stock.

    Pass codes=all to see every transaction type.
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
    """
    transactions = await client.get_insider_transactions(ticker, limit=50, codes=["S"])
    sells = [t for t in transactions if (t.price_per_share or 0) * t.shares >= min_value]
    return {
        "ticker": ticker.upper(),
        "sells": [t.model_dump(mode="json") for t in sells[:limit]],
        "count": len(sells[:limit]),
        "codes_filter": ["S"],
    }


@router.get("/sentiment")
@limiter.limit("60/minute")
async def get_insider_sentiment(
    request: Request,
    client: SECEdgarClientDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
) -> dict[str, Any]:
    """Get computed insider sentiment from Form 4 data."""
    sentiment = await client.get_insider_sentiment(ticker)
    if sentiment is None:
        raise HTTPException(status_code=404, detail=f"No insider data for {ticker.upper()}")
    return sentiment


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
) -> dict[str, Any]:
    """Full-text search across SEC filings."""
    form_list = [f.strip() for f in forms.split(",") if f.strip()] if forms else None
    results = await client.search_filings(
        query=query,
        forms=form_list,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    return {
        "query": query,
        "results": results,
        "count": len(results),
    }


@router.get("/earnings")
@limiter.limit("10/minute")
async def get_earnings_releases(
    request: Request,
    client: SECEdgarClientDep,
    crawler: Crawl4AICrawlerDep,
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Get earnings press releases (8-K Item 2.02) with full content."""
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
    """Get the most recent earnings press release with full content."""
    releases = await client.get_earnings_releases(ticker, limit=1, crawler=crawler)
    if not releases:
        raise HTTPException(
            status_code=404,
            detail=f"No earnings releases found for {ticker.upper()}",
        )
    return releases[0].model_dump(mode="json")
