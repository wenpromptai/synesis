"""FactSet data endpoints."""

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from synesis.core.dependencies import FactSetProviderDep
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)

router = APIRouter()


@router.get("/securities/search", response_model=list[FactSetSecurity])
async def search_securities(
    provider: FactSetProviderDep,
    query: str = Query(..., description="Search query (company name or ticker)"),
    limit: int = Query(default=20, le=100),
) -> list[FactSetSecurity]:
    return await provider.search_securities(query, limit)


@router.get("/securities/prices/batch", response_model=dict[str, FactSetPrice])
async def get_latest_prices(
    provider: FactSetProviderDep,
    tickers: str = Query(..., description="Comma-separated tickers (e.g. AAPL,MSFT,NVDA)"),
) -> dict[str, FactSetPrice]:
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    return await provider.get_latest_prices(ticker_list)


@router.get("/securities/{ticker}", response_model=FactSetSecurity)
async def get_security(ticker: str, provider: FactSetProviderDep) -> FactSetSecurity:
    security = await provider.resolve_ticker(ticker)
    if not security:
        raise HTTPException(404, detail=f"Ticker '{ticker}' not found")
    return security


@router.get("/securities/{ticker}/price", response_model=FactSetPrice)
async def get_price(
    ticker: str,
    provider: FactSetProviderDep,
    price_date: date | None = Query(
        default=None, alias="date", description="Price date (YYYY-MM-DD)"
    ),
) -> FactSetPrice:
    price = await provider.get_price(ticker, price_date=price_date)
    if not price:
        raise HTTPException(404, detail=f"No price data for '{ticker}'")
    return price


@router.get("/securities/{ticker}/price-history", response_model=list[FactSetPrice])
async def get_price_history(
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date | None = Query(default=None, description="End date (YYYY-MM-DD)"),
) -> list[FactSetPrice]:
    return await provider.get_price_history(ticker, start, end)


@router.get("/securities/{ticker}/price-history/adjusted", response_model=list[FactSetPrice])
async def get_adjusted_price_history(
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date | None = Query(default=None, description="End date (YYYY-MM-DD)"),
) -> list[FactSetPrice]:
    return await provider.get_adjusted_price_history(ticker, start, end)


@router.get("/securities/{ticker}/fundamentals", response_model=list[FactSetFundamentals])
async def get_fundamentals(
    ticker: str,
    provider: FactSetProviderDep,
    period: str = Query(default="annual", pattern="^(annual|quarterly|ltm)$"),
    limit: int = Query(default=4, le=20),
) -> list[FactSetFundamentals]:
    return await provider.get_fundamentals(ticker, period_type=period, limit=limit)


@router.get("/securities/{ticker}/profile")
async def get_company_profile(ticker: str, provider: FactSetProviderDep) -> dict[str, str | None]:
    profile = await provider.get_company_profile(ticker)
    if profile is None:
        raise HTTPException(404, detail=f"No profile for '{ticker}'")
    return {"ticker": ticker, "description": profile}


@router.get("/securities/{ticker}/corporate-actions", response_model=list[FactSetCorporateAction])
async def get_corporate_actions(
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=50, le=200),
) -> list[FactSetCorporateAction]:
    return await provider.get_corporate_actions(ticker, limit=limit)


@router.get("/securities/{ticker}/dividends", response_model=list[FactSetCorporateAction])
async def get_dividends(
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=20, le=100),
) -> list[FactSetCorporateAction]:
    return await provider.get_dividends(ticker, limit=limit)


@router.get("/securities/{ticker}/splits", response_model=list[FactSetCorporateAction])
async def get_splits(
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=20, le=100),
) -> list[FactSetCorporateAction]:
    return await provider.get_splits(ticker, limit=limit)


@router.get("/securities/{ticker}/shares-outstanding", response_model=FactSetSharesOutstanding)
async def get_shares_outstanding(
    ticker: str, provider: FactSetProviderDep
) -> FactSetSharesOutstanding:
    shares = await provider.get_shares_outstanding(ticker)
    if not shares:
        raise HTTPException(404, detail=f"No shares data for '{ticker}'")
    return shares


@router.get("/securities/{ticker}/market-cap")
async def get_market_cap(ticker: str, provider: FactSetProviderDep) -> dict[str, object]:
    mcap = await provider.get_market_cap(ticker)
    if mcap is None:
        raise HTTPException(404, detail=f"Cannot compute market cap for '{ticker}'")
    return {"ticker": ticker, "market_cap": mcap}


@router.get("/securities/{ticker}/adjustment-factors")
async def get_adjustment_factors(
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date = Query(..., description="End date (YYYY-MM-DD)"),
) -> dict[str, float]:
    factors = await provider.get_adjustment_factors(ticker, start, end)
    return {d.isoformat(): f for d, f in factors.items()}
