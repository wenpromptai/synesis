"""FactSet data endpoints.

Provides access to security information, pricing data, fundamentals, and corporate actions
from FactSet's financial database.
"""

from datetime import date

from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from synesis.core.dependencies import FactSetProviderDep
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/securities/search", response_model=list[FactSetSecurity])
@limiter.limit("60/minute")
async def search_securities(
    request: Request,
    provider: FactSetProviderDep,
    query: str = Query(..., description="Search query (company name or ticker)"),
    limit: int = Query(default=20, le=100, description="Maximum number of results to return"),
) -> list[FactSetSecurity]:
    """Search for securities by company name or ticker symbol.

    Args:
        query: Company name or ticker to search for (e.g., "Apple" or "AAPL")
        limit: Maximum number of results (default 20, max 100)

    Returns:
        List of matching securities with ticker, name, and metadata.

    Example:
        GET /api/v1/factset/securities/search?query=NVDA&limit=5
    """
    return await provider.search_securities(query, limit)


@router.get("/securities/prices/batch", response_model=dict[str, FactSetPrice])
@limiter.limit("10/minute")
async def get_latest_prices(
    request: Request,
    provider: FactSetProviderDep,
    tickers: str = Query(..., description="Comma-separated tickers (e.g. AAPL,MSFT,NVDA)"),
) -> dict[str, FactSetPrice]:
    """Get latest prices for multiple securities in a single request.

    Args:
        tickers: Comma-separated list of ticker symbols (max 100)

    Returns:
        Dictionary mapping ticker symbols to their latest price data.

    Raises:
        HTTPException: 400 if more than 100 tickers requested.

    Example:
        GET /api/v1/factset/securities/prices/batch?tickers=AAPL,MSFT,NVDA
    """
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) > 100:
        raise HTTPException(400, detail="Too many tickers requested (max 100)")
    return await provider.get_latest_prices(ticker_list)


@router.get("/securities/{ticker}", response_model=FactSetSecurity)
@limiter.limit("60/minute")
async def get_security(
    request: Request, ticker: str, provider: FactSetProviderDep
) -> FactSetSecurity:
    """Get security details by ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Security details including name, exchange, and identifiers.

    Raises:
        HTTPException: 404 if ticker not found.

    Example:
        GET /api/v1/factset/securities/AAPL
    """
    security = await provider.resolve_ticker(ticker)
    if not security:
        raise HTTPException(404, detail=f"Ticker '{ticker}' not found")
    return security


@router.get("/securities/{ticker}/price", response_model=FactSetPrice)
@limiter.limit("60/minute")
async def get_price(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    price_date: date | None = Query(
        default=None, alias="date", description="Price date (YYYY-MM-DD), defaults to latest"
    ),
) -> FactSetPrice:
    """Get price data for a security on a specific date.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        date: Optional date for historical price (defaults to latest available)

    Returns:
        Price data including open, high, low, close, and volume.

    Raises:
        HTTPException: 404 if no price data available.

    Example:
        GET /api/v1/factset/securities/AAPL/price
        GET /api/v1/factset/securities/AAPL/price?date=2024-01-15
    """
    price = await provider.get_price(ticker, price_date=price_date)
    if not price:
        raise HTTPException(404, detail=f"No price data for '{ticker}'")
    return price


@router.get("/securities/{ticker}/price-history", response_model=list[FactSetPrice])
@limiter.limit("30/minute")
async def get_price_history(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date | None = Query(default=None, description="End date (YYYY-MM-DD), defaults to today"),
) -> list[FactSetPrice]:
    """Get historical price data for a security over a date range.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start: Start date for the price history
        end: Optional end date (defaults to today)

    Returns:
        List of daily price data sorted by date ascending.

    Example:
        GET /api/v1/factset/securities/AAPL/price-history?start=2024-01-01&end=2024-01-31
    """
    return await provider.get_price_history(ticker, start, end)


@router.get("/securities/{ticker}/price-history/adjusted", response_model=list[FactSetPrice])
@limiter.limit("30/minute")
async def get_adjusted_price_history(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date | None = Query(default=None, description="End date (YYYY-MM-DD), defaults to today"),
) -> list[FactSetPrice]:
    """Get split and dividend adjusted historical prices.

    Prices are adjusted for stock splits and dividends to provide
    accurate return calculations across corporate actions.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start: Start date for the price history
        end: Optional end date (defaults to today)

    Returns:
        List of adjusted daily price data sorted by date ascending.

    Example:
        GET /api/v1/factset/securities/AAPL/price-history/adjusted?start=2024-01-01
    """
    return await provider.get_adjusted_price_history(ticker, start, end)


@router.get("/securities/{ticker}/fundamentals", response_model=list[FactSetFundamentals])
@limiter.limit("60/minute")
async def get_fundamentals(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    period: str = Query(
        default="annual",
        pattern="^(annual|quarterly|ltm)$",
        description="Reporting period: annual, quarterly, or ltm (last twelve months)",
    ),
    limit: int = Query(default=4, le=20, description="Number of periods to return"),
) -> list[FactSetFundamentals]:
    """Get fundamental financial data for a security.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: Reporting period type (annual, quarterly, or ltm)
        limit: Number of periods to return (default 4, max 20)

    Returns:
        List of fundamental data including revenue, earnings, and ratios.

    Example:
        GET /api/v1/factset/securities/AAPL/fundamentals?period=quarterly&limit=8
    """
    return await provider.get_fundamentals(ticker, period_type=period, limit=limit)


@router.get("/securities/{ticker}/profile")
@limiter.limit("60/minute")
async def get_company_profile(
    request: Request, ticker: str, provider: FactSetProviderDep
) -> dict[str, str | None]:
    """Get company profile and business description.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with ticker and company description text.

    Raises:
        HTTPException: 404 if no profile available.

    Example:
        GET /api/v1/factset/securities/AAPL/profile
    """
    profile = await provider.get_company_profile(ticker)
    if profile is None:
        raise HTTPException(404, detail=f"No profile for '{ticker}'")
    return {"ticker": ticker, "description": profile}


@router.get("/securities/{ticker}/corporate-actions", response_model=list[FactSetCorporateAction])
@limiter.limit("60/minute")
async def get_corporate_actions(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=50, le=200, description="Maximum number of actions to return"),
) -> list[FactSetCorporateAction]:
    """Get all corporate actions for a security.

    Includes dividends, stock splits, spinoffs, and other corporate events.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        limit: Maximum number of actions to return (default 50, max 200)

    Returns:
        List of corporate actions sorted by date descending.

    Example:
        GET /api/v1/factset/securities/AAPL/corporate-actions?limit=100
    """
    return await provider.get_corporate_actions(ticker, limit=limit)


@router.get("/securities/{ticker}/dividends", response_model=list[FactSetCorporateAction])
@limiter.limit("60/minute")
async def get_dividends(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=20, le=100, description="Maximum number of dividends to return"),
) -> list[FactSetCorporateAction]:
    """Get dividend history for a security.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        limit: Maximum number of dividends to return (default 20, max 100)

    Returns:
        List of dividend payments sorted by date descending.

    Example:
        GET /api/v1/factset/securities/AAPL/dividends?limit=10
    """
    return await provider.get_dividends(ticker, limit=limit)


@router.get("/securities/{ticker}/splits", response_model=list[FactSetCorporateAction])
@limiter.limit("60/minute")
async def get_splits(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    limit: int = Query(default=20, le=100, description="Maximum number of splits to return"),
) -> list[FactSetCorporateAction]:
    """Get stock split history for a security.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        limit: Maximum number of splits to return (default 20, max 100)

    Returns:
        List of stock splits sorted by date descending.

    Example:
        GET /api/v1/factset/securities/AAPL/splits
    """
    return await provider.get_splits(ticker, limit=limit)


@router.get("/securities/{ticker}/shares-outstanding", response_model=FactSetSharesOutstanding)
@limiter.limit("60/minute")
async def get_shares_outstanding(
    request: Request, ticker: str, provider: FactSetProviderDep
) -> FactSetSharesOutstanding:
    """Get current shares outstanding for a security.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Shares outstanding data including basic and diluted counts.

    Raises:
        HTTPException: 404 if no shares data available.

    Example:
        GET /api/v1/factset/securities/AAPL/shares-outstanding
    """
    shares = await provider.get_shares_outstanding(ticker)
    if not shares:
        raise HTTPException(404, detail=f"No shares data for '{ticker}'")
    return shares


@router.get("/securities/{ticker}/market-cap")
@limiter.limit("60/minute")
async def get_market_cap(
    request: Request, ticker: str, provider: FactSetProviderDep
) -> dict[str, object]:
    """Get current market capitalization for a security.

    Calculated as shares outstanding multiplied by current price.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with ticker and market_cap value in USD.

    Raises:
        HTTPException: 404 if market cap cannot be computed.

    Example:
        GET /api/v1/factset/securities/AAPL/market-cap
    """
    mcap = await provider.get_market_cap(ticker)
    if mcap is None:
        raise HTTPException(404, detail=f"Cannot compute market cap for '{ticker}'")
    return {"ticker": ticker, "market_cap": mcap}


@router.get("/securities/{ticker}/adjustment-factors")
@limiter.limit("60/minute")
async def get_adjustment_factors(
    request: Request,
    ticker: str,
    provider: FactSetProviderDep,
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date = Query(..., description="End date (YYYY-MM-DD)"),
) -> dict[str, float]:
    """Get price adjustment factors for splits and dividends.

    Factors can be used to manually adjust historical prices for
    accurate return calculations.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start: Start date for adjustment factors
        end: End date for adjustment factors

    Returns:
        Dictionary mapping dates (ISO format) to adjustment factors.

    Example:
        GET /api/v1/factset/securities/AAPL/adjustment-factors?start=2020-01-01&end=2024-01-01
    """
    factors = await provider.get_adjustment_factors(ticker, start, end)
    return {d.isoformat(): f for d, f in factors.items()}
