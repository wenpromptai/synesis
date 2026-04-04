"""Massive.com REST API client — free tier (5 calls/min).

API docs: https://massive.com/docs/rest
Polygon.io-compatible endpoint structure.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.massive.models import (
    Bar,
    BarsResponse,
    DailySummary,
    Dividend,
    FinancialResult,
    IndicatorValue,
    MACDValue,
    MarketHoliday,
    MarketStatus,
    NewsArticle,
    NewsInsight,
    OptionsContractRef,
    ShortInterest,
    ShortVolume,
    Split,
    TickerEvent,
    TickerInfo,
    TickerOverview,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

CACHE_PREFIX = "synesis:massive"
MASSIVE_RATE_LIMIT_PER_MINUTE = 5


class _MassiveRateLimiter:
    """Token bucket rate limiter for Massive API (5 req/min)."""

    def __init__(self, max_calls: int = MASSIVE_RATE_LIMIT_PER_MINUTE, period: float = 60.0):
        self._max_calls = max_calls
        self._period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
                if len(self._calls) < self._max_calls:
                    self._calls.append(time.monotonic())
                    return
                sleep_time = self._period - (now - self._calls[0]) + 0.05
            await asyncio.sleep(sleep_time)


_rate_limiter = _MassiveRateLimiter()


class MassiveClient:
    """Client for the Massive.com REST API (free tier).

    Usage:
        client = MassiveClient(redis=redis_client)
        bars = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-03-01")
        overview = await client.get_ticker_overview("AAPL")
        await client.close()
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._http: httpx.AsyncClient | None = None
        settings = get_settings()
        if not settings.massive_api_key:
            raise ValueError("MASSIVE_API_KEY is required")
        self._api_key = settings.massive_api_key.get_secret_value()
        self._base_url = settings.massive_api_url.rstrip("/")

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._http

    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        """Build query params, mapping _gte/_gt/_lte/_lt/_any_of to dot notation."""
        params: dict[str, Any] = {}
        for key, val in kwargs.items():
            if val is None:
                continue
            for suffix in ("_any_of", "_gte", "_gt", "_lte", "_lt"):
                if key.endswith(suffix):
                    api_key = key[: -len(suffix)] + "." + suffix[1:]
                    params[api_key] = val
                    break
            else:
                params[key] = val
        return params

    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        """Rate-limited GET request to Massive API."""
        await _rate_limiter.acquire()
        client = self._get_http()
        req_params: dict[str, Any] = {"apiKey": self._api_key}
        if params:
            req_params.update(params)
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            resp = await client.get(url, params=req_params)
            resp.raise_for_status()
            return orjson.loads(resp.content)  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            logger.warning("Massive API error", path=path, status=e.response.status_code)
            return {}
        except Exception as e:
            logger.warning("Massive API request failed", path=path, error=str(e))
            return {}

    async def _get_cached(
        self, cache_key: str, path: str, params: dict[str, Any] | None, ttl: int
    ) -> dict[str, Any] | list[Any]:
        """GET with Redis cache layer."""
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return orjson.loads(cached)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))
        data = await self._get(path, params)
        if data:
            await self._redis.set(cache_key, orjson.dumps(data), ex=ttl)
        return data

    async def _get_all_pages(
        self, path: str, params: dict[str, Any] | None, max_pages: int = 10
    ) -> list[dict[str, Any]]:
        """Follow next_url to collect all pages of results."""
        all_results: list[dict[str, Any]] = []
        data = await self._get(path, params)
        if not data or not isinstance(data, dict):
            return all_results
        all_results.extend(data.get("results", []))
        pages = 1
        while data.get("next_url") and pages < max_pages:
            next_url = data["next_url"]
            await _rate_limiter.acquire()
            try:
                resp = await self._get_http().get(
                    next_url, params={"apiKey": self._api_key}, timeout=30.0
                )
                resp.raise_for_status()
                data = orjson.loads(resp.content)
                all_results.extend(data.get("results", []))
                pages += 1
            except Exception as e:
                logger.warning("Massive pagination failed", page=pages + 1, error=str(e))
                break
        return all_results

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._http:
            await self._http.aclose()
            self._http = None

    # -------------------------------------------------------------------------
    # Aggregates
    # -------------------------------------------------------------------------

    async def get_bars(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        *,
        adjusted: bool = True,
        limit: int = 5000,
        sort: str = "asc",
    ) -> BarsResponse | None:
        """Fetch OHLCV aggregate bars for a ticker over a date range.

        Args:
            ticker: Ticker symbol (e.g. "AAPL") or options ticker (e.g. "O:AAPL260418C00200000").
            multiplier: Bar size multiplier (e.g. 1 for 1-day bars).
            timespan: Bar timespan — "minute", "hour", "day", "week", "month", "quarter", "year".
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            adjusted: Whether to adjust for splits.
            limit: Max number of bars to return (max 50000).
            sort: Sort order — "asc" or "desc".

        Returns:
            BarsResponse with ticker, bars list, and adjusted flag. None if no data.

        Example::

            bars = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-03-01")
            bars = await client.get_bars("AAPL", 5, "minute", "2026-03-01", "2026-03-01", limit=500)
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:bars:{ticker}:{multiplier}:{timespan}:{from_date}:{to_date}:{adjusted}:{limit}:{sort}"
        data = await self._get_cached(
            cache_key,
            f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            {"adjusted": str(adjusted).lower(), "limit": limit, "sort": sort},
            settings.massive_cache_ttl_bars,
        )
        if not data or not isinstance(data, dict):
            return None
        bars = [
            Bar(
                open=r["o"],
                high=r["h"],
                low=r["l"],
                close=r["c"],
                volume=r["v"],
                vwap=r.get("vw"),
                timestamp=r["t"],
                transactions=r.get("n"),
            )
            for r in data.get("results", [])
        ]
        return BarsResponse(
            ticker=data.get("ticker", ticker), bars=bars, adjusted=data.get("adjusted", adjusted)
        )

    async def get_previous_day(self, ticker: str, adjusted: bool = True) -> BarsResponse | None:
        """Fetch the previous trading day's OHLCV bar for a ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL") or options ticker.
            adjusted: Whether to adjust for splits.

        Returns:
            BarsResponse containing a single bar for the previous day. None if no data.

        Example::

            prev = await client.get_previous_day("AAPL")
            prev = await client.get_previous_day("O:AAPL260418C00200000", adjusted=False)
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:prev:{ticker}:{adjusted}"
        data = await self._get_cached(
            cache_key,
            f"v2/aggs/ticker/{ticker}/prev",
            {"adjusted": str(adjusted).lower()},
            settings.massive_cache_ttl_bars,
        )
        if not data or not isinstance(data, dict):
            return None
        bars = [
            Bar(
                open=r["o"],
                high=r["h"],
                low=r["l"],
                close=r["c"],
                volume=r["v"],
                vwap=r.get("vw"),
                timestamp=r["t"],
                transactions=r.get("n"),
            )
            for r in data.get("results", [])
        ]
        return BarsResponse(
            ticker=data.get("ticker", ticker), bars=bars, adjusted=data.get("adjusted", adjusted)
        )

    async def get_daily_summary(
        self, ticker: str, date: str, *, adjusted: bool = True
    ) -> DailySummary | None:
        """Fetch open/close prices with after-hours and pre-market for a single date.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            date: Date in YYYY-MM-DD format.
            adjusted: Whether to adjust for splits.

        Returns:
            DailySummary with OHLCV plus after_hours and pre_market prices. None if no data.

        Example::

            summary = await client.get_daily_summary("AAPL", "2026-03-28")
            summary = await client.get_daily_summary("AAPL", "2026-03-28", adjusted=False)
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:daily:{ticker}:{date}:{adjusted}"
        data = await self._get_cached(
            cache_key,
            f"v1/open-close/{ticker}/{date}",
            {"adjusted": str(adjusted).lower()},
            settings.massive_cache_ttl_bars,
        )
        if not data or not isinstance(data, dict) or data.get("status") != "OK":
            return None
        return DailySummary(
            ticker=data.get("symbol", ticker),
            date=data.get("from", date),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            after_hours=data.get("afterHours"),
            pre_market=data.get("preMarket"),
        )

    async def get_grouped_daily(
        self, date: str, *, adjusted: bool = True, include_otc: bool = False
    ) -> list[Bar] | None:
        """Fetch OHLCV bars for all tickers on a given date.

        Args:
            date: Date in YYYY-MM-DD format.
            adjusted: Whether to adjust for splits.
            include_otc: Whether to include OTC securities in the response.

        Returns:
            List of Bar objects for every ticker traded that day. None if no data.

        Example::

            bars = await client.get_grouped_daily("2026-03-28")
            bars = await client.get_grouped_daily("2026-03-28", include_otc=True)
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:grouped:{date}:{adjusted}:{include_otc}"
        params = self._build_params(
            adjusted=str(adjusted).lower(), include_otc=str(include_otc).lower()
        )
        data = await self._get_cached(
            cache_key,
            f"v2/aggs/grouped/locale/us/market/stocks/{date}",
            params if params else None,
            settings.massive_cache_ttl_bars,
        )
        if not data or not isinstance(data, dict):
            return None
        return [
            Bar(
                open=r["o"],
                high=r["h"],
                low=r["l"],
                close=r["c"],
                volume=r["v"],
                vwap=r.get("vw"),
                timestamp=r["t"],
                transactions=r.get("n"),
            )
            for r in data.get("results", [])
        ]

    # -------------------------------------------------------------------------
    # Reference Data
    # -------------------------------------------------------------------------

    async def search_tickers(
        self,
        query: str | None = None,
        *,
        market: str | None = None,
        ticker_type: str | None = None,
        active: bool | None = None,
        limit: int = 100,
        ticker: str | None = None,
        exchange: str | None = None,
        cusip: str | None = None,
        cik: str | None = None,
        date: str | None = None,
        order: str | None = None,
        sort: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
    ) -> list[TickerInfo]:
        """Search and filter ticker symbols.

        Args:
            query: Free-text search string (e.g. "Apple").
            market: Filter by market — "stocks", "crypto", "fx", "otc", "indices".
            ticker_type: Filter by ticker type code (e.g. "CS" for common stock).
            active: Filter by active status.
            limit: Max results to return (max 1000).
            ticker: Exact ticker filter.
            exchange: Filter by primary exchange MIC code.
            cusip: Filter by CUSIP.
            cik: Filter by CIK number.
            date: Point-in-time date for ticker state (YYYY-MM-DD).
            order: Sort order — "asc" or "desc".
            sort: Sort field — "ticker", "name", "market", "locale", "type", etc.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.

        Returns:
            List of TickerInfo objects. Empty list if no matches.

        Example::

            results = await client.search_tickers("Apple", market="stocks")
            results = await client.search_tickers(ticker_gte="AA", ticker_lt="AB", limit=50)
        """
        settings = get_settings()
        params = self._build_params(
            limit=limit,
            ticker=ticker,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            order=order,
            sort=sort,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
        )
        if query:
            params["search"] = query
        if market:
            params["market"] = market
        if ticker_type:
            params["type"] = ticker_type
        if active is not None:
            params["active"] = str(active).lower()

        cache_key = f"{CACHE_PREFIX}:tickers:{query}:{market}:{ticker_type}:{active}:{limit}:{sort}"
        data = await self._get_cached(
            cache_key, "v3/reference/tickers", params, settings.massive_cache_ttl_reference
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            TickerInfo(
                ticker=r["ticker"],
                name=r.get("name", ""),
                market=r.get("market", ""),
                locale=r.get("locale", ""),
                type=r.get("type", ""),
                active=r.get("active", True),
                primary_exchange=r.get("primary_exchange"),
                currency=r.get("currency_name"),
                cik=r.get("cik"),
                composite_figi=r.get("composite_figi"),
            )
            for r in data.get("results", [])
        ]

    async def get_ticker_overview(
        self, ticker: str, *, date: str | None = None
    ) -> TickerOverview | None:
        """Fetch full company details including fundamentals, branding, and contact info.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            date: Point-in-time date for historical ticker state (YYYY-MM-DD).

        Returns:
            TickerOverview with market cap, description, employees, branding, etc. None if not found.

        Example::

            overview = await client.get_ticker_overview("AAPL")
            overview = await client.get_ticker_overview("AAPL", date="2025-01-01")
        """
        settings = get_settings()
        params = self._build_params(date=date)
        cache_key = f"{CACHE_PREFIX}:overview:{ticker}:{date}"
        data = await self._get_cached(
            cache_key,
            f"v3/reference/tickers/{ticker}",
            params if params else None,
            settings.massive_cache_ttl_reference,
        )
        if not data or not isinstance(data, dict):
            return None
        r = data.get("results", data)
        if not r or not isinstance(r, dict):
            return None
        return TickerOverview(
            ticker=r.get("ticker", ticker),
            name=r.get("name", ""),
            market=r.get("market", ""),
            locale=r.get("locale", ""),
            type=r.get("type", ""),
            active=r.get("active", True),
            primary_exchange=r.get("primary_exchange"),
            currency=r.get("currency_name"),
            cik=r.get("cik"),
            composite_figi=r.get("composite_figi"),
            market_cap=r.get("market_cap"),
            description=r.get("description"),
            homepage_url=r.get("homepage_url"),
            total_employees=r.get("total_employees"),
            list_date=r.get("list_date"),
            sic_code=r.get("sic_code"),
            sic_description=r.get("sic_description"),
            weighted_shares_outstanding=r.get("weighted_shares_outstanding"),
            phone_number=r.get("phone_number"),
            address=r.get("address"),
            branding=r.get("branding"),
            ticker_root=r.get("ticker_root"),
            ticker_suffix=r.get("ticker_suffix"),
            share_class_figi=r.get("share_class_figi"),
            share_class_shares_outstanding=r.get("share_class_shares_outstanding"),
            round_lot=r.get("round_lot"),
            delisted_utc=r.get("delisted_utc"),
        )

    async def get_ticker_events(
        self, ticker: str, *, types: str | None = None
    ) -> list[TickerEvent]:
        """Fetch historical events timeline for a ticker (e.g. ticker changes, delistings).

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            types: Comma-separated event type filter (e.g. "ticker_change,delisted").

        Returns:
            List of TickerEvent objects. Empty list if no events found.

        Example::

            events = await client.get_ticker_events("META")
            events = await client.get_ticker_events("META", types="ticker_change")
        """
        settings = get_settings()
        params = self._build_params(types=types)
        cache_key = f"{CACHE_PREFIX}:events:{ticker}:{types}"
        data = await self._get_cached(
            cache_key,
            f"vX/reference/tickers/{ticker}/events",
            params if params else None,
            settings.massive_cache_ttl_static,
        )
        if not data or not isinstance(data, dict):
            return []
        results = data.get("results", {})
        events = results.get("events", []) if isinstance(results, dict) else []
        return [
            TickerEvent(
                type=e.get("type", ""),
                date=e.get("date", ""),
                ticker_change=e.get("ticker_change"),
            )
            for e in events
        ]

    async def get_related_tickers(self, ticker: str) -> list[str]:
        """Fetch related company tickers based on sector, industry, and similar attributes.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").

        Returns:
            List of related ticker strings (e.g. ["MSFT", "GOOG"]). Empty list if none found.

        Example::

            related = await client.get_related_tickers("AAPL")
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:related:{ticker}"
        data = await self._get_cached(
            cache_key, f"v1/related-companies/{ticker}", None, settings.massive_cache_ttl_static
        )
        if not data or not isinstance(data, dict):
            return []
        return [r["ticker"] for r in data.get("results", []) if "ticker" in r]

    async def get_ticker_types(
        self, *, asset_class: str | None = None, locale: str | None = None
    ) -> list[dict[str, str]]:
        """Fetch all supported ticker type codes.

        Args:
            asset_class: Filter by asset class — "stocks", "options", "crypto", "fx", "indices".
            locale: Filter by locale — "us", "global".

        Returns:
            List of dicts with "code", "description", "asset_class_type", "locale" keys.

        Example::

            types = await client.get_ticker_types()
            types = await client.get_ticker_types(asset_class="stocks", locale="us")
        """
        settings = get_settings()
        params = self._build_params(asset_class=asset_class, locale=locale)
        cache_key = f"{CACHE_PREFIX}:ticker_types:{asset_class}:{locale}"
        data = await self._get_cached(
            cache_key,
            "v3/reference/tickers/types",
            params if params else None,
            settings.massive_cache_ttl_static,
        )
        if not data or not isinstance(data, dict):
            return []
        results: list[dict[str, str]] = data.get("results", [])
        return results

    async def get_exchanges(
        self, *, asset_class: str | None = None, locale: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch exchange metadata including name, MIC code, and operating hours.

        Args:
            asset_class: Filter by asset class — "stocks", "options", "crypto", "fx".
            locale: Filter by locale — "us", "global".

        Returns:
            List of exchange metadata dicts. Empty list if none found.

        Example::

            exchanges = await client.get_exchanges()
            exchanges = await client.get_exchanges(asset_class="stocks", locale="us")
        """
        settings = get_settings()
        params = self._build_params(asset_class=asset_class, locale=locale)
        cache_key = f"{CACHE_PREFIX}:exchanges:{asset_class}:{locale}"
        data = await self._get_cached(
            cache_key,
            "v3/reference/exchanges",
            params if params else None,
            settings.massive_cache_ttl_static,
        )
        if not data or not isinstance(data, dict):
            return []
        results: list[dict[str, Any]] = data.get("results", [])
        return results

    async def get_conditions(
        self,
        *,
        asset_class: str | None = None,
        data_type: str | None = None,
        id: int | None = None,  # noqa: A002
        sip: str | None = None,
        order: str | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch trade/quote condition code definitions.

        Args:
            asset_class: Filter by asset class — "stocks", "options", "crypto", "fx".
            data_type: Filter by data type — "trade", "bbo", "nbbo".
            id: Filter by condition ID.
            sip: Filter by SIP — "CTA", "UTP", "OPRA".
            order: Sort order — "asc" or "desc".
            limit: Max results to return.
            sort: Sort field.

        Returns:
            List of condition metadata dicts with "id", "type", "name", etc. Empty list if none found.

        Example::

            conditions = await client.get_conditions()
            conditions = await client.get_conditions(asset_class="stocks", data_type="trade")
        """
        settings = get_settings()
        params = self._build_params(
            asset_class=asset_class,
            data_type=data_type,
            id=id,
            sip=sip,
            order=order,
            limit=limit,
            sort=sort,
        )
        cache_key = f"{CACHE_PREFIX}:conditions:{asset_class}:{data_type}:{id}:{sip}"
        data = await self._get_cached(
            cache_key,
            "v3/reference/conditions",
            params if params else None,
            settings.massive_cache_ttl_static,
        )
        if not data or not isinstance(data, dict):
            return []
        results: list[dict[str, Any]] = data.get("results", [])
        return results

    # -------------------------------------------------------------------------
    # Fundamentals & Corporate Actions
    # -------------------------------------------------------------------------

    async def get_financials(
        self,
        ticker: str,
        *,
        timeframe: str | None = None,
        limit: int = 5,
        cik: str | None = None,
        company_name: str | None = None,
        company_name_search: str | None = None,
        sic: str | None = None,
        filing_date: str | None = None,
        period_of_report_date: str | None = None,
        include_sources: bool | None = None,
        order: str | None = None,
        sort: str | None = None,
        filing_date_gte: str | None = None,
        filing_date_gt: str | None = None,
        filing_date_lte: str | None = None,
        filing_date_lt: str | None = None,
        period_of_report_date_gte: str | None = None,
        period_of_report_date_gt: str | None = None,
        period_of_report_date_lte: str | None = None,
        period_of_report_date_lt: str | None = None,
    ) -> list[FinancialResult]:
        """Fetch combined financial statements from SEC XBRL filings.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timeframe: Period — "annual", "quarterly", "ttm".
            limit: Max results to return.
            cik: Filter by CIK number.
            company_name: Exact company name filter.
            company_name_search: Fuzzy search on company name.
            sic: Filter by SIC code.
            filing_date: Exact filing date (YYYY-MM-DD).
            period_of_report_date: Exact period-of-report date (YYYY-MM-DD).
            include_sources: Whether to include XBRL source data.
            order: Sort order — "asc" or "desc".
            sort: Sort field — "filing_date", "period_of_report_date".
            filing_date_gte: Filing date >= range filter.
            filing_date_gt: Filing date > range filter.
            filing_date_lte: Filing date <= range filter.
            filing_date_lt: Filing date < range filter.
            period_of_report_date_gte: Period-of-report date >= range filter.
            period_of_report_date_gt: Period-of-report date > range filter.
            period_of_report_date_lte: Period-of-report date <= range filter.
            period_of_report_date_lt: Period-of-report date < range filter.

        Returns:
            List of FinancialResult objects with income/balance/cashflow data. Empty list if none found.

        Example::

            fins = await client.get_financials("AAPL", timeframe="quarterly", limit=4)
            fins = await client.get_financials("AAPL", filing_date_gte="2025-01-01")
            fins = await client.get_financials("AAPL", company_name_search="apple")
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            timeframe=timeframe,
            limit=limit,
            cik=cik,
            company_name=company_name,
            sic=sic,
            filing_date=filing_date,
            period_of_report_date=period_of_report_date,
            include_sources=include_sources,
            order=order,
            sort=sort,
            filing_date_gte=filing_date_gte,
            filing_date_gt=filing_date_gt,
            filing_date_lte=filing_date_lte,
            filing_date_lt=filing_date_lt,
            period_of_report_date_gte=period_of_report_date_gte,
            period_of_report_date_gt=period_of_report_date_gt,
            period_of_report_date_lte=period_of_report_date_lte,
            period_of_report_date_lt=period_of_report_date_lt,
        )
        if company_name_search:
            params["company_name.search"] = company_name_search
        cache_key = f"{CACHE_PREFIX}:financials:{ticker}:{timeframe}:{limit}:{filing_date}:{sort}"
        data = await self._get_cached(
            cache_key, "vX/reference/financials", params, settings.massive_cache_ttl_fundamentals
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            FinancialResult(
                tickers=r.get("tickers", []),
                start_date=r.get("start_date", ""),
                end_date=r.get("end_date", ""),
                filing_date=r.get("filing_date"),
                fiscal_period=r.get("fiscal_period"),
                fiscal_year=r.get("fiscal_year"),
                timeframe=r.get("timeframe"),
                financials=r.get("financials", {}),
            )
            for r in data.get("results", [])
        ]

    async def get_dividends(
        self,
        ticker: str,
        *,
        limit: int = 100,
        ex_dividend_date: str | None = None,
        frequency: int | None = None,
        distribution_type: str | None = None,
        sort: str | None = None,
        ticker_any_of: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
        frequency_gte: int | None = None,
        frequency_gt: int | None = None,
        frequency_lte: int | None = None,
        frequency_lt: int | None = None,
        distribution_type_any_of: str | None = None,
        ex_dividend_date_gte: str | None = None,
        ex_dividend_date_gt: str | None = None,
        ex_dividend_date_lte: str | None = None,
        ex_dividend_date_lt: str | None = None,
    ) -> list[Dividend]:
        """Fetch historical cash dividend distributions.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            limit: Max results to return (max 1000).
            ex_dividend_date: Exact ex-dividend date (YYYY-MM-DD).
            frequency: Exact dividend frequency (0=one-time, 1=annual, 2=bi-annual, 4=quarterly, 12=monthly).
            distribution_type: Exact distribution type (e.g. "CD" for cash dividend).
            sort: Sort field — "ex_dividend_date", "pay_date", "ticker".
            ticker_any_of: Comma-separated list of tickers to include.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.
            frequency_gte: Frequency >= range filter.
            frequency_gt: Frequency > range filter.
            frequency_lte: Frequency <= range filter.
            frequency_lt: Frequency < range filter.
            distribution_type_any_of: Comma-separated list of distribution types.
            ex_dividend_date_gte: Ex-dividend date >= range filter.
            ex_dividend_date_gt: Ex-dividend date > range filter.
            ex_dividend_date_lte: Ex-dividend date <= range filter.
            ex_dividend_date_lt: Ex-dividend date < range filter.

        Returns:
            List of Dividend objects. Empty list if none found.

        Example::

            divs = await client.get_dividends("AAPL", limit=10)
            divs = await client.get_dividends("AAPL", ex_dividend_date_gte="2025-01-01", frequency=4)
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            limit=limit,
            ex_dividend_date=ex_dividend_date,
            frequency=frequency,
            distribution_type=distribution_type,
            sort=sort,
            ticker_any_of=ticker_any_of,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
            frequency_gte=frequency_gte,
            frequency_gt=frequency_gt,
            frequency_lte=frequency_lte,
            frequency_lt=frequency_lt,
            distribution_type_any_of=distribution_type_any_of,
            ex_dividend_date_gte=ex_dividend_date_gte,
            ex_dividend_date_gt=ex_dividend_date_gt,
            ex_dividend_date_lte=ex_dividend_date_lte,
            ex_dividend_date_lt=ex_dividend_date_lt,
        )
        cache_key = f"{CACHE_PREFIX}:dividends:{ticker}:{limit}:{ex_dividend_date}:{sort}"
        data = await self._get_cached(
            cache_key,
            "stocks/v1/dividends",
            params,
            settings.massive_cache_ttl_fundamentals,
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            Dividend(
                ticker=r.get("ticker", ticker),
                ex_dividend_date=r["ex_dividend_date"],
                pay_date=r.get("pay_date"),
                record_date=r.get("record_date"),
                declaration_date=r.get("declaration_date"),
                cash_amount=r["cash_amount"],
                currency=r.get("currency", "USD"),
                frequency=r.get("frequency"),
                distribution_type=r.get("distribution_type"),
            )
            for r in data.get("results", [])
        ]

    async def get_splits(
        self,
        ticker: str,
        *,
        limit: int = 100,
        execution_date: str | None = None,
        adjustment_type: str | None = None,
        sort: str | None = None,
        ticker_any_of: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
        adjustment_type_any_of: str | None = None,
        execution_date_gte: str | None = None,
        execution_date_gt: str | None = None,
        execution_date_lte: str | None = None,
        execution_date_lt: str | None = None,
    ) -> list[Split]:
        """Fetch stock split history for a ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            limit: Max results to return (max 1000).
            execution_date: Exact execution date (YYYY-MM-DD).
            adjustment_type: Exact adjustment type filter.
            sort: Sort field — "execution_date", "ticker".
            ticker_any_of: Comma-separated list of tickers to include.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.
            adjustment_type_any_of: Comma-separated list of adjustment types.
            execution_date_gte: Execution date >= range filter.
            execution_date_gt: Execution date > range filter.
            execution_date_lte: Execution date <= range filter.
            execution_date_lt: Execution date < range filter.

        Returns:
            List of Split objects. Empty list if none found.

        Example::

            splits = await client.get_splits("AAPL")
            splits = await client.get_splits("AAPL", execution_date_gte="2020-01-01")
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            limit=limit,
            execution_date=execution_date,
            adjustment_type=adjustment_type,
            sort=sort,
            ticker_any_of=ticker_any_of,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
            adjustment_type_any_of=adjustment_type_any_of,
            execution_date_gte=execution_date_gte,
            execution_date_gt=execution_date_gt,
            execution_date_lte=execution_date_lte,
            execution_date_lt=execution_date_lt,
        )
        cache_key = f"{CACHE_PREFIX}:splits:{ticker}:{limit}:{execution_date}:{sort}"
        data = await self._get_cached(
            cache_key,
            "stocks/v1/splits",
            params,
            settings.massive_cache_ttl_fundamentals,
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            Split(
                ticker=r.get("ticker", ticker),
                execution_date=r["execution_date"],
                split_from=r["split_from"],
                split_to=r["split_to"],
                adjustment_type=r.get("adjustment_type"),
            )
            for r in data.get("results", [])
        ]

    async def get_short_interest(
        self,
        ticker: str,
        *,
        limit: int = 10,
        settlement_date: str | None = None,
        days_to_cover: float | None = None,
        avg_daily_volume: int | None = None,
        sort: str | None = None,
        ticker_any_of: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
        settlement_date_any_of: str | None = None,
        days_to_cover_any_of: str | None = None,
        avg_daily_volume_any_of: str | None = None,
        settlement_date_gte: str | None = None,
        settlement_date_gt: str | None = None,
        settlement_date_lte: str | None = None,
        settlement_date_lt: str | None = None,
        days_to_cover_gte: float | None = None,
        days_to_cover_gt: float | None = None,
        days_to_cover_lte: float | None = None,
        days_to_cover_lt: float | None = None,
        avg_daily_volume_gte: int | None = None,
        avg_daily_volume_gt: int | None = None,
        avg_daily_volume_lte: int | None = None,
        avg_daily_volume_lt: int | None = None,
    ) -> list[ShortInterest]:
        """Fetch bi-monthly FINRA short interest data.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            limit: Max results to return.
            settlement_date: Exact settlement date (YYYY-MM-DD).
            days_to_cover: Exact days-to-cover value.
            avg_daily_volume: Exact average daily volume.
            sort: Sort field — "settlement_date", "ticker".
            ticker_any_of: Comma-separated list of tickers.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.
            settlement_date_any_of: Comma-separated list of settlement dates.
            days_to_cover_any_of: Comma-separated list of days-to-cover values.
            avg_daily_volume_any_of: Comma-separated list of average daily volume values.
            settlement_date_gte: Settlement date >= range filter.
            settlement_date_gt: Settlement date > range filter.
            settlement_date_lte: Settlement date <= range filter.
            settlement_date_lt: Settlement date < range filter.
            days_to_cover_gte: Days-to-cover >= range filter.
            days_to_cover_gt: Days-to-cover > range filter.
            days_to_cover_lte: Days-to-cover <= range filter.
            days_to_cover_lt: Days-to-cover < range filter.
            avg_daily_volume_gte: Average daily volume >= range filter.
            avg_daily_volume_gt: Average daily volume > range filter.
            avg_daily_volume_lte: Average daily volume <= range filter.
            avg_daily_volume_lt: Average daily volume < range filter.

        Returns:
            List of ShortInterest objects. Empty list if none found.

        Example::

            si = await client.get_short_interest("AAPL", limit=5)
            si = await client.get_short_interest("AAPL", settlement_date_gte="2026-01-01")
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            limit=limit,
            settlement_date=settlement_date,
            days_to_cover=days_to_cover,
            avg_daily_volume=avg_daily_volume,
            sort=sort,
            ticker_any_of=ticker_any_of,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
            settlement_date_any_of=settlement_date_any_of,
            days_to_cover_any_of=days_to_cover_any_of,
            avg_daily_volume_any_of=avg_daily_volume_any_of,
            settlement_date_gte=settlement_date_gte,
            settlement_date_gt=settlement_date_gt,
            settlement_date_lte=settlement_date_lte,
            settlement_date_lt=settlement_date_lt,
            days_to_cover_gte=days_to_cover_gte,
            days_to_cover_gt=days_to_cover_gt,
            days_to_cover_lte=days_to_cover_lte,
            days_to_cover_lt=days_to_cover_lt,
            avg_daily_volume_gte=avg_daily_volume_gte,
            avg_daily_volume_gt=avg_daily_volume_gt,
            avg_daily_volume_lte=avg_daily_volume_lte,
            avg_daily_volume_lt=avg_daily_volume_lt,
        )
        cache_key = f"{CACHE_PREFIX}:short_interest:{ticker}:{limit}:{settlement_date}:{sort}"
        data = await self._get_cached(
            cache_key,
            "stocks/v1/short-interest",
            params,
            settings.massive_cache_ttl_fundamentals,
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            ShortInterest(
                ticker=r.get("ticker", ticker),
                settlement_date=r["settlement_date"],
                short_interest=r["short_interest"],
                avg_daily_volume=r.get("avg_daily_volume"),
                days_to_cover=r.get("days_to_cover"),
            )
            for r in data.get("results", [])
        ]

    async def get_short_volume(
        self,
        ticker: str,
        *,
        limit: int = 10,
        date: str | None = None,
        sort: str | None = None,
        ticker_any_of: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
        date_any_of: str | None = None,
        short_volume_ratio: float | None = None,
        short_volume_ratio_any_of: str | None = None,
        short_volume_ratio_gte: float | None = None,
        short_volume_ratio_gt: float | None = None,
        short_volume_ratio_lte: float | None = None,
        short_volume_ratio_lt: float | None = None,
        date_gte: str | None = None,
        date_gt: str | None = None,
        date_lte: str | None = None,
        date_lt: str | None = None,
    ) -> list[ShortVolume]:
        """Fetch daily short sale volume data.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            limit: Max results to return.
            date: Exact date filter (YYYY-MM-DD).
            sort: Sort field — "date", "ticker".
            ticker_any_of: Comma-separated list of tickers.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.
            date_any_of: Comma-separated list of dates.
            short_volume_ratio: Exact short volume ratio filter.
            short_volume_ratio_any_of: Comma-separated list of short volume ratios.
            short_volume_ratio_gte: Short volume ratio >= range filter.
            short_volume_ratio_gt: Short volume ratio > range filter.
            short_volume_ratio_lte: Short volume ratio <= range filter.
            short_volume_ratio_lt: Short volume ratio < range filter.
            date_gte: Date >= range filter.
            date_gt: Date > range filter.
            date_lte: Date <= range filter.
            date_lt: Date < range filter.

        Returns:
            List of ShortVolume objects. Empty list if none found.

        Example::

            sv = await client.get_short_volume("AAPL", limit=5)
            sv = await client.get_short_volume("AAPL", date_gte="2026-03-01", short_volume_ratio_gte=0.5)
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            limit=limit,
            date=date,
            sort=sort,
            ticker_any_of=ticker_any_of,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
            date_any_of=date_any_of,
            short_volume_ratio=short_volume_ratio,
            short_volume_ratio_any_of=short_volume_ratio_any_of,
            short_volume_ratio_gte=short_volume_ratio_gte,
            short_volume_ratio_gt=short_volume_ratio_gt,
            short_volume_ratio_lte=short_volume_ratio_lte,
            short_volume_ratio_lt=short_volume_ratio_lt,
            date_gte=date_gte,
            date_gt=date_gt,
            date_lte=date_lte,
            date_lt=date_lt,
        )
        cache_key = f"{CACHE_PREFIX}:short_volume:{ticker}:{limit}:{date}:{sort}"
        data = await self._get_cached(
            cache_key,
            "stocks/v1/short-volume",
            params,
            settings.massive_cache_ttl_fundamentals,
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            ShortVolume(
                ticker=r.get("ticker", ticker),
                date=r["date"],
                short_volume=r["short_volume"],
                total_volume=r["total_volume"],
                exempt_volume=r.get("exempt_volume"),
            )
            for r in data.get("results", [])
        ]

    # -------------------------------------------------------------------------
    # News
    # -------------------------------------------------------------------------

    async def get_news(
        self,
        ticker: str | None = None,
        *,
        published_utc_gte: str | None = None,
        limit: int = 10,
        order: str | None = None,
        sort: str | None = None,
        published_utc: str | None = None,
        published_utc_gt: str | None = None,
        published_utc_lte: str | None = None,
        published_utc_lt: str | None = None,
        ticker_gte: str | None = None,
        ticker_gt: str | None = None,
        ticker_lte: str | None = None,
        ticker_lt: str | None = None,
    ) -> list[NewsArticle]:
        """Fetch news articles with per-ticker sentiment analysis.

        Args:
            ticker: Filter by ticker symbol (e.g. "AAPL").
            published_utc_gte: Published date >= range filter (YYYY-MM-DD or ISO 8601).
            limit: Max articles to return (max 1000).
            order: Sort order — "asc" or "desc".
            sort: Sort field — "published_utc".
            published_utc: Exact published date filter.
            published_utc_gt: Published date > range filter.
            published_utc_lte: Published date <= range filter.
            published_utc_lt: Published date < range filter.
            ticker_gte: Ticker >= range filter.
            ticker_gt: Ticker > range filter.
            ticker_lte: Ticker <= range filter.
            ticker_lt: Ticker < range filter.

        Returns:
            List of NewsArticle objects with sentiment insights. Empty list if none found.

        Example::

            news = await client.get_news("AAPL", limit=5)
            news = await client.get_news(published_utc_gte="2026-03-01", ticker_gte="A", ticker_lt="B")
        """
        settings = get_settings()
        params = self._build_params(
            ticker=ticker,
            limit=limit,
            order=order,
            sort=sort,
            published_utc=published_utc,
            published_utc_gte=published_utc_gte,
            published_utc_gt=published_utc_gt,
            published_utc_lte=published_utc_lte,
            published_utc_lt=published_utc_lt,
            ticker_gte=ticker_gte,
            ticker_gt=ticker_gt,
            ticker_lte=ticker_lte,
            ticker_lt=ticker_lt,
        )
        cache_key = f"{CACHE_PREFIX}:news:{ticker}:{published_utc_gte}:{limit}:{sort}"
        data = await self._get_cached(
            cache_key, "v2/reference/news", params, settings.massive_cache_ttl_news
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            NewsArticle(
                id=r.get("id", ""),
                title=r.get("title", ""),
                published_utc=r.get("published_utc", ""),
                article_url=r.get("article_url", ""),
                tickers=r.get("tickers", []),
                description=r.get("description"),
                keywords=r.get("keywords", []),
                insights=[
                    NewsInsight(
                        ticker=i.get("ticker", ""),
                        sentiment=i.get("sentiment"),
                        sentiment_reasoning=i.get("sentiment_reasoning"),
                    )
                    for i in r.get("insights", [])
                ],
                author=r.get("author"),
                image_url=r.get("image_url"),
            )
            for r in data.get("results", [])
        ]

    # -------------------------------------------------------------------------
    # Technical Indicators
    # -------------------------------------------------------------------------

    async def _get_indicator(
        self, indicator: str, ticker: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Shared helper for technical indicator endpoints."""
        settings = get_settings()
        window = params.get("window", "")
        timespan = params.get("timespan", "day")
        limit = params.get("limit", 10)
        series_type = params.get("series_type", "close")
        cache_key = (
            f"{CACHE_PREFIX}:ind:{indicator}:{ticker}:{timespan}:{window}:{limit}:{series_type}"
        )
        data = await self._get_cached(
            cache_key,
            f"v1/indicators/{indicator}/{ticker}",
            params,
            settings.massive_cache_ttl_indicators,
        )
        if not data or not isinstance(data, dict):
            return []
        results = data.get("results", {})
        return results.get("values", []) if isinstance(results, dict) else []

    async def get_sma(
        self,
        ticker: str,
        *,
        timespan: str = "day",
        window: int = 50,
        limit: int = 10,
        series_type: str = "close",
        order: str | None = None,
        adjusted: bool = True,
        expand_underlying: bool = False,
        timestamp: str | None = None,
        timestamp_gte: str | None = None,
        timestamp_gt: str | None = None,
        timestamp_lte: str | None = None,
        timestamp_lt: str | None = None,
    ) -> list[IndicatorValue]:
        """Fetch Simple Moving Average values for a ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timespan: Bar timespan — "minute", "hour", "day", "week", "month", "quarter", "year".
            window: Lookback window size (number of bars).
            limit: Max data points to return.
            series_type: Price type — "close", "open", "high", "low".
            order: Sort order — "asc" or "desc".
            adjusted: Whether to adjust for splits.
            expand_underlying: Whether to include underlying aggregate bars in response.
            timestamp: Exact timestamp filter (YYYY-MM-DD or Unix ms).
            timestamp_gte: Timestamp >= range filter.
            timestamp_gt: Timestamp > range filter.
            timestamp_lte: Timestamp <= range filter.
            timestamp_lt: Timestamp < range filter.

        Returns:
            List of IndicatorValue objects with timestamp and SMA value. Empty list if none found.

        Example::

            sma = await client.get_sma("AAPL", window=200, limit=30)
            sma = await client.get_sma("AAPL", timestamp_gte="2026-01-01", timestamp_lte="2026-03-01")
        """
        params = self._build_params(
            timespan=timespan,
            window=window,
            limit=limit,
            series_type=series_type,
            order=order,
            adjusted=str(adjusted).lower(),
            expand_underlying=str(expand_underlying).lower(),
            timestamp=timestamp,
            timestamp_gte=timestamp_gte,
            timestamp_gt=timestamp_gt,
            timestamp_lte=timestamp_lte,
            timestamp_lt=timestamp_lt,
        )
        values = await self._get_indicator("sma", ticker, params)
        return [IndicatorValue(timestamp=v["timestamp"], value=v["value"]) for v in values]

    async def get_ema(
        self,
        ticker: str,
        *,
        timespan: str = "day",
        window: int = 20,
        limit: int = 10,
        series_type: str = "close",
        order: str | None = None,
        adjusted: bool = True,
        expand_underlying: bool = False,
        timestamp: str | None = None,
        timestamp_gte: str | None = None,
        timestamp_gt: str | None = None,
        timestamp_lte: str | None = None,
        timestamp_lt: str | None = None,
    ) -> list[IndicatorValue]:
        """Fetch Exponential Moving Average values for a ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timespan: Bar timespan — "minute", "hour", "day", "week", "month", "quarter", "year".
            window: Lookback window size (number of bars).
            limit: Max data points to return.
            series_type: Price type — "close", "open", "high", "low".
            order: Sort order — "asc" or "desc".
            adjusted: Whether to adjust for splits.
            expand_underlying: Whether to include underlying aggregate bars in response.
            timestamp: Exact timestamp filter (YYYY-MM-DD or Unix ms).
            timestamp_gte: Timestamp >= range filter.
            timestamp_gt: Timestamp > range filter.
            timestamp_lte: Timestamp <= range filter.
            timestamp_lt: Timestamp < range filter.

        Returns:
            List of IndicatorValue objects with timestamp and EMA value. Empty list if none found.

        Example::

            ema = await client.get_ema("AAPL", window=12, limit=30)
            ema = await client.get_ema("AAPL", timestamp_gte="2026-01-01")
        """
        params = self._build_params(
            timespan=timespan,
            window=window,
            limit=limit,
            series_type=series_type,
            order=order,
            adjusted=str(adjusted).lower(),
            expand_underlying=str(expand_underlying).lower(),
            timestamp=timestamp,
            timestamp_gte=timestamp_gte,
            timestamp_gt=timestamp_gt,
            timestamp_lte=timestamp_lte,
            timestamp_lt=timestamp_lt,
        )
        values = await self._get_indicator("ema", ticker, params)
        return [IndicatorValue(timestamp=v["timestamp"], value=v["value"]) for v in values]

    async def get_rsi(
        self,
        ticker: str,
        *,
        timespan: str = "day",
        window: int = 14,
        limit: int = 10,
        series_type: str = "close",
        order: str | None = None,
        adjusted: bool = True,
        expand_underlying: bool = False,
        timestamp: str | None = None,
        timestamp_gte: str | None = None,
        timestamp_gt: str | None = None,
        timestamp_lte: str | None = None,
        timestamp_lt: str | None = None,
    ) -> list[IndicatorValue]:
        """Fetch Relative Strength Index values for a ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timespan: Bar timespan — "minute", "hour", "day", "week", "month", "quarter", "year".
            window: Lookback window size (number of bars).
            limit: Max data points to return.
            series_type: Price type — "close", "open", "high", "low".
            order: Sort order — "asc" or "desc".
            adjusted: Whether to adjust for splits.
            expand_underlying: Whether to include underlying aggregate bars in response.
            timestamp: Exact timestamp filter (YYYY-MM-DD or Unix ms).
            timestamp_gte: Timestamp >= range filter.
            timestamp_gt: Timestamp > range filter.
            timestamp_lte: Timestamp <= range filter.
            timestamp_lt: Timestamp < range filter.

        Returns:
            List of IndicatorValue objects with timestamp and RSI value. Empty list if none found.

        Example::

            rsi = await client.get_rsi("AAPL", window=14, limit=30)
            rsi = await client.get_rsi("AAPL", timestamp_gte="2026-01-01", timestamp_lte="2026-03-01")
        """
        params = self._build_params(
            timespan=timespan,
            window=window,
            limit=limit,
            series_type=series_type,
            order=order,
            adjusted=str(adjusted).lower(),
            expand_underlying=str(expand_underlying).lower(),
            timestamp=timestamp,
            timestamp_gte=timestamp_gte,
            timestamp_gt=timestamp_gt,
            timestamp_lte=timestamp_lte,
            timestamp_lt=timestamp_lt,
        )
        values = await self._get_indicator("rsi", ticker, params)
        return [IndicatorValue(timestamp=v["timestamp"], value=v["value"]) for v in values]

    async def get_macd(
        self,
        ticker: str,
        *,
        timespan: str = "day",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        limit: int = 10,
        series_type: str = "close",
        order: str | None = None,
        adjusted: bool = True,
        expand_underlying: bool = False,
        timestamp: str | None = None,
        timestamp_gte: str | None = None,
        timestamp_gt: str | None = None,
        timestamp_lte: str | None = None,
        timestamp_lt: str | None = None,
    ) -> list[MACDValue]:
        """Fetch MACD indicator values with signal line and histogram.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timespan: Bar timespan — "minute", "hour", "day", "week", "month", "quarter", "year".
            short_window: Fast EMA period.
            long_window: Slow EMA period.
            signal_window: Signal line EMA period.
            limit: Max data points to return.
            series_type: Price type — "close", "open", "high", "low".
            order: Sort order — "asc" or "desc".
            adjusted: Whether to adjust for splits.
            expand_underlying: Whether to include underlying aggregate bars in response.
            timestamp: Exact timestamp filter (YYYY-MM-DD or Unix ms).
            timestamp_gte: Timestamp >= range filter.
            timestamp_gt: Timestamp > range filter.
            timestamp_lte: Timestamp <= range filter.
            timestamp_lt: Timestamp < range filter.

        Returns:
            List of MACDValue objects with value, signal, and histogram. Empty list if none found.

        Example::

            macd = await client.get_macd("AAPL", limit=30)
            macd = await client.get_macd("AAPL", short_window=8, long_window=21, timestamp_gte="2026-01-01")
        """
        settings = get_settings()
        params = self._build_params(
            timespan=timespan,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
            limit=limit,
            series_type=series_type,
            order=order,
            adjusted=str(adjusted).lower(),
            expand_underlying=str(expand_underlying).lower(),
            timestamp=timestamp,
            timestamp_gte=timestamp_gte,
            timestamp_gt=timestamp_gt,
            timestamp_lte=timestamp_lte,
            timestamp_lt=timestamp_lt,
        )
        cache_key = f"{CACHE_PREFIX}:ind:macd:{ticker}:{timespan}:{short_window}:{long_window}:{signal_window}:{limit}:{series_type}"
        data = await self._get_cached(
            cache_key, f"v1/indicators/macd/{ticker}", params, settings.massive_cache_ttl_indicators
        )
        if not data or not isinstance(data, dict):
            return []
        results = data.get("results", {})
        values = results.get("values", []) if isinstance(results, dict) else []
        return [
            MACDValue(
                timestamp=v["timestamp"],
                value=v["value"],
                signal=v["signal"],
                histogram=v["histogram"],
            )
            for v in values
        ]

    # -------------------------------------------------------------------------
    # Market Operations
    # -------------------------------------------------------------------------

    async def get_market_status(self) -> MarketStatus | None:
        """Fetch current market open/closed/after-hours status.

        Returns:
            MarketStatus with market state, server time, and per-exchange statuses. None if unavailable.

        Example::

            status = await client.get_market_status()
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:market_status"
        data = await self._get_cached(
            cache_key, "v1/marketstatus/now", None, settings.massive_cache_ttl_market_status
        )
        if not data or not isinstance(data, dict) or "market" not in data:
            return None
        return MarketStatus(
            market=data["market"],
            early_hours=data.get("earlyHours", False),
            after_hours=data.get("afterHours", False),
            server_time=data.get("serverTime", ""),
            exchanges=data.get("exchanges", {}),
            currencies=data.get("currencies", {}),
        )

    async def get_market_holidays(self) -> list[MarketHoliday]:
        """Fetch upcoming market holidays and early close dates.

        Returns:
            List of MarketHoliday objects. Empty list if none found.

        Example::

            holidays = await client.get_market_holidays()
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:holidays"
        data = await self._get_cached(
            cache_key, "v1/marketstatus/upcoming", None, settings.massive_cache_ttl_static
        )
        if not data or not isinstance(data, list):
            return []
        return [
            MarketHoliday(
                date=h.get("date", ""),
                exchange=h.get("exchange", ""),
                name=h.get("name", ""),
                status=h.get("status", ""),
            )
            for h in data
        ]

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------

    async def get_options_contracts(
        self,
        underlying_ticker: str,
        *,
        option_ticker: str | None = None,
        contract_type: str | None = None,
        expiration_date: str | None = None,
        strike_price: float | None = None,
        expired: bool = False,
        limit: int = 1000,
        as_of: str | None = None,
        order: str | None = None,
        sort: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_gt: float | None = None,
        strike_price_lte: float | None = None,
        strike_price_lt: float | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_gt: str | None = None,
        expiration_date_lte: str | None = None,
        expiration_date_lt: str | None = None,
        underlying_ticker_gte: str | None = None,
        underlying_ticker_gt: str | None = None,
        underlying_ticker_lte: str | None = None,
        underlying_ticker_lt: str | None = None,
    ) -> list[OptionsContractRef]:
        """Fetch options contract listings (reference data, no pricing).

        Args:
            underlying_ticker: Underlying stock ticker (e.g. "AAPL").
            option_ticker: Specific options contract ticker (maps to API param "ticker").
            contract_type: Filter by type — "call" or "put".
            expiration_date: Exact expiration date (YYYY-MM-DD).
            strike_price: Exact strike price.
            expired: Whether to include expired contracts.
            limit: Max results to return (max 1000).
            as_of: Point-in-time date (YYYY-MM-DD).
            order: Sort order — "asc" or "desc".
            sort: Sort field — "ticker", "expiration_date", "strike_price".
            strike_price_gte: Strike price >= range filter.
            strike_price_gt: Strike price > range filter.
            strike_price_lte: Strike price <= range filter.
            strike_price_lt: Strike price < range filter.
            expiration_date_gte: Expiration date >= range filter.
            expiration_date_gt: Expiration date > range filter.
            expiration_date_lte: Expiration date <= range filter.
            expiration_date_lt: Expiration date < range filter.
            underlying_ticker_gte: Underlying ticker >= range filter.
            underlying_ticker_gt: Underlying ticker > range filter.
            underlying_ticker_lte: Underlying ticker <= range filter.
            underlying_ticker_lt: Underlying ticker < range filter.

        Returns:
            List of OptionsContractRef objects. Empty list if none found.

        Example::

            contracts = await client.get_options_contracts("AAPL", contract_type="call")
            contracts = await client.get_options_contracts("AAPL", strike_price_gte=150, strike_price_lte=200)
        """
        settings = get_settings()
        params = self._build_params(
            underlying_ticker=underlying_ticker,
            limit=limit,
            contract_type=contract_type,
            expiration_date=expiration_date,
            strike_price=strike_price,
            as_of=as_of,
            order=order,
            sort=sort,
            strike_price_gte=strike_price_gte,
            strike_price_gt=strike_price_gt,
            strike_price_lte=strike_price_lte,
            strike_price_lt=strike_price_lt,
            expiration_date_gte=expiration_date_gte,
            expiration_date_gt=expiration_date_gt,
            expiration_date_lte=expiration_date_lte,
            expiration_date_lt=expiration_date_lt,
            underlying_ticker_gte=underlying_ticker_gte,
            underlying_ticker_gt=underlying_ticker_gt,
            underlying_ticker_lte=underlying_ticker_lte,
            underlying_ticker_lt=underlying_ticker_lt,
        )
        if option_ticker:
            params["ticker"] = option_ticker
        if expired:
            params["expired"] = "true"
        cache_key = f"{CACHE_PREFIX}:opts:{underlying_ticker}:{contract_type}:{expiration_date}:{strike_price}:{expired}:{limit}:{sort}"
        data = await self._get_cached(
            cache_key, "v3/reference/options/contracts", params, settings.massive_cache_ttl_options
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            OptionsContractRef(
                ticker=r["ticker"],
                underlying_ticker=r.get("underlying_ticker", underlying_ticker),
                contract_type=r.get("contract_type", ""),
                exercise_style=r.get("exercise_style", ""),
                expiration_date=r.get("expiration_date", ""),
                strike_price=r.get("strike_price", 0.0),
                shares_per_contract=r.get("shares_per_contract", 100),
                primary_exchange=r.get("primary_exchange"),
                cfi=r.get("cfi"),
            )
            for r in data.get("results", [])
        ]

    async def get_full_options_chain(
        self,
        underlying_ticker: str,
        *,
        option_ticker: str | None = None,
        contract_type: str | None = None,
        expiration_date: str | None = None,
        strike_price: float | None = None,
        expired: bool = False,
        limit: int = 1000,
        as_of: str | None = None,
        order: str | None = None,
        sort: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_gt: float | None = None,
        strike_price_lte: float | None = None,
        strike_price_lt: float | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_gt: str | None = None,
        expiration_date_lte: str | None = None,
        expiration_date_lt: str | None = None,
        underlying_ticker_gte: str | None = None,
        underlying_ticker_gt: str | None = None,
        underlying_ticker_lte: str | None = None,
        underlying_ticker_lt: str | None = None,
    ) -> list[OptionsContractRef]:
        """Fetch ALL options contracts by auto-paginating through all result pages.

        Args:
            underlying_ticker: Underlying stock ticker (e.g. "AAPL").
            option_ticker: Specific options contract ticker (maps to API param "ticker").
            contract_type: Filter by type — "call" or "put".
            expiration_date: Exact expiration date (YYYY-MM-DD).
            strike_price: Exact strike price.
            expired: Whether to include expired contracts.
            limit: Results per page (max 1000).
            as_of: Point-in-time date (YYYY-MM-DD).
            order: Sort order — "asc" or "desc".
            sort: Sort field — "ticker", "expiration_date", "strike_price".
            strike_price_gte: Strike price >= range filter.
            strike_price_gt: Strike price > range filter.
            strike_price_lte: Strike price <= range filter.
            strike_price_lt: Strike price < range filter.
            expiration_date_gte: Expiration date >= range filter.
            expiration_date_gt: Expiration date > range filter.
            expiration_date_lte: Expiration date <= range filter.
            expiration_date_lt: Expiration date < range filter.
            underlying_ticker_gte: Underlying ticker >= range filter.
            underlying_ticker_gt: Underlying ticker > range filter.
            underlying_ticker_lte: Underlying ticker <= range filter.
            underlying_ticker_lt: Underlying ticker < range filter.

        Returns:
            List of all OptionsContractRef objects across all pages. Empty list if none found.

        Example::

            chain = await client.get_full_options_chain("AAPL", contract_type="call")
            chain = await client.get_full_options_chain("AAPL", expiration_date_gte="2026-06-01", strike_price_gte=150)
        """
        params = self._build_params(
            underlying_ticker=underlying_ticker,
            limit=limit,
            contract_type=contract_type,
            expiration_date=expiration_date,
            strike_price=strike_price,
            as_of=as_of,
            order=order,
            sort=sort,
            strike_price_gte=strike_price_gte,
            strike_price_gt=strike_price_gt,
            strike_price_lte=strike_price_lte,
            strike_price_lt=strike_price_lt,
            expiration_date_gte=expiration_date_gte,
            expiration_date_gt=expiration_date_gt,
            expiration_date_lte=expiration_date_lte,
            expiration_date_lt=expiration_date_lt,
            underlying_ticker_gte=underlying_ticker_gte,
            underlying_ticker_gt=underlying_ticker_gt,
            underlying_ticker_lte=underlying_ticker_lte,
            underlying_ticker_lt=underlying_ticker_lt,
        )
        if option_ticker:
            params["ticker"] = option_ticker
        if expired:
            params["expired"] = "true"
        raw = await self._get_all_pages("v3/reference/options/contracts", params)
        return [
            OptionsContractRef(
                ticker=r["ticker"],
                underlying_ticker=r.get("underlying_ticker", underlying_ticker),
                contract_type=r.get("contract_type", ""),
                exercise_style=r.get("exercise_style", ""),
                expiration_date=r.get("expiration_date", ""),
                strike_price=r.get("strike_price", 0.0),
                shares_per_contract=r.get("shares_per_contract", 100),
                primary_exchange=r.get("primary_exchange"),
                cfi=r.get("cfi"),
            )
            for r in raw
        ]
