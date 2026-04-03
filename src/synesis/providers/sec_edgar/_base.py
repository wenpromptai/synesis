"""SEC EDGAR base client — shared infrastructure for all mixins.

Provides rate-limited HTTP fetching, CIK mapping, and XML helpers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar.models import CompanyInfo

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_EFTS_URL = "https://efts.sec.gov/LATEST"
CACHE_PREFIX = "synesis:sec_edgar"


class _SECRateLimiter:
    """Token bucket rate limiter for SEC EDGAR (10 req/sec)."""

    def __init__(self, calls_per_second: int = 10) -> None:
        self._max_calls = calls_per_second
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            self._calls = [t for t in self._calls if t > now - 1.0]
            if len(self._calls) >= self._max_calls:
                sleep_time = 1.0 - (now - self._calls[0]) + 0.05
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = asyncio.get_event_loop().time()
                self._calls = [t for t in self._calls if t > now - 1.0]
            self._calls.append(now)


_sec_rate_limiter = _SECRateLimiter()


class SECEdgarBase:
    """Base class for SEC EDGAR client — provides HTTP, CIK mapping, lifecycle."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None
        self._cik_map: dict[str, str] | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            settings = get_settings()
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": settings.sec_edgar_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
            )
        return self._http_client

    async def _fetch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Rate-limited HTTP GET."""
        await _sec_rate_limiter.acquire()
        client = self._get_http_client()
        return await client.get(url, **kwargs)

    # ─────────────────────────────────────────────────────────────
    # CIK Mapping
    # ─────────────────────────────────────────────────────────────

    async def _load_cik_mapping(self) -> dict[str, str]:
        """Load ticker→CIK mapping from SEC, with Redis caching.

        Uses company_tickers_exchange.json which includes exchange data.
        Format: {"fields": ["cik", "name", "ticker", "exchange"], "data": [[cik, name, ticker, exchange], ...]}
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:cik_map"

        # Check Redis cache
        cached = await self._redis.get(cache_key)
        if cached:
            self._cik_map = orjson.loads(cached)
            return self._cik_map

        # Fetch from SEC (exchange version includes exchange field)
        try:
            resp = await self._fetch(f"{SEC_WWW_URL}/files/company_tickers_exchange.json")
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch SEC CIK mapping", error=str(e))
            return {}

        # Build {TICKER: "CIK_padded_to_10"}
        cik_map: dict[str, str] = {}
        for row in data.get("data", []):
            if len(row) < 3:
                continue
            cik_val, _name, ticker_val = row[0], row[1], row[2]
            ticker = str(ticker_val).upper()
            if ticker and cik_val is not None:
                cik_map[ticker] = str(cik_val).zfill(10)

        # Cache in Redis
        await self._redis.set(
            cache_key,
            orjson.dumps(cik_map),
            ex=settings.sec_edgar_cache_ttl_cik_map,
        )
        self._cik_map = cik_map
        logger.debug("Loaded SEC CIK mapping", count=len(cik_map))
        return cik_map

    async def _get_cik(self, ticker: str) -> str | None:
        """Get padded CIK for a ticker."""
        ticker = ticker.upper()
        if self._cik_map is None:
            await self._load_cik_mapping()
        if self._cik_map is None:
            return None
        return self._cik_map.get(ticker)

    # ─────────────────────────────────────────────────────────────
    # Submissions (shared helper for overflow pagination)
    # ─────────────────────────────────────────────────────────────

    async def _fetch_submissions(self, cik: str) -> dict[str, Any] | None:
        """Fetch full submissions for a CIK, with overflow pagination.

        The SEC submissions endpoint returns the most recent ~1000 filings
        in ``recent``. Older filings are split across paginated files listed
        in ``filings.files``. This helper merges them into a single dict.
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:submissions:{cik}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return orjson.loads(cached)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            resp = await self._fetch(f"{SEC_BASE_URL}/submissions/CIK{cik}.json")
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch SEC submissions", cik=cik, error=str(e))
            return None

        # Merge overflow pages into `recent`
        recent = data.get("filings", {}).get("recent", {})
        overflow_files = data.get("filings", {}).get("files", [])
        for overflow in overflow_files:
            name = overflow.get("name")
            if not name:
                continue
            try:
                overflow_resp = await self._fetch(f"{SEC_BASE_URL}/submissions/{name}")
                overflow_resp.raise_for_status()
                overflow_data: dict[str, Any] = orjson.loads(overflow_resp.content)
                for key in recent:
                    if key in overflow_data and isinstance(recent[key], list):
                        recent[key].extend(overflow_data[key])
            except Exception as e:
                logger.warning("Failed to fetch overflow submissions", name=name, error=str(e))

        # Cache merged submissions
        await self._redis.set(
            cache_key,
            orjson.dumps(data),
            ex=settings.sec_edgar_cache_ttl_submissions,
        )
        return data

    # ─────────────────────────────────────────────────────────────
    # Company Info (extracted from submissions metadata)
    # ─────────────────────────────────────────────────────────────

    async def get_company_info(self, ticker: str) -> CompanyInfo | None:
        """Get company metadata from SEC submissions.

        Extracts SIC code, fiscal year end, entity type, exchanges, and other
        metadata from the submissions JSON (already fetched and cached).
        """

        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            return None

        data = await self._fetch_submissions(cik)
        if not data:
            return None

        return CompanyInfo(
            ticker=ticker,
            cik=cik,
            name=data.get("name", ""),
            entity_type=data.get("entityType", ""),
            sic=data.get("sic", ""),
            sic_description=data.get("sicDescription", ""),
            category=data.get("category", ""),
            fiscal_year_end=data.get("fiscalYearEnd", ""),
            state_of_incorporation=data.get("stateOfIncorporation", ""),
            exchanges=data.get("exchanges", []),
            tickers=data.get("tickers", []),
            ein=data.get("ein", ""),
            former_names=data.get("formerNames", []),
            phone=data.get("phone", ""),
            website=data.get("website", ""),
        )

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("SECEdgarClient closed")


# ─────────────────────────────────────────────────────────────
# XML Helpers
# ─────────────────────────────────────────────────────────────


def _el_text(parent: Element, tag: str) -> str:
    """Get text from a child element, or empty string."""
    el = parent.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def _build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Build a full SEC Archives URL for a filing document."""
    acc = accession_number.replace("-", "")
    if not primary_document:
        return ""
    return f"{SEC_WWW_URL}/Archives/edgar/data/{cik}/{acc}/{primary_document}"


def _parse_acceptance_datetime(acc_dt_str: str, ticker: str = "") -> datetime:
    """Parse SEC acceptance datetime string, returning datetime.min on failure."""
    try:
        return datetime.fromisoformat(acc_dt_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning(
            "Failed to parse acceptance datetime, using sentinel",
            ticker=ticker,
            acc_dt_str=acc_dt_str,
        )
        return datetime.min
