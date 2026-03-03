"""FRED API client for Federal Reserve Economic Data.

API docs: https://fred.stlouisfed.org/docs/api/fred/
Rate limit: 120 requests/minute.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.fred.models import (
    FREDObservation,
    FREDObservations,
    FREDRelease,
    FREDReleaseDate,
    FREDSeries,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred"
CACHE_PREFIX = "synesis:fred"

FRED_RATE_LIMIT_PER_MINUTE = 120


class _FREDRateLimiter:
    """Token bucket rate limiter for FRED API (120 req/min)."""

    def __init__(self, max_calls: int = FRED_RATE_LIMIT_PER_MINUTE, period: float = 60.0):
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


_rate_limiter = _FREDRateLimiter()


class FREDClient:
    """Client for the FRED API.

    Usage:
        client = FREDClient(redis=redis_client)
        series = await client.search_series("CPI")
        obs = await client.get_observations("CPIAUCSL")
        await client.close()
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None
        settings = get_settings()
        if settings.fred_api_key is None:
            raise ValueError("FRED_API_KEY is required")
        self._api_key = settings.fred_api_key.get_secret_value()

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._http_client

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a rate-limited GET request to FRED API."""
        await _rate_limiter.acquire()
        client = self._get_http_client()
        req_params: dict[str, Any] = {"api_key": self._api_key, "file_type": "json"}
        if params:
            req_params.update(params)
        resp = await client.get(f"{FRED_API_URL}/{endpoint}", params=req_params)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            raise httpx.HTTPStatusError(
                message=f"FRED API {resp.status_code} for /{endpoint}",
                request=resp.request,
                response=resp,
            ) from None
        return orjson.loads(resp.content)  # type: ignore[no-any-return]

    async def search_series(
        self,
        query: str,
        limit: int = 20,
        filter_variable: str | None = None,
        filter_value: str | None = None,
    ) -> list[FREDSeries]:
        """Search for FRED series by keyword."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:search:{query}:{limit}:{filter_variable}:{filter_value}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return [FREDSeries.model_validate(s) for s in orjson.loads(cached)]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        params: dict[str, Any] = {"search_text": query, "limit": limit}
        if filter_variable:
            params["filter_variable"] = filter_variable
        if filter_value:
            params["filter_value"] = filter_value

        try:
            data = await self._get("series/search", params)
        except Exception as e:
            logger.warning("FRED search failed", query=query, error=str(e))
            return []

        series_list = [_parse_series(s) for s in data.get("seriess", [])]

        await self._redis.set(
            cache_key,
            orjson.dumps([s.model_dump(mode="json") for s in series_list]),
            ex=settings.fred_cache_ttl_search,
        )
        return series_list

    async def get_series_info(self, series_id: str) -> FREDSeries | None:
        """Get metadata for a single FRED series."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:series:{series_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return FREDSeries.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            data = await self._get("series", {"series_id": series_id})
        except Exception as e:
            logger.warning("FRED series info failed", series_id=series_id, error=str(e))
            return None

        items = data.get("seriess", [])
        if not items:
            return None

        series = _parse_series(items[0])
        await self._redis.set(
            cache_key,
            orjson.dumps(series.model_dump(mode="json")),
            ex=settings.fred_cache_ttl_series,
        )
        return series

    async def get_observations(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
        units: str | None = None,
        sort_order: str = "asc",
        limit: int = 100000,
    ) -> FREDObservations:
        """Get time-series observations for a FRED series."""
        settings = get_settings()
        cache_key = (
            f"{CACHE_PREFIX}:obs:{series_id}:{start}:{end}:{frequency}:{units}:{sort_order}:{limit}"
        )

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return FREDObservations.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        params: dict[str, Any] = {"series_id": series_id, "sort_order": sort_order, "limit": limit}
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        if frequency:
            params["frequency"] = frequency
        if units:
            params["units"] = units

        try:
            data = await self._get("series/observations", params)
        except Exception as e:
            logger.warning("FRED observations failed", series_id=series_id, error=str(e))
            return FREDObservations(series_id=series_id)

        obs = []
        for item in data.get("observations", []):
            value = item.get("value")
            parsed_value: float | None = None
            if value is not None and value != ".":
                try:
                    parsed_value = float(value)
                except (ValueError, TypeError):
                    pass
            obs.append(FREDObservation(date=item["date"], value=parsed_value))

        series_info = await self.get_series_info(series_id)
        result = FREDObservations(
            series_id=series_id,
            title=series_info.title if series_info else "",
            units=units or (series_info.units if series_info else ""),
            frequency=frequency or (series_info.frequency if series_info else ""),
            observations=obs,
        )

        await self._redis.set(
            cache_key,
            orjson.dumps(result.model_dump(mode="json")),
            ex=settings.fred_cache_ttl_observations,
        )
        return result

    async def get_releases(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "release_id",
        sort_order: str = "asc",
    ) -> tuple[list[FREDRelease], int]:
        """List all FRED releases. Returns (releases, total_count)."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:releases:{limit}:{offset}:{order_by}:{sort_order}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                parsed = orjson.loads(cached)
                releases = [FREDRelease.model_validate(r) for r in parsed["releases"]]
                return releases, parsed["count"]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            data = await self._get(
                "releases",
                {"limit": limit, "offset": offset, "order_by": order_by, "sort_order": sort_order},
            )
        except Exception as e:
            logger.warning("FRED releases failed", error=str(e))
            return [], 0

        total = data.get("count", 0)
        releases = [_parse_release(r) for r in data.get("releases", [])]

        await self._redis.set(
            cache_key,
            orjson.dumps(
                {"releases": [r.model_dump(mode="json") for r in releases], "count": total}
            ),
            ex=settings.fred_cache_ttl_releases,
        )
        return releases, total

    async def get_release(self, release_id: int) -> FREDRelease | None:
        """Get a single FRED release by ID."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:release:{release_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return FREDRelease.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            data = await self._get("release", {"release_id": release_id})
        except Exception as e:
            logger.warning("FRED release failed", release_id=release_id, error=str(e))
            return None

        items = data.get("releases", [])
        if not items:
            return None

        release = _parse_release(items[0])
        await self._redis.set(
            cache_key,
            orjson.dumps(release.model_dump(mode="json")),
            ex=settings.fred_cache_ttl_releases,
        )
        return release

    async def get_release_series(
        self,
        release_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FREDSeries]:
        """Get all series within a FRED release."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:release_series:{release_id}:{limit}:{offset}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return [FREDSeries.model_validate(s) for s in orjson.loads(cached)]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            data = await self._get(
                "release/series",
                {"release_id": release_id, "limit": limit, "offset": offset},
            )
        except Exception as e:
            logger.warning("FRED release series failed", release_id=release_id, error=str(e))
            return []

        series_list = [_parse_series(s) for s in data.get("seriess", [])]

        await self._redis.set(
            cache_key,
            orjson.dumps([s.model_dump(mode="json") for s in series_list]),
            ex=settings.fred_cache_ttl_releases,
        )
        return series_list

    async def get_release_dates(
        self,
        release_id: int,
        include_future: bool = True,
        limit: int = 100,
    ) -> list[FREDReleaseDate]:
        """Get scheduled dates for a FRED release."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:release_dates:{release_id}:{include_future}:{limit}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return [FREDReleaseDate.model_validate(d) for d in orjson.loads(cached)]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        release = await self.get_release(release_id)
        release_name = release.name if release else ""

        try:
            params: dict[str, Any] = {
                "release_id": release_id,
                "limit": limit,
                "sort_order": "desc",
                "include_release_dates_with_no_data": "true" if include_future else "false",
            }
            data = await self._get("release/dates", params)
        except Exception as e:
            logger.warning("FRED release dates failed", release_id=release_id, error=str(e))
            return []

        dates = [
            FREDReleaseDate(
                release_id=release_id,
                release_name=release_name,
                date=d["date"],
            )
            for d in data.get("release_dates", [])
        ]

        await self._redis.set(
            cache_key,
            orjson.dumps([d.model_dump(mode="json") for d in dates]),
            ex=settings.fred_cache_ttl_release_dates,
        )
        return dates

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FREDClient closed")


def _parse_series(raw: dict[str, Any]) -> FREDSeries:
    """Parse a FRED API series object into FREDSeries model."""
    obs_start = raw.get("observation_start")
    obs_end = raw.get("observation_end")
    return FREDSeries(
        id=raw.get("id", ""),
        title=raw.get("title", ""),
        frequency=raw.get("frequency", ""),
        units=raw.get("units", ""),
        seasonal_adjustment=raw.get("seasonal_adjustment", ""),
        last_updated=raw.get("last_updated", ""),
        popularity=raw.get("popularity", 0),
        notes=raw.get("notes", ""),
        observation_start=obs_start if obs_start and obs_start != "1776-07-04" else None,
        observation_end=obs_end if obs_end and obs_end != "9999-12-31" else None,
    )


def _parse_release(raw: dict[str, Any]) -> FREDRelease:
    """Parse a FRED API release object into FREDRelease model."""
    return FREDRelease(
        id=raw.get("id", 0),
        name=raw.get("name", ""),
        press_release=raw.get("press_release", False),
        link=raw.get("link", ""),
    )
