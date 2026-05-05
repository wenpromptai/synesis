"""FRED API endpoints (Federal Reserve Economic Data)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Path, Query
from starlette.requests import Request

from synesis.core.dependencies import FREDClientDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter()


@router.get("/search")
@limiter.limit("30/minute")
async def search_series(
    request: Request,
    client: FREDClientDep,
    q: str = Query(..., min_length=1, description="Search text"),
    limit: int = Query(20, ge=1, le=1000, description="Max results"),
    filter_variable: Literal["frequency", "units", "seasonal_adjustment"] | None = Query(
        None, description="Filter by: frequency, units, or seasonal_adjustment"
    ),
    filter_value: str | None = Query(None, description="Value for filter_variable"),
) -> dict[str, Any]:
    """Free-text search across FRED economic series.

    **Query params:**
    - `q` (str, required, min 1 char): search text — series name or keyword.
    - `limit` (int, 1–1000, default 20): max series returned.
    - `filter_variable` (optional): `frequency` | `units` | `seasonal_adjustment`.
    - `filter_value` (optional, str): paired with `filter_variable`
      (e.g. `frequency=Monthly`).

    **Returns:**
    - `query` (str): echoed.
    - `results` (list): each `{id, title, frequency, units, seasonal_adjustment,
      observation_start, observation_end, popularity}`.
    - `count` (int).

    **Example:** `curl 'http://localhost:7337/api/v1/fred/search?q=unemployment&limit=5'`
    """
    series = await client.search_series(
        query=q,
        limit=limit,
        filter_variable=filter_variable,
        filter_value=filter_value,
    )
    return {
        "query": q,
        "results": [s.model_dump(mode="json") for s in series],
        "count": len(series),
    }


@router.get("/series/{series_id}")
@limiter.limit("60/minute")
async def get_series_info(
    request: Request,
    client: FREDClientDep,
    series_id: str = Path(..., description="FRED series ID (e.g., CPIAUCSL, GDP, UNRATE)"),
) -> dict[str, Any]:
    """Metadata for a single FRED series.

    **Path params:**
    - `series_id` (str): FRED series ID, e.g. `CPIAUCSL`, `GDP`, `UNRATE`,
      `DGS10`. Case-insensitive (uppercased server-side).

    **Returns:** the series object — `id`, `title`, `frequency`, `units`,
    `seasonal_adjustment`, `observation_start`, `observation_end`,
    `last_updated`, `notes`, `popularity`.

    **Errors:**
    - `404` if the series ID is unknown.

    **Example:** `curl http://localhost:7337/api/v1/fred/series/CPIAUCSL`
    """
    series = await client.get_series_info(series_id.upper())
    if series is None:
        raise HTTPException(status_code=404, detail=f"Series '{series_id}' not found")
    return series.model_dump(mode="json")


@router.get("/series/{series_id}/observations")
@limiter.limit("30/minute")
async def get_observations(
    request: Request,
    client: FREDClientDep,
    series_id: str = Path(..., description="FRED series ID"),
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
    frequency: Literal["d", "w", "bw", "m", "q", "sa", "a"] | None = Query(
        None,
        description="Aggregation frequency: d, w, bw, m, q, sa, a",
    ),
    units: Literal["lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"] | None = Query(
        None,
        description="Data transformation: lin, chg, ch1, pch, pc1, pca, cch, cca, log",
    ),
    sort_order: Literal["asc", "desc"] = Query("asc", description="Sort: asc or desc"),
    limit: int = Query(100000, ge=1, le=100000, description="Max observations"),
) -> dict[str, Any]:
    """Time-series observations for a FRED series.

    **Path params:**
    - `series_id` (str): e.g. `UNRATE`.

    **Query params:**
    - `start` (YYYY-MM-DD, optional): inclusive start.
    - `end` (YYYY-MM-DD, optional): inclusive end.
    - `frequency` (optional): aggregation override —
      `d`=daily, `w`=weekly, `bw`=biweekly, `m`=monthly, `q`=quarterly,
      `sa`=semiannual, `a`=annual.
    - `units` (optional): transformation —
      `lin` (no change, default), `chg` (Δ), `ch1` (Δ y/y), `pch` (% Δ),
      `pc1` (% Δ y/y), `pca` (% Δ ann.), `cch` (cum. Δ), `cca`, `log`.
    - `sort_order` (`asc` | `desc`, default `asc`).
    - `limit` (int, 1–100000, default 100000).

    **Returns:** `FREDObservations` —
    - `series_id`, `title`, `units`, `frequency`,
    - `observations` (list[`FREDObservation`]): each `{date, value}` —
      `value` is null for missing data points.
    - `count` (int, computed).

    **Example (10y treasury, year-over-year change):**
    ```bash
    curl 'http://localhost:7337/api/v1/fred/series/DGS10/observations?start=2020-01-01&units=ch1'
    ```
    """
    obs = await client.get_observations(
        series_id=series_id.upper(),
        start=start,
        end=end,
        frequency=frequency,
        units=units,
        sort_order=sort_order,
        limit=limit,
    )
    return obs.model_dump(mode="json")


@router.get("/releases")
@limiter.limit("30/minute")
async def list_releases(
    request: Request,
    client: FREDClientDep,
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    order_by: Literal["release_id", "name"] = Query(
        "release_id", description="Order by: release_id, name"
    ),
    sort_order: Literal["asc", "desc"] = Query("asc", description="Sort: asc or desc"),
) -> dict[str, Any]:
    """Paginated list of FRED data releases.

    **Query params:**
    - `limit` (int, 1–1000, default 100): page size.
    - `offset` (int, ≥0, default 0): pagination offset.
    - `order_by` (`release_id` | `name`, default `release_id`).
    - `sort_order` (`asc` | `desc`, default `asc`).

    **Returns:**
    - `releases` (list): each `{id, name, press_release, link}`.
    - `count` (int): items in this page.
    - `total` (int): total releases across all pages.
    - `offset` (int).

    **Example:** `curl 'http://localhost:7337/api/v1/fred/releases?limit=20&order_by=name'`
    """
    releases, total = await client.get_releases(
        limit=limit, offset=offset, order_by=order_by, sort_order=sort_order
    )
    return {
        "releases": [r.model_dump(mode="json") for r in releases],
        "count": len(releases),
        "total": total,
        "offset": offset,
    }


@router.get("/releases/{release_id}")
@limiter.limit("60/minute")
async def get_release(
    request: Request,
    client: FREDClientDep,
    release_id: int = Path(..., description="FRED release ID"),
) -> dict[str, Any]:
    """Single FRED release by numeric ID.

    **Path params:**
    - `release_id` (int): FRED release ID (e.g. `10` for "Consumer Price Index").

    **Returns:** the release object — `id`, `name`, `press_release`, `link`.

    **Errors:**
    - `404` if release ID is unknown.

    **Example:** `curl http://localhost:7337/api/v1/fred/releases/10`
    """
    release = await client.get_release(release_id)
    if release is None:
        raise HTTPException(status_code=404, detail=f"Release {release_id} not found")
    return release.model_dump(mode="json")


@router.get("/releases/{release_id}/series")
@limiter.limit("30/minute")
async def get_release_series(
    request: Request,
    client: FREDClientDep,
    release_id: int = Path(..., description="FRED release ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> dict[str, Any]:
    """All series belonging to a FRED release.

    **Path params:**
    - `release_id` (int).

    **Query params:**
    - `limit` (int, 1–1000, default 100).
    - `offset` (int, ≥0, default 0).

    **Returns:**
    - `release_id` (int), `series` (list of series objects), `count` (int).

    **Example:** `curl 'http://localhost:7337/api/v1/fred/releases/10/series?limit=50'`
    """
    series = await client.get_release_series(release_id, limit=limit, offset=offset)
    return {
        "release_id": release_id,
        "series": [s.model_dump(mode="json") for s in series],
        "count": len(series),
    }


@router.get("/releases/{release_id}/dates")
@limiter.limit("30/minute")
async def get_release_dates(
    request: Request,
    client: FREDClientDep,
    release_id: int = Path(..., description="FRED release ID"),
    include_future: bool = Query(True, description="Include future scheduled dates"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
) -> dict[str, Any]:
    """Scheduled publication dates for a FRED release.

    Useful for forward-looking calendars — e.g. when is the next CPI print?

    **Path params:**
    - `release_id` (int).

    **Query params:**
    - `include_future` (bool, default `true`): include scheduled future dates.
    - `limit` (int, 1–1000, default 100).

    **Returns:**
    - `release_id` (int), `dates` (list of `{release_id, release_name, date}`), `count` (int).

    **Example:** `curl 'http://localhost:7337/api/v1/fred/releases/10/dates?include_future=true'`
    """
    dates = await client.get_release_dates(release_id, include_future=include_future, limit=limit)
    return {
        "release_id": release_id,
        "dates": [d.model_dump(mode="json") for d in dates],
        "count": len(dates),
    }
