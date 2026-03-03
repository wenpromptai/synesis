"""Tests for FRED provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest

from synesis.providers.fred.client import FREDClient, _parse_release, _parse_series

# ---------------------------------------------------------------------------
# Sample API Responses
# ---------------------------------------------------------------------------

SAMPLE_SEARCH_RESPONSE = {
    "seriess": [
        {
            "id": "CPIAUCSL",
            "title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
            "frequency": "Monthly",
            "units": "Index 1982-1984=100",
            "seasonal_adjustment": "Seasonally Adjusted",
            "last_updated": "2026-02-12 07:41:03-06",
            "popularity": 95,
            "notes": "The Consumer Price Index...",
            "observation_start": "1947-01-01",
            "observation_end": "2026-01-01",
        },
        {
            "id": "CPILFESL",
            "title": "Consumer Price Index: All Items Less Food and Energy",
            "frequency": "Monthly",
            "units": "Index 1982-1984=100",
            "seasonal_adjustment": "Seasonally Adjusted",
            "last_updated": "2026-02-12 07:41:04-06",
            "popularity": 82,
            "notes": "",
            "observation_start": "1957-01-01",
            "observation_end": "2026-01-01",
        },
    ]
}

SAMPLE_SERIES_RESPONSE = {
    "seriess": [
        {
            "id": "GDP",
            "title": "Gross Domestic Product",
            "frequency": "Quarterly",
            "units": "Billions of Dollars",
            "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
            "last_updated": "2026-01-30 07:46:02-06",
            "popularity": 93,
            "notes": "",
            "observation_start": "1947-01-01",
            "observation_end": "2025-10-01",
        }
    ]
}

SAMPLE_OBSERVATIONS_RESPONSE = {
    "observations": [
        {"date": "2025-10-01", "value": "29719.921"},
        {"date": "2025-07-01", "value": "29374.391"},
        {"date": "2025-04-01", "value": "."},
    ]
}

SAMPLE_RELEASES_RESPONSE = {
    "count": 300,
    "releases": [
        {
            "id": 10,
            "name": "Consumer Price Index",
            "press_release": True,
            "link": "http://www.bls.gov/cpi/",
        },
        {
            "id": 11,
            "name": "Employment Cost Index",
            "press_release": True,
            "link": "http://www.bls.gov/eci/",
        },
    ],
}

SAMPLE_RELEASE_DATES_RESPONSE = {
    "release_dates": [
        {"date": "2026-03-12"},
        {"date": "2026-02-12"},
        {"date": "2026-01-15"},
    ]
}

SAMPLE_RELEASE_SERIES_RESPONSE = {
    "seriess": [
        {
            "id": "CPIAUCSL",
            "title": "Consumer Price Index for All Urban Consumers",
            "frequency": "Monthly",
            "units": "Index 1982-1984=100",
            "seasonal_adjustment": "Seasonally Adjusted",
            "last_updated": "2026-02-12",
            "popularity": 95,
            "notes": "",
            "observation_start": "1947-01-01",
            "observation_end": "2026-01-01",
        },
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture()
def client(mock_redis):
    with patch("synesis.providers.fred.client.get_settings") as mock_settings:
        settings = MagicMock()
        settings.fred_api_key.get_secret_value.return_value = "test_api_key"
        settings.fred_cache_ttl_search = 3600
        settings.fred_cache_ttl_series = 43200
        settings.fred_cache_ttl_observations = 21600
        settings.fred_cache_ttl_releases = 43200
        settings.fred_cache_ttl_release_dates = 21600
        mock_settings.return_value = settings
        return FREDClient(redis=mock_redis)


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------


class TestParsingHelpers:
    def test_parse_series(self):
        series = _parse_series(SAMPLE_SEARCH_RESPONSE["seriess"][0])
        assert series.id == "CPIAUCSL"
        assert "Consumer Price Index" in series.title
        assert series.frequency == "Monthly"
        assert series.popularity == 95

    def test_parse_series_sentinel_dates(self):
        """FRED uses 1776-07-04 and 9999-12-31 as sentinel dates."""
        series = _parse_series(
            {"id": "TEST", "observation_start": "1776-07-04", "observation_end": "9999-12-31"}
        )
        assert series.observation_start is None
        assert series.observation_end is None

    def test_parse_release(self):
        release = _parse_release(SAMPLE_RELEASES_RESPONSE["releases"][0])
        assert release.id == 10
        assert release.name == "Consumer Price Index"
        assert release.press_release is True


# ---------------------------------------------------------------------------
# search_series
# ---------------------------------------------------------------------------


class TestSearchSeries:
    async def test_search(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_SEARCH_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            results = await client.search_series("CPI")

        assert len(results) == 2
        assert results[0].id == "CPIAUCSL"
        assert results[1].id == "CPILFESL"

    async def test_search_from_cache(self, client: FREDClient, mock_redis):
        cached = [{"id": "CPIAUCSL", "title": "CPI"}]
        mock_redis.get.return_value = orjson.dumps(cached)
        results = await client.search_series("CPI")
        assert len(results) == 1
        assert results[0].id == "CPIAUCSL"

    async def test_search_http_error(self, client: FREDClient, mock_redis):
        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            results = await client.search_series("CPI")
        assert results == []


# ---------------------------------------------------------------------------
# get_series_info
# ---------------------------------------------------------------------------


class TestGetSeriesInfo:
    async def test_get_series(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_SERIES_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            series = await client.get_series_info("GDP")

        assert series is not None
        assert series.id == "GDP"
        assert series.title == "Gross Domestic Product"

    async def test_get_series_not_found(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps({"seriess": []})
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            series = await client.get_series_info("NONEXISTENT")

        assert series is None


# ---------------------------------------------------------------------------
# get_observations
# ---------------------------------------------------------------------------


class TestGetObservations:
    async def test_get_observations(self, client: FREDClient, mock_redis):
        series_resp = MagicMock()
        series_resp.content = orjson.dumps(SAMPLE_SERIES_RESPONSE)
        series_resp.raise_for_status = MagicMock()

        obs_resp = MagicMock()
        obs_resp.content = orjson.dumps(SAMPLE_OBSERVATIONS_RESPONSE)
        obs_resp.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=[obs_resp, series_resp])
            obs = await client.get_observations("GDP", limit=3)

        assert obs.series_id == "GDP"
        assert obs.count == 3
        assert obs.observations[0].value == 29719.921
        assert obs.observations[1].value == 29374.391
        assert obs.observations[2].value is None  # "." -> None

    async def test_get_observations_missing_dot(self, client: FREDClient, mock_redis):
        """FRED uses '.' for missing values — should parse as None."""
        obs_resp = MagicMock()
        obs_resp.content = orjson.dumps({"observations": [{"date": "2025-01-01", "value": "."}]})
        obs_resp.raise_for_status = MagicMock()

        series_resp = MagicMock()
        series_resp.content = orjson.dumps({"seriess": []})
        series_resp.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=[obs_resp, series_resp])
            obs = await client.get_observations("TEST")

        assert obs.observations[0].value is None


# ---------------------------------------------------------------------------
# get_releases
# ---------------------------------------------------------------------------


class TestGetReleases:
    async def test_get_releases(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_RELEASES_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            releases, total = await client.get_releases()

        assert len(releases) == 2
        assert total == 300
        assert releases[0].name == "Consumer Price Index"
        assert releases[0].press_release is True

    async def test_get_releases_http_error(self, client: FREDClient, mock_redis):
        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            releases, total = await client.get_releases()
        assert releases == []
        assert total == 0


# ---------------------------------------------------------------------------
# get_release
# ---------------------------------------------------------------------------


class TestGetRelease:
    async def test_get_release(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(
            {
                "releases": [
                    {
                        "id": 10,
                        "name": "Consumer Price Index",
                        "press_release": True,
                        "link": "http://www.bls.gov/cpi/",
                    }
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            release = await client.get_release(10)

        assert release is not None
        assert release.id == 10
        assert release.name == "Consumer Price Index"

    async def test_get_release_not_found(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps({"releases": []})
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            release = await client.get_release(99999)

        assert release is None


# ---------------------------------------------------------------------------
# get_release_series
# ---------------------------------------------------------------------------


class TestGetReleaseSeries:
    async def test_get_release_series(self, client: FREDClient, mock_redis):
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_RELEASE_SERIES_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            series = await client.get_release_series(10)

        assert len(series) == 1
        assert series[0].id == "CPIAUCSL"


# ---------------------------------------------------------------------------
# get_release_dates
# ---------------------------------------------------------------------------


class TestGetReleaseDates:
    async def test_get_release_dates(self, client: FREDClient, mock_redis):
        release_resp = MagicMock()
        release_resp.content = orjson.dumps(
            {
                "releases": [
                    {
                        "id": 10,
                        "name": "Consumer Price Index",
                        "press_release": True,
                        "link": "",
                    }
                ]
            }
        )
        release_resp.raise_for_status = MagicMock()

        dates_resp = MagicMock()
        dates_resp.content = orjson.dumps(SAMPLE_RELEASE_DATES_RESPONSE)
        dates_resp.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=[release_resp, dates_resp])
            dates = await client.get_release_dates(10)

        assert len(dates) == 3
        assert dates[0].release_name == "Consumer Price Index"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_close(self, client: FREDClient):
        mock_http = AsyncMock()
        client._http_client = mock_http
        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._http_client is None

    async def test_close_no_client(self, client: FREDClient):
        await client.close()  # Should not raise
