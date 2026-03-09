"""Tests for Events API routes."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.events import router
from synesis.core.dependencies import get_agent_state, get_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_event_row(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": 1,
        "title": "FOMC Rate Decision",
        "description": "Federal Reserve interest rate decision",
        "event_date": date(2026, 3, 19),
        "event_end_date": None,
        "category": "fed",
        "sector": None,
        "region": ["US"],
        "tickers": [],
        "source_urls": ["https://federalreserve.gov"],
        "discovered_at": datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return defaults


class _FakeRecord(dict):
    """Mimics asyncpg.Record for dict(row) conversion."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


def _make_record(**kwargs: Any) -> _FakeRecord:
    row = _sample_event_row(**kwargs)
    return _FakeRecord(row)


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.get_upcoming_events = AsyncMock(return_value=[])
    db.get_events_by_date_range = AsyncMock(return_value=[])
    db.get_event_by_id = AsyncMock(return_value=None)
    db.upsert_calendar_event = AsyncMock(return_value=1)
    return db


@pytest.fixture
def mock_agent_state() -> MagicMock:
    state = MagicMock()
    state.trigger_fns = {}
    return state


@pytest.fixture
def app(mock_db: AsyncMock, mock_agent_state: MagicMock) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/events")
    test_app.dependency_overrides[get_db] = lambda: mock_db
    test_app.dependency_overrides[get_agent_state] = lambda: mock_agent_state
    return test_app


@pytest.fixture
async def client(app: FastAPI) -> httpx.AsyncClient:  # type: ignore[misc]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c  # type: ignore[misc]


PREFIX = "/api/v1/events"


# ---------------------------------------------------------------------------
# Tests: GET /upcoming
# ---------------------------------------------------------------------------


class TestUpcomingEvents:
    @pytest.mark.asyncio
    async def test_upcoming_returns_events(
        self, mock_db: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_db.get_upcoming_events.return_value = [_make_record(), _make_record(id=2, title="CPI")]

        r = await client.get(f"{PREFIX}/upcoming")

        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["title"] == "FOMC Rate Decision"

    @pytest.mark.asyncio
    async def test_upcoming_with_filters(
        self, mock_db: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_db.get_upcoming_events.return_value = []

        r = await client.get(f"{PREFIX}/upcoming?days=14&region=US,JP&category=fed")

        assert r.status_code == 200
        mock_db.get_upcoming_events.assert_called_once_with(
            14, region=["US", "JP"], category="fed", sector=None
        )

    @pytest.mark.asyncio
    async def test_upcoming_empty(self, mock_db: AsyncMock, client: httpx.AsyncClient) -> None:
        r = await client.get(f"{PREFIX}/upcoming")

        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# Tests: GET /calendar
# ---------------------------------------------------------------------------


class TestCalendar:
    @pytest.mark.asyncio
    async def test_calendar_returns_events(
        self, mock_db: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_db.get_events_by_date_range.return_value = [_make_record()]

        r = await client.get(f"{PREFIX}/calendar?start=2026-03-01&end=2026-03-31")

        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_calendar_too_wide_range(
        self, mock_db: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        r = await client.get(f"{PREFIX}/calendar?start=2026-01-01&end=2026-12-31")

        assert r.status_code == 400
        assert "90 days" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_calendar_missing_params(
        self, mock_db: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        r = await client.get(f"{PREFIX}/calendar")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /{event_id}
# ---------------------------------------------------------------------------


class TestGetEvent:
    @pytest.mark.asyncio
    async def test_get_event_found(self, mock_db: AsyncMock, client: httpx.AsyncClient) -> None:
        mock_db.get_event_by_id.return_value = _make_record(id=42)

        r = await client.get(f"{PREFIX}/42")

        assert r.status_code == 200
        assert r.json()["id"] == 42

    @pytest.mark.asyncio
    async def test_get_event_not_found(self, mock_db: AsyncMock, client: httpx.AsyncClient) -> None:
        mock_db.get_event_by_id.return_value = None

        r = await client.get(f"{PREFIX}/999")

        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /discover
# ---------------------------------------------------------------------------


class TestTriggerDiscovery:
    @pytest.mark.asyncio
    async def test_discover_triggered(
        self, mock_agent_state: MagicMock, client: httpx.AsyncClient
    ) -> None:
        trigger = AsyncMock(return_value={"structured": 5, "curated": 3})
        mock_agent_state.trigger_fns = {"event_discover": trigger}

        r = await client.post(f"{PREFIX}/discover")

        assert r.status_code == 200
        assert r.json()["status"] == "triggered"
        # Give background task time to start
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_discover_not_configured(
        self, mock_agent_state: MagicMock, client: httpx.AsyncClient
    ) -> None:
        mock_agent_state.trigger_fns = {}

        r = await client.post(f"{PREFIX}/discover")

        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Tests: POST / (add event)
# ---------------------------------------------------------------------------


class TestAddEvent:
    @pytest.mark.asyncio
    async def test_add_event(self, mock_db: AsyncMock, client: httpx.AsyncClient) -> None:
        event_payload = {
            "title": "NVIDIA GTC 2026",
            "event_date": "2026-03-17",
            "category": "conference",
            "region": ["US"],
        }
        mock_db.upsert_calendar_event.return_value = 42

        r = await client.post(f"{PREFIX}", json=event_payload)

        assert r.status_code == 200
        assert r.json()["status"] == "created"
        assert r.json()["id"] == 42

    @pytest.mark.asyncio
    async def test_add_event_invalid(self, mock_db: AsyncMock, client: httpx.AsyncClient) -> None:
        r = await client.post(f"{PREFIX}", json={"title": "Missing fields"})
        assert r.status_code == 422
