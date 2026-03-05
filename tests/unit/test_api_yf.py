"""Tests for yfinance API endpoints — options snapshot route."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.yf import router
from synesis.core.dependencies import get_yfinance_client
from synesis.providers.yfinance.models import (
    OptionsContract,
    OptionsGreeks,
    OptionsSnapshot,
)


def _make_contract(strike: float, option_type: str = "call") -> OptionsContract:
    return OptionsContract(
        contract_symbol=f"AAPL260320{'C' if option_type == 'call' else 'P'}{int(strike * 1000):08d}",
        strike=strike,
        last_price=5.50,
        bid=5.30,
        ask=5.70,
        volume=1200,
        open_interest=8500,
        implied_volatility=0.28,
        in_the_money=False,
        greeks=OptionsGreeks(
            delta=0.48,
            gamma=0.02,
            theta=-0.15,
            vega=0.35,
            rho=0.05,
            implied_volatility=0.28,
        ),
    )


def _make_snapshot(**kwargs: object) -> OptionsSnapshot:
    defaults: dict = {
        "ticker": "AAPL",
        "spot": 264.72,
        "realized_vol_30d": 0.3245,
        "expiration": "2026-03-20",
        "days_to_expiry": 15,
        "calls": [_make_contract(265.0, "call"), _make_contract(270.0, "call")],
        "puts": [_make_contract(265.0, "put"), _make_contract(260.0, "put")],
    }
    defaults.update(kwargs)
    return OptionsSnapshot(**defaults)


@pytest.fixture()
def mock_yf_client():
    client = AsyncMock()
    client.get_options_snapshot = AsyncMock(return_value=_make_snapshot())
    return client


@pytest.fixture()
def app(mock_yf_client):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/yf")
    test_app.dependency_overrides[get_yfinance_client] = lambda: mock_yf_client
    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


PREFIX = "/yf"


class TestOptionsSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_returns_200(self, client: httpx.AsyncClient) -> None:
        r = await client.get(f"{PREFIX}/options/AAPL/snapshot")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_snapshot_response_shape(self, client: httpx.AsyncClient) -> None:
        r = await client.get(f"{PREFIX}/options/AAPL/snapshot")
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["spot"] == 264.72
        assert data["realized_vol_30d"] == 0.3245
        assert data["expiration"] == "2026-03-20"
        assert data["days_to_expiry"] == 15
        assert len(data["calls"]) == 2
        assert len(data["puts"]) == 2

    @pytest.mark.asyncio
    async def test_snapshot_calls_have_greeks(self, client: httpx.AsyncClient) -> None:
        r = await client.get(f"{PREFIX}/options/AAPL/snapshot")
        data = r.json()
        call = data["calls"][0]
        assert call["greeks"] is not None
        assert "delta" in call["greeks"]

    @pytest.mark.asyncio
    async def test_snapshot_greeks_param_forwarded(
        self, mock_yf_client: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        await client.get(f"{PREFIX}/options/AAPL/snapshot?greeks=false")
        mock_yf_client.get_options_snapshot.assert_called_once_with("AAPL", greeks=False)

    @pytest.mark.asyncio
    async def test_snapshot_greeks_default_true(
        self, mock_yf_client: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        await client.get(f"{PREFIX}/options/AAPL/snapshot")
        mock_yf_client.get_options_snapshot.assert_called_once_with("AAPL", greeks=True)

    @pytest.mark.asyncio
    async def test_snapshot_empty_chain(
        self, mock_yf_client: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_yf_client.get_options_snapshot.return_value = _make_snapshot(
            expiration="", days_to_expiry=0, calls=[], puts=[]
        )
        r = await client.get(f"{PREFIX}/options/AAPL/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert data["calls"] == []
        assert data["puts"] == []

    @pytest.mark.asyncio
    async def test_snapshot_null_realized_vol(
        self, mock_yf_client: AsyncMock, client: httpx.AsyncClient
    ) -> None:
        mock_yf_client.get_options_snapshot.return_value = _make_snapshot(
            realized_vol_30d=None
        )
        r = await client.get(f"{PREFIX}/options/AAPL/snapshot")
        data = r.json()
        assert data["realized_vol_30d"] is None
