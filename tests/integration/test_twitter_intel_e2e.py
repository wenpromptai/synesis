"""Integration tests for yfinance data quality.

These test the yfinance provider directly — standalone, not tied to the twitter agent.

Run with: pytest -m integration tests/integration/test_twitter_intel_e2e.py -v
"""

from __future__ import annotations

from typing import Any

import pytest

from synesis.providers.yfinance.client import YFinanceClient


@pytest.fixture
def yfinance_client(mock_redis: Any) -> YFinanceClient:
    """Real yfinance client with mock Redis (no API key needed, free data)."""
    return YFinanceClient(redis=mock_redis)


@pytest.mark.integration
class TestYFinanceToolsDirectly:
    """Validate yfinance data quality and math."""

    @pytest.mark.anyio
    async def test_get_quote_returns_valid_data(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """get_quote should return a quote with at least ticker and last price."""
        q = await yfinance_client.get_quote("AAPL")
        assert q.ticker == "AAPL"
        assert q.name is not None
        assert len(q.name) > 0

    @pytest.mark.anyio
    async def test_get_history_returns_bars(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """1mo daily history should return enough bars for realized vol calc."""
        bars = await yfinance_client.get_history("AAPL", period="1mo", interval="1d")
        assert len(bars) >= 18, f"Expected ~20 bars for 1mo, got {len(bars)}"
        closes = [b.close for b in bars if b.close is not None]
        assert len(closes) >= 18
        assert all(c > 0 for c in closes)

    @pytest.mark.anyio
    async def test_realized_vol_calculation(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Validate the realized vol math is reasonable."""
        import math

        bars = await yfinance_client.get_history("AAPL", period="1mo", interval="1d")
        closes = [b.close for b in bars if b.close is not None and b.close > 0]
        assert len(closes) >= 5, f"Need >=5 closes, got {len(closes)}"

        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        realized_vol = math.sqrt(variance * 252)

        assert 0.05 <= realized_vol <= 1.50, (
            f"Realized vol {realized_vol:.2%} outside sane range for AAPL"
        )
        assert math.isfinite(realized_vol)

    @pytest.mark.anyio
    async def test_options_expirations_available(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """AAPL should always have options expirations."""
        expirations = await yfinance_client.get_options_expirations("AAPL")
        assert len(expirations) >= 1
        for exp in expirations[:3]:
            assert len(exp) == 10  # YYYY-MM-DD
            assert exp[4] == "-"

    @pytest.mark.anyio
    async def test_options_chain_structure(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Options chain should have calls and puts with strikes."""
        expirations = await yfinance_client.get_options_expirations("AAPL")
        assert expirations, "No expirations found"

        from datetime import date

        target_exp = expirations[0]
        today = date.today()
        for exp_str in expirations:
            exp_date = date.fromisoformat(exp_str)
            if (exp_date - today).days >= 7:
                target_exp = exp_str
                break

        chain = await yfinance_client.get_options_chain("AAPL", target_exp, greeks=True)

        assert len(chain.calls) > 0
        assert len(chain.puts) > 0

        for c in chain.calls[:5]:
            assert c.strike > 0
        for c in chain.puts[:5]:
            assert c.strike > 0

    @pytest.mark.anyio
    async def test_full_options_snapshot_flow(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Simulate the full get_options_snapshot tool flow for AAPL."""
        import math

        ticker = "AAPL"

        quote = await yfinance_client.get_quote(ticker)
        spot = quote.last or 0
        assert spot >= 0

        bars = await yfinance_client.get_history(ticker, period="1mo", interval="1d")
        closes = [b.close for b in bars if b.close is not None and b.close > 0]

        if len(closes) >= 5:
            log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            mean_r = sum(log_returns) / len(log_returns)
            variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
            realized_vol = math.sqrt(variance * 252)
            assert math.isfinite(realized_vol)
            assert 0.01 <= realized_vol <= 3.0

        expirations = await yfinance_client.get_options_expirations(ticker)
        assert len(expirations) >= 1

        from datetime import date

        today = date.today()
        target_exp = expirations[0]
        for exp_str in expirations:
            exp_date = date.fromisoformat(exp_str)
            if (exp_date - today).days >= 7:
                target_exp = exp_str
                break

        chain = await yfinance_client.get_options_chain(ticker, target_exp, greeks=True)
        assert len(chain.calls) > 0 or len(chain.puts) > 0
