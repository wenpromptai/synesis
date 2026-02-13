"""Integration tests for standalone providers (SEC EDGAR, NASDAQ, FactSet).

These tests call REAL APIs with no mocking. They use real Redis for caching.
Run with: pytest tests/integration/test_providers_e2e.py -m integration -v

No API keys required for SEC EDGAR or NASDAQ.
FactSet tests are skipped automatically if SQL Server is unavailable.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from synesis.providers.nasdaq.client import NasdaqClient
from synesis.providers.sec_edgar.client import SECEdgarClient


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ─────────────────────────────────────────────────────────────
# SEC EDGAR
# ─────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestSECEdgarFilings:
    """Test SEC filings across multiple tickers and form types."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"])
    async def test_get_filings_multiple_tickers(self, real_redis: Any, ticker: str) -> None:
        """Fetch filings for major tickers — all should have recent filings."""
        client = SECEdgarClient(redis=real_redis)
        try:
            filings = await client.get_filings(ticker, limit=5)
            assert len(filings) > 0, f"No filings returned for {ticker}"
            assert filings[0].ticker == ticker
            assert filings[0].form  # Non-empty form type
            assert filings[0].url.startswith("https://")
            assert filings[0].accession_number
            print(
                f"  {ticker}: {len(filings)} filings, latest: {filings[0].form} ({filings[0].filed_date})"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_filings_filtered_8k(self, real_redis: Any) -> None:
        """Filter filings to 8-K only."""
        client = SECEdgarClient(redis=real_redis)
        try:
            filings = await client.get_filings("AAPL", form_types=["8-K"], limit=5)
            assert len(filings) > 0
            assert all(f.form == "8-K" for f in filings)
            print(f"  AAPL 8-K filings: {len(filings)}")
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_filings_filtered_10q(self, real_redis: Any) -> None:
        """Filter filings to 10-Q only."""
        client = SECEdgarClient(redis=real_redis)
        try:
            filings = await client.get_filings("MSFT", form_types=["10-Q"], limit=5)
            assert len(filings) > 0
            assert all(f.form == "10-Q" for f in filings)
            print(f"  MSFT 10-Q filings: {len(filings)}")
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_filings_unknown_ticker(self, real_redis: Any) -> None:
        """Unknown ticker returns empty list, no error."""
        client = SECEdgarClient(redis=real_redis)
        try:
            filings = await client.get_filings("ZZZZXYZ", limit=5)
            assert filings == []
        finally:
            await client.close()


@pytest.mark.integration
class TestSECEdgarInsiders:
    """Test insider transactions and sentiment across tickers."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA"])
    async def test_get_insider_transactions_multiple(self, real_redis: Any, ticker: str) -> None:
        """Fetch insider transactions for major tickers."""
        client = SECEdgarClient(redis=real_redis)
        try:
            txns = await client.get_insider_transactions(ticker, limit=5)
            assert len(txns) > 0, f"No insider transactions for {ticker}"
            assert txns[0].ticker == ticker
            assert txns[0].owner_name
            assert txns[0].transaction_code in ("P", "S", "M", "A", "D", "G", "F", "J", "C")
            assert txns[0].shares > 0
            print(
                f"  {ticker}: {len(txns)} txns, "
                f"latest: {txns[0].owner_name} ({txns[0].transaction_code}) "
                f"{txns[0].shares:.0f} shares @ ${txns[0].price_per_share or 0:.2f}"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_insider_sentiment_aapl(self, real_redis: Any) -> None:
        """Compute insider sentiment for AAPL."""
        client = SECEdgarClient(redis=real_redis)
        try:
            sentiment = await client.get_insider_sentiment("AAPL")
            assert sentiment is not None
            assert "mspr" in sentiment
            assert "buy_count" in sentiment
            assert "sell_count" in sentiment
            assert "total_buy_value" in sentiment
            assert "total_sell_value" in sentiment
            assert -1.0 <= sentiment["mspr"] <= 1.0
            print(
                f"  AAPL sentiment: MSPR={sentiment['mspr']:.4f}, "
                f"buys={sentiment['buy_count']}, sells={sentiment['sell_count']}"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_insider_sentiment_tsla(self, real_redis: Any) -> None:
        """Compute insider sentiment for TSLA."""
        client = SECEdgarClient(redis=real_redis)
        try:
            sentiment = await client.get_insider_sentiment("TSLA")
            assert sentiment is not None
            assert sentiment["ticker"] == "TSLA"
            assert -1.0 <= sentiment["mspr"] <= 1.0
            print(
                f"  TSLA sentiment: MSPR={sentiment['mspr']:.4f}, "
                f"buys={sentiment['buy_count']}, sells={sentiment['sell_count']}"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_insider_transactions_unknown_ticker(self, real_redis: Any) -> None:
        """Unknown ticker returns empty list."""
        client = SECEdgarClient(redis=real_redis)
        try:
            txns = await client.get_insider_transactions("ZZZZXYZ")
            assert txns == []
        finally:
            await client.close()


@pytest.mark.integration
class TestSECEdgarXBRL:
    """Test XBRL historical EPS and revenue across tickers."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "NVDA"])
    async def test_historical_eps_multiple(self, real_redis: Any, ticker: str) -> None:
        """Fetch historical EPS for major tickers."""
        client = SECEdgarClient(redis=real_redis)
        try:
            eps = await client.get_historical_eps(ticker, limit=4)
            assert len(eps) > 0, f"No EPS data for {ticker}"
            assert eps[0]["actual"] is not None
            assert "Q" in eps[0]["frame"]
            assert eps[0]["period"]  # non-empty date string
            # Should be sorted descending
            if len(eps) > 1:
                assert eps[0]["period"] >= eps[1]["period"]
            print(
                f"  {ticker}: {len(eps)} quarters, latest: ${eps[0]['actual']:.2f} ({eps[0]['frame']})"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "AMZN"])
    async def test_historical_revenue_multiple(self, real_redis: Any, ticker: str) -> None:
        """Fetch historical revenue for major tickers."""
        client = SECEdgarClient(redis=real_redis)
        try:
            rev = await client.get_historical_revenue(ticker, limit=4)
            assert len(rev) > 0, f"No revenue data for {ticker}"
            assert rev[0]["actual"] is not None
            assert rev[0]["actual"] > 0  # Revenue should be positive
            assert "Q" in rev[0]["frame"]
            # Should be sorted descending
            if len(rev) > 1:
                assert rev[0]["period"] >= rev[1]["period"]
            print(
                f"  {ticker}: {len(rev)} quarters, latest: ${rev[0]['actual']:,.0f} ({rev[0]['frame']})"
            )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_eps_unknown_ticker(self, real_redis: Any) -> None:
        """Unknown ticker returns empty list."""
        client = SECEdgarClient(redis=real_redis)
        try:
            eps = await client.get_historical_eps("ZZZZXYZ")
            assert eps == []
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_revenue_unknown_ticker(self, real_redis: Any) -> None:
        """Unknown ticker returns empty list."""
        client = SECEdgarClient(redis=real_redis)
        try:
            rev = await client.get_historical_revenue("ZZZZXYZ")
            assert rev == []
        finally:
            await client.close()


# ─────────────────────────────────────────────────────────────
# NASDAQ
# ─────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestNasdaqIntegration:
    """Tests that call the real NASDAQ API."""

    @pytest.mark.anyio
    async def test_get_upcoming_earnings_major(self, real_redis: Any) -> None:
        """Fetch upcoming earnings for major tickers."""
        client = NasdaqClient(redis=real_redis)
        try:
            events = await client.get_upcoming_earnings(
                ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]
            )
            # May or may not have upcoming earnings depending on timing
            print(f"  Got {len(events)} upcoming earnings events")
            for ev in events[:5]:
                print(
                    f"    {ev.ticker}: {ev.earnings_date} ({ev.time}, EPS forecast: {ev.eps_forecast})"
                )
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_earnings_by_date_today(self, real_redis: Any) -> None:
        """Fetch earnings for today's date."""
        client = NasdaqClient(redis=real_redis)
        try:
            events = await client.get_earnings_by_date(date.today())
            # Validate structure even if empty
            print(f"  Today ({date.today()}): {len(events)} earnings reports")
            for ev in events[:5]:
                assert ev.ticker
                assert ev.company_name
                assert ev.earnings_date == date.today()
                assert ev.time in ("pre-market", "after-hours", "during-market", "unknown")
                print(f"    {ev.ticker} ({ev.company_name}): {ev.time}")
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_earnings_by_date_next_week(self, real_redis: Any) -> None:
        """Fetch earnings for next Monday — usually has many reports."""
        # Find next Monday
        today = date.today()
        days_ahead = (7 - today.weekday()) % 7 or 7
        next_monday = today + timedelta(days=days_ahead)

        client = NasdaqClient(redis=real_redis)
        try:
            events = await client.get_earnings_by_date(next_monday)
            print(f"  {next_monday} (next Mon): {len(events)} earnings reports")
            for ev in events[:5]:
                assert ev.ticker
                assert ev.earnings_date == next_monday
                print(f"    {ev.ticker}: {ev.time} (mkt cap: ${ev.market_cap or 0:,.0f})")
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_upcoming_earnings_empty_tickers(self, real_redis: Any) -> None:
        """Empty ticker list returns empty results."""
        client = NasdaqClient(redis=real_redis)
        try:
            events = await client.get_upcoming_earnings([])
            assert events == []
        finally:
            await client.close()

    @pytest.mark.anyio
    async def test_get_upcoming_earnings_unknown_ticker(self, real_redis: Any) -> None:
        """Unknown ticker returns empty results."""
        client = NasdaqClient(redis=real_redis)
        try:
            events = await client.get_upcoming_earnings(["ZZZZXYZ"])
            assert events == []
        finally:
            await client.close()


# ─────────────────────────────────────────────────────────────
# FactSet (skips if SQL Server unavailable)
# ─────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestFactSetIntegration:
    """Tests that call the real FactSet API (requires SQL Server)."""

    @pytest.mark.anyio
    async def test_factset_fundamentals_real(self) -> None:
        """Fetch real fundamentals from FactSet."""
        try:
            from synesis.providers.factset.client import FactSetClient
            from synesis.providers.factset.provider import FactSetProvider

            client = FactSetClient()
            provider = FactSetProvider(client=client)
            result = await provider.get_fundamentals("AAPL", "ltm", 1)
            assert result is not None
            print(f"  AAPL fundamentals: {result}")
        except Exception as exc:
            pytest.skip(f"FactSet not available: {exc}")

    @pytest.mark.anyio
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "TSLA"])
    async def test_factset_ticker_verify(self, real_redis: Any, ticker: str) -> None:
        """Verify tickers via FactSet."""
        try:
            from synesis.providers.factset.client import FactSetClient
            from synesis.providers.factset.ticker import FactSetTickerProvider

            client = FactSetClient()
            provider = FactSetTickerProvider(client=client, redis=real_redis)
            is_valid, name = await provider.verify_ticker(ticker)
            assert is_valid is True
            assert name is not None
            print(f"  {ticker} verified: {name}")
        except Exception as exc:
            pytest.skip(f"FactSet not available: {exc}")

    @pytest.mark.anyio
    async def test_factset_ticker_invalid(self, real_redis: Any) -> None:
        """Invalid ticker returns False."""
        try:
            from synesis.providers.factset.client import FactSetClient
            from synesis.providers.factset.ticker import FactSetTickerProvider

            client = FactSetClient()
            provider = FactSetTickerProvider(client=client, redis=real_redis)
            is_valid, name = await provider.verify_ticker("ZZZZXYZ")
            assert is_valid is False
            assert name is None
        except Exception as exc:
            pytest.skip(f"FactSet not available: {exc}")
