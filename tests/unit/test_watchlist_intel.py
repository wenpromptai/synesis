"""Tests for Flow 4: Watchlist Intelligence Processor."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from synesis.processing.watchlist.models import (
    CatalystAlert,
    TickerIntelligence,
    TickerReport,
    WatchlistSignal,
)
from synesis.processing.watchlist.processor import WatchlistProcessor


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for watchlist intelligence models."""

    def test_ticker_intelligence_defaults(self) -> None:
        intel = TickerIntelligence(ticker="AAPL")
        assert intel.ticker == "AAPL"
        assert intel.company_name is None
        assert intel.market_cap is None
        assert intel.recent_filings == []

    def test_catalyst_alert(self) -> None:
        alert = CatalystAlert(
            ticker="AAPL",
            alert_type="earnings_imminent",
            severity="high",
            summary="Earnings in 2 days",
        )
        assert alert.ticker == "AAPL"
        assert alert.alert_type == "earnings_imminent"
        assert alert.data == {}

    def test_ticker_report(self) -> None:
        report = TickerReport(
            ticker="AAPL",
            company_name="Apple Inc",
            fundamental_summary="Strong margins",
            overall_outlook="bullish",
            confidence=0.8,
        )
        assert report.overall_outlook == "bullish"
        assert report.confidence == 0.8

    def test_watchlist_signal(self) -> None:
        signal = WatchlistSignal(
            timestamp=datetime.now(UTC),
            tickers_analyzed=5,
            summary="Test summary",
        )
        assert signal.tickers_analyzed == 5
        assert signal.ticker_reports == []
        assert signal.alerts == []


# =============================================================================
# Processor Tests
# =============================================================================


class TestWatchlistProcessor:
    """Tests for WatchlistProcessor."""

    @pytest.fixture
    def mock_watchlist(self) -> AsyncMock:
        wl = AsyncMock()
        wl.get_all = AsyncMock(return_value=["AAPL", "TSLA", "NVDA"])
        return wl

    @pytest.fixture
    def mock_factset(self) -> AsyncMock:
        fs = AsyncMock()
        security = MagicMock()
        security.configure_mock(name="Apple Inc.")
        fs.resolve_ticker = AsyncMock(return_value=security)
        fs.get_market_cap = AsyncMock(return_value=3_000_000_000_000.0)
        fundamentals_obj = MagicMock(
            eps_diluted=6.5,
            price_to_book=45.0,
            price_to_sales=8.0,
            ev_to_ebitda=25.0,
            roe=150.0,
            net_margin=25.0,
            gross_margin=45.0,
            debt_to_equity=1.5,
        )
        fs.get_fundamentals = AsyncMock(return_value=[fundamentals_obj])
        price = MagicMock(one_day_pct=1.2, one_mth_pct=5.5)
        fs.get_price = AsyncMock(return_value=price)
        fs.close = AsyncMock()
        return fs

    @pytest.fixture
    def mock_sec_edgar(self) -> AsyncMock:
        sec = AsyncMock()
        filing = MagicMock()
        filing.form = "10-Q"
        filing.filed_date = datetime.now(UTC).date()
        sec.get_filings = AsyncMock(return_value=[filing])
        txn = MagicMock()
        txn.owner_name = "Tim Cook"
        txn.transaction_code = "S"
        txn.shares = 50000
        txn.price_per_share = 190.0
        txn.filing_date = datetime.now(UTC).date()
        sec.get_insider_transactions = AsyncMock(return_value=[txn])
        sec.get_insider_sentiment = AsyncMock(return_value={"mspr": -0.10, "change": -50000})
        sec.get_historical_eps = AsyncMock(
            return_value=[{"period": "Q4 2025", "actual": 2.10, "frame": "CY2025Q4"}]
        )
        sec.close = AsyncMock()
        return sec

    @pytest.fixture
    def mock_nasdaq(self) -> AsyncMock:
        nq = AsyncMock()
        event = MagicMock()
        event.earnings_date = (datetime.now(UTC) + timedelta(days=5)).date()
        event.time = "after-hours"
        event.eps_forecast = 2.15
        nq.get_upcoming_earnings = AsyncMock(return_value=[event])
        nq.close = AsyncMock()
        return nq

    @pytest.fixture
    def processor(
        self,
        mock_watchlist: AsyncMock,
        mock_factset: AsyncMock,
        mock_sec_edgar: AsyncMock,
        mock_nasdaq: AsyncMock,
    ) -> WatchlistProcessor:
        return WatchlistProcessor(
            watchlist=mock_watchlist,
            factset=mock_factset,
            sec_edgar=mock_sec_edgar,
            nasdaq=mock_nasdaq,
        )

    @pytest.mark.asyncio
    async def test_gather_intelligence(self, processor: WatchlistProcessor) -> None:
        """Test gathering intelligence for a single ticker."""
        intel = await processor.gather_intelligence("AAPL")

        assert intel.ticker == "AAPL"
        assert intel.company_name == "Apple Inc."
        assert intel.market_cap == 3_000_000_000_000.0
        assert intel.fundamentals is not None
        assert intel.fundamentals["ev_to_ebitda"] == 25.0
        assert intel.price_change_1d == 1.2
        assert len(intel.recent_filings) == 1
        assert len(intel.recent_insider_txns) == 1
        assert intel.insider_sentiment is not None
        assert len(intel.eps_history) == 1
        assert intel.next_earnings is not None

    @pytest.mark.asyncio
    async def test_gather_intelligence_no_providers(self, mock_watchlist: AsyncMock) -> None:
        """Test gathering intelligence with no providers configured."""
        processor = WatchlistProcessor(watchlist=mock_watchlist)
        intel = await processor.gather_intelligence("AAPL")

        assert intel.ticker == "AAPL"
        assert intel.company_name is None
        assert intel.market_cap is None
        assert intel.recent_filings == []

    @pytest.mark.asyncio
    async def test_gather_intelligence_provider_errors(self, mock_watchlist: AsyncMock) -> None:
        """Test that provider errors are handled gracefully."""
        bad_factset = AsyncMock()
        bad_factset.resolve_ticker = AsyncMock(side_effect=Exception("DB down"))
        bad_factset.get_market_cap = AsyncMock(side_effect=Exception("DB down"))
        bad_factset.get_fundamentals = AsyncMock(side_effect=Exception("DB down"))
        bad_factset.get_price = AsyncMock(side_effect=Exception("DB down"))

        processor = WatchlistProcessor(watchlist=mock_watchlist, factset=bad_factset)
        intel = await processor.gather_intelligence("AAPL")

        # Should not crash, just return empty intel
        assert intel.ticker == "AAPL"
        assert intel.company_name is None

    @pytest.mark.asyncio
    async def test_gather_intelligence_resolves_ticker_region(
        self,
        mock_watchlist: AsyncMock,
        mock_factset: AsyncMock,
    ) -> None:
        """Test that ticker_provider resolves bare ticker to ticker-region for FactSet."""
        mock_tp = AsyncMock()
        mock_tp.verify_ticker = AsyncMock(return_value=(True, "AAPL-US", "Apple Inc."))

        processor = WatchlistProcessor(
            watchlist=mock_watchlist,
            factset=mock_factset,
            ticker_provider=mock_tp,
        )
        intel = await processor.gather_intelligence("AAPL")

        # ticker_provider was called with bare ticker
        mock_tp.verify_ticker.assert_awaited_once_with("AAPL")
        # FactSet was called with resolved ticker-region
        mock_factset.resolve_ticker.assert_awaited_once_with("AAPL-US")
        mock_factset.get_market_cap.assert_awaited_once_with("AAPL-US")
        assert intel.company_name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_gather_intelligence_ticker_provider_failure_falls_back(
        self,
        mock_watchlist: AsyncMock,
        mock_factset: AsyncMock,
    ) -> None:
        """Test that FactSet falls back to bare ticker when ticker_provider fails."""
        mock_tp = AsyncMock()
        mock_tp.verify_ticker = AsyncMock(side_effect=Exception("Provider down"))

        processor = WatchlistProcessor(
            watchlist=mock_watchlist,
            factset=mock_factset,
            ticker_provider=mock_tp,
        )
        intel = await processor.gather_intelligence("AAPL")

        # FactSet should be called with bare ticker as fallback
        mock_factset.resolve_ticker.assert_awaited_once_with("AAPL")
        assert intel.company_name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_gather_intelligence_ticker_provider_invalid_falls_back(
        self,
        mock_watchlist: AsyncMock,
        mock_factset: AsyncMock,
    ) -> None:
        """Test that FactSet falls back to bare ticker when verify returns invalid."""
        mock_tp = AsyncMock()
        mock_tp.verify_ticker = AsyncMock(return_value=(False, None, None))

        processor = WatchlistProcessor(
            watchlist=mock_watchlist,
            factset=mock_factset,
            ticker_provider=mock_tp,
        )
        await processor.gather_intelligence("AAPL")

        # FactSet should be called with bare ticker
        mock_factset.resolve_ticker.assert_awaited_once_with("AAPL")


class TestDetectAlerts:
    """Tests for rule-based alert detection."""

    @pytest.fixture
    def processor(self) -> WatchlistProcessor:
        return WatchlistProcessor(watchlist=AsyncMock())

    def test_earnings_imminent(self, processor: WatchlistProcessor) -> None:
        """Test earnings imminent alert."""
        intel = TickerIntelligence(
            ticker="AAPL",
            next_earnings={
                "date": (datetime.now(UTC) + timedelta(days=5)).date().isoformat(),
                "time": "after-hours",
                "eps_forecast": 2.15,
            },
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "earnings_imminent"
        assert alerts[0].severity == "medium"

    def test_earnings_imminent_high_severity(self, processor: WatchlistProcessor) -> None:
        """Test earnings within 2 days gets high severity."""
        intel = TickerIntelligence(
            ticker="AAPL",
            next_earnings={
                "date": (datetime.now(UTC) + timedelta(days=1)).date().isoformat(),
            },
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].severity == "high"

    def test_no_earnings_alert_far_away(self, processor: WatchlistProcessor) -> None:
        """Test no alert when earnings are far away."""
        intel = TickerIntelligence(
            ticker="AAPL",
            next_earnings={
                "date": (datetime.now(UTC) + timedelta(days=30)).date().isoformat(),
            },
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 0

    def test_insider_buying(self, processor: WatchlistProcessor) -> None:
        """Test insider buying alert."""
        intel = TickerIntelligence(
            ticker="AAPL",
            insider_sentiment={"mspr": 0.60, "change": 100000},
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "insider_buying"
        assert alerts[0].severity == "high"

    def test_insider_selling(self, processor: WatchlistProcessor) -> None:
        """Test insider selling alert."""
        intel = TickerIntelligence(
            ticker="AAPL",
            insider_sentiment={"mspr": -0.30, "change": -50000},
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "insider_selling"
        assert alerts[0].severity == "medium"

    def test_new_8k_filing(self, processor: WatchlistProcessor) -> None:
        """Test 8-K filing alert."""
        intel = TickerIntelligence(
            ticker="AAPL",
            recent_filings=[
                {"form": "8-K", "filed_date": datetime.now(UTC).isoformat()},
            ],
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "new_sec_filing"

    def test_valuation_extreme(self, processor: WatchlistProcessor) -> None:
        """Test valuation extreme alert."""
        intel = TickerIntelligence(
            ticker="MEME",
            fundamentals={"ev_to_ebitda": 75.0},
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "valuation_extreme"

    def test_no_alerts_normal_data(self, processor: WatchlistProcessor) -> None:
        """Test no alerts for normal data."""
        intel = TickerIntelligence(
            ticker="AAPL",
            insider_sentiment={"mspr": 0.05, "change": 1000},
            fundamentals={"ev_to_ebitda": 20.0},
        )
        alerts = processor.detect_alerts(intel)
        assert len(alerts) == 0


class TestRunAnalysis:
    """Tests for full analysis cycle."""

    @pytest.mark.asyncio
    async def test_empty_watchlist(self) -> None:
        """Test run_analysis with empty watchlist."""
        mock_wl = AsyncMock()
        mock_wl.get_all = AsyncMock(return_value=[])
        processor = WatchlistProcessor(watchlist=mock_wl)

        signal = await processor.run_analysis()

        assert signal.tickers_analyzed == 0
        assert "empty" in signal.summary.lower()

    @pytest.mark.asyncio
    async def test_run_analysis_full(self) -> None:
        """Test full analysis cycle with mocked LLM."""
        mock_wl = AsyncMock()
        mock_wl.get_all = AsyncMock(return_value=["AAPL"])

        processor = WatchlistProcessor(watchlist=mock_wl)

        # Mock gather_intelligence
        mock_intel = TickerIntelligence(
            ticker="AAPL",
            company_name="Apple Inc.",
            market_cap=3e12,
            fundamentals={"ev_to_ebitda": 25.0},
        )
        processor.gather_intelligence = AsyncMock(return_value=mock_intel)  # type: ignore[method-assign]

        # Mock the LLM agent
        mock_synthesis = MagicMock()
        mock_synthesis.output = MagicMock(
            ticker_reports=[
                TickerReport(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    fundamental_summary="Strong margins",
                    overall_outlook="bullish",
                    confidence=0.8,
                )
            ],
            summary="AAPL looks strong.",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_synthesis)
        processor._agent = mock_agent

        signal = await processor.run_analysis()

        assert signal.tickers_analyzed == 1
        assert len(signal.ticker_reports) == 1
        assert signal.ticker_reports[0].ticker == "AAPL"
        assert signal.summary == "AAPL looks strong."

    @pytest.mark.asyncio
    async def test_run_analysis_llm_failure(self) -> None:
        """Test that LLM failure is handled gracefully."""
        mock_wl = AsyncMock()
        mock_wl.get_all = AsyncMock(return_value=["AAPL"])

        processor = WatchlistProcessor(watchlist=mock_wl)

        mock_intel = TickerIntelligence(ticker="AAPL")
        processor.gather_intelligence = AsyncMock(return_value=mock_intel)  # type: ignore[method-assign]

        # LLM fails
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM Error"))
        processor._agent = mock_agent

        signal = await processor.run_analysis()

        assert signal.tickers_analyzed == 1
        assert len(signal.ticker_reports) == 0
        assert "failed" in signal.summary.lower()
