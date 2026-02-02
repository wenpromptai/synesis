"""Tests for Telegram notification service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.notifications.telegram import (
    format_combined_signal,
    format_investment_signal,
    format_prediction_alert,
    format_sentiment_signal,
    send_telegram,
)
from synesis.processing.news import (
    Direction,
    EventType,
    ImpactLevel,
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    TickerAnalysis,
    SectorImplication,
    SourcePlatform,
    SourceType,
    UnifiedMessage,
)
from synesis.processing.sentiment import SentimentSignal, TickerSentimentSummary


class TestSendTelegram:
    """Tests for send_telegram function."""

    @pytest.mark.anyio
    async def test_no_bot_token(self) -> None:
        """Test that missing bot token returns False."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = None
            mock_settings.return_value.telegram_chat_id = "123"

            result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_no_chat_id(self) -> None:
        """Test that missing chat ID returns False."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = MagicMock()
            mock_settings.return_value.telegram_chat_id = None

            result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_successful_send(self) -> None:
        """Test successful message send."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is True

    @pytest.mark.anyio
    async def test_api_returns_error(self) -> None:
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Bad request"}
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_http_error(self) -> None:
        """Test handling HTTP error."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is False


class TestFormatInvestmentSignal:
    """Tests for format_investment_signal."""

    def test_bullish_signal(self) -> None:
        """Test formatting bullish signal."""
        result = format_investment_signal(
            source="@DeItaone",
            summary="Fed cuts rates",
            primary_thesis="Dovish pivot bullish for equities",
            tickers=["SPY", "QQQ"],
            direction="bullish",
            confidence=0.85,
        )

        assert "@DeItaone" in result
        assert "Fed cuts rates" in result
        assert "Dovish pivot" in result
        assert "$SPY" in result
        assert "$QQQ" in result
        assert "BULLISH" in result
        assert "85%" in result

    def test_bearish_signal(self) -> None:
        """Test formatting bearish signal."""
        result = format_investment_signal(
            source="@test",
            summary="Bad news",
            primary_thesis="Market decline expected",
            tickers=["SPY"],
            direction="bearish",
            confidence=0.7,
        )

        assert "BEARISH" in result

    def test_no_tickers(self) -> None:
        """Test formatting signal without tickers."""
        result = format_investment_signal(
            source="@test",
            summary="News",
            primary_thesis="Thesis",
            tickers=[],
            direction="neutral",
            confidence=0.5,
        )

        assert "N/A" in result


class TestFormatPredictionAlert:
    """Tests for format_prediction_alert."""

    def test_undervalued_market(self) -> None:
        """Test formatting undervalued market alert."""
        result = format_prediction_alert(
            market_question="Will the Fed cut rates in March?",
            verdict="undervalued",
            current_price=0.35,
            fair_price=0.55,
            edge=0.20,
            recommended_side="yes",
            reasoning="Fed guidance suggests cut is likely",
        )

        assert "Fed cut rates" in result
        assert "UNDERVALUED" in result
        assert "$0.35" in result
        assert "$0.55" in result
        assert "+20.0%" in result
        assert "YES" in result

    def test_overvalued_market(self) -> None:
        """Test formatting overvalued market alert."""
        result = format_prediction_alert(
            market_question="Will event happen?",
            verdict="overvalued",
            current_price=0.80,
            fair_price=0.60,
            edge=-0.20,
            recommended_side="no",
            reasoning="Market is overpriced",
        )

        assert "OVERVALUED" in result
        assert "NO" in result

    def test_long_reasoning_truncated(self) -> None:
        """Test that long reasoning is truncated."""
        long_reasoning = "A" * 300

        result = format_prediction_alert(
            market_question="Question?",
            verdict="fair",
            current_price=0.50,
            fair_price=0.50,
            edge=0.0,
            recommended_side="skip",
            reasoning=long_reasoning,
        )

        assert "..." in result


class TestFormatCombinedSignal:
    """Tests for format_combined_signal."""

    def _create_message(self, source: str = "@test", text: str = "Test") -> UnifiedMessage:
        """Helper to create a UnifiedMessage for testing."""
        return UnifiedMessage(
            external_id="test_123",
            source_platform=SourcePlatform.twitter,
            source_account=source,
            text=text,
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

    def _create_extraction(self, summary: str = "Test summary") -> LightClassification:
        """Helper to create a LightClassification for testing."""
        return LightClassification(
            event_type=EventType.macro,
            summary=summary,
            confidence=0.9,
            primary_entity="Test",
        )

    def test_basic_signal(self) -> None:
        """Test formatting basic combined signal."""
        message = self._create_message(source="@DeItaone", text="Apple beats earnings")
        extraction = self._create_extraction(summary="Apple beats earnings")
        analysis = SmartAnalysis(
            tickers=["AAPL"],
            sectors=["technology"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Apple earnings beat bullish for tech",
            thesis_confidence=0.8,
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "SIGNAL ALERT" in result
        assert "@DeItaone" in result
        assert "Apple" in result
        assert "HIGH" in result
        assert "BULLISH" in result
        assert "80%" in result

    def test_signal_with_ticker_analyses(self) -> None:
        """Test signal with ticker analyses."""
        message = self._create_message()
        extraction = self._create_extraction()
        analysis = SmartAnalysis(
            tickers=["AAPL"],
            sectors=[],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.bullish,
            primary_thesis="Test thesis",
            thesis_confidence=0.7,
            ticker_analyses=[
                TickerAnalysis(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    bull_thesis="Strong iPhone sales",
                    bear_thesis="China risk",
                    net_direction=Direction.bullish,
                    conviction=0.8,
                    time_horizon="days",
                ),
            ],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "$AAPL" in result
        assert "TICKERS" in result
        assert "Bull:" in result
        assert "Bear:" in result

    def test_signal_with_sector_implications(self) -> None:
        """Test signal with sector implications."""
        message = self._create_message()
        extraction = self._create_extraction(summary="Rate hike")
        analysis = SmartAnalysis(
            tickers=[],
            sectors=["financials"],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.bearish,
            primary_thesis="Rate hike negative for banks",
            thesis_confidence=0.6,
            sector_implications=[
                SectorImplication(
                    sector="financials",
                    direction=Direction.bearish,
                    reasoning="NIM compression expected",
                ),
            ],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "SECTORS" in result
        assert "financials" in result
        assert "BEARISH" in result

    def test_signal_with_market_evaluations(self) -> None:
        """Test signal with market evaluations showing edge."""
        message = self._create_message()
        extraction = self._create_extraction(summary="Fed news")
        market_eval = MarketEvaluation(
            market_id="mkt_123",
            market_question="Will Fed cut rates?",
            is_relevant=True,
            relevance_reasoning="Directly relevant",
            current_price=0.40,
            estimated_fair_price=0.60,
            edge=0.20,
            verdict="undervalued",
            confidence=0.8,
            reasoning="Strong edge",
            recommended_side="yes",
        )

        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Fed pivot",
            thesis_confidence=0.8,
            market_evaluations=[market_eval],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "POLYMARKET" in result
        assert "Fed cut" in result
        assert "+20.0%" in result
        assert "YES" in result

    def test_signal_with_historical_context(self) -> None:
        """Test signal with historical context."""
        message = self._create_message()
        extraction = self._create_extraction()
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
            primary_thesis="Test",
            thesis_confidence=0.5,
            historical_context="In 2019, similar Fed action led to a 10% rally",
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "HISTORICAL CONTEXT" in result
        assert "2019" in result


class TestFormatSentimentSignal:
    """Tests for format_sentiment_signal."""

    def _create_signal(self, **kwargs: object) -> SentimentSignal:
        """Helper to create a SentimentSignal for testing."""
        now = datetime.now(timezone.utc)
        defaults: dict[str, object] = {
            "timestamp": now,
            "period_start": now,
            "period_end": now,
            "watchlist": [],
            "ticker_sentiments": [],
            "total_posts_analyzed": 0,
            "high_quality_posts": 0,
            "spam_posts": 0,
            "overall_sentiment": "neutral",
        }
        defaults.update(kwargs)
        return SentimentSignal(**defaults)

    def test_basic_signal(self) -> None:
        """Test basic sentiment signal formatting."""
        signal = self._create_signal(
            overall_sentiment="bullish",
            narrative_summary="Market is optimistic about tech earnings",
            total_posts_analyzed=150,
            high_quality_posts=30,
            spam_posts=10,
        )
        result = format_sentiment_signal(signal)

        assert "REDDIT SENTIMENT" in result
        assert "BULLISH" in result
        assert "Market is optimistic" in result
        assert "150 analyzed" in result
        assert "30 high quality" in result
        assert "10 spam filtered" in result

    def test_ticker_sorting_by_mention_count(self) -> None:
        """Test tickers are sorted by mention count descending."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(ticker="LOW", mention_count=10, avg_sentiment=0.1),
                TickerSentimentSummary(ticker="HIGH", mention_count=100, avg_sentiment=0.2),
                TickerSentimentSummary(ticker="MED", mention_count=50, avg_sentiment=0.15),
            ],
        )
        result = format_sentiment_signal(signal)

        # HIGH should appear before MED before LOW
        high_pos = result.index("$HIGH")
        med_pos = result.index("$MED")
        low_pos = result.index("$LOW")
        assert high_pos < med_pos < low_pos

    def test_extreme_bullish_badge(self) -> None:
        """Test extreme bullish badge is displayed."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="BULL",
                    mention_count=20,
                    avg_sentiment=0.9,
                    is_extreme_bullish=True,
                ),
            ],
        )
        result = format_sentiment_signal(signal)

        assert "EXTREME BULLISH" in result

    def test_extreme_bearish_badge(self) -> None:
        """Test extreme bearish badge is displayed."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="BEAR",
                    mention_count=20,
                    avg_sentiment=-0.9,
                    is_extreme_bearish=True,
                ),
            ],
        )
        result = format_sentiment_signal(signal)

        assert "EXTREME BEARISH" in result

    def test_watchlist_added(self) -> None:
        """Test watchlist added display."""
        signal = self._create_signal(
            watchlist_added=["AAPL", "TSLA"],
        )
        result = format_sentiment_signal(signal)

        assert "Watchlist Changes" in result
        assert "Added:" in result
        assert "AAPL" in result
        assert "TSLA" in result

    def test_watchlist_removed(self) -> None:
        """Test watchlist removed display."""
        signal = self._create_signal(
            watchlist_removed=["NFLX", "META"],
        )
        result = format_sentiment_signal(signal)

        assert "Watchlist Changes" in result
        assert "Removed:" in result
        assert "NFLX" in result
        assert "META" in result

    def test_watchlist_both_added_and_removed(self) -> None:
        """Test watchlist with both added and removed tickers."""
        signal = self._create_signal(
            watchlist_added=["AAPL"],
            watchlist_removed=["NFLX"],
        )
        result = format_sentiment_signal(signal)

        assert "Watchlist Changes" in result
        assert "Added:" in result
        assert "AAPL" in result
        assert "Removed:" in result
        assert "NFLX" in result

    def test_key_themes(self) -> None:
        """Test key themes formatting."""
        signal = self._create_signal(
            key_themes=["silver crash", "earnings season", "meme stocks"],
        )
        result = format_sentiment_signal(signal)

        assert "Key Themes" in result
        assert "silver crash" in result
        assert "earnings season" in result
        assert "meme stocks" in result

    def test_subreddit_breakdown(self) -> None:
        """Test subreddit breakdown in stats section."""
        signal = self._create_signal(
            subreddits={
                "wallstreetbets": 100,
                "stocks": 50,
                "investing": 25,
            },
            total_posts_analyzed=175,
        )
        result = format_sentiment_signal(signal)

        assert "Sources:" in result
        assert "r/wallstreetbets" in result
        assert "r/stocks" in result
        assert "r/investing" in result

    def test_ticker_with_company_name(self) -> None:
        """Test ticker display with company name."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    mention_count=50,
                    avg_sentiment=0.3,
                ),
            ],
        )
        result = format_sentiment_signal(signal)

        assert "$AAPL" in result
        assert "Apple Inc." in result

    def test_ticker_with_catalysts(self) -> None:
        """Test ticker display with key catalysts."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="TSLA",
                    mention_count=80,
                    avg_sentiment=0.5,
                    key_catalysts=["earnings beat", "FSD release"],
                ),
            ],
        )
        result = format_sentiment_signal(signal)

        assert "Catalysts:" in result
        assert "earnings beat" in result
        assert "FSD release" in result

    def test_empty_ticker_sentiments(self) -> None:
        """Test signal with no ticker sentiments."""
        signal = self._create_signal(
            overall_sentiment="neutral",
            narrative_summary="Low activity period",
            total_posts_analyzed=5,
        )
        result = format_sentiment_signal(signal)

        # Should still format without error
        assert "REDDIT SENTIMENT" in result
        assert "NEUTRAL" in result
        assert "Tickers" not in result  # No tickers section

    def test_long_narrative_not_truncated(self) -> None:
        """Test that long narrative is not truncated."""
        long_narrative = "A" * 500 + "B" * 500
        signal = self._create_signal(
            narrative_summary=long_narrative,
        )
        result = format_sentiment_signal(signal)

        # Full narrative should be present (no truncation)
        assert "A" * 500 in result
        assert "B" * 500 in result

    def test_html_escaping(self) -> None:
        """Test that special characters are HTML escaped."""
        signal = self._create_signal(
            narrative_summary="<script>alert('XSS')</script> & more",
            key_themes=["theme <b>bold</b>"],
        )
        result = format_sentiment_signal(signal)

        # HTML entities should be escaped
        assert "&lt;script&gt;" in result
        assert "&amp;" in result
        assert "<script>" not in result  # Raw HTML should not appear

    def test_html_escaping_in_company_name(self) -> None:
        """Test that special characters in company names are HTML escaped."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="T",
                    company_name="AT&T Inc.",
                    mention_count=25,
                    avg_sentiment=0.2,
                ),
                TickerSentimentSummary(
                    ticker="TEST",
                    company_name="<Test> Corp",
                    mention_count=10,
                    avg_sentiment=0.1,
                ),
            ],
        )
        result = format_sentiment_signal(signal)

        # Company names should be HTML escaped
        assert "AT&amp;T Inc." in result
        assert "&lt;Test&gt; Corp" in result
        assert "AT&T Inc." not in result  # Raw & should not appear
        assert "<Test>" not in result  # Raw HTML should not appear

    def test_bearish_overall_sentiment(self) -> None:
        """Test bearish overall sentiment display."""
        signal = self._create_signal(
            overall_sentiment="bearish",
        )
        result = format_sentiment_signal(signal)

        assert "BEARISH" in result
        assert "🔴" in result

    def test_mixed_overall_sentiment(self) -> None:
        """Test mixed overall sentiment display."""
        signal = self._create_signal(
            overall_sentiment="mixed",
        )
        result = format_sentiment_signal(signal)

        assert "MIXED" in result
        assert "🟡" in result

    def test_ticker_sentiment_labels(self) -> None:
        """Test ticker sentiment labels from avg_sentiment."""
        signal = self._create_signal(
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker="BULL", mention_count=10, avg_sentiment=0.5
                ),  # > 0.1 = bullish
                TickerSentimentSummary(
                    ticker="BEAR", mention_count=10, avg_sentiment=-0.5
                ),  # < -0.1 = bearish
                TickerSentimentSummary(
                    ticker="NEUT", mention_count=10, avg_sentiment=0.05
                ),  # -0.1 to 0.1 = neutral
            ],
        )
        result = format_sentiment_signal(signal)

        # Find positions and check labels
        bull_section = result[result.index("$BULL") : result.index("$BULL") + 100]
        bear_section = result[result.index("$BEAR") : result.index("$BEAR") + 100]
        neut_section = result[result.index("$NEUT") : result.index("$NEUT") + 100]

        assert "bullish" in bull_section
        assert "bearish" in bear_section
        assert "neutral" in neut_section
