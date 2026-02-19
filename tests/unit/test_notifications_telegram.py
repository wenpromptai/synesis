"""Tests for Telegram notification service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.notifications.telegram import (
    format_sentiment_signal,
    send_telegram,
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
                    bullish_ratio=0.9,
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
                    bearish_ratio=0.9,
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


class TestSplitMessageAtSections:
    """Tests for _split_message_at_sections."""

    def test_short_message_no_split(self) -> None:
        """Messages under 4096 chars should not be split."""
        from synesis.notifications.telegram import _split_message_at_sections

        msg = "Short message"
        chunks = _split_message_at_sections(msg)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_long_message_splits_at_section_separator(self) -> None:
        """Long messages should split at section separators."""
        from synesis.notifications.telegram import (
            SECTION_SEPARATOR,
            TELEGRAM_MAX_LENGTH,
            _split_message_at_sections,
        )

        # Build a message with two sections, each ~3000 chars
        section1 = "📊 Header\n" + "A" * 2990
        section2 = f"\n{SECTION_SEPARATOR}\n📈 Second\n" + "B" * 2990
        msg = section1 + section2
        assert len(msg) > TELEGRAM_MAX_LENGTH

        chunks = _split_message_at_sections(msg)
        assert len(chunks) == 2
        # Each chunk should be under the limit
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MAX_LENGTH
        # First chunk should have part indicator
        assert "1/2" in chunks[0]
        # Second chunk should start with part indicator then separator
        assert "2/2" in chunks[1]
        assert SECTION_SEPARATOR in chunks[1]

    def test_part_indicators_on_multi_part(self) -> None:
        """Multi-part messages should have part indicators."""
        from synesis.notifications.telegram import (
            SECTION_SEPARATOR,
            _split_message_at_sections,
        )

        # Build 3 sections of ~2000 chars each
        sections = []
        for i in range(3):
            sections.append(f"📊 Section {i}\n" + f"{'X' * 1980}")
        msg = f"\n{SECTION_SEPARATOR}\n".join(sections)

        chunks = _split_message_at_sections(msg)
        assert len(chunks) >= 2
        total = len(chunks)
        # First chunk ends with part indicator
        assert f"1/{total}" in chunks[0]
        # Last chunk starts with part indicator
        assert f"{total}/{total}" in chunks[-1]

    def test_sentiment_signal_splits_cleanly(self) -> None:
        """A realistic sentiment signal with many tickers should split cleanly."""
        # Build a large signal with 30 tickers
        signal = SentimentSignal(
            timestamp=datetime.now(timezone.utc),
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            watchlist=[f"TK{i}" for i in range(30)],
            ticker_sentiments=[
                TickerSentimentSummary(
                    ticker=f"TK{i}",
                    company_name=f"Test Company {i} International Holdings Corp",
                    mention_count=100 - i,
                    avg_sentiment=0.3,
                    key_catalysts=["earnings beat", "new product launch"],
                )
                for i in range(30)
            ],
            total_posts_analyzed=500,
            high_quality_posts=100,
            spam_posts=50,
            overall_sentiment="bullish",
            narrative_summary="Markets are extremely bullish on tech earnings this quarter with multiple companies beating expectations significantly. "
            * 3,
            key_themes=["AI infrastructure", "cloud spending", "semiconductor demand", "rate cuts"],
            subreddits={"wallstreetbets": 200, "stocks": 150, "options": 100},
            watchlist_added=["AAPL", "MSFT", "NVDA"],
            watchlist_removed=["NFLX"],
        )
        msg = format_sentiment_signal(signal)

        from synesis.notifications.telegram import (
            TELEGRAM_MAX_LENGTH,
            _split_message_at_sections,
        )

        chunks = _split_message_at_sections(msg)
        # Should need splitting with 30 detailed tickers
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MAX_LENGTH, f"Chunk too long: {len(chunk)} chars"
