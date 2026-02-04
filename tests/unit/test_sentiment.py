"""Unit tests for sentiment analyzer."""

import pytest

from synesis.processing.sentiment import SentimentAnalyzer, SentimentResult


@pytest.fixture
def analyzer() -> SentimentAnalyzer:
    """Create analyzer instance."""
    return SentimentAnalyzer()


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating valid result."""
        result = SentimentResult(
            compound=0.5,
            positive=0.6,
            negative=0.1,
            neutral=0.3,
            confidence=0.8,
        )
        assert result.compound == 0.5
        assert result.is_bullish

    def test_invalid_compound_raises(self) -> None:
        """Test that invalid compound raises ValueError."""
        with pytest.raises(ValueError, match="compound must be"):
            SentimentResult(compound=1.5, positive=0.5, negative=0.3, neutral=0.2)

    def test_is_bullish(self) -> None:
        """Test bullish detection."""
        result = SentimentResult(compound=0.3, positive=0.5, negative=0.2, neutral=0.3)
        assert result.is_bullish
        assert not result.is_bearish
        assert not result.is_neutral

    def test_is_bearish(self) -> None:
        """Test bearish detection."""
        result = SentimentResult(compound=-0.3, positive=0.2, negative=0.5, neutral=0.3)
        assert result.is_bearish
        assert not result.is_bullish

    def test_is_neutral(self) -> None:
        """Test neutral detection."""
        result = SentimentResult(compound=0.02, positive=0.3, negative=0.3, neutral=0.4)
        assert result.is_neutral

    def test_strength(self) -> None:
        """Test sentiment strength categorization."""
        strong = SentimentResult(compound=0.7, positive=0.8, negative=0.1, neutral=0.1)
        assert strong.strength == "strong"

        moderate = SentimentResult(compound=0.4, positive=0.5, negative=0.2, neutral=0.3)
        assert moderate.strength == "moderate"

        weak = SentimentResult(compound=0.1, positive=0.4, negative=0.3, neutral=0.3)
        assert weak.strength == "weak"


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    @pytest.mark.asyncio
    async def test_empty_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test empty text returns neutral."""
        result = await analyzer.analyze("")
        assert result.compound == 0.0
        assert result.neutral == 1.0

    @pytest.mark.asyncio
    async def test_bullish_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test bullish trading text."""
        result = await analyzer.analyze("TSLA to the moon! 🚀🚀🚀")
        assert result.compound > 0.8, f"Expected > 0.8, got {result.compound}"
        assert result.is_bullish

    @pytest.mark.asyncio
    async def test_bearish_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test bearish text."""
        result = await analyzer.analyze("This stock is absolute garbage")
        assert result.compound < -0.5, f"Expected < -0.5, got {result.compound}"
        assert result.is_bearish

    @pytest.mark.asyncio
    async def test_neutral_mixed_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test mixed sentiment text."""
        result = await analyzer.analyze("Not bad, but not great either")
        assert -0.3 < result.compound < 0.3, f"Expected near 0, got {result.compound}"

    @pytest.mark.asyncio
    async def test_breaking_news_neutral(self, analyzer: SentimentAnalyzer) -> None:
        """Test breaking news extracts entities."""
        result = await analyzer.analyze("BREAKING: Fed cuts rates")
        # Fed news is slightly positive due to 'cut' being dovish
        assert result.compound > -0.2

    @pytest.mark.asyncio
    async def test_guh_bearish(self, analyzer: SentimentAnalyzer) -> None:
        """Test WSB 'GUH' is very bearish."""
        result = await analyzer.analyze("GUH my calls are worthless")
        assert result.compound < -0.7, f"Expected < -0.7, got {result.compound}"

    @pytest.mark.asyncio
    async def test_ticker_extraction(self, analyzer: SentimentAnalyzer) -> None:
        """Test ticker extraction."""
        result = await analyzer.analyze("$TSLA and NVDA are mooning!")
        assert "TSLA" in result.tickers_mentioned
        assert "NVDA" in result.tickers_mentioned

    @pytest.mark.asyncio
    async def test_ticker_blacklist(self, analyzer: SentimentAnalyzer) -> None:
        """Test blacklisted words aren't extracted as tickers."""
        result = await analyzer.analyze("YOLO into CEO IPO")
        assert "YOLO" not in result.tickers_mentioned
        assert "CEO" not in result.tickers_mentioned
        assert "IPO" not in result.tickers_mentioned

    @pytest.mark.asyncio
    async def test_emoji_sentiment(self, analyzer: SentimentAnalyzer) -> None:
        """Test emoji contribute to sentiment."""
        rocket = await analyzer.analyze("🚀🚀🚀")
        assert rocket.compound > 0.5

        skull = await analyzer.analyze("💀💀💀")
        assert skull.compound < -0.5

    @pytest.mark.asyncio
    async def test_caps_emphasis(self, analyzer: SentimentAnalyzer) -> None:
        """Test ALL CAPS increases intensity."""
        normal = await analyzer.analyze("This is amazing")
        caps = await analyzer.analyze("This is AMAZING")
        assert caps.compound > normal.compound

    @pytest.mark.asyncio
    async def test_negation(self, analyzer: SentimentAnalyzer) -> None:
        """Test negation flips sentiment."""
        positive = await analyzer.analyze("This is good")
        negated = await analyzer.analyze("This is not good")
        assert positive.compound > 0
        assert negated.compound < positive.compound

    @pytest.mark.asyncio
    async def test_booster(self, analyzer: SentimentAnalyzer) -> None:
        """Test boosters increase intensity."""
        normal = await analyzer.analyze("This is good")
        boosted = await analyzer.analyze("This is very good")
        assert boosted.compound > normal.compound

    @pytest.mark.asyncio
    async def test_slang_phrases(self, analyzer: SentimentAnalyzer) -> None:
        """Test slang phrase detection."""
        result = await analyzer.analyze("diamond hands baby, we're going to the moon")
        assert result.compound > 0.5

    @pytest.mark.asyncio
    async def test_but_clause(self, analyzer: SentimentAnalyzer) -> None:
        """Test but-clause weighting."""
        # Post-but content weighted higher
        result = await analyzer.analyze("The product is good but the stock is terrible")
        assert result.compound < 0  # Negative wins due to but-clause

    @pytest.mark.asyncio
    async def test_confidence(self, analyzer: SentimentAnalyzer) -> None:
        """Test confidence reflects lexicon coverage."""
        high_coverage = await analyzer.analyze("amazing wonderful fantastic great")
        low_coverage = await analyzer.analyze("xyzzy foobar baz")
        assert high_coverage.confidence > low_coverage.confidence

    @pytest.mark.asyncio
    async def test_finance_terms(self, analyzer: SentimentAnalyzer) -> None:
        """Test finance-specific vocabulary."""
        bullish = await analyzer.analyze("The stock is breaking out, squeeze incoming")
        assert bullish.compound > 0.5

        bearish = await analyzer.analyze("Total rugpull, everyone got rekt")
        assert bearish.compound < -0.7


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_only_punctuation(self, analyzer: SentimentAnalyzer) -> None:
        """Test text with only punctuation."""
        result = await analyzer.analyze("!!!")
        # Just punctuation adds small positive modifier
        assert result.compound >= 0

    @pytest.mark.asyncio
    async def test_unicode_handling(self, analyzer: SentimentAnalyzer) -> None:
        """Test unicode text doesn't crash."""
        result = await analyzer.analyze("好的 très bien molto bene")
        assert isinstance(result, SentimentResult)

    @pytest.mark.asyncio
    async def test_very_long_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test long text doesn't timeout."""
        long_text = "amazing " * 1000
        result = await analyzer.analyze(long_text)
        assert result.compound > 0.5

    @pytest.mark.asyncio
    async def test_mixed_case_tickers(self, analyzer: SentimentAnalyzer) -> None:
        """Test ticker extraction handles case."""
        result = await analyzer.analyze("$tsla AAPL")
        # Cashtags work lowercase, bare tickers need uppercase
        assert "TSLA" in result.tickers_mentioned or "AAPL" in result.tickers_mentioned


class TestMentionCountFromRealData:
    """Test that mention_count in generate_signal() uses real Gate 1 data, not LLM output."""

    @pytest.mark.asyncio
    async def test_mention_count_equals_raw_ticker_scores(self) -> None:
        """mention_count must equal len(scores) from _raw_tickers, not an LLM field."""
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock, Mock, patch

        from synesis.ingestion.reddit import RedditPost
        from synesis.processing.sentiment.models import (
            SentimentRefinement,
            ValidatedTicker,
        )
        from synesis.processing.sentiment.processor import SentimentProcessor

        mock_redis = AsyncMock()
        # WatchlistManager.get_all() uses redis.smembers → returns set of tickers
        mock_redis.smembers = AsyncMock(return_value=set())
        # WatchlistManager.cleanup_expired() calls redis.register_script() (sync)
        # which returns a callable script object that is then awaited
        mock_redis.register_script = Mock(return_value=AsyncMock(return_value=0))

        processor = SentimentProcessor(
            settings=AsyncMock(),
            redis=mock_redis,
        )

        # Simulate Gate 1 output: AAPL mentioned in 3 posts, TSLA in 7 posts
        processor._raw_tickers = {
            "AAPL": [0.5, -0.2, 0.8],  # 3 posts
            "TSLA": [0.3, 0.1, -0.4, 0.6, 0.2, -0.1, 0.9],  # 7 posts
        }

        # Simulate buffered posts (count doesn't matter for this test,
        # but buffer must be non-empty so process_posts is called)
        now = datetime.now(timezone.utc)
        fake_post = RedditPost(
            post_id="p1",
            subreddit="wallstreetbets",
            author="user",
            title="test",
            content="test content",
            url="https://reddit.com/r/test/1",
            permalink="/r/test/1",
            timestamp=now,
            raw={},
        )
        processor._post_buffer = [
            (
                fake_post,
                SentimentResult(
                    compound=0.5,
                    positive=0.6,
                    negative=0.1,
                    neutral=0.3,
                    confidence=0.8,
                ),
            )
        ]
        processor._buffer_start = now

        # Mock process_posts to return a refinement with validated tickers
        # Includes a hallucinated ticker (MSFT) that Gate 1 never extracted
        mock_refinement = SentimentRefinement(
            validated_tickers=[
                ValidatedTicker(
                    ticker="AAPL",
                    company_name="Apple Inc",
                    is_valid_ticker=True,
                    avg_sentiment=0.37,
                    sentiment_label="bullish",
                    confidence=0.9,
                ),
                ValidatedTicker(
                    ticker="TSLA",
                    company_name="Tesla Inc",
                    is_valid_ticker=True,
                    avg_sentiment=0.23,
                    sentiment_label="bullish",
                    confidence=0.85,
                ),
                ValidatedTicker(
                    ticker="MSFT",
                    company_name="Microsoft Corp",
                    is_valid_ticker=True,
                    avg_sentiment=0.10,
                    sentiment_label="neutral",
                    confidence=0.7,
                ),
            ],
            rejected_tickers=[],
            overall_sentiment="bullish",
            narrative_summary="Test narrative",
        )

        # Patch process_posts so we skip the real LLM call
        with patch.object(processor, "process_posts", return_value=mock_refinement):
            signal = await processor.generate_signal()

        # The key assertion: mention_count must come from _raw_tickers, not LLM
        ticker_map = {ts.ticker: ts for ts in signal.ticker_sentiments}

        assert "AAPL" in ticker_map
        assert "TSLA" in ticker_map
        assert ticker_map["AAPL"].mention_count == 3, (
            f"AAPL mention_count should be 3 (from raw_tickers), got {ticker_map['AAPL'].mention_count}"
        )
        assert ticker_map["TSLA"].mention_count == 7, (
            f"TSLA mention_count should be 7 (from raw_tickers), got {ticker_map['TSLA'].mention_count}"
        )

        # MSFT was hallucinated by the LLM (not in _raw_tickers) — must be excluded
        assert "MSFT" not in ticker_map, (
            "MSFT should be excluded from signal because it has no Gate 1 raw_tickers scores"
        )
