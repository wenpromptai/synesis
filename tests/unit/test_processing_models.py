"""Tests for processing models."""

from datetime import datetime, timezone


from synesis.processing.news import (
    Direction,
    LightClassification,
    MarketEvaluation,
    NewsCategory,
    NewsSignal,
    PrimaryTopic,
    ResearchQuality,
    SmartAnalysis,
    SourcePlatform,
    TickerAnalysis,
    UnifiedMessage,
)


class TestSourceEnums:
    """Tests for source platform enum."""

    def test_source_platform_values(self) -> None:
        assert SourcePlatform.telegram.value == "telegram"
        assert SourcePlatform.twitter.value == "twitter"


class TestUnifiedMessage:
    """Tests for UnifiedMessage model."""

    def test_create_telegram_message(self) -> None:
        msg = UnifiedMessage(
            external_id="12345",
            source_platform=SourcePlatform.telegram,
            source_account="@DeItaone",
            text="Breaking: Fed announces rate cut",
            timestamp=datetime.now(timezone.utc),
        )

        assert msg.external_id == "12345"
        assert msg.source_platform == SourcePlatform.telegram

    def test_message_with_raw_data(self) -> None:
        msg = UnifiedMessage(
            external_id="67890",
            source_platform=SourcePlatform.telegram,
            source_account="marketfeed",
            text="Market update",
            timestamp=datetime.now(timezone.utc),
            raw={"channel": "news"},
        )

        assert msg.raw == {"channel": "news"}

    def test_telegram_message(self) -> None:
        msg = UnifiedMessage(
            external_id="tg_123",
            source_platform=SourcePlatform.telegram,
            source_account="marketfeed",
            text="Market update",
            timestamp=datetime.now(timezone.utc),
        )

        assert msg.source_platform == SourcePlatform.telegram


class TestNewsSignal:
    """Tests for NewsSignal model.

    Uses the 2-stage architecture:
    - extraction: LightClassification (Stage 1, required)
    - analysis: SmartAnalysis (Stage 2, optional)
    """

    def test_create_signal(self) -> None:
        """Test creating a signal with 2-stage architecture."""
        # Stage 1: LightClassification (entity extraction only)
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Federal Reserve",
            all_entities=["Federal Reserve"],
            polymarket_keywords=["Fed", "rate cut"],
            search_keywords=["Fed rate cut"],
        )

        # Stage 2: SmartAnalysis (all informed judgments)
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sentiment=Direction.bullish,
            sentiment_score=0.7,
            primary_thesis="Fed pivot bullish for equities",
            thesis_confidence=0.8,
        )

        signal = NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.telegram,
            source_account="@DeItaone",
            raw_text="Breaking: Fed announces rate cut",
            external_id="12345",
            extraction=extraction,
            analysis=analysis,
        )

        assert PrimaryTopic.monetary_policy in signal.extraction.primary_topics
        # Tickers come from analysis (Stage 2)
        assert signal.tickers == ["SPY"]

    def test_signal_serialization(self) -> None:
        """Test JSON serialization of NewsSignal."""
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Test",
            confidence=0.9,
            primary_entity="Federal Reserve",
        )

        signal = NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            raw_text="Test message",
            external_id="123",
            extraction=extraction,
        )

        # Should be serializable to JSON
        data = signal.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["source_platform"] == "telegram"
        assert "monetary_policy" in data["extraction"]["primary_topics"]

    def test_signal_without_analysis(self) -> None:
        """Test signal with only Stage 1 (no Stage 2 analysis)."""
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.other],
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )

        signal = NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            raw_text="Test",
            external_id="1",
            extraction=extraction,
            # No analysis - Stage 2 was skipped
        )

        assert signal.analysis is None
        assert signal.tickers == []  # Empty when no analysis
        assert signal.has_edge is False


class TestTickerAnalysis:
    """Tests for TickerAnalysis model."""

    def test_create_ticker_analysis(self) -> None:
        ticker = TickerAnalysis(
            ticker="AAPL",
            company_name="Apple Inc.",
            bull_thesis="Strong iPhone sales expected",
            bear_thesis="China regulatory concerns",
            net_direction=Direction.bullish,
            conviction=0.85,
            time_horizon="weeks",
            catalysts=["Earnings", "WWDC"],
            risk_factors=["Trade tensions", "Competition"],
        )

        assert ticker.ticker == "AAPL"
        assert ticker.company_name == "Apple Inc."
        assert ticker.conviction == 0.85
        assert ticker.net_direction == Direction.bullish
        assert len(ticker.catalysts) == 2
        assert len(ticker.risk_factors) == 2

    def test_conviction_bounds(self) -> None:
        # Test minimum
        ticker = TickerAnalysis(
            ticker="TEST",
            bull_thesis="Test",
            bear_thesis="Test",
            net_direction=Direction.neutral,
            conviction=0.0,
            time_horizon="days",
        )
        assert ticker.conviction == 0.0

        # Test maximum
        ticker2 = TickerAnalysis(
            ticker="TEST",
            bull_thesis="Test",
            bear_thesis="Test",
            net_direction=Direction.neutral,
            conviction=1.0,
            time_horizon="days",
        )
        assert ticker2.conviction == 1.0

    def test_optional_fields_defaults(self) -> None:
        ticker = TickerAnalysis(
            ticker="TEST",
            bull_thesis="Bull",
            bear_thesis="Bear",
            net_direction=Direction.neutral,
            conviction=0.5,
            time_horizon="days",
        )

        assert ticker.company_name == ""
        assert ticker.relevance_score == 0.7  # Default
        assert ticker.relevance_reason == ""
        assert ticker.catalysts == []
        assert ticker.risk_factors == []


class TestResearchQuality:
    """Tests for ResearchQuality enum."""

    def test_research_quality_values(self) -> None:
        assert ResearchQuality.high.value == "high"
        assert ResearchQuality.medium.value == "medium"
        assert ResearchQuality.low.value == "low"


class TestMarketEvaluation:
    """Tests for MarketEvaluation model."""

    def test_create_relevant_evaluation(self) -> None:
        eval = MarketEvaluation(
            market_id="market_123",
            market_question="Will Fed cut rates in March?",
            is_relevant=True,
            relevance_reasoning="Directly related to Fed news",
            current_price=0.35,
            estimated_fair_price=0.55,
            edge=0.20,
            verdict="undervalued",
            confidence=0.7,
            reasoning="Rate cut likely based on guidance",
            recommended_side="yes",
        )

        assert eval.is_relevant is True
        assert eval.edge == 0.20
        assert eval.verdict == "undervalued"
        assert eval.recommended_side == "yes"

    def test_create_irrelevant_evaluation(self) -> None:
        eval = MarketEvaluation(
            market_id="market_456",
            market_question="Will Trump win 2024?",
            is_relevant=False,
            relevance_reasoning="Not related to Fed rate news",
            current_price=0.50,
            estimated_fair_price=None,
            edge=None,
            verdict="skip",
            confidence=0.0,
            reasoning="Market not relevant to this news",
            recommended_side="skip",
        )

        assert eval.is_relevant is False
        assert eval.edge is None
        assert eval.verdict == "skip"


class TestSmartAnalysisEdgeBoundaries:
    """Tests for SmartAnalysis has_tradable_edge boundary conditions.

    Edge threshold: > 0.05 (5%)
    Confidence threshold: > 0.5 (50%)
    """

    def test_has_tradable_edge_above_thresholds(self) -> None:
        """Test has_tradable_edge True when both thresholds exceeded."""
        eval_good = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.3,
            estimated_fair_price=0.5,
            edge=0.10,  # > 0.05 ✓
            verdict="undervalued",
            confidence=0.7,  # > 0.5 ✓
            reasoning="Good",
            recommended_side="yes",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval_good],
        )

        assert analysis.has_tradable_edge is True

    def test_has_tradable_edge_exactly_at_edge_boundary(self) -> None:
        """Test edge = 0.05 exactly (should fail > check)."""
        eval_at_boundary = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.45,
            estimated_fair_price=0.50,
            edge=0.05,  # Exactly 0.05 - NOT > 0.05
            verdict="fair",
            confidence=0.8,
            reasoning="Edge at boundary",
            recommended_side="yes",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval_at_boundary],
        )

        # 0.05 is NOT > 0.05, so should be False
        assert analysis.has_tradable_edge is False

    def test_has_tradable_edge_confidence_exactly_at_threshold(self) -> None:
        """Test confidence = 0.5 exactly (should fail > check)."""
        eval_confidence_boundary = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.3,
            estimated_fair_price=0.5,
            edge=0.10,  # > 0.05 ✓
            verdict="undervalued",
            confidence=0.5,  # Exactly 0.5 - NOT > 0.5
            reasoning="Confidence at boundary",
            recommended_side="yes",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval_confidence_boundary],
        )

        # confidence 0.5 is NOT > 0.5, so should be False
        assert analysis.has_tradable_edge is False

    def test_has_tradable_edge_with_negative_edge(self) -> None:
        """Test overvalued markets with negative edge.

        Negative edge means current price is HIGHER than fair price.
        The trader should buy NO side. Implementation uses abs(edge) > 0.05,
        so negative edges with |edge| > 0.05 ARE tradable.
        """
        eval_overvalued = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.7,
            estimated_fair_price=0.5,
            edge=-0.20,  # Negative edge (overvalued), abs(-0.20) = 0.20 > 0.05
            verdict="overvalued",
            confidence=0.8,  # > 0.5
            reasoning="Overvalued - buy NO",
            recommended_side="no",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval_overvalued],
        )

        # Implementation uses abs(edge) > 0.05, so negative edges are tradable
        assert analysis.has_tradable_edge is True

    def test_has_tradable_edge_no_evaluations(self) -> None:
        """Test has_tradable_edge False when no market evaluations."""
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[],
        )

        assert analysis.has_tradable_edge is False

    def test_best_opportunity_selects_highest_edge(self) -> None:
        """Test best_opportunity returns market with highest edge."""
        eval1 = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.4,
            estimated_fair_price=0.5,
            edge=0.10,  # Lower edge
            verdict="undervalued",
            confidence=0.7,
            reasoning="Good",
            recommended_side="yes",
        )
        eval2 = MarketEvaluation(
            market_id="m2",
            market_question="Q2",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.3,
            estimated_fair_price=0.55,
            edge=0.25,  # Highest edge
            verdict="undervalued",
            confidence=0.8,
            reasoning="Best",
            recommended_side="yes",
        )
        eval3 = MarketEvaluation(
            market_id="m3",
            market_question="Q3",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.35,
            estimated_fair_price=0.5,
            edge=0.15,  # Middle edge
            verdict="undervalued",
            confidence=0.6,
            reasoning="OK",
            recommended_side="yes",
        )

        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval1, eval2, eval3],
        )

        best = analysis.best_opportunity
        assert best is not None
        assert best.market_id == "m2"  # Highest edge
        assert best.edge == 0.25
