"""Tests for processing models."""

from datetime import datetime, timezone


from synesis.processing.models import (
    BreakingClassification,
    Direction,
    EvaluatorOutput,
    EventType,
    Flow1Signal,
    ImpactLevel,
    InvestmentAnalysis,
    LightClassification,
    MarketEvaluation,
    MarketOpportunity,
    NewsCategory,
    ResearchQuality,
    SectorImplication,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    TickerAnalysis,
    UnifiedMessage,
)


class TestSourceEnums:
    """Tests for source type enums."""

    def test_source_platform_values(self) -> None:
        assert SourcePlatform.twitter.value == "twitter"
        assert SourcePlatform.telegram.value == "telegram"

    def test_source_type_values(self) -> None:
        assert SourceType.news.value == "news"
        assert SourceType.analysis.value == "analysis"


class TestUnifiedMessage:
    """Tests for UnifiedMessage model."""

    def test_create_twitter_message(self) -> None:
        msg = UnifiedMessage(
            external_id="12345",
            source_platform=SourcePlatform.twitter,
            source_account="@DeItaone",
            text="Breaking: Fed announces rate cut",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

        assert msg.external_id == "12345"
        assert msg.source_platform == SourcePlatform.twitter
        assert msg.source_type == SourceType.news
        assert msg.urgency == "high"

    def test_create_analysis_message(self) -> None:
        msg = UnifiedMessage(
            external_id="67890",
            source_platform=SourcePlatform.twitter,
            source_account="@analyst",
            text="My analysis on the market",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.analysis,
        )

        assert msg.source_type == SourceType.analysis
        assert msg.urgency == "normal"

    def test_telegram_message(self) -> None:
        msg = UnifiedMessage(
            external_id="tg_123",
            source_platform=SourcePlatform.telegram,
            source_account="marketfeed",
            text="Market update",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

        assert msg.source_platform == SourcePlatform.telegram
        assert msg.urgency == "high"


class TestBreakingClassification:
    """Tests for BreakingClassification model."""

    def test_create_classification(self) -> None:
        classification = BreakingClassification(
            event_type=EventType.macro,
            summary="Fed cuts rates by 25bps",
            confidence=0.95,
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            tickers=["SPY", "QQQ"],
            sectors=["financials"],
            search_keywords=["Fed", "rate cut"],
            related_markets=["Fed rate cut March 2025"],
        )

        assert classification.event_type == EventType.macro
        assert classification.confidence == 0.95
        assert "SPY" in classification.tickers
        assert classification.search_keywords == ["Fed", "rate cut"]

    def test_confidence_validation(self) -> None:
        # Valid confidence
        classification = BreakingClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            predicted_impact=ImpactLevel.low,
            market_direction=Direction.neutral,
        )
        assert classification.confidence == 0.5

    def test_empty_lists_default(self) -> None:
        classification = BreakingClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            predicted_impact=ImpactLevel.low,
            market_direction=Direction.neutral,
        )

        assert classification.tickers == []
        assert classification.sectors == []
        assert classification.search_keywords == []
        assert classification.related_markets == []


class TestMarketOpportunity:
    """Tests for MarketOpportunity model."""

    def test_create_opportunity(self) -> None:
        opp = MarketOpportunity(
            market_id="market_123",
            platform="polymarket",
            question="Will the Fed cut rates in March?",
            slug="fed-rate-cut-march",
            yes_price=0.35,
            no_price=0.65,
            volume_24h=100000.0,
            suggested_direction="yes",
            reason="News indicates rate cut likely",
            end_date=datetime(2025, 3, 15, tzinfo=timezone.utc),
        )

        assert opp.market_id == "market_123"
        assert opp.yes_price == 0.35
        assert opp.suggested_direction == "yes"


class TestFlow1Signal:
    """Tests for Flow1Signal model.

    Uses the 2-stage architecture:
    - extraction: LightClassification (Stage 1, required)
    - analysis: SmartAnalysis (Stage 2, optional)
    """

    def test_create_signal(self) -> None:
        """Test creating a signal with 2-stage architecture."""
        # Stage 1: LightClassification (entity extraction only)
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            event_type=EventType.macro,
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
            sectors=["financials"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Fed pivot bullish for equities",
            thesis_confidence=0.8,
        )

        signal = Flow1Signal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@DeItaone",
            source_type=SourceType.news,
            raw_text="Breaking: Fed announces rate cut",
            external_id="12345",
            extraction=extraction,
            analysis=analysis,
            watchlist_tickers=["SPY"],
            watchlist_sectors=["financials"],
        )

        assert signal.urgency == "high"
        assert signal.extraction.event_type == EventType.macro
        assert "SPY" in signal.watchlist_tickers
        # Tickers/sectors now come from analysis
        assert signal.tickers == ["SPY"]
        assert signal.sectors == ["financials"]

    def test_signal_urgency_from_source_type(self) -> None:
        """Test that urgency is derived from source_type, not LLM."""
        extraction = LightClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )

        # News source = high urgency
        news_signal = Flow1Signal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@news",
            source_type=SourceType.news,
            raw_text="Test",
            external_id="1",
            extraction=extraction,
        )
        assert news_signal.urgency == "high"

        # Analysis source = normal urgency
        analysis_signal = Flow1Signal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@analyst",
            source_type=SourceType.analysis,
            raw_text="Test",
            external_id="2",
            extraction=extraction,
        )
        assert analysis_signal.urgency == "normal"

    def test_signal_serialization(self) -> None:
        """Test JSON serialization of Flow1Signal."""
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            event_type=EventType.macro,
            summary="Test",
            confidence=0.9,
            primary_entity="Federal Reserve",
        )

        signal = Flow1Signal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            source_type=SourceType.news,
            raw_text="Test message",
            external_id="123",
            extraction=extraction,
        )

        # Should be serializable to JSON
        data = signal.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["source_platform"] == "twitter"
        assert data["extraction"]["event_type"] == "macro"

    def test_signal_without_analysis(self) -> None:
        """Test signal with only Stage 1 (no Stage 2 analysis)."""
        extraction = LightClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )

        signal = Flow1Signal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            source_type=SourceType.news,
            raw_text="Test",
            external_id="1",
            extraction=extraction,
            # No analysis - Stage 2 was skipped
        )

        assert signal.analysis is None
        assert signal.tickers == []  # Empty when no analysis
        assert signal.sectors == []
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

        assert ticker.company_name is None
        assert ticker.catalysts == []
        assert ticker.risk_factors == []


class TestSectorImplication:
    """Tests for SectorImplication model."""

    def test_create_sector_implication(self) -> None:
        sector = SectorImplication(
            sector="technology",
            direction=Direction.bullish,
            reasoning="AI spending accelerating",
            affected_subsectors=["semiconductors", "cloud", "software"],
        )

        assert sector.sector == "technology"
        assert sector.direction == Direction.bullish
        assert len(sector.affected_subsectors) == 3

    def test_empty_subsectors_default(self) -> None:
        sector = SectorImplication(
            sector="energy",
            direction=Direction.bearish,
            reasoning="Oil prices falling",
        )

        assert sector.affected_subsectors == []


class TestResearchQuality:
    """Tests for ResearchQuality enum."""

    def test_research_quality_values(self) -> None:
        assert ResearchQuality.high.value == "high"
        assert ResearchQuality.medium.value == "medium"
        assert ResearchQuality.low.value == "low"


class TestInvestmentAnalysis:
    """Tests for InvestmentAnalysis model."""

    def test_create_investment_analysis(self) -> None:
        analysis = InvestmentAnalysis(
            ticker_analyses=[
                TickerAnalysis(
                    ticker="SPY",
                    bull_thesis="Rate cuts bullish",
                    bear_thesis="Recession risk",
                    net_direction=Direction.bullish,
                    conviction=0.8,
                    time_horizon="days",
                ),
            ],
            sector_implications=[
                SectorImplication(
                    sector="financials",
                    direction=Direction.bearish,
                    reasoning="NIM compression",
                ),
            ],
            historical_precedent="2019 cuts led to rally",
            similar_events=["2019 rate cuts"],
            typical_market_reaction="Initial rally",
            actionable_insights=["Buy SPY calls"],
            primary_thesis="Fed dovish pivot",
            thesis_confidence=0.75,
            research_quality=ResearchQuality.high,
        )

        assert len(analysis.ticker_analyses) == 1
        assert len(analysis.sector_implications) == 1
        assert analysis.thesis_confidence == 0.75
        assert analysis.research_quality == ResearchQuality.high

    def test_has_tradable_tickers_high_conviction(self) -> None:
        analysis = InvestmentAnalysis(
            ticker_analyses=[
                TickerAnalysis(
                    ticker="HIGH",
                    bull_thesis="Strong",
                    bear_thesis="Weak",
                    net_direction=Direction.bullish,
                    conviction=0.85,  # >= 0.7
                    time_horizon="days",
                ),
            ],
            primary_thesis="Test",
        )

        assert analysis.has_tradable_tickers is True

    def test_has_tradable_tickers_low_conviction(self) -> None:
        analysis = InvestmentAnalysis(
            ticker_analyses=[
                TickerAnalysis(
                    ticker="LOW",
                    bull_thesis="Weak",
                    bear_thesis="Strong",
                    net_direction=Direction.neutral,
                    conviction=0.5,  # < 0.7
                    time_horizon="days",
                ),
            ],
            primary_thesis="Test",
        )

        assert analysis.has_tradable_tickers is False

    def test_top_ticker(self) -> None:
        analysis = InvestmentAnalysis(
            ticker_analyses=[
                TickerAnalysis(
                    ticker="LOW",
                    bull_thesis="A",
                    bear_thesis="B",
                    net_direction=Direction.neutral,
                    conviction=0.3,
                    time_horizon="days",
                ),
                TickerAnalysis(
                    ticker="HIGH",
                    bull_thesis="A",
                    bear_thesis="B",
                    net_direction=Direction.bullish,
                    conviction=0.9,
                    time_horizon="days",
                ),
                TickerAnalysis(
                    ticker="MID",
                    bull_thesis="A",
                    bear_thesis="B",
                    net_direction=Direction.bearish,
                    conviction=0.6,
                    time_horizon="days",
                ),
            ],
            primary_thesis="Test",
        )

        assert analysis.top_ticker is not None
        assert analysis.top_ticker.ticker == "HIGH"

    def test_top_ticker_empty(self) -> None:
        analysis = InvestmentAnalysis(primary_thesis="Test")
        assert analysis.top_ticker is None

    def test_defaults(self) -> None:
        analysis = InvestmentAnalysis(primary_thesis="Test thesis")

        assert analysis.ticker_analyses == []
        assert analysis.sector_implications == []
        assert analysis.similar_events == []
        assert analysis.actionable_insights == []
        assert analysis.historical_precedent == ""
        assert analysis.typical_market_reaction == ""
        assert analysis.thesis_confidence == 0.5
        assert analysis.research_quality == ResearchQuality.medium


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


class TestEvaluatorOutput:
    """Tests for EvaluatorOutput model."""

    def test_create_evaluator_output(self) -> None:
        eval1 = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.4,
            estimated_fair_price=0.6,
            edge=0.2,
            verdict="undervalued",
            confidence=0.8,
            reasoning="Strong edge",
            recommended_side="yes",
        )
        eval2 = MarketEvaluation(
            market_id="m2",
            market_question="Q2",
            is_relevant=False,
            relevance_reasoning="Not relevant",
            current_price=0.5,
            verdict="skip",
            confidence=0.0,
            reasoning="Skip",
            recommended_side="skip",
        )

        output = EvaluatorOutput(
            markets_found=2,
            markets_relevant=1,
            evaluations=[eval1, eval2],
            has_tradable_edge=True,
            best_opportunity=eval1,
        )

        assert output.markets_found == 2
        assert output.markets_relevant == 1
        assert output.has_tradable_edge is True
        assert output.best_opportunity == eval1

    def test_relevant_evaluations_property(self) -> None:
        eval_relevant = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.5,
            verdict="fair",
            confidence=0.5,
            reasoning="Fair",
            recommended_side="skip",
        )
        eval_irrelevant = MarketEvaluation(
            market_id="m2",
            market_question="Q2",
            is_relevant=False,
            relevance_reasoning="Not relevant",
            current_price=0.5,
            verdict="skip",
            confidence=0.0,
            reasoning="Skip",
            recommended_side="skip",
        )

        output = EvaluatorOutput(
            markets_found=2,
            markets_relevant=1,
            evaluations=[eval_relevant, eval_irrelevant],
        )

        relevant = output.relevant_evaluations
        assert len(relevant) == 1
        assert relevant[0].market_id == "m1"

    def test_opportunities_with_edge_property(self) -> None:
        eval_with_edge = MarketEvaluation(
            market_id="m1",
            market_question="Q1",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.3,
            estimated_fair_price=0.5,
            edge=0.2,  # > 0.05
            verdict="undervalued",
            confidence=0.8,
            reasoning="Good edge",
            recommended_side="yes",
        )
        eval_no_edge = MarketEvaluation(
            market_id="m2",
            market_question="Q2",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.5,
            estimated_fair_price=0.52,
            edge=0.02,  # < 0.05
            verdict="fair",
            confidence=0.6,
            reasoning="No edge",
            recommended_side="skip",
        )

        output = EvaluatorOutput(
            markets_found=2,
            markets_relevant=2,
            evaluations=[eval_with_edge, eval_no_edge],
        )

        opps = output.opportunities_with_edge
        assert len(opps) == 1
        assert opps[0].market_id == "m1"


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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
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
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
            primary_thesis="Test",
            thesis_confidence=0.5,
            market_evaluations=[eval1, eval2, eval3],
        )

        best = analysis.best_opportunity
        assert best is not None
        assert best.market_id == "m2"  # Highest edge
        assert best.edge == 0.25
