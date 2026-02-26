"""Tests for numeric extraction and urgency classification.

Tests the new Stage 1 features:
- Numeric extraction for economic/earnings data (BeatMissStatus, MetricReading, NumericExtraction)
- Urgency classification (UrgencyLevel, classify_urgency_by_rules)
"""

from synesis.processing.news.categorizer import classify_urgency_by_rules
from synesis.processing.news import (
    BeatMissStatus,
    LightClassification,
    MetricReading,
    NewsCategory,
    NumericExtraction,
    PrimaryTopic,
    UrgencyLevel,
)


class TestBeatMissStatus:
    """Tests for BeatMissStatus enum."""

    def test_beat_miss_status_values(self) -> None:
        assert BeatMissStatus.beat.value == "beat"
        assert BeatMissStatus.miss.value == "miss"
        assert BeatMissStatus.inline.value == "inline"
        assert BeatMissStatus.unknown.value == "unknown"


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""

    def test_urgency_level_values(self) -> None:
        assert UrgencyLevel.critical.value == "critical"
        assert UrgencyLevel.high.value == "high"
        assert UrgencyLevel.normal.value == "normal"
        assert UrgencyLevel.low.value == "low"


class TestMetricReading:
    """Tests for MetricReading model."""

    def test_create_metric_reading_full(self) -> None:
        metric = MetricReading(
            metric_name="CPI Y/Y",
            actual=2.4,
            estimate=2.5,
            previous=2.8,
            unit="%",
            period="Q4",
            beat_miss=BeatMissStatus.beat,
            surprise_magnitude=-0.1,
        )

        assert metric.metric_name == "CPI Y/Y"
        assert metric.actual == 2.4
        assert metric.estimate == 2.5
        assert metric.previous == 2.8
        assert metric.unit == "%"
        assert metric.period == "Q4"
        assert metric.beat_miss == BeatMissStatus.beat
        assert metric.surprise_magnitude == -0.1

    def test_create_metric_reading_minimal(self) -> None:
        """Test creating metric with only required fields."""
        metric = MetricReading(
            metric_name="GDP",
            actual=3.2,
        )

        assert metric.metric_name == "GDP"
        assert metric.actual == 3.2
        assert metric.estimate is None
        assert metric.previous is None
        assert metric.unit == "%"  # default
        assert metric.period is None
        assert metric.beat_miss == BeatMissStatus.unknown  # default
        assert metric.surprise_magnitude is None

    def test_create_metric_reading_different_units(self) -> None:
        """Test metric with different units."""
        # Basis points
        metric_bps = MetricReading(
            metric_name="Fed Funds Rate",
            actual=5.25,
            estimate=5.00,
            unit="bps",
            beat_miss=BeatMissStatus.miss,  # Higher rate when cut expected = miss
        )
        assert metric_bps.unit == "bps"

        # Dollar billions
        metric_dollar = MetricReading(
            metric_name="Revenue",
            actual=125.5,
            estimate=120.0,
            unit="B",
            beat_miss=BeatMissStatus.beat,
        )
        assert metric_dollar.unit == "B"


class TestNumericExtraction:
    """Tests for NumericExtraction model."""

    def test_create_empty_extraction(self) -> None:
        extraction = NumericExtraction()

        assert extraction.metrics == []
        assert extraction.headline_metric is None
        assert extraction.overall_beat_miss == BeatMissStatus.unknown
        assert extraction.has_surprise is False
        assert extraction.beats == []
        assert extraction.misses == []

    def test_create_extraction_with_metrics(self) -> None:
        """Test extraction with multiple metrics."""
        cpi_yoy = MetricReading(
            metric_name="CPI Y/Y",
            actual=2.4,
            estimate=2.5,
            previous=2.8,
            beat_miss=BeatMissStatus.beat,  # Lower inflation = beat
        )
        cpi_qoq = MetricReading(
            metric_name="CPI Q/Q",
            actual=0.2,
            estimate=0.3,
            beat_miss=BeatMissStatus.beat,
        )
        trimmed_mean = MetricReading(
            metric_name="Trimmed Mean CPI Y/Y",
            actual=3.2,
            estimate=3.3,
            beat_miss=BeatMissStatus.beat,
        )

        extraction = NumericExtraction(
            metrics=[cpi_yoy, cpi_qoq, trimmed_mean],
            headline_metric="CPI Y/Y",
            overall_beat_miss=BeatMissStatus.beat,
        )

        assert len(extraction.metrics) == 3
        assert extraction.headline_metric == "CPI Y/Y"
        assert extraction.overall_beat_miss == BeatMissStatus.beat
        assert extraction.has_surprise is True  # All have estimates
        assert len(extraction.beats) == 3
        assert len(extraction.misses) == 0

    def test_has_surprise_property(self) -> None:
        """Test has_surprise property with mixed metrics."""
        # One metric with estimate, one without
        with_estimate = MetricReading(
            metric_name="CPI Y/Y",
            actual=2.4,
            estimate=2.5,
        )
        without_estimate = MetricReading(
            metric_name="Core CPI",
            actual=3.0,
        )

        extraction = NumericExtraction(metrics=[with_estimate, without_estimate])
        assert extraction.has_surprise is True  # At least one has estimate

        # No estimates at all
        extraction_no_surprise = NumericExtraction(metrics=[without_estimate])
        assert extraction_no_surprise.has_surprise is False

    def test_beats_and_misses_filters(self) -> None:
        """Test beats and misses filter properties."""
        beat_metric = MetricReading(
            metric_name="GDP",
            actual=3.5,
            estimate=3.0,
            beat_miss=BeatMissStatus.beat,
        )
        miss_metric = MetricReading(
            metric_name="Employment",
            actual=150,
            estimate=200,
            beat_miss=BeatMissStatus.miss,
        )
        inline_metric = MetricReading(
            metric_name="Inflation",
            actual=2.0,
            estimate=2.0,
            beat_miss=BeatMissStatus.inline,
        )
        unknown_metric = MetricReading(
            metric_name="Other",
            actual=1.0,
            beat_miss=BeatMissStatus.unknown,
        )

        extraction = NumericExtraction(
            metrics=[beat_metric, miss_metric, inline_metric, unknown_metric]
        )

        assert len(extraction.beats) == 1
        assert extraction.beats[0].metric_name == "GDP"
        assert len(extraction.misses) == 1
        assert extraction.misses[0].metric_name == "Employment"


class TestLightClassificationWithNumericData:
    """Tests for LightClassification with numeric extraction and urgency."""

    def test_light_classification_with_numeric_data(self) -> None:
        """Test LightClassification includes numeric_data field."""
        numeric = NumericExtraction(
            metrics=[
                MetricReading(
                    metric_name="CPI Y/Y",
                    actual=2.4,
                    estimate=2.5,
                    beat_miss=BeatMissStatus.beat,
                )
            ],
            headline_metric="CPI Y/Y",
            overall_beat_miss=BeatMissStatus.beat,
        )

        classification = LightClassification(
            news_category=NewsCategory.economic_calendar,
            primary_topics=[PrimaryTopic.economic_data],
            summary="Australia CPI comes in below expectations",
            confidence=0.95,
            primary_entity="Australia",
            all_entities=["Australia", "Reserve Bank of Australia"],
            numeric_data=numeric,
            urgency=UrgencyLevel.high,
            urgency_reasoning="Scheduled economic data with surprise",
        )

        assert classification.numeric_data is not None
        assert len(classification.numeric_data.metrics) == 1
        assert classification.numeric_data.headline_metric == "CPI Y/Y"
        assert classification.urgency == UrgencyLevel.high
        assert classification.urgency_reasoning == "Scheduled economic data with surprise"

    def test_light_classification_without_numeric_data(self) -> None:
        """Test LightClassification works without numeric_data (default None)."""
        classification = LightClassification(
            primary_topics=[PrimaryTopic.corporate_actions],
            summary="Company announces merger",
            confidence=0.9,
            primary_entity="Acme Corp",
        )

        assert classification.numeric_data is None
        assert classification.urgency == UrgencyLevel.normal  # default
        assert classification.urgency_reasoning == ""  # default

    def test_light_classification_serialization(self) -> None:
        """Test JSON serialization with numeric data."""
        numeric = NumericExtraction(
            metrics=[
                MetricReading(
                    metric_name="EPS",
                    actual=1.50,
                    estimate=1.40,
                    beat_miss=BeatMissStatus.beat,
                    surprise_magnitude=0.10,
                )
            ],
            headline_metric="EPS",
            overall_beat_miss=BeatMissStatus.beat,
        )

        classification = LightClassification(
            news_category=NewsCategory.economic_calendar,
            primary_topics=[PrimaryTopic.earnings],
            summary="Apple beats EPS estimates",
            confidence=0.95,
            primary_entity="Apple",
            numeric_data=numeric,
            urgency=UrgencyLevel.high,
            urgency_reasoning="Earnings beat",
        )

        data = classification.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["numeric_data"]["headline_metric"] == "EPS"
        assert data["numeric_data"]["metrics"][0]["actual"] == 1.50
        assert data["numeric_data"]["metrics"][0]["beat_miss"] == "beat"
        assert data["urgency"] == "high"


class TestClassifyUrgencyByRules:
    """Tests for classify_urgency_by_rules function."""

    # Critical patterns
    def test_urgency_critical_breaking(self) -> None:
        """Test critical urgency for breaking news markers."""
        assert (
            classify_urgency_by_rules("*BREAKING: Fed announces emergency cut")
            == UrgencyLevel.critical
        )
        assert (
            classify_urgency_by_rules("**FLASH: Market circuit breaker hit")
            == UrgencyLevel.critical
        )
        assert classify_urgency_by_rules("ALERT: Major bank failure") == UrgencyLevel.critical

    def test_urgency_critical_fed_decision(self) -> None:
        """Test critical urgency for Fed rate decisions."""
        assert classify_urgency_by_rules("Fed rate decision: 50bps cut") == UrgencyLevel.critical
        assert classify_urgency_by_rules("FOMC decision: rates unchanged") == UrgencyLevel.critical
        assert classify_urgency_by_rules("Fed rate hike surprises markets") == UrgencyLevel.critical

    def test_urgency_critical_surprise(self) -> None:
        """Test critical urgency for surprise announcements."""
        assert classify_urgency_by_rules("SURPRISE rate cut from ECB") == UrgencyLevel.critical
        assert classify_urgency_by_rules("UNEXPECTED earnings revision") == UrgencyLevel.critical
        assert classify_urgency_by_rules("EMERGENCY meeting called") == UrgencyLevel.critical

    # High patterns
    def test_urgency_high_economic_data(self) -> None:
        """Test high urgency for economic data WITH numbers."""
        assert (
            classify_urgency_by_rules("CPI comes in at 2.4% vs 2.5% expected") == UrgencyLevel.high
        )
        assert classify_urgency_by_rules("NFP: 200K jobs added") == UrgencyLevel.high
        assert classify_urgency_by_rules("GDP growth 3.5% in Q4") == UrgencyLevel.high
        assert classify_urgency_by_rules("PPI rises 0.3% m/m") == UrgencyLevel.high
        assert classify_urgency_by_rules("PCE inflation 2.8%") == UrgencyLevel.high

    def test_urgency_high_earnings_with_outcome(self) -> None:
        """Test high urgency for earnings with beat/miss."""
        assert classify_urgency_by_rules("AAPL EARNINGS BEAT expectations") == UrgencyLevel.high
        assert classify_urgency_by_rules("Tesla earnings miss by wide margin") == UrgencyLevel.high

    def test_urgency_high_basis_points(self) -> None:
        """Test high urgency for basis point moves."""
        assert classify_urgency_by_rules("10Y yield up 15bps") == UrgencyLevel.high
        assert classify_urgency_by_rules("Spread widened by 25 bps") == UrgencyLevel.high

    # Low patterns
    def test_urgency_low_opinions(self) -> None:
        """Test low urgency for opinions and analysis."""
        assert classify_urgency_by_rules("In my OPINION, rates will fall") == UrgencyLevel.low
        assert classify_urgency_by_rules("My ANALYSIS suggests bullish") == UrgencyLevel.low
        assert classify_urgency_by_rules("COMMENTARY on the Fed") == UrgencyLevel.low
        assert classify_urgency_by_rules("IMO the market is overreacting") == UrgencyLevel.low
        assert classify_urgency_by_rules("I think we'll see a pullback") == UrgencyLevel.low

    def test_urgency_low_reposts_and_threads(self) -> None:
        """Test low urgency for forwarded reposts and threads."""
        assert (
            classify_urgency_by_rules("RT @analyst: Great insight on markets") == UrgencyLevel.low
        )
        assert classify_urgency_by_rules("THREAD on Fed policy 1/10") == UrgencyLevel.low

    # Ambiguous (returns None - let LLM decide)
    def test_urgency_ambiguous(self) -> None:
        """Test ambiguous cases that return None."""
        assert classify_urgency_by_rules("Markets open higher today") is None
        assert classify_urgency_by_rules("Apple announces new product") is None
        assert classify_urgency_by_rules("Oil prices stable") is None
        # Just CPI mention without numbers - not confident enough
        assert classify_urgency_by_rules("CPI report coming tomorrow") is None

    # Edge cases
    def test_urgency_case_insensitive(self) -> None:
        """Test patterns are case insensitive."""
        # Breaking at start with colon is critical
        assert classify_urgency_by_rules("BREAKING: Major news") == UrgencyLevel.critical
        # Opinion patterns are case insensitive
        assert classify_urgency_by_rules("OPINION piece") == UrgencyLevel.low
        assert classify_urgency_by_rules("opinion piece") == UrgencyLevel.low
        # Economic data with numbers
        assert classify_urgency_by_rules("cpi comes in at 2.5%") == UrgencyLevel.high
        assert classify_urgency_by_rules("CPI comes in at 2.5%") == UrgencyLevel.high

    def test_urgency_critical_takes_priority(self) -> None:
        """Test critical patterns take priority over high patterns."""
        # Contains both critical (BREAKING) and high (CPI with number) patterns
        text = "*BREAKING: CPI 2.4% vs 2.5% expected"
        assert classify_urgency_by_rules(text) == UrgencyLevel.critical

    def test_urgency_economic_data_without_numbers_is_ambiguous(self) -> None:
        """Economic data indicators without actual numbers should be ambiguous."""
        # "CPI" alone without a number - could be scheduled release announcement
        assert classify_urgency_by_rules("CPI report scheduled for tomorrow") is None
        assert classify_urgency_by_rules("Waiting for NFP data") is None
