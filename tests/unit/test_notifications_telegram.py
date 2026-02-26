"""Tests for Telegram notification formatting."""

from datetime import datetime, timezone

from synesis.notifications.telegram import format_stage1_signal
from synesis.processing.news import (
    LightClassification,
    PrimaryTopic,
    SecondaryTopic,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)


def _make_message(text: str = "Test message", account: str = "@test") -> UnifiedMessage:
    return UnifiedMessage(
        external_id="test_123",
        source_platform=SourcePlatform.telegram,
        source_account=account,
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def _make_extraction(**kwargs: object) -> LightClassification:
    defaults: dict = {
        "primary_topics": [PrimaryTopic.monetary_policy],
        "summary": "Fed cuts rates 25bps",
        "confidence": 0.9,
        "primary_entity": "Federal Reserve",
        "urgency": UrgencyLevel.high,
    }
    defaults.update(kwargs)
    return LightClassification(**defaults)


class TestFormatStage1Signal:
    """Tests for format_stage1_signal formatter."""

    def test_critical_urgency_marker(self) -> None:
        """Critical urgency uses 🚨 prefix."""
        extraction = _make_extraction(urgency=UrgencyLevel.critical)
        result = format_stage1_signal(_make_message(), extraction)
        assert "🚨[1st pass]" in result

    def test_high_urgency_marker(self) -> None:
        """High urgency uses no prefix."""
        extraction = _make_extraction(urgency=UrgencyLevel.high)
        result = format_stage1_signal(_make_message(), extraction)
        assert "[1st pass]" in result
        assert "🚨[1st pass]" not in result

    def test_primary_topics_in_output(self) -> None:
        """Primary topics appear in the formatted message."""
        extraction = _make_extraction(
            primary_topics=[PrimaryTopic.earnings, PrimaryTopic.corporate_actions]
        )
        result = format_stage1_signal(_make_message(), extraction)
        assert "earnings" in result
        assert "corporate_actions" in result

    def test_secondary_topics_included_when_present(self) -> None:
        """Secondary topics line is present when secondary_topics is non-empty."""
        extraction = _make_extraction(secondary_topics=[SecondaryTopic.semiconductors])
        result = format_stage1_signal(_make_message(), extraction)
        assert "semiconductors" in result

    def test_secondary_topics_omitted_when_empty(self) -> None:
        """Secondary topics line is omitted when secondary_topics is empty."""
        extraction = _make_extraction(secondary_topics=[])
        result = format_stage1_signal(_make_message(), extraction)
        # No secondary line — just primary icon line and entities
        lines = result.splitlines()
        secondary_lines = [ln for ln in lines if "semiconductors" in ln or "biotech" in ln]
        assert secondary_lines == []

    def test_entities_from_all_entities(self) -> None:
        """When all_entities is populated, first 5 are used."""
        extraction = _make_extraction(
            all_entities=["Apple", "Tim Cook", "Goldman Sachs", "JPMorgan", "Buffett", "Gates"]
        )
        result = format_stage1_signal(_make_message(), extraction)
        assert "Apple" in result
        assert "Buffett" in result
        assert "Gates" not in result  # 6th entity is excluded

    def test_entities_fallback_to_primary_entity(self) -> None:
        """When all_entities is empty, primary_entity is used."""
        extraction = _make_extraction(
            all_entities=[],
            primary_entity="Federal Reserve",
        )
        result = format_stage1_signal(_make_message(), extraction)
        assert "Federal Reserve" in result

    def test_html_escaping_in_source_account(self) -> None:
        """Special HTML characters in source_account are escaped."""
        result = format_stage1_signal(_make_message(account="<bold>"), _make_extraction())
        assert "<bold>" not in result
        assert "&lt;bold&gt;" in result

    def test_html_escaping_in_summary(self) -> None:
        """Special HTML characters in summary are escaped."""
        extraction = _make_extraction(summary="Price > $100 & rising")
        result = format_stage1_signal(_make_message(), extraction)
        assert "Price > $100 & rising" not in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_html_escaping_in_entities(self) -> None:
        """Ampersands in entity names (e.g. AT&T) are escaped."""
        extraction = _make_extraction(
            all_entities=["AT&T"],
            primary_entity="AT&T",
        )
        result = format_stage1_signal(_make_message(), extraction)
        assert "AT&T" not in result.split("👤")[1]  # raw & not in entities line
        assert "AT&amp;T" in result

    def test_summary_in_output(self) -> None:
        """Summary text appears in the formatted message."""
        extraction = _make_extraction(summary="Central bank cuts by 25bps unexpectedly")
        result = format_stage1_signal(_make_message(), extraction)
        assert "Central bank cuts by 25bps unexpectedly" in result

    def test_source_account_in_header(self) -> None:
        """Source account appears in the header line."""
        result = format_stage1_signal(_make_message(account="@marketfeed"), _make_extraction())
        assert "@marketfeed" in result
