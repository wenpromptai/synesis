"""Tests for Telegram notification formatting."""

from datetime import datetime, timezone

from synesis.notifications.telegram import format_stage1_signal
from synesis.processing.news import (
    LightClassification,
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
        "matched_tickers": ["AAPL"],
        "impact_score": 60,
        "impact_reasons": ["wire_prefix:+15"],
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
        """High urgency uses no 🚨 prefix."""
        extraction = _make_extraction(urgency=UrgencyLevel.high)
        result = format_stage1_signal(_make_message(), extraction)
        assert "[1st pass]" in result
        assert "🚨[1st pass]" not in result

    def test_matched_tickers_in_output(self) -> None:
        """Matched tickers appear in the formatted message."""
        extraction = _make_extraction(matched_tickers=["NVDA", "MRVL"])
        result = format_stage1_signal(_make_message(), extraction)
        assert "NVDA" in result
        assert "MRVL" in result

    def test_tickers_omitted_when_empty(self) -> None:
        """Tickers line is omitted when matched_tickers is empty."""
        extraction = _make_extraction(matched_tickers=[])
        result = format_stage1_signal(_make_message(), extraction)
        assert "Tickers" not in result

    def test_impact_score_in_output(self) -> None:
        """Impact score appears in the formatted message."""
        extraction = _make_extraction(impact_score=75)
        result = format_stage1_signal(_make_message(), extraction)
        assert "75/100" in result

    def test_html_escaping_in_source_account(self) -> None:
        """Special HTML characters in source_account are escaped."""
        result = format_stage1_signal(_make_message(account="<bold>"), _make_extraction())
        assert "<bold>" not in result
        assert "&lt;bold&gt;" in result

    def test_source_account_in_header(self) -> None:
        """Source account appears in the header line."""
        result = format_stage1_signal(_make_message(account="@marketfeed"), _make_extraction())
        assert "@marketfeed" in result
