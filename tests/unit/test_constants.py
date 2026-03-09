"""Smoke tests for the constants package — importability, value guards."""

from __future__ import annotations

from synesis.core.constants import (
    BENCHMARK_TICKERS,
    COLOR_HEADER,
    DAY_NAMES,
    DEDUP_CACHE_TTL_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE,
    SEC_13F_DEADLINES,
    SECTOR_COLORS,
    SECTOR_LABELS,
    SECTOR_TICKERS,
    SENTIMENT_ICON,
    SOURCE_LABEL,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    THEME_EMOJI,
)


class TestConstantsImportability:
    """All expected names are importable from synesis.core.constants."""

    def test_general_constants_importable(self) -> None:
        assert FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE == 60
        assert TELEGRAM_MAX_MESSAGE_LENGTH == 4096
        assert isinstance(DEFAULT_SIMILARITY_THRESHOLD, float)

    def test_display_constants_importable(self) -> None:
        assert isinstance(COLOR_HEADER, int)
        assert isinstance(SECTOR_COLORS, dict)
        assert isinstance(THEME_EMOJI, dict)
        assert isinstance(SOURCE_LABEL, dict)

    def test_events_constants_importable(self) -> None:
        assert isinstance(SEC_13F_DEADLINES, dict)

    def test_market_constants_importable(self) -> None:
        assert isinstance(BENCHMARK_TICKERS, list)
        assert isinstance(SECTOR_TICKERS, list)
        assert isinstance(SECTOR_LABELS, dict)
        assert isinstance(DAY_NAMES, list)


class TestSentimentIconValues:
    """Guard SENTIMENT_ICON emoji values used by discord.py sentiment_labels derivation."""

    def test_bullish_icon(self) -> None:
        assert SENTIMENT_ICON["bullish"] == "\U0001f7e2"  # green circle

    def test_bearish_icon(self) -> None:
        assert SENTIMENT_ICON["bearish"] == "\U0001f534"  # red circle

    def test_neutral_icon(self) -> None:
        assert SENTIMENT_ICON["neutral"] == "\u26aa"  # white circle

    def test_all_keys_present(self) -> None:
        assert set(SENTIMENT_ICON.keys()) == {"bullish", "bearish", "neutral"}


class TestDedupCacheTTL:
    """Prevent silent value drift of DEDUP_CACHE_TTL_SECONDS."""

    def test_value_is_one_hour(self) -> None:
        assert DEDUP_CACHE_TTL_SECONDS == 3600
