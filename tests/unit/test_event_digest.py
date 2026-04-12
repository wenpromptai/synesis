"""Tests for Event Radar digest (forward-looking calendar digest)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.events.digest import (
    _format_whats_coming_embeds,
    _get_13f_deadline_reminder,
    _split_content,
    send_event_digest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_row(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": 1,
        "title": "FOMC Rate Decision",
        "description": "Federal Reserve interest rate decision",
        "event_date": date(2026, 3, 19),
        "event_end_date": None,
        "category": "fed",
        "sector": None,
        "region": ["US"],
        "tickers": [],
        "source_urls": [],
    }
    defaults.update(overrides)
    return defaults


class _FakeRecord(dict):
    """Mimics asyncpg.Record."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _record(**kwargs: Any) -> _FakeRecord:
    return _FakeRecord(_make_event_row(**kwargs))


# ---------------------------------------------------------------------------
# Tests: _split_content
# ---------------------------------------------------------------------------


class TestSplitContent:
    def test_short_content_no_split(self) -> None:
        assert _split_content("hello", 100) == ["hello"]

    def test_long_content_splits(self) -> None:
        lines = [f"Line {i}" for i in range(100)]
        text = "\n".join(lines)
        chunks = _split_content(text, 200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_empty_string(self) -> None:
        assert _split_content("", 100) == [""]


# ---------------------------------------------------------------------------
# Tests: _get_13f_deadline_reminder
# ---------------------------------------------------------------------------


class TestGet13FDeadlineReminder:
    def test_deadline_within_window(self) -> None:
        # Q4 deadline is Feb 14 — test with today = Feb 1
        result = _get_13f_deadline_reminder(date(2026, 2, 1), 14)
        assert result is not None
        assert "13F Filing Deadline" in result
        assert "Feb 14" in result
        assert "Q4" in result
        assert "Berkshire Hathaway" in result

    def test_no_deadline_in_window(self) -> None:
        # No deadline near Apr 15
        result = _get_13f_deadline_reminder(date(2026, 4, 15), 14)
        assert result is None

    def test_deadline_exactly_on_boundary(self) -> None:
        # Q1 deadline is May 15 — test with today = May 1
        result = _get_13f_deadline_reminder(date(2026, 5, 1), 14)
        assert result is not None
        assert "May 15" in result


# ---------------------------------------------------------------------------
# Tests: _format_whats_coming_embeds
# ---------------------------------------------------------------------------


class TestFormatWhatsComingEmbeds:
    def test_groups_by_date(self) -> None:
        rows = [
            _record(id=1, event_date=date(2026, 3, 19), title="FOMC"),
            _record(id=2, event_date=date(2026, 3, 20), title="CPI Release"),
        ]
        messages = _format_whats_coming_embeds(
            rows, set(), None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )

        assert len(messages) >= 1
        desc = messages[0][0]["description"]
        assert "Mar 19" in desc
        assert "Mar 20" in desc
        assert "FOMC" in desc
        assert "CPI Release" in desc

    def test_new_badge_applied(self) -> None:
        rows = [_record(id=42, title="New Event")]
        messages = _format_whats_coming_embeds(
            rows, {42}, None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        desc = messages[0][0]["description"]
        assert "\U0001f195" in desc  # NEW emoji

    def test_deadline_reminder_included(self) -> None:
        rows = [_record()]
        reminder = "\u23f0 **13F Filing Deadline: Feb 14 (Q4 2025)**"
        messages = _format_whats_coming_embeds(
            rows, set(), reminder, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        desc = messages[0][0]["description"]
        assert "13F Filing Deadline" in desc

    def test_title_set_correctly(self) -> None:
        rows = [_record()]
        messages = _format_whats_coming_embeds(
            rows, set(), None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        assert "What's Coming" in messages[0][0]["title"]


# ---------------------------------------------------------------------------
# Tests: send_event_digest (orchestrator)
# ---------------------------------------------------------------------------


class TestSendEventDigest:
    @pytest.mark.asyncio
    async def test_returns_false_no_webhook(self) -> None:
        db = AsyncMock()
        with patch("synesis.processing.events.digest.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                discord_brief_webhook_url=None,
                discord_webhook_url=None,
            )
            result = await send_event_digest(db)

        assert result is False

    @pytest.mark.asyncio
    async def test_sends_whats_coming(self) -> None:
        db = AsyncMock()
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        with (
            patch("synesis.processing.events.digest.get_settings") as mock_settings,
            patch(
                "synesis.processing.events.digest._send_whats_coming", AsyncMock(return_value=True)
            ) as mock_coming,
        ):
            mock_settings.return_value = MagicMock(
                discord_brief_webhook_url=MagicMock(get_secret_value=lambda: "https://webhook"),
                discord_webhook_url=None,
            )
            result = await send_event_digest(db, redis=redis)

        assert result is True
        mock_coming.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_whats_coming_fails(self) -> None:
        db = AsyncMock()

        with (
            patch("synesis.processing.events.digest.get_settings") as mock_settings,
            patch(
                "synesis.processing.events.digest._send_whats_coming",
                AsyncMock(return_value=False),
            ),
        ):
            mock_settings.return_value = MagicMock(
                discord_brief_webhook_url=MagicMock(get_secret_value=lambda: "https://webhook"),
                discord_webhook_url=None,
            )
            result = await send_event_digest(db)

        assert result is False
