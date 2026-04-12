"""Tests for trade idea tracking and direction parsing."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from synesis.processing.intelligence.job import _parse_direction
from synesis.processing.intelligence.tracking import run_tracking_review


class _FakeRecord(dict):
    """Mimics asyncpg.Record."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


def _make_idea(
    *,
    idea_id: int = 1,
    ticker: str = "NVDA",
    direction: str = "long",
    brief_date: date | None = None,
    entry_price: float | None = 135.0,
    target_price: float | None = 165.0,
    stop_price: float | None = 120.0,
    price_at_1w: float | None = None,
    price_at_2w: float | None = None,
    price_at_1m: float | None = None,
    days_ago: int = 3,
) -> _FakeRecord:
    """Helper to build a fake open trade idea record."""
    return _FakeRecord(
        id=idea_id,
        ticker=ticker,
        direction=direction,
        trade_structure=f"{direction} {ticker}",
        brief_date=(brief_date or datetime.now(UTC).date() - timedelta(days=days_ago)),
        entry_price=entry_price,
        target_price=target_price,
        stop_price=stop_price,
        conviction_tier=2,
        price_at_1w=price_at_1w,
        price_at_2w=price_at_2w,
        price_at_1m=price_at_1m,
    )


def _make_quote(last: float | None = 150.0) -> MagicMock:
    quote = MagicMock()
    quote.last = last
    return quote


class TestParseDirection:
    """Tests for _parse_direction helper."""

    def test_long_explicit(self) -> None:
        assert _parse_direction("long NVDA") == "long"

    def test_short_explicit(self) -> None:
        assert _parse_direction("short AMD") == "short"

    def test_short_case_insensitive(self) -> None:
        assert _parse_direction("Short TSLA") == "short"

    def test_long_is_default(self) -> None:
        assert _parse_direction("NVDA") == "long"

    def test_empty_string_defaults_to_long(self) -> None:
        # Should not happen in practice (min_length=1) but defensive
        assert _parse_direction("") == "long"

    def test_whitespace_trimmed(self) -> None:
        assert _parse_direction("  short AMD  ") == "short"


class TestTrackingReviewNoOpenIdeas:
    """Edge case: no open ideas returns early."""

    @pytest.mark.anyio
    async def test_returns_zeros(self) -> None:
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = []
        yf = AsyncMock()

        result = await run_tracking_review(db, yf)

        assert result == {"reviewed": 0, "closed": 0, "updated": 0}
        yf.get_quote.assert_not_called()


class TestTrackingReviewLongTarget:
    """Long position hits target."""

    @pytest.mark.anyio
    async def test_long_target_hit(self) -> None:
        idea = _make_idea(entry_price=100.0, target_price=120.0, stop_price=90.0)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=125.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        call_args = db.update_trade_tracking.call_args
        updates = call_args[0][1]
        assert updates["status"] == "hit_target"
        assert updates["pnl_at_close_pct"] == pytest.approx(25.0)
        assert updates["close_reason"] == "Target hit"


class TestTrackingReviewLongStop:
    """Long position hits stop."""

    @pytest.mark.anyio
    async def test_long_stop_hit(self) -> None:
        idea = _make_idea(entry_price=100.0, target_price=120.0, stop_price=90.0)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=85.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "hit_stop"
        assert updates["pnl_at_close_pct"] == pytest.approx(-15.0)
        assert updates["close_reason"] == "Stop hit"


class TestTrackingReviewShort:
    """Short position target/stop detection and P&L math."""

    @pytest.mark.anyio
    async def test_short_target_hit(self) -> None:
        idea = _make_idea(
            direction="short",
            entry_price=150.0,
            target_price=120.0,
            stop_price=165.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=115.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "hit_target"
        # Short P&L: (entry - current) / entry * 100 = (150 - 115) / 150 * 100
        assert updates["pnl_at_close_pct"] == pytest.approx(23.333, rel=1e-2)

    @pytest.mark.anyio
    async def test_short_stop_hit(self) -> None:
        idea = _make_idea(
            direction="short",
            entry_price=150.0,
            target_price=120.0,
            stop_price=165.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=170.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "hit_stop"
        # Short stop P&L: (entry - current) / entry * 100 = (150 - 170) / 150 * 100
        assert updates["pnl_at_close_pct"] == pytest.approx(-13.333, rel=1e-2)


class TestTrackingReviewCheckpoints:
    """Price checkpoint updates at 7, 14, 30 days."""

    @pytest.mark.anyio
    async def test_1w_checkpoint_written(self) -> None:
        idea = _make_idea(days_ago=8, entry_price=None, target_price=None, stop_price=None)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=140.0)

        await run_tracking_review(db, yf)

        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["price_at_1w"] == 140.0
        assert "price_at_2w" not in updates

    @pytest.mark.anyio
    async def test_2w_checkpoint_written(self) -> None:
        idea = _make_idea(
            days_ago=15,
            entry_price=None,
            target_price=None,
            stop_price=None,
            price_at_1w=138.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=142.0)

        await run_tracking_review(db, yf)

        updates = db.update_trade_tracking.call_args[0][1]
        assert "price_at_1w" not in updates  # already set
        assert updates["price_at_2w"] == 142.0

    @pytest.mark.anyio
    async def test_1m_checkpoint_written(self) -> None:
        idea = _make_idea(
            days_ago=31,
            entry_price=None,
            target_price=None,
            stop_price=None,
            price_at_1w=138.0,
            price_at_2w=140.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=145.0)

        await run_tracking_review(db, yf)

        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["price_at_1m"] == 145.0

    @pytest.mark.anyio
    async def test_checkpoint_not_overwritten(self) -> None:
        """Already-written checkpoints are never overwritten."""
        idea = _make_idea(
            days_ago=31,
            entry_price=None,
            target_price=None,
            stop_price=None,
            price_at_1w=138.0,
            price_at_2w=140.0,
            price_at_1m=143.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=150.0)

        await run_tracking_review(db, yf)

        # No updates needed — nothing to write
        db.update_trade_tracking.assert_not_called()


class TestTrackingReviewExpiry:
    """Auto-expiry after 90 days."""

    @pytest.mark.anyio
    async def test_long_auto_expired(self) -> None:
        idea = _make_idea(
            days_ago=91,
            entry_price=100.0,
            target_price=130.0,
            stop_price=85.0,
            price_at_1w=105.0,
            price_at_2w=108.0,
            price_at_1m=110.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=112.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "expired"
        # Long expiry P&L: (112 - 100) / 100 * 100 = 12%
        assert updates["pnl_at_close_pct"] == pytest.approx(12.0)

    @pytest.mark.anyio
    async def test_short_auto_expired(self) -> None:
        idea = _make_idea(
            days_ago=91,
            direction="short",
            entry_price=150.0,
            target_price=120.0,
            stop_price=165.0,
            price_at_1w=148.0,
            price_at_2w=147.0,
            price_at_1m=145.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=140.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "expired"
        # Short expiry P&L: (150 - 140) / 150 * 100 = 6.67%
        assert updates["pnl_at_close_pct"] == pytest.approx(6.667, rel=1e-2)

    @pytest.mark.anyio
    async def test_not_expired_at_exactly_90_days(self) -> None:
        """90 days is NOT expired — only > 90."""
        idea = _make_idea(
            days_ago=90,
            entry_price=None,
            target_price=None,
            stop_price=None,
            price_at_1w=105.0,
            price_at_2w=108.0,
            price_at_1m=110.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=115.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 0
        db.update_trade_tracking.assert_not_called()


class TestTrackingReviewErrorIsolation:
    """Quote failures and DB errors don't abort the entire review."""

    @pytest.mark.anyio
    async def test_quote_failure_skips_ticker(self) -> None:
        idea_fail = _make_idea(idea_id=1, ticker="BAD")
        idea_ok = _make_idea(
            idea_id=2, ticker="NVDA", entry_price=100.0, target_price=120.0, stop_price=90.0
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea_fail, idea_ok]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.side_effect = [RuntimeError("API down"), _make_quote(last=125.0)]

        result = await run_tracking_review(db, yf)

        # Only the second idea was reviewed
        assert result["reviewed"] == 1
        assert result["closed"] == 1

    @pytest.mark.anyio
    async def test_null_price_skips_ticker(self) -> None:
        idea = _make_idea()
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=None)

        result = await run_tracking_review(db, yf)

        assert result["reviewed"] == 0
        db.update_trade_tracking.assert_not_called()

    @pytest.mark.anyio
    async def test_db_update_failure_does_not_abort(self) -> None:
        idea1 = _make_idea(idea_id=1, ticker="NVDA", days_ago=8, entry_price=None)
        idea2 = _make_idea(idea_id=2, ticker="AMD", days_ago=8, entry_price=None)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea1, idea2]
        db.update_trade_tracking.side_effect = [RuntimeError("DB error"), None]
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=140.0)

        result = await run_tracking_review(db, yf)

        # Both reviewed, but only second succeeded
        assert result["reviewed"] == 2
        assert result["updated"] == 1


class TestTrackingReviewEdgeCases:
    """Edge cases: zero entry price, missing prices."""

    @pytest.mark.anyio
    async def test_zero_entry_price_skips_target_stop_check(self) -> None:
        """entry_price=0 should not cause ZeroDivisionError."""
        idea = _make_idea(entry_price=0.0, target_price=120.0, stop_price=90.0, days_ago=3)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=130.0)

        result = await run_tracking_review(db, yf)

        # Reviewed but no target/stop check, no updates
        assert result["reviewed"] == 1
        assert result["closed"] == 0
        db.update_trade_tracking.assert_not_called()

    @pytest.mark.anyio
    async def test_missing_prices_skips_target_stop(self) -> None:
        """Ideas without entry/target/stop still get checkpoint updates."""
        idea = _make_idea(entry_price=None, target_price=None, stop_price=None, days_ago=8)
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=140.0)

        result = await run_tracking_review(db, yf)

        assert result["updated"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert "status" not in updates
        assert updates["price_at_1w"] == 140.0

    @pytest.mark.anyio
    async def test_zero_entry_expiry_skips_pnl(self) -> None:
        """Auto-expiry with entry_price=0 should not compute P&L."""
        idea = _make_idea(
            entry_price=0.0,
            target_price=120.0,
            stop_price=90.0,
            days_ago=91,
            price_at_1w=105.0,
            price_at_2w=108.0,
            price_at_1m=110.0,
        )
        db = AsyncMock()
        db.get_open_trade_ideas.return_value = [idea]
        db.update_trade_tracking.return_value = None
        yf = AsyncMock()
        yf.get_quote.return_value = _make_quote(last=130.0)

        result = await run_tracking_review(db, yf)

        assert result["closed"] == 1
        updates = db.update_trade_tracking.call_args[0][1]
        assert updates["status"] == "expired"
        # P&L should NOT be present (would cause ZeroDivisionError)
        assert "pnl_at_close_pct" not in updates
