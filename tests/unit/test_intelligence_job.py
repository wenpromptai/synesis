"""Tests for the intelligence pipeline job runner."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_brief(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid brief dict."""
    base: dict[str, Any] = {
        "date": "2026-04-08",
        "macro": {"regime": "risk_on"},
        "debates": [],
        "l1_summary": {"social": "", "news": ""},
        "tickers_analyzed": ["NVDA"],
        "trade_ideas": [{"tickers": ["NVDA"], "trade_structure": "buy NVDA"}],
        "errors": {
            "social_failed": False,
            "news_failed": False,
            "company_failures": [],
            "price_failures": [],
            "bull_failures": [],
            "bear_failures": [],
            "macro_failed": False,
            "trader_failures": [],
        },
    }
    base.update(overrides)
    return base


def _make_graph_result(brief: dict[str, Any]) -> dict[str, Any]:
    return {"brief": brief}


class TestRunIntelligenceBrief:
    @pytest.fixture()
    def mock_deps(self):
        return {
            "db": MagicMock(),
            "sec_edgar": MagicMock(),
            "yfinance": MagicMock(),
        }

    @patch("synesis.processing.intelligence.job._save_brief_to_kg")
    @patch("synesis.processing.intelligence.job.send_discord", new_callable=AsyncMock)
    @patch("synesis.processing.intelligence.job.build_intelligence_graph")
    async def test_sends_discord_batches(
        self,
        mock_build: MagicMock,
        mock_send: AsyncMock,
        _mock_kg: MagicMock,
        mock_deps: dict[str, Any],
    ) -> None:
        brief = _make_brief()
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = _make_graph_result(brief)
        mock_build.return_value = mock_graph

        from synesis.processing.intelligence.job import run_intelligence_brief

        result = await run_intelligence_brief(**mock_deps)
        assert result["date"] == "2026-04-08"
        assert mock_send.called

    @patch("synesis.processing.intelligence.job._save_brief_to_kg")
    @patch("synesis.processing.intelligence.job.send_discord", new_callable=AsyncMock)
    @patch("synesis.processing.intelligence.job.build_intelligence_graph")
    async def test_empty_brief_skips_discord(
        self,
        mock_build: MagicMock,
        mock_send: AsyncMock,
        _mock_kg: MagicMock,
        mock_deps: dict[str, Any],
    ) -> None:
        """When brief has no date (compiler didn't run), skip Discord send."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"brief": {}}
        mock_build.return_value = mock_graph

        from synesis.processing.intelligence.job import run_intelligence_brief

        result = await run_intelligence_brief(**mock_deps)
        assert result == {}
        mock_send.assert_not_called()

    @patch("synesis.processing.intelligence.job._save_brief_to_kg")
    @patch("synesis.processing.intelligence.job.send_discord", new_callable=AsyncMock)
    @patch("synesis.processing.intelligence.job.build_intelligence_graph")
    async def test_batch_isolation_on_failure(
        self,
        mock_build: MagicMock,
        mock_send: AsyncMock,
        _mock_kg: MagicMock,
        mock_deps: dict[str, Any],
    ) -> None:
        """If one Discord batch fails, remaining batches should still send."""
        brief = _make_brief()
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = _make_graph_result(brief)
        mock_build.return_value = mock_graph

        # First call raises, second succeeds
        mock_send.side_effect = [Exception("webhook down"), True]

        from synesis.processing.intelligence.job import run_intelligence_brief

        # Patch formatter to return 2 batches
        with patch(
            "synesis.processing.intelligence.job.format_intelligence_brief",
            return_value=[[{"title": "batch1"}], [{"title": "batch2"}]],
        ):
            result = await run_intelligence_brief(**mock_deps)

        assert result["date"] == "2026-04-08"
        assert mock_send.call_count == 2
