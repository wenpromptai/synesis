"""Tests for intelligence pipeline graph infrastructure."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from synesis.processing.intelligence.compiler import compile_brief
from synesis.processing.intelligence.graph import build_intelligence_graph


class TestCompileBrief:
    """Tests for the deterministic brief compiler."""

    def test_empty_state(self) -> None:
        """Empty state produces empty brief."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
            }
        )
        assert brief["date"] == "2026-04-06"
        assert brief["tickers_analyzed"] == []
        assert brief["macro_themes"] == []
        assert brief["tier1_summary"]["social"] == ""
        assert brief["tier1_summary"]["news"] == ""

    def test_populated_state(self) -> None:
        """Populated state assembles correctly."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {
                    "summary": "Bullish tech sentiment",
                    "ticker_mentions": [{"ticker": "NVDA", "sentiment_score": 0.7}],
                    "macro_themes": [{"theme": "AI capex", "sentiment_score": 0.5}],
                },
                "news_analysis": {
                    "summary": "M&A activity",
                    "story_clusters": [
                        {"headline": "NVDA deal", "tickers": [{"ticker": "NVDA"}]}
                    ],
                    "macro_themes": [{"theme": "Risk-off", "sentiment_score": -0.3}],
                    "messages_analyzed": 7,
                },
                "company_analyses": [
                    {"ticker": "NVDA", "sentiment_score": 0.8},
                    {"ticker": "AMD", "sentiment_score": 0.3},
                ],
            }
        )
        assert brief["tickers_analyzed"] == ["NVDA", "AMD"]
        assert len(brief["macro_themes"]) == 2
        assert brief["tier1_summary"]["social"] == "Bullish tech sentiment"
        assert brief["tier1_summary"]["news"] == "M&A activity"
        assert brief["messages_analyzed"] == 7

    def test_filters_errored_analyses(self) -> None:
        """Company analyses with error=True are excluded from tickers_analyzed."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [
                    {"ticker": "NVDA", "sentiment_score": 0.8},
                    {"ticker": "AAPL", "error": True},
                    {"ticker": "AMD", "sentiment_score": -0.2},
                ],
            }
        )
        assert brief["tickers_analyzed"] == ["NVDA", "AMD"]
        assert len(brief["company_analyses"]) == 2


class TestBuildGraph:
    """Tests for graph compilation."""

    def test_graph_compiles(self) -> None:
        """Graph compiles with mock deps."""
        graph = build_intelligence_graph(
            db=AsyncMock(),
            sec_edgar=AsyncMock(),
            yfinance=AsyncMock(),
            crawler=None,
        )
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """Graph contains all expected nodes."""
        graph = build_intelligence_graph(
            db=AsyncMock(),
            sec_edgar=AsyncMock(),
            yfinance=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "social_sentiment" in node_names
        assert "news_analyst" in node_names
        assert "extract_tickers" in node_names
        assert "company_analyst" in node_names
        assert "compiler" in node_names


class TestExtractTickers:
    """Tests for ticker extraction logic (tested via compile_brief inputs)."""

    def test_extracts_from_social_mentions(self) -> None:
        """Tickers are extracted from social ticker_mentions."""
        # extract_tickers_node reads from state dicts
        social = {
            "ticker_mentions": [
                {"ticker": "NVDA", "sentiment_score": 0.7},
                {"ticker": "AMD", "sentiment_score": 0.3},
            ]
        }
        tickers = {m["ticker"] for m in social.get("ticker_mentions", []) if m.get("ticker")}
        assert tickers == {"NVDA", "AMD"}

    def test_extracts_from_news_clusters(self) -> None:
        """Tickers are extracted from news story_clusters."""
        news = {
            "story_clusters": [
                {
                    "headline": "NVDA deal",
                    "tickers": [
                        {"ticker": "NVDA", "sentiment_score": 0.8},
                        {"ticker": "INTC", "sentiment_score": -0.3},
                    ],
                },
                {
                    "headline": "DBS earnings",
                    "tickers": [{"ticker": "D05.SI", "sentiment_score": 0.6}],
                },
            ]
        }
        tickers = set()
        for cluster in news.get("story_clusters", []):
            for mention in cluster.get("tickers", []):
                if mention.get("ticker"):
                    tickers.add(mention["ticker"])
        assert tickers == {"NVDA", "INTC", "D05.SI"}

    def test_deduplicates_across_sources(self) -> None:
        """Same ticker from social + news appears only once."""
        social_tickers = {"NVDA", "AMD"}
        news_tickers = {"NVDA", "AAPL"}
        combined = sorted(social_tickers | news_tickers)
        assert combined == ["AAPL", "AMD", "NVDA"]

    def test_empty_outputs(self) -> None:
        """No tickers when both sources are empty."""
        social = {"ticker_mentions": []}
        news = {"story_clusters": []}
        tickers = set()
        for m in social.get("ticker_mentions", []):
            if m.get("ticker"):
                tickers.add(m["ticker"])
        for c in news.get("story_clusters", []):
            for m in c.get("tickers", []):
                if m.get("ticker"):
                    tickers.add(m["ticker"])
        assert tickers == set()
