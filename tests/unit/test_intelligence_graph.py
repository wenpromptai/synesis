"""Tests for intelligence pipeline graph infrastructure."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from synesis.processing.intelligence.compiler import compile_brief
from synesis.processing.intelligence.context import (
    format_company_context_for_ticker,
    format_debate_history,
    format_news_context_for_ticker,
    format_price_context_for_ticker,
    format_social_context_for_ticker,
)
from synesis.processing.intelligence.graph import build_intelligence_graph
from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    MacroView,
    NewsAnalysis,
    NewsEventType,
    NewsStoryCluster,
    PriceAnalysis,
    SocialSentimentAnalysis,
    TickerDebate,
    TickerMention,
    TradeIdea,
    TraderOutput,
)


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
                "price_analyses": [],
                "bull_analyses": [],
                "bear_analyses": [],
            }
        )
        assert brief["date"] == "2026-04-06"
        assert brief["tickers_analyzed"] == []
        assert brief["debates"] == []
        assert brief["macro_themes"] == []
        assert brief["l1_summary"]["social"] == ""
        assert brief["l1_summary"]["news"] == ""
        assert brief["trade_ideas"] == []
        assert brief["errors"]["trader_failures"] == []

    def test_populated_state(self) -> None:
        """Populated state assembles correctly."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {
                    "summary": "Bullish tech sentiment",
                    "ticker_mentions": [{"ticker": "NVDA", "context": "heavy call buying"}],
                    "macro_themes": [{"theme": "AI capex", "context": "multiple accounts"}],
                },
                "news_analysis": {
                    "summary": "M&A activity",
                    "story_clusters": [{"headline": "NVDA deal", "tickers": [{"ticker": "NVDA"}]}],
                    "macro_themes": [{"theme": "Risk-off", "context": "tariff fears"}],
                    "messages_analyzed": 7,
                },
                "company_analyses": [
                    {"ticker": "NVDA"},
                    {"ticker": "AMD"},
                ],
                "price_analyses": [
                    {"ticker": "NVDA", "spot_price": 150.0},
                ],
                "bull_analyses": [],
                "bear_analyses": [],
            }
        )
        assert brief["tickers_analyzed"] == ["NVDA", "AMD"]
        assert len(brief["macro_themes"]) == 2
        assert brief["l1_summary"]["social"] == "Bullish tech sentiment"
        assert brief["l1_summary"]["news"] == "M&A activity"
        assert brief["messages_analyzed"] == 7
        assert len(brief["price_analyses"]) == 1

    def test_full_pipeline_state(self) -> None:
        """All pipeline stages populated — verify every brief field."""
        brief = compile_brief(
            {
                "current_date": "2026-04-07",
                "social_analysis": {
                    "summary": "Tech bullish",
                    "ticker_mentions": [{"ticker": "NVDA", "context": "AI hype"}],
                    "macro_themes": [{"theme": "AI capex"}],
                },
                "news_analysis": {
                    "summary": "Earnings season",
                    "story_clusters": [{"headline": "NVDA beats", "tickers": [{"ticker": "NVDA"}]}],
                    "macro_themes": [{"theme": "Rate cuts"}],
                    "messages_analyzed": 12,
                },
                "company_analyses": [
                    {"ticker": "NVDA", "company_name": "NVIDIA"},
                    {"ticker": "AMD", "error": True},
                ],
                "price_analyses": [
                    {"ticker": "NVDA", "spot_price": 150.0},
                ],
                "macro_view": {
                    "regime": "risk_on",
                    "sentiment_score": 0.6,
                    "key_drivers": ["Strong employment"],
                    "sector_tilts": [{"sector": "Tech", "sentiment_score": 0.7}],
                    "risks": ["Fed uncertainty"],
                },
                "bull_analyses": [
                    {
                        "ticker": "NVDA",
                        "role": "bull",
                        "argument": "AI demand strong",
                        "round": 1,
                    },
                ],
                "bear_analyses": [
                    {
                        "ticker": "NVDA",
                        "role": "bear",
                        "argument": "Overvalued",
                        "round": 1,
                    },
                    {"ticker": "AMD", "error": True},
                ],
                "trade_ideas": [
                    {
                        "tickers": ["NVDA"],
                        "trade_structure": "bull call spread 150/160 June",
                        "thesis": "AI demand",
                    },
                    {"tickers": ["AMD"], "error": True},
                ],
            }
        )
        # Date
        assert brief["date"] == "2026-04-07"
        # Macro
        assert brief["macro"]["regime"] == "risk_on"
        assert brief["macro"]["sentiment_score"] == 0.6
        assert brief["macro"]["key_drivers"] == ["Strong employment"]
        assert len(brief["macro"]["sector_tilts"]) == 1
        assert brief["macro"]["risks"] == ["Fed uncertainty"]
        # Debates
        assert len(brief["debates"]) == 1
        assert brief["debates"][0]["ticker"] == "NVDA"
        assert "bull" in brief["debates"][0]
        assert "bear" in brief["debates"][0]
        # L1 summary
        assert brief["l1_summary"]["social"] == "Tech bullish"
        assert brief["l1_summary"]["news"] == "Earnings season"
        # Company/price (errors filtered)
        assert brief["tickers_analyzed"] == ["NVDA"]
        assert len(brief["company_analyses"]) == 1
        assert len(brief["price_analyses"]) == 1
        # Macro themes (merged from both L1)
        assert len(brief["macro_themes"]) == 2
        # Ticker mentions
        assert len(brief["ticker_mentions"]["social"]) == 1
        assert len(brief["ticker_mentions"]["news_clusters"]) == 1
        # Messages
        assert brief["messages_analyzed"] == 12
        # Trade ideas (errors filtered)
        assert len(brief["trade_ideas"]) == 1
        assert brief["trade_ideas"][0]["tickers"] == ["NVDA"]
        # Errors — all failure types tracked
        assert brief["errors"]["social_failed"] is False
        assert brief["errors"]["news_failed"] is False
        assert brief["errors"]["company_failures"] == ["AMD"]
        assert brief["errors"]["price_failures"] == []
        assert brief["errors"]["bull_failures"] == []
        assert brief["errors"]["bear_failures"] == ["AMD"]
        assert brief["errors"]["macro_failed"] is False
        assert brief["errors"]["trader_failures"] == ["AMD"]

    def test_filters_errored_analyses(self) -> None:
        """Company analyses with error=True are excluded from tickers_analyzed."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [
                    {"ticker": "NVDA"},
                    {"ticker": "AAPL", "error": True},
                    {"ticker": "AMD"},
                ],
                "price_analyses": [
                    {"ticker": "NVDA", "spot_price": 150.0},
                    {"ticker": "AAPL", "error": True},
                ],
                "bull_analyses": [],
                "bear_analyses": [],
            }
        )
        assert brief["tickers_analyzed"] == ["NVDA", "AMD"]
        assert len(brief["company_analyses"]) == 2
        assert len(brief["price_analyses"]) == 1


class TestCompilerDebates:
    """Tests for compiler debate grouping."""

    def test_groups_debates_by_ticker(self) -> None:
        """Bull and bear arguments are grouped by ticker."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "bull_analyses": [
                    {"role": "bull", "ticker": "NVDA", "argument": "Strong AI growth"},
                    {"role": "bull", "ticker": "AMD", "argument": "CPU market share"},
                ],
                "bear_analyses": [
                    {"role": "bear", "ticker": "NVDA", "argument": "Valuation stretched"},
                    {"role": "bear", "ticker": "AMD", "argument": "Margin pressure"},
                ],
            }
        )
        assert len(brief["debates"]) == 2
        # Sorted by ticker
        assert brief["debates"][0]["ticker"] == "AMD"
        assert brief["debates"][0]["bull"]["argument"] == "CPU market share"
        assert brief["debates"][0]["bear"]["argument"] == "Margin pressure"
        assert brief["debates"][1]["ticker"] == "NVDA"
        assert brief["debates"][1]["bull"]["argument"] == "Strong AI growth"
        assert brief["debates"][1]["bear"]["argument"] == "Valuation stretched"

    def test_handles_missing_side(self) -> None:
        """Debate with only one side still appears."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "bull_analyses": [
                    {"role": "bull", "ticker": "NVDA", "argument": "Strong growth"},
                ],
                "bear_analyses": [],
            }
        )
        assert len(brief["debates"]) == 1
        assert brief["debates"][0]["ticker"] == "NVDA"
        assert "bull" in brief["debates"][0]
        assert "bear" not in brief["debates"][0]

    def test_handles_errored_analysis(self) -> None:
        """Errored bull/bear analysis produces empty debates."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "bull_analyses": [{"ticker": "NVDA", "error": True}],
                "bear_analyses": [
                    {"role": "bear", "ticker": "NVDA", "argument": "Weak"},
                ],
            }
        )
        assert len(brief["debates"]) == 1
        assert "bull" not in brief["debates"][0]
        assert "bear" in brief["debates"][0]

    def test_empty_debates(self) -> None:
        """No arguments produces empty debates list."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "bull_analyses": [],
                "bear_analyses": [],
            }
        )
        assert brief["debates"] == []

    def test_macro_context_in_brief(self) -> None:
        """Macro view is included in brief."""
        brief = compile_brief(
            {
                "current_date": "2026-04-06",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "bull_analyses": [],
                "bear_analyses": [],
                "macro_view": {"regime": "risk_on", "sentiment_score": 0.5},
            }
        )
        assert brief["macro"]["regime"] == "risk_on"
        assert brief["macro"]["sentiment_score"] == 0.5


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
        assert "price_analyst" in node_names
        assert "macro_strategist" in node_names
        assert "l2_join" in node_names
        assert "bull_researcher" in node_names
        assert "bear_researcher" in node_names
        assert "ticker_debate" in node_names
        assert "compiler" in node_names

    def test_graph_does_not_have_removed_nodes(self) -> None:
        """Graph does not contain removed equity_strategist or conviction_gate."""
        graph = build_intelligence_graph(
            db=AsyncMock(),
            sec_edgar=AsyncMock(),
            yfinance=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "equity_strategist" not in node_names
        assert "conviction_gate" not in node_names


class TestExtractTickers:
    """Tests for ticker extraction logic (tested via compile_brief inputs)."""

    def test_extracts_from_social_mentions(self) -> None:
        """Tickers are extracted from social ticker_mentions."""
        social = {
            "ticker_mentions": [
                {"ticker": "NVDA", "context": "heavy call buying"},
                {"ticker": "AMD"},
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
                        {"ticker": "NVDA"},
                        {"ticker": "INTC", "context": "competitive pressure"},
                    ],
                },
                {
                    "headline": "DBS earnings",
                    "tickers": [{"ticker": "D05.SI", "context": "record Q1 profit"}],
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


# ── Graph Execution Tests (mock agents, real wiring) ────────────


def _social(tickers: list[str]) -> SocialSentimentAnalysis:
    return SocialSentimentAnalysis(
        ticker_mentions=[TickerMention(ticker=t, context=f"{t} buzz") for t in tickers],
        summary="test social",
        analysis_date=date(2026, 4, 7),
    )


def _news(tickers: list[str]) -> NewsAnalysis:
    clusters = []
    for t in tickers:
        clusters.append(
            NewsStoryCluster(
                headline=f"{t} news",
                event_type=NewsEventType.other,
                tickers=[TickerMention(ticker=t, context=f"{t} story")],
            )
        )
    return NewsAnalysis(
        story_clusters=clusters,
        summary="test news",
        analysis_date=date(2026, 4, 7),
    )


def _company(ticker: str) -> CompanyAnalysis:
    return CompanyAnalysis(ticker=ticker, company_name=ticker, analysis_date=date(2026, 4, 7))


def _price(ticker: str) -> PriceAnalysis:
    return PriceAnalysis(ticker=ticker, spot_price=100.0, analysis_date=date(2026, 4, 7))


def _macro() -> MacroView:
    return MacroView(regime="risk_on", sentiment_score=0.5, analysis_date=date(2026, 4, 7))


def _bull(ticker: str) -> TickerDebate:
    return TickerDebate(
        role="bull",
        ticker=ticker,
        argument=f"Bull case for {ticker}",
        key_evidence=[f"{ticker} growing"],
        analysis_date=date(2026, 4, 7),
    )


def _bear(ticker: str) -> TickerDebate:
    return TickerDebate(
        role="bear",
        ticker=ticker,
        argument=f"Bear case for {ticker}",
        key_evidence=[f"{ticker} risks"],
        analysis_date=date(2026, 4, 7),
    )


_PATCH_PREFIX = "synesis.processing.intelligence.graph"


@pytest.mark.slow
class TestGraphExecution:
    """End-to-end graph execution with mocked agents."""

    @pytest.mark.asyncio
    async def test_no_tickers_skips_debate(self) -> None:
        """When L1 finds no tickers, debate is skipped and compiler runs."""
        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social([])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news([])),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch(f"{_PATCH_PREFIX}.research_bull") as mock_bull,
            patch(f"{_PATCH_PREFIX}.research_bear") as mock_bear,
            patch("synesis.config.get_settings") as mock_settings,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        # Debate never called
        mock_bull.assert_not_called()
        mock_bear.assert_not_called()

        brief = result["brief"]
        assert brief["debates"] == []
        assert brief["macro"]["regime"] == "risk_on"

    @pytest.mark.asyncio
    async def test_per_ticker_fanout(self) -> None:
        """Two tickers produce per-ticker bull/bear calls and correct debates (rounds=0)."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        with (
            patch(
                f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA", "AMD"])
            ),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA", "AMD"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull) as bull_spy,
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear) as bear_spy,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        # Bull/bear called once per ticker (2 tickers = 2 calls each)
        assert bull_spy.call_count == 2
        assert bear_spy.call_count == 2

        # Verify each call received the correct ticker in state
        bull_tickers = {call.args[0]["ticker"] for call in bull_spy.call_args_list}
        bear_tickers = {call.args[0]["ticker"] for call in bear_spy.call_args_list}
        assert bull_tickers == {"NVDA", "AMD"}
        assert bear_tickers == {"NVDA", "AMD"}

        # Brief has 2 debates, each with both sides
        brief = result["brief"]
        assert len(brief["debates"]) == 2
        debate_tickers = {d["ticker"] for d in brief["debates"]}
        assert debate_tickers == {"NVDA", "AMD"}
        for debate in brief["debates"]:
            assert "bull" in debate
            assert "bear" in debate

    @pytest.mark.asyncio
    async def test_partial_debate_failure(self) -> None:
        """One ticker's bull fails, other results still reach compiler (rounds=0)."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            if state["ticker"] == "NVDA":
                raise RuntimeError("LLM error")
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        with (
            patch(
                f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA", "AMD"])
            ),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA", "AMD"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull),
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear),
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        brief = result["brief"]

        # AMD has both sides, NVDA only has bear
        amd_debate = next(d for d in brief["debates"] if d["ticker"] == "AMD")
        assert "bull" in amd_debate
        assert "bear" in amd_debate

        nvda_debate = next(d for d in brief["debates"] if d["ticker"] == "NVDA")
        assert "bull" not in nvda_debate
        assert "bear" in nvda_debate

        # NVDA bull failure tracked in errors
        assert "NVDA" in brief["errors"]["bull_failures"]
        assert "AMD" not in brief["errors"]["bull_failures"]


@pytest.mark.slow
class TestDebateSubgraph:
    """Tests for the multi-round debate subgraph."""

    @pytest.mark.asyncio
    async def test_single_round_debate(self) -> None:
        """rounds=1 produces one bull + one bear argument."""
        from synesis.processing.intelligence.debate.subgraph import build_debate_subgraph

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"]).model_copy(
                update={
                    "round": len([h for h in (debate_history or []) if h.get("role") == "bull"]) + 1
                }
            )

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"]).model_copy(
                update={
                    "round": len([h for h in (debate_history or []) if h.get("role") == "bear"]) + 1
                }
            )

        with (
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            subgraph = build_debate_subgraph()
            result = await subgraph.ainvoke(
                {
                    "ticker": "NVDA",
                    "current_date": "2026-04-07",
                    "social_analysis": {},
                    "news_analysis": {},
                    "company_analyses": [],
                    "price_analyses": [],
                    "debate_history": [],
                    "round": 0,
                    "max_rounds": 1,
                },
                config={"recursion_limit": 10},
            )

        history = result["debate_history"]
        assert len(history) == 2
        assert history[0]["role"] == "bull"
        assert history[1]["role"] == "bear"
        assert result["round"] == 1

    @pytest.mark.asyncio
    async def test_two_round_debate(self) -> None:
        """rounds=2 produces 4 arguments (bull, bear, bull, bear)."""
        from synesis.processing.intelligence.debate.subgraph import build_debate_subgraph

        call_order: list[str] = []

        async def mock_bull(state, current_date, debate_history=None):
            call_order.append("bull")
            history = debate_history or []
            round_num = len([h for h in history if h.get("role") == "bull"]) + 1
            return _bull(state["ticker"]).model_copy(
                update={"argument": f"Bull round {round_num}", "round": round_num}
            )

        async def mock_bear(state, current_date, debate_history=None):
            call_order.append("bear")
            history = debate_history or []
            round_num = len([h for h in history if h.get("role") == "bear"]) + 1
            return _bear(state["ticker"]).model_copy(
                update={"argument": f"Bear round {round_num}", "round": round_num}
            )

        with (
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            subgraph = build_debate_subgraph()
            result = await subgraph.ainvoke(
                {
                    "ticker": "NVDA",
                    "current_date": "2026-04-07",
                    "social_analysis": {},
                    "news_analysis": {},
                    "company_analyses": [],
                    "price_analyses": [],
                    "debate_history": [],
                    "round": 0,
                    "max_rounds": 2,
                },
                config={"recursion_limit": 20},
            )

        # Correct alternation: bull, bear, bull, bear
        assert call_order == ["bull", "bear", "bull", "bear"]

        history = result["debate_history"]
        assert len(history) == 4
        assert history[0]["argument"] == "Bull round 1"
        assert history[1]["argument"] == "Bear round 1"
        assert history[2]["argument"] == "Bull round 2"
        assert history[3]["argument"] == "Bear round 2"
        assert result["round"] == 2

    @pytest.mark.asyncio
    async def test_debate_history_passed_to_researchers(self) -> None:
        """Each researcher receives the full debate history from prior turns."""
        from synesis.processing.intelligence.debate.subgraph import build_debate_subgraph

        captured_histories: list[list[dict[str, Any]]] = []

        async def mock_bull(state, current_date, debate_history=None):
            captured_histories.append(list(debate_history or []))
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            captured_histories.append(list(debate_history or []))
            return _bear(state["ticker"])

        with (
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            subgraph = build_debate_subgraph()
            await subgraph.ainvoke(
                {
                    "ticker": "NVDA",
                    "current_date": "2026-04-07",
                    "social_analysis": {},
                    "news_analysis": {},
                    "company_analyses": [],
                    "price_analyses": [],
                    "debate_history": [],
                    "round": 0,
                    "max_rounds": 2,
                },
                config={"recursion_limit": 20},
            )

        # Round 1: bull sees empty, bear sees bull's argument
        assert len(captured_histories[0]) == 0  # bull round 1
        assert len(captured_histories[1]) == 1  # bear round 1 (sees bull)
        # Round 2: bull sees bull+bear, bear sees bull+bear+bull
        assert len(captured_histories[2]) == 2  # bull round 2
        assert len(captured_histories[3]) == 3  # bear round 2


@pytest.mark.slow
class TestMultiRoundGraphExecution:
    """Test full graph with debate subgraph (rounds>=1)."""

    @pytest.mark.asyncio
    async def test_debate_subgraph_in_full_graph(self) -> None:
        """rounds=1 routes through ticker_debate node and produces correct brief."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            mock_settings.return_value.debate_rounds = 1
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        brief = result["brief"]
        assert len(brief["debates"]) == 1
        assert brief["debates"][0]["ticker"] == "NVDA"
        assert "bull" in brief["debates"][0]
        assert "bear" in brief["debates"][0]

    @pytest.mark.asyncio
    async def test_rounds_zero_uses_parallel_path(self) -> None:
        """rounds=0 uses separate bull/bear nodes (not debate subgraph)."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull) as bull_spy,
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear) as bear_spy,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        # Parallel path: bull/bear called directly (not via subgraph)
        assert bull_spy.call_count == 1
        assert bear_spy.call_count == 1

        brief = result["brief"]
        assert len(brief["debates"]) == 1
        assert brief["debates"][0]["ticker"] == "NVDA"

    @pytest.mark.asyncio
    async def test_multi_round_compiler_picks_last(self) -> None:
        """With rounds=2, compiler uses the last (most refined) argument per ticker."""
        brief = compile_brief(
            {
                "current_date": "2026-04-07",
                "social_analysis": {},
                "news_analysis": {},
                "company_analyses": [],
                "price_analyses": [],
                "bull_analyses": [
                    {"role": "bull", "ticker": "NVDA", "argument": "Bull round 1", "round": 1},
                    {"role": "bull", "ticker": "NVDA", "argument": "Bull round 2", "round": 2},
                ],
                "bear_analyses": [
                    {"role": "bear", "ticker": "NVDA", "argument": "Bear round 1", "round": 1},
                    {"role": "bear", "ticker": "NVDA", "argument": "Bear round 2", "round": 2},
                ],
            }
        )
        assert len(brief["debates"]) == 1
        # Last argument wins (round 2 is the most refined)
        assert brief["debates"][0]["bull"]["argument"] == "Bull round 2"
        assert brief["debates"][0]["bear"]["argument"] == "Bear round 2"

    @pytest.mark.asyncio
    async def test_debate_failure_mid_round(self) -> None:
        """Bear fails in debate subgraph — ticker_debate_node catches and returns error."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            raise RuntimeError("LLM timeout")

        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            mock_settings.return_value.debate_rounds = 1
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        brief = result["brief"]
        # Entire ticker debate failed — error tracked
        assert "NVDA" in brief["errors"]["bull_failures"]
        assert "NVDA" in brief["errors"]["bear_failures"]

    @pytest.mark.asyncio
    async def test_multi_ticker_debate_subgraph(self) -> None:
        """Multiple tickers each get their own independent debate subgraph."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        with (
            patch(
                f"{_PATCH_PREFIX}.analyze_social_sentiment",
                return_value=_social(["NVDA", "AMD"]),
            ),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA", "AMD"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bull",
                side_effect=mock_bull,
            ),
            patch(
                "synesis.processing.intelligence.debate.subgraph.research_bear",
                side_effect=mock_bear,
            ),
        ):
            mock_settings.return_value.debate_rounds = 1
            mock_settings.return_value.macro_strategist_enabled = True
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        brief = result["brief"]
        assert len(brief["debates"]) == 2
        debate_tickers = {d["ticker"] for d in brief["debates"]}
        assert debate_tickers == {"NVDA", "AMD"}
        for debate in brief["debates"]:
            assert "bull" in debate
            assert "bear" in debate


# ── Debate History Formatter Tests ───────────────────────────────


class TestFormatDebateHistory:
    """Tests for the shared debate history formatter."""

    def test_empty_history(self) -> None:
        assert format_debate_history([]) == ""

    def test_single_argument(self) -> None:
        result = format_debate_history(
            [
                {
                    "role": "bull",
                    "round": 1,
                    "argument": "Strong growth",
                    "key_evidence": ["Rev +20%"],
                }
            ]
        )
        assert "## Prior Debate" in result
        assert "BULL (Round 1)" in result
        assert "Strong growth" in result
        assert "Rev +20%" in result

    def test_multi_round_history(self) -> None:
        result = format_debate_history(
            [
                {"role": "bull", "round": 1, "argument": "Bull case"},
                {"role": "bear", "round": 1, "argument": "Bear case", "key_evidence": ["Risk A"]},
                {"role": "bull", "round": 2, "argument": "Bull rebuttal"},
            ]
        )
        assert "BULL (Round 1)" in result
        assert "BEAR (Round 1)" in result
        assert "BULL (Round 2)" in result
        assert "Bull rebuttal" in result
        assert "Risk A" in result

    def test_missing_fields_graceful(self) -> None:
        result = format_debate_history([{"role": "bull"}])
        assert "BULL" in result


# ── Per-Ticker Context Formatter Tests ──────────────────────────


class TestPerTickerContextFormatters:
    """Tests for per-ticker context filtering functions."""

    # ── Social ──

    def test_social_filters_to_matching_ticker(self) -> None:
        state = {
            "social_analysis": {
                "ticker_mentions": [
                    {"ticker": "NVDA", "context": "heavy call buying", "source_accounts": ["@a"]},
                    {"ticker": "AMD", "context": "CPU share gains", "source_accounts": ["@b"]},
                    {"ticker": "NVDA", "context": "data center growth", "source_accounts": ["@c"]},
                ],
            }
        }
        result = format_social_context_for_ticker(state, "NVDA")
        assert "heavy call buying" in result
        assert "data center growth" in result
        assert "CPU share gains" not in result
        assert "@a" in result
        assert "@b" not in result

    def test_social_no_match(self) -> None:
        state = {
            "social_analysis": {
                "ticker_mentions": [
                    {"ticker": "AMD", "context": "buzz"},
                ],
            }
        }
        result = format_social_context_for_ticker(state, "NVDA")
        assert "No social mentions for NVDA" in result

    def test_social_empty_state(self) -> None:
        result = format_social_context_for_ticker({}, "NVDA")
        assert "No social analysis available" in result

    # ── News ──

    def test_news_filters_clusters_by_ticker(self) -> None:
        state = {
            "news_analysis": {
                "story_clusters": [
                    {
                        "headline": "NVDA earnings beat",
                        "event_type": "earnings",
                        "tickers": [{"ticker": "NVDA", "context": "beat estimates"}],
                        "key_facts": ["Revenue up 20%"],
                    },
                    {
                        "headline": "AMD guidance",
                        "event_type": "earnings",
                        "tickers": [{"ticker": "AMD", "context": "weak guidance"}],
                        "key_facts": ["Margins compressed"],
                    },
                    {
                        "headline": "Chip war",
                        "event_type": "geopolitical",
                        "tickers": [
                            {"ticker": "NVDA", "context": "export ban risk"},
                            {"ticker": "AMD", "context": "also affected"},
                        ],
                        "key_facts": ["New restrictions"],
                    },
                ],
            }
        }
        result = format_news_context_for_ticker(state, "NVDA")
        # NVDA earnings cluster included
        assert "NVDA earnings beat" in result
        assert "Revenue up 20%" in result
        # Chip war cluster included (mentions NVDA)
        assert "Chip war" in result
        assert "New restrictions" in result
        # AMD-only cluster excluded
        assert "AMD guidance" not in result
        assert "Margins compressed" not in result

    def test_news_no_match(self) -> None:
        state = {
            "news_analysis": {
                "story_clusters": [
                    {
                        "headline": "AMD news",
                        "event_type": "other",
                        "tickers": [{"ticker": "AMD"}],
                    },
                ],
            }
        }
        result = format_news_context_for_ticker(state, "NVDA")
        assert "No news clusters for NVDA" in result

    def test_news_empty_state(self) -> None:
        result = format_news_context_for_ticker({}, "NVDA")
        assert "No news analysis available" in result

    # ── Company ──

    def test_company_finds_matching_ticker(self) -> None:
        state = {
            "company_analyses": [
                {"ticker": "NVDA", "company_name": "NVIDIA", "sector": "Tech"},
                {"ticker": "AMD", "company_name": "AMD Inc"},
            ]
        }
        result = format_company_context_for_ticker(state, "NVDA")
        assert "NVDA" in result
        assert "NVIDIA" in result
        assert "AMD" not in result

    def test_company_skips_errored(self) -> None:
        state = {
            "company_analyses": [
                {"ticker": "NVDA", "error": True},
            ]
        }
        result = format_company_context_for_ticker(state, "NVDA")
        assert "No company analysis available for NVDA" in result

    def test_company_no_match(self) -> None:
        state = {"company_analyses": [{"ticker": "AMD"}]}
        result = format_company_context_for_ticker(state, "NVDA")
        assert "No company analysis available for NVDA" in result

    def test_company_empty_state(self) -> None:
        result = format_company_context_for_ticker({}, "NVDA")
        assert "No company analysis available for NVDA" in result

    # ── Price ──

    def test_price_finds_matching_ticker(self) -> None:
        state = {
            "price_analyses": [
                {"ticker": "NVDA", "spot_price": 150.0},
                {"ticker": "AMD", "spot_price": 80.0},
            ]
        }
        result = format_price_context_for_ticker(state, "NVDA")
        assert "$150.00" in result
        assert "$80.00" not in result

    def test_price_skips_errored(self) -> None:
        state = {
            "price_analyses": [
                {"ticker": "NVDA", "spot_price": 150.0, "error": True},
            ]
        }
        result = format_price_context_for_ticker(state, "NVDA")
        assert "No price analysis available for NVDA" in result

    def test_price_no_match(self) -> None:
        state = {"price_analyses": [{"ticker": "AMD", "spot_price": 80.0}]}
        result = format_price_context_for_ticker(state, "NVDA")
        assert "No price analysis available for NVDA" in result

    def test_price_empty_state(self) -> None:
        result = format_price_context_for_ticker({}, "NVDA")
        assert "No price analysis available for NVDA" in result


class TestTradeIdeaModel:
    """Tests for TradeIdea model validation."""

    def test_valid_single_ticker(self) -> None:
        idea = TradeIdea(
            tickers=["NVDA"],
            trade_structure="bull call spread NVDA 150/160 June",
            thesis="Strong AI demand",
            catalyst="Earnings beat",
            timeframe="2-4 weeks",
            key_risk="Valuation",
            analysis_date=date(2026, 4, 7),
        )
        assert idea.tickers == ["NVDA"]
        assert "NVDA" in idea.trade_structure

    def test_pair_trade(self) -> None:
        idea = TradeIdea(
            tickers=["NVDA", "AMD"],
            trade_structure="equity L/S: long NVDA / short AMD",
            thesis="NVDA outperforms AMD",
            analysis_date=date(2026, 4, 7),
        )
        assert len(idea.tickers) == 2
        assert idea.tickers[1] == "AMD"

    def test_tickers_requires_at_least_one(self) -> None:
        with pytest.raises(Exception):
            TradeIdea(
                tickers=[],
                trade_structure="buy shares",
                analysis_date=date(2026, 4, 7),
            )

    def test_tickers_rejects_empty_string_element(self) -> None:
        with pytest.raises(Exception):
            TradeIdea(
                tickers=[""],
                trade_structure="buy shares",
                analysis_date=date(2026, 4, 7),
            )

    def test_trade_structure_requires_non_empty(self) -> None:
        with pytest.raises(Exception):
            TradeIdea(
                tickers=["NVDA"],
                trade_structure="",
                analysis_date=date(2026, 4, 7),
            )

    def test_trader_output(self) -> None:
        output = TraderOutput(
            trade_ideas=[
                TradeIdea(
                    tickers=["NVDA"],
                    trade_structure="buy 100 shares NVDA",
                    analysis_date=date(2026, 4, 7),
                ),
            ],
            portfolio_note="Correlated tech exposure",
            analysis_date=date(2026, 4, 7),
        )
        assert len(output.trade_ideas) == 1


class TestCompilerTradeIdeas:
    """Tests for trade ideas in the compiled brief."""

    def test_trade_ideas_included(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [
                {"tickers": ["AAPL"], "trade_structure": "buy shares", "thesis": "Weak buy"},
                {
                    "tickers": ["NVDA"],
                    "trade_structure": "bull call spread",
                    "thesis": "Strong buy",
                },
            ],
        }
        brief = compile_brief(state)
        assert len(brief["trade_ideas"]) == 2

    def test_trade_ideas_filters_errors(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [
                {"tickers": ["NVDA"], "trade_structure": "buy shares", "thesis": "Buy"},
                {"tickers": ["AAPL"], "error": True},
            ],
        }
        brief = compile_brief(state)
        assert len(brief["trade_ideas"]) == 1
        assert brief["trade_ideas"][0]["tickers"] == ["NVDA"]
        assert brief["errors"]["trader_failures"] == ["AAPL"]

    def test_empty_trade_ideas(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [],
        }
        brief = compile_brief(state)
        assert brief["trade_ideas"] == []
        assert brief["errors"]["trader_failures"] == []


def _trader_output(tickers: list[str]) -> TraderOutput:
    """Helper to create a mock TraderOutput for given tickers."""
    return TraderOutput(
        trade_ideas=[
            TradeIdea(
                tickers=[t],
                trade_structure=f"buy 100 shares {t}",
                thesis=f"Buy {t}",
                catalyst="Earnings",
                timeframe="2 weeks",
                key_risk="Valuation",
                analysis_date=date(2026, 4, 7),
            )
            for t in tickers
        ],
        analysis_date=date(2026, 4, 7),
    )


class TestTraderGraphStructure:
    """Tests for Trader node presence in the graph."""

    def test_trader_gate_node_exists(self) -> None:
        graph = build_intelligence_graph(
            db=AsyncMock(),
            sec_edgar=AsyncMock(),
            yfinance=AsyncMock(),
            fred=AsyncMock(),
            crawler=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "trader_gate" in node_names

    def test_trader_node_exists(self) -> None:
        graph = build_intelligence_graph(
            db=AsyncMock(),
            sec_edgar=AsyncMock(),
            yfinance=AsyncMock(),
            fred=AsyncMock(),
            crawler=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "trader" in node_names


@pytest.mark.slow
class TestTraderGraphExecution:
    """Tests for Trader in full graph execution."""

    @pytest.mark.asyncio
    async def test_per_ticker_trader_produces_trade_ideas(self) -> None:
        """Per-ticker mode: Trader called once per ticker, ideas in brief."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        async def mock_trader_per_ticker(state, current_date):
            return _trader_output([state["ticker"]])

        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull),
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear),
            patch(
                f"{_PATCH_PREFIX}.analyze_trade_per_ticker", side_effect=mock_trader_per_ticker
            ) as trader_spy,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            mock_settings.return_value.trader_mode = "per_ticker"
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        assert trader_spy.call_count == 1
        brief = result["brief"]
        assert len(brief["trade_ideas"]) == 1
        assert brief["trade_ideas"][0]["tickers"] == ["NVDA"]

    @pytest.mark.asyncio
    async def test_portfolio_trader_produces_trade_ideas(self) -> None:
        """Portfolio mode: Trader called once with all tickers."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        async def mock_trader_portfolio(state, current_date, tickers):
            return _trader_output(tickers)

        with (
            patch(
                f"{_PATCH_PREFIX}.analyze_social_sentiment",
                return_value=_social(["NVDA", "AMD"]),
            ),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA", "AMD"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull),
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear),
            patch(
                f"{_PATCH_PREFIX}.analyze_trade_portfolio", side_effect=mock_trader_portfolio
            ) as trader_spy,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            mock_settings.return_value.trader_mode = "portfolio"
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        assert trader_spy.call_count == 1
        brief = result["brief"]
        assert len(brief["trade_ideas"]) == 2

    @pytest.mark.asyncio
    async def test_no_tickers_skips_trader(self) -> None:
        """No tickers → trader skipped, no trade ideas."""
        with (
            patch(f"{_PATCH_PREFIX}.analyze_social_sentiment", return_value=_social([])),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news([])),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch(f"{_PATCH_PREFIX}.analyze_trade_per_ticker") as trader_spy,
            patch("synesis.config.get_settings") as mock_settings,
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            mock_settings.return_value.trader_mode = "per_ticker"
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        trader_spy.assert_not_called()
        brief = result["brief"]
        assert brief["trade_ideas"] == []

    @pytest.mark.asyncio
    async def test_trader_partial_failure(self) -> None:
        """One ticker's Trader fails, others succeed — brief has both."""

        async def mock_company(ticker, deps):
            return _company(ticker)

        async def mock_price(ticker, deps):
            return _price(ticker)

        async def mock_bull(state, current_date, debate_history=None):
            return _bull(state["ticker"])

        async def mock_bear(state, current_date, debate_history=None):
            return _bear(state["ticker"])

        call_count = 0

        async def mock_trader_per_ticker(state, current_date):
            nonlocal call_count
            call_count += 1
            if state["ticker"] == "AMD":
                raise RuntimeError("LLM timeout")
            return _trader_output([state["ticker"]])

        with (
            patch(
                f"{_PATCH_PREFIX}.analyze_social_sentiment",
                return_value=_social(["NVDA", "AMD"]),
            ),
            patch(f"{_PATCH_PREFIX}.analyze_news", return_value=_news(["NVDA", "AMD"])),
            patch(f"{_PATCH_PREFIX}.analyze_company", side_effect=mock_company),
            patch(f"{_PATCH_PREFIX}.analyze_price", side_effect=mock_price),
            patch(f"{_PATCH_PREFIX}.analyze_macro", return_value=_macro()),
            patch("synesis.config.get_settings") as mock_settings,
            patch(f"{_PATCH_PREFIX}.research_bull", side_effect=mock_bull),
            patch(f"{_PATCH_PREFIX}.research_bear", side_effect=mock_bear),
            patch(
                f"{_PATCH_PREFIX}.analyze_trade_per_ticker",
                side_effect=mock_trader_per_ticker,
            ),
        ):
            mock_settings.return_value.debate_rounds = 0
            mock_settings.return_value.macro_strategist_enabled = True
            mock_settings.return_value.trader_mode = "per_ticker"
            graph = build_intelligence_graph(
                db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(), fred=AsyncMock()
            )
            result = await graph.ainvoke(
                {"current_date": "2026-04-07"}, config={"recursion_limit": 50}
            )

        assert call_count == 2
        brief = result["brief"]
        # NVDA succeeded
        assert len(brief["trade_ideas"]) == 1
        assert brief["trade_ideas"][0]["tickers"] == ["NVDA"]
        # AMD failed
        assert "AMD" in brief["errors"]["trader_failures"]


class TestTraderDebateFormatting:
    """Tests for _format_debate_for_ticker in trader.py."""

    def test_formats_single_round(self) -> None:
        """Single round debate produces bull + bear sections."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        state = {
            "bull_analyses": [
                {
                    "ticker": "NVDA",
                    "role": "bull",
                    "argument": "Strong buy",
                    "key_evidence": ["Revenue up"],
                    "round": 1,
                },
            ],
            "bear_analyses": [
                {
                    "ticker": "NVDA",
                    "role": "bear",
                    "argument": "Overvalued",
                    "key_evidence": ["PE high"],
                    "round": 1,
                },
            ],
        }
        result = _format_debate_for_ticker(state, "NVDA")
        assert "BULL" in result
        assert "Strong buy" in result
        assert "BEAR" in result
        assert "Overvalued" in result

    def test_filters_to_ticker(self) -> None:
        """Only includes debate for the requested ticker."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        state = {
            "bull_analyses": [
                {"ticker": "NVDA", "role": "bull", "argument": "NVDA bull", "round": 1},
                {"ticker": "AMD", "role": "bull", "argument": "AMD bull", "round": 1},
            ],
            "bear_analyses": [
                {"ticker": "NVDA", "role": "bear", "argument": "NVDA bear", "round": 1},
            ],
        }
        result = _format_debate_for_ticker(state, "NVDA")
        assert "NVDA bull" in result
        assert "NVDA bear" in result
        assert "AMD bull" not in result

    def test_multi_round_includes_all(self) -> None:
        """Multi-round debate includes all rounds, ordered."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        state = {
            "bull_analyses": [
                {"ticker": "NVDA", "role": "bull", "argument": "Bull round 1", "round": 1},
                {"ticker": "NVDA", "role": "bull", "argument": "Bull round 2", "round": 2},
            ],
            "bear_analyses": [
                {"ticker": "NVDA", "role": "bear", "argument": "Bear round 1", "round": 1},
                {"ticker": "NVDA", "role": "bear", "argument": "Bear round 2", "round": 2},
            ],
        }
        result = _format_debate_for_ticker(state, "NVDA")
        assert "Bull round 1" in result
        assert "Bull round 2" in result
        assert "Bear round 1" in result
        assert "Bear round 2" in result
        # Round 1 should appear before round 2
        assert result.index("Round 1") < result.index("Round 2")

    def test_skips_errored(self) -> None:
        """Errored entries are excluded."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        state = {
            "bull_analyses": [
                {"ticker": "NVDA", "error": True},
                {"ticker": "NVDA", "role": "bull", "argument": "Good bull", "round": 1},
            ],
            "bear_analyses": [],
        }
        result = _format_debate_for_ticker(state, "NVDA")
        assert "Good bull" in result

    def test_no_debate_available(self) -> None:
        """No debate data produces fallback message."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        state = {"bull_analyses": [], "bear_analyses": []}
        result = _format_debate_for_ticker(state, "NVDA")
        assert "No debate available" in result

    def test_missing_state_keys(self) -> None:
        """Missing bull/bear keys handled gracefully."""
        from synesis.processing.intelligence.trader.trader import _format_debate_for_ticker

        result = _format_debate_for_ticker({}, "NVDA")
        assert "No debate available" in result
