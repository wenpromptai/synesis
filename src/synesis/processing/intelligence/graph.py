"""LangGraph pipeline builder for the daily intelligence pipeline.

Builds a compiled StateGraph that orchestrates:
  Tier 1: SocialSentiment + News (parallel)
  → extract_tickers (deterministic)
  → Tier 2: CompanyAnalyst per extracted ticker (dynamic fan-out via Send)
  → Layer 2: MacroStrategist → EquityStrategist (sequential)
  → Conviction gate → Compiler (deterministic)

Provider clients (DB, SEC EDGAR, yfinance, FRED, crawler) are captured via
closure — they never appear in state (not serializable).

Usage:
    graph = build_intelligence_graph(db, sec_edgar, yfinance, fred, crawler)
    result = await graph.ainvoke(
        {"current_date": "2026-04-06", ...},
        config={"recursion_limit": 50},
    )
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from synesis.core.logging import get_logger
from synesis.processing.intelligence.compiler import compile_brief
from synesis.processing.intelligence.specialists.company import CompanyDeps, analyze_company
from synesis.processing.intelligence.specialists.news import NewsDeps, analyze_news
from synesis.processing.intelligence.specialists.price import PriceDeps, analyze_price
from synesis.processing.intelligence.specialists.social_sentiment import (
    SocialSentimentDeps,
    analyze_social_sentiment,
)
from synesis.processing.intelligence.state import IntelligenceState
from synesis.processing.intelligence.strategists.equity import (
    EquityStrategistDeps,
    analyze_equity,
)
from synesis.processing.intelligence.strategists.macro import (
    MacroStrategistDeps,
    analyze_macro,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.massive.client import MassiveClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


def build_intelligence_graph(
    db: "Database",
    sec_edgar: "SECEdgarClient",
    yfinance: "YFinanceClient",
    fred: "FREDClient | None" = None,
    massive: "MassiveClient | None" = None,
    crawler: "Crawl4AICrawlerProvider | None" = None,
) -> Any:
    """Build the intelligence pipeline graph. Call once at startup.

    Provider clients are captured via closure so they never appear in
    serializable state. The compiled graph is reusable for each daily run.
    """

    # ── Tier 1 Nodes (parallel signal discovery) ─────────────────

    async def social_sentiment_node(state: IntelligenceState) -> dict[str, Any]:
        """Run SocialSentimentAnalyst on recent tweets."""
        try:
            current = date.fromisoformat(state["current_date"])
            deps = SocialSentimentDeps(db=db, current_date=current)
            result = await analyze_social_sentiment(deps)
            return {"social_analysis": result.model_dump(mode="json")}
        except Exception:
            logger.exception("SocialSentimentAnalyst failed")
            return {"social_analysis": {}}

    async def news_analyst_node(state: IntelligenceState) -> dict[str, Any]:
        """Run NewsAnalyst on recent pre-scored messages."""
        try:
            current = date.fromisoformat(state["current_date"])
            deps = NewsDeps(db=db, current_date=current)
            result = await analyze_news(deps)
            return {"news_analysis": result.model_dump(mode="json")}
        except Exception:
            logger.exception("NewsAnalyst failed")
            return {"news_analysis": {}}

    # ── Ticker Extraction (deterministic) ────────────────────────

    async def extract_tickers_node(state: IntelligenceState) -> dict[str, Any]:
        """Collect all unique tickers from Tier 1 outputs."""
        tickers: set[str] = set()

        social = state.get("social_analysis", {})
        for mention in social.get("ticker_mentions", []):
            if mention.get("ticker"):
                tickers.add(mention["ticker"])

        news = state.get("news_analysis", {})
        for cluster in news.get("story_clusters", []):
            for mention in cluster.get("tickers", []):
                if mention.get("ticker"):
                    tickers.add(mention["ticker"])

        target = sorted(tickers)
        logger.info("Tickers extracted", count=len(target), tickers=target)
        return {"target_tickers": target}

    # ── Ticker Fan-Out Router ────────────────────────────────────

    def route_to_tier2(state: IntelligenceState) -> list[Send]:
        """Fan-out: CompanyAnalyst + PriceAnalyst per ticker, all parallel."""
        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No tickers to analyze — skipping to macro_strategist")
            return [Send("macro_strategist", state)]
        logger.info("Routing to Tier 2", tickers=tickers)
        sends: list[Send] = []
        for t in tickers:
            sends.append(Send("company_analyst", {**state, "ticker": t}))
            sends.append(Send("price_analyst", {**state, "ticker": t}))
        return sends

    # ── Tier 2: Company Analyst (per-ticker via Send) ────────────

    async def company_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run CompanyAnalyst for a single ticker. Called via Send."""
        ticker = state["ticker"]
        current = date.fromisoformat(state["current_date"])
        deps = CompanyDeps(
            sec_edgar=sec_edgar,
            yfinance=yfinance,
            crawler=crawler,
            current_date=current,
        )
        try:
            result = await analyze_company(ticker, deps)
            return {"company_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("CompanyAnalyst failed", ticker=ticker)
            return {"company_analyses": [{"ticker": ticker, "error": True}]}

    # ── Tier 2: Price Analyst (per-ticker via Send) ──────────────

    async def price_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run PriceAnalyst for a single ticker. Called via Send.

        yfinance calls are unlimited. Massive calls (up to 3 per ticker) are
        rate-limited by the MassiveClient's built-in token bucket.
        """
        ticker = state["ticker"]
        current = date.fromisoformat(state["current_date"])
        deps = PriceDeps(yfinance=yfinance, massive=massive, current_date=current)
        try:
            result = await analyze_price(ticker, deps)
            return {"price_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("PriceAnalyst failed", ticker=ticker)
            return {"price_analyses": [{"ticker": ticker, "error": True}]}

    # ── Layer 2: Strategists (sequential) ────────────────────────

    async def macro_strategist_node(state: IntelligenceState) -> dict[str, Any]:
        """Run MacroStrategist — assess regime from FRED data + Tier 1 themes."""
        try:
            from synesis.config import get_settings

            current = date.fromisoformat(state["current_date"])
            if not get_settings().macro_strategist_enabled or fred is None:
                logger.info("MacroStrategist disabled or FRED unavailable — skipping")
                return {"macro_view": {}}

            deps = MacroStrategistDeps(fred=fred, current_date=current)
            result = await analyze_macro(dict(state), deps)
            return {"macro_view": result.model_dump(mode="json")}
        except Exception:
            logger.exception("MacroStrategist failed")
            return {"macro_view": {}}

    async def equity_strategist_node(state: IntelligenceState) -> dict[str, Any]:
        """Run EquityStrategist — produce ranked trade ideas."""
        try:
            current = date.fromisoformat(state["current_date"])
            deps = EquityStrategistDeps(current_date=current)
            result = await analyze_equity(dict(state), deps)
            return {"equity_ideas": result.model_dump(mode="json")}
        except Exception:
            logger.exception("EquityStrategist failed")
            return {"equity_ideas": {}}

    # ── Conviction Gate ──────────────────────────────────────────

    def conviction_gate(state: IntelligenceState) -> Literal["compiler"]:
        """Route high-conviction ideas to debate, rest to compiler.

        Phase 3B: placeholder — routes everything to compiler.
        Phase 3C will add: abs(sentiment_score) >= 0.7 → debate branch.
        """
        return "compiler"

    # ── Compiler Node ────────────────────────────────────────────

    async def compiler_node(state: IntelligenceState) -> dict[str, Any]:
        """Assemble the final brief from all pipeline outputs."""
        brief = compile_brief(dict(state))
        logger.info(
            "Brief compiled",
            tickers_analyzed=len(brief.get("tickers_analyzed", [])),
            trade_ideas=len(brief.get("trade_ideas", [])),
        )
        return {"brief": brief}

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(IntelligenceState)

    # Add nodes
    graph.add_node("social_sentiment", social_sentiment_node)
    graph.add_node("news_analyst", news_analyst_node)
    graph.add_node("extract_tickers", extract_tickers_node)
    graph.add_node("company_analyst", company_analyst_node)  # type: ignore[type-var]
    graph.add_node("price_analyst", price_analyst_node)  # type: ignore[type-var]
    graph.add_node("macro_strategist", macro_strategist_node, defer=True)
    graph.add_node("equity_strategist", equity_strategist_node)
    graph.add_node("conviction_gate", conviction_gate)
    graph.add_node("compiler", compiler_node)

    # Tier 1: parallel fan-out from START
    graph.add_edge(START, "social_sentiment")
    graph.add_edge(START, "news_analyst")

    # Fan-in: both Tier 1 nodes must complete before ticker extraction
    graph.add_edge("social_sentiment", "extract_tickers")
    graph.add_edge("news_analyst", "extract_tickers")

    # Dynamic fan-out: CompanyAnalysts (parallel) + PriceAnalyst (sequential)
    graph.add_conditional_edges("extract_tickers", route_to_tier2)

    # Layer 2: strategists (defer on macro waits for all Tier 2 to complete)
    graph.add_edge("company_analyst", "macro_strategist")
    graph.add_edge("price_analyst", "macro_strategist")
    graph.add_edge("macro_strategist", "equity_strategist")

    # Gate + compiler
    graph.add_edge("equity_strategist", "conviction_gate")
    graph.add_conditional_edges("conviction_gate", conviction_gate)
    graph.add_edge("compiler", END)

    return graph.compile()
