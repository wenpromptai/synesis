"""LangGraph pipeline builder for the daily intelligence pipeline.

Builds a compiled StateGraph that orchestrates:
  Tier 1: SocialSentiment + News (parallel)
  → extract_tickers (deterministic)
  → Tier 2: CompanyAnalyst per extracted ticker (dynamic fan-out via Send)
  → Compiler (deterministic)

Provider clients (DB, SEC EDGAR, yfinance, crawler) are captured via
closure — they never appear in state (not serializable).

Usage:
    graph = build_intelligence_graph(db, sec_edgar, yfinance, crawler)
    result = await graph.ainvoke(
        {"current_date": "2026-04-06", ...},
        config={"recursion_limit": 50},
    )
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from synesis.core.logging import get_logger
from synesis.processing.intelligence.compiler import compile_brief
from synesis.processing.intelligence.specialists.company import CompanyDeps, analyze_company
from synesis.processing.intelligence.specialists.news import NewsDeps, analyze_news
from synesis.processing.intelligence.specialists.social_sentiment import (
    SocialSentimentDeps,
    analyze_social_sentiment,
)
from synesis.processing.intelligence.state import IntelligenceState

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


def build_intelligence_graph(
    db: "Database",
    sec_edgar: "SECEdgarClient",
    yfinance: "YFinanceClient",
    crawler: "Crawl4AICrawlerProvider | None" = None,
) -> Any:
    """Build the intelligence pipeline graph. Call once at startup.

    Provider clients are captured via closure so they never appear in
    serializable state. The compiled graph is reusable for each daily run.
    """

    # ── Tier 1 Nodes (parallel signal discovery) ─────────────────

    async def social_sentiment_node(state: IntelligenceState) -> dict[str, Any]:
        """Run SocialSentimentAnalyst on recent tweets."""
        current = date.fromisoformat(state["current_date"])
        deps = SocialSentimentDeps(db=db, current_date=current)
        result = await analyze_social_sentiment(deps)
        return {"social_analysis": result.model_dump(mode="json")}

    async def news_analyst_node(state: IntelligenceState) -> dict[str, Any]:
        """Run NewsAnalyst on recent pre-scored messages."""
        current = date.fromisoformat(state["current_date"])
        deps = NewsDeps(db=db, current_date=current)
        result = await analyze_news(deps)
        return {"news_analysis": result.model_dump(mode="json")}

    # ── Ticker Extraction (deterministic) ────────────────────────

    async def extract_tickers_node(state: IntelligenceState) -> dict[str, Any]:
        """Collect all unique tickers from Tier 1 outputs."""
        tickers: set[str] = set()

        # From social: flat ticker_mentions
        social = state.get("social_analysis", {})
        for mention in social.get("ticker_mentions", []):
            if mention.get("ticker"):
                tickers.add(mention["ticker"])

        # From news: nested in story_clusters
        news = state.get("news_analysis", {})
        for cluster in news.get("story_clusters", []):
            for mention in cluster.get("tickers", []):
                if mention.get("ticker"):
                    tickers.add(mention["ticker"])

        target = sorted(tickers)
        logger.info("Tickers extracted", count=len(target), tickers=target)
        return {"target_tickers": target}

    # ── Ticker Fan-Out Router ────────────────────────────────────

    def route_to_company_analysts(
        state: IntelligenceState,
    ) -> list[Send]:
        """Fan-out: one CompanyAnalyst per extracted ticker via Send."""
        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No tickers to analyze — skipping to compiler")
            return [Send("compiler", state)]
        logger.info("Routing to CompanyAnalyst", tickers=tickers)
        return [Send("company_analyst", {**state, "ticker": t}) for t in tickers]

    # ── Tier 2 Node (per-ticker deep dive) ───────────────────────

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

    # ── Compiler Node ────────────────────────────────────────────

    async def compiler_node(state: IntelligenceState) -> dict[str, Any]:
        """Assemble the final brief from all pipeline outputs."""
        brief = compile_brief(dict(state))
        logger.info(
            "Brief compiled",
            tickers_analyzed=len(brief.get("tickers_analyzed", [])),
        )
        return {"brief": brief}

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(IntelligenceState)

    # Add nodes
    graph.add_node("social_sentiment", social_sentiment_node)
    graph.add_node("news_analyst", news_analyst_node)
    graph.add_node("extract_tickers", extract_tickers_node)
    graph.add_node("company_analyst", company_analyst_node)  # type: ignore[type-var]
    graph.add_node("compiler", compiler_node, defer=True)

    # Tier 1: parallel fan-out from START
    graph.add_edge(START, "social_sentiment")
    graph.add_edge(START, "news_analyst")

    # Fan-in: both Tier 1 nodes must complete before ticker extraction
    graph.add_edge("social_sentiment", "extract_tickers")
    graph.add_edge("news_analyst", "extract_tickers")

    # Dynamic fan-out: one CompanyAnalyst per ticker
    graph.add_conditional_edges("extract_tickers", route_to_company_analysts)

    # All company analysts feed into compiler (defer=True waits for all)
    graph.add_edge("company_analyst", "compiler")

    # Done
    graph.add_edge("compiler", END)

    return graph.compile()
