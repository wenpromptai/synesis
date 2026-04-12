"""LangGraph pipeline builder for the daily intelligence pipeline.

Builds a compiled StateGraph that orchestrates:
  Layer 1: SocialSentiment + News (parallel signal discovery)
  → extract_tickers (deterministic)
  → Layer 2 + Macro (ALL parallel via Send):
      CompanyAnalyst (per ticker) | PriceAnalyst (per ticker) | MacroStrategist
  → l2_join (defer=True, waits for all Layer 2 + Macro)
  → Layer 3 (configurable via DEBATE_ROUNDS):
      rounds=0: BullResearcher | BearResearcher (parallel, no debate)
      rounds≥1: TickerDebate subgraph per ticker (bull ⇄ bear loop)
  → trader_gate (defer=True, waits for all debate output)
  → Trader (configurable via TRADER_MODE):
      per_ticker: one Send per ticker
      portfolio: single Send with all tickers
  → Compiler (defer=True)

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
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from synesis.core.logging import get_logger
from synesis.processing.intelligence.compiler import compile_brief
from synesis.processing.intelligence.debate.bear import research_bear
from synesis.processing.intelligence.debate.bull import research_bull
from synesis.processing.intelligence.debate.subgraph import build_debate_subgraph
from synesis.processing.intelligence.trader.trader import (
    analyze_trade_per_ticker,
    analyze_trade_portfolio,
)
from synesis.processing.intelligence.specialists.company import CompanyDeps, analyze_company
from synesis.processing.intelligence.specialists.news import NewsDeps, analyze_news
from synesis.processing.intelligence.specialists.price import PriceDeps, analyze_price
from synesis.processing.intelligence.specialists.social_sentiment import (
    SocialSentimentDeps,
    analyze_social_sentiment,
)
from synesis.processing.intelligence.state import IntelligenceState
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

    # ── Layer 1 (parallel signal discovery) ──────────────────────

    async def social_sentiment_node(state: IntelligenceState) -> dict[str, Any]:
        """Run SocialSentimentAnalyst on recent tweets."""
        logger.info("SocialSentimentAnalyst starting")
        try:
            current = date.fromisoformat(state["current_date"])
            deps = SocialSentimentDeps(db=db, current_date=current)
            result = await analyze_social_sentiment(deps)
            data = result.model_dump(mode="json")
            logger.info(
                "SocialSentimentAnalyst complete",
                ticker_mentions=len(data.get("ticker_mentions", [])),
            )
            return {"social_analysis": data}
        except Exception:
            logger.exception("SocialSentimentAnalyst failed", date=state.get("current_date"))
            return {"social_analysis": {"error": True}}

    async def news_analyst_node(state: IntelligenceState) -> dict[str, Any]:
        """Run NewsAnalyst on recent pre-scored messages."""
        logger.info("NewsAnalyst starting")
        try:
            current = date.fromisoformat(state["current_date"])
            deps = NewsDeps(db=db, current_date=current)
            result = await analyze_news(deps)
            data = result.model_dump(mode="json")
            logger.info(
                "NewsAnalyst complete",
                story_clusters=len(data.get("story_clusters", [])),
            )
            return {"news_analysis": data}
        except Exception:
            logger.exception("NewsAnalyst failed", date=state.get("current_date"))
            return {"news_analysis": {"error": True}}

    # ── Ticker Extraction (deterministic) ────────────────────────

    async def extract_tickers_node(state: IntelligenceState) -> dict[str, Any]:
        """Collect all unique tickers from Layer 1 outputs."""
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

    # ── Layer 2 Fan-Out Router ───────────────────────────────────

    def route_to_L2(state: IntelligenceState) -> list[Send]:  # noqa: N802
        """Fan-out: CompanyAnalyst + PriceAnalyst per ticker + MacroStrategist, all parallel."""
        tickers = state.get("target_tickers", [])
        sends: list[Send] = []

        # MacroStrategist always runs (only needs FRED + Layer 1 themes)
        sends.append(Send("macro_strategist", state))

        if not tickers:
            logger.info("No tickers to analyze — macro only")
            return sends

        logger.info("Routing to Layer 2", tickers=tickers)
        for t in tickers:
            sends.append(Send("company_analyst", {**state, "ticker": t}))
            sends.append(Send("price_analyst", {**state, "ticker": t}))
        return sends

    # ── Layer 2: Company Analyst (per-ticker via Send) ───────────

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
        logger.info("CompanyAnalyst starting", ticker=ticker)
        try:
            result = await analyze_company(ticker, deps)
            logger.info("CompanyAnalyst complete", ticker=ticker)
            return {"company_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("CompanyAnalyst failed", ticker=ticker)
            return {"company_analyses": [{"ticker": ticker, "error": True}]}

    # ── Layer 2: Price Analyst (per-ticker via Send) ─────────────

    async def price_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run PriceAnalyst for a single ticker. Called via Send.

        yfinance calls are unlimited. Massive calls (up to 3 per ticker) are
        rate-limited by the MassiveClient's built-in token bucket.
        """
        ticker = state["ticker"]
        current = date.fromisoformat(state["current_date"])
        deps = PriceDeps(yfinance=yfinance, massive=massive, current_date=current)
        logger.info("PriceAnalyst starting", ticker=ticker)
        try:
            result = await analyze_price(ticker, deps)
            logger.info("PriceAnalyst complete", ticker=ticker)
            return {"price_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("PriceAnalyst failed", ticker=ticker)
            return {"price_analyses": [{"ticker": ticker, "error": True}]}

    # ── MacroStrategist (parallel with Layer 2) ──────────────────

    async def macro_strategist_node(state: IntelligenceState) -> dict[str, Any]:
        """Run MacroStrategist — assess regime from FRED data + Layer 1 themes."""
        try:
            from synesis.config import get_settings

            current = date.fromisoformat(state["current_date"])
            if not get_settings().macro_strategist_enabled or fred is None:
                logger.info("MacroStrategist disabled or FRED unavailable — skipping")
                return {"macro_view": {}}

            logger.info("MacroStrategist starting")
            deps = MacroStrategistDeps(
                fred=fred,
                db=db,
                yfinance=yfinance,
                sec_edgar=sec_edgar,
                crawler=crawler,
                current_date=current,
            )
            result = await analyze_macro(dict(state), deps)
            logger.info("MacroStrategist complete", regime=result.regime)
            return {"macro_view": result.model_dump(mode="json")}
        except Exception:
            logger.exception("MacroStrategist failed", date=state.get("current_date"))
            return {"macro_view": {"error": True}}

    # ── Layer 2 Join (waits for all analysts + macro) ────────────

    async def l2_join_node(state: IntelligenceState) -> dict[str, Any]:
        """Sync barrier: waits for all Layer 2 analysts + MacroStrategist (defer=True)."""
        logger.info(
            "L2 join complete",
            company_analyses=len(state.get("company_analyses", [])),
            price_analyses=len(state.get("price_analyses", [])),
            has_macro=bool(state.get("macro_view")),
        )
        return {}

    def l2_router(state: IntelligenceState) -> list[str | Send]:
        """Route to per-ticker debate if tickers exist, otherwise skip to compiler."""
        from synesis.config import get_settings

        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No priority tickers — skipping debate + trader, going to compiler")
            return ["compiler"]

        rounds = get_settings().debate_rounds
        sends: list[str | Send] = []
        for t in tickers:
            if rounds == 0:
                # Parallel: independent bull + bear (no debate)
                sends.append(Send("bull_researcher", {**state, "ticker": t}))
                sends.append(Send("bear_researcher", {**state, "ticker": t}))
            else:
                # Sequential debate subgraph
                sends.append(Send("ticker_debate", {**state, "ticker": t}))
        return sends

    # ── BullResearcher (per-ticker via Send, rounds=0 only) ─────

    async def bull_researcher_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run BullResearcher for a single ticker. Called via Send (rounds=0)."""
        ticker = state["ticker"]
        logger.info("BullResearcher starting", ticker=ticker)
        try:
            current = date.fromisoformat(state["current_date"])
            result = await research_bull(state, current)
            logger.info("BullResearcher complete", ticker=ticker)
            return {"bull_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("BullResearcher failed", ticker=ticker)
            return {"bull_analyses": [{"ticker": ticker, "error": True}]}

    # ── BearResearcher (per-ticker via Send, rounds=0 only) ─────

    async def bear_researcher_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run BearResearcher for a single ticker. Called via Send (rounds=0)."""
        ticker = state["ticker"]
        logger.info("BearResearcher starting", ticker=ticker)
        try:
            current = date.fromisoformat(state["current_date"])
            result = await research_bear(state, current)
            logger.info("BearResearcher complete", ticker=ticker)
            return {"bear_analyses": [result.model_dump(mode="json")]}
        except Exception:
            logger.exception("BearResearcher failed", ticker=ticker)
            return {"bear_analyses": [{"ticker": ticker, "error": True}]}

    # ── Ticker Debate (per-ticker subgraph, rounds>=1) ──────────

    debate_subgraph = build_debate_subgraph()

    async def ticker_debate_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run multi-round debate subgraph for a single ticker. Called via Send."""
        from synesis.config import get_settings

        ticker = state["ticker"]
        try:
            rounds = get_settings().debate_rounds
            logger.info("Debate starting", ticker=ticker, rounds=rounds)
            debate_input = {
                "ticker": ticker,
                "current_date": state["current_date"],
                "social_analysis": state.get("social_analysis", {}),
                "news_analysis": state.get("news_analysis", {}),
                "company_analyses": state.get("company_analyses", []),
                "price_analyses": state.get("price_analyses", []),
                "debate_history": [],
                "round": 0,
                "max_rounds": rounds,
            }
            result = await debate_subgraph.ainvoke(
                debate_input, config={"recursion_limit": 4 * rounds + 5}
            )
            history = result.get("debate_history", [])
            bulls = [h for h in history if h.get("role") == "bull"]
            bears = [h for h in history if h.get("role") == "bear"]
            logger.info(
                "Debate complete",
                ticker=ticker,
                rounds_completed=result.get("round", 0),
                bull_entries=len(bulls),
                bear_entries=len(bears),
            )
            return {"bull_analyses": bulls, "bear_analyses": bears}
        except Exception:
            logger.exception("Debate failed", ticker=ticker)
            return {
                "bull_analyses": [{"ticker": ticker, "error": True}],
                "bear_analyses": [{"ticker": ticker, "error": True}],
            }

    # ── Trader Gate (waits for all debate output) ──────────────────

    async def trader_gate_node(state: IntelligenceState) -> dict[str, Any]:
        """Sync barrier: waits for all debate output (defer=True)."""
        logger.info(
            "Trader gate complete",
            bull_analyses=len(state.get("bull_analyses", [])),
            bear_analyses=len(state.get("bear_analyses", [])),
        )
        return {}

    def trader_router(state: IntelligenceState) -> list[str | Send]:
        """Route to per-ticker or portfolio Trader based on config."""
        from synesis.config import get_settings

        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No tickers — skipping trader, going to compiler")
            return ["compiler"]

        mode = get_settings().trader_mode
        if mode == "portfolio":
            logger.info("Routing to Trader (portfolio)", tickers=tickers)
            return [Send("trader", {**state, "tickers": tickers, "mode": "portfolio"})]
        else:
            logger.info("Routing to Trader (per_ticker)", tickers=tickers)
            return [Send("trader", {**state, "ticker": t, "mode": "per_ticker"}) for t in tickers]

    # ── Trader (per-ticker or portfolio via Send) ────────────────

    async def trader_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run Trader for one ticker or all tickers. Called via Send."""
        mode = state.get("mode", "per_ticker")
        ticker = state.get("ticker", "UNKNOWN")
        tickers = state.get("tickers", [])
        logger.info("Trader starting", mode=mode, ticker=ticker, tickers=tickers)
        try:
            current = date.fromisoformat(state["current_date"])
            if mode == "portfolio":
                result = await analyze_trade_portfolio(state, current, tickers)
            else:
                result = await analyze_trade_per_ticker(state, current)
            logger.info(
                "Trader complete",
                mode=mode,
                trade_ideas=len(result.trade_ideas),
                portfolio_note_len=len(result.portfolio_note) if result.portfolio_note else 0,
            )
            out: dict[str, Any] = {
                "trade_ideas": [idea.model_dump(mode="json") for idea in result.trade_ideas],
            }
            if mode == "portfolio":
                out["portfolio_note"] = result.portfolio_note
            return out
        except Exception:
            if mode == "portfolio":
                logger.exception("Trader failed (portfolio)", tickers=tickers)
                return {
                    "trade_ideas": [{"tickers": [t], "error": True} for t in tickers],
                    "portfolio_note": "",
                }
            else:
                logger.exception("Trader failed", ticker=ticker)
                return {"trade_ideas": [{"tickers": [ticker], "error": True}]}

    # ── Compiler ─────────────────────────────────────────────────

    async def compiler_node(state: IntelligenceState) -> dict[str, Any]:
        """Assemble the final brief from all pipeline outputs."""
        try:
            brief = compile_brief(dict(state))
            return {"brief": brief}
        except Exception:
            logger.exception("Compiler failed", date=state.get("current_date"))
            return {"brief": {}}

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(IntelligenceState)

    # Add nodes
    graph.add_node("social_sentiment", social_sentiment_node)
    graph.add_node("news_analyst", news_analyst_node)
    graph.add_node("extract_tickers", extract_tickers_node)
    graph.add_node("company_analyst", company_analyst_node)  # type: ignore[type-var]
    graph.add_node("price_analyst", price_analyst_node)  # type: ignore[type-var]
    graph.add_node("macro_strategist", macro_strategist_node)
    graph.add_node("l2_join", l2_join_node, defer=True)
    graph.add_node("bull_researcher", bull_researcher_node)  # type: ignore[type-var]
    graph.add_node("bear_researcher", bear_researcher_node)  # type: ignore[type-var]
    graph.add_node("ticker_debate", ticker_debate_node)  # type: ignore[type-var]
    graph.add_node("trader_gate", trader_gate_node, defer=True)
    graph.add_node("trader", trader_node)  # type: ignore[type-var]
    graph.add_node("compiler", compiler_node, defer=True)

    # Layer 1: parallel fan-out from START
    graph.add_edge(START, "social_sentiment")
    graph.add_edge(START, "news_analyst")

    # Fan-in: both Layer 1 nodes must complete before ticker extraction
    graph.add_edge("social_sentiment", "extract_tickers")
    graph.add_edge("news_analyst", "extract_tickers")

    # Dynamic fan-out: Layer 2 analysts + MacroStrategist (all parallel)
    graph.add_conditional_edges("extract_tickers", route_to_L2)

    # Join: all Layer 2 + Macro feed into l2_join (defer waits for all)
    graph.add_edge("company_analyst", "l2_join")
    graph.add_edge("price_analyst", "l2_join")
    graph.add_edge("macro_strategist", "l2_join")

    # Layer 3: route to debate (rounds>=1) or parallel bull/bear (rounds=0)
    graph.add_conditional_edges(
        "l2_join",
        l2_router,
        ["bull_researcher", "bear_researcher", "ticker_debate", "compiler"],
    )

    # Debate paths feed into trader_gate (defer waits for all debate output)
    graph.add_edge("bull_researcher", "trader_gate")
    graph.add_edge("bear_researcher", "trader_gate")
    graph.add_edge("ticker_debate", "trader_gate")

    # Trader routing: per-ticker Sends or portfolio Send, or skip to compiler
    graph.add_conditional_edges(
        "trader_gate",
        trader_router,
        ["trader", "compiler"],
    )

    # Trader feeds into compiler (defer waits for all trader Sends)
    graph.add_edge("trader", "compiler")
    graph.add_edge("compiler", END)

    return graph.compile()
