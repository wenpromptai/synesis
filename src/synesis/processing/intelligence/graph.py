"""LangGraph pipeline builders for the intelligence system.

Two separate graphs:

1. **Scan Graph** (daily scheduled):
   Social + News (parallel) → extract_tickers → MacroStrategist → scan_compiler
   Outputs macro regime, thematic tilts, and watchlist tickers for Discord.

2. **Analyze Graph** (on-demand POST endpoint):
   Input tickers → CompanyAnalyst + PriceAnalyst (parallel per ticker via Send)
   → debate (bull/bear) → Trader → analyze_compiler
   Returns trade ideas as JSON + saves to KG.

Provider clients are captured via closure — never serialized in state.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.processing.intelligence.compiler import compile_brief, compile_scan_brief
from synesis.processing.intelligence.debate.bear import research_bear
from synesis.processing.intelligence.debate.bull import research_bull
from synesis.processing.intelligence.debate.subgraph import build_debate_subgraph
from synesis.processing.intelligence.specialists.company import CompanyDeps, analyze_company
from synesis.processing.intelligence.specialists.news import NewsDeps, analyze_news
from synesis.processing.intelligence.specialists.price import PriceDeps, analyze_price
from synesis.ingestion.twitterapi import TwitterClient
from synesis.processing.intelligence.specialists.ticker_research import (
    TickerResearchDeps,
    analyze_ticker_research,
)
from synesis.processing.intelligence.specialists.social_sentiment import (
    SocialSentimentDeps,
    analyze_social_sentiment,
)
from synesis.processing.intelligence.state import AnalyzeState, ScanState
from synesis.processing.intelligence.strategists.macro import (
    MacroStrategistDeps,
    analyze_macro,
)
from synesis.processing.intelligence.trader.trader import (
    analyze_trade_per_ticker,
    analyze_trade_portfolio,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.massive.client import MassiveClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Scan Graph — daily signal discovery + macro regime + watchlist
# ═══════════════════════════════════════════════════════════════════


def build_scan_graph(
    db: "Database",
    sec_edgar: "SECEdgarClient",
    yfinance: "YFinanceClient",
    fred: "FREDClient | None" = None,
    crawler: "Crawl4AICrawlerProvider | None" = None,
) -> Any:
    """Build the daily scan pipeline graph.

    START → social_sentiment + news_analyst (parallel)
      → extract_tickers → macro_strategist → scan_compiler → END
    """

    async def social_sentiment_node(state: ScanState) -> dict[str, Any]:
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

    async def news_analyst_node(state: ScanState) -> dict[str, Any]:
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

    async def extract_tickers_node(state: ScanState) -> dict[str, Any]:
        try:
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
            return {"target_tickers": target, "l1_tickers": target}
        except Exception:
            logger.exception("Ticker extraction failed")
            return {"target_tickers": [], "l1_tickers": []}

    async def macro_strategist_node(state: ScanState) -> dict[str, Any]:
        try:
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
            data = result.model_dump(mode="json")
            logger.info("MacroStrategist complete", regime=result.regime)

            output: dict[str, Any] = {"macro_view": data}

            # Save watchlist context (all picks — no culling)
            if result.watchlist_picks:
                output["watchlist_context"] = {
                    "selected": [p.model_dump(mode="json") for p in result.watchlist_picks],
                    "themes": [t.theme for t in result.thematic_tilts],
                    "dropped": result.tickers_dropped,
                    "drop_reasons": result.drop_reasons,
                }
                logger.info(
                    "Watchlist",
                    tickers=[p.ticker for p in result.watchlist_picks],
                    dropped=result.tickers_dropped,
                )

            return output
        except Exception:
            logger.exception("MacroStrategist failed", date=state.get("current_date"))
            return {"macro_view": {"error": True}}

    async def scan_compiler_node(state: ScanState) -> dict[str, Any]:
        try:
            brief = compile_scan_brief(dict(state))
            return {"brief": brief}
        except Exception:
            logger.exception("Scan compiler failed", date=state.get("current_date"))
            return {"brief": {}}

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(ScanState)

    graph.add_node("social_sentiment", social_sentiment_node)
    graph.add_node("news_analyst", news_analyst_node)
    graph.add_node("extract_tickers", extract_tickers_node)
    graph.add_node("macro_strategist", macro_strategist_node)
    graph.add_node("scan_compiler", scan_compiler_node)

    graph.add_edge(START, "social_sentiment")
    graph.add_edge(START, "news_analyst")
    graph.add_edge("social_sentiment", "extract_tickers")
    graph.add_edge("news_analyst", "extract_tickers")
    graph.add_edge("extract_tickers", "macro_strategist")
    graph.add_edge("macro_strategist", "scan_compiler")
    graph.add_edge("scan_compiler", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════
#  Analyze Graph — on-demand deep ticker analysis
# ═══════════════════════════════════════════════════════════════════


def build_analyze_graph(
    sec_edgar: "SECEdgarClient",
    yfinance: "YFinanceClient",
    massive: "MassiveClient | None" = None,
    crawler: "Crawl4AICrawlerProvider | None" = None,
    twitter_api_key: str | None = None,
) -> Any:
    """Build the on-demand ticker analysis graph.

    START → ticker_research (all tickers, parallel with L2)
    START → route_to_L2 (Send per ticker: company + price)
      → l2_join → debate (bull/bear) → trader_gate → trader
      → analyze_compiler → END

    Input state must include ``target_tickers`` (from POST body).
    """

    # ── Ticker Research (all tickers, runs parallel with L2) ────

    async def ticker_research_node(state: AnalyzeState) -> dict[str, Any]:
        tickers = state.get("target_tickers", [])
        if not tickers:
            return {"ticker_research": {}}

        logger.info("TickerResearchAnalyst starting", tickers=tickers)
        twitter_client = TwitterClient(api_key=twitter_api_key) if twitter_api_key else None
        try:
            current = date.fromisoformat(state["current_date"])
            deps = TickerResearchDeps(twitter_client=twitter_client, current_date=current)
            result = await analyze_ticker_research(tickers, deps)
            logger.info(
                "TickerResearchAnalyst complete",
                tickers_researched=len(result.research),
            )
            return {"ticker_research": result.model_dump(mode="json")}
        except Exception:
            logger.exception("TickerResearchAnalyst failed", tickers=tickers)
            return {"ticker_research": {"error": True}}
        finally:
            if twitter_client:
                await twitter_client.stop()

    # ── Layer 2: Company + Price per ticker ─────────────────────

    async def company_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
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

    async def price_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
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

    # ── Fan-out router ──────────────────────────────────────────

    def route_to_L2(state: AnalyzeState) -> list[Send] | str:  # noqa: N802
        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No tickers provided — skipping to compiler")
            return "analyze_compiler"
        logger.info("Routing to Layer 2", tickers=tickers)
        sends: list[Send] = []
        for t in tickers:
            sends.append(Send("company_analyst", {**state, "ticker": t}))
            sends.append(Send("price_analyst", {**state, "ticker": t}))
        return sends

    # ── L2 Join ─────────────────────────────────────────────────

    async def l2_join_node(state: AnalyzeState) -> dict[str, Any]:
        logger.info(
            "L2 join complete",
            company_analyses=len(state.get("company_analyses", [])),
            price_analyses=len(state.get("price_analyses", [])),
        )
        return {}

    # ── Debate routing ──────────────────────────────────────────

    debate_subgraph = build_debate_subgraph()

    def debate_router(state: AnalyzeState) -> list[str | Send]:
        tickers = state.get("target_tickers", [])
        if not tickers:
            return ["analyze_compiler"]

        rounds = get_settings().debate_rounds
        sends: list[str | Send] = []
        for t in tickers:
            if rounds == 0:
                sends.append(Send("bull_researcher", {**state, "ticker": t}))
                sends.append(Send("bear_researcher", {**state, "ticker": t}))
            else:
                sends.append(Send("ticker_debate", {**state, "ticker": t}))
        return sends

    # ── Bull / Bear / Debate nodes ──────────────────────────────

    async def bull_researcher_node(state: dict[str, Any]) -> dict[str, Any]:
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

    async def bear_researcher_node(state: dict[str, Any]) -> dict[str, Any]:
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

    async def ticker_debate_node(state: dict[str, Any]) -> dict[str, Any]:
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
                "watchlist_context": state.get("watchlist_context", {}),
                "ticker_research": state.get("ticker_research", {}),
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

    # ── Trader gate + routing ───────────────────────────────────

    async def trader_gate_node(state: AnalyzeState) -> dict[str, Any]:
        logger.info(
            "Trader gate complete",
            bull_analyses=len(state.get("bull_analyses", [])),
            bear_analyses=len(state.get("bear_analyses", [])),
        )
        return {}

    def trader_router(state: AnalyzeState) -> list[str | Send]:
        tickers = state.get("target_tickers", [])
        if not tickers:
            return ["analyze_compiler"]

        mode = get_settings().trader_mode
        if mode == "portfolio":
            logger.info("Routing to Trader (portfolio)", tickers=tickers)
            return [Send("trader", {**state, "tickers": tickers, "mode": "portfolio"})]
        else:
            logger.info("Routing to Trader (per_ticker)", tickers=tickers)
            return [Send("trader", {**state, "ticker": t, "mode": "per_ticker"}) for t in tickers]

    async def trader_node(state: dict[str, Any]) -> dict[str, Any]:
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

    # ── Compiler ────────────────────────────────────────────────

    async def analyze_compiler_node(state: AnalyzeState) -> dict[str, Any]:
        try:
            brief = compile_brief(dict(state))
            return {"brief": brief}
        except Exception:
            logger.exception("Analyze compiler failed", date=state.get("current_date"))
            return {"brief": {}}

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(AnalyzeState)

    graph.add_node("ticker_research", ticker_research_node)
    graph.add_node("company_analyst", company_analyst_node)  # type: ignore[type-var]
    graph.add_node("price_analyst", price_analyst_node)  # type: ignore[type-var]
    graph.add_node("l2_join", l2_join_node, defer=True)
    graph.add_node("bull_researcher", bull_researcher_node)  # type: ignore[type-var]
    graph.add_node("bear_researcher", bear_researcher_node)  # type: ignore[type-var]
    graph.add_node("ticker_debate", ticker_debate_node)  # type: ignore[type-var]
    graph.add_node("trader_gate", trader_gate_node, defer=True)
    graph.add_node("trader", trader_node)  # type: ignore[type-var]
    graph.add_node("analyze_compiler", analyze_compiler_node, defer=True)

    # START → ticker_research (all tickers) in parallel with L2 fan-out
    graph.add_edge(START, "ticker_research")
    graph.add_conditional_edges(
        START, route_to_L2, ["company_analyst", "price_analyst", "analyze_compiler"]
    )

    graph.add_edge("ticker_research", "l2_join")
    graph.add_edge("company_analyst", "l2_join")
    graph.add_edge("price_analyst", "l2_join")

    # Debate routing
    graph.add_conditional_edges(
        "l2_join",
        debate_router,
        ["bull_researcher", "bear_researcher", "ticker_debate", "analyze_compiler"],
    )

    graph.add_edge("bull_researcher", "trader_gate")
    graph.add_edge("bear_researcher", "trader_gate")
    graph.add_edge("ticker_debate", "trader_gate")

    # Trader routing
    graph.add_conditional_edges(
        "trader_gate",
        trader_router,
        ["trader", "analyze_compiler"],
    )

    graph.add_edge("trader", "analyze_compiler")
    graph.add_edge("analyze_compiler", END)

    return graph.compile()
