"""Stage 2: Smart analysis with LLM — entity extraction, sentiment, ETF impact, Polymarket.

Takes a message + Stage 1 extraction (impact score + matched tickers) and produces
a full SmartAnalysis using a smart LLM model with web_search, web_read, and
search_polymarket tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.news.models import (
    LightClassification,
    SmartAnalysis,
    UnifiedMessage,
)

if TYPE_CHECKING:
    from synesis.markets.polymarket import PolymarketClient

logger = get_logger(__name__)

# Tool budgets per analysis
_WEB_SEARCH_CAP = 3  # Brave search calls


@dataclass
class AnalyzerDeps:
    """Dependencies for Stage 2 Smart Analyzer."""

    message: UnifiedMessage
    extraction: LightClassification
    http_client: httpx.AsyncClient | None = None
    web_search_calls: int = 0  # Budget: 3 searches max


# =============================================================================
# System Prompt
# =============================================================================

SMART_ANALYZER_SYSTEM_PROMPT = """You are an expert financial analyst. Your job is to analyze breaking news and produce actionable intelligence.

## Your Tools

- `web_search(query, recency)` — Search the web. Returns titles, snippets, and URLs. Budget: 3 calls.
  Recency: "day" (last 24h), "week" (7d), "month" (30d), "year" (12mo), "all" (no filter).
- `web_read(url)` — Read a web page in full (~4000 chars of article content). Unlimited calls.
- `search_polymarket(query)` — Search Polymarket for active prediction markets.

## Workflow (follow this order)

### Step 1: Identify entities

Extract `all_entities` — every company, person, institution, or country directly involved.
- INCLUDE: named companies (NVIDIA, Marvell), people (Trump, Powell), institutions (Fed, ECB, OPEC), countries
- EXCLUDE: news sources (@DeItaone, Bloomberg), generic references ("analysts", "investors", "markets")

### Step 2: Research context

BEFORE forming any thesis, use your tools to understand the situation:

1. `web_search` for the specific event to get current context and immediate reactions
2. `web_search` for historical precedent — similar events in the past. Prefer RECENT precedents (last 1-2 years) over older ones. A recent event under similar market conditions is more valuable than an older but closer event-type match.
3. `web_read` the most relevant 1-2 URLs to get full article content

Put your findings in:
- `historical_context`: Cite 1-3 precedent events. For each: date, what happened, market reaction (quantified: "SPY -2.3% in 2 days"), and how current conditions compare (rate environment, VIX level, positioning). Rank by recency — most recent first.
- `typical_market_reaction`: The pattern — initial move magnitude, whether it held/reversed, timeframe, which sectors led/lagged.

If no relevant precedent exists, say so honestly. Do NOT fabricate data.

### Step 3: Form thesis

Now that you have research context, write `primary_thesis`:
- ONE sentence, ≤150 characters
- Be specific: name the expected outcome, affected assets, and rough timeframe
- Example: "Fed 50bp surprise cut likely triggers 3-5% SPY rally over 5 days as market prices in dovish pivot"

### Step 4: Assess macro & sector ETF impact

For each ETF that is DIRECTLY and MATERIALLY affected by this news, provide:
- `ticker`: the ETF symbol
- `sentiment_score`: -1.0 (max bearish) to +1.0 (max bullish)
  - ±0.8 to ±1.0: Strong, high-conviction directional impact
  - ±0.5 to ±0.7: Moderate impact, clear causal link
  - ±0.2 to ±0.4: Mild/indirect impact
  - 0.0: No meaningful directional bias
- `reason`: ≤100 chars explaining the causal link

Only include ETFs where you can articulate a clear cause-and-effect. Leave out ETFs with vague or tangential connections.

**Macro ETFs** → `macro_impact`:
| ETF  | Asset Class     | Triggers                                                    |
|------|-----------------|-------------------------------------------------------------|
| GLD  | Gold            | Geopolitical risk, inflation surprise, real-rate shifts     |
| USO  | Crude Oil       | Middle-East conflict, OPEC decisions, sanctions, supply shock|
| SPY  | US Equities     | Broad risk-on/off, recession fears, major policy shifts     |
| TLT  | Long Treasuries | Rate decisions, flight-to-safety, inflation expectations    |
| UUP  | US Dollar       | Fed divergence, trade policy, reserve-currency flows        |
| VIXY | Volatility      | Uncertainty spikes, surprise events, tail risk              |
| EEM  | EM Equities     | Dollar strength, tariffs, EM contagion                      |

**Sector ETFs** → `sector_impact`:
| ETF  | Sector              | ETF  | Sector              |
|------|---------------------|------|---------------------|
| XLE  | Energy              | XLF  | Financials          |
| XLB  | Materials           | XLY  | Consumer Disc.      |
| XLI  | Industrials         | XLP  | Consumer Staples    |
| XLU  | Utilities           | XLK  | Technology          |
| XLV  | Healthcare          | XLC  | Communications      |
|      |                     | XLRE | Real Estate         |

### Step 5: Evaluate prediction markets

After your analysis is formed, search for relevant prediction markets:

1. Call `search_polymarket(query)` with terms derived from your entity/event understanding
   - Try more than 1 searches with different angles (e.g. entity name, then event type) to get the relevant markets. If you find 0 markets, that's a valid outcome as well.
2. For EVERY market returned, create a MarketEvaluation:
   - **Verify relevance**: Does this news DIRECTLY affect this market's outcome? Many keyword matches are false positives. A market about "Trump approval" is NOT relevant to "Trump tariff" news.
   - **Check if still actionable**: Ignore markets that appear settled (YES price > 0.95 or < 0.05) or have negligible volume.
   - **If relevant and actionable**: Estimate fair probability, calculate edge (fair − current). Only recommend if |edge| > 5% and confidence > 0.5.
   - **verdict**: undervalued (buy YES) | overvalued (buy NO) | fair | skip"""


class SmartAnalyzer:
    """Stage 2: LLM analysis — entities, topics, sentiment, ETF impact, Polymarket.

    Polymarket search is done internally (no prefetch needed from processor).
    """

    def __init__(self, polymarket_client: "PolymarketClient | None" = None) -> None:
        self._agent: Agent[AnalyzerDeps, SmartAnalysis] | None = None
        self._polymarket = polymarket_client
        self._own_polymarket = polymarket_client is None

    @property
    def agent(self) -> Agent[AnalyzerDeps, SmartAnalysis]:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    @property
    def polymarket(self) -> "PolymarketClient":
        if self._polymarket is None:
            from synesis.markets.polymarket import PolymarketClient

            self._polymarket = PolymarketClient()
            self._own_polymarket = True
        return self._polymarket

    async def close(self) -> None:
        if self._own_polymarket and self._polymarket:
            await self._polymarket.close()
            self._polymarket = None

    def _create_agent(self) -> Agent[AnalyzerDeps, SmartAnalysis]:
        model = create_model(smart=True)

        agent: Agent[AnalyzerDeps, SmartAnalysis] = Agent(
            model,
            deps_type=AnalyzerDeps,
            output_type=PromptedOutput(SmartAnalysis),
            system_prompt=SMART_ANALYZER_SYSTEM_PROMPT,
        )

        # Dynamic system prompt: inject message + Stage 1 context
        @agent.system_prompt
        def inject_context(ctx: RunContext[AnalyzerDeps]) -> str:
            msg = ctx.deps.message
            ext = ctx.deps.extraction
            tickers_str = ", ".join(ext.matched_tickers) if ext.matched_tickers else "None"

            return f"""
## Breaking News
Source: {msg.source_account} ({msg.source_platform.value})
Timestamp: {msg.timestamp.isoformat()}

Message:
{msg.text}

## Stage 1 (Rule-Based)
Matched Tickers: {tickers_str}
Impact Score: {ext.impact_score}/100"""

        # Tool: Web search — returns titles + snippets + URLs
        @agent.tool
        async def web_search(
            ctx: RunContext[AnalyzerDeps],
            query: str,
            recency: str = "week",
        ) -> str:
            """Search the web for information. Returns titles, short snippets, and URLs.

            Use this to find relevant articles, then call web_read(url) on the best ones
            to get the full content.

            Budget: 3 searches max.
            Recency: "day" | "week" | "month" | "year" | "all"

            Args:
                query: Search query (be specific, e.g. "Fed 50bps cut 2024 market reaction SPY")
                recency: Time range (default: "week")
            """
            if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
                return f"Search limit reached ({_WEB_SEARCH_CAP} calls). Use web_read on URLs you already have."
            ctx.deps.web_search_calls += 1

            try:
                recency_map = {"all": "none"}
                mapped = recency_map.get(recency, recency)
                valid_recency: Recency = (
                    mapped if mapped in ("day", "week", "month", "year", "none") else "week"  # type: ignore[assignment]
                )
                results = await search_market_impact(query, count=5, recency=valid_recency)
                return format_search_results(results)
            except Exception as e:
                logger.warning("Web search failed", query=query, error=str(e))
                return f"Search failed: {e}"

        # Tool: Web read — crawls a URL and returns full article content
        @agent.tool
        async def web_read(
            ctx: RunContext[AnalyzerDeps],
            url: str,
        ) -> str:
            """Read the full content of a web page. Use after web_search to read
            the most relevant articles in depth.

            Returns ~4000 chars of article content (nav/images stripped).
            No limit on reads — use as many as needed.

            Args:
                url: The URL to read (from web_search results)
            """
            try:
                return await read_web_page(url)
            except Exception as e:
                logger.warning("web_read failed", url=url, error=str(e))
                return f"Failed to read page: {e}"

        # Tool: Search Polymarket for prediction markets
        @agent.tool
        async def search_polymarket(
            ctx: RunContext[AnalyzerDeps],
            query: str,
        ) -> str:
            """Search Polymarket for prediction markets related to a query.

            Call this after understanding the news to find relevant prediction markets.
            Use entity names, event descriptions, or topic keywords as the query.

            Args:
                query: Search query (e.g. "NVIDIA acquisition", "Fed rate cut", "Iran oil")
            """
            try:
                markets = await self.polymarket.search_markets(query, limit=5)
            except Exception as e:
                logger.warning("Polymarket search failed", query=query, error=str(e))
                return f"Polymarket search failed: {e}"

            # Filter to active markets
            active = [m for m in markets if m.is_active and not m.is_closed]
            if not active:
                return f"No active Polymarket markets found for '{query}'."

            lines = [
                f"Found {len(active)} markets for '{query}'. Create a MarketEvaluation for each:\n",
                "| Market ID | Question | YES Price | 24h Volume |",
                "|-----------|----------|-----------|------------|",
            ]
            for m in active[:7]:
                lines.append(
                    f"| {m.id} | {m.question} | ${m.yes_price:.2f} | ${m.volume_24h:,.0f} |"
                )

            lines.append("")
            lines.append("For each row, create a MarketEvaluation with the exact market_id.")

            logger.debug("Polymarket search complete", query=query, markets_found=len(active))
            return "\n".join(lines)

        return agent

    async def analyze(
        self,
        message: UnifiedMessage,
        extraction: LightClassification,
        http_client: httpx.AsyncClient | None = None,
    ) -> SmartAnalysis | None:
        """Run Stage 2 analysis. All searching (web, Polymarket) done by LLM via tools.

        Args:
            message: Original news message
            extraction: Stage 1 classification (impact score + tickers)
            http_client: Optional HTTP client for web searches
        """
        log = logger.bind(message_id=message.external_id)

        log.info(
            "Stage 2 analysis starting",
            matched_tickers=extraction.matched_tickers,
        )

        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            http_client=http_client,
        )

        user_prompt = """Analyze this breaking news following the workflow:
1. Identify entities
2. Research context (web_search + web_read) — do this BEFORE forming opinions
3. Form thesis based on research
4. Assess macro & sector ETF impact with sentiment scores
5. Search and evaluate Polymarket prediction markets"""

        try:
            result = await self.agent.run(user_prompt, deps=deps)

            log.info(
                "Stage 2 complete",
                entities=result.output.all_entities[:5],
                thesis=result.output.primary_thesis[:100],
                macro_impact=[e.ticker for e in result.output.macro_impact],
                sector_impact=[e.ticker for e in result.output.sector_impact],
                markets_evaluated=len(result.output.market_evaluations),
            )

            return result.output
        except Exception:
            log.exception("Stage 2 analysis failed")
            return None
