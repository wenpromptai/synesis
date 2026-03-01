"""Stage 2: Smart Analyzer - Consolidated analysis with research context.

This module provides the unified Stage 2 analysis that:
- Takes message + extraction + web results + polymarket markets
- Makes ALL informed judgments: tickers, sentiment, thesis, historical context
- Generates ticker-level bull/bear analysis
- Evaluates prediction market opportunities
- Returns unified SmartAnalysis output

Architecture follows PydanticAI best practices:
- Typed deps via `deps_type=AnalyzerDeps` dataclass
- Dynamic system prompt injection via `@agent.system_prompt` decorator
- Pre-fetched context passed via deps, not user prompt
- Simple user prompt focused on the task
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker as _verify_ticker
from synesis.processing.news.models import (
    MACRO_TOPICS,
    LightClassification,
    SmartAnalysis,
    UnifiedMessage,
)

if TYPE_CHECKING:
    from synesis.markets.polymarket import PolymarketClient, SimpleMarket
    from synesis.providers.base import TickerProvider

logger = get_logger(__name__)


# =============================================================================
# Dependencies (PydanticAI typed deps pattern)
# =============================================================================


@dataclass
class AnalyzerDeps:
    """Dependencies for Stage 2 Smart Analyzer.

    Following PydanticAI best practices, all pre-fetched context is passed
    via this typed dataclass rather than embedded in the user prompt.
    """

    message: UnifiedMessage
    extraction: LightClassification
    web_results: list[str]
    markets_text: str
    http_client: httpx.AsyncClient | None = None  # Optional for additional searches
    ticker_provider: "TickerProvider | None" = None


# =============================================================================
# System Prompt (Static Base)
# =============================================================================

SMART_ANALYZER_SYSTEM_PROMPT = """You are an expert financial analyst with a disciplined institutional approach.

## Your Investment Philosophy
- Value early signal detection — identify trends and tickers before they become consensus
- Balance conviction with risk: high confidence for direct impacts, speculative for emerging trends
- Consider what is already priced in vs. new information (surprise drives alpha)
- Second-order effects can be valuable if the causal chain is clear

You have been given:
- Breaking news with entity extraction (Stage 1)
- Web research results (analyst estimates, historical data)
- Polymarket prediction markets (pre-searched)

Your job is to make ALL informed judgments about this news.

## Research Process (MANDATORY before making predictions)

Use `web_search` (with recency="year" or "month") to find historical precedent BEFORE forming your thesis.
You need to understand how markets reacted to similar events in the past.

**What to search for, by event type:**

1. **Macro events** (Fed, CPI, NFP, geopolitical): Search for the same event type at similar magnitude.
   Find how SPY, bonds, and sector ETFs reacted. Note the market conditions at the time
   (e.g., was the market already pricing in cuts? was volatility elevated?).

2. **Earnings events**: Search for the company's past earnings surprises at similar magnitude.
   Find the stock's typical reaction pattern (gap up/down, fade, follow-through).

3. **Corporate events** (M&A, guidance, regulatory): Search for similar events at comparable companies.
   Find how acquirer/target stocks moved and over what timeframe.

**Recency & Regime Matching:**
- Prioritize recent precedents (last 1-2 years) — recent market structure is more representative
- Look for precedents under SIMILAR conditions: same rate environment, volatility regime, sentiment
- A recent event under similar conditions >> an older event from a different regime, even if the
  older event is a closer event-type match

If no relevant precedent exists, say so — do NOT fabricate or force irrelevant data.

## Your Tasks

### 1. Identify Affected Securities

Only include tickers with DIRECT, MATERIAL impact.

**Ticker Verification Workflow:**
1. Extract potential ticker from the news text
2. Call `verify_ticker(ticker)` to confirm it exists
3. If VERIFIED: include with the company name returned
4. If NOT FOUND: use `web_search("{ticker} stock ticker price")` to verify
5. If still unclear: exclude the ticker
6. Do NOT verify macro ETF proxies (GLD, USO, SPY, TLT, UUP, VIXY, EEM) — they are pre-defined

**Causal Link Requirements** (must meet at least ONE):
- Revenue or earnings impact (>5% expected change)
- Regulatory/legal status change
- Competitive position materially affected
- Supply chain or key partnerships disrupted/formed

**Exclusion Criteria** (DO NOT include if):
- Competitor mentioned for context only
- Company in same sector but not directly affected
- Parent/subsidiary unless news specifically affects them
- Historical comparison ("similar to when X happened to AAPL")
- Tangentially related through industry trends

**Output per ticker** — `relevance_score` ≥ 0.6 required:
- 0.9–1.0: Primary subject of the news
- 0.7–0.89: Directly named or materially affected
- 0.6–0.69: Clear secondary impact
- `relevance_reason`: ONE sentence, ≤100 characters

**Examples**:
- "Apple announces 20% revenue miss in China"
  ✅ AAPL (0.95): Primary subject, direct earnings impact
  ❌ MSFT: Competitor, not directly affected
  ❌ TSM: Supplier, but no specific impact mentioned

- "Fed cuts rates by 50bps"
  ✅ Tickers directly named or with clear NIM/rate sensitivity
  ❌ Individual banks unless specifically named with impact

**Macro Asset-Class ETFs** (when primary_topics include macro themes):

When the news is a broad macro event, ALSO assess impact on major asset classes using
these ETF proxies inside `ticker_analyses` (regular TickerAnalysis entries):

| ETF  | Asset Class     | When to include                                             |
|------|-----------------|-------------------------------------------------------------|
| GLD  | Gold            | Geopolitical risk, inflation, real-rate shifts              |
| USO  | Crude Oil       | Middle-East conflict, OPEC, sanctions, supply disruption    |
| SPY  | US Equities     | Broad risk-on / risk-off, recession fears, policy shifts    |
| TLT  | Long Treasuries | Rate decisions, flight-to-safety, duration bets             |
| UUP  | US Dollar       | Fed policy divergence, trade policy, reserve-currency flows |
| VIXY | Volatility      | Uncertainty spikes, tail-risk events                        |
| EEM  | EM Equities     | Dollar strength, tariffs, EM-specific contagion             |

Rules:
- Only include ETFs with a CLEAR causal link — do NOT add all 7 for every macro story
- Apply the SAME relevance scoring (≥ 0.6) and analysis depth as stock tickers
- Set `company_name` to the asset class label (e.g., "Gold", "US Equities")

### 2. Investment Thesis & Sentiment

- `primary_thesis`: ONE sentence, ≤150 characters. Be specific about expected outcome and timeframe.
  Example: "Fed 25bp cut signals dovish pivot; financials may underperform as NIM compression accelerates"
- `thesis_confidence`: 0.0–1.0 (use calibration table)
- `sentiment`: bullish | bearish | neutral
- `sentiment_score`: -1.0 (max bearish) to 1.0 (max bullish)

**Confidence Calibration:**

| Score     | Criteria                                                          |
|-----------|-------------------------------------------------------------------|
| 0.9–1.0   | Unambiguous direct impact, clear causal link, historical precedent |
| 0.7–0.89  | Strong relationship, some uncertainty in magnitude or timing       |
| 0.5–0.69  | Plausible connection, emerging trend, multiple interpretations     |
| 0.3–0.49  | Weak but interesting signal, early trend detection opportunity     |
| <0.3      | Very tenuous connection, no clear causal path — do not include     |

Lower confidence plays (0.3–0.69) can still be valuable for early trend detection.

### 3. Ticker-Level Analysis

For each ticker from Task 1:
- `bull_thesis`: Why positive (≤100 characters)
- `bear_thesis`: Why negative (≤100 characters)
- `net_direction`: bullish | bearish | neutral
- `conviction`: 0.0–1.0
- `time_horizon`: intraday | days | weeks | months
- `catalysts`: Key catalysts to watch
- `risk_factors`: Key risks to the thesis

### 4. Historical Context & Market Patterns

Synthesize the historical research you did earlier (Research Process) into structured output.
If you haven't searched yet, use `web_search` now — do NOT skip this for high/medium impact events.

**a) Precedent Events** — cite 1-3 events, most-recent-first, with SIMILAR characteristics
  - Include the date, what happened, and the market conditions at the time
  - Example: "March 2020: Emergency 50bp cut during COVID sell-off; SPY rallied 9% in 3 days but
    gave it all back within a week as panic resumed. VIX was at 40+ (elevated fear)."

**b) Quantified Market Reactions** — for each precedent:
  - Immediate (first 15-60 min), short-term (1-5 days), extended (1-4 weeks)
  - Whether the move held, reversed, or accelerated

**c) Regime Comparison** — for each precedent, compare to today's conditions:
  - Was the event expected or a surprise? Compounding factors?
  - Rate cycle (hiking/cutting/paused), VIX regime (<15 complacent, 15-25 normal, >25 elevated)
  - Sentiment/positioning (risk-on, risk-off, crowded trades), inflation backdrop
  - Similar conditions → weight heavily; different conditions → discount and explain why

**d) Key Differences** — how does the current situation differ from the precedents?
  - Different magnitude, regime, or positioning? Higher or lower impact expected?

Output fields:
- `historical_context`: Precedent events with dates, market conditions, and quantified reactions
- `typical_market_reaction`: Reaction pattern — initial move, reversal probability, sector rotation

If no relevant precedent: "No relevant historical precedent found — analysis based on first principles"

### 5. Evaluate Prediction Markets

**CRITICAL: You MUST return a MarketEvaluation for EVERY market in the Polymarket table.**

For each market row:
1. **Check Relevance**: Does the news DIRECTLY affect this market outcome?
   - Many keyword matches are FALSE POSITIVES
   - "Trump praises Visa" ≠ "Trump deportation" markets
   - Meta/meme markets are NEVER relevant unless thesis is specifically about prediction market behavior
2. **If Relevant**: Evaluate mispricing
   - undervalued: YES price too low (buy YES)
   - overvalued: YES price too high (buy NO)
   - fair: Within 5% of estimate
3. **Edge**: fair_price − current_price. Only recommend trades with |edge| > 5% and confidence > 0.5

**MarketEvaluation fields:**
- `market_id`: Copy EXACTLY from the table
- `market_question`: The question text
- `is_relevant`: true/false
- `relevance_reasoning`: Why is/isn't this relevant
- `current_price`: YES price from table
- `estimated_fair_price`: Your estimate (null if not relevant)
- `edge`: fair − current (null if not relevant)
- `verdict`: undervalued | overvalued | fair | skip
- `confidence`: 0.0–1.0
- `reasoning`: ≤120 characters
- `recommended_side`: yes | no | skip"""


class SmartAnalyzer:
    """Stage 2: Consolidated smart analyzer.

    Takes message + extraction + web results + polymarket markets
    and produces a unified SmartAnalysis with all informed judgments.

    Follows PydanticAI best practices:
    - Typed deps via AnalyzerDeps dataclass
    - Dynamic system prompt injection for context
    - Simple user prompt focused on the task
    """

    def __init__(self, polymarket_client: PolymarketClient | None = None) -> None:
        self._agent: Agent[AnalyzerDeps, SmartAnalysis] | None = None
        self._polymarket = polymarket_client
        self._own_polymarket = polymarket_client is None

    @property
    def agent(self) -> Agent[AnalyzerDeps, SmartAnalysis]:
        """Get or create the analyzer agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    @property
    def polymarket(self) -> PolymarketClient:
        """Get or create the Polymarket client."""
        if self._polymarket is None:
            from synesis.markets.polymarket import PolymarketClient

            self._polymarket = PolymarketClient()
            self._own_polymarket = True
        return self._polymarket

    async def close(self) -> None:
        """Close resources."""
        if self._own_polymarket and self._polymarket:
            await self._polymarket.close()
            self._polymarket = None

    def _create_agent(self) -> Agent[AnalyzerDeps, SmartAnalysis]:
        """Create the PydanticAI agent for smart analysis.

        Uses PydanticAI patterns:
        - deps_type=AnalyzerDeps for typed dependencies
        - @agent.system_prompt decorator for dynamic context injection
        - Tools access deps via RunContext[AnalyzerDeps]
        """
        # Use smart model for complex reasoning
        model = create_model(smart=True)

        agent: Agent[AnalyzerDeps, SmartAnalysis] = Agent(
            model,
            deps_type=AnalyzerDeps,
            output_type=PromptedOutput(SmartAnalysis),
            system_prompt=SMART_ANALYZER_SYSTEM_PROMPT,
        )

        # Dynamic system prompt: inject pre-fetched research context
        @agent.system_prompt
        def inject_research_context(ctx: RunContext[AnalyzerDeps]) -> str:
            """Dynamically inject pre-fetched research context.

            This follows PydanticAI best practice of using @agent.system_prompt
            to inject context from deps rather than embedding in user prompt.
            """
            msg = ctx.deps.message
            ext = ctx.deps.extraction

            # Format web results
            web_section = ""
            if ctx.deps.web_results:
                for i, result in enumerate(ctx.deps.web_results, 1):
                    web_section += f"\n### Research {i}:\n{result}\n"
            else:
                web_section = "\nNo web research available.\n"

            # Format entities
            entities_str = ", ".join(ext.all_entities) if ext.all_entities else "None"

            return f"""
## Breaking News (Current Analysis Subject)
Source: {msg.source_account} ({msg.source_platform.value})
Timestamp: {msg.timestamp.isoformat()}

Message:
{msg.text}

## Stage 1 Extraction
Primary Entity: {ext.primary_entity}
All Entities: {entities_str}
Primary Topics: {", ".join(t.value for t in ext.primary_topics) or "other"}
Secondary Topics: {", ".join(t.value for t in ext.secondary_topics) or "none"}
News Category: {ext.news_category.value}
Summary: {ext.summary}

## Web Research (Pre-Fetched)
{web_section}

## Polymarket Markets (Pre-Searched)
{ctx.deps.markets_text or "No prediction markets found."}""" + (
                "\n\n⚠️ MACRO EVENT — include macro asset-class ETF proxies in Task 1"
                if set(ext.primary_topics) & MACRO_TOPICS
                else ""
            )

        # Tool: Check Relevance (structured thinking tool)
        @agent.tool
        async def check_market_relevance(
            ctx: RunContext[AnalyzerDeps],
            market_question: str,
            reasoning: str,
        ) -> str:
            """Check if a prediction market is relevant to the news.

            Use this to filter out false positive keyword matches.
            Think about whether the news DIRECTLY affects the market outcome.

            Args:
                market_question: The prediction market question
                reasoning: Your reasoning for why it might be relevant

            Returns:
                Guidance on relevance assessment
            """
            news_summary = ctx.deps.extraction.summary

            return f"""Relevance Check Framework:

News: {news_summary}
Market: {market_question}
Your reasoning: {reasoning}

To determine relevance:
1. Does the news DIRECTLY affect the market outcome?
2. Is the connection causal or just coincidental keywords?
3. Would this news change a rational person's probability estimate?

If indirect or keywords match but topics differ, mark as NOT relevant."""

        # Tool: Verify ticker
        @agent.tool
        async def verify_ticker(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Verify if a US ticker symbol exists.

            Use this tool to validate US tickers BEFORE including them in your analysis.
            For non-US tickers, use web_search instead.

            Args:
                ticker: The US ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")

            Returns:
                Verification result - either VERIFIED with company name, NOT FOUND, or error
            """
            return await _verify_ticker(ticker, ctx.deps.ticker_provider)

        # Tool: Web search for additional context
        @agent.tool
        async def web_search(
            ctx: RunContext[AnalyzerDeps],
            query: str,
            recency: str = "week",
        ) -> str:
            """Search the web for additional information.

            Use this tool to gather context when pre-fetched research is insufficient.

            RECENCY GUIDE (default: "week"):
            - "day": Breaking news, very recent developments (last 24h)
            - "week": Recent analysis, market commentary (last 7 days) - GOOD DEFAULT
            - "month": Recent context, analyst reports (last 30 days)
            - "year": Historical precedent, similar past events (last 12 months)

            WHEN TO USE:
            1. Find historical precedent for similar events (recency="year" or "month")
               - Search for events with SIMILAR characteristics (magnitude, surprise, sector)
               - "Fed rate cut 25bps market reaction"
               - "earnings beat 10% stock reaction"
            2. Verify tickers or companies not in pre-fetch
            3. Get additional context on unfamiliar situations

            IMPORTANT:
            - Only include historical data if it MATCHES the current context
            - If no relevant matches found, it's better to have NO historical context
            - Don't force irrelevant historical data just to fill space

            Args:
                query: Search query (be specific about what pattern you're looking for)
                recency: Time range - "day", "week", "month", or "year" (default: "week")

            Returns:
                Formatted search results
            """
            if ctx.deps.http_client is None:
                return "Web search not available (no HTTP client configured)."

            try:
                from synesis.processing.common.web_search import (
                    format_search_results,
                    search_market_impact,
                )

                # Validate recency parameter
                valid_recency: Literal["day", "week", "month", "year"] = (
                    recency if recency in ("day", "week", "month", "year") else "week"  # type: ignore[assignment]
                )
                results = await search_market_impact(query, count=5, recency=valid_recency)
                return format_search_results(results)
            except Exception as e:
                logger.warning("Web search failed", query=query, error=str(e))
                return f"Search failed: {e}"

        return agent

    async def search_polymarket(self, keywords: list[str]) -> str:
        """Search Polymarket for prediction markets in parallel.

        Args:
            keywords: Keywords to search

        Returns:
            Formatted list of markets with questions and prices
        """
        import asyncio

        from synesis.config import get_settings

        settings = get_settings()
        keywords_to_search = keywords[: settings.polymarket_max_keywords]
        logger.debug("Searching Polymarket", keywords=keywords_to_search)

        async def safe_search(keyword: str) -> list[SimpleMarket]:
            try:
                return await self.polymarket.search_markets(keyword, limit=5)
            except Exception as e:
                logger.warning("Polymarket search failed", keyword=keyword, error=str(e))
                return []

        # Parallel search all keywords
        search_results = await asyncio.gather(*[safe_search(kw) for kw in keywords_to_search])

        # Dedupe results
        all_markets: list[SimpleMarket] = []
        seen_ids: set[str] = set()
        for markets in search_results:
            for m in markets:
                # Filter: only include active, non-closed markets (defense in depth)
                # The search_markets() method should already filter, but double-check here
                if m.id not in seen_ids and m.is_active and not m.is_closed:
                    seen_ids.add(m.id)
                    all_markets.append(m)

        if not all_markets:
            return "No markets found for these keywords."

        # Format as structured table for better LLM parsing
        # Include status column as defense in depth (all should be ACTIVE after filtering)
        lines = [
            f"Found {len(all_markets)} markets. You MUST create a MarketEvaluation for each:\n",
            "| Market ID | Question | YES Price | Status | 24h Volume |",
            "|-----------|----------|-----------|--------|------------|",
        ]
        for m in all_markets[:7]:  # Limit to 7 markets for faster LLM analysis
            question = m.question
            status = "ACTIVE" if m.is_active and not m.is_closed else "CLOSED"
            lines.append(
                f"| {m.id} | {question} | ${m.yes_price:.2f} | {status} | ${m.volume_24h:,.0f} |"
            )

        lines.append("")
        lines.append("**IMPORTANT**: For each row above, create a MarketEvaluation with:")
        lines.append("- market_id: Copy the exact ID from the 'Market ID' column")
        lines.append("- current_price: Use the 'YES Price' value (without $)")
        lines.append("- market_question: Use the full question text")

        logger.debug("Polymarket search complete", markets_found=len(all_markets))
        return "\n".join(lines)

    async def analyze(
        self,
        message: UnifiedMessage,
        extraction: LightClassification,
        web_results: list[str],
        markets_text: str,
        http_client: httpx.AsyncClient | None = None,
        ticker_provider: "TickerProvider | None" = None,
    ) -> SmartAnalysis | None:
        """Analyze news with full research context.

        This is the main Stage 2 entry point. It takes all pre-fetched context
        and produces a unified SmartAnalysis with all informed judgments.

        Args:
            message: Original news message
            extraction: Stage 1 extraction result
            web_results: Pre-fetched web search results
            markets_text: Pre-fetched Polymarket search results
            http_client: Optional HTTP client for additional searches
            ticker_provider: Optional TickerProvider for ticker verification

        Returns:
            SmartAnalysis with tickers, sentiment, thesis, and market evaluations
        """
        log = logger.bind(
            message_id=message.external_id,
            primary_entity=extraction.primary_entity,
        )

        log.info(
            "Stage 2 smart analysis starting",
            all_entities=extraction.all_entities,
            web_results_count=len(web_results),
        )

        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            web_results=web_results,
            markets_text=markets_text,
            http_client=http_client,
            ticker_provider=ticker_provider,
        )

        # Build user prompt — market instructions only when Polymarket was searched
        if markets_text:
            market_instructions = """5. Evaluate EACH prediction market from the table
   - Return a MarketEvaluation for EVERY market in the Polymarket table
   - Copy the market_id EXACTLY from the table
   - Set is_relevant=false and verdict="skip" for unrelated markets"""
        else:
            market_instructions = ""

        user_prompt = f"""Analyze this news. Determine:
1. Affected tickers (include macro ETFs if macro event)
2. Investment thesis, sentiment, and confidence
3. Ticker-level analysis with bull/bear thesis for each
4. Historical context from web research
{market_instructions}

Focus on DIRECT impacts. Be conservative with confidence scores."""

        try:
            result = await self.agent.run(user_prompt, deps=deps)
            output = result.output

            # Post-process: Filter low-relevance tickers (threshold: 0.6)
            original_ticker_count = len(output.ticker_analyses)
            output.ticker_analyses = [t for t in output.ticker_analyses if t.relevance_score >= 0.6]
            output.tickers = [t.ticker for t in output.ticker_analyses]

            filtered_count = original_ticker_count - len(output.ticker_analyses)
            if filtered_count > 0:
                log.debug(
                    "Filtered low-relevance tickers",
                    filtered=filtered_count,
                    remaining=len(output.ticker_analyses),
                )

            log.info(
                "Stage 2 smart analysis complete",
                tickers=output.tickers,
                sentiment=output.sentiment.value,
                sentiment_score=output.sentiment_score,
                thesis_confidence=f"{output.thesis_confidence:.0%}",
                markets_evaluated=len(output.market_evaluations),
                has_edge=output.has_tradable_edge,
            )

            return output

        except Exception as e:
            log.exception("Stage 2 analysis failed", error=str(e))
            # Return None to signal failure - callers must handle this
            return None
