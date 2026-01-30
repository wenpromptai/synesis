"""Stage 2: Smart Analyzer - Consolidated analysis with research context.

This module provides the unified Stage 2 analysis that:
- Takes message + extraction + web results + polymarket markets
- Makes ALL informed judgments: tickers, sectors, impact, direction
- Generates thesis and ticker analyses
- Evaluates market opportunities
- Returns unified SmartAnalysis output

This consolidates the old Stage 2A (InvestmentAnalyzer) and Stage 2B (SmartEvaluator).

Architecture follows PydanticAI best practices:
- Typed deps via `deps_type=AnalyzerDeps` dataclass
- Dynamic system prompt injection via `@agent.system_prompt` decorator
- Pre-fetched context passed via deps, not user prompt
- Simple user prompt focused on the task
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.llm_factory import create_model
from synesis.processing.models import (
    LightClassification,
    SmartAnalysis,
    UnifiedMessage,
)

if TYPE_CHECKING:
    from synesis.markets.polymarket import PolymarketClient, SimpleMarket

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


# =============================================================================
# System Prompt (Static Base)
# =============================================================================

# System prompt for Stage 2: Smart Analyzer
SMART_ANALYZER_SYSTEM_PROMPT = """You are an expert financial analyst. You have been given:
- Breaking news with entity extraction (Stage 1)
- Web research results (analyst estimates, historical data)
- Polymarket prediction markets (pre-searched)

Your job is to make ALL informed judgments about this news.

## Your Tasks

### 1. Determine Affected Securities (with research context)
- **tickers**: Stock tickers directly affected (e.g., AAPL, TSLA)
  - Be specific and accurate
  - Only include tickers that are DIRECTLY affected
  - Use your research to validate ticker relevance
- **sectors**: Sectors affected (e.g., semiconductors, financials)
  - Consider second-order effects

### 2. Assess Impact (with research context)
- **predicted_impact**: high | medium | low
  - high: Major policy changes, earnings surprises >10%, M&A, market-moving
  - medium: Notable sector news, guidance changes, regulatory actions
  - low: Commentary, minor updates, already priced in
- **market_direction**: bullish | bearish | neutral
  - Consider overall market effect, not just one stock

### 3. Generate Investment Thesis
- **primary_thesis**: ONE clear thesis statement
  - Be specific about expected outcome and timeframe
  - Example: "Fed 25bp cut signals dovish pivot; financials may underperform as NIM compression accelerates"
- **thesis_confidence**: 0.0 to 1.0
  - Be conservative - only high confidence for clear, unambiguous impact

### 4. Ticker-Level Analysis
For each directly affected ticker:
- Bull thesis: Why positive
- Bear thesis: Why negative
- Net direction: bullish | bearish | neutral
- Conviction: 0.0 to 1.0
- Time horizon: intraday | days | weeks | months

### 5. Sector Implications
For affected sectors:
- Direction and reasoning
- Affected subsectors

### 6. Historical Context
- What happened historically with similar news
- Typical market reaction patterns

### 7. Evaluate Prediction Markets
**CRITICAL: You MUST return a MarketEvaluation object for EVERY market in the table below.**

For each market row in the Polymarket table:
1. **Extract the Market ID**: Use the EXACT market_id from the "Market ID" column
2. **Check Relevance**: Does the news DIRECTLY affect this market?
   - Many keyword matches are FALSE POSITIVES
   - "Trump praises Visa" does NOT relate to "Trump deportation" markets
3. **If Relevant**: Evaluate if odds are mispriced
   - undervalued: YES price too low (buy YES)
   - overvalued: YES price too high (buy NO)
   - fair: Within 5% of estimate
4. **Edge Calculation**: fair_price - current_price
   - Only recommend trades with |edge| > 5% and confidence > 0.5

**MarketEvaluation Fields** (fill in for EACH market):
- market_id: Copy EXACTLY from the table (e.g., "abc123def456")
- market_question: The question text
- is_relevant: true/false
- relevance_reasoning: Why is/isn't this market relevant
- current_price: The YES price from the table
- estimated_fair_price: Your estimate (null if not relevant)
- edge: fair_price - current_price (null if not relevant)
- verdict: "undervalued" | "overvalued" | "fair" | "skip"
- confidence: 0.0 to 1.0
- reasoning: Full reasoning
- recommended_side: "yes" | "no" | "skip"

## Guidelines
- Be CONSERVATIVE with conviction and confidence scores
- Consider counter-arguments and risks
- Use web research to inform historical patterns
- Focus on DIRECT impacts, not tangential effects
- For markets, be strict about relevance filtering
- **ALWAYS return market_evaluations array with one entry per market in the table**"""


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
            output_type=SmartAnalysis,
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
Type: {msg.source_type.value}

Message:
{msg.text}

## Stage 1 Extraction
Primary Entity: {ext.primary_entity}
All Entities: {entities_str}
Event Type: {ext.event_type.value}
News Category: {ext.news_category.value}
Summary: {ext.summary}

## Web Research (Pre-Fetched)
{web_section}

## Polymarket Markets (Pre-Searched)
{ctx.deps.markets_text}"""

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

        # Tool: Additional web search (hybrid approach for edge cases)
        @agent.tool
        async def search_additional(
            ctx: RunContext[AnalyzerDeps],
            query: str,
        ) -> str:
            """Search for additional information if the pre-fetched context is insufficient.

            Use sparingly - prefer using the pre-fetched research when possible.
            Only use this if you need specific information not covered by the
            pre-fetched web research.

            Args:
                query: Search query for additional information

            Returns:
                Formatted search results or error message
            """
            if ctx.deps.http_client is None:
                return "Additional search not available (no HTTP client configured)."

            try:
                from synesis.processing.web_search import (
                    format_search_results,
                    search_market_impact,
                )

                results = await search_market_impact(query, count=3)
                return format_search_results(results)
            except Exception as e:
                logger.warning("Additional search failed", query=query, error=str(e))
                return f"Search failed: {e}"

        return agent

    async def search_polymarket(self, keywords: list[str]) -> str:
        """Search Polymarket for prediction markets.

        Args:
            keywords: Keywords to search

        Returns:
            Formatted list of markets with questions and prices
        """
        from synesis.config import get_settings

        settings = get_settings()
        logger.debug("Searching Polymarket", keywords=keywords)

        all_markets: list[SimpleMarket] = []
        seen_ids: set[str] = set()

        # Search each keyword (limit to avoid rate limits)
        for keyword in keywords[: settings.polymarket_max_keywords]:
            try:
                markets = await self.polymarket.search_markets(keyword, limit=5)
                for m in markets:
                    # Filter: only include active, non-closed markets (defense in depth)
                    # The search_markets() method should already filter, but double-check here
                    if m.id not in seen_ids and m.is_active and not m.is_closed:
                        seen_ids.add(m.id)
                        all_markets.append(m)
            except Exception as e:
                logger.warning("Polymarket search failed", keyword=keyword, error=str(e))

        if not all_markets:
            return "No markets found for these keywords."

        # Format as structured table for better LLM parsing
        # Include status column as defense in depth (all should be ACTIVE after filtering)
        lines = [
            f"Found {len(all_markets)} markets. You MUST create a MarketEvaluation for each:\n",
            "| Market ID | Question | YES Price | Status | 24h Volume |",
            "|-----------|----------|-----------|--------|------------|",
        ]
        for m in all_markets[:15]:  # Limit to 15 markets
            # Truncate question if too long for table
            question = m.question[:55] + "..." if len(m.question) > 55 else m.question
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
    ) -> SmartAnalysis | None:
        """Analyze news with full research context.

        This is the main Stage 2 entry point. It takes all pre-fetched context
        and produces a unified SmartAnalysis with all informed judgments.

        Follows PydanticAI best practices:
        - Context passed via typed AnalyzerDeps dataclass
        - Dynamic system prompt injects research context
        - Simple user prompt focuses on the task

        Args:
            message: Original news message
            extraction: Stage 1 extraction result
            web_results: Pre-fetched web search results
            markets_text: Pre-fetched Polymarket search results
            http_client: Optional HTTP client for additional searches (hybrid pattern)

        Returns:
            SmartAnalysis with tickers, sectors, impact, markets, and thesis
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

        # Create typed deps (PydanticAI pattern)
        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            web_results=web_results,
            markets_text=markets_text,
            http_client=http_client,
        )

        # Simple user prompt - context is injected via dynamic system prompt
        user_prompt = """Analyze this news. Determine:
1. Affected tickers and sectors (use research to validate)
2. Impact level and market direction (use research for context)
3. Primary investment thesis with confidence score
4. Ticker-level analysis with bull/bear thesis for each
5. Sector implications
6. Historical precedents from web research
7. Evaluate EACH prediction market from the table

**CRITICAL for market_evaluations**:
- You MUST return a MarketEvaluation for EVERY market in the Polymarket table
- Copy the market_id EXACTLY from the table (e.g., "0x123abc...")
- Set is_relevant=false for markets unrelated to this specific news
- Set verdict="skip" and recommended_side="skip" for irrelevant markets

Focus on DIRECT impacts. Be conservative with confidence scores."""

        try:
            result = await self.agent.run(user_prompt, deps=deps)
            output = result.output

            log.info(
                "Stage 2 smart analysis complete",
                tickers=output.tickers,
                sectors=output.sectors,
                impact=output.predicted_impact.value,
                direction=output.market_direction.value,
                thesis_confidence=f"{output.thesis_confidence:.0%}",
                markets_evaluated=len(output.market_evaluations),
                has_edge=output.has_tradable_edge,
            )

            return output

        except Exception as e:
            log.exception("Stage 2 analysis failed", error=str(e))
            # Return None to signal failure - callers must handle this
            return None


@lru_cache(maxsize=1)
def get_smart_analyzer() -> SmartAnalyzer:
    """Get the singleton smart analyzer instance."""
    return SmartAnalyzer()


async def analyze_with_context(
    message: UnifiedMessage,
    extraction: LightClassification,
    web_results: list[str],
    markets_text: str,
    polymarket_client: PolymarketClient | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> SmartAnalysis | None:
    """Convenience function for Stage 2 smart analysis.

    Args:
        message: The news message
        extraction: Stage 1 extraction result
        web_results: Pre-fetched web search results
        markets_text: Pre-fetched Polymarket search results
        polymarket_client: Optional Polymarket client
        http_client: Optional HTTP client for additional searches (hybrid pattern)

    Returns:
        SmartAnalysis with all informed judgments
    """
    analyzer = SmartAnalyzer(polymarket_client=polymarket_client)
    try:
        return await analyzer.analyze(
            message, extraction, web_results, markets_text, http_client=http_client
        )
    finally:
        await analyzer.close()
