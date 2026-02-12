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
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker as _verify_ticker
from synesis.processing.news.models import (
    LightClassification,
    SmartAnalysis,
    UnifiedMessage,
)

if TYPE_CHECKING:
    from synesis.markets.polymarket import PolymarketClient, SimpleMarket
    from synesis.providers import FinnhubService

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
    finnhub: "FinnhubService | None" = None  # Optional for fundamental data tools


# =============================================================================
# System Prompt (Static Base)
# =============================================================================

# System prompt for Stage 2: Smart Analyzer
SMART_ANALYZER_SYSTEM_PROMPT = """You are an expert financial analyst with a disciplined institutional approach.

## Your Investment Philosophy
- Value early signal detection - identify trends and tickers before they become consensus
- Balance conviction with risk: high confidence for direct impacts, speculative for emerging trends
- Consider what is already priced in vs. new information (surprise drives alpha)
- Second-order effects can be valuable if the causal chain is clear

You have been given:
- Breaking news with entity extraction (Stage 1)
- Web research results (analyst estimates, historical data)
- Polymarket prediction markets (pre-searched)

Your job is to make ALL informed judgments about this news.

## Before Making Decisions (REQUIRED Research Process)

**CRITICAL: You MUST research historical precedent before making predictions.**

### Step 1: Research Historical Context (MANDATORY for high/medium impact)

Use the `web_search` tool to find SIMILAR past events that match the current context:

1. **For MACRO events** (Fed, CPI, NFP, geopolitical):
   ```
   web_search("Fed rate cut {magnitude} market reaction {year}")
   web_search("{event_type} similar events SPY reaction")
   web_search("{event_type} historical pattern")
   ```

2. **For EARNINGS events**:
   ```
   web_search("{ticker} earnings beat miss historical reaction")
   web_search("{ticker} earnings surprise stock movement")
   ```

3. **For CORPORATE events** (M&A, guidance, regulatory):
   ```
   web_search("{company} {event} similar historical")
   web_search("{event_type} market impact precedent")
   ```

**IMPORTANT**: 
- Search for SPECIFIC patterns that match the current context (magnitude, surprise level, sector)
- If no similar events found, state "No relevant historical precedent" - do NOT force irrelevant data
- Better to have NO historical context than misleading context

### Step 2: Analyze Chain of Thought

After gathering historical context, reason through:
1. What is the PRIMARY causal relationship? (company X → event Y → impact Z)
2. What happened in SIMILAR historical events? (only if relevant matches found)
3. What is the expected magnitude? (% move, based on precedent if available)
4. What is the time horizon? (immediate, days, weeks)
5. What is already priced in? (expected vs. surprise)
6. What could go wrong? (key risks to thesis)

### Step 3: Ground Predictions in Data

- Cite SPECIFIC historical events with dates (only if they match current context)
- Quantify expected moves based on precedent
- If no relevant historical data found, make predictions based on first principles
- DO NOT make up historical data
- DO NOT cite irrelevant historical events just to fill space

## Your Tasks

### 1. Identify Affected Securities (STRICT RELEVANCE)

**CRITICAL: Only include tickers with DIRECT, MATERIAL impact.**

**Ticker Verification Workflow:**
1. Extract potential ticker from the news text
2. If likely a US ticker, call `verify_ticker_finnhub(ticker)` to confirm it exists
3. If VERIFIED: include in analysis with the company name returned
4. If NOT FOUND or error: use `web_search("{ticker} stock ticker price")` to verify
5. If still unclear: exclude the ticker or note uncertainty

For each potential ticker, apply this relevance test:

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

**Required Output for Each Ticker**:
- `ticker`: Stock symbol (e.g., AAPL)
- `company_name`: Full company name
- `relevance_score`: 0.0-1.0 (only include if >= 0.6)
  - 0.9-1.0: Primary subject of the news
  - 0.7-0.89: Directly named or materially affected
  - 0.6-0.69: Clear secondary impact
  - <0.6: DO NOT include
- `relevance_reason`: ONE sentence explaining the direct causal link

**Examples**:
- News: "Apple announces 20% revenue miss in China"
  - ✅ AAPL (0.95): Primary subject, direct earnings impact
  - ❌ MSFT: Competitor, not directly affected
  - ❌ TSM: Supplier, but no specific impact mentioned

- News: "Fed cuts rates by 50bps"
  - ✅ Sector-level play on Financials, Real Estate
  - ❌ Individual banks unless specifically named with impact

**When to use web_search:**
- SEARCH if a company is mentioned but you're unsure of direct impact
- SKIP search for obvious primary subjects or well-known sector plays
- Use for non-US tickers after verify_ticker_finnhub returns NOT FOUND
- Limit to 1-2 searches max to maintain analysis speed

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
- **thesis_confidence**: 0.0 to 1.0 (use calibration table below)

## Confidence Calibration

| Score | Criteria | Include? |
|-------|----------|----------|
| 0.9-1.0 | Unambiguous direct impact, clear causal link, historical precedent | Yes - high conviction |
| 0.7-0.89 | Strong relationship, some uncertainty in magnitude or timing | Yes - solid play |
| 0.5-0.69 | Plausible connection, emerging trend, multiple interpretations | Yes - speculative |
| 0.3-0.49 | Weak but interesting signal, early trend detection opportunity | Maybe - flag as speculative |
| <0.3 | Very tenuous connection, no clear causal path | No - too noisy |

Note: Lower confidence plays (0.3-0.69) can still be valuable for early trend detection. Flag them appropriately but don't exclude them entirely.

### 4. Ticker-Level Analysis
For each directly affected ticker:
- Bull thesis: Why positive
- Bear thesis: Why negative
- Net direction: bullish | bearish | neutral
- Conviction: 0.0 to 1.0
- Time horizon: intraday | days | weeks | months

### 5. Sector & Subsector Mapping

**Use GICS sectors as top-level classification:**
Energy | Materials | Industrials | Utilities | Healthcare |
Financials | Consumer Discretionary | Consumer Staples |
Information Technology | Communication Services | Real Estate

**For each affected sector provide:**
- `sector`: GICS sector name (e.g., "Information Technology")
- `subsectors`: 1-3 specific subsectors for granularity
- `direction`: bullish | bearish | neutral
- `reasoning`: One sentence explaining the sector impact

**Subsector Examples by Sector:**
- Information Technology: AI chips, cloud computing, cybersecurity, enterprise software, consumer electronics
- Financials: regional banks, investment banks, insurance, fintech, payment processors, asset managers
- Healthcare: biotech, pharma, medical devices, healthcare services, managed care
- Energy: oil & gas exploration, refiners, renewables, utilities
- Consumer Discretionary: luxury goods, auto manufacturers, e-commerce, restaurants, travel
- Communication Services: social media, streaming, telecom, advertising, gaming
- Industrials: defense, aerospace, construction, logistics, machinery
- Materials: mining, chemicals, steel, packaging
- Real Estate: REITs, commercial, residential, data centers
- Consumer Staples: food & beverage, household products, tobacco, retail grocery
- Utilities: electric, gas, water, renewable energy generators

### 6. Historical Context & Market Reaction Patterns

**Use the web_search tool to find RELEVANT historical precedent. If nothing matches, skip this section.**

Provide structured historical analysis ONLY if you found relevant similar events:

**a) Precedent Events** (only include if they match current context)
- Cite events with SIMILAR characteristics (magnitude, surprise level, sector impact)
- Example: "July 2023: Fed cut 25bps (similar magnitude); SPY +1.2% over 3 days"
- Do NOT cite events that are tangentially related but fundamentally different

**b) Quantified Market Reactions** (extract from relevant precedents)
- Immediate reaction (first 15-60 min): cite actual % moves
- Short-term (1-5 days): cite actual sector performance
- Extended (1-4 weeks): cite sustained moves

**c) Typical Reaction Pattern** (based on relevant precedents)
- Initial spike/drop magnitude
- Reversal probability
- Sector rotation patterns

**d) Key Differences from Precedents**
- What's different about current context?
- Higher/lower impact expected and why?

**IMPORTANT**:
- If no relevant historical precedent found, state: "No relevant historical precedent found - analysis based on first principles"
- DO NOT fabricate or force irrelevant historical data
- Better to have NO historical section than misleading context

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

## Output Constraints (IMPORTANT)

- primary_thesis: ONE sentence, max 150 characters
- relevance_reason: ONE sentence, max 100 characters
- bull_thesis/bear_thesis: max 100 characters each
- reasoning (for markets): max 120 characters

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

        # Tool: Verify ticker via Finnhub
        @agent.tool
        async def verify_ticker_finnhub(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Verify if a US ticker symbol exists using Finnhub.

            Use this tool to validate US tickers BEFORE including them in your analysis.
            For non-US tickers, use web_search instead.

            Args:
                ticker: The US ticker symbol to verify (e.g., "AAPL", "GME", "TSLA")

            Returns:
                Verification result - either VERIFIED with company name, NOT FOUND, or error
            """
            return await _verify_ticker(ticker, ctx.deps.finnhub)

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
                valid_recency = recency if recency in ("day", "week", "month", "year") else "week"
                results = await search_market_impact(query, count=5, recency=valid_recency)
                return format_search_results(results)
            except Exception as e:
                logger.warning("Web search failed", query=query, error=str(e))
                return f"Search failed: {e}"

        # Tool: Get stock fundamentals
        @agent.tool
        async def get_stock_fundamentals(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Get key financial metrics for a stock (P/E ratio, market cap, 52-week range).

            Use this to assess if a stock is overvalued/undervalued relative to the news impact.

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

            Returns:
                Formatted string with key financial metrics
            """
            if ctx.deps.finnhub is None:
                return "Finnhub not available - fundamental data tools require FINNHUB_API_KEY."

            data = await ctx.deps.finnhub.get_basic_financials(ticker)
            if not data:
                return f"No financial data available for {ticker}"

            # Format key metrics for LLM consumption
            lines = [f"**{ticker} Fundamentals:**"]

            pe = data.get("peRatio")
            if pe:
                lines.append(f"- P/E Ratio: {pe:.1f}")

            mktcap = data.get("marketCap")
            if mktcap:
                # Convert from millions to billions
                lines.append(f"- Market Cap: ${mktcap / 1000:.1f}B")

            high52 = data.get("52WeekHigh")
            low52 = data.get("52WeekLow")
            if high52 and low52:
                lines.append(f"- 52-Week Range: ${low52:.2f} - ${high52:.2f}")

            beta = data.get("beta")
            if beta:
                lines.append(f"- Beta: {beta:.2f}")

            rev_growth = data.get("revenueGrowth")
            if rev_growth:
                lines.append(f"- Revenue Growth (TTM): {rev_growth:.1f}%")

            roe = data.get("roeTTM")
            if roe:
                lines.append(f"- ROE (TTM): {roe:.1f}%")

            return "\n".join(lines)

        # Tool: Get insider activity
        @agent.tool
        async def get_insider_activity(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Get recent insider transactions and sentiment for a stock.

            Use this to see if insiders are buying or selling ahead of news.
            Insider buying is typically a bullish signal; selling can be bearish
            (but insiders often sell for non-informational reasons).

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

            Returns:
                Summary of recent insider activity and MSPR sentiment score
            """
            if ctx.deps.finnhub is None:
                return "Finnhub not available - fundamental data tools require FINNHUB_API_KEY."

            # Fetch both transactions and sentiment in parallel
            txns = await ctx.deps.finnhub.get_insider_transactions(ticker, limit=5)
            sentiment = await ctx.deps.finnhub.get_insider_sentiment(ticker)

            lines = [f"**{ticker} Insider Activity:**"]

            # Add sentiment score if available
            if sentiment and sentiment.get("mspr") is not None:
                mspr = sentiment["mspr"]
                change = sentiment.get("change", 0)
                if mspr > 0:
                    signal = "bullish (net buying)"
                elif mspr < 0:
                    signal = "bearish (net selling)"
                else:
                    signal = "neutral"
                lines.append(f"- MSPR Score: {mspr:.2f} ({signal})")
                lines.append(f"- Net Share Change: {change:+,.0f}")

            # Add recent transactions
            if txns:
                lines.append("\n**Recent Transactions:**")
                for txn in txns[:5]:
                    name = txn.get("name", "Unknown")
                    code = txn.get("transactionCode", "?")
                    shares = txn.get("shares", 0)
                    date = txn.get("filingDate", "")
                    price = txn.get("transactionPrice")

                    action = "bought" if code == "P" else "sold" if code == "S" else code
                    price_str = f" @ ${price:.2f}" if price else ""
                    lines.append(f"- {name}: {action} {shares:,} shares{price_str} ({date})")
            else:
                lines.append("- No recent insider transactions found")

            return "\n".join(lines)

        # Tool: Get earnings info
        @agent.tool
        async def get_earnings_info(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Get earnings calendar and recent EPS surprises.

            Use this when news might relate to earnings expectations.

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

            Returns:
                Upcoming earnings date and recent EPS surprise history
            """
            if ctx.deps.finnhub is None:
                return "Finnhub not available - fundamental data tools require FINNHUB_API_KEY."

            calendar = await ctx.deps.finnhub.get_earnings_calendar(ticker)
            surprises = await ctx.deps.finnhub.get_eps_surprises(ticker, limit=4)

            lines = [f"**{ticker} Earnings Info:**"]

            # Upcoming earnings
            if calendar and calendar.get("date"):
                date = calendar["date"]
                hour = calendar.get("hour", "")
                hour_str = (
                    " (before market)"
                    if hour == "bmo"
                    else " (after market)"
                    if hour == "amc"
                    else ""
                )
                estimate = calendar.get("epsEstimate")
                estimate_str = f", EPS estimate: ${estimate:.2f}" if estimate else ""
                lines.append(f"- Next Earnings: {date}{hour_str}{estimate_str}")
            else:
                lines.append("- Next Earnings: Not scheduled")

            # EPS surprise history
            if surprises:
                lines.append("\n**Recent EPS Surprises:**")
                for s in surprises:
                    period = s.get("period", "")
                    actual = s.get("actual")
                    estimate = s.get("estimate")
                    surprise_pct = s.get("surprisePercent")

                    if actual is not None and estimate is not None:
                        beat = (
                            "beat"
                            if actual > estimate
                            else "missed"
                            if actual < estimate
                            else "met"
                        )
                        pct_str = f" ({surprise_pct:+.1f}%)" if surprise_pct else ""
                        lines.append(
                            f"- {period}: ${actual:.2f} vs ${estimate:.2f} est ({beat}{pct_str})"
                        )
            else:
                lines.append("- No EPS history available")

            return "\n".join(lines)

        # Tool: Get SEC filings
        @agent.tool
        async def get_sec_filings(
            ctx: RunContext[AnalyzerDeps],
            ticker: str,
        ) -> str:
            """Get recent SEC filings (10-K, 8-K, etc).

            Use this when news references regulatory filings or disclosures.

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")

            Returns:
                List of recent SEC filings with dates and form types
            """
            if ctx.deps.finnhub is None:
                return "Finnhub not available - fundamental data tools require FINNHUB_API_KEY."

            filings = await ctx.deps.finnhub.get_sec_filings(ticker, limit=5)

            if not filings:
                return f"No recent SEC filings found for {ticker}"

            lines = [f"**{ticker} Recent SEC Filings:**"]
            for filing in filings:
                form = filing.get("form", "Unknown")
                date = filing.get("filedDate", "")
                lines.append(f"- {form} filed {date}")

            return "\n".join(lines)

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
        finnhub: "FinnhubService | None" = None,
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
            finnhub: Optional FinnhubService for fundamental data tools

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
            finnhub_available=finnhub is not None,
        )

        # Create typed deps (PydanticAI pattern)
        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            web_results=web_results,
            markets_text=markets_text,
            http_client=http_client,
            finnhub=finnhub,
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
    polymarket_client: "PolymarketClient | None" = None,
    http_client: httpx.AsyncClient | None = None,
    finnhub: "FinnhubService | None" = None,
) -> SmartAnalysis | None:
    """Convenience function for Stage 2 smart analysis.

    Args:
        message: The news message
        extraction: Stage 1 extraction result
        web_results: Pre-fetched web search results
        markets_text: Pre-fetched Polymarket search results
        polymarket_client: Optional Polymarket client
        http_client: Optional HTTP client for additional searches (hybrid pattern)
        finnhub: Optional FinnhubService for fundamental data tools

    Returns:
        SmartAnalysis with all informed judgments
    """
    analyzer = SmartAnalyzer(polymarket_client=polymarket_client)
    try:
        return await analyzer.analyze(
            message,
            extraction,
            web_results,
            markets_text,
            http_client=http_client,
            finnhub=finnhub,
        )
    finally:
        await analyzer.close()
