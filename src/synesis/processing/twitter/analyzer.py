"""Twitter agent daily digest analyzer.

PydanticAI agent that consolidates tweets from multiple accounts and
produces a structured investment digest with themes, tickers, and
research-driven trading ideas backed by live market data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.ingestion.twitterapi import Tweet
from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker as _verify_ticker
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    search_market_impact,
)
from synesis.processing.twitter.accounts import get_profile
from synesis.processing.twitter.models import TwitterAgentAnalysis

if TYPE_CHECKING:
    from synesis.providers.base import TickerProvider
    from synesis.providers.yfinance.client import YFinanceClient

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a research analyst producing a daily investment digest from curated
Twitter/X accounts. Each account has a profile in the Account Directory below
showing their category, expertise, and known biases.

Your job is NOT to summarize tweets — it is to BUILD ON them. Use the tools
to verify claims, pull live market data, and construct actionable trade ideas.

## Ticker classification — this is critical

**Deep-research tickers (TOP 3 MAX):** Pick the 3 individual stocks where options
analysis adds the most alpha — catalyst with known timing, likely vol mispricing,
asymmetric payoff, or IV surface anomaly. These 3 get the full treatment:
- `get_quote()` + `get_options_snapshot()` (snapshot internally fetches 1mo history,
  realized vol, expirations, and chain — 4 yfinance calls per ticker)
- Specific trade ideas with strikes, structure, reasoning
- Selection criteria: options data should *change* the recommendation, not just confirm
  what you already know from equity analysis alone

**Other individual stocks** (beyond the top 3):
- `get_quote()` + `web_search()` only — no `get_options_snapshot()`
- Direction + reasoning, equity-style (similar to ETF treatment)
- If you have a strong options thesis but no budget, note it as a "watch for next session"

**Macro ETFs** (SPY, QQQ, TLT, GLD, USO, UUP, VIXY, EEM):
- Direction only (bullish/bearish/neutral) based on your thesis
- Do NOT call `get_quote()` or `get_options_snapshot()`
- Use `web_search()` to verify macro claims if needed
- Just state the view with reasoning from the tweets + web research

**Sector ETFs** (XLF, XLK, XLE, XLV, XLI, XLU, XLP, XLY, XLC, XLB, XLRE, SMH, IBB, KRE, XHB, etc.):
- Same as macro — direction only, no quote/history/options tool calls
- Use `web_search()` to verify sector-level claims if needed
- Just state bullish/bearish/neutral with reasoning

The goal: spend tool budget on the 3 names where options data will most change the trade,
not on broad ETFs or stocks where the thesis is already clear from equity analysis.

## Workflow per theme

1. **Identify themes** (3-7) from the tweets. Merge related topics.
2. **For each theme**, follow this research loop:
   a. **Verify tickers**: `verify_ticker()` for any individual stock ticker you're not 100% certain about — do this BEFORE fetching quotes/history/options so you don't waste calls on invalid symbols
   b. **Verify claims**: tweet says "DRAM prices falling" → `web_search("DRAM spot prices 2026")`
   c. **Price context** (individual stocks only): `get_quote()` — current price, MA levels, today's move
   d. **Expand thesis**: `web_search()` for deeper context on the investment thesis
   e. **Options & trade idea** (individual stocks only): see "Options strategy thinking" below
   f. **Macro/sector ETFs**: just assign direction based on your analysis, no tool calls needed
3. **Synthesize** a concise market overview (3-5 sentences).

## Tool budget
~12-18 total tool calls. Breakdown:
- 3 `get_options_snapshot()` calls (top 3 deep-research tickers) = ~12 yfinance calls
- ~3-5 `get_quote()` for remaining individual stocks
- ~4-6 `web_search()` for claim verification and macro context
- ~1-3 `verify_ticker()` as needed
Zero tool calls on ETFs (macro or sector).

## Search tips
- Include the current year/month in web search queries for time-sensitive data
  (e.g. "DRAM spot prices March 2026" not just "DRAM spot prices").
- The `recency` param filters by time window but does not add date context to the query itself.

## Options strategy thinking

You have a budget of 3 `get_options_snapshot()` calls. Use them on the tickers where
the options data will most change the trade recommendation — not just confirm it.

**Selecting the top 3 for deep research:**
Before calling any tools, rank individual stocks by how much the options snapshot will
add. Prioritize tickers where:
- There's a **dated catalyst** (earnings, FDA, macro event) → expiry selection matters
- **Vol mispricing is likely** (tweet mentions IV crush, unusual activity, "cheap puts")
- **Asymmetric payoff** (binary outcome, could double or zero) → options structure matters
- **High-priced stock** ($500+) → options for capital efficiency
Skip options research when the thesis is "slow grind up/down" with no catalyst — equity
is fine and the snapshot won't change your recommendation.

**For the 3 chosen tickers:**
- ALWAYS call `get_options_snapshot()` — the snapshot includes 30d realized vol alongside
  IV so you can assess richness/cheapness
- Suggest a SPECIFIC strategy with strikes and expiry reasoning:
  * Which strike? Pick based on delta/IV tradeoff (e.g. "30-delta calls for leverage"
    or "ATM puts because IV is only 25% vs 40% realized")
  * Which structure? Bull call spread if IV > realized (cap cost via selling),
    naked calls if IV < realized (cheap), put spreads for defined-risk bearish,
    straddles/strangles for vol plays
  * Why this over equity? Explain the edge (e.g. "IV at 30% is cheap vs 45% realized,
    calls give 5x leverage vs stock")

**For other individual stocks (beyond top 3):**
- Direction + reasoning only (like ETFs but with `get_quote()` data)
- If you have a strong options thesis, note it: "Options research warranted next session —
  IV likely cheap into earnings" so it's queued for future analysis

- When you do suggest options, be specific:
  * "Buy AAPL Jun $260 calls at $8.50, IV 28% < 35% 30d realized, 0.40 delta"
  * "Sell NVDA Mar $950/$900 put spread for $12 credit, IV 55% > 38% realized"
  * "Long straddle TSLA May $250, IV 42% vs 50% realized, cheap into earnings"
- When equity is better, say so: "Long AAPL at $262, IV 35% > 30% realized — premium not justified"

## Guidelines
- Source attribution: use usernames WITHOUT the @ prefix (e.g. "NickTimiraos" not "@NickTimiraos").
- Prioritize themes mentioned by MULTIPLE accounts (cross-confirmation).
- Distinguish between informed analysis and speculation.
- For trade ideas, note the time horizon and risk factors.
- Category assignment: macro (Fed, rates, CPI), sector (industry-wide),
  earnings (company results), geopolitical (tariffs, sanctions, wars),
  trade_idea (specific actionable setup), technical (chart patterns, flows).
- Theme conviction: high = multiple sources + verified + clear catalyst,
  medium = plausible + partially verified, low = single source or speculative.
- Ticker-level conviction: high = strong data support + clear setup,
  medium = reasonable thesis, low = speculative.
- Keep ticker symbols uppercase US-listed (e.g. AAPL not $AAPL).
- On slow days with few actionable tweets, return fewer themes (even 0) with a
  brief market overview. Don't force themes that aren't there.
- Fill in `price_context` for individual stock tickers where you called `get_quote()`.
  Leave it None for macro/sector ETFs.
- Fill in `trade_idea` and `time_horizon` for individual stocks when you have data.
  Be specific with strikes, expiries, and reasoning when suggesting options.
- Fill in `research_notes` on themes where web search yielded useful findings.

## Account bias & credibility weighting
- The Account Directory shows each poster's expertise and biases. Use it to:
  * Weight credibility: a Fed reporter on rate policy > general macro account.
  * Flag conflicts of interest: short-sellers posting bearish = they have a position.
    Accounts marked BIAS: always short (FuzzyPanda, MuddyWaters) should be flagged.
  * Cross-confirm: if macro AND technical accounts agree on direction, higher conviction.
  * Distinguish signal from noise: Elon Musk shitposting != market signal (unless
    about Tesla/SpaceX/DOGE). NickTimiraos on Fed = near-official signal.
  * Activist funds (Kerrisdale) talk their book — note they have a position.
  * Perma-bears (michaeljburry, zerohedge) — discount bearish takes unless backed by data.
"""


@dataclass
class TwitterAgentDeps:
    """Dependencies for the Twitter agent analyzer."""

    tweets: list[Tweet]
    yfinance: YFinanceClient | None = field(default=None, repr=False)
    ticker_provider: TickerProvider | None = field(default=None, repr=False)

    @property
    def accounts(self) -> list[str]:
        return sorted({t.username for t in self.tweets})


class TwitterAgentAnalyzer:
    """Daily Twitter agent digest analyzer."""

    def __init__(self) -> None:
        self._agent: Agent[TwitterAgentDeps, TwitterAgentAnalysis] | None = None

    @property
    def agent(self) -> Agent[TwitterAgentDeps, TwitterAgentAnalysis]:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def _create_agent(self) -> Agent[TwitterAgentDeps, TwitterAgentAnalysis]:
        model = create_model(tier="vsmart")

        agent: Agent[TwitterAgentDeps, TwitterAgentAnalysis] = Agent(
            model,
            deps_type=TwitterAgentDeps,
            output_type=PromptedOutput(TwitterAgentAnalysis),
            system_prompt=SYSTEM_PROMPT,
        )

        @agent.system_prompt
        def inject_tweets(ctx: RunContext[TwitterAgentDeps]) -> str:
            # Account directory — gives the LLM context on who each poster is
            directory_lines = ["# Account Directory\n"]
            for username in sorted(ctx.deps.accounts):
                profile = get_profile(username)
                if profile:
                    directory_lines.append(
                        f"- **@{username}** [{profile.category}]: {profile.description}"
                    )
                else:
                    directory_lines.append(f"- **@{username}**: (no profile)")

            # Tweets grouped by account
            by_account: dict[str, list[Tweet]] = {}
            for t in ctx.deps.tweets:
                by_account.setdefault(t.username, []).append(t)

            sections: list[str] = []
            for username, tweets in by_account.items():
                tweets.sort(key=lambda t: t.timestamp)
                lines = [f"## @{username}"]
                for t in tweets:
                    ts = t.timestamp.strftime("%Y-%m-%d %H:%M UTC")
                    lines.append(f"[{ts}] {t.text}")
                sections.append("\n".join(lines))

            today = datetime.now(UTC).strftime("%Y-%m-%d")

            return (
                f"# Current Date: {today}\n\n"
                + "\n".join(directory_lines)
                + "\n\n# Tweets (last 24 hours)\n\n"
                + "\n\n".join(sections)
            )

        @agent.tool
        async def web_search(
            ctx: RunContext[TwitterAgentDeps],
            query: str,
            recency: str = "day",
        ) -> str:
            """Search the web to verify claims or find supporting/contradicting evidence.

            Use this to fact-check earnings numbers, macro data, deal announcements,
            or any specific claim made in the tweets. Also use it to expand on
            investment theses with deeper context.

            Args:
                query: Specific search query (e.g. "AAOI Q4 2025 earnings revenue")
                recency: Time filter — "day", "week", "month", "year", or "all"
            """
            try:
                recency_map = {"all": "none"}
                mapped = recency_map.get(recency, recency)
                valid_recency: Recency = (
                    mapped if mapped in ("day", "week", "month", "year", "none") else "day"  # type: ignore[assignment]
                )
                results = await search_market_impact(query, count=5, recency=valid_recency)
                return format_search_results(results)
            except Exception as e:
                logger.warning("Web search failed in twitter analyzer", query=query, error=str(e))
                return "Search failed — try a different query"

        @agent.tool
        async def get_quote(
            ctx: RunContext[TwitterAgentDeps],
            ticker: str,
        ) -> str:
            """Get a live quote snapshot for a US ticker.

            Returns current price, daily change%, 50d/200d moving averages, and market cap.
            Use this to check "is ticker X actually up/down?" and provide price context.

            Args:
                ticker: US ticker symbol (e.g. "AAPL", "EWY", "SPY")
            """
            if not ctx.deps.yfinance:
                return "Quote data unavailable — yfinance client not configured."
            try:
                q = await ctx.deps.yfinance.get_quote(ticker.upper())
                parts = [f"{q.ticker}"]
                if q.name:
                    parts[0] += f" ({q.name})"
                if q.last is not None:
                    parts.append(f"Last: ${q.last:.2f}")
                if q.last is not None and q.prev_close:
                    chg = q.last - q.prev_close
                    chg_pct = (chg / q.prev_close) * 100
                    parts.append(f"Change: {chg:+.2f} ({chg_pct:+.1f}%)")
                if q.avg_50d is not None:
                    parts.append(f"50d MA: ${q.avg_50d:.2f}")
                if q.avg_200d is not None:
                    parts.append(f"200d MA: ${q.avg_200d:.2f}")
                if q.market_cap is not None:
                    if q.market_cap >= 1e12:
                        parts.append(f"Mkt Cap: ${q.market_cap / 1e12:.1f}T")
                    elif q.market_cap >= 1e9:
                        parts.append(f"Mkt Cap: ${q.market_cap / 1e9:.1f}B")
                    else:
                        parts.append(f"Mkt Cap: ${q.market_cap / 1e6:.0f}M")
                if q.volume is not None:
                    parts.append(f"Volume: {q.volume:,}")
                return " | ".join(parts)
            except Exception as e:
                logger.warning("get_quote failed", ticker=ticker, error=str(e))
                return f"Failed to fetch quote for {ticker}: {e}"

        @agent.tool
        async def get_options_snapshot(
            ctx: RunContext[TwitterAgentDeps],
            ticker: str,
        ) -> str:
            """Get options snapshot with IV, Greeks, and 30d realized vol for comparison.

            Use for individual stock names only (NOT macro/sector ETFs).
            Returns ATM calls/puts from the nearest monthly expiry (skips weeklies
            expiring within 7 days) plus 30d realized vol so you can assess IV
            richness/cheapness.

            Args:
                ticker: US stock ticker (e.g. "AAPL", "NVDA") — not ETFs
            """
            if not ctx.deps.yfinance:
                return "Options data unavailable — yfinance client not configured."
            try:
                snap = await ctx.deps.yfinance.get_options_snapshot(ticker.upper())

                if not snap.expiration:
                    spot_str = f"${snap.spot:.2f}" if snap.spot else "N/A"
                    return f"No options available for {ticker} (spot {spot_str})"

                spot = snap.spot or 0
                rv_str = (
                    f"{snap.realized_vol_30d:.0%}" if snap.realized_vol_30d is not None else "N/A"
                )

                header = (
                    f"{snap.ticker} options — exp {snap.expiration} "
                    f"(spot ${spot:.2f}, 30d realized vol: {rv_str}):"
                )
                lines = [header]

                for label, contracts in [("CALLS", snap.calls), ("PUTS", snap.puts)]:
                    if not contracts:
                        continue
                    sorted_c = sorted(contracts, key=lambda c: abs(c.strike - spot))[:3]
                    lines.append(f"  {label}:")
                    for c in sorted_c:
                        parts = [f"    ${c.strike:.0f}"]
                        if c.bid is not None and c.ask is not None:
                            parts.append(f"bid/ask: ${c.bid:.2f}/${c.ask:.2f}")
                        if c.implied_volatility is not None:
                            parts.append(f"IV: {c.implied_volatility:.0%}")
                        if c.greeks:
                            if c.greeks.delta is not None:
                                parts.append(f"Δ:{c.greeks.delta:.2f}")
                            if c.greeks.theta is not None:
                                parts.append(f"Θ:{c.greeks.theta:.2f}")
                        if c.open_interest is not None:
                            parts.append(f"OI:{c.open_interest:,}")
                        lines.append(" | ".join(parts))
                return "\n".join(lines)
            except Exception as e:
                logger.warning("get_options_snapshot failed", ticker=ticker, error=str(e))
                return f"Failed to fetch options for {ticker}: {e}"

        @agent.tool
        async def verify_ticker(
            ctx: RunContext[TwitterAgentDeps],
            ticker: str,
        ) -> str:
            """Verify if a US ticker symbol exists.

            Use this tool to validate US tickers BEFORE including them in your analysis.
            For non-US tickers, use web_search instead.

            Args:
                ticker: The US ticker symbol to verify (e.g. "AAPL", "GME", "TSLA")

            Returns:
                Verification result — either VERIFIED with company name, NOT FOUND, or error
            """
            return await _verify_ticker(ticker, ctx.deps.ticker_provider)

        return agent

    async def analyze_tweets(
        self,
        tweets: list[Tweet],
        yfinance: YFinanceClient | None = None,
        ticker_provider: TickerProvider | None = None,
    ) -> TwitterAgentAnalysis | None:
        """Run the digest analysis on a batch of tweets.

        Args:
            tweets: All tweets to analyze (already filtered to last 24hrs)
            yfinance: Optional YFinanceClient for live quote/history/options data
            ticker_provider: Optional provider for ticker verification

        Returns:
            TwitterAgentAnalysis or None on failure
        """
        if not tweets:
            return None

        deps = TwitterAgentDeps(
            tweets=tweets,
            yfinance=yfinance,
            ticker_provider=ticker_provider,
        )
        log = logger.bind(tweet_count=len(tweets), accounts=deps.accounts)
        log.info("Starting Twitter agent analysis")

        try:
            result = await self.agent.run(
                "Analyze these tweets and produce a daily investment digest. "
                "Use get_quote, web_search, and other tools to verify claims "
                "and build research-backed trade ideas.",
                deps=deps,
            )
            output = result.output
            output.raw_tweet_count = len(tweets)

            log.info(
                "Twitter agent analysis complete",
                themes=len(output.themes),
                tickers_mentioned=sum(len(th.tickers) for th in output.themes),
            )
            return output

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Twitter agent analysis failed", error=str(e))
            return None
