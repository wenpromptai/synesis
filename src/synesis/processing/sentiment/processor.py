"""Flow 2: Reddit Sentiment Intelligence Processor.

This module implements the two-gate sentiment analysis pipeline:

Gate 1 (Lexicon): Fast, free lexicon-based sentiment scoring using
the existing SentimentAnalyzer. Extracts tickers and sentiment scores
but may produce false positives.

Gate 2 (LLM): Smart model refinement that validates tickers, removes
false positives, assesses post quality, and generates a narrative summary.

Architecture follows PydanticAI best practices:
- Typed deps via `deps_type=SentimentRefinementDeps` dataclass
- Dynamic system prompt injection via `@agent.system_prompt` decorator
- Pre-fetched context passed via deps, not user prompt
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.ingestion.reddit import RedditPost
from synesis.processing.sentiment.analyzer import SentimentAnalyzer
from synesis.processing.sentiment.models import (
    SentimentResult,
    SentimentSignal,
    SentimentRefinement,
    SentimentRefinementDeps,
    TickerSentimentSummary,
)
from synesis.processing.common.llm import create_model
from synesis.processing.common.watchlist import WatchlistManager

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.config import Settings
    from synesis.providers.base import TickerProvider
    from synesis.storage.database import Database

logger = get_logger(__name__)


# =============================================================================
# Gate 2 System Prompt
# =============================================================================

SENTIMENT_REFINER_SYSTEM_PROMPT = """You are an expert retail sentiment analyst specializing in Reddit finance communities (r/wallstreetbets, r/stocks, r/options).

You have been given:
- Reddit posts from finance subreddits
- Lexicon-based sentiment analysis results (Gate 1)
- Raw ticker mentions with sentiment scores
- Pre-verified ticker information (tickers matched via FactSet with company names)

## 1. Context-Check Each Pre-Verified Ticker

Each ticker below was matched by FactSet to a company name — but the match may be WRONG.
Read the post content to determine what the author actually means by that symbol.

For each pre-verified ticker, output a ValidatedTicker with:
- **is_valid_ticker = True** if the symbol has ANY valid financial meaning in context:
  - If the FactSet company name matches what the author means → keep company_name as-is
  - If the FactSet match is WRONG but the symbol refers to a real financial instrument → set is_valid_ticker=True and **correct company_name** to what the author actually means:
    - SPX + "Spirax Group" → correct company_name to "S&P 500 Index"
    - VIX + "VIX Securities" → correct company_name to "CBOE Volatility Index"
    - BTC + some obscure equity → correct company_name to "Bitcoin"
    - ETH + random match → correct company_name to "Ethereum"
    - DXY + wrong match → correct company_name to "US Dollar Index"
  - The key question is: does this symbol carry financial meaning in the post?
- **is_valid_ticker = False** + rejection_reason ONLY if:
  - The symbol has ZERO financial meaning in the post (e.g., used as a random word or acronym unrelated to markets)

Confidence guidelines:
- 0.9+ : Post explicitly names the company/instrument or discusses specific news about it
- 0.7-0.9 : Context strongly implies the financial instrument (earnings, price targets, DD)
- 0.5-0.7 : Ambiguous but plausible financial reference
- Below 0.5 : Reject (is_valid_ticker = False)

## 1b. Validate Unverified Tickers (if any)

If there are UNVERIFIED tickers (ticker provider was unavailable), validate them using your own financial knowledge:
1. If you recognize the ticker as a well-known instrument (e.g., NVDA → NVIDIA, AAPL → Apple) → output ValidatedTicker with is_valid_ticker=True and correct company_name
2. If the ticker is ambiguous or you're unsure → use the `web_search` tool to check (search "{TICKER} stock ticker company")
3. If neither your knowledge nor web_search confirms it → output with is_valid_ticker=False

Do NOT web_search every unverified ticker — only search when genuinely unsure.

## 2. Assess Post Quality

**HIGH**: DD with research, earnings analysis, technical analysis with levels, news with implications
**MEDIUM**: Position posts with reasoning, sentiment-revealing questions, news with commentary
**LOW**: Memes without substance, one-word reactions, off-topic
**SPAM**: Promotional, crypto shilling, bot-like content

## 3. Generate Narrative Summary

Create a 2-3 sentence narrative:
- Sentence 1: Overall market mood and dominant theme
- Sentence 2: Top 2-3 tickers with strongest sentiment and why
- Sentence 3: Key catalysts or upcoming events driving sentiment

## 4. Identify Themes

Extract key market themes (e.g., "earnings season", "rate cut expectations", "AI momentum").

## 5. Identify Extreme Sentiment

Flag tickers with >85% one-directional sentiment as potential contrarian signals.

## Sentiment Aggregation

1. Weight by quality: HIGH=3x, MEDIUM=1x, LOW/SPAM=ignore
2. Calculate bullish vs bearish ratio per ticker
3. Flag "crowded" positions (>80% one direction)
4. Distinguish event-driven sentiment (earnings, FDA) vs. general mood

## Guidelines
- Context is key: the FactSet company name is provided but may not match the author's intent — CORRECT the company_name rather than rejecting the ticker
- WSB culture: "regarded" = regarded, "apes" = retail traders, "tendies" = gains
- Confidence scores should be conservative (0.7+ for high confidence)
- ONLY output ValidatedTickers for tickers in the Pre-Verified Tickers table or the UNVERIFIED Tickers table. Do NOT add tickers you find in the post content that aren't in either table."""


# =============================================================================
# Flow 2 Processor
# =============================================================================


class SentimentProcessor:
    """Sentiment: Reddit sentiment → Watchlist → 6-hour signals.

    This processor implements the two-gate sentiment analysis pipeline:
    1. Gate 1: Fast lexicon analysis for initial sentiment and ticker extraction
    2. Gate 2: LLM refinement for ticker validation and narrative generation

    Usage:
        processor = SentimentProcessor(settings, redis, db=database)
        refinement = await processor.process_posts(posts)
        signal = await processor.generate_signal()
    """

    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        db: Database | None = None,
        ticker_provider: "TickerProvider | None" = None,
        watchlist: "WatchlistManager | None" = None,
    ) -> None:
        """Initialize Flow 2 processor.

        Args:
            settings: Application settings
            redis: Redis client for watchlist storage
            db: Optional PostgreSQL database for persistence
            ticker_provider: Optional TickerProvider for ticker verification
            watchlist: Optional shared WatchlistManager (created in __main__.py)
        """
        self.settings = settings
        self.redis = redis
        self.db = db
        self._ticker_provider = ticker_provider

        # Use provided watchlist or create new one (for standalone use)
        if watchlist is not None:
            self.watchlist = watchlist
        else:
            self.watchlist = WatchlistManager(redis, db=db)

        self.lexicon_analyzer = SentimentAnalyzer()
        self._refiner_agent: Agent[SentimentRefinementDeps, SentimentRefinement] | None = None

        # In-memory buffer for posts between signal generations
        self._post_buffer: list[tuple[RedditPost, SentimentResult]] = []
        self._buffer_start: datetime | None = None

        # Store raw ticker scores for accurate ratio calculation in generate_signal()
        self._raw_tickers: dict[str, list[float]] = {}

    @property
    def refiner_agent(self) -> Agent[SentimentRefinementDeps, SentimentRefinement]:
        """Get or create the Gate 2 refiner agent."""
        if self._refiner_agent is None:
            self._refiner_agent = self._create_refiner_agent()
        return self._refiner_agent

    def _create_refiner_agent(
        self,
    ) -> Agent[SentimentRefinementDeps, SentimentRefinement]:
        """Create the PydanticAI agent for Gate 2 refinement.

        Uses smart model for complex reasoning about ticker validation
        and narrative generation.
        """
        # Use smart model for refinement
        model = create_model(smart=True)

        agent: Agent[SentimentRefinementDeps, SentimentRefinement] = Agent(
            model,
            deps_type=SentimentRefinementDeps,
            output_type=PromptedOutput(SentimentRefinement),
            system_prompt=SENTIMENT_REFINER_SYSTEM_PROMPT,
        )

        # Dynamic system prompt: inject Gate 1 results
        @agent.system_prompt
        def inject_gate1_context(ctx: RunContext[SentimentRefinementDeps]) -> str:
            """Inject Gate 1 lexicon results into system prompt."""
            deps = ctx.deps

            # Format posts with lexicon results
            posts_section = "## Reddit Posts with Lexicon Analysis\n\n"
            for i, (post, result) in enumerate(deps.lexicon_results[:30], 1):
                sentiment_label = (
                    "bullish"
                    if result.is_bullish
                    else ("bearish" if result.is_bearish else "neutral")
                )
                tickers_str = ", ".join(result.tickers_mentioned) or "none"
                posts_section += f"""### Post {i} (r/{post.subreddit})
**Title**: {post.title}
**Content**: {post.content[:500]}{"..." if len(post.content) > 500 else ""}
**Lexicon Sentiment**: {result.compound:.2f} ({sentiment_label})
**Tickers Extracted**: {tickers_str}
**Post ID**: {post.post_id}

"""

            # Format raw ticker aggregates
            tickers_section = "## Aggregated Ticker Mentions (from Lexicon)\n\n"
            tickers_section += "| Ticker | Mentions | Avg Sentiment | Scores |\n"
            tickers_section += "|--------|----------|---------------|--------|\n"
            for ticker, scores in sorted(
                deps.raw_tickers.items(), key=lambda x: len(x[1]), reverse=True
            )[:50]:
                avg = sum(scores) / len(scores) if scores else 0
                scores_preview = ", ".join(f"{s:.2f}" for s in scores[:5])
                if len(scores) > 5:
                    scores_preview += f" (+{len(scores) - 5} more)"
                tickers_section += f"| {ticker} | {len(scores)} | {avg:.2f} | {scores_preview} |\n"

            # Format pre-verified tickers (single table — all validated via FactSet)
            verified_section = "## Pre-Verified Tickers\n\n"
            if deps.pre_verified:
                verified_section += "All tickers below were matched by FactSet. Context-check each one against the post content — the company name may be wrong.\n\n"
                verified_section += "| Ticker | Company Name |\n"
                verified_section += "|--------|--------------|\n"
                for ticker in sorted(deps.pre_verified.keys()):
                    verified_section += f"| {ticker} | {deps.pre_verified[ticker]} |\n"
            else:
                verified_section += "_No verified tickers this batch._\n"

            # Format NOT FOUND tickers (auto-rejected — excludes unverified)
            not_found = [
                t
                for t in deps.raw_tickers
                if t not in deps.pre_verified and t not in deps.unverified_tickers
            ]
            not_found_section = ""
            if not_found:
                not_found_section = "## NOT FOUND Tickers (Auto-Rejected)\n\n"
                not_found_section += "| Ticker |\n"
                not_found_section += "|--------|\n"
                for ticker in sorted(not_found):
                    not_found_section += f"| {ticker} |\n"

            # Format UNVERIFIED tickers (provider down — LLM validates from knowledge)
            unverified_section = ""
            if deps.unverified_tickers:
                unverified_section = "## UNVERIFIED Tickers (Ticker Provider Unavailable)\n\n"
                unverified_section += (
                    "These tickers could NOT be checked (ticker provider down/unavailable). "
                    "Validate using your own knowledge — only use `web_search` if genuinely unsure.\n\n"
                )
                unverified_section += "| Ticker |\n|--------|\n"
                for ticker in sorted(deps.unverified_tickers):
                    unverified_section += f"| {ticker} |\n"

            subreddits_str = ", ".join(deps.subreddits) if deps.subreddits else "various"

            return f"""
## Analysis Context
Total Posts: {len(deps.posts)}
Subreddits: {subreddits_str}
Unique Tickers (raw): {len(deps.raw_tickers)}
Pre-verified (valid): {len(deps.pre_verified)}
Not found (rejected): {len(not_found)}
Unverified (provider down): {len(deps.unverified_tickers)}

{verified_section}

{not_found_section}

{unverified_section}

{posts_section}

{tickers_section}"""

        # Tool: Web search for ticker verification fallback
        @agent.tool
        async def web_search(
            ctx: RunContext[SentimentRefinementDeps],
            query: str,
        ) -> str:
            """Search the web for additional research, context, or ticker verification.

            Use to:
            - Verify unverified tickers when the ticker provider is unavailable
              (e.g., "NVDA stock ticker company" to confirm it's NVIDIA)
            - Fetch background info, recent news, or historical context

            Args:
                query: Search query (e.g., "NVDA earnings Q4 2026 results")

            Returns:
                Formatted search results
            """
            try:
                from synesis.processing.common.web_search import (
                    SearchProvidersExhaustedError,
                    format_search_results,
                    search_market_impact,
                )

                results = await search_market_impact(query, count=3, recency="week")

                if not results:
                    return f"No web results found for '{query}'."

                return format_search_results(results)

            except SearchProvidersExhaustedError:
                return "Web search unavailable: all providers failed or not configured."
            except Exception as e:
                logger.warning("Web search failed", query=query, error=str(e))
                return f"Web search failed: {e}"

        return agent

    async def process_post(self, post: RedditPost) -> SentimentResult:
        """Process a single Reddit post through Gate 1.

        This is called for each incoming post. The post and result
        are buffered for batch Gate 2 processing.

        Args:
            post: Reddit post to process

        Returns:
            Gate 1 lexicon sentiment result
        """
        # Gate 1: Lexicon analysis
        result = await self.lexicon_analyzer.analyze(post.full_text)

        # Buffer for batch processing
        if self._buffer_start is None:
            self._buffer_start = datetime.now(UTC)
        self._post_buffer.append((post, result))

        logger.debug(
            "Flow 2 Gate 1 complete",
            post_id=post.post_id,
            subreddit=post.subreddit,
            sentiment=result.compound,
            tickers=result.tickers_mentioned,
        )

        return result

    async def process_posts(self, posts: list[RedditPost]) -> SentimentRefinement:
        """Process multiple posts through Gate 1 + Gate 2 pipeline.

        Args:
            posts: List of Reddit posts to process

        Returns:
            Gate 2 refined sentiment analysis
        """
        if not posts:
            logger.warning("No posts to process")
            return SentimentRefinement()

        log = logger.bind(post_count=len(posts))
        log.info("Flow 2 processing started")

        # Gate 1: Lexicon analysis (fast, free)
        lexicon_results: list[tuple[RedditPost, SentimentResult]] = []
        raw_tickers: dict[str, list[float]] = {}

        for post in posts:
            result = await self.lexicon_analyzer.analyze(post.full_text)
            lexicon_results.append((post, result))

            # Aggregate tickers
            for ticker in result.tickers_mentioned:
                if ticker not in raw_tickers:
                    raw_tickers[ticker] = []
                raw_tickers[ticker].append(result.compound)

        # Get unique subreddits
        subreddits = list({post.subreddit for post in posts})

        log.info(
            "Gate 1 complete",
            unique_tickers=len(raw_tickers),
            subreddits=subreddits,
        )

        # Store raw_tickers for ratio calculation in generate_signal()
        self._raw_tickers = raw_tickers

        # Pre-verify tickers in batch (before Gate 2, eliminates tool calls)
        # Three-way split: pre_verified / not_found / unverified
        pre_verified: dict[str, str] = {}
        unverified_tickers: list[str] = []

        if self._ticker_provider and raw_tickers:
            for ticker in raw_tickers:
                try:
                    is_valid, _region, company_name = await self._ticker_provider.verify_ticker(
                        ticker
                    )
                    if is_valid and company_name:
                        pre_verified[ticker] = company_name
                except Exception as e:
                    logger.warning("Pre-verify failed", ticker=ticker, error=str(e))
                    unverified_tickers.append(ticker)
        elif raw_tickers:
            # No provider available — all tickers are unverified
            unverified_tickers = list(raw_tickers.keys())
            log.warning(
                "No ticker provider available, all tickers unverified",
                count=len(unverified_tickers),
            )

        # NOT FOUND = provider explicitly rejected (not in pre_verified AND not unverified)
        not_found_tickers = [
            t for t in raw_tickers if t not in pre_verified and t not in unverified_tickers
        ]

        log.info(
            "Pre-verification complete",
            verified=len(pre_verified),
            not_found=len(not_found_tickers),
            unverified=len(unverified_tickers),
        )

        # Gate 2: LLM context-checks pre-verified tickers + narrative/quality/themes
        deps = SentimentRefinementDeps(
            posts=posts,
            lexicon_results=lexicon_results,
            raw_tickers=raw_tickers,
            subreddits=subreddits,
            pre_verified=pre_verified,
            unverified_tickers=unverified_tickers,
        )

        user_prompt = """Analyze the Reddit posts and lexicon results above.

For each pre-verified ticker, output a ValidatedTicker:
- Context-check the FactSet company name against post content
- If the FactSet match is correct, keep company_name as-is with is_valid_ticker=True
- If the FactSet match is WRONG but the symbol has real financial meaning, set is_valid_ticker=True and CORRECT company_name to what the author actually means
- Only set is_valid_ticker=False if the symbol has zero financial meaning in context

For each UNVERIFIED ticker (if any), validate using your knowledge (only `web_search` if genuinely unsure), then output a ValidatedTicker.

Also:
1. **Assess post quality** (high/medium/low/spam)
2. **Generate narrative summary** (2-3 sentences covering validated tickers)
3. **Identify key themes** (market themes, catalysts, sector rotations)
4. **Flag extreme sentiment** (>85% one direction for any ticker)"""

        try:
            agent_result = await self.refiner_agent.run(user_prompt, deps=deps)
            refinement = agent_result.output

            # Append NOT FOUND tickers to LLM's rejected list
            refinement.rejected_tickers.extend(not_found_tickers)

            # Update watchlist with validated tickers
            for validated_ticker in refinement.validated_tickers:
                if validated_ticker.is_valid_ticker and validated_ticker.confidence >= 0.6:
                    await self.watchlist.add_ticker(
                        validated_ticker.ticker,
                        source="reddit",
                        subreddit=subreddits[0] if subreddits else None,
                        company_name=validated_ticker.company_name or None,
                    )

            log.info(
                "Gate 2 complete",
                validated_tickers=len(
                    [t for t in refinement.validated_tickers if t.is_valid_ticker]
                ),
                rejected_tickers=len(refinement.rejected_tickers),
                overall_sentiment=refinement.overall_sentiment,
                narrative_length=len(refinement.narrative_summary),
            )

            return refinement

        except Exception as e:
            log.exception("Gate 2 refinement failed", error=str(e))
            # Return empty validated + all raw tickers as rejected
            return SentimentRefinement(
                validated_tickers=[],
                rejected_tickers=list(raw_tickers.keys()),
                narrative_summary=f"Gate 2 analysis failed: {e}",
            )

    async def generate_signal(self) -> SentimentSignal:
        """Generate 6-hour signal with watchlist sentiments.

        This should be called on a 6-hour schedule. It processes
        buffered posts and generates the signal output.

        Returns:
            SentimentSignal with aggregated sentiment data
        """
        now = datetime.now(UTC)
        period_start = self._buffer_start or (now - timedelta(hours=6))

        log = logger.bind(
            period_start=period_start.isoformat(),
            period_end=now.isoformat(),
        )
        log.info("Generating Flow 2 signal")

        # Get previous watchlist for delta tracking
        previous_watchlist = set(await self.watchlist.get_all())

        # Process buffered posts if any
        buffered_posts = [post for post, _ in self._post_buffer]
        refinement = None

        if buffered_posts:
            refinement = await self.process_posts(buffered_posts)

        # Clean up expired watchlist entries
        removed = await self.watchlist.cleanup_expired()

        # Get current watchlist
        current_watchlist = set(await self.watchlist.get_all())

        # Calculate deltas
        watchlist_added = list(current_watchlist - previous_watchlist)
        watchlist_removed = list(previous_watchlist - current_watchlist)

        # Build ticker sentiments
        ticker_sentiments: list[TickerSentimentSummary] = []
        if refinement:
            for ticker in refinement.validated_tickers:
                if ticker.is_valid_ticker:
                    # Calculate actual ratios from raw post-level sentiment scores
                    scores = self._raw_tickers.get(ticker.ticker, [])
                    if not scores:
                        logger.info(
                            "Skipping ticker not found in Gate 1 raw_tickers",
                            ticker=ticker.ticker,
                            company_name=ticker.company_name,
                            confidence=ticker.confidence,
                        )
                        continue

                    # VADER compound scores range from -1.0 to 1.0
                    # Dead zone: scores in (-0.1, 0.1) are classified as neutral
                    bullish_count = sum(1 for s in scores if s > 0.1)
                    bearish_count = sum(1 for s in scores if s < -0.1)
                    total = len(scores)
                    bullish = bullish_count / total
                    bearish = bearish_count / total

                    ticker_sentiments.append(
                        TickerSentimentSummary(
                            ticker=ticker.ticker,
                            company_name=ticker.company_name,
                            mention_count=len(scores),
                            bullish_ratio=bullish,
                            bearish_ratio=bearish,
                            avg_sentiment=ticker.avg_sentiment,
                            key_catalysts=ticker.key_catalysts,
                        )
                    )

        # Build subreddit breakdown
        subreddit_counts: dict[str, int] = {}
        for post, _ in self._post_buffer:
            subreddit_counts[post.subreddit] = subreddit_counts.get(post.subreddit, 0) + 1

        # Build signal
        signal = SentimentSignal(
            timestamp=now,
            signal_period="6h",
            period_start=period_start,
            period_end=now,
            watchlist=sorted(current_watchlist),
            watchlist_added=watchlist_added,
            watchlist_removed=watchlist_removed + removed,
            ticker_sentiments=ticker_sentiments,
            total_posts_analyzed=len(self._post_buffer),
            high_quality_posts=refinement.high_quality_posts if refinement else 0,
            spam_posts=refinement.spam_posts if refinement else 0,
            overall_sentiment=(refinement.overall_sentiment if refinement else "neutral"),
            narrative_summary=(refinement.narrative_summary if refinement else ""),
            key_themes=refinement.key_themes if refinement else [],
            sources={"reddit": len(self._post_buffer)},
            subreddits=subreddit_counts,
        )

        # Persist to PostgreSQL if available
        if self.db:
            try:
                # 1. Store the full signal
                await self.db.insert_sentiment_signal(signal)

                # 2. Store per-ticker sentiment snapshots
                for ts in signal.ticker_sentiments:
                    await self.db.insert_sentiment_snapshot(
                        ticker=ts.ticker,
                        snapshot_time=signal.timestamp,
                        bullish_ratio=ts.bullish_ratio,
                        bearish_ratio=ts.bearish_ratio,
                        mention_count=ts.mention_count,
                    )

                log.debug(
                    "Flow 2 signal persisted to PostgreSQL",
                    ticker_snapshots=len(signal.ticker_sentiments),
                )
            except Exception as e:
                log.error(
                    "Failed to persist Flow 2 signal to PostgreSQL",
                    error=str(e),
                    exc_info=True,
                )

        # Clear buffer for next period
        self._post_buffer = []
        self._buffer_start = None

        log.info(
            "Flow 2 signal generated",
            watchlist_size=len(signal.watchlist),
            added=len(signal.watchlist_added),
            removed=len(signal.watchlist_removed),
            tickers_analyzed=len(signal.ticker_sentiments),
            overall=signal.overall_sentiment,
            persisted=self.db is not None,
        )

        return signal

    async def close(self) -> None:
        """Clean up resources."""
        # Nothing to clean up currently
        pass


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_sentiment_processor(
    settings: Settings,
    redis: Redis,
    db: Database | None = None,
    ticker_provider: "TickerProvider | None" = None,
    watchlist: "WatchlistManager | None" = None,
) -> SentimentProcessor:
    """Create a Sentiment processor instance.

    Args:
        settings: Application settings
        redis: Redis client
        db: Optional PostgreSQL database for persistence
        ticker_provider: Optional TickerProvider for ticker verification
        watchlist: Optional shared WatchlistManager (created in __main__.py)

    Returns:
        Configured SentimentProcessor
    """
    processor = SentimentProcessor(
        settings,
        redis,
        db=db,
        ticker_provider=ticker_provider,
        watchlist=watchlist,
    )

    # Sync watchlist from database on startup (only if we created our own)
    if db and watchlist is None:
        await processor.watchlist.sync_from_db()

    return processor
