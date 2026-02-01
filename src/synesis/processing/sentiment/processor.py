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
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.ingestion.reddit import RedditPost
from synesis.intelligence.sentiment.analyzer import SentimentAnalyzer
from synesis.intelligence.sentiment.models import SentimentResult
from synesis.processing.sentiment.models import (
    SentimentSignal,
    SentimentRefinement,
    SentimentRefinementDeps,
    StockEmotion,
    TickerSentimentSummary,
)
from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker_finnhub as _verify_ticker
from synesis.processing.common.watchlist import WatchlistManager

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.config import Settings
    from synesis.ingestion.finnhub import FinnhubService
    from synesis.ingestion.prices import PriceService
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

Your job is to REFINE the lexicon analysis by:

## 1. Validate Tickers (CRITICAL)

The lexicon extracts ticker-like patterns but produces FALSE POSITIVES.

**Common False Positives to REJECT**:
- Common words in caps: WEEK, WHAT, OPEN, CLOSE, VS, FOR, THE, NOW, EOD, ATH, DTE, OTM, ITM
- Subreddit/platform terms: DD, YOLO, FOMO, WSB, OP, IMO, TL, DR, ELI, TLDR
- Finance terms not tickers: CEO, CFO, IPO, ETF, SEC, FED, GDP, CPI, EPS, PE, IV, DCA, LEAPS
- Partial matches: IT (not Intel), AM, PM, US, UK, EU, AI, EV, ER

**IMPORTANT - Meme Stocks ARE VALID**:
- GME (GameStop) - THE original meme stock
- AMC (AMC Entertainment) - movie theater meme stock
- DJT (Trump Media) - legitimate NASDAQ ticker, popular political meme stock
- BBBY (Bed Bath & Beyond) - even if delisted, was heavily discussed
- DWAC - Trump SPAC before merger
- Other politically/culturally significant stocks popular on WSB

**Validation Criteria**:
- Is traded on major exchange (NYSE, NASDAQ, etc.) - OR recently delisted but still discussed
- Context suggests stock discussion (mentions of calls, puts, positions, tendies, etc.)
- If unsure, USE THE verify_ticker_finnhub TOOL first, then web_search as fallback

For each ticker, provide:
- is_valid_ticker: true/false
- rejection_reason: Why it's not valid (if rejected)
- confidence: How confident you are (0.6+ to include)

**Ticker Verification Workflow**:
1. Call `verify_ticker_finnhub(ticker)` for US ticker verification
2. If VERIFIED: include with the company name returned
3. If NOT FOUND: use `web_search("{ticker} stock ticker price")` to check non-US/delisted
4. If still unclear: reject the ticker

**When to use verify_ticker_finnhub tool**:
- Ticker looks legitimate but you're not 100% sure it exists
- Short tickers (2-3 letters) that could be words OR stocks
- Political/cultural stocks you haven't heard of
- ANY ticker where rejection would be borderline

## 2. Assess Post Quality

Reddit posts vary wildly in quality:

**HIGH quality**:
- DD (Due Diligence) with research and thesis
- Earnings analysis with numbers
- Technical analysis with specific levels
- News with market implications

**MEDIUM quality**:
- Simple position posts with reasoning
- Questions that reveal sentiment
- News shares with brief commentary

**LOW quality**:
- Memes without substance
- One-word reactions
- Off-topic posts

**SPAM**:
- Promotional content
- Crypto shilling unrelated to stocks
- Bot-like repetitive content

## 3. Generate Narrative Summary

Create a 2-3 sentence narrative that:
- Summarizes the dominant sentiment and WHY
- Highlights any extreme positions or unusual activity
- Notes key themes (earnings, macro events, specific stocks)

Example:
"WSB sentiment is strongly bearish on silver (SLV) following today's flash crash, with multiple loss porn posts. PayPal (PYPL) seeing renewed interest ahead of earnings. Overall market mood is cautious with focus on Fed rate decision."

## 4. Identify Extreme Sentiment

Flag tickers with extreme sentiment (>85% one direction):
- These are potential contrarian signals
- High conviction crowd positions often precede reversals

## Sentiment Aggregation (for multi-post analysis)

When synthesizing sentiment across posts:
1. Weight by quality: HIGH=3x, MEDIUM=1x, LOW/SPAM=ignore
2. Calculate bullish vs bearish ratio per ticker
3. Flag "crowded" positions (>80% one direction) as contrarian signals
4. Distinguish event-driven sentiment (earnings, FDA) vs. general mood

## Narrative Summary Format

Your narrative_summary MUST follow this structure:
- Sentence 1: Overall market mood and dominant theme
- Sentence 2: Top 2-3 tickers with strongest sentiment and why
- Sentence 3: Key catalysts or upcoming events driving sentiment

Example: "WSB sentiment is cautiously bullish with focus on tech earnings. NVDA and AMD dominate discussion with bullish calls ahead of AI chip demand reports. Key catalyst is Fed decision Wednesday, with most expecting a hawkish hold."

## Confidence Calibration for Tickers

| Score | Criteria |
|-------|----------|
| 0.9-1.0 | verify_ticker_finnhub returned VERIFIED |
| 0.7-0.89 | Strong context (calls, puts, DD, positions mentioned) |
| 0.6-0.69 | Likely ticker but borderline context |
| <0.6 | Do not include |

## Guidelines
- Use verify_ticker_finnhub tool first, then web_search as fallback - better to verify than wrongly reject
- Context is key: "bought 100 DJT calls" = valid ticker, "THE market is up" = not a ticker
- WSB culture: "regarded" = regarded, "apes" = retail traders, "tendies" = gains
- Meme stocks and politically significant stocks ARE legitimate tickers
- Confidence scores should be conservative (0.7+ for high confidence)"""


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
        price_service: "PriceService | None" = None,
        finnhub: "FinnhubService | None" = None,
        watchlist: "WatchlistManager | None" = None,
    ) -> None:
        """Initialize Flow 2 processor.

        Args:
            settings: Application settings
            redis: Redis client for watchlist storage
            db: Optional PostgreSQL database for persistence
            price_service: Optional PriceService for fetching ticker prices
            finnhub: Optional FinnhubService for ticker verification
            watchlist: Optional shared WatchlistManager (created in __main__.py)
        """
        self.settings = settings
        self.redis = redis
        self.db = db
        self.price_service = price_service
        self._finnhub = finnhub

        # Use provided watchlist or create new one (for standalone use)
        if watchlist is not None:
            self.watchlist = watchlist
        else:
            # Create with callbacks for dynamic WebSocket subscription management
            on_ticker_added = None
            on_ticker_removed = None
            if price_service:

                async def subscribe_ticker(ticker: str) -> None:
                    await price_service.subscribe([ticker])

                async def unsubscribe_ticker(ticker: str) -> None:
                    await price_service.unsubscribe([ticker])

                on_ticker_added = subscribe_ticker
                on_ticker_removed = unsubscribe_ticker

            self.watchlist = WatchlistManager(
                redis,
                db=db,
                on_ticker_added=on_ticker_added,
                on_ticker_removed=on_ticker_removed,
            )

        self.lexicon_analyzer = SentimentAnalyzer()
        self._refiner_agent: Agent[SentimentRefinementDeps, SentimentRefinement] | None = None

        # In-memory buffer for posts between signal generations
        self._post_buffer: list[tuple[RedditPost, SentimentResult]] = []
        self._buffer_start: datetime | None = None

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
            output_type=SentimentRefinement,
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

            subreddits_str = ", ".join(deps.subreddits) if deps.subreddits else "various"

            return f"""
## Analysis Context
Total Posts: {len(deps.posts)}
Subreddits: {subreddits_str}
Unique Tickers (raw): {len(deps.raw_tickers)}

{posts_section}

{tickers_section}"""

        # Tool: Verify ticker via Finnhub (US exchanges only)
        @agent.tool
        async def verify_ticker_finnhub(
            ctx: RunContext[SentimentRefinementDeps],
            ticker: str,
        ) -> str:
            """Verify if a US ticker symbol exists using Finnhub.

            Use this tool to validate US tickers. For non-US or delisted tickers,
            use web_search as a fallback.

            Args:
                ticker: The US ticker symbol to verify (e.g., "AAPL", "GME", "DJT")

            Returns:
                Verification result - VERIFIED with company name, NOT FOUND, or error
            """
            return await _verify_ticker(ticker, self._finnhub)

        # Tool: Web search for ticker verification fallback
        @agent.tool
        async def web_search(
            ctx: RunContext[SentimentRefinementDeps],
            query: str,
        ) -> str:
            """Search the web for ticker verification or additional context.

            Use this as a fallback when verify_ticker_finnhub returns NOT FOUND,
            or when you need to verify non-US tickers or delisted stocks.

            Args:
                query: Search query (e.g., "DJT stock ticker price")

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

        # Gate 2: LLM refinement (smart, validates tickers, generates narrative)
        deps = SentimentRefinementDeps(
            posts=posts,
            lexicon_results=lexicon_results,
            raw_tickers=raw_tickers,
            subreddits=subreddits,
        )

        user_prompt = """Analyze the Reddit posts and lexicon results above.

1. **Validate each ticker** in the aggregated mentions table:
   - Mark false positives as is_valid_ticker=false
   - Only include tickers with confidence >= 0.6

2. **Assess post quality** for each post:
   - Label as high/medium/low/spam
   - Note if it's DD, YOLO, or contains thesis

3. **Generate narrative summary**:
   - 2-3 sentences summarizing the sentiment
   - Include specific tickers and catalysts

4. **Flag extreme sentiment**:
   - List tickers with >85% bullish or bearish

Be strict on ticker validation. Reject if unsure."""

        try:
            agent_result = await self.refiner_agent.run(user_prompt, deps=deps)
            refinement = agent_result.output

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
            # Return partial result from Gate 1 analysis
            return SentimentRefinement(
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
                    # Calculate ratios from sentiment label
                    bullish = 1.0 if ticker.sentiment_label == "bullish" else 0.0
                    bearish = 1.0 if ticker.sentiment_label == "bearish" else 0.0
                    neutral = 1.0 if ticker.sentiment_label == "neutral" else 0.0

                    # Determine emotion from sentiment
                    emotion = self._sentiment_to_emotion(ticker.avg_sentiment)

                    ticker_sentiments.append(
                        TickerSentimentSummary(
                            ticker=ticker.ticker,
                            company_name=ticker.company_name,
                            mention_count=ticker.mention_count,
                            bullish_ratio=bullish,
                            bearish_ratio=bearish,
                            neutral_ratio=neutral,
                            avg_sentiment=ticker.avg_sentiment,
                            dominant_emotion=emotion,
                            is_extreme_bullish=ticker.ticker
                            in (refinement.extreme_bullish_tickers or []),
                            is_extreme_bearish=ticker.ticker
                            in (refinement.extreme_bearish_tickers or []),
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
            extreme_sentiments=(
                (refinement.extreme_bullish_tickers or [])
                + (refinement.extreme_bearish_tickers or [])
                if refinement
                else []
            ),
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

                # 2. Fetch prices for all tickers (batch fetch for efficiency)
                ticker_prices: dict[str, Decimal] = {}
                if self.price_service and signal.ticker_sentiments:
                    try:
                        tickers = [ts.ticker for ts in signal.ticker_sentiments]
                        ticker_prices = await self.price_service.get_prices(
                            tickers,
                            fallback_to_rest=True,
                        )
                        if ticker_prices:
                            log.debug(
                                "Fetched prices for sentiment snapshots",
                                tickers=list(ticker_prices.keys()),
                            )
                    except Exception as e:
                        log.warning(
                            "Failed to fetch prices for sentiment snapshots",
                            error=str(e),
                        )

                # 3. Store per-ticker sentiment snapshots
                for ts in signal.ticker_sentiments:
                    price_at_signal = ticker_prices.get(ts.ticker.upper())
                    await self.db.insert_sentiment_snapshot(
                        ticker=ts.ticker,
                        snapshot_time=signal.timestamp,
                        bullish_ratio=ts.bullish_ratio,
                        bearish_ratio=ts.bearish_ratio,
                        neutral_ratio=ts.neutral_ratio,
                        mention_count=ts.mention_count,
                        dominant_emotion=ts.dominant_emotion.value if ts.dominant_emotion else None,
                        sentiment_delta_6h=ts.sentiment_delta_6h,
                        is_extreme_bullish=ts.is_extreme_bullish,
                        is_extreme_bearish=ts.is_extreme_bearish,
                        price_at_signal=price_at_signal,
                    )

                log.debug(
                    "Flow 2 signal persisted to PostgreSQL",
                    ticker_snapshots=len(signal.ticker_sentiments),
                    tickers_with_prices=len(ticker_prices),
                )
            except Exception as e:
                log.warning("Failed to persist Flow 2 signal to PostgreSQL", error=str(e))

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

    def _sentiment_to_emotion(self, sentiment: float) -> StockEmotion:
        """Convert sentiment score to emotion category."""
        if sentiment >= 0.6:
            return StockEmotion.euphoric
        elif sentiment >= 0.2:
            return StockEmotion.bullish
        elif sentiment <= -0.6:
            return StockEmotion.panic
        elif sentiment <= -0.2:
            return StockEmotion.fearful
        return StockEmotion.neutral

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
    price_service: "PriceService | None" = None,
    finnhub: "FinnhubService | None" = None,
    watchlist: "WatchlistManager | None" = None,
) -> SentimentProcessor:
    """Create a Sentiment processor instance.

    Args:
        settings: Application settings
        redis: Redis client
        db: Optional PostgreSQL database for persistence
        price_service: Optional PriceService for fetching ticker prices
        finnhub: Optional FinnhubService for ticker verification
        watchlist: Optional shared WatchlistManager (created in __main__.py)

    Returns:
        Configured SentimentProcessor
    """
    processor = SentimentProcessor(
        settings,
        redis,
        db=db,
        price_service=price_service,
        finnhub=finnhub,
        watchlist=watchlist,
    )

    # Sync watchlist from database on startup (only if we created our own)
    if db and watchlist is None:
        await processor.watchlist.sync_from_db()

    return processor
