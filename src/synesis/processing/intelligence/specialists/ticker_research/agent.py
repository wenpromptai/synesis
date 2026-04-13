"""TickerResearchAnalyst — pre-gathers social + news context for debate agents.

Searches Twitter ($TICKER, min 500 likes, last 5 days) and the web (last week)
for each ticker. Extracts key analysis, thesis, narratives, and claims.
The output is passed to bull/bear debate agents so they can focus on
building investment cases instead of gathering basic information.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.ingestion.twitterapi import TwitterClient
from synesis.processing.common.llm import create_model, web_search_config
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.models import TickerResearchAnalysis

logger = get_logger(__name__)

_WEB_SEARCH_PER_TICKER = 3
_SEARCH_DESC = "find latest news, analyst commentary, earnings updates, or regulatory actions"


@dataclass
class TickerResearchDeps:
    """Dependencies for TickerResearchAnalyst."""

    twitter_client: TwitterClient | None
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = field(default=0, init=False)
    max_web_searches: int = 15


# ── Data Gathering ───────────────────────────────────────────────


async def _gather_tweets_for_ticker(
    client: TwitterClient,
    ticker: str,
    since_date: date,
) -> list[dict[str, Any]]:
    """Search Twitter for a ticker's cashtag with quality filter."""
    query = f"${ticker} min_faves:500 -filter:replies since:{since_date.isoformat()}"
    try:
        tweets, _ = await client.search_tweets(query, query_type="Latest")
        return [
            {
                "username": t.username,
                "text": t.text,
                "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "likes": t.raw.get("likeCount", 0),
                "retweets": t.raw.get("retweetCount", 0),
            }
            for t in tweets
        ]
    except Exception:
        logger.exception("Twitter search failed", ticker=ticker)
        return []


async def _gather_all_tweets(
    client: TwitterClient | None,
    tickers: list[str],
    since_date: date,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch tweets for all tickers sequentially with rate-limit pauses."""
    if client is None:
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = await _gather_tweets_for_ticker(client, ticker, since_date)
        if i < len(tickers) - 1:
            await asyncio.sleep(0.5)
    return results


def _format_tweets_for_prompt(
    tweets_by_ticker: dict[str, list[dict[str, Any]]],
) -> str:
    """Format gathered tweets into structured text for the LLM."""
    if not tweets_by_ticker:
        return "No Twitter data available."

    sections: list[str] = []
    for ticker, tweets in tweets_by_ticker.items():
        if not tweets:
            sections.append(f"## ${ticker}\nNo high-engagement tweets found.\n")
            continue

        sections.append(f"## ${ticker} ({len(tweets)} tweets, 500+ likes)")
        for tw in sorted(tweets, key=lambda t: t.get("likes", 0), reverse=True):
            sections.append(
                f"- @{tw['username']} ({tw['timestamp']}) "
                f"[{tw['likes']} likes, {tw['retweets']} RT]\n"
                f"  {tw['text'][:500]}"
            )
        sections.append("")

    return "\n".join(sections)


# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research analyst pre-gathering intelligence on specific tickers. \
Your output feeds directly into bull/bear debate researchers — give them clean, \
structured, fact-rich context so they can build investment cases without \
searching for basic information themselves.

Today's date: {current_date}
Tickers to research: {tickers}

## Your Job

1. **Analyze the Twitter data** provided below for each ticker:
   - Extract key thesis points and investment narratives being discussed
   - Note who is saying what (credibility matters — fund managers > random accounts)
   - Identify bullish vs bearish arguments with specific evidence cited
   - Flag any claims that seem unverified or exaggerated

2. **Web search for latest news** on each ticker (use your budget wisely):
   - Search for each ticker's latest news, earnings, analyst actions, insider moves
   - Prioritize material events: earnings, guidance changes, analyst upgrades/downgrades, \
regulatory actions, insider transactions, M&A, product launches
   - Read articles to get full details — headlines are not enough

3. **Verify key claims** from Twitter if they cite specific numbers or events:
   - If a tweet claims "AAOI is up 332% YTD" or "Citron is shorting", verify it
   - Mark claims as verified or unverified in your output

4. **For each ticker, produce:**
   - `social_highlights`: Key points from Twitter (with attribution)
   - `news_highlights`: Key recent news items (with dates and sources)
   - `key_narratives`: The main investment narrative(s) being discussed
   - `sentiment_lean`: Overall social+news sentiment (bullish/bearish/mixed/neutral)
   - `verified_claims`: Claims you confirmed via web search
   - `unverified_claims`: Claims that couldn't be verified

## Tools
{search_docs}\
- `web_read(url)` — Read a web page for full article content. Unlimited calls.

## Rules
- Be specific: "$AAOI rallied 10% on Lumentum backlog report" not "stock went up"
- Preserve numbers, dates, names, and sources
- If Twitter data is unavailable for a ticker, focus on web search for news
- Do NOT form investment opinions — just gather and organize facts
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[TickerResearchDeps],
    query: str,
    recency: str = "week",
) -> str:
    """Search the web for recent news and context."""
    if ctx.deps.web_search_calls >= ctx.deps.max_web_searches:
        return f"Web search budget exhausted ({ctx.deps.max_web_searches} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "week"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[TickerResearchDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Public API ───────────────────────────────────────────────────


async def analyze_ticker_research(
    tickers: list[str],
    deps: TickerResearchDeps,
) -> TickerResearchAnalysis:
    """Run the TickerResearchAnalyst for a set of tickers.

    Gathers Twitter data and web news, then uses an LLM to extract
    key intelligence for downstream debate agents.
    """
    logger.info("Starting TickerResearchAnalyst", tickers=tickers)

    # Scale web search budget by ticker count
    deps.max_web_searches = min(len(tickers) * _WEB_SEARCH_PER_TICKER, 15)

    # Gather Twitter data (last 5 days)
    since_date = deps.current_date - timedelta(days=5)
    tweets_by_ticker = await _gather_all_tweets(deps.twitter_client, tickers, since_date)

    total_tweets = sum(len(v) for v in tweets_by_ticker.values())
    logger.info(
        "Twitter data gathered",
        tickers=len(tweets_by_ticker),
        total_tweets=total_tweets,
    )

    formatted_tweets = _format_tweets_for_prompt(tweets_by_ticker)

    # Build LLM agent
    search = web_search_config(deps.max_web_searches, _SEARCH_DESC)
    tools: list[Any] = [_tool_web_read]
    if not search.native:
        tools.append(_tool_web_search)

    agent: Agent[TickerResearchDeps, TickerResearchAnalysis] = Agent(
        model=create_model(smart=True),
        deps_type=TickerResearchDeps,
        output_type=TickerResearchAnalysis,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            tickers=", ".join(f"${t}" for t in tickers),
            search_docs=search.prompt_docs,
        ),
        tools=tools,
        builtin_tools=search.builtin_tools,
    )

    try:
        result = await agent.run(formatted_tweets, deps=deps)
        output: TickerResearchAnalysis = result.output
    except Exception:
        logger.exception("TickerResearchAnalyst LLM call failed")
        return TickerResearchAnalysis(
            research=[],
            summary="[LLM synthesis failed — ticker research unavailable]",
            analysis_date=deps.current_date,
        )

    # Ensure analysis_date is set
    if not output.analysis_date:
        output.analysis_date = deps.current_date

    logger.info(
        "TickerResearchAnalyst complete",
        tickers_researched=len(output.research),
        web_searches_used=deps.web_search_calls,
    )

    return output
