"""SocialSentimentAnalyst — analyzes Twitter/X feeds for trading signals and macro themes.

Reads raw tweets from the last 24h, applies account bias/credibility weighting,
uses web tools to verify context, and outputs ticker mentions + macro themes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.ticker_tools import verify_ticker
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.models import SocialSentimentAnalysis
from synesis.processing.intelligence.specialists.social_sentiment.x_accounts import (
    get_profile,
)

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 5


@dataclass
class SocialSentimentDeps:
    """Dependencies for SocialSentimentAnalyst."""

    db: Database
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = 0


# ── Data Gathering ───────────────────────────────────────────────


async def _gather_tweets(db: Database, since_hours: int = 24) -> list[dict[str, Any]]:
    """Fetch raw tweets and return as list of dicts."""
    rows = await db.get_raw_tweets(since_hours=since_hours)
    return [dict(r) for r in rows]


def _format_tweets_by_account(tweets: list[dict[str, Any]]) -> str:
    """Format tweets grouped by account with profile context.

    Output:
    ## Tweets by Account (last 24h)

    ### @javierblas (macro_commodities)
    Bloomberg commodities columnist. Oil, gas, metals, energy geopolitics.

    - [2026-04-06 08:30] "Oil prices surging on OPEC+ supply cut..."
    """
    if not tweets:
        return "No tweets found in the last 24 hours."

    # Group by account
    by_account: dict[str, list[dict[str, Any]]] = {}
    for tw in tweets:
        username = tw.get("account_username", "unknown")
        by_account.setdefault(username, []).append(tw)

    sections: list[str] = []
    sections.append(
        f"## Tweets by Account ({len(tweets)} tweets from {len(by_account)} accounts)\n"
    )

    for username, account_tweets in sorted(by_account.items()):
        profile = get_profile(username)
        if profile:
            sections.append(f"### @{username} ({profile.category})")
            sections.append(f"{profile.description}\n")
        else:
            sections.append(f"### @{username} (uncategorized)\n")

        for tw in sorted(account_tweets, key=lambda t: t.get("tweet_timestamp", "")):
            ts = tw.get("tweet_timestamp", "")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            text = tw.get("tweet_text", "").replace("\n", " ")
            sections.append(f'- [{ts}] "{text}"')

        sections.append("")  # blank line between accounts

    return "\n".join(sections)


# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a financial analyst extracting key intelligence from curated Twitter/X accounts.

Today's date: {current_date}

## Your Job

Extract the most valuable information from the tweets provided:

1. **Ticker Mentions**: Every stock ticker mentioned or implied by any account.
   - What was said, by whom, and the context (e.g. "heavy call buying ahead of earnings",
     "fraud allegations in accounting", "record revenue guidance").
   - Note the account's expertise and any known bias (e.g. short-seller — still include
     the mention but note the bias).
   - If multiple independent accounts mention the same ticker, highlight the convergence.

2. **Macro Themes**: Broad market themes without specific tickers.
   - The theme, who's discussing it, and the reasoning (e.g. "risk-off rotation —
     @elerianm and @lizannsonders both flagging defensive positioning").

3. **Summary**: 2-3 sentences capturing the overall mood and key takeaways.

## Tools

- `verify_ticker(ticker)` — Verify a ticker exists or find a company's ticker symbol.
- `web_search(query, recency)` — Search for context on developing stories. Budget: {web_search_cap} calls.
- `web_read(url)` — Read a web page for full article content. Unlimited calls.

## Guidelines
- Include ALL tickers mentioned.
- Context quality matters: "NVDA heavy call buying ahead of earnings" is useful;
  "NVDA mentioned" is not.
- Highlight when accounts with different expertise areas converge on the same ticker or theme.
- Use web_search to verify significant claims and add context.
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_verify_ticker(ctx: RunContext[SocialSentimentDeps], ticker: str) -> str:
    """Verify if a ticker symbol exists."""
    return await verify_ticker(ticker)


async def _tool_web_search(
    ctx: RunContext[SocialSentimentDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for market context. Budget: limited calls."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[SocialSentimentDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Public API ───────────────────────────────────────────────────


async def analyze_social_sentiment(deps: SocialSentimentDeps) -> SocialSentimentAnalysis:
    """Run the SocialSentimentAnalyst on recent tweets.

    Fetches tweets from the last 24h, formats them by account with
    profile context, then runs the LLM agent to extract ticker
    mentions and macro themes.
    """
    logger.info("Starting SocialSentimentAnalyst")

    # Gather and format tweets
    tweets = await _gather_tweets(deps.db)
    if not tweets:
        logger.warning("No tweets found in last 24h — returning empty analysis")
        return SocialSentimentAnalysis(
            ticker_mentions=[],
            macro_themes=[],
            summary="No tweets available for analysis.",
            analysis_date=deps.current_date,
        )

    formatted = _format_tweets_by_account(tweets)
    logger.info("Tweets formatted", tweet_count=len(tweets))

    # Construct agent at runtime with formatted system prompt
    agent: Agent[SocialSentimentDeps, SocialSentimentAnalysis] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=SocialSentimentDeps,
        output_type=SocialSentimentAnalysis,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            web_search_cap=_WEB_SEARCH_CAP,
        ),
        tools=[_tool_verify_ticker, _tool_web_search, _tool_web_read],
    )

    try:
        result = await agent.run(formatted, deps=deps)
        output: SocialSentimentAnalysis = result.output
    except Exception:
        logger.exception("SocialSentimentAnalyst LLM call failed")
        return SocialSentimentAnalysis(
            ticker_mentions=[],
            macro_themes=[],
            summary="[LLM synthesis failed — social sentiment analysis unavailable]",
            analysis_date=deps.current_date,
        )

    logger.info(
        "SocialSentimentAnalyst complete",
        ticker_mentions=len(output.ticker_mentions),
        macro_themes=len(output.macro_themes),
    )

    # Ensure analysis_date is set
    if output.analysis_date != deps.current_date:
        output = output.model_copy(update={"analysis_date": deps.current_date})

    return output
