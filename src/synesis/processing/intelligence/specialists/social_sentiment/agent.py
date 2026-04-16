"""SocialSentimentAnalyst — extracts explicit tickers and macro themes from Twitter/X.

Reads raw tweets from the last 24h, extracts only explicitly mentioned tickers,
verifies them against tweet context, and identifies macro themes.
No analysis or scoring — information gathering only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model, web_search_config
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
_SEARCH_DESC = (
    "research thematic theses in depth, find supporting data, and discover related themes"
)


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
You extract key intelligence from curated Twitter/X financial accounts AND research \
the most important themes in depth. Your output feeds directly into bull/bear \
researchers who will form the investment thesis — your job is to give them clean, \
structured, fact-rich context with deep thematic research. \
Always preserve specifics: dollar amounts, percentages, dates, names, and sources.

Today's date: {current_date}

## Your Job

Extract key information from the tweets, then research the most compelling themes.

1. **Ticker Mentions**: Only tickers EXPLICITLY mentioned by name or cashtag ($NVDA) \
in the tweets.
   - Do NOT infer or guess tickers. A tweet about "oil stocks are red" should NOT \
     produce XOM, CVX, etc. unless those tickers are actually named in the tweet.
   - Do NOT include ETFs or indices (QQQ, SPY, SPX, IWM, DIA, VOO, etc.).
   - For each ticker, extract the CONTEXT of why it was mentioned — this context \
     passes downstream with the ticker. (e.g. "heavy call buying ahead of earnings", \
     "fraud allegations in accounting", "record revenue guidance").
   - Note the account's expertise and any known bias (e.g. short-seller — still include \
     the mention but note the bias).
   - If multiple independent accounts mention the same ticker, note the convergence.

2. **Macro Themes**: Broad market themes without specific tickers.
   - The theme, who's discussing it, and the key reasoning. Keep it factual.

3. **Summary**: 4-6 sentences capturing today's key intelligence. This is NOT a generic \
overview — it must be source-rich and specific:
   - Name the accounts behind each key claim: "@KobeissiLetter flags Strait of Hormuz \
blockade risk pushing crude above $115", not "oil prices may rise due to geopolitical tensions."
   - Include specific data points, dates, and figures from the tweets.
   - When you researched a theme via web search, weave the findings into the summary: \
"@jukan05 flagged HBM4 validation delays — DigiTimes reports Samsung 2nm yields remain \
below 30% as of Apr 10, confirming the bottleneck."
   - Highlight convergences: "3 independent accounts (@a, @b, @c) flagged X" is a stronger \
signal than a single mention.

4. **Thematic Research** (populate `research_context` and `discovered_themes`):
   This is where you add real value. When you find a compelling thesis — especially when \
2+ accounts converge on a theme, or a credible account pushes a non-obvious thesis:
   - USE web_search to dig into the underlying story. Don't just note the claim — research \
the full picture: supply chain data, recent filings, industry reports, earnings context.
   - Record findings in `research_context` with source attribution: \
"HBM4 supply: Samsung 2nm yield issues confirmed by DigiTimes Apr 12; TSMC CoWoS capacity \
reportedly sold out through Q3 per industry sources; SK Hynix HBM4 sampling delayed to Q2."
   - Surface RELATED themes the tweets missed. Researching one thesis often reveals connected \
angles. Record these in `discovered_themes`: e.g. researching HBM4 delays might surface \
"Intel Foundry 18A as alternative advanced packaging source" or "DRAM pricing power shifting \
to SK Hynix due to HBM allocation." These help downstream analysts see the full picture.

## Tools

- `verify_ticker(ticker)` — Verify a ticker exists or find a company's ticker symbol.
{search_docs}\
- `web_read(url)` — Read a web page for full article content. Unlimited calls.

## When to web_search (use your budget for thematic depth)
- **Research the underlying thesis** — not just "is it true?" but "what's the full picture?" \
Search for supply chain data, industry reports, recent filings, and earnings context. \
Record findings in `research_context` and related angles in `discovered_themes`.
- **Follow convergence signals** — when 2+ accounts discuss the same theme, search for the \
deeper story they might have missed.
- **Cross-check bold claims** — data points cited by short-sellers, permabulls, or anyone \
making outsized claims. Look for the actual filing, report, or data source.
- Do NOT search for routine market commentary that already has sufficient detail in the tweets.

## Rules
- ONLY extract explicitly mentioned tickers. NEVER infer additional tickers.
- NEVER include ETFs/indices (QQQ, SPY, IWM, DIA, VOO, VTI, XLF, XLE, XLK, etc.).
- Prioritize depth over breadth — 2-3 well-researched themes beat 5 shallow ones.
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
    search = web_search_config(_WEB_SEARCH_CAP, _SEARCH_DESC)
    tools: list[Any] = [_tool_verify_ticker, _tool_web_read]
    if not search.native:
        tools.append(_tool_web_search)

    agent: Agent[SocialSentimentDeps, SocialSentimentAnalysis] = Agent(
        model=create_model(smart=True),
        deps_type=SocialSentimentDeps,
        output_type=SocialSentimentAnalysis,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            search_docs=search.prompt_docs,
        ),
        tools=tools,
        builtin_tools=search.builtin_tools,
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
