"""NewsAnalyst — synthesizes pre-scored news messages into story clusters.

Reads raw_messages (last 24h, impact_score >= 20) that have been pre-scored
by Flow 1 Stage 1. Groups related messages into story clusters, extracts
tickers with sentiment/magnitude/confidence, and identifies macro themes.
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
from synesis.processing.intelligence.models import NewsAnalysis

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 5


@dataclass
class NewsDeps:
    """Dependencies for NewsAnalyst."""

    db: Database
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = 0


# ── Data Gathering ───────────────────────────────────────────────


async def _gather_messages(
    db: Database, since_hours: int = 24, min_impact_score: int = 20
) -> list[dict[str, Any]]:
    """Fetch pre-scored raw messages from DB."""
    return await db.get_raw_messages(since_hours=since_hours, min_impact_score=min_impact_score)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Format messages chronologically with metadata.

    Output:
    ## News Messages (47 messages, last 24h, impact >= 20)

    - [2026-04-06 08:30] impact=85, tickers=[NVDA, AMD], source=@DeItaone
      "BREAKING: NVIDIA to acquire AMD for $300B..."
    """
    if not messages:
        return "No high-impact news messages found in the last 24 hours."

    sections: list[str] = []
    sections.append(f"## News Messages ({len(messages)} messages, last 24h, impact >= 20)\n")

    for msg in sorted(messages, key=lambda m: m.get("source_timestamp", "")):
        ts = msg.get("source_timestamp", "")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%Y-%m-%d %H:%M")
        impact = msg.get("impact_score", 0)
        tickers = msg.get("tickers", [])
        tickers_str = f", tickers=[{', '.join(tickers)}]" if tickers else ""
        source = msg.get("source_account", "unknown")
        text = msg.get("raw_text", "").replace("\n", " ")
        sections.append(f"- [{ts}] impact={impact}{tickers_str}, source={source}")
        sections.append(f'  "{text}"')

    return "\n".join(sections)


# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a financial news analyst. You synthesize breaking news into structured intelligence.

Today's date: {current_date}

## Your Job

You receive pre-scored news messages from the last 24 hours. Each has an impact score (0-100)
and pre-extracted tickers from a rule-based system. Your job is deeper synthesis:

1. **Story Clusters**: Group related messages about the same event into one story.
   Multiple messages about the same deal = 1 cluster with key facts consolidated.
   - Classify by event_type: earnings, m&a, regulatory, macro, geopolitical, management,
     legal, product, financing, other
   - Assess urgency: critical (act now), high (today), normal (this week), low (background)
   - Extract key facts: deal size, percentage changes, deadlines, named parties, regulatory status

2. **Ticker Extraction**: For each story, identify which companies are involved and HOW.
   - Context matters: "NVDA is the acquirer at $300B" vs "INTC dropped on competitive concerns"
     are very different — capture the nature of involvement.
   - The pre-extracted tickers are a starting point but may have false positives or miss non-US
     tickers. Use `verify_ticker` to check unfamiliar ones.

3. **Macro Themes**: Broad themes spanning multiple stories.
   - What's happening, why it matters, and what it implies (e.g. "tariff escalation across
     multiple sectors suggests broader risk-off positioning").

4. **Summary**: 2-3 sentences capturing the most important market-moving news today.

## Tools

- `verify_ticker(ticker)` — Verify a ticker or find a company's ticker symbol.
- `web_search(query, recency)` — Search for additional context. Budget: {web_search_cap} calls.
- `web_read(url)` — Read a web page for full article content. Unlimited calls.

## Guidelines
- Group aggressively — 3 messages about the same topic = 1 cluster, not 3.
- Context quality matters: "AAPL acquiring Perplexity AI for $12B per WSJ sources" is useful;
  "AAPL mentioned in news" is not.
- Include ALL tickers mentioned.
- Use web_search + web_read to verify major claims and add detail.
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_verify_ticker(ctx: RunContext[NewsDeps], ticker: str) -> str:
    """Verify if a ticker symbol exists."""
    return await verify_ticker(ticker)


async def _tool_web_search(
    ctx: RunContext[NewsDeps],
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


async def _tool_web_read(ctx: RunContext[NewsDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


# ── Public API ───────────────────────────────────────────────────


async def analyze_news(deps: NewsDeps) -> NewsAnalysis:
    """Run the NewsAnalyst on recent pre-scored messages.

    Fetches high-impact messages from the last 24h, formats them
    by source channel, then runs the LLM agent to group into
    story clusters and extract ticker signals.
    """
    logger.info("Starting NewsAnalyst")

    # Gather and format messages
    messages = await _gather_messages(deps.db)
    if not messages:
        logger.warning("No high-impact messages found — returning empty analysis")
        return NewsAnalysis(
            story_clusters=[],
            macro_themes=[],
            summary="No high-impact news in the last 24 hours.",
            analysis_date=deps.current_date,
            messages_analyzed=0,
        )

    formatted = _format_messages(messages)
    logger.info("Messages formatted", message_count=len(messages))

    # Construct agent at runtime with formatted system prompt
    agent: Agent[NewsDeps, NewsAnalysis] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=NewsDeps,
        output_type=NewsAnalysis,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            web_search_cap=_WEB_SEARCH_CAP,
        ),
        tools=[_tool_verify_ticker, _tool_web_search, _tool_web_read],
    )

    try:
        result = await agent.run(formatted, deps=deps)
        output: NewsAnalysis = result.output
    except Exception:
        logger.exception("NewsAnalyst LLM call failed")
        return NewsAnalysis(
            story_clusters=[],
            macro_themes=[],
            summary="[LLM synthesis failed — news analysis unavailable]",
            analysis_date=deps.current_date,
            messages_analyzed=len(messages),
        )

    logger.info(
        "NewsAnalyst complete",
        story_clusters=len(output.story_clusters),
        macro_themes=len(output.macro_themes),
        messages_analyzed=len(messages),
    )

    # Ensure metadata is set
    if output.analysis_date != deps.current_date:
        output = output.model_copy(update={"analysis_date": deps.current_date})
    if output.messages_analyzed != len(messages):
        output = output.model_copy(update={"messages_analyzed": len(messages)})

    return output
