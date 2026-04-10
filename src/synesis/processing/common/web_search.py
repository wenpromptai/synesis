"""Web search utility for market impact analysis.

Provides web search via Brave Search API (2000 req/month free tier).
Circuit breaker trips after 3 consecutive failures — all subsequent calls
return empty results for the rest of the process lifetime.

Crawl4AI page reading is handled separately by the Stage 2 LLM's web_read tool —
the LLM decides which URLs to read in full after reviewing search results.

The Brave rate limiter (_brave_lock, _brave_last_call) is a module-level singleton
shared across all callers. Interval: settings.brave_min_interval (default 1.5s).
"""

import asyncio
import json
import re
import time
from datetime import date, timedelta
from typing import Any, Literal

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)

# Fixed API URL
BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# Brave rate limiter — module-level singleton shared across all processors.
_brave_lock: asyncio.Lock | None = None
_brave_last_call: float = 0.0

# Circuit breaker — trips after _BRAVE_CB_THRESHOLD consecutive failures.
_BRAVE_CB_THRESHOLD = 3
_brave_fail_count: int = 0
_brave_tripped: bool = False


# Timeouts
SEARCH_TIMEOUT = 10.0

# Recency options
Recency = Literal["day", "week", "month", "year", "none"]

# Max chars of crawled article content to pass to LLM
_CRAWL_MAX_CHARS = 2000


async def search_market_impact(
    query: str,
    count: int = 5,
    recency: Recency = "day",
) -> list[dict[str, Any]]:
    """Search for market impact info via Brave Search.

    Returns empty list if the circuit breaker has tripped (quota/rate exhausted).

    Args:
        query: Search query (e.g., "Fed rate cut stocks affected")
        count: Number of results to return
        recency: Time range filter - "day", "week", "month", "year", or "none"

    Returns:
        List of dicts with 'title', 'snippet', and 'url' keys
    """
    if _brave_tripped:
        logger.info("Brave circuit breaker tripped — skipping search", query=query)
        return []

    settings = get_settings()

    if not settings.brave_api_key:
        logger.info("Brave API key not configured — skipping search")
        return []

    try:
        results = await _search_brave(
            query, count, settings.brave_api_key.get_secret_value(), recency
        )
        if results:
            logger.debug("Brave search successful", query=query, results=len(results))
            return results
    except (httpx.HTTPError, httpx.RequestError, json.JSONDecodeError) as e:
        logger.warning("Brave search failed", error=str(e))

    return []


def _get_date_range(recency: Recency) -> tuple[date | None, date]:
    """Get start date based on recency setting."""
    today = date.today()
    if recency == "day":
        return today - timedelta(days=1), today
    elif recency == "week":
        return today - timedelta(weeks=1), today
    elif recency == "month":
        return today - timedelta(days=30), today
    elif recency == "year":
        return today - timedelta(days=365), today
    return None, today


def _extract_article_content(markdown: str, max_chars: int = _CRAWL_MAX_CHARS) -> str:
    """Extract article body from crawled markdown, skipping nav and image noise.

    Finds the first heading to skip nav menus, then strips image/social-share
    lines and collapses blank lines.
    """
    # Start from first heading to skip nav menus at the top
    heading_match = re.search(r"^#{1,3}\s+\S", markdown, re.MULTILINE)
    start = heading_match.start() if heading_match else 0
    content = markdown[start:]

    # Strip pure image lines: ![alt](url)
    # Strip bare-link lines used for social share / nav (standalone or as list items):
    #   [ ](url)  or  * [ ](url)  or  - [ ](url)
    lines = content.split("\n")
    cleaned = [
        line
        for line in lines
        if not re.match(r"^\s*!\[.*?\]\(.*?\)\s*$", line)
        and not re.match(r"^\s*(?:[\*\-]\s*)*\[\s*\]\(.*?\)\s*$", line)
    ]

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result[:max_chars].strip()


# Default max chars for page reads
_READ_MAX_CHARS = 4000


async def read_web_page(url: str, max_chars: int = _READ_MAX_CHARS) -> str:
    """Read the full content of a web page via Crawl4AI.

    Shared utility used by LLM tool implementations across analyzers.
    Returns cleaned article content (~4000 chars) or an error message.

    Args:
        url: The URL to read
        max_chars: Maximum chars to return (default 4000)
    """
    settings = get_settings()
    if not settings.crawl4ai_url:
        return "Web reading not available (Crawl4AI not configured)."

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

    crawler = Crawl4AICrawlerProvider(base_url=settings.crawl4ai_url, timeout=15.0)
    try:
        result = await crawler.crawl(url)
        if result.success and result.markdown:
            content = _extract_article_content(result.markdown, max_chars=max_chars)
            if content:
                logger.debug("read_web_page success", url=url, chars=len(content))
                return content
        return "Page crawled but no readable content extracted."
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("read_web_page failed", url=url, error=str(e))
        return f"Failed to read page: {e}"
    finally:
        await crawler.close()


async def _search_brave(
    query: str, count: int, api_key: str, recency: Recency
) -> list[dict[str, Any]]:
    """Search using Brave Search API.

    Rate-limited via lazy-initialized asyncio.Lock + settings.brave_min_interval.
    Circuit breaker trips after _BRAVE_CB_THRESHOLD consecutive failures.
    """
    global _brave_lock, _brave_last_call, _brave_fail_count, _brave_tripped

    if _brave_lock is None:
        _brave_lock = asyncio.Lock()

    min_interval = get_settings().brave_min_interval
    async with _brave_lock:
        elapsed = time.monotonic() - _brave_last_call
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        _brave_last_call = time.monotonic()

    params: dict[str, Any] = {"q": query, "count": count}

    # Brave uses freshness parameter: pd (past day), pw (past week), pm (past month), py (past year)
    if recency != "none":
        freshness_map = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
        params["freshness"] = freshness_map.get(recency, "pw")

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.get(
            BRAVE_API_URL,
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params=params,
        )
        if response.status_code == 429:
            _brave_fail_count += 1
            if _brave_fail_count >= _BRAVE_CB_THRESHOLD:
                _brave_tripped = True
                logger.warning(
                    "Brave circuit breaker tripped — disabling web search for this process",
                    consecutive_failures=_brave_fail_count,
                )
            response.raise_for_status()

        response.raise_for_status()

        # Success (2xx) — reset failure counter
        _brave_fail_count = 0
        data = response.json()

        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("description", ""),
                "url": r.get("url", ""),
                "published_date": r.get("page_age", r.get("age", "")),
            }
            for r in data.get("web", {}).get("results", [])
        ]


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results for LLM — includes URLs so it can call web_read on them."""
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        line = f"{i}. **{title}**"
        if snippet:
            line += f"\n   {snippet[:200]}"
        if url:
            line += f"\n   URL: {url}"
        lines.append(line)

    lines.append(
        "\nCall web_read(url) on the most relevant URLs above to read full article content."
    )
    return "\n".join(lines)
