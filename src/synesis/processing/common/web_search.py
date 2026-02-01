"""Web search utility for market impact analysis.

Provides web search with fallback chain: SearXNG -> Exa -> Brave.
Used by the LLM classifier to enrich news with:
- Affected tickers/stocks
- Historical precedent (similar events and outcomes)
- Market movement analysis
"""

import json
from datetime import date, timedelta
from typing import Any, Literal

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)


class SearchProvidersExhaustedError(Exception):
    """Raised when all search providers fail or are not configured."""

    pass


# Timeouts
SEARCH_TIMEOUT = 10.0

# Recency options
Recency = Literal["day", "week", "month", "year", "none"]


async def search_market_impact(
    query: str,
    count: int = 5,
    recency: Recency = "day",
) -> list[dict[str, Any]]:
    """Search for market impact info with fallback chain.

    Tries SearXNG first (self-hosted, no limits), then Exa, then Brave.

    Args:
        query: Search query (e.g., "Fed rate cut stocks affected")
        count: Number of results to return
        recency: Time range filter - "day", "week", "month", "year", or "none"

    Returns:
        List of dicts with 'title', 'snippet', and 'url' keys
    """
    settings = get_settings()

    # Try SearXNG first (self-hosted, free, no rate limits)
    if settings.searxng_url:
        try:
            results = await _search_searxng(query, count, settings.searxng_url, recency)
            if results:
                logger.debug("SearXNG search successful", query=query, results=len(results))
                return results
        except (httpx.HTTPError, httpx.RequestError, json.JSONDecodeError, KeyError) as e:
            logger.warning("SearXNG search failed, trying Exa", error=str(e))

    # Fallback to Exa (best for financial/news content)
    if settings.exa_api_key:
        try:
            results = await _search_exa(
                query, count, settings.exa_api_key.get_secret_value(), recency
            )
            if results:
                logger.debug("Exa search successful", query=query, results=len(results))
                return results
        except (httpx.HTTPError, httpx.RequestError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Exa search failed, trying Brave", error=str(e))

    # Fallback to Brave
    if settings.brave_api_key:
        try:
            results = await _search_brave(
                query, count, settings.brave_api_key.get_secret_value(), recency
            )
            if results:
                logger.debug("Brave search successful", query=query, results=len(results))
                return results
        except (httpx.HTTPError, httpx.RequestError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Brave search failed", error=str(e))

    logger.error("All search providers failed or not configured", query=query)
    raise SearchProvidersExhaustedError(
        "All search providers failed or not configured. "
        "Configure at least one of: SEARXNG_URL, EXA_API_KEY, or BRAVE_API_KEY"
    )


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


async def _search_searxng(
    query: str, count: int, base_url: str, recency: Recency
) -> list[dict[str, Any]]:
    """Search using self-hosted SearXNG instance.

    SearXNG aggregates results from multiple search engines (Google, Bing, DDG, etc.)
    and provides a JSON API with no rate limits when self-hosted.
    """
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": "general,news",
    }

    # Add time range filter for recent results
    if recency != "none":
        params["time_range"] = recency

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.get(f"{base_url.rstrip('/')}/search", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", [])[:count]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")[:300] if r.get("content") else "",
                    "url": r.get("url", ""),
                }
            )
        return results


async def _search_exa(
    query: str, count: int, api_key: str, recency: Recency
) -> list[dict[str, Any]]:
    """Search using Exa API (neural search, great for financial content)."""
    payload: dict[str, Any] = {
        "query": query,
        "numResults": count,
        "type": "neural",
        "useAutoprompt": True,
        "contents": {"text": {"maxCharacters": 300}},
    }

    # Add date filter for recent results
    if recency != "none":
        start_date, _ = _get_date_range(recency)
        if start_date:
            payload["startPublishedDate"] = start_date.isoformat()

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("text", "")[:300],
                "url": r.get("url", ""),
            }
            for r in data.get("results", [])
        ]


async def _search_brave(
    query: str, count: int, api_key: str, recency: Recency
) -> list[dict[str, Any]]:
    """Search using Brave Search API."""
    params: dict[str, Any] = {"q": query, "count": count}

    # Brave uses freshness parameter: pd (past day), pw (past week), pm (past month), py (past year)
    if recency != "none":
        freshness_map = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
        params["freshness"] = freshness_map.get(recency, "pw")

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("description", ""),
                "url": r.get("url", ""),
            }
            for r in data.get("web", {}).get("results", [])
        ]


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results as text for LLM consumption."""
    if not results:
        return "No search results found."

    lines = []
    for r in results:
        title = r.get("title", "Untitled")
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"- {title}: {snippet}")
        else:
            lines.append(f"- {title}")

    return "\n".join(lines)
