"""Surprise event detection via web search.

Discovers major market events from yesterday that aren't in our calendar DB.
Uses the existing SearXNG/Exa/Brave search fallback chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson

from synesis.core.constants import SURPRISE_MAX_RESULTS, SURPRISE_SEARCH_QUERIES
from synesis.core.logging import get_logger
from synesis.processing.common.web_search import SearchProvidersExhaustedError, search_market_impact

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

SURPRISE_CACHE_TTL = 21600  # 6 hours


async def detect_surprise_events(redis: Redis) -> list[dict[str, Any]]:
    """Search for major market events from yesterday not in our calendar.

    Returns list of dicts: {title, snippet, url, published_date}
    """
    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)
    cache_key = f"synesis:event_radar:surprise:{today.isoformat()}"

    # Check cache first
    cached = await redis.get(cache_key)
    if cached:
        try:
            return orjson.loads(cached)  # type: ignore[no-any-return]
        except Exception:
            pass

    # Build date range string: "March 5 2026 to March 6 2026"
    date_range = f"{yesterday.strftime('%B %-d %Y')} to {today.strftime('%B %-d %Y')}"

    all_results: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for query_template in SURPRISE_SEARCH_QUERIES:
        query = query_template.format(date_range=date_range)
        try:
            results = await search_market_impact(query, count=5, recency="day")
            for r in results:
                title = r.get("title", "").strip()
                if not title:
                    continue
                # Simple dedup: skip if title is substring of or contains an existing title
                title_lower = title.lower()
                is_dup = any(
                    title_lower in existing or existing in title_lower for existing in seen_titles
                )
                if not is_dup:
                    seen_titles.add(title_lower)
                    all_results.append(r)
        except SearchProvidersExhaustedError:
            logger.warning("Search providers exhausted during surprise detection")
            break
        except Exception:
            logger.warning("Surprise search query failed", query=query, exc_info=True)
            continue

    # Trim to max results
    surprises = all_results[:SURPRISE_MAX_RESULTS]

    # Cache results
    try:
        await redis.set(cache_key, orjson.dumps(surprises), ex=SURPRISE_CACHE_TTL)
    except Exception:
        logger.warning("Failed to cache surprise events")

    logger.info("Surprise event detection complete", count=len(surprises))
    return surprises
