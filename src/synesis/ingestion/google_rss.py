"""Google News RSS feed poller.

Ingests news from Google News RSS feeds (topic feeds and search feeds) on a
configurable interval. Each new article is converted to a UnifiedMessage and
pushed through a callback into the same processing queue as Telegram messages.

Deduplication is two-layered:
  1. GUID dedup (this module) — each RSS item has a unique GUID. On every poll
     cycle the feed returns ~100 items, most already seen. We store seen GUIDs
     as Redis keys with a 48h TTL so repeated items are skipped before they
     ever hit the processing queue.
  2. Semantic dedup (downstream) — the existing MessageDeduplicator uses
     Model2Vec cosine similarity (threshold 0.85) to catch cross-source
     duplicates, e.g. the same story arriving via both Telegram and RSS.

Google News RSS specifics:
  - <link> values are Google redirect URLs, not real article URLs. We resolve
    them via HTTP HEAD + follow_redirects, rate-limited to 2 concurrent
    requests to avoid 429s. Resolved URL stored in raw.resolved_url.
  - <title> includes the source as a suffix ("Headline - Bloomberg.com").
    We strip it so the impact scorer sees a clean headline.
  - <source> element provides the publisher name (e.g. "Reuters") which
    becomes source_account and feeds into SOURCE_RELIABILITY scoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import defusedxml.ElementTree as ET
import httpx
from defusedxml import DefusedXmlException
from redis.asyncio import Redis
from redis.exceptions import RedisError

from synesis.core.logging import get_logger
from synesis.processing.news.models import SourcePlatform, UnifiedMessage

logger = get_logger(__name__)

# Each seen GUID stored as a Redis key with this prefix and 48h TTL.
# After expiry the GUID could be re-ingested, but by then the article
# is stale and the semantic dedup would catch it anyway.
# NOTE: This is separate from the semantic dedup in deduplication.py
# (synesis:news:dedup:emb:*) which catches cross-source duplicates.
_SEEN_PREFIX = "synesis:google_rss:seen:"
_SEEN_TTL_SECONDS = 48 * 3600

# Max concurrent HTTP HEAD requests for resolving Google redirect URLs.
_RESOLVE_SEMAPHORE_LIMIT = 2


@dataclass
class RSSItem:
    """Single parsed item from a Google News RSS feed."""

    guid: str  # unique identifier (base64-encoded by Google)
    title: str  # headline with trailing source name stripped
    link: str  # Google redirect URL (not the real article URL)
    pub_date: datetime
    source_name: str  # publisher from <source> element, e.g. "Reuters"
    source_url: str  # publisher domain from <source url="...">
    description: str  # HTML of clustered related headlines (not article body)


def _clean_title(title: str, source_name: str) -> str:
    """Strip trailing ' - SourceName' that Google News appends to titles.

    Google titles look like "Fed Raises Rates by 25bps - Reuters".
    We want just the headline for the impact scorer and ticker matcher.
    """
    if source_name and title.endswith(f" - {source_name}"):
        return title[: -(len(source_name) + 3)]
    # Fallback: split on last ' - ' when source_name doesn't match exactly
    parts = title.rsplit(" - ", 1)
    if len(parts) == 2:
        return parts[0]
    return title


def _parse_pub_date(text: str) -> datetime:
    """Parse RFC 2822 date string (e.g. 'Fri, 03 Apr 2026 16:55:44 GMT')."""
    try:
        return parsedate_to_datetime(text).astimezone(UTC)
    except Exception:
        return datetime.now(UTC)


def parse_feed_xml(xml_bytes: bytes) -> list[RSSItem]:
    """Parse Google News RSS 2.0 XML into a list of RSSItem.

    Skips items missing <title> or <guid>. Items missing optional fields
    (<source>, <pubDate>, <description>) get safe defaults.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except (ET.ParseError, DefusedXmlException):
        logger.warning("Failed to parse RSS XML")
        return []

    items: list[RSSItem] = []
    for item_el in root.iter("item"):
        title_el = item_el.find("title")
        link_el = item_el.find("link")
        guid_el = item_el.find("guid")
        pub_date_el = item_el.find("pubDate")
        source_el = item_el.find("source")
        desc_el = item_el.find("description")

        if title_el is None or title_el.text is None:
            continue
        if guid_el is None or guid_el.text is None:
            continue

        source_name = source_el.text if source_el is not None and source_el.text else ""
        source_url = source_el.get("url", "") if source_el is not None else ""
        title = _clean_title(title_el.text, source_name)

        items.append(
            RSSItem(
                guid=guid_el.text,
                title=title,
                link=link_el.text if link_el is not None and link_el.text else "",
                pub_date=_parse_pub_date(pub_date_el.text)
                if pub_date_el is not None and pub_date_el.text
                else datetime.now(UTC),
                source_name=source_name,
                source_url=source_url,
                description=desc_el.text if desc_el is not None and desc_el.text else "",
            )
        )

    return items


class GoogleRSSPoller:
    """Polls Google News RSS feeds and pushes new articles to a callback.

    Same interface as TelegramListener: register a callback via on_message(),
    then start()/stop() to control the polling loop.

    Each poll cycle fetches all configured feeds, parses the XML, checks each
    item's GUID against Redis to skip already-seen articles, resolves the
    Google redirect URL, and invokes the callback with a UnifiedMessage.
    """

    def __init__(
        self,
        feeds: list[str],
        poll_interval: int,
        redis: Redis,
    ) -> None:
        """Initialize the poller.

        Args:
            feeds: Google News RSS feed URLs to poll.
            poll_interval: Minutes between poll cycles.
            redis: Redis client for GUID dedup storage.
        """
        self._feeds = feeds
        self._poll_interval = poll_interval * 60  # minutes → seconds
        self._redis = redis
        self._message_callback: Any = None
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._resolve_semaphore = asyncio.Semaphore(_RESOLVE_SEMAPHORE_LIMIT)
        self._seeded = False  # first poll seeds the GUID cache without processing

    async def start(self) -> None:
        """Start the polling loop as a background asyncio task."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={"User-Agent": "Synesis/1.0 (+https://github.com/synesis)"},
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Google RSS poller started",
            feeds=len(self._feeds),
            interval_min=self._poll_interval // 60,
        )

    async def stop(self) -> None:
        """Cancel the polling task and close the HTTP client."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("Google RSS poller stopped")

    def on_message(self, callback: Any) -> None:
        """Register callback for new articles (same interface as TelegramListener)."""
        self._message_callback = callback

    async def _poll_loop(self) -> None:
        """Infinite loop: fetch all feeds, process new items, sleep.

        First poll is a seed run — marks all current GUIDs as seen without
        processing. This prevents a flood of old articles on startup. Only
        subsequent polls push genuinely new items through the callback.
        """
        # Small delay so other services (Redis, DB) finish initializing
        await asyncio.sleep(5)

        while True:
            try:
                if not self._seeded:
                    self._seeded = await self._seed_seen_cache()
                else:
                    total_new = 0
                    for feed_url in self._feeds:
                        total_new += await self._poll_feed(feed_url)

                    if total_new > 0:
                        logger.info("RSS poll complete", new_items=total_new)
                    else:
                        logger.debug("RSS poll complete", new_items=0)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("RSS poll cycle failed")

            await asyncio.sleep(self._poll_interval)

    async def _seed_seen_cache(self) -> bool:
        """First-run seed: fetch all feeds and mark every GUID as seen.

        This ensures that when the app starts up, existing articles are cached
        for dedup but NOT processed or sent to Discord. Only articles that
        appear in subsequent polls (i.e. genuinely new) will be processed.

        Returns True if at least one feed was successfully fetched.
        """
        if not self._client:
            return False

        total_seeded = 0
        feeds_ok = 0
        for url in self._feeds:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("RSS seed fetch failed", url=url[:80], error=str(e))
                continue

            feeds_ok += 1
            items = parse_feed_xml(resp.content)
            for item in items:
                if not await self._is_seen(item.guid):
                    await self._mark_seen(item.guid)
                    total_seeded += 1

        if feeds_ok == 0:
            logger.warning(
                "RSS seed failed — no feeds reachable, will retry next cycle",
                feeds=len(self._feeds),
            )
            return False

        logger.info(
            "RSS seed complete — cached existing articles, will only process new ones",
            seeded=total_seeded,
            feeds_ok=feeds_ok,
            feeds_total=len(self._feeds),
        )
        return True

    async def _poll_feed(self, url: str) -> int:
        """Fetch one feed, deduplicate by GUID, push new items to callback."""
        if not self._client:
            return 0

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("RSS feed fetch failed", url=url[:80], error=str(e))
            return 0

        items = parse_feed_xml(resp.content)
        new_count = 0

        for item in items:
            # Layer 1 dedup: skip if this GUID was already processed
            if await self._is_seen(item.guid):
                continue

            await self._mark_seen(item.guid)

            resolved_url = await self._resolve_url(item.link)
            message = self._to_unified(item, resolved_url)

            if self._message_callback:
                try:
                    await self._message_callback(message)
                    new_count += 1
                except Exception:
                    logger.exception(
                        "RSS callback error",
                        guid=item.guid,
                        title=item.title[:80],
                    )

        return new_count

    async def _is_seen(self, guid: str) -> bool:
        """Check Redis for an existing GUID key (layer 1 dedup)."""
        try:
            return bool(await self._redis.exists(f"{_SEEN_PREFIX}{guid}"))
        except RedisError:
            logger.warning("Redis error in GUID dedup check, treating as unseen", guid=guid)
            return False

    async def _mark_seen(self, guid: str) -> None:
        """Store GUID in Redis with 48h TTL so it's skipped on future polls."""
        try:
            await self._redis.set(f"{_SEEN_PREFIX}{guid}", "1", ex=_SEEN_TTL_SECONDS)
        except RedisError:
            logger.warning("Redis error storing seen GUID", guid=guid)

    async def _resolve_url(self, google_url: str) -> str:
        """Follow Google redirect to get the real article URL.

        Rate-limited to 2 concurrent requests. Returns the original
        Google URL on failure (non-critical for processing).
        """
        if not self._client or not google_url:
            return google_url

        async with self._resolve_semaphore:
            try:
                resp = await self._client.head(google_url, follow_redirects=True)
                return str(resp.url)
            except httpx.HTTPError:
                return google_url

    @staticmethod
    def _to_unified(item: RSSItem, resolved_url: str) -> UnifiedMessage:
        """Convert a parsed RSSItem into a UnifiedMessage for the queue.

        Text is just the cleaned headline — Google News RSS provides no
        article body. Stage 2's web_read tool fetches full content when needed.
        """
        return UnifiedMessage(
            external_id=item.guid,
            source_platform=SourcePlatform.google_rss,
            source_account=item.source_name,
            text=item.title,
            timestamp=item.pub_date,
            raw={
                "guid": item.guid,
                "link": item.link,
                "resolved_url": resolved_url,
                "source_name": item.source_name,
                "source_url": item.source_url,
            },
        )
