"""Google News RSS feed poller.

Ingests news from Google News RSS feeds (topic feeds and search feeds) on a
configurable interval. Each new article is converted to a UnifiedMessage and
pushed through the same processing queue as Telegram messages.

Deduplication:
  1. Freshness filter (this module) — articles older than 15 minutes are
     skipped. Google News RSS returns up to 24h of articles, but we only
     care about fresh ones. This eliminates stale articles on every poll
     and on startup (no seed mechanism needed).
  2. Semantic dedup (shared with processor) — the MessageDeduplicator uses
     Model2Vec cosine similarity (threshold 0.85) to catch duplicates.
     Google News rotates GUIDs for some articles between polls, so GUID-
     based dedup is unreliable. Semantic dedup catches these rotating-GUID
     duplicates as well as cross-source duplicates (e.g. same story from
     both Telegram and RSS).

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
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import defusedxml.ElementTree as ET
import httpx
from defusedxml import DefusedXmlException

from synesis.core.logging import get_logger
from synesis.processing.news.deduplication import MessageDeduplicator
from synesis.processing.news.models import SourcePlatform, UnifiedMessage

logger = get_logger(__name__)

# Articles older than this are skipped. Keeps the 60-min dedup TTL
# sufficient to catch rotating GUIDs for fresh articles.
_FRESHNESS_MINUTES = 15

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
    """Parse RFC 2822 date string (e.g. 'Fri, 03 Apr 2026 16:55:44 GMT').

    Falls back to epoch on parse failure so the article is treated as stale
    by the freshness filter (safe default — never accidentally process junk).
    """
    try:
        return parsedate_to_datetime(text).astimezone(UTC)
    except Exception:
        logger.warning("Failed to parse pubDate, treating as stale", raw_date=text[:80])
        return datetime.min.replace(tzinfo=UTC)


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
                else datetime.min.replace(tzinfo=UTC),
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

    Each poll cycle fetches all configured feeds, parses the XML, filters
    out stale articles, runs semantic dedup, and invokes the callback with
    a UnifiedMessage for each unique fresh article.
    """

    def __init__(
        self,
        feeds: list[str],
        poll_interval: int,
        deduplicator: MessageDeduplicator,
    ) -> None:
        """Initialize the poller.

        Args:
            feeds: Google News RSS feed URLs to poll.
            poll_interval: Minutes between poll cycles.
            deduplicator: Semantic deduplicator shared with the processor.
        """
        self._feeds = feeds
        self._poll_interval = poll_interval * 60  # minutes → seconds
        self._deduplicator = deduplicator
        self._message_callback: Any = None
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._resolve_semaphore = asyncio.Semaphore(_RESOLVE_SEMAPHORE_LIMIT)

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
        """Infinite loop: fetch all feeds, process new items, sleep."""
        # Small delay so other services (Redis, DB) finish initializing
        await asyncio.sleep(5)

        while True:
            try:
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

    async def _poll_feed(self, url: str) -> int:
        """Fetch one feed, filter stale articles, dedup, push new items."""
        if not self._client:
            return 0

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("RSS feed fetch failed", url=url[:80], error=str(e))
            return 0

        items = parse_feed_xml(resp.content)
        cutoff = datetime.now(UTC) - timedelta(minutes=_FRESHNESS_MINUTES)
        new_count = 0

        for item in items:
            # Skip stale articles — only process fresh ones
            if item.pub_date < cutoff:
                continue

            # Semantic dedup first (check only — processor stores after processing)
            message = self._to_unified(item, item.link)
            dedup_result = await self._deduplicator.check_duplicate(message)
            if dedup_result.is_duplicate:
                logger.debug(
                    "RSS item dropped by semantic dedup",
                    title=item.title[:80],
                    similarity=f"{dedup_result.similarity:.3f}"
                    if dedup_result.similarity
                    else None,
                )
                continue

            # Resolve Google redirect URL only for unique articles
            resolved_url = await self._resolve_url(item.link)
            message.raw["resolved_url"] = resolved_url

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
