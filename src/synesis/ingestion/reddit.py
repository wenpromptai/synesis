"""Reddit RSS client for polling subreddit feeds.

Uses Reddit's free native RSS feeds (no API key required).
Suitable for low-frequency polling (6 hours) with zero rate limit concerns.
"""

import asyncio
import html
import re
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

try:
    import feedparser
except ImportError as e:
    raise ImportError("feedparser is required: pip install feedparser") from e

logger = structlog.get_logger(__name__)


def _parse_rss_timestamp(time_struct: Any) -> datetime:
    """Parse RSS timestamp (struct_time or string) to datetime."""
    if time_struct is None:
        return datetime.now(timezone.utc)

    # feedparser returns time.struct_time
    if hasattr(time_struct, "tm_year"):
        try:
            from time import mktime

            return datetime.fromtimestamp(mktime(time_struct), tz=timezone.utc)
        except (ValueError, OverflowError):
            return datetime.now(timezone.utc)

    return datetime.now(timezone.utc)


def _clean_html(text: str) -> str:
    """Clean HTML content from Reddit RSS.

    Reddit RSS content includes HTML entities and some tags.
    """
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove common HTML tags but preserve content
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<a[^>]*href=[\"']([^\"']*)[\"'][^>]*>([^<]*)</a>", r"\2 (\1)", text)
    text = re.sub(r"<[^>]+>", "", text)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def _extract_selftext(entry: dict[str, Any]) -> str:
    """Extract selftext content from RSS entry.

    Reddit RSS puts self-post content in different places depending on format.
    """
    # Try content first (usually has the selftext)
    if content := entry.get("content"):
        if isinstance(content, list) and content:
            return _clean_html(content[0].get("value", ""))
        if isinstance(content, dict):
            return _clean_html(content.get("value", ""))

    # Try summary
    if summary := entry.get("summary"):
        return _clean_html(summary)

    return ""


@dataclass
class RedditPost:
    """Represents a Reddit post from RSS feed."""

    post_id: str
    subreddit: str
    author: str | None
    title: str
    content: str  # selftext (may be empty for link posts)
    url: str
    permalink: str
    timestamp: datetime
    raw: dict[str, Any]

    @property
    def full_text(self) -> str:
        """Get combined title and content for analysis."""
        if self.content:
            return f"{self.title}\n\n{self.content}"
        return self.title


@dataclass
class RedditRSSClient:
    """Poll Reddit RSS feeds for new posts.

    Uses Reddit's native RSS at https://www.reddit.com/r/{subreddit}/new/.rss
    """

    subreddits: list[str]
    poll_interval: int = 21600  # 6 hours in seconds
    user_agent: str = "Synesis/1.0 (Financial News Monitor)"

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _callbacks: list[Callable[[RedditPost], Coroutine[Any, Any, None]]] = field(
        default_factory=list, init=False, repr=False
    )
    _running: bool = field(default=False, init=False, repr=False)
    _poll_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _seen_ids: OrderedDict[str, None] = field(default_factory=OrderedDict, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def on_post(self, callback: Callable[[RedditPost], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for new posts.

        Args:
            callback: Async function called with each new post
        """
        self._callbacks.append(callback)

    async def fetch_subreddit(self, subreddit: str) -> list[RedditPost]:
        """Fetch posts from a subreddit RSS feed.

        Args:
            subreddit: Subreddit name (without r/)

        Returns:
            List of RedditPost objects
        """
        client = self._get_client()
        url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
        log = logger.bind(subreddit=subreddit, url=url)

        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("reddit_rss_http_error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            log.error("reddit_rss_request_error", error=str(e))
            return []

        # Parse RSS feed
        try:
            feed = feedparser.parse(response.text)
        except Exception as e:
            log.error("reddit_rss_parse_error", error=str(e))
            return []

        posts: list[RedditPost] = []
        for entry in feed.entries:
            try:
                post = self._parse_entry(entry, subreddit)
                posts.append(post)
            except Exception as e:
                log.warning("reddit_entry_parse_error", error=str(e), entry_id=entry.get("id"))

        log.debug("reddit_rss_fetched", count=len(posts))
        return posts

    def _parse_entry(self, entry: dict[str, Any], subreddit: str) -> RedditPost:
        """Parse RSS entry into RedditPost."""
        # Extract post ID - Reddit RSS uses full thing names like "t3_abc123"
        # or URLs like "https://www.reddit.com/r/.../comments/abc123/..."
        post_id = entry.get("id", "")

        # Try to extract from URL pattern first
        if "/comments/" in post_id:
            match = re.search(r"/comments/([a-z0-9]+)", post_id)
            if match:
                post_id = match.group(1)
        # Handle Reddit thing names (t3_xxx for links/posts)
        elif post_id.startswith("t3_"):
            post_id = post_id[3:]  # Remove "t3_" prefix

        # Parse author (Reddit RSS includes "/u/" prefix)
        author = None
        if author_detail := entry.get("author_detail"):
            author = author_detail.get("name")
        elif author_str := entry.get("author"):
            author = author_str

        # Strip /u/ prefix if present
        if author and author.startswith("/u/"):
            author = author[3:]

        # Parse timestamp
        timestamp = _parse_rss_timestamp(entry.get("published_parsed"))

        return RedditPost(
            post_id=post_id,
            subreddit=subreddit,
            author=author,
            title=entry.get("title", ""),
            content=_extract_selftext(entry),
            url=entry.get("link", ""),
            permalink=entry.get("link", ""),
            timestamp=timestamp,
            raw=dict(entry),
        )

    async def poll_all_subreddits(self) -> list[RedditPost]:
        """Poll all configured subreddits and return new posts.

        Returns:
            List of new posts (not seen before)
        """
        all_posts: list[RedditPost] = []

        # Fetch from all subreddits
        for subreddit in self.subreddits:
            posts = await self.fetch_subreddit(subreddit)
            all_posts.extend(posts)
            # Small delay between requests to be polite
            await asyncio.sleep(1.0)

        # Filter to only new posts
        new_posts = [p for p in all_posts if p.post_id not in self._seen_ids]

        # Update seen IDs (OrderedDict maintains insertion order)
        for post in all_posts:
            self._seen_ids[post.post_id] = None
            self._seen_ids.move_to_end(post.post_id)

        # Limit seen IDs size (keep newest 5000 when exceeding 10000)
        while len(self._seen_ids) > 10000:
            self._seen_ids.popitem(last=False)  # Remove oldest entries

        logger.info(
            "reddit_poll_complete",
            total_posts=len(all_posts),
            new_posts=len(new_posts),
            subreddits=len(self.subreddits),
        )

        return new_posts

    async def _poll_loop(self) -> None:
        """Internal polling loop."""
        log = logger.bind(
            subreddits=self.subreddits,
            interval_hours=self.poll_interval / 3600,
        )
        log.info("reddit_polling_started")

        # Initial poll
        first_run = True

        while self._running:
            try:
                posts = await self.poll_all_subreddits()

                # Invoke callbacks for all posts (including first run for sentiment analysis)
                for post in posts:
                    for callback in self._callbacks:
                        try:
                            await callback(post)
                        except Exception as e:
                            logger.error(
                                "reddit_callback_error",
                                error=str(e),
                                post_id=post.post_id,
                            )

                if first_run:
                    log.info(
                        "reddit_initial_poll_complete",
                        seen_ids=len(self._seen_ids),
                        posts_processed=len(posts),
                    )
                    first_run = False

            except Exception as e:
                log.error("reddit_poll_error", error=str(e))

            if self._running:
                await asyncio.sleep(self.poll_interval)

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("reddit_client_started", subreddits=self.subreddits)

    async def stop(self) -> None:
        """Stop the polling loop and cleanup."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("reddit_client_stopped")

    async def __aenter__(self) -> "RedditRSSClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()
