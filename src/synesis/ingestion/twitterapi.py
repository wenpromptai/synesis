"""Twitter client using twitterapi.io API (WebSocket Stream + REST)."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
import websockets
from websockets.asyncio.client import ClientConnection

logger = structlog.get_logger(__name__)

# Twitter's standard timestamp format: "Sun Jan 18 13:14:48 +0000 2026"
TWITTER_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_twitter_timestamp(timestamp_str: str | None) -> datetime:
    """Parse timestamp from Twitter API, handling multiple formats."""
    if not timestamp_str:
        return datetime.now(timezone.utc)

    # Try ISO format first (with Z or +00:00)
    try:
        timestamp_str = timestamp_str.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        pass

    # Try Twitter's standard format: "Sun Jan 18 13:14:48 +0000 2026"
    try:
        return datetime.strptime(timestamp_str, TWITTER_TIMESTAMP_FORMAT)
    except ValueError:
        pass

    # Fallback to current time
    logger.warning("unknown_timestamp_format", timestamp=timestamp_str)
    return datetime.now(timezone.utc)


def _extract_full_text(data: dict[str, Any]) -> str:
    """Extract full tweet text, handling extended tweets and Twitter Notes.

    Twitter provides full text in different fields depending on tweet type:
    - note_tweet.text: Twitter Notes (long-form, 4000+ chars)
    - extended_tweet.full_text: Legacy extended format
    - full_text: Direct extended text field
    - text: Standard tweet text (may be truncated at ~280 chars)
    """
    # Twitter Notes (long-form, 4000+ chars)
    if note_tweet := data.get("note_tweet"):
        if note_text := note_tweet.get("text"):
            return str(note_text)

    # Extended tweet format
    if extended := data.get("extended_tweet"):
        if full_text := extended.get("full_text"):
            return str(full_text)

    # Direct full_text field
    if full_text := data.get("full_text"):
        return str(full_text)

    # Fallback to standard text
    text: str = data.get("text", "")
    return text


@dataclass
class Tweet:
    """Represents a tweet from the Twitter API."""

    tweet_id: str
    user_id: str
    username: str
    text: str
    timestamp: datetime
    raw: dict[str, Any]


@dataclass
class TwitterClient:
    """Async Twitter client using twitterapi.io API with polling-based updates."""

    api_key: str
    base_url: str = "https://api.twitterapi.io"
    accounts: list[str] = field(default_factory=list)
    poll_interval: float = 60.0
    request_delay: float = 2.0  # Delay between requests to avoid rate limits

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _last_seen: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _callbacks: list[Callable[[Tweet], Coroutine[Any, Any, None]]] = field(
        default_factory=list, init=False, repr=False
    )
    _running: bool = field(default=False, init=False, repr=False)
    _poll_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"x-api-key": self.api_key},
                timeout=30.0,
            )
        return self._client

    async def get_user_tweets(
        self, username: str, cursor: str | None = None
    ) -> tuple[list[Tweet], str | None]:
        """Fetch tweets for a user.

        Args:
            username: Twitter username (without @)
            cursor: Pagination cursor for subsequent pages

        Returns:
            Tuple of (tweets, next_cursor)
        """
        client = self._get_client()
        params: dict[str, str] = {"userName": username}
        if cursor:
            params["cursor"] = cursor

        log = logger.bind(username=username)

        try:
            response = await client.get("/twitter/user/last_tweets", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            log.error("twitter_api_error", status_code=e.response.status_code)
            raise
        except httpx.RequestError as e:
            log.error("twitter_request_error", error=str(e))
            raise

        tweets: list[Tweet] = []
        for tweet_data in data.get("tweets", []):
            try:
                tweet = self._parse_tweet(tweet_data)
                tweets.append(tweet)
            except (KeyError, ValueError) as e:
                log.warning("tweet_parse_error", error=str(e), tweet_id=tweet_data.get("id"))
                continue

        next_cursor = data.get("next_cursor")
        log.debug("fetched_tweets", count=len(tweets), has_next=bool(next_cursor))

        return tweets, next_cursor

    def _parse_tweet(self, data: dict[str, Any]) -> Tweet:
        """Parse raw API response into Tweet object."""
        timestamp_str = data.get("createdAt") or data.get("created_at")
        timestamp = parse_twitter_timestamp(timestamp_str)

        author = data.get("author", {})

        return Tweet(
            tweet_id=data["id"],
            user_id=author.get("id", ""),
            username=author.get("userName", author.get("username", "")),
            text=_extract_full_text(data),
            timestamp=timestamp,
            raw=data,
        )

    async def _fetch_account_tweets(self, username: str) -> list[Tweet]:
        """Fetch new tweets for a single account."""
        log = logger.bind(username=username)

        try:
            tweets, _ = await self.get_user_tweets(username)
        except (httpx.HTTPStatusError, httpx.RequestError):
            return []

        if not tweets:
            return []

        # Sort by ID (newer tweets have higher IDs)
        tweets.sort(key=lambda t: t.tweet_id)

        last_seen = self._last_seen.get(username)
        new_tweets = []

        for tweet in tweets:
            # Skip tweets we've already seen
            if last_seen and tweet.tweet_id <= last_seen:
                continue
            new_tweets.append(tweet)

        # Update last seen to newest tweet
        if tweets:
            self._last_seen[username] = tweets[-1].tweet_id
            log.debug("updated_last_seen", tweet_id=tweets[-1].tweet_id)

        return new_tweets

    async def poll_accounts(self) -> AsyncIterator[Tweet]:
        """Poll all configured accounts and yield new tweets.

        Fetches up to 5 accounts concurrently to speed up polling.
        """
        sem = asyncio.Semaphore(5)

        async def fetch_with_sem(username: str) -> list[Tweet]:
            async with sem:
                return await self._fetch_account_tweets(username)

        # Fetch all accounts concurrently (max 5 at a time)
        tasks = [fetch_with_sem(username) for username in self.accounts]
        results = await asyncio.gather(*tasks)

        # Yield tweets from all accounts
        for tweets in results:
            for tweet in tweets:
                yield tweet

    def on_tweet(self, callback: Callable[[Tweet], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for new tweets.

        Args:
            callback: Async function called with each new tweet
        """
        self._callbacks.append(callback)

    async def _poll_loop(self) -> None:
        """Internal polling loop."""
        log = logger.bind(accounts=self.accounts, interval=self.poll_interval)
        log.info("polling_started")

        while self._running:
            try:
                async for tweet in self.poll_accounts():
                    for callback in self._callbacks:
                        try:
                            await callback(tweet)
                        except Exception as e:
                            logger.error("callback_error", error=str(e), tweet_id=tweet.tweet_id)
            except Exception as e:
                log.error("poll_error", error=str(e))

            if self._running:
                await asyncio.sleep(self.poll_interval)

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("twitter_client_started", accounts=self.accounts)

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

        logger.info("twitter_client_stopped")

    async def __aenter__(self) -> "TwitterClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()


@dataclass
class TwitterStreamClient:
    """Real-time Twitter stream client using twitterapi.io WebSocket API.

    Requires filter rules to be configured at twitterapi.io web interface.
    This is the recommended approach for monitoring multiple accounts.
    """

    api_key: str
    ws_url: str = "wss://ws.twitterapi.io/twitter/tweet/websocket"
    reconnect_delay: float = 5.0
    max_reconnect_delay: float = 60.0

    _ws: ClientConnection | None = field(default=None, init=False, repr=False)
    _callbacks: list[Callable[[Tweet], Coroutine[Any, Any, None]]] = field(
        default_factory=list, init=False, repr=False
    )
    _running: bool = field(default=False, init=False, repr=False)
    _listen_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def _parse_tweet(self, data: dict[str, Any]) -> Tweet:
        """Parse raw tweet data from WebSocket into Tweet object."""
        timestamp_str = data.get("createdAt") or data.get("created_at")
        timestamp = parse_twitter_timestamp(timestamp_str)

        author = data.get("author", {})

        return Tweet(
            tweet_id=data.get("id", ""),
            user_id=author.get("id", ""),
            username=author.get("userName", author.get("username", "")),
            text=_extract_full_text(data),
            timestamp=timestamp,
            raw=data,
        )

    def on_tweet(self, callback: Callable[[Tweet], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for new tweets."""
        self._callbacks.append(callback)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.warning("ws_invalid_json", error=str(e))
            return

        event_type = data.get("event_type")

        if event_type == "connected":
            logger.info("ws_connected")
        elif event_type == "ping":
            logger.debug("ws_ping", timestamp=data.get("timestamp"))
        elif event_type == "tweet":
            # Process tweets from the event
            tweets_data = data.get("tweets", [])
            rule_tag = data.get("rule_tag", "")

            for tweet_data in tweets_data:
                try:
                    tweet = self._parse_tweet(tweet_data)
                    logger.info(
                        "tweet_received",
                        tweet_id=tweet.tweet_id,
                        username=tweet.username,
                        rule_tag=rule_tag,
                    )

                    for callback in self._callbacks:
                        try:
                            await callback(tweet)
                        except Exception as e:
                            logger.error("callback_error", error=str(e), tweet_id=tweet.tweet_id)
                except Exception as e:
                    logger.warning("tweet_parse_error", error=str(e))
        else:
            logger.debug("ws_unknown_event", event_type=event_type, data=data)

    async def _listen_loop(self) -> None:
        """Main WebSocket listen loop with reconnection."""
        delay = self.reconnect_delay

        while self._running:
            try:
                logger.info("ws_connecting", url=self.ws_url)

                async with websockets.connect(
                    self.ws_url,
                    additional_headers={"x-api-key": self.api_key},
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    delay = self.reconnect_delay  # Reset delay on successful connect
                    logger.info("ws_connected")

                    async for message in ws:
                        if not self._running:
                            break
                        if isinstance(message, bytes):
                            message = message.decode("utf-8")
                        await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning("ws_connection_closed", code=e.code, reason=e.reason)
            except Exception as e:
                logger.error("ws_error", error=str(e))

            self._ws = None

            if self._running:
                logger.info("ws_reconnecting", delay=delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def start(self) -> None:
        """Start the WebSocket stream."""
        if self._running:
            return

        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("twitter_stream_started")

    async def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        logger.info("twitter_stream_stopped")

    async def __aenter__(self) -> "TwitterStreamClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()
