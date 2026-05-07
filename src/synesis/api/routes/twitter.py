"""Twitter agent endpoints."""

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.requests import Request

from synesis.api.utils import create_tracked_task
from synesis.config import get_settings
from synesis.core.dependencies import AgentStateDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter
from synesis.ingestion.twitterapi import TwitterClient

router = APIRouter()

logger = get_logger(__name__)

# Hold references to background tasks so they aren't GC'd
_background_tasks: set[asyncio.Task[None]] = set()

# A ticker is allowed to be letters/numbers with optional dots (for tickers like BRK.B).
# Keeping this strict prevents a user-supplied ticker from turning into a Twitter
# advanced-search operator such as "OR" or "from:someone".
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_MAX_TICKERS_PER_SEARCH = 10


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TweetItem(BaseModel):
    """Single tweet result."""

    id: str = Field(..., description="Tweet ID")
    username: str = Field(..., description="Author username")
    text: str = Field(..., description="Tweet text (full if available)")
    timestamp: str = Field(..., description="ISO 8601 creation timestamp")
    likes: int = Field(default=0, description="Like count")
    retweets: int = Field(default=0, description="Retweet count")


class PerTickerResult(BaseModel):
    """Search result for a single ticker."""

    query: str = Field(..., description="The exact Twitter search query executed")
    count: int = Field(..., description="Number of tweets returned")
    tweets: list[TweetItem] = Field(default_factory=list, description="Tweet results")
    error: str | None = Field(default=None, description="Error message if this ticker search failed")


class TweetSearchResponse(BaseModel):
    """Response for multi-ticker Twitter search."""

    results: dict[str, PerTickerResult] = Field(
        default_factory=dict,
        description="Map of uppercase ticker → search result",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze")
@limiter.limit("5/minute")
async def trigger_twitter_agent(request: Request, state: AgentStateDep) -> dict[str, str]:
    """Manually trigger the daily Twitter agent.

    Fires the same job the scheduler runs at 10:00 ET. Pulls recent tweets from
    accounts in `TWITTER_ACCOUNTS`, runs them through the LLM digest, and posts
    to the Twitter Discord webhook. Runs in the background — returns immediately.

    **Inputs:** none.

    **Returns:**
    - `status` (str): `"triggered"` on success.
    - `message` (str): human-readable confirmation.

    **Errors:**
    - `503` if `TWITTERAPI_API_KEY` is missing or `TWITTER_ACCOUNTS` is empty
      (the trigger isn't registered at startup in that case).

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/twitter/analyze
    # {"status":"triggered","message":"Twitter data collection job started in background"}
    ```
    """
    trigger = state.trigger_fns.get("twitter_agent")
    if trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Twitter agent not configured (missing TWITTERAPI_API_KEY or TWITTER_ACCOUNTS)",
        )

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        if exc := t.exception():
            logger.error(
                "Twitter agent background task failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    create_tracked_task(trigger(), _background_tasks, _on_done)
    return {"status": "triggered", "message": "Twitter data collection job started in background"}


@router.get("/search", response_model=TweetSearchResponse)
@limiter.limit("30/minute")
async def search_twitter_by_ticker(
    request: Request,
    tickers: Annotated[
        str,
        Query(description="Comma-separated ticker symbols, e.g. NVDA,AMD,AAPL"),
    ],
    min_faves: Annotated[
        int,
        Query(ge=0, description="Minimum likes filter added as min_faves: inside the search query"),
    ] = 200,
    since_days: Annotated[
        int,
        Query(ge=1, le=30, description="Search tweets from this many days ago through now"),
    ] = 5,
    query_type: Literal["Latest", "Top"] = "Top",
    exclude_replies: bool = True,
) -> TweetSearchResponse:
    """Search Twitter for multiple ticker cashtags via twitterapi.io.

    Builds an advanced search query per ticker like `$TICKER min_faves:200
    -filter:replies since_time:1777670400` and returns raw tweets directly.
    Tickers are searched sequentially with a small rate-limit pause between
    requests. If one ticker fails, the others still return.

    **Query params:**
    - `tickers` (str): Comma-separated tickers, e.g. `NVDA,AMD,AAPL`.
    - `min_faves` (int, default `200`): Minimum likes filter (`min_faves:`).
    - `since_days` (int, default `5`): How many days back to search.
    - `query_type` (str, default `"Top"`): `"Latest"` or `"Top"`.
    - `exclude_replies` (bool, default `true`): Exclude reply tweets.

    **Returns:** `TweetSearchResponse` with a `results` map keyed by uppercase ticker.

    **Errors:**
    - `503` if `TWITTERAPI_API_KEY` is not configured.
    - `422` if no tickers provided or more than 10.

    **Example:**
    ```bash
    curl "http://localhost:7337/api/v1/twitter/search?tickers=NVDA,AMD&min_faves=200&since_days=3"
    ```
    """
    parsed = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    parsed = list(dict.fromkeys(parsed))  # dedup preserve order
    if not parsed:
        raise HTTPException(status_code=422, detail="No valid tickers provided")
    if len(parsed) > _MAX_TICKERS_PER_SEARCH:
        raise HTTPException(status_code=422, detail=f"Max {_MAX_TICKERS_PER_SEARCH} tickers per request")

    invalid_tickers = [ticker for ticker in parsed if not _TICKER_RE.fullmatch(ticker)]
    if invalid_tickers:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ticker(s): {', '.join(invalid_tickers)}",
        )

    settings = get_settings()
    api_key = settings.twitterapi_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured (missing TWITTERAPI_API_KEY)",
        )

    # twitterapi.io Advanced Search expects time filters as Unix seconds inside
    # the query string (`since_time:`), not as a separate API parameter.
    since_time = int((datetime.now(UTC) - timedelta(days=since_days)).timestamp())
    client = TwitterClient(
        api_key=api_key.get_secret_value(),
        base_url=settings.twitter_api_base_url,
    )

    results: dict[str, PerTickerResult] = {}

    try:
        for i, ticker in enumerate(parsed):
            query_parts = [f"${ticker}"]
            if min_faves > 0:
                query_parts.append(f"min_faves:{min_faves}")
            if exclude_replies:
                query_parts.append("-filter:replies")
            query_parts.append(f"since_time:{since_time}")
            query = " ".join(query_parts)

            try:
                tweets, _ = await client.search_tweets(query, query_type=query_type)
                items = [
                    TweetItem(
                        id=t.tweet_id,
                        username=t.username,
                        text=t.text,
                        timestamp=t.timestamp.isoformat(),
                        likes=t.raw.get("likeCount", 0),
                        retweets=t.raw.get("retweetCount", 0),
                    )
                    for t in tweets
                ]
                results[ticker] = PerTickerResult(
                    query=query,
                    count=len(items),
                    tweets=items,
                )
            except Exception as exc:
                logger.error(
                    "twitter_search_failed",
                    ticker=ticker,
                    query=query,
                    error=str(exc),
                )
                results[ticker] = PerTickerResult(
                    query=query,
                    count=0,
                    tweets=[],
                    error=str(exc),
                )

            # Rate-limit pause between tickers (not after the last one)
            if i < len(parsed) - 1:
                await asyncio.sleep(0.5)
    finally:
        await client.stop()

    return TweetSearchResponse(results=results)
