"""Shared fixtures for integration tests.

These fixtures provide stateful mock implementations of Redis and PostgreSQL
that maintain internal state for verification, while allowing real API calls
(LLM, Finnhub, Polymarket, Telegram) to proceed.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async tests."""
    return "asyncio"


@pytest.fixture
def mock_redis() -> Any:
    """Create mock Redis with stateful behavior for verification.

    This mock maintains internal state so tests can verify what was written.
    All operations are in-memory for easy inspection.

    Internal state attributes:
        _test_sets: Dict of set key -> set of values
        _test_strings: Dict of string key -> value
        _test_hashes: Dict of hash key -> dict of field -> value
        _test_pubsub: List of (channel, message) tuples published
    """
    redis = AsyncMock()

    # Internal state for verification (accessed via redis._test_* in tests)
    _test_sets: dict[str, set[str]] = {}
    _test_strings: dict[str, str] = {}
    _test_hashes: dict[str, dict[str, str]] = {}
    _test_pubsub: list[tuple[str, str]] = []

    # Expose state for test assertions
    redis._test_sets = _test_sets
    redis._test_strings = _test_strings
    redis._test_hashes = _test_hashes
    redis._test_pubsub = _test_pubsub

    async def mock_sadd(key: str, *members: str) -> int:
        if key not in _test_sets:
            _test_sets[key] = set()
        _test_sets[key].update(members)
        return len(members)

    async def mock_smembers(key: str) -> set[str]:
        return _test_sets.get(key, set())

    async def mock_sismember(key: str, member: str) -> bool:
        return member in _test_sets.get(key, set())

    async def mock_srem(key: str, *members: str) -> int:
        if key in _test_sets:
            before = len(_test_sets[key])
            _test_sets[key] -= set(members)
            return before - len(_test_sets[key])
        return 0

    async def mock_set(key: str, value: str, ex: int | None = None, nx: bool = False) -> bool:
        # nx=True means "only set if key does not exist"
        if nx and key in _test_strings:
            return False
        _test_strings[key] = value
        return True

    async def mock_get(key: str) -> str | None:
        return _test_strings.get(key)

    async def mock_exists(*keys: str) -> int:
        count = 0
        for key in keys:
            if key in _test_strings or key in _test_sets or key in _test_hashes:
                count += 1
        return count

    async def mock_delete(*keys: str) -> int:
        count = 0
        for key in keys:
            if key in _test_strings:
                del _test_strings[key]
                count += 1
            if key in _test_sets:
                del _test_sets[key]
                count += 1
            if key in _test_hashes:
                del _test_hashes[key]
                count += 1
        return count

    async def mock_hset(key: str, mapping: dict[str, str] | None = None, **kwargs: str) -> int:
        if key not in _test_hashes:
            _test_hashes[key] = {}
        data = mapping or kwargs
        _test_hashes[key].update(data)
        return len(data)

    async def mock_hgetall(key: str) -> dict[str, str]:
        return _test_hashes.get(key, {})

    async def mock_hincrby(key: str, field: str, increment: int = 1) -> int:
        if key not in _test_hashes:
            _test_hashes[key] = {}
        current = int(_test_hashes[key].get(field, "0"))
        new_val = current + increment
        _test_hashes[key][field] = str(new_val)
        return new_val

    async def mock_publish(channel: str, message: str) -> int:
        _test_pubsub.append((channel, message))
        return 1

    async def mock_expire(key: str, seconds: int) -> bool:
        return True

    async def mock_scan_iter(
        match: str = "*",
        count: int = 100,  # noqa: ARG001
    ) -> Any:
        """Async generator that yields keys matching a pattern.

        This is used by deduplication to find embedding keys.
        """
        import fnmatch

        # Collect all keys from all stores
        all_keys: set[str] = set()
        all_keys.update(_test_strings.keys())
        all_keys.update(_test_sets.keys())
        all_keys.update(_test_hashes.keys())

        # Match against pattern (Redis uses glob-style matching)
        for key in sorted(all_keys):
            if fnmatch.fnmatch(key, match):
                yield key

    # Assign mock methods
    redis.sadd = mock_sadd
    redis.smembers = mock_smembers
    redis.sismember = mock_sismember
    redis.srem = mock_srem
    redis.set = mock_set
    redis.get = mock_get
    redis.exists = mock_exists
    redis.delete = mock_delete
    redis.hset = mock_hset
    redis.hgetall = mock_hgetall
    redis.hincrby = mock_hincrby
    redis.publish = mock_publish
    redis.expire = mock_expire
    redis.scan_iter = mock_scan_iter
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()

    return redis


@pytest.fixture
def mock_db() -> Any:
    """Create mock Database with stateful behavior for verification.

    Captures all database operations for test assertions.

    Internal state attributes:
        _test_raw_messages: List of inserted raw messages
        _test_signals: List of inserted signals (Flow 1 and Flow 2)
        _test_predictions: List of inserted predictions (market evaluations)
        _test_watchlist: Dict of ticker -> watchlist record
        _test_sentiment_snapshots: List of sentiment snapshot records
    """
    db = AsyncMock()

    # Internal state for verification (accessed via db._test_* in tests)
    _test_raw_messages: list[dict[str, Any]] = []
    _test_signals: list[dict[str, Any]] = []
    _test_predictions: list[dict[str, Any]] = []
    _test_watchlist: dict[str, dict[str, Any]] = {}
    _test_sentiment_snapshots: list[dict[str, Any]] = []

    # Expose state for test assertions
    db._test_raw_messages = _test_raw_messages
    db._test_signals = _test_signals
    db._test_predictions = _test_predictions
    db._test_watchlist = _test_watchlist
    db._test_sentiment_snapshots = _test_sentiment_snapshots

    async def mock_insert_raw_message(
        message: Any,
        embedding: Any = None,
        is_duplicate: bool = False,
        duplicate_of: str | None = None,
    ) -> str:
        _test_raw_messages.append(
            {
                "message": message,
                "embedding": embedding,
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of,
                "inserted_at": datetime.now(timezone.utc),
            }
        )
        return f"test-uuid-{len(_test_raw_messages)}"

    async def mock_insert_signal(signal: Any, prices: dict[str, Any] | None = None) -> None:
        _test_signals.append(
            {
                "signal": signal,
                "prices": prices,
                "inserted_at": datetime.now(timezone.utc),
            }
        )

    async def mock_insert_prediction(evaluation: Any, timestamp: datetime) -> None:
        _test_predictions.append(
            {
                "evaluation": evaluation,
                "timestamp": timestamp,
                "inserted_at": datetime.now(timezone.utc),
            }
        )

    async def mock_upsert_watchlist_ticker(
        ticker: str,
        company_name: str | None = None,
        added_by: str | None = None,
        added_reason: str | None = None,
        expires_at: datetime | None = None,
        ttl_days: int = 7,
    ) -> None:
        _test_watchlist[ticker] = {
            "ticker": ticker,
            "company_name": company_name,
            "added_by": added_by,
            "added_reason": added_reason,
            "expires_at": expires_at,
            "ttl_days": ttl_days,
            "upserted_at": datetime.now(timezone.utc),
        }

    async def mock_insert_sentiment_snapshot(
        ticker: str,
        snapshot_time: datetime,
        **kwargs: Any,
    ) -> None:
        _test_sentiment_snapshots.append(
            {
                "ticker": ticker,
                "snapshot_time": snapshot_time,
                **kwargs,
                "inserted_at": datetime.now(timezone.utc),
            }
        )

    async def mock_insert_sentiment_signal(signal: Any) -> None:
        _test_signals.append(
            {
                "signal": signal,
                "flow_id": "sentiment",
                "inserted_at": datetime.now(timezone.utc),
            }
        )

    # Assign mock methods
    db.insert_raw_message = mock_insert_raw_message
    db.insert_signal = mock_insert_signal
    db.insert_prediction = mock_insert_prediction
    db.upsert_watchlist_ticker = mock_upsert_watchlist_ticker
    db.insert_sentiment_snapshot = mock_insert_sentiment_snapshot
    db.insert_sentiment_signal = mock_insert_sentiment_signal
    db.get_active_watchlist = AsyncMock(return_value=[])
    db.get_active_watchlist_with_metadata = AsyncMock(return_value=[])
    db.deactivate_expired_watchlist = AsyncMock(return_value=[])

    return db


@pytest.fixture
def test_settings() -> Any:
    """Create test settings with real API keys from environment.

    This loads settings from environment variables / .env file,
    which should include real API keys for:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY (LLM)
    - FINNHUB_API_KEY (ticker validation)
    - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (notifications)
    """
    from synesis.config import get_settings

    return get_settings()


@pytest.fixture
def finnhub_service(mock_redis: Any) -> Any:
    """Create real Finnhub service for ticker validation.

    Returns None if FINNHUB_API_KEY is not set.
    Uses mock_redis for caching (Finnhub API uses Redis caching).
    """
    from synesis.config import get_settings

    settings = get_settings()
    if not settings.finnhub_api_key:
        return None

    from synesis.ingestion.finnhub import FinnhubService

    api_key = settings.finnhub_api_key.get_secret_value()
    return FinnhubService(api_key=api_key, redis=mock_redis)
