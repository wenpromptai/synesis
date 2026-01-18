"""Event bus using Redis Streams for inter-service communication."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

import orjson

from redis.asyncio import Redis

from synesis.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    """Event types for the event bus."""

    # Ingestion events
    TELEGRAM_MESSAGE = "telegram.message"
    TWITTER_TWEET = "twitter.tweet"

    # Processing events
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_DEDUPLICATED = "message.deduplicated"
    MESSAGE_ANALYZED = "message.analyzed"

    # Market events
    MARKET_MATCHED = "market.matched"
    PRICE_UPDATE = "price.update"

    # Trading events
    SIGNAL_GENERATED = "signal.generated"
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"

    # Feedback events
    OUTCOME_RESOLVED = "outcome.resolved"


@dataclass
class Event:
    """An event in the system."""

    event_type: EventType
    payload: dict[str, Any]
    timestamp: datetime
    event_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert event to dictionary for Redis."""
        return {
            "event_type": self.event_type.value,
            "payload": orjson.dumps(self.payload).decode(),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[bytes, bytes], event_id: str) -> "Event":
        """Create event from Redis stream entry."""
        return cls(
            event_type=EventType(data[b"event_type"].decode()),
            payload=orjson.loads(data[b"payload"]),
            timestamp=datetime.fromisoformat(data[b"timestamp"].decode()),
            event_id=event_id,
        )


class EventBus:
    """Event bus using Redis Streams."""

    STREAM_KEY = "synesis:events"
    CONSUMER_GROUP = "synesis-workers"

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def ensure_stream(self) -> None:
        """Ensure the stream and consumer group exist."""
        try:
            await self._redis.xgroup_create(
                self.STREAM_KEY,
                self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info("Created consumer group", group=self.CONSUMER_GROUP)
        except Exception as e:
            # Group already exists
            if "BUSYGROUP" not in str(e):
                raise

    async def publish(self, event_type: EventType, payload: dict[str, Any]) -> str:
        """Publish an event to the stream."""
        event = Event(
            event_type=event_type,
            payload=payload,
            timestamp=datetime.now(UTC),
        )
        event_id = await self._redis.xadd(
            self.STREAM_KEY,
            cast(
                dict[
                    bytes | bytearray | memoryview | str | int | float,
                    bytes | bytearray | memoryview | str | int | float,
                ],
                event.to_dict(),
            ),
        )
        event_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id

        logger.debug(
            "Published event",
            event_type=event_type.value,
            event_id=event_id_str,
        )
        return event_id_str

    async def consume(
        self,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> list[Event]:
        """Consume events from the stream."""
        results = await self._redis.xreadgroup(
            groupname=self.CONSUMER_GROUP,
            consumername=consumer_name,
            streams={self.STREAM_KEY: ">"},
            count=count,
            block=block_ms,
        )

        events = []
        for _stream_name, messages in results:
            for msg_id, data in messages:
                msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                try:
                    event = Event.from_dict(data, msg_id_str)
                    events.append(event)
                except Exception:
                    logger.exception("Failed to parse event", event_id=msg_id_str)

        return events

    async def ack(self, event_id: str) -> None:
        """Acknowledge an event has been processed."""
        await self._redis.xack(self.STREAM_KEY, self.CONSUMER_GROUP, event_id)

    @asynccontextmanager
    async def subscribe(
        self,
        consumer_name: str,
        event_types: list[EventType] | None = None,
    ) -> AsyncIterator["EventSubscription"]:
        """Subscribe to events with automatic acknowledgment."""
        await self.ensure_stream()
        yield EventSubscription(self, consumer_name, event_types)


class EventSubscription:
    """An active subscription to the event bus."""

    def __init__(
        self,
        bus: EventBus,
        consumer_name: str,
        event_types: list[EventType] | None,
    ) -> None:
        self._bus = bus
        self._consumer_name = consumer_name
        self._event_types = set(event_types) if event_types else None

    async def __aiter__(self) -> AsyncIterator[Event]:
        """Iterate over events."""
        while True:
            events = await self._bus.consume(self._consumer_name)
            for event in events:
                if self._event_types is None or event.event_type in self._event_types:
                    yield event
                    if event.event_id:
                        await self._bus.ack(event.event_id)
