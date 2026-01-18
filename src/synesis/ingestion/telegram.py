"""Telegram channel listener using Telethon."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from telethon import TelegramClient, events
from telethon.tl.types import Channel, Message

from synesis.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TelegramMessage:
    """Normalized Telegram message."""

    message_id: int
    channel_id: int
    channel_name: str
    text: str
    timestamp: datetime
    raw: dict[str, Any]


class TelegramListener:
    """Listens to Telegram channels for financial news."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "synesis",
        channels: list[str] | None = None,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_name = session_name
        self._channels = channels or []
        self._client: TelegramClient | None = None
        self._running = False
        self._message_callback: Any = None

    async def start(self) -> None:
        """Start the Telegram client and connect."""
        self._client = TelegramClient(
            self._session_name,
            self._api_id,
            self._api_hash,
        )
        await self._client.start()
        self._running = True

        logger.info(
            "Telegram client started",
            session=self._session_name,
            channels=self._channels,
        )

        # Register event handler
        self._client.add_event_handler(
            self._handle_message,
            events.NewMessage(chats=self._channels if self._channels else None),
        )

    async def stop(self) -> None:
        """Stop the Telegram client."""
        self._running = False
        if self._client:
            await self._client.disconnect()
            logger.info("Telegram client stopped")

    def on_message(self, callback: Any) -> None:
        """Register a callback for new messages."""
        self._message_callback = callback

    async def _handle_message(self, event: events.NewMessage.Event) -> None:
        """Handle incoming messages from channels."""
        message: Message = event.message
        chat = await event.get_chat()

        # Only process channel messages
        if not isinstance(chat, Channel):
            return

        # Skip empty messages
        if not message.text:
            return

        telegram_msg = TelegramMessage(
            message_id=message.id,
            channel_id=chat.id,
            channel_name=chat.username or str(chat.id),
            text=message.text,
            timestamp=message.date.replace(tzinfo=UTC) if message.date else datetime.now(UTC),
            raw={
                "message_id": message.id,
                "channel_id": chat.id,
                "channel_title": chat.title,
                "channel_username": chat.username,
                "date": message.date.isoformat() if message.date else None,
                "views": message.views,
                "forwards": message.forwards,
            },
        )

        logger.debug(
            "Received Telegram message",
            channel=telegram_msg.channel_name,
            message_id=telegram_msg.message_id,
            text_preview=telegram_msg.text[:100] if telegram_msg.text else None,
        )

        # Call the registered callback
        if self._message_callback:
            try:
                await self._message_callback(telegram_msg)
            except Exception:
                logger.exception("Error in message callback")

    async def run_forever(self) -> None:
        """Run until disconnected."""
        if not self._client:
            raise RuntimeError("Client not started. Call start() first.")
        await self._client.run_until_disconnected()

    async def get_channel_info(self, channel: str) -> dict[str, Any] | None:
        """Get information about a channel."""
        if not self._client:
            return None

        try:
            entity = await self._client.get_entity(channel)
            if isinstance(entity, Channel):
                return {
                    "id": entity.id,
                    "title": entity.title,
                    "username": entity.username,
                    "participants_count": getattr(entity, "participants_count", None),
                }
        except Exception:
            logger.exception("Failed to get channel info", channel=channel)
        return None


async def create_telegram_listener(
    api_id: int,
    api_hash: str,
    session_name: str,
    channels: list[str],
) -> TelegramListener:
    """Create and start a Telegram listener."""
    listener = TelegramListener(
        api_id=api_id,
        api_hash=api_hash,
        session_name=session_name,
        channels=channels,
    )
    await listener.start()
    return listener
