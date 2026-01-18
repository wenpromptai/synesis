"""Custom exceptions for Synesis."""


class SynesisError(Exception):
    """Base exception for all Synesis errors."""

    def __init__(self, message: str, *args: object) -> None:
        self.message = message
        super().__init__(message, *args)


# Ingestion errors
class IngestionError(SynesisError):
    """Base error for ingestion layer."""


class TelegramConnectionError(IngestionError):
    """Failed to connect to Telegram."""


class TelegramAuthError(IngestionError):
    """Telegram authentication failed."""


class TwitterConnectionError(IngestionError):
    """Failed to connect to Twitter API."""


class TwitterRateLimitError(IngestionError):
    """Twitter API rate limit exceeded."""


# Processing errors
class ProcessingError(SynesisError):
    """Base error for processing layer."""


class LLMError(ProcessingError):
    """LLM API call failed."""


class ClassificationError(ProcessingError):
    """Failed to classify message."""


# Market errors
class MarketError(SynesisError):
    """Base error for market layer."""


class PolymarketConnectionError(MarketError):
    """Failed to connect to Polymarket."""


class MarketNotFoundError(MarketError):
    """No matching market found."""


class OrderError(MarketError):
    """Order execution failed."""


# Trading errors
class TradingError(SynesisError):
    """Base error for trading layer."""


class TradingDisabledError(TradingError):
    """Trading is disabled but trade was attempted."""


class InsufficientFundsError(TradingError):
    """Insufficient funds for trade."""


class RiskLimitError(TradingError):
    """Trade would exceed risk limits."""


# Storage errors
class StorageError(SynesisError):
    """Base error for storage layer."""


class DatabaseConnectionError(StorageError):
    """Failed to connect to database."""


class RedisConnectionError(StorageError):
    """Failed to connect to Redis."""
