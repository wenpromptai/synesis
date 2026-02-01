"""Application configuration via pydantic-settings."""

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core
    env: Literal["development", "staging", "production"] = Field(
        default="development", alias="SYNESIS_ENV"
    )
    debug: bool = Field(default=False, alias="SYNESIS_DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="SYNESIS_LOG_LEVEL"
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://synesis:synesis_dev@localhost:5435/synesis"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Telegram (ingestion)
    telegram_api_id: int | None = Field(default=None)
    telegram_api_hash: SecretStr | None = Field(default=None)
    telegram_session_name: str = Field(default="shared/sessions/synesis")
    telegram_channels: list[str] = Field(default_factory=list)

    # Telegram (notifications)
    telegram_bot_token: SecretStr | None = Field(
        default=None,
        description="Telegram bot token for sending notifications",
    )
    telegram_chat_id: str | None = Field(
        default=None,
        description="Telegram chat ID to send notifications to",
    )

    @field_validator("telegram_channels", mode="before")
    @classmethod
    def parse_telegram_channels(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                v = json.loads(v)
            else:
                v = [c.strip() for c in v.split(",") if c.strip()]
        return [c.lstrip("@") for c in v]

    # Twitter (twitterapi.io)
    twitterapi_api_key: SecretStr | None = Field(default=None)
    twitter_api_base_url: str = Field(default="https://api.twitterapi.io")
    twitter_accounts: list[str] = Field(default_factory=list)

    # Twitter source categorization for Flow 1
    # News accounts = high urgency (breaking news, act fast)
    twitter_news_accounts: list[str] = Field(
        default=["DeItaone", "realDonaldTrump"],
        description="Twitter accounts treated as breaking news sources (high urgency)",
    )
    # Analysis accounts = normal urgency (insights, consider)
    twitter_analysis_accounts: list[str] = Field(
        default=["elonmusk", "NickTimiraos", "charliebilello", "KobeissiLetter"],
        description="Twitter accounts treated as analysis sources (normal urgency)",
    )

    @field_validator("twitter_accounts", mode="before")
    @classmethod
    def parse_twitter_accounts(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                v = json.loads(v)
            else:
                v = [a.strip() for a in v.split(",") if a.strip()]
        return [a.lstrip("@") for a in v]

    @field_validator("twitter_news_accounts", "twitter_analysis_accounts", mode="before")
    @classmethod
    def parse_twitter_categorized_accounts(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                v = json.loads(v)
            else:
                v = [a.strip() for a in v.split(",") if a.strip()]
        return [a.lstrip("@") for a in v]

    def get_twitter_source_type(self, username: str) -> str:
        """Get source type (news/analysis) for a Twitter username."""
        username_clean = username.lstrip("@").lower()
        if username_clean in [a.lower() for a in self.twitter_news_accounts]:
            return "news"
        return "analysis"

    def get_telegram_source_type(self, channel: str) -> str:
        """Get source type for a Telegram channel. All Telegram channels are news by default."""
        # All configured Telegram channels are treated as breaking news
        return "news"

    # Polymarket
    polymarket_api_key: SecretStr | None = Field(default=None)
    polymarket_api_secret: SecretStr | None = Field(default=None)
    polymarket_private_key: SecretStr | None = Field(default=None)
    polymarket_chain_id: int = Field(default=137)

    # LLM Provider
    llm_provider: Literal["anthropic", "openai"] = Field(default="anthropic")

    # API Keys
    anthropic_api_key: SecretStr | None = Field(default=None)
    openai_api_key: SecretStr | None = Field(default=None)

    # OpenAI-compatible base URL (for ZAI, use https://api.z.ai/api/coding/paas/v4)
    openai_base_url: str | None = Field(default=None)

    # Model names
    llm_model: str = Field(default="claude-3-5-haiku-20241022")
    llm_model_smart: str = Field(default="claude-sonnet-4-20250514")

    # Web Search APIs (for LLM tool use)
    # SearXNG (self-hosted, primary search - no API key, no rate limits)
    searxng_url: str | None = Field(default="http://localhost:8080")
    # External APIs (fallbacks)
    exa_api_key: SecretStr | None = Field(default=None)
    brave_api_key: SecretStr | None = Field(default=None)

    # Stock Price Data (Finnhub)
    finnhub_api_key: SecretStr | None = Field(
        default=None,
        description="Finnhub API key for real-time stock prices",
    )

    @field_validator("searxng_url")
    @classmethod
    def validate_searxng_url(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate SearXNG URL to prevent SSRF in production."""
        if v is None:
            return v
        # Get env from the values being validated
        env = info.data.get("env", "development")
        if env == "production":
            # Block private IPs in production
            private_patterns = [
                # IPv4
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "192.168.",
                "10.",
                "172.16.",
                "172.17.",
                "172.18.",
                "172.19.",
                "172.20.",
                "172.21.",
                "172.22.",
                "172.23.",
                "172.24.",
                "172.25.",
                "172.26.",
                "172.27.",
                "172.28.",
                "172.29.",
                "172.30.",
                "172.31.",
                # IPv6
                "::1",
                "[::1]",
                "fc00:",
                "[fc00:",
                "fe80:",
                "[fe80:",
            ]
            if any(pattern in v.lower() for pattern in private_patterns):
                raise ValueError("SearXNG URL cannot point to internal network in production")
        return v

    # Reddit RSS (Flow 2 - Sentiment Intelligence)
    reddit_subreddits: list[str] = Field(
        default=[
            # Degen/Options (high retail sentiment, meme stocks)
            "wallstreetbets",
            "options",
            "smallstreetbets",
            "thetagang",
            # Active trading
            "Daytrading",
            "pennystocks",
            # Mainstream investing
            "stocks",
            "StockMarket",
        ],
        description="Subreddits to monitor for sentiment",
    )
    reddit_poll_interval: int = Field(
        default=21600,  # 6 hours in seconds
        description="Reddit RSS poll interval in seconds",
    )

    @field_validator("reddit_subreddits", mode="before")
    @classmethod
    def parse_reddit_subreddits(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                v = json.loads(v)
            else:
                v = [s.strip() for s in v.split(",") if s.strip()]
        return [s.lstrip("r/").lstrip("/r/") for s in v]

    # Trading
    trading_enabled: bool = Field(default=False)
    max_position_size: float = Field(default=100.0)
    min_edge_threshold: float = Field(default=0.05)
    confidence_threshold: float = Field(default=0.7)

    # Processing concurrency
    processing_workers: int = Field(
        default=5,
        description="Number of concurrent message processor workers",
    )
    processing_queue_size: int = Field(
        default=100,
        description="Max size of internal processing queue",
    )
    web_search_max_queries: int = Field(
        default=2,
        description="Max web search queries per message",
    )
    polymarket_max_keywords: int = Field(
        default=5,
        description="Max Polymarket keywords to search per message",
    )

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
