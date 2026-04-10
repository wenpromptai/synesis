"""Application configuration via pydantic-settings."""

import ipaddress
import json
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from synesis.config_cache import CacheTTLSettings
from synesis.core.constants import (
    DEFAULT_FINNHUB_API_URL,
    DEFAULT_FINNHUB_WS_URL,
    DEFAULT_POLYMARKET_GAMMA_API_URL,
)


class Settings(CacheTTLSettings, BaseSettings):
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

    # Notification channel selection
    notification_channel: Literal["telegram", "discord"] = Field(
        default="telegram",
        description="Notification output channel: 'telegram' or 'discord'",
    )
    stage2_enabled: bool = Field(
        default=True,
        description="Enable Stage 2 processing (LLM analysis, market matching, notification)",
    )
    macro_strategist_enabled: bool = Field(
        default=True,
        description="Enable MacroStrategist in intelligence pipeline (FRED regime assessment + sector tilts)",
    )
    debate_rounds: int = Field(
        default=1,
        ge=0,
        description="Bull/bear debate rounds per ticker (0=parallel/no debate, 1+=sequential rounds)",
    )
    trader_mode: Literal["per_ticker", "portfolio"] = Field(
        default="portfolio",
        description="Trader evaluation mode: 'per_ticker' or 'portfolio'",
    )
    kg_briefs_dir: str = Field(
        default="docs/kg/raw/synesis_briefs",
        description="Directory for saving intelligence brief markdown files (relative to cwd or absolute)",
    )
    # Discord (notifications)
    discord_webhook_url: SecretStr | None = Field(
        default=None,
        description="Discord webhook URL for Stage 1 notifications",
    )
    discord2_webhook_url: SecretStr | None = Field(
        default=None,
        description="Discord webhook URL for Stage 2 notifications (falls back to discord_webhook_url)",
    )
    discord_twitter_webhook_url: SecretStr | None = Field(
        default=None,
        description="Discord webhook URL for daily Twitter agent digest",
    )
    discord_events_webhook_url: SecretStr | None = Field(
        default=None,
        description="Discord webhook URL for Event Radar daily digest",
    )

    @field_validator("telegram_channels", mode="before")
    @classmethod
    def parse_telegram_channels(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                parsed: list[str] = json.loads(v)
                v = parsed
            else:
                v = [c.strip() for c in v.split(",") if c.strip()]
        return [c.lstrip("@") for c in v]

    # RSS ingestion (Google News)
    rss_enabled: bool = Field(
        default=False,
        description="Enable Google News RSS feed polling",
    )
    rss_poll_interval_minutes: int = Field(
        default=1,
        description="Minutes between RSS poll cycles",
    )
    rss_feeds: list[str] = Field(
        default_factory=lambda: [
            # AI mega-deals: M&A, investments, big orders ($M/$B)
            'https://news.google.com/rss/search?q="AI"+OR+"artificial+intelligence"+"billion"+OR+"million"+acquisition+OR+order+OR+investment+OR+deal+when:24h&hl=en-US&gl=US&ceid=US:en',
            # AI infrastructure supply chain: data centers, semis, GPUs, optics
            'https://news.google.com/rss/search?q="data+center"+OR+semiconductor+OR+GPU+OR+transceiver+OR+"optical"+order+OR+deal+OR+contract+OR+investment+when:24h&hl=en-US&gl=US&ceid=US:en',
        ],
        description="Google News RSS feed URLs to poll (topic or search feeds)",
    )

    @field_validator("rss_feeds", mode="before")
    @classmethod
    def parse_rss_feeds(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                parsed: list[str] = json.loads(v)
                v = parsed
            else:
                v = [u.strip() for u in v.split(",") if u.strip()]
        return list(v)

    # Twitter (twitterapi.io) — ingestion client credentials (not active in Flow 1)
    twitterapi_api_key: SecretStr | None = Field(default=None)
    twitter_api_base_url: str = Field(default="https://api.twitterapi.io")
    twitter_accounts: list[str] = Field(default_factory=list)

    @field_validator("twitter_accounts", mode="before")
    @classmethod
    def parse_twitter_accounts(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                parsed: list[str] = json.loads(v)
                v = parsed
            else:
                v = [a.strip() for a in v.split(",") if a.strip()]
        return [a.lstrip("@") for a in v]

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
    llm_model: str = Field(default="gpt-5-nano")
    llm_model_smart: str = Field(default="gpt-4o-mini")
    llm_model_vsmart: str = Field(
        default="gpt-5.2",
        description="Most capable model for complex tasks (event synthesis, Twitter agent)",
    )

    # Web Search APIs (for LLM tool use)
    # Brave (primary, 2000 req/month free tier). SearXNG for ticker searches only.
    searxng_url: str | None = Field(default="http://localhost:8080")
    brave_api_key: SecretStr | None = Field(default=None)

    # Stock Price Data (Finnhub)
    finnhub_api_key: SecretStr | None = Field(
        default=None,
        description="Finnhub API key for real-time stock prices",
    )

    # FRED (Federal Reserve Economic Data)
    fred_api_key: SecretStr | None = Field(
        default=None,
        description="FRED API key (free, register at https://fredaccount.stlouisfed.org)",
    )

    # Massive (market data — free tier: 5 calls/min)
    massive_api_key: SecretStr | None = Field(
        default=None,
        description="Massive.com API key (free at https://massive.com)",
    )
    massive_api_url: str = Field(
        default="https://api.massive.com",
        description="Massive.com REST API base URL",
    )
    # SEC EDGAR (free, no key required)
    sec_edgar_user_agent: str = Field(
        default="Synesis synesis@example.com",
        description="User-Agent header for SEC EDGAR API (required by SEC)",
    )

    # Crawl4AI (web crawling)
    crawl4ai_url: str = Field(
        default="http://localhost:11235",
        description="Crawl4AI service URL",
    )

    @field_validator("searxng_url")
    @classmethod
    def validate_searxng_url(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate SearXNG URL to prevent SSRF in production."""
        if v is None:
            return v
        env = info.data.get("env", "development")
        if env == "production":
            host = urlparse(v).hostname or ""
            if host in ("localhost", "0.0.0.0"):
                raise ValueError("SearXNG URL cannot point to internal network in production")
            try:
                addr = ipaddress.ip_address(host)
                if addr.is_private or addr.is_loopback:
                    raise ValueError("SearXNG URL cannot point to internal network in production")
            except ValueError as e:
                if "internal network" in str(e):
                    raise
        return v

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
    brave_min_interval: float = Field(
        default=1.5,
        description=(
            "Minimum seconds between Brave API calls. Shared across all processors "
            "via a module-level rate limiter in web_search.py (Brave limit: 1 req/s)."
        ),
    )
    polymarket_max_keywords: int = Field(
        default=5,
        description="Max Polymarket keywords to search per message",
    )

    # API URLs (environment-configurable)
    polymarket_gamma_api_url: str = Field(
        default=DEFAULT_POLYMARKET_GAMMA_API_URL,
        description="Polymarket Gamma API base URL",
    )
    finnhub_ws_url: str = Field(
        default=DEFAULT_FINNHUB_WS_URL,
        description="Finnhub WebSocket URL",
    )
    finnhub_api_url: str = Field(
        default=DEFAULT_FINNHUB_API_URL,
        description="Finnhub REST API base URL",
    )

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
