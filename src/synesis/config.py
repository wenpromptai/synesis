"""Application configuration via pydantic-settings."""

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from synesis.core.constants import (
    DEFAULT_FINNHUB_API_URL,
    DEFAULT_FINNHUB_WS_URL,
    DEFAULT_POLYMARKET_GAMMA_API_URL,
)


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

    # Notification channel selection
    notification_channel: Literal["telegram", "discord"] = Field(
        default="telegram",
        description="Notification output channel: 'telegram' or 'discord'",
    )
    stage2_enabled: bool = Field(
        default=True,
        description="Enable Stage 2 processing (LLM analysis, market matching, notification)",
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
                v = json.loads(v)
            else:
                v = [c.strip() for c in v.split(",") if c.strip()]
        return [c.lstrip("@") for c in v]

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
                v = json.loads(v)
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
    llm_model: str = Field(default="claude-3-5-haiku-20241022")
    llm_model_smart: str = Field(default="claude-sonnet-4-20250514")
    llm_model_vsmart: str = Field(
        default="claude-sonnet-4-20250514",
        description="Most capable model for complex tasks (event synthesis, Twitter agent)",
    )

    # Web Search APIs (for LLM tool use)
    # Priority: Brave (primary, 2000 req/month) → Exa keys (fallback). SearXNG for ticker searches only.
    searxng_url: str | None = Field(default="http://localhost:8080")
    exa_api_key: SecretStr | None = Field(default=None)
    exa_wenprompt_api_key: SecretStr | None = Field(default=None, alias="EXA_WENPROMPT_API_KEY")
    exa_wenpromptai_api_key: SecretStr | None = Field(default=None, alias="EXA_WENPROMPTAI_API_KEY")
    exa_wangwhpt_api_key: SecretStr | None = Field(default=None, alias="EXA_WANGWHPT_API_KEY")
    brave_api_key: SecretStr | None = Field(default=None)

    @property
    def exa_api_keys(self) -> list[str]:
        """All configured Exa API keys, deduplicated and in order."""
        keys = []
        seen: set[str] = set()
        for field in [
            self.exa_api_key,
            self.exa_wenprompt_api_key,
            self.exa_wenpromptai_api_key,
            self.exa_wangwhpt_api_key,
        ]:
            if field:
                v = field.get_secret_value()
                if v not in seen:
                    seen.add(v)
                    keys.append(v)
        return keys

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
    fred_cache_ttl_search: int = Field(
        default=3600,
        description="Cache TTL for FRED series search results (seconds)",
    )
    fred_cache_ttl_series: int = Field(
        default=43200,
        description="Cache TTL for FRED series metadata (seconds)",
    )
    fred_cache_ttl_observations: int = Field(
        default=21600,
        description="Cache TTL for FRED observations (seconds)",
    )
    fred_cache_ttl_releases: int = Field(
        default=43200,
        description="Cache TTL for FRED releases (seconds)",
    )
    fred_cache_ttl_release_dates: int = Field(
        default=21600,
        description="Cache TTL for FRED release dates (seconds)",
    )

    # SEC EDGAR (free, no key required)
    sec_edgar_user_agent: str = Field(
        default="Synesis synesis@example.com",
        description="User-Agent header for SEC EDGAR API (required by SEC)",
    )
    sec_edgar_cache_ttl_submissions: int = Field(
        default=3600,
        description="Cache TTL for SEC EDGAR submissions (seconds)",
    )
    sec_edgar_cache_ttl_cik_map: int = Field(
        default=86400,
        description="Cache TTL for SEC ticker→CIK mapping (seconds)",
    )
    sec_edgar_cache_ttl_company_facts: int = Field(
        default=21600,
        description="Cache TTL for SEC XBRL company facts (seconds, 6h)",
    )
    sec_edgar_cache_ttl_xbrl_frames: int = Field(
        default=86400,
        description="Cache TTL for SEC XBRL frames (seconds, 24h)",
    )
    sec_edgar_cache_ttl_filing_content: int = Field(
        default=604800,
        description="Cache TTL for SEC filing content (seconds, 7d — filings don't change)",
    )

    # NASDAQ (free, no key required)
    nasdaq_cache_ttl_earnings: int = Field(
        default=21600,
        description="Cache TTL for NASDAQ earnings calendar per date (seconds)",
    )
    nasdaq_earnings_lookahead_days: int = Field(
        default=14,
        description="Number of days to look ahead for upcoming earnings",
    )

    # yfinance (free, no key required)
    yfinance_cache_ttl_quote: int = Field(
        default=60,
        description="Cache TTL for yfinance quote snapshots (seconds)",
    )
    yfinance_cache_ttl_history: int = Field(
        default=300,
        description="Cache TTL for yfinance OHLCV history (seconds)",
    )
    yfinance_cache_ttl_options: int = Field(
        default=120,
        description="Cache TTL for yfinance options data (seconds)",
    )
    yfinance_cache_ttl_fx: int = Field(
        default=30,
        description="Cache TTL for yfinance FX rates (seconds)",
    )
    yfinance_cache_ttl_movers: int = Field(
        default=300,
        description="Cache TTL for yfinance market movers screener (seconds)",
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
