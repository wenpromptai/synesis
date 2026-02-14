"""Application configuration via pydantic-settings."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from synesis.core.constants import (
    DEFAULT_FINNHUB_API_URL,
    DEFAULT_FINNHUB_WS_URL,
    DEFAULT_KALSHI_API_URL,
    DEFAULT_KALSHI_WS_URL,
    DEFAULT_POLYMARKET_CLOB_WS_URL,
    DEFAULT_POLYMARKET_DATA_API_URL,
    DEFAULT_POLYMARKET_GAMMA_API_URL,
    DEFAULT_SIGNALS_OUTPUT_DIR,
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

    # Provider Selection (allows swapping data providers)
    ticker_provider: Literal["factset", "finnhub"] = Field(
        default="factset",
        description="Ticker validation provider (factset or finnhub)",
    )
    fundamentals_provider: Literal["factset", "finnhub", "sec_edgar", "none"] = Field(
        default="sec_edgar",
        description="Fundamentals data provider (factset, finnhub, sec_edgar, none)",
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

    # NASDAQ (free, no key required)
    nasdaq_cache_ttl_earnings: int = Field(
        default=21600,
        description="Cache TTL for NASDAQ earnings calendar per date (seconds)",
    )
    nasdaq_earnings_lookahead_days: int = Field(
        default=14,
        description="Number of days to look ahead for upcoming earnings",
    )

    # Crawl4AI (web crawling)
    crawl4ai_url: str = Field(
        default="http://localhost:11235",
        description="Crawl4AI service URL",
    )
    # FactSet SQL Server
    sqlserver_host: str = Field(
        default="",
        description="FactSet SQL Server host",
    )
    sqlserver_port: int = Field(
        default=1433,
        description="FactSet SQL Server port",
    )
    sqlserver_database: str = Field(
        default="",
        description="FactSet SQL Server database name",
    )
    sqlserver_user: str = Field(
        default="",
        description="FactSet SQL Server username",
    )
    sqlserver_password: SecretStr | None = Field(
        default=None,
        description="FactSet SQL Server password",
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
        return [s.removeprefix("/r/").removeprefix("r/") for s in v]

    # Kalshi
    kalshi_api_key: SecretStr | None = Field(default=None)
    kalshi_private_key_path: str | None = Field(default=None)
    kalshi_api_url: str = Field(
        default=DEFAULT_KALSHI_API_URL,
        description="Kalshi REST API base URL",
    )
    kalshi_ws_url: str = Field(
        default=DEFAULT_KALSHI_WS_URL,
        description="Kalshi WebSocket URL",
    )

    # Polymarket Data API + WebSocket
    polymarket_data_api_url: str = Field(
        default=DEFAULT_POLYMARKET_DATA_API_URL,
        description="Polymarket Data API base URL",
    )
    polymarket_clob_ws_url: str = Field(
        default=DEFAULT_POLYMARKET_CLOB_WS_URL,
        description="Polymarket CLOB WebSocket URL",
    )

    # Market Intelligence (Flow 3: mkt_intel)
    mkt_intel_enabled: bool = Field(default=True)
    mkt_intel_interval: int = Field(
        default=3600,
        gt=0,
        description="Market intelligence scan interval in seconds (1 hour)",
    )
    mkt_intel_volume_spike_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Volume spike detection threshold (1.0 = 100% increase from previous hour)",
    )
    mkt_intel_insider_score_min: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum insider score to flag wallet activity",
    )
    mkt_intel_expiring_hours: int = Field(
        default=24,
        gt=0,
        description="Hours before expiration to flag markets",
    )
    mkt_intel_ws_enabled: bool = Field(
        default=True,
        description="Enable real-time WebSocket streams for market data",
    )
    mkt_intel_auto_watch_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Insider score threshold for auto-watching discovered wallets",
    )
    mkt_intel_unwatch_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Insider score below which watched wallets are demoted",
    )

    # Watchlist Intelligence (Flow 4)
    watchlist_intel_enabled: bool = Field(default=False)
    watchlist_intel_interval: int = Field(
        default=21600,
        gt=0,
        description="Watchlist analysis interval in seconds (default 6h)",
    )
    watchlist_intel_earnings_alert_days: int = Field(
        default=7,
        gt=0,
        description="Alert if earnings within N days",
    )
    watchlist_intel_max_tickers: int = Field(
        default=30,
        gt=0,
        description="Max tickers to analyze per cycle",
    )

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
        default=3,
        description="Max web search queries per message (pre-fetch current context)",
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
    signals_output_dir: Path = Field(
        default=Path(DEFAULT_SIGNALS_OUTPUT_DIR),
        description="Directory for signal output files",
    )

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
