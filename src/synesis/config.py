"""Application configuration via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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

    # Telegram
    telegram_api_id: int | None = Field(default=None)
    telegram_api_hash: SecretStr | None = Field(default=None)
    telegram_session_name: str = Field(default="synesis")
    telegram_channels: list[str] = Field(default_factory=list)

    @field_validator("telegram_channels", mode="before")
    @classmethod
    def parse_telegram_channels(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    # Twitter
    twitter_api_key: SecretStr | None = Field(default=None)
    twitter_api_base_url: str = Field(default="https://api.twitterapi.io")
    twitter_accounts: list[str] = Field(default_factory=list)

    @field_validator("twitter_accounts", mode="before")
    @classmethod
    def parse_twitter_accounts(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v

    # Polymarket
    polymarket_api_key: SecretStr | None = Field(default=None)
    polymarket_api_secret: SecretStr | None = Field(default=None)
    polymarket_private_key: SecretStr | None = Field(default=None)
    polymarket_chain_id: int = Field(default=137)

    # LLM
    anthropic_api_key: SecretStr | None = Field(default=None)
    openai_api_key: SecretStr | None = Field(default=None)
    llm_model: str = Field(default="claude-3-5-haiku-20241022")

    # Trading
    trading_enabled: bool = Field(default=False)
    max_position_size: float = Field(default=100.0)
    min_edge_threshold: float = Field(default=0.05)
    confidence_threshold: float = Field(default=0.7)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
