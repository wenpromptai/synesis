"""Unit tests for Settings validators and env-var loading in config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from synesis.config import Settings


# ─────────────────────────────────────────────────────────────
# Validator tests (existing)
# ─────────────────────────────────────────────────────────────


class TestParseTelegramChannels:
    """Tests for parse_telegram_channels validator."""

    def test_none_returns_empty_list(self) -> None:
        result = Settings.parse_telegram_channels(None)
        assert result == []

    def test_csv_string(self) -> None:
        result = Settings.parse_telegram_channels("chan1, chan2, chan3")
        assert result == ["chan1", "chan2", "chan3"]

    def test_json_array_string(self) -> None:
        result = Settings.parse_telegram_channels('["chan1", "chan2"]')
        assert result == ["chan1", "chan2"]

    def test_strips_at_prefix(self) -> None:
        result = Settings.parse_telegram_channels("@chan1, @chan2")
        assert result == ["chan1", "chan2"]

    def test_list_passthrough(self) -> None:
        result = Settings.parse_telegram_channels(["a", "b"])
        assert result == ["a", "b"]


class TestParseTwitterAccounts:
    """Tests for parse_twitter_accounts validator."""

    def test_none_returns_empty_list(self) -> None:
        result = Settings.parse_twitter_accounts(None)
        assert result == []

    def test_csv_string(self) -> None:
        result = Settings.parse_twitter_accounts("user1, user2")
        assert result == ["user1", "user2"]

    def test_json_array_string(self) -> None:
        result = Settings.parse_twitter_accounts('["u1", "u2"]')
        assert result == ["u1", "u2"]

    def test_strips_at_prefix(self) -> None:
        result = Settings.parse_twitter_accounts("@alice, @bob")
        assert result == ["alice", "bob"]

    def test_list_passthrough(self) -> None:
        result = Settings.parse_twitter_accounts(["x", "y"])
        assert result == ["x", "y"]


class TestParseTwitterCategorizedAccounts:
    """Tests for parse_twitter_categorized_accounts validator (news + analysis fields)."""

    def test_none_returns_empty_list(self) -> None:
        result = Settings.parse_twitter_categorized_accounts(None)
        assert result == []

    def test_csv_string(self) -> None:
        result = Settings.parse_twitter_categorized_accounts("a, b")
        assert result == ["a", "b"]

    def test_json_array_string(self) -> None:
        result = Settings.parse_twitter_categorized_accounts('["c", "d"]')
        assert result == ["c", "d"]

    def test_strips_at_prefix(self) -> None:
        result = Settings.parse_twitter_categorized_accounts("@foo, @bar")
        assert result == ["foo", "bar"]


class TestParseRedditSubreddits:
    """Tests for parse_reddit_subreddits validator."""

    def test_none_returns_empty_list(self) -> None:
        result = Settings.parse_reddit_subreddits(None)
        assert result == []

    def test_csv_string(self) -> None:
        result = Settings.parse_reddit_subreddits("stocks, options")
        assert result == ["stocks", "options"]

    def test_json_array_string(self) -> None:
        result = Settings.parse_reddit_subreddits('["wsb", "stocks"]')
        assert result == ["wsb", "stocks"]

    def test_strips_r_prefix(self) -> None:
        result = Settings.parse_reddit_subreddits("r/stocks, r/options")
        assert result == ["stocks", "options"]

    def test_strips_slash_r_prefix(self) -> None:
        result = Settings.parse_reddit_subreddits("/r/wsb, /r/stocks")
        assert result == ["wsb", "stocks"]


class TestGetTwitterSourceType:
    """Tests for get_twitter_source_type method."""

    def _make_settings(self, **kwargs: object) -> Settings:
        return Settings.model_construct(**kwargs)

    def test_news_account_returns_news(self) -> None:
        s = self._make_settings(twitter_news_accounts=["DeItaone", "realDonaldTrump"])
        assert s.get_twitter_source_type("DeItaone") == "news"

    def test_news_account_case_insensitive(self) -> None:
        s = self._make_settings(twitter_news_accounts=["DeItaone"])
        assert s.get_twitter_source_type("deitaone") == "news"
        assert s.get_twitter_source_type("DEITAONE") == "news"

    def test_unknown_account_returns_analysis(self) -> None:
        s = self._make_settings(twitter_news_accounts=["DeItaone"])
        assert s.get_twitter_source_type("someRandomUser") == "analysis"

    def test_strips_at_in_lookup(self) -> None:
        s = self._make_settings(twitter_news_accounts=["DeItaone"])
        assert s.get_twitter_source_type("@DeItaone") == "news"


class TestGetTelegramSourceType:
    """Tests for get_telegram_source_type method."""

    def test_always_returns_news(self) -> None:
        s = Settings.model_construct()
        assert s.get_telegram_source_type("any_channel") == "news"
        assert s.get_telegram_source_type("another") == "news"


class TestValidateSearxngUrl:
    """Tests for validate_searxng_url validator."""

    def _validate(self, url: str | None, env: str = "development") -> str | None:
        """Call the validator directly with a mock ValidationInfo."""
        from unittest.mock import MagicMock

        info = MagicMock()
        info.data = {"env": env}
        return Settings.validate_searxng_url(url, info)

    def test_none_returns_none(self) -> None:
        assert self._validate(None) is None

    def test_dev_mode_allows_localhost(self) -> None:
        result = self._validate("http://localhost:8080", env="development")
        assert result == "http://localhost:8080"

    def test_prod_rejects_localhost(self) -> None:
        with pytest.raises(ValueError, match="internal network"):
            self._validate("http://localhost:8080", env="production")

    def test_prod_rejects_127(self) -> None:
        with pytest.raises(ValueError, match="internal network"):
            self._validate("http://127.0.0.1:8080", env="production")

    def test_prod_rejects_10_x(self) -> None:
        with pytest.raises(ValueError, match="internal network"):
            self._validate("http://10.0.0.5:8080", env="production")

    def test_prod_rejects_192_168(self) -> None:
        with pytest.raises(ValueError, match="internal network"):
            self._validate("http://192.168.1.1:8080", env="production")

    def test_prod_rejects_ipv6_loopback(self) -> None:
        with pytest.raises(ValueError, match="internal network"):
            self._validate("http://[::1]:8080", env="production")

    def test_prod_allows_public_url(self) -> None:
        result = self._validate("https://search.example.com", env="production")
        assert result == "https://search.example.com"


# ─────────────────────────────────────────────────────────────
# Comprehensive env-var loading test
# ─────────────────────────────────────────────────────────────

# Every Settings field mapped to (ENV_VAR_NAME, test_value_string, expected_python_value).
# Aliased fields use their alias; all others use UPPER_CASE(field_name).
# SecretStr fields are checked via .get_secret_value().
# List fields use JSON array syntax.
_ENV_FIELD_SPECS: list[tuple[str, str, str, object]] = [
    # (field_name, env_var_name, env_string_value, expected_value)
    # --- Core (aliased) ---
    ("env", "SYNESIS_ENV", "staging", "staging"),
    ("debug", "SYNESIS_DEBUG", "true", True),
    ("log_level", "SYNESIS_LOG_LEVEL", "WARNING", "WARNING"),
    # --- Database ---
    (
        "database_url",
        "DATABASE_URL",
        "postgresql+asyncpg://u:p@h:1/db",
        "postgresql+asyncpg://u:p@h:1/db",
    ),
    # --- Redis ---
    ("redis_url", "REDIS_URL", "redis://localhost:6380/1", "redis://localhost:6380/1"),
    # --- Telegram ingestion ---
    ("telegram_api_id", "TELEGRAM_API_ID", "12345", 12345),
    ("telegram_api_hash", "TELEGRAM_API_HASH", "abc123hash", "abc123hash"),  # SecretStr
    ("telegram_session_name", "TELEGRAM_SESSION_NAME", "mysess", "mysess"),
    ("telegram_channels", "TELEGRAM_CHANNELS", '["chan1","chan2"]', ["chan1", "chan2"]),
    # --- Telegram notifications ---
    ("telegram_bot_token", "TELEGRAM_BOT_TOKEN", "bot:token123", "bot:token123"),  # SecretStr
    ("telegram_chat_id", "TELEGRAM_CHAT_ID", "99999", "99999"),
    # --- Twitter ---
    ("twitterapi_api_key", "TWITTERAPI_API_KEY", "twkey", "twkey"),  # SecretStr
    (
        "twitter_api_base_url",
        "TWITTER_API_BASE_URL",
        "https://tw.example.com",
        "https://tw.example.com",
    ),
    ("twitter_accounts", "TWITTER_ACCOUNTS", '["alice","bob"]', ["alice", "bob"]),
    ("twitter_news_accounts", "TWITTER_NEWS_ACCOUNTS", '["cnn"]', ["cnn"]),
    ("twitter_analysis_accounts", "TWITTER_ANALYSIS_ACCOUNTS", '["analyst1"]', ["analyst1"]),
    # --- Polymarket ---
    ("polymarket_api_key", "POLYMARKET_API_KEY", "pmkey", "pmkey"),  # SecretStr
    ("polymarket_api_secret", "POLYMARKET_API_SECRET", "pmsec", "pmsec"),  # SecretStr
    ("polymarket_private_key", "POLYMARKET_PRIVATE_KEY", "pmpk", "pmpk"),  # SecretStr
    ("polymarket_chain_id", "POLYMARKET_CHAIN_ID", "80001", 80001),
    # --- LLM ---
    ("llm_provider", "LLM_PROVIDER", "openai", "openai"),
    ("anthropic_api_key", "ANTHROPIC_API_KEY", "sk-ant-xxx", "sk-ant-xxx"),  # SecretStr
    ("openai_api_key", "OPENAI_API_KEY", "sk-xxx", "sk-xxx"),  # SecretStr
    (
        "openai_base_url",
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
        "https://api.openai.com/v1",
    ),
    ("llm_model", "LLM_MODEL", "gpt-4", "gpt-4"),
    ("llm_model_smart", "LLM_MODEL_SMART", "gpt-4-turbo", "gpt-4-turbo"),
    # --- Web search ---
    ("searxng_url", "SEARXNG_URL", "http://search.local:8080", "http://search.local:8080"),
    ("exa_api_key", "EXA_API_KEY", "exa123", "exa123"),  # SecretStr
    ("brave_api_key", "BRAVE_API_KEY", "brave456", "brave456"),  # SecretStr
    # --- Finnhub ---
    ("finnhub_api_key", "FINNHUB_API_KEY", "fhkey", "fhkey"),  # SecretStr
    # --- Providers ---
    ("price_provider", "PRICE_PROVIDER", "finnhub", "finnhub"),
    ("ticker_provider", "TICKER_PROVIDER", "finnhub", "finnhub"),
    ("fundamentals_provider", "FUNDAMENTALS_PROVIDER", "sec_edgar", "sec_edgar"),
    # --- SEC EDGAR ---
    ("sec_edgar_user_agent", "SEC_EDGAR_USER_AGENT", "Test test@test.com", "Test test@test.com"),
    ("sec_edgar_cache_ttl_submissions", "SEC_EDGAR_CACHE_TTL_SUBMISSIONS", "7200", 7200),
    ("sec_edgar_cache_ttl_cik_map", "SEC_EDGAR_CACHE_TTL_CIK_MAP", "43200", 43200),
    # --- NASDAQ ---
    ("nasdaq_cache_ttl_earnings", "NASDAQ_CACHE_TTL_EARNINGS", "10800", 10800),
    ("nasdaq_earnings_lookahead_days", "NASDAQ_EARNINGS_LOOKAHEAD_DAYS", "7", 7),
    # --- Crawl4AI ---
    ("crawl4ai_url", "CRAWL4AI_URL", "http://crawl:11235", "http://crawl:11235"),
    # --- FactSet SQL Server ---
    ("sqlserver_host", "SQLSERVER_HOST", "db.example.com", "db.example.com"),
    ("sqlserver_port", "SQLSERVER_PORT", "1434", 1434),
    ("sqlserver_database", "SQLSERVER_DATABASE", "mydb", "mydb"),
    ("sqlserver_user", "SQLSERVER_USER", "admin", "admin"),
    ("sqlserver_password", "SQLSERVER_PASSWORD", "s3cret", "s3cret"),  # SecretStr
    # --- Reddit ---
    ("reddit_subreddits", "REDDIT_SUBREDDITS", '["wsb","stocks"]', ["wsb", "stocks"]),
    ("reddit_poll_interval", "REDDIT_POLL_INTERVAL", "3600", 3600),
    # --- Kalshi ---
    ("kalshi_api_key", "KALSHI_API_KEY", "kalkey", "kalkey"),  # SecretStr
    ("kalshi_private_key_path", "KALSHI_PRIVATE_KEY_PATH", "/tmp/k.pem", "/tmp/k.pem"),
    ("kalshi_api_url", "KALSHI_API_URL", "https://kalshi.test/v2", "https://kalshi.test/v2"),
    ("kalshi_ws_url", "KALSHI_WS_URL", "wss://kalshi.test/ws", "wss://kalshi.test/ws"),
    # --- Polymarket Data/WS ---
    (
        "polymarket_data_api_url",
        "POLYMARKET_DATA_API_URL",
        "https://pm-data.test",
        "https://pm-data.test",
    ),
    ("polymarket_clob_ws_url", "POLYMARKET_CLOB_WS_URL", "wss://pm-ws.test", "wss://pm-ws.test"),
    # --- Market Intelligence (Flow 3) ---
    ("mkt_intel_enabled", "MKT_INTEL_ENABLED", "false", False),
    ("mkt_intel_interval", "MKT_INTEL_INTERVAL", "1800", 1800),
    ("mkt_intel_volume_spike_threshold", "MKT_INTEL_VOLUME_SPIKE_THRESHOLD", "2.5", 2.5),
    ("mkt_intel_insider_score_min", "MKT_INTEL_INSIDER_SCORE_MIN", "0.7", 0.7),
    ("mkt_intel_expiring_hours", "MKT_INTEL_EXPIRING_HOURS", "12", 12),
    ("mkt_intel_ws_enabled", "MKT_INTEL_WS_ENABLED", "false", False),
    ("mkt_intel_auto_watch_threshold", "MKT_INTEL_AUTO_WATCH_THRESHOLD", "0.8", 0.8),
    ("mkt_intel_unwatch_threshold", "MKT_INTEL_UNWATCH_THRESHOLD", "0.2", 0.2),
    # --- Watchlist Intelligence (Flow 4) ---
    ("watchlist_intel_enabled", "WATCHLIST_INTEL_ENABLED", "true", True),
    ("watchlist_intel_interval", "WATCHLIST_INTEL_INTERVAL", "10800", 10800),
    ("watchlist_intel_earnings_alert_days", "WATCHLIST_INTEL_EARNINGS_ALERT_DAYS", "3", 3),
    ("watchlist_intel_max_tickers", "WATCHLIST_INTEL_MAX_TICKERS", "50", 50),
    # --- Trading ---
    ("trading_enabled", "TRADING_ENABLED", "true", True),
    ("max_position_size", "MAX_POSITION_SIZE", "250.0", 250.0),
    ("min_edge_threshold", "MIN_EDGE_THRESHOLD", "0.1", 0.1),
    ("confidence_threshold", "CONFIDENCE_THRESHOLD", "0.8", 0.8),
    # --- Processing ---
    ("processing_workers", "PROCESSING_WORKERS", "10", 10),
    ("processing_queue_size", "PROCESSING_QUEUE_SIZE", "200", 200),
    ("web_search_max_queries", "WEB_SEARCH_MAX_QUERIES", "5", 5),
    ("polymarket_max_keywords", "POLYMARKET_MAX_KEYWORDS", "3", 3),
    # --- API URLs ---
    (
        "polymarket_gamma_api_url",
        "POLYMARKET_GAMMA_API_URL",
        "https://gamma.test",
        "https://gamma.test",
    ),
    ("finnhub_ws_url", "FINNHUB_WS_URL", "wss://fh.test", "wss://fh.test"),
    ("finnhub_api_url", "FINNHUB_API_URL", "https://fh.test/v1", "https://fh.test/v1"),
    ("signals_output_dir", "SIGNALS_OUTPUT_DIR", "/tmp/signals", Path("/tmp/signals")),
]

# Fields that are SecretStr (need .get_secret_value() to compare)
_SECRET_FIELDS = {
    "telegram_api_hash",
    "telegram_bot_token",
    "twitterapi_api_key",
    "polymarket_api_key",
    "polymarket_api_secret",
    "polymarket_private_key",
    "anthropic_api_key",
    "openai_api_key",
    "exa_api_key",
    "brave_api_key",
    "finnhub_api_key",
    "kalshi_api_key",
    "sqlserver_password",
}


class TestSettingsEnvLoading:
    """Verify every Settings field can be loaded from its env var."""

    def test_all_fields_loadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set every env var, create Settings, assert each field got the value."""
        # Set all env vars
        for _field, env_var, env_val, _expected in _ENV_FIELD_SPECS:
            monkeypatch.setenv(env_var, env_val)

        # Create settings without reading .env file
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Check each field
        for field_name, env_var, _env_val, expected in _ENV_FIELD_SPECS:
            actual = getattr(settings, field_name)
            if field_name in _SECRET_FIELDS:
                actual = actual.get_secret_value()
            assert actual == expected, (
                f"Field {field_name!r} (env={env_var}): expected {expected!r}, got {actual!r}"
            )

    def test_field_spec_covers_all_settings_fields(self) -> None:
        """Ensure _ENV_FIELD_SPECS covers every field in Settings."""
        model_fields = set(Settings.model_fields.keys())
        spec_fields = {field_name for field_name, *_ in _ENV_FIELD_SPECS}
        missing = model_fields - spec_fields
        assert not missing, (
            f"Fields missing from _ENV_FIELD_SPECS: {missing}. "
            "Add them to keep the env-loading test comprehensive."
        )
