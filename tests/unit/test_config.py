"""Unit tests for Settings validators in config.py."""

import pytest

from synesis.config import Settings


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
