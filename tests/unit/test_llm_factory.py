"""Tests for LLM model factory."""

from unittest.mock import MagicMock, patch

from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

import pytest

from synesis.processing.common.llm import (
    SearchConfig,
    create_model,
    is_native_openai,
    web_search_config,
)


class TestCreateModel:
    """Tests for create_model function."""

    def test_anthropic_model_default(self) -> None:
        """Test creating Anthropic model (default, not smart)."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "anthropic"
            settings.llm_model = "claude-3-5-haiku-20241022"
            settings.llm_model_smart = "claude-3-5-sonnet-20241022"
            mock_settings.return_value = settings

            result = create_model(smart=False)

        assert isinstance(result, str)
        assert result == "anthropic:claude-3-5-haiku-20241022"

    def test_anthropic_model_smart(self) -> None:
        """Test creating Anthropic smart model."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "anthropic"
            settings.llm_model = "claude-3-5-haiku-20241022"
            settings.llm_model_smart = "claude-3-5-sonnet-20241022"
            mock_settings.return_value = settings

            result = create_model(smart=True)

        assert isinstance(result, str)
        assert result == "anthropic:claude-3-5-sonnet-20241022"

    def test_openai_model_with_base_url(self) -> None:
        """Test creating OpenAI-compatible model with custom base URL."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = "https://custom.openai.com/v1"
            mock_settings.return_value = settings

            result = create_model(smart=False)

        assert isinstance(result, OpenAIChatModel)

    def test_openai_model_without_base_url(self) -> None:
        """Test native OpenAI (no base URL) returns OpenAIResponsesModel."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            result = create_model(smart=False)

        assert isinstance(result, OpenAIResponsesModel)

    def test_openai_model_with_native_base_url(self) -> None:
        """Test explicit OpenAI base URL still returns OpenAIResponsesModel."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = "https://api.openai.com/v1"
            mock_settings.return_value = settings

            result = create_model(smart=False)

        assert isinstance(result, OpenAIResponsesModel)

    def test_openai_model_smart_flag(self) -> None:
        """Test that smart flag selects correct model."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            # Regular model (native OpenAI → ResponsesModel)
            result_regular = create_model(smart=False)
            assert isinstance(result_regular, OpenAIResponsesModel)

            # Smart model (native OpenAI → ResponsesModel)
            result_smart = create_model(smart=True)
            assert isinstance(result_smart, OpenAIResponsesModel)

    def test_openai_model_no_api_key_raises(self) -> None:
        """Test creating OpenAI model without API key raises error."""
        import openai

        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = None
            settings.openai_base_url = None
            mock_settings.return_value = settings

            # OpenAI client validates API key at creation time
            with pytest.raises(openai.OpenAIError, match="api_key"):
                create_model(smart=False)


class TestIsNativeOpenAI:
    """Tests for is_native_openai helper."""

    @pytest.mark.parametrize(
        ("provider", "base_url", "expected"),
        [
            ("openai", None, True),
            ("openai", "", True),
            ("openai", "https://api.openai.com/v1", True),
            ("openai", "https://api.openai.com/v1/", True),
            ("openai", "https://api.z.ai/api/coding/paas/v4", False),
            ("openai", "https://custom.proxy.com/v1", False),
            ("anthropic", None, False),
            ("anthropic", "https://api.openai.com/v1", False),
        ],
    )
    def test_is_native_openai(self, provider: str, base_url: str | None, expected: bool) -> None:
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = provider
            settings.openai_base_url = base_url
            mock_settings.return_value = settings

            assert is_native_openai() is expected


class TestWebSearchConfig:
    """Tests for web_search_config helper."""

    def test_native_openai_returns_builtin_tools(self) -> None:
        """Native OpenAI should return WebSearchTool and native=True."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            config = web_search_config(3, "test description")

        assert config.native is True
        assert len(config.builtin_tools) == 1
        assert isinstance(config.builtin_tools[0], WebSearchTool)
        assert "built-in web search" in config.prompt_docs
        assert "test description" in config.prompt_docs
        assert "3" in config.prompt_docs

    def test_non_native_returns_brave_docs(self) -> None:
        """Non-native (custom base URL) should return empty builtin_tools and native=False."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.openai_base_url = "https://api.z.ai/v1"
            mock_settings.return_value = settings

            config = web_search_config(5, "verify claims")

        assert config.native is False
        assert config.builtin_tools == []
        assert "web_search(query, recency)" in config.prompt_docs
        assert "verify claims" in config.prompt_docs
        assert "5" in config.prompt_docs

    def test_anthropic_returns_brave_docs(self) -> None:
        """Anthropic provider should return Brave search docs."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "anthropic"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            config = web_search_config(2, "find info")

        assert config.native is False
        assert config.builtin_tools == []
        assert "web_search(query, recency)" in config.prompt_docs

    def test_search_config_is_frozen(self) -> None:
        """SearchConfig should be immutable."""
        config = SearchConfig(prompt_docs="test", builtin_tools=[], native=False)
        with pytest.raises(AttributeError):
            config.native = True  # type: ignore[misc]

    def test_native_with_explicit_openai_url(self) -> None:
        """Explicit https://api.openai.com/v1 should still be native."""
        with patch("synesis.processing.common.llm.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.openai_base_url = "https://api.openai.com/v1"
            mock_settings.return_value = settings

            config = web_search_config(3, "test")

        assert config.native is True
        assert len(config.builtin_tools) == 1
