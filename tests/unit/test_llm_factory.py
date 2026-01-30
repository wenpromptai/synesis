"""Tests for LLM model factory."""

from unittest.mock import MagicMock, patch

from pydantic_ai.models.openai import OpenAIChatModel

from synesis.processing.llm_factory import create_model


class TestCreateModel:
    """Tests for create_model function."""

    def test_anthropic_model_default(self) -> None:
        """Test creating Anthropic model (default, not smart)."""
        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
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
        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
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
        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
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
        """Test creating OpenAI model without custom base URL."""
        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            result = create_model(smart=False)

        assert isinstance(result, OpenAIChatModel)

    def test_openai_model_smart_flag(self) -> None:
        """Test that smart flag selects correct model."""
        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = MagicMock()
            settings.openai_api_key.get_secret_value.return_value = "test-key"
            settings.openai_base_url = None
            mock_settings.return_value = settings

            # Regular model
            result_regular = create_model(smart=False)
            assert isinstance(result_regular, OpenAIChatModel)

            # Smart model
            result_smart = create_model(smart=True)
            assert isinstance(result_smart, OpenAIChatModel)

    def test_openai_model_no_api_key_raises(self) -> None:
        """Test creating OpenAI model without API key raises error."""
        import openai

        with patch("synesis.processing.llm_factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = "openai"
            settings.llm_model = "gpt-4o-mini"
            settings.llm_model_smart = "gpt-4o"
            settings.openai_api_key = None
            settings.openai_base_url = None
            mock_settings.return_value = settings

            # OpenAI client validates API key at creation time
            import pytest

            with pytest.raises(openai.OpenAIError, match="api_key"):
                create_model(smart=False)
