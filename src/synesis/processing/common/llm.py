"""LLM model factory for PydanticAI.

Supports:
- Anthropic (Claude) - default
- OpenAI-compatible APIs (ZAI, etc.)
"""

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)


def create_model(smart: bool = False) -> str | Model:
    """Create a PydanticAI model based on configuration.

    Args:
        smart: Use the smart/capable model (llm_model_smart) for complex tasks.
               Default uses llm_model for faster/cheaper tasks.

    Returns:
        Model string for Anthropic (e.g., "anthropic:claude-3-5-haiku-20241022")
        or OpenAIChatModel instance for OpenAI-compatible APIs.
    """
    settings = get_settings()
    model_name = settings.llm_model_smart if smart else settings.llm_model

    if settings.llm_provider == "anthropic":
        # PydanticAI uses "anthropic:model-name" format
        model_str = f"anthropic:{model_name}"
        logger.debug("Using Anthropic model", model=model_str, smart=smart)
        return model_str

    # OpenAI-compatible (including ZAI)
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None

    if settings.openai_base_url:
        provider = OpenAIProvider(
            base_url=settings.openai_base_url,
            api_key=api_key,
        )
        logger.debug(
            "Using OpenAI-compatible model",
            model=model_name,
            base_url=settings.openai_base_url,
            smart=smart,
        )
    else:
        provider = OpenAIProvider(api_key=api_key)
        logger.debug("Using OpenAI model", model=model_name, smart=smart)

    return OpenAIChatModel(model_name, provider=provider)
