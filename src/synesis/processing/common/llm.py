"""LLM model factory for PydanticAI.

Supports:
- Anthropic (Claude) - default
- OpenAI-compatible APIs (ZAI, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)

_OPENAI_BASE_URL = "https://api.openai.com/v1"


def is_native_openai() -> bool:
    """True when the configured provider is OpenAI's own API (not a proxy)."""
    settings = get_settings()
    base = (settings.openai_base_url or "").rstrip("/")
    return settings.llm_provider == "openai" and (not base or base == _OPENAI_BASE_URL)


# ── Web Search Configuration ──────────────────────────────────────


@dataclass(frozen=True)
class SearchConfig:
    """Web search configuration for a PydanticAI agent.

    Usage::

        search = web_search_config(cap=3, description="verify claims")
        tools = [_tool_web_read]
        if not search.native:
            tools.append(_tool_web_search)
        agent = Agent(
            ...,
            system_prompt=PROMPT.format(search_docs=search.prompt_docs, ...),
            tools=tools,
            builtin_tools=search.builtin_tools,
        )
    """

    prompt_docs: str
    builtin_tools: list[WebSearchTool] = field(default_factory=list)
    native: bool = False


def web_search_config(cap: int, description: str) -> SearchConfig:
    """Build web search config based on the current LLM provider.

    If native OpenAI: returns ``WebSearchTool`` in builtin_tools, prompt docs
    reference built-in search. Agent should NOT include Brave ``_tool_web_search``.

    Otherwise: returns empty builtin_tools, prompt docs reference Brave
    ``web_search(query, recency)`` tool. Agent must include ``_tool_web_search``.
    """
    if is_native_openai():
        return SearchConfig(
            prompt_docs=(
                f"- **Native web search** — you have built-in web search to {description}. "
                f"Budget: {cap} searches.\n"
            ),
            builtin_tools=[WebSearchTool(max_uses=cap)],
            native=True,
        )
    return SearchConfig(
        prompt_docs=(
            f"- `web_search(query, recency)` — {description}. "
            f"Budget: {cap} calls. Recency filter: day/week/month/year.\n"
        ),
        builtin_tools=[],
        native=False,
    )


def create_model(smart: bool = False, tier: str | None = None) -> str | Model:
    """Create a PydanticAI model based on configuration.

    Args:
        smart: Use the smart/capable model (llm_model_smart) for complex tasks.
               Default uses llm_model for faster/cheaper tasks.
        tier: Explicit model tier override. Currently supports "vsmart" for
              llm_model_vsmart (most capable model). Takes precedence over smart.

    Returns:
        Model string for Anthropic (e.g., "anthropic:claude-3-5-haiku-20241022")
        or OpenAI model instance for OpenAI-compatible APIs. Returns
        OpenAIResponsesModel (supports WebSearchTool) when on native OpenAI,
        OpenAIChatModel otherwise.
    """
    settings = get_settings()
    if tier == "vsmart":
        model_name = settings.llm_model_vsmart
    elif smart:
        model_name = settings.llm_model_smart
    else:
        model_name = settings.llm_model

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

    if is_native_openai():
        return OpenAIResponsesModel(model_name, provider=provider)
    return OpenAIChatModel(model_name, provider=provider)
