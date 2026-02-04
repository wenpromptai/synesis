"""Agent module for news processing using PydanticAI.

Provides `agent_lifespan()` — an async context manager that starts and stops
the full ingestion + processing pipeline. Used by the FastAPI server lifespan.

Usage:
    uv run synesis
"""

from synesis.agent.__main__ import AgentState, agent_lifespan

__all__ = ["AgentState", "agent_lifespan"]
