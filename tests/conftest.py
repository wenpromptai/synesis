"""Pytest fixtures and configuration."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async tests."""
    return "asyncio"
