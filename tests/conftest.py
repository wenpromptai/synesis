"""Pytest fixtures and configuration."""

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests by default unless -m integration is specified."""
    # Check if user explicitly requested integration tests
    markexpr = config.getoption("-m", default="")
    if "integration" in markexpr:
        # User wants integration tests, don't skip
        return

    # Skip integration tests by default
    skip_integration = pytest.mark.skip(reason="Integration test - run with: pytest -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async tests."""
    return "asyncio"
