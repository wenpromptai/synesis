"""Shared API utilities."""

import asyncio
from collections.abc import Callable
from typing import Any


def create_tracked_task(
    coro: Any,
    task_set: set[asyncio.Task[Any]],
    on_done: Callable[[asyncio.Task[Any]], None],
) -> asyncio.Task[Any]:
    """Create a background task, track it in task_set, and call on_done when finished."""
    task = asyncio.create_task(coro)
    task_set.add(task)

    def _done(t: asyncio.Task[Any]) -> None:
        task_set.discard(t)
        on_done(t)

    task.add_done_callback(_done)
    return task
