"""
Task supervision for the long-running asyncio pipelines.

asyncio.gather(..., return_exceptions=True) lets a crashed task die silently
while the rest of the system keeps looking alive — news counts up, status
exports, but nothing trades. supervise() restarts crashed tasks with a delay
and logs loudly instead.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


async def supervise(
    name: str,
    factory: Callable[[], Awaitable],
    stats: dict | None = None,
    restart_delay: float = 5.0,
) -> None:
    """Run factory() until it returns cleanly; restart it whenever it raises.

    A clean return is treated as intentional completion (e.g. a disabled news
    source) and ends supervision. Exceptions are logged with traceback,
    counted in stats["task_restarts"] when a stats dict is given, and the
    task is restarted after restart_delay seconds.
    """
    while True:
        try:
            await factory()
            log.info(f"[supervisor] Task {name!r} completed; not restarting")
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f"[supervisor] Task {name!r} crashed; restarting in {restart_delay}s")
            if stats is not None:
                stats["task_restarts"] = stats.get("task_restarts", 0) + 1
            await asyncio.sleep(restart_delay)
