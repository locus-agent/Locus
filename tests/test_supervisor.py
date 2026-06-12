"""supervise() must restart crashed tasks and stop on clean completion."""
import asyncio

from locus.supervisor import supervise


def test_restarts_after_crashes():
    calls = {"n": 0}
    stats = {}

    async def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError(f"boom {calls['n']}")
        # third run completes cleanly

    asyncio.run(supervise("flaky", flaky, stats, restart_delay=0))
    assert calls["n"] == 3
    assert stats["task_restarts"] == 2


def test_clean_exit_is_not_restarted():
    calls = {"n": 0}

    async def once():
        calls["n"] += 1

    asyncio.run(supervise("once", once, restart_delay=0))
    assert calls["n"] == 1


def test_cancellation_propagates():
    async def forever():
        await asyncio.sleep(3600)

    async def main():
        task = asyncio.ensure_future(supervise("forever", forever, restart_delay=0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "swallowed"

    assert asyncio.run(main()) == "cancelled"
