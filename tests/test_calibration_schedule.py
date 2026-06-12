"""The in-pipeline calibration schedule: startup delay, cadence, crash immunity."""
import asyncio
import time

from locus.core.pipeline import run_calibration_schedule


def _summary(resolved=0, graded=0, total=0, accuracy=0.0):
    return {"resolved": resolved, "graded": graded, "total": total, "accuracy": accuracy}


def test_runs_after_startup_delay_then_on_interval():
    calls = []

    def runner():
        calls.append(time.monotonic())  # runs in an executor thread: no event loop here
        return _summary(resolved=1, graded=2, total=3, accuracy=66.7)

    async def main():
        task = asyncio.ensure_future(
            run_calibration_schedule(startup_delay=0.05, interval_seconds=0.05, runner=runner)
        )
        await asyncio.sleep(0.02)
        assert calls == [], "must not run before the startup delay"
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    assert len(calls) >= 3, f"expected repeated runs, got {len(calls)}"
    # spacing respects the interval (allowing scheduler jitter)
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert all(g >= 0.04 for g in gaps), gaps


def test_runner_crash_does_not_kill_the_schedule():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("gamma down")
        return _summary()

    async def main():
        task = asyncio.ensure_future(
            run_calibration_schedule(startup_delay=0, interval_seconds=0.03, runner=flaky)
        )
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    assert calls["n"] >= 3, "loop must survive a crashing cycle and keep running"


def test_cancellation_stops_the_schedule():
    calls = {"n": 0}

    def runner():
        calls["n"] += 1
        return _summary()

    async def main():
        task = asyncio.ensure_future(
            run_calibration_schedule(startup_delay=0, interval_seconds=0.02, runner=runner)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(main()) is True
