"""The watch command's crash watchdog restarts the pipeline on crash, stops on
a clean return or Ctrl-C, and gives up after the restart limit."""
import pytest

from cli import run_with_watchdog


def test_clean_return_does_not_restart():
    calls = {"n": 0}

    def run():
        calls["n"] += 1  # returns cleanly

    run_with_watchdog(run, max_restarts=10, delay_seconds=0)
    assert calls["n"] == 1


def test_crash_then_recover_restarts():
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("loky semaphore leak")
        # 4th call returns cleanly

    run_with_watchdog(run, max_restarts=10, delay_seconds=0)
    assert calls["n"] == 4  # 3 crashes + 1 clean run


def test_gives_up_after_max_restarts():
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        raise RuntimeError("always crashes")

    with pytest.raises(RuntimeError, match="always crashes"):
        run_with_watchdog(run, max_restarts=3, delay_seconds=0)
    # initial attempt + 3 restarts = 4 calls
    assert calls["n"] == 4


def test_keyboard_interrupt_stops_immediately():
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        raise KeyboardInterrupt

    run_with_watchdog(run, max_restarts=10, delay_seconds=0)  # must not raise
    assert calls["n"] == 1
