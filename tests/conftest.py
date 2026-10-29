import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _disable_telegram(monkeypatch):
    """Keep Telegram notifications off by default so the suite never touches the
    network (a developer .env may carry a real bot token). Tests that exercise
    the bot re-enable it explicitly."""
    from locus import config

    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "", raising=False)
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "", raising=False)


@pytest.fixture(autouse=True)
def _force_dry_run(monkeypatch):
    """Default every test to dry-run so the exit/close paths simulate fills
    instead of hitting the real CLOB — a developer .env may set DRY_RUN=false
    (they trade --live), and without this the suite would place/await real
    orders. Tests that exercise the live path set DRY_RUN=false explicitly and
    fake py_clob_client_v2 / executor.close_position_live."""
    from locus import config

    monkeypatch.setattr(config, "DRY_RUN", True, raising=False)


@pytest.fixture(autouse=True)
def _pin_passive_defaults(monkeypatch):
    """Pin the passive limit-order subsystem to its shipped defaults so the
    suite is hermetic against a developer .env that enables it (a live-trading
    .env may set PASSIVE_LIMIT_ENABLED=true, which reroutes every executor BUY
    to the passive path and breaks the aggressive-path tests). Tests that
    exercise the passive path re-enable the flag explicitly via monkeypatch."""
    from locus import config

    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "PASSIVE_MIN_HOURS_TO_RESOLUTION", 72.0, raising=False)
    monkeypatch.setattr(config, "PASSIVE_LIMIT_TIMEOUT_HOURS", 6.0, raising=False)
    monkeypatch.setattr(config, "PASSIVE_PRICE_IMPROVE_TICKS", 1, raising=False)
    monkeypatch.setattr(config, "PASSIVE_CHASE_AWAY_PCT", 10.0, raising=False)
    monkeypatch.setattr(config, "PASSIVE_FILL_TOLERANCE_PCT", 0.5, raising=False)


@pytest.fixture(autouse=True)
def _pin_legacy_models(monkeypatch):
    """The pre-existing suite was written against the legacy edge model and the
    single-shot GTC close. Pin those (both still shipped, selectable via
    EDGE_MODEL=legacy / EXIT_LADDER_STEPS=1) so the old assertions keep
    testing what they were written for; the new behavior has its own tests
    (test_calibrated_edge.py, test_exit_ladder.py, ...) which re-enable it."""
    from locus import config

    monkeypatch.setattr(config, "EDGE_MODEL", "legacy", raising=False)
    monkeypatch.setattr(config, "EXIT_LADDER_STEPS", 1, raising=False)
    monkeypatch.setattr(config, "EXIT_RETRY_DELAY_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(config, "ORDERBOOK_FAIL_OPEN", False, raising=False)


@pytest.fixture(autouse=True)
def _hermetic_book(monkeypatch):
    """No test may hit the public CLOB /book endpoint. Default to a balanced
    two-sided book (imbalance 0) so pipeline tests that aren't about the
    orderbook gate pass it, exactly as they did when the SDK-less fetch
    returned None and the gate failed open. Orderbook tests override this."""
    from locus.core import orderbook

    monkeypatch.setattr(
        orderbook, "_http_get_book",
        lambda token_id: {"bids": [{"price": "0.49", "size": "100"}],
                          "asks": [{"price": "0.51", "size": "100"}]},
    )


@pytest.fixture(autouse=True)
def _disable_momentum(monkeypatch):
    """Keep the momentum hybrid off by default so detect_edge_v2 never reaches
    for the live price-history API during tests. Tests that exercise momentum
    re-enable it explicitly and stub edge.get_price_momentum."""
    from locus import config

    monkeypatch.setattr(config, "MOMENTUM_ENABLED", False, raising=False)


@pytest.fixture(autouse=True)
def _clear_close_backoff():
    """The close-retry backoff and last-close-failure maps are in-memory and
    keyed by position id, and position ids restart at 1 in every tmp_db —
    clear them around each test so state from one test can't leak into
    another."""
    from locus.core import positions

    positions._close_backoff.clear()
    positions._last_close_failure.clear()
    yield
    positions._close_backoff.clear()
    positions._last_close_failure.clear()


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the SQLite layer at a throwaway database."""
    from locus.memory import logger

    monkeypatch.setattr(logger, "DB_PATH", tmp_path / "test.db")
    logger.init_db()
    return logger
