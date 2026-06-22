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
    fake py_clob_client / executor.close_position_live."""
    from locus import config

    monkeypatch.setattr(config, "DRY_RUN", True, raising=False)


@pytest.fixture(autouse=True)
def _disable_momentum(monkeypatch):
    """Keep the momentum hybrid off by default so detect_edge_v2 never reaches
    for the live price-history API during tests. Tests that exercise momentum
    re-enable it explicitly and stub edge.get_price_momentum."""
    from locus import config

    monkeypatch.setattr(config, "MOMENTUM_ENABLED", False, raising=False)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the SQLite layer at a throwaway database."""
    from locus.memory import logger

    monkeypatch.setattr(logger, "DB_PATH", tmp_path / "test.db")
    logger.init_db()
    return logger
