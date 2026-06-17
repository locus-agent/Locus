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


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the SQLite layer at a throwaway database."""
    from locus.memory import logger

    monkeypatch.setattr(logger, "DB_PATH", tmp_path / "test.db")
    logger.init_db()
    return logger
