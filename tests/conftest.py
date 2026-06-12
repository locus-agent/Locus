import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the SQLite layer at a throwaway database."""
    from locus.memory import logger

    monkeypatch.setattr(logger, "DB_PATH", tmp_path / "test.db")
    logger.init_db()
    return logger
