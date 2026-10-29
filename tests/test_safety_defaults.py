"""Safe defaults: paper mode unless explicitly acknowledged, auto-push off,
momentum cached, edge computed off the event loop and fail-closed on no book,
atomic daily-spend reservation."""
import asyncio
import importlib
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from locus import config


def _reload_config(monkeypatch, **env):
    for k in ("DRY_RUN", "LOCUS_LIVE_ACK", "AUTO_PUSH_STATUS", "ORDERBOOK_FAIL_OPEN", "EDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    return importlib.reload(config)


@pytest.fixture
def restore_config(monkeypatch):
    yield
    monkeypatch.undo()
    importlib.reload(config)


def test_defaults_are_safe(monkeypatch, restore_config):
    c = _reload_config(monkeypatch)
    assert c.DRY_RUN is True
    assert c.AUTO_PUSH_STATUS is False
    assert c.ORDERBOOK_FAIL_OPEN is False
    assert c.EDGE_MODEL == "calibrated"


def test_dry_run_false_alone_does_not_go_live(monkeypatch, restore_config):
    assert _reload_config(monkeypatch, DRY_RUN="false").DRY_RUN is True


def test_live_needs_ack(monkeypatch, restore_config):
    c = _reload_config(monkeypatch, DRY_RUN="false", LOCUS_LIVE_ACK="I_UNDERSTAND_REAL_MONEY")
    assert c.DRY_RUN is False


def test_cli_live_flag_refused_without_ack(monkeypatch):
    import cli
    monkeypatch.setattr(config, "LIVE_ACK", False)
    monkeypatch.setattr(config, "DRY_RUN", True)
    started = {}
    monkeypatch.setattr("locus.core.pipeline.run_pipeline_v2", lambda: started.setdefault("run", True))
    monkeypatch.setattr("locus.core.telegram_bot.start_bot_thread", lambda *a, **k: None, raising=False)
    try:
        cli.cmd_watch(SimpleNamespace(live=True, threshold=None))
    except Exception:
        pass  # anything after the mode switch is irrelevant here
    assert config.DRY_RUN is True


def test_momentum_is_cached(monkeypatch):
    from locus.core import edge
    calls = []
    monkeypatch.setattr(edge, "_fetch_price_momentum", lambda t, lb: calls.append(t) or 0.1)
    monkeypatch.setattr(config, "MOMENTUM_CACHE_TTL_SECONDS", 60)
    edge._momentum_cache.clear()
    m = SimpleNamespace(tokens=[{"token_id": "tY", "outcome": "Yes"}])
    assert edge.get_price_momentum(m, 60) == 0.1
    assert edge.get_price_momentum(m, 60) == 0.1
    assert calls == ["tY"]
    edge._momentum_cache.clear()


def test_compute_edge_runs_off_loop_and_fails_closed(monkeypatch):
    from locus.core import pipeline, orderbook
    monkeypatch.setattr(config, "EDGE_MODEL", "calibrated")
    main_thread = threading.get_ident()
    seen = {}

    def fake_fetch(token):
        seen["thread"] = threading.get_ident()
        return None  # book unavailable

    monkeypatch.setattr(orderbook, "fetch_book", fake_fetch)
    called = []
    monkeypatch.setattr(pipeline, "detect_edge_v2", lambda *a, **k: called.append(1))
    market = SimpleNamespace(tokens=[{"token_id": "tY", "outcome": "Yes"}], slug="s", condition_id="c")
    cls = SimpleNamespace(direction="bullish")
    out = asyncio.run(pipeline.compute_edge(market, cls, None))
    assert out is None and called == []
    assert seen["thread"] != main_thread


def test_compute_edge_passes_book(monkeypatch):
    from locus.core import pipeline, orderbook
    monkeypatch.setattr(config, "EDGE_MODEL", "calibrated")
    book = {"bids": [(0.4, 1)], "asks": [(0.5, 1)]}
    monkeypatch.setattr(orderbook, "fetch_book", lambda t: book)
    got = {}
    monkeypatch.setattr(pipeline, "detect_edge_v2",
                        lambda m, c, e, book=None: got.setdefault("book", book))
    market = SimpleNamespace(tokens=[{"token_id": "tY", "outcome": "Yes"}], slug="s", condition_id="c")
    asyncio.run(pipeline.compute_edge(market, SimpleNamespace(direction="bearish"), None))
    assert got["book"] is book


# --- daily spend race -----------------------------------------------------------

def test_daily_spend_reservation_is_atomic(tmp_db, monkeypatch):
    from locus.core import executor
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 100.0)
    monkeypatch.setattr(executor, "_reserved_usd", 0.0)

    def slow_execute(signal):
        time.sleep(0.05)  # widen the window the old check-then-act raced in
        return executor._log_and_return(signal, status="dry_run", order_id=None)

    monkeypatch.setattr(executor, "_execute_reserved", slow_execute)
    from locus.core.edge import Signal
    from locus.markets.gamma import Market

    def sig(i):
        m = Market(f"c{i}", f"Q{i}?", "ai", 0.5, 0.5, 5000, "", True, [])
        return Signal(market=m, claude_score=0.5, market_price=0.5, edge=0.1, side="YES",
                      bet_amount=30.0, reasoning="", headlines=f"h{i}")

    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(executor.execute_trade(sig(i))))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    accepted = [r for r in results if r.get("status") == "dry_run"]
    assert len(accepted) == 3            # 3 x $30 = $90 <= $100; a 4th would exceed
    assert abs(tmp_db.get_daily_pnl()) == pytest.approx(90.0)
    assert executor._reserved_usd == pytest.approx(0.0)
