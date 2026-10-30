"""startup_position_sync: a cheap, report-only DB-vs-chain check run at
pipeline startup via the public Data API (no CLOB auth needed). Mismatches
(outside the reconcile tolerance) are logged and sent to Telegram ONCE per
call — never per-position, never auto-closed. No wallet configured, or the
Data API unreachable, is a silent skip."""
import pytest

from locus import config
from locus.core import positions, telegram_bot
from locus.markets.gamma import Market


WALLET = "0xWALLET"


def _mkt(condition_id: str, yes_price: float = 0.50) -> Market:
    return Market(condition_id, f"Will {condition_id} happen?", "ai",
                  yes_price, 1 - yes_price, 5000, "", True, [],
                  slug=f"slug-{condition_id}")


def _open(tmp_db, condition_id: str, side: str = "YES", amount: float = 25.0,
          entry: float = 0.50, token_count: float | None = None) -> int:
    trade_id = tmp_db.log_trade(
        market_id=condition_id, market_question=f"Will {condition_id} happen?",
        claude_score=0.7, market_price=entry, edge=0.2, side=side,
        amount_usd=amount, status="executed", classification="bullish",
        materiality=0.7,
    )
    positions.open_position(trade_id, _mkt(condition_id, entry), side, amount,
                            headline="h", reasoning="r", token_count=token_count)
    return positions.get_open_positions()[0]["id"]


def _chain_row(cid: str, outcome_index: int, size: float) -> dict:
    return {"conditionId": cid, "outcomeIndex": outcome_index, "size": size}


@pytest.fixture(autouse=True)
def _telegram_capture(monkeypatch):
    """Capture any sent Telegram message instead of hitting the network, and
    enable the bot so notify_reconcile_mismatch actually tries to send."""
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "123")
    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "_send", lambda text: (sent.append(text), True)[1])
    return sent


def test_matching_holding_is_silent(tmp_db, _telegram_capture):
    _open(tmp_db, "c1", side="YES", entry=0.50, token_count=50.0)
    chain = [_chain_row("c1", 0, 50.0)]

    report = positions.startup_position_sync(wallet=WALLET,
                                              fetcher=lambda w: chain)

    assert report == {"checked": 1, "mismatches": []}
    assert _telegram_capture == []
    # Read-only: the position is untouched.
    assert positions.get_open_positions()[0]["status"] == "open"


def test_divergence_notifies_once_and_never_closes(tmp_db, _telegram_capture):
    # Position-57 shape: DB expects ~1.995, chain holds 0.005 — far outside
    # the 1%/0.5-token tolerance.
    pid = _open(tmp_db, "c57", side="YES", entry=0.266, token_count=1.995)
    chain = [_chain_row("c57", 0, 0.005)]

    report = positions.startup_position_sync(wallet=WALLET,
                                              fetcher=lambda w: chain)

    assert report["checked"] == 1
    assert len(report["mismatches"]) == 1
    assert report["mismatches"][0]["id"] == pid
    assert report["mismatches"][0]["expected"] == pytest.approx(1.995)
    assert report["mismatches"][0]["held"] == pytest.approx(0.005)
    # Exactly one Telegram message for the whole run, not one per position.
    assert len(_telegram_capture) == 1
    assert "STARTUP RECONCILE" in _telegram_capture[0]
    assert f"#{pid}" in _telegram_capture[0]
    # Never auto-closed.
    assert positions.get_open_positions()[0]["status"] == "open"


def test_multiple_mismatches_still_one_notification(tmp_db, _telegram_capture):
    _open(tmp_db, "c1", side="YES", entry=0.50, token_count=50.0)
    _open(tmp_db, "c2", side="NO", entry=0.50, token_count=50.0)
    chain = [_chain_row("c1", 0, 0.0), _chain_row("c2", 1, 0.0)]

    report = positions.startup_position_sync(wallet=WALLET,
                                              fetcher=lambda w: chain)

    assert len(report["mismatches"]) == 2
    assert len(_telegram_capture) == 1  # still a single message


def test_position_absent_from_chain_is_treated_as_zero(tmp_db, _telegram_capture):
    _open(tmp_db, "c_missing", side="YES", entry=0.50, token_count=50.0)

    report = positions.startup_position_sync(wallet=WALLET, fetcher=lambda w: [])

    assert len(report["mismatches"]) == 1
    assert report["mismatches"][0]["held"] == 0.0


def test_rounding_dust_within_tolerance_is_silent(tmp_db, _telegram_capture):
    _open(tmp_db, "c_dust", side="NO", entry=0.64, token_count=96.0)
    chain = [_chain_row("c_dust", 1, 95.9)]  # 0.1 sh off, 0.1%

    report = positions.startup_position_sync(wallet=WALLET,
                                              fetcher=lambda w: chain)

    assert report["mismatches"] == []
    assert _telegram_capture == []


def test_no_open_positions_skips_fetch(tmp_db, _telegram_capture):
    called = []
    report = positions.startup_position_sync(
        wallet=WALLET, fetcher=lambda w: called.append(w) or [])

    assert report == {"checked": 0, "mismatches": []}
    assert called == []  # the Data API was never hit — nothing to check


def test_no_wallet_configured_is_skipped(tmp_db, monkeypatch, _telegram_capture):
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "", raising=False)
    _open(tmp_db, "c1", side="YES", entry=0.50, token_count=50.0)
    called = []

    report = positions.startup_position_sync(
        fetcher=lambda w: called.append(w) or [])

    assert report is None
    assert called == []
    assert _telegram_capture == []


def test_fetch_failure_is_swallowed(tmp_db, _telegram_capture):
    _open(tmp_db, "c1", side="YES", entry=0.50, token_count=50.0)

    def boom(wallet):
        raise RuntimeError("network down")

    report = positions.startup_position_sync(wallet=WALLET, fetcher=boom)

    assert report is None
    assert _telegram_capture == []
    assert positions.get_open_positions()[0]["status"] == "open"


def test_fetch_wallet_positions_passes_size_threshold_zero(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"conditionId": "c1", "outcomeIndex": 0, "size": 10.0}]

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda timeout=15.0: FakeClient())

    rows = positions.fetch_wallet_positions(WALLET)
    assert rows == [{"conditionId": "c1", "outcomeIndex": 0, "size": 10.0}]
    assert captured["url"] == f"{config.POLYMARKET_DATA_HOST}/positions"
    assert captured["params"]["user"] == WALLET
    # sizeThreshold=0 so dust holdings (position 57's 0.005) aren't dropped
    # by the API's own default filtering.
    assert captured["params"]["sizeThreshold"] == 0


def test_fetch_wallet_positions_paginates_until_short_page(monkeypatch):
    pages = {
        0: [{"conditionId": f"c{i}", "outcomeIndex": 0, "size": 1.0} for i in range(500)],
        500: [{"conditionId": "c500", "outcomeIndex": 0, "size": 1.0}],
    }
    captured_offsets = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            offset = params["offset"]
            captured_offsets.append(offset)
            return FakeResponse(pages[offset])

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda timeout=15.0: FakeClient())

    rows = positions.fetch_wallet_positions(WALLET, page_size=500)
    assert len(rows) == 501  # both pages concatenated
    assert captured_offsets == [0, 500]  # stopped after the short page


def test_fetch_wallet_positions_single_short_page_no_second_call(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"conditionId": "c1", "outcomeIndex": 0, "size": 1.0}]

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls.append(params["offset"])
            return FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", lambda timeout=15.0: FakeClient())

    rows = positions.fetch_wallet_positions(WALLET, page_size=500)
    assert len(rows) == 1
    assert calls == [0]  # a single short page ends pagination immediately
