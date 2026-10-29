"""Telethon news listener with a mocked client: graceful disable, connect-only
(never logs in), message -> NewsEvent attribution/latency, channel filter,
priority routing ahead of RSS, cross-source dedup, latency metrics."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import latency
from locus.sources import telegram_mtproto as tm
from locus.sources.news_stream import NewsEvent, PriorityNewsQueue


def _listener(tmp_path, **kw):
    sess = tmp_path / "news.session"
    if kw.pop("make_session", True):
        sess.write_bytes(b"x")
    args = dict(api_id="123", api_hash="abc", session=str(sess), channels=["WalterBloomberg", "@financialjuice"])
    args.update(kw)
    return tm.TelegramNewsListener(**args)


class FakeClient:
    instances = []

    def __init__(self, session, api_id, api_hash, authorized=True):
        self.session, self.api_id, self.api_hash = session, api_id, api_hash
        self.authorized = authorized
        self.handlers = []
        self.calls = []
        FakeClient.instances.append(self)

    async def connect(self):
        self.calls.append("connect")

    async def is_user_authorized(self):
        return self.authorized

    async def disconnect(self):
        self.calls.append("disconnect")

    def add_event_handler(self, cb, builder):
        self.handlers.append((cb, builder))

    async def run_until_disconnected(self):
        self.calls.append("run")
        await asyncio.sleep(3600)

    async def start(self, *a, **k):  # must never be called by the pipeline
        raise AssertionError("listener tried to log in")

    sign_in = start


class FakeEvents:
    class NewMessage:
        def __init__(self, chats=None):
            self.chats = chats


@pytest.fixture
def fake_telethon(monkeypatch):
    FakeClient.instances.clear()
    monkeypatch.setattr(tm, "_load_telethon", lambda: (FakeClient, FakeEvents))
    return FakeClient


def _msg(text, mid=7, age_s=2.0):
    return SimpleNamespace(message=text, id=mid,
                           date=datetime.now(timezone.utc) - timedelta(seconds=age_s))


# --- graceful disable ------------------------------------------------------------

@pytest.mark.parametrize("kw,needle", [
    ({"api_id": ""}, "TELEGRAM_API_ID"),
    ({"api_hash": ""}, "TELEGRAM_API_ID"),
    ({"channels": []}, "TELEGRAM_NEWS_CHANNELS"),
    ({"session": ""}, "TELEGRAM_SESSION"),
    ({"make_session": False}, "telegram_login.py"),
])
def test_disabled_without_creds(tmp_path, fake_telethon, kw, needle):
    lst = _listener(tmp_path, **kw)
    assert not lst.enabled and needle in lst.disabled_reason()
    asyncio.run(asyncio.wait_for(lst.stream(asyncio.Queue()), 1))  # returns, no client
    assert FakeClient.instances == [] and lst.status.startswith("disabled")


def test_missing_telethon_disables(tmp_path, monkeypatch):
    def boom():
        raise ImportError("no telethon")
    monkeypatch.setattr(tm, "_load_telethon", boom)
    lst = _listener(tmp_path)
    asyncio.run(asyncio.wait_for(lst.stream(asyncio.Queue()), 1))
    assert "telethon" in lst.status


def test_unauthorized_session_never_logs_in(tmp_path, monkeypatch):
    clients = []

    def factory(*a):
        c = FakeClient(*a, authorized=False)
        clients.append(c)
        return c
    monkeypatch.setattr(tm, "_load_telethon", lambda: (factory, FakeEvents))
    lst = _listener(tmp_path)
    asyncio.run(asyncio.wait_for(lst.stream(asyncio.Queue()), 1))
    assert clients[0].calls == ["connect", "disconnect"]
    assert lst.status == "disabled: session not authorized"


# --- listening ----------------------------------------------------------------------

def test_listens_and_pushes_attributed_events(tmp_path, fake_telethon):
    lst = _listener(tmp_path)
    q = asyncio.Queue()

    async def scenario():
        task = asyncio.create_task(lst.stream(q))
        await asyncio.sleep(0.05)
        client = FakeClient.instances[0]
        assert client.calls == ["connect", "run"]
        cb, builder = client.handlers[0]
        assert builder.chats == ["WalterBloomberg", "financialjuice"]
        chat = SimpleNamespace(username="WalterBloomberg")
        await cb(SimpleNamespace(chat=chat, message=_msg("🚨 BREAKING: FED CUTS RATES BY 50BPS", mid=42)))
        # other channel (e.g. a forwarded chat) is filtered out
        await cb(SimpleNamespace(chat=SimpleNamespace(username="spam"), message=_msg("Totally unrelated news")))
        # media-only post -> nothing
        await cb(SimpleNamespace(chat=chat, message=_msg("")))
        task.cancel()
    asyncio.run(scenario())
    assert q.qsize() == 1
    ev = q.get_nowait()
    assert ev.source == "telegram"
    assert ev.headline == "FED CUTS RATES BY 50BPS"
    assert ev.url == "https://t.me/walterbloomberg/42"
    assert ev.raw_data == {"channel": "walterbloomberg", "message_id": 42, "via": "mtproto"}
    assert 1500 <= ev.latency_ms <= 10000
    assert ev.published_at < ev.received_at


def test_session_path_resolution(tmp_path):
    assert tm.session_file(str(tmp_path / "a")).name == "a.session"
    assert tm.session_file("x.session") == config.PROJECT_ROOT / "x.session"


@pytest.mark.parametrize("raw,expected", [
    ("⚡️ JUST IN: Trump signs executive order on AI", "Trump signs executive order on AI"),
    ("**BREAKING** - Ceasefire announced in Gaza\nmore text", "Ceasefire announced in Gaza"),
    ("🇺🇸 US CPI 3.1% VS 3.2% EXPECTED https://t.co/x", "US CPI 3.1% VS 3.2% EXPECTED"),
    ("ok\n\nSEC approves spot SOL ETF", "SEC approves spot SOL ETF"),
    ("🔥🔥", ""),
])
def test_normalize_headline(raw, expected):
    assert tm.normalize_headline(raw) == expected


# --- aggregator integration ------------------------------------------------------------

def _ev(source, headline):
    now = datetime.now(timezone.utc)
    return NewsEvent(headline=headline, source=source, url="", received_at=now,
                     published_at=now, latency_ms=1200)


def test_priority_queue_routes_telegram_before_rss():
    async def go():
        q = PriorityNewsQueue()
        for i in range(3):
            await q.put(_ev("rss", f"rss {i}"))
        await q.put(_ev("telegram", "tg first"))
        await q.put(_ev("twitter", "tw second"))
        return [(await q.get()).headline for _ in range(5)]
    assert asyncio.run(go()) == ["tg first", "tw second", "rss 0", "rss 1", "rss 2"]


def test_aggregator_dedups_rss_after_telegram(monkeypatch):
    from locus.sources import news_stream as ns
    latency.reset()
    agg = ns.NewsAggregator(asyncio.Queue())
    monkeypatch.setattr(agg._deduper, "_encode", lambda t: (_ for _ in ()).throw(RuntimeError("no model")))

    async def go():
        await agg._internal_queue.put(_ev("rss", "Fed cuts rates by 50bps"))
        await agg._internal_queue.put(_ev("telegram", "Fed cuts rates by 50bps"))
        router = asyncio.create_task(agg._dedup_router())
        await asyncio.sleep(0.1)
        router.cancel()
        out = []
        while not agg.output_queue.empty():
            out.append(agg.output_queue.get_nowait())
        return out
    out = asyncio.run(go())
    assert [e.source for e in out] == ["telegram"]   # RSS copy of the same story dropped
    assert agg.stats["deduped"] == 1
    assert latency.summary()["telegram"]["publish_to_receive_ms_p50"] == 1200


def test_rss_slows_down_when_realtime_source_configured(monkeypatch, tmp_path):
    from locus.sources import news_stream as ns
    sess = tmp_path / "s.session"
    sess.write_bytes(b"x")
    monkeypatch.setattr(config, "TELEGRAM_API_ID", "1")
    monkeypatch.setattr(config, "TELEGRAM_API_HASH", "h")
    monkeypatch.setattr(config, "TELEGRAM_SESSION", str(sess))
    monkeypatch.setattr(config, "TELEGRAM_NEWS_CHANNELS", ["WalterBloomberg"])
    monkeypatch.setattr(config, "TWITTER_BEARER_TOKEN", "")
    assert ns.NewsAggregator(asyncio.Queue()).rss.interval == config.RSS_INTERVAL_WITH_REALTIME_SECONDS
    monkeypatch.setattr(config, "TELEGRAM_API_ID", "")
    assert ns.NewsAggregator(asyncio.Queue()).rss.interval == config.RSS_INTERVAL_SECONDS


def test_default_channels_are_verified_list():
    assert "WalterBloomberg" in config.DEFAULT_TELEGRAM_NEWS_CHANNELS
    assert all(not c.startswith("@") for c in config.TELEGRAM_NEWS_CHANNELS)


# --- latency metrics -------------------------------------------------------------------

def test_latency_summary_percentiles():
    latency.reset()
    for ms in (100, 200, 300, 400, 5000):
        latency.record_receive("telegram", ms)
    latency.record_receive("rss", -1)  # unknown publish time ignored
    t0 = datetime.now(timezone.utc)
    latency.record_classified("telegram", t0 - timedelta(seconds=4), now=t0)
    s = latency.summary()
    assert s["telegram"]["publish_to_receive_ms_p50"] == 300
    assert s["telegram"]["publish_to_receive_ms_p90"] == 5000
    assert s["telegram"]["receive_to_classify_ms_p50"] == 4000
    assert s.get("rss", {}).get("n_received", 0) == 0
    latency.reset()
