"""Real-time Telegram channel listener (MTProto user client via Telethon).

Why MTProto and not the Bot API: a bot only sees channels it was added to as
admin, so it cannot follow public news channels like @WalterBloomberg. A user
client subscribes to any public channel and gets pushes in ~1 s.

Safety rules baked in:
- Never logs in and never creates a session. It only CONNECTS with an existing
  session file made once by `scripts/telegram_login.py`. Missing creds, missing
  session file, or an unauthorized session -> the source disables itself with a
  log line (the rest of the pipeline keeps running).
- Read-only: it registers a NewMessage handler, nothing is ever sent.
- Telethon is imported lazily, so the dependency is optional.

Each message becomes a NewsEvent with source="telegram", the channel username
and message link as attribution, published_at = Telegram's message date and
received_at = local receive time (latency_ms = receive - publish).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from locus import config

log = logging.getLogger(__name__)

# Leading noise news channels put in front of the actual headline. Stripped so
# the cross-source deduper sees the same text an RSS/Twitter item would carry.
_PREFIX_RE = re.compile(
    r"^\s*(?:[\W_]+\s*)?(?:(?:breaking|just in|urgent|alert|update|flash|developing)"
    r"\s*(?:news)?\s*[:\-–—|]\s*)+",
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]+"
)
_MD_RE = re.compile(r"[*_`~]+|\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\S+")


def normalize_headline(text: str, max_len: int = 500) -> str:
    """First meaningful line of a channel post, without emojis, markdown,
    links and 'BREAKING:'-style prefixes."""
    if not text:
        return ""
    for line in text.splitlines():
        line = _MD_RE.sub(lambda m: m.group(1) or "", line)
        line = _URL_RE.sub("", line)
        line = _EMOJI_RE.sub("", line)
        line = _PREFIX_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip(" -–—|:•")
        if len(line) >= 12:
            return line[:max_len]
    return ""


def _load_telethon():
    """(TelegramClient, events). Isolated so tests inject a fake."""
    from telethon import TelegramClient, events
    return TelegramClient, events


def session_file(session: str) -> Path:
    p = Path(session)
    if not p.is_absolute():
        p = config.PROJECT_ROOT / p
    return p if p.suffix == ".session" else p.with_name(p.name + ".session")


class TelegramNewsListener:
    def __init__(self, api_id: int | str | None, api_hash: str | None,
                 session: str | None, channels: list[str]):
        self.api_id = int(api_id) if str(api_id or "").strip().isdigit() else None
        self.api_hash = (api_hash or "").strip()
        self.session = (session or "").strip()
        self.channels = [c.strip().lstrip("@") for c in channels if c and c.strip()]
        self.client = None
        self.status = "init"

    def disabled_reason(self) -> str | None:
        if not self.api_id or not self.api_hash:
            return "TELEGRAM_API_ID / TELEGRAM_API_HASH not set"
        if not self.channels:
            return "TELEGRAM_NEWS_CHANNELS is empty"
        if not self.session:
            return "TELEGRAM_SESSION not set"
        if not session_file(self.session).exists():
            return (f"session file {session_file(self.session)} not found — "
                    "run `python scripts/telegram_login.py` once")
        return None

    @property
    def enabled(self) -> bool:
        return self.disabled_reason() is None

    def to_event(self, message, channel: str, now: datetime | None = None):
        """Telethon Message -> NewsEvent (None for empty/media-only posts)."""
        from locus.sources.news_stream import NewsEvent

        headline = normalize_headline(getattr(message, "message", None) or getattr(message, "text", "") or "")
        if not headline:
            return None
        now = now or datetime.now(timezone.utc)
        pub = getattr(message, "date", None)
        if isinstance(pub, datetime):
            pub = pub if pub.tzinfo else pub.replace(tzinfo=timezone.utc)
            latency = max(0, int((now - pub).total_seconds() * 1000))
        else:
            pub, latency = now, -1
        msg_id = getattr(message, "id", None)
        return NewsEvent(
            headline=headline,
            source="telegram",
            url=f"https://t.me/{channel}/{msg_id}" if channel and msg_id else "",
            received_at=now,
            published_at=pub,
            summary=(getattr(message, "message", "") or "")[:1000],
            latency_ms=latency,
            raw_data={"channel": channel, "message_id": msg_id, "via": "mtproto"},
        )

    async def stream(self, queue: asyncio.Queue):
        reason = self.disabled_reason()
        if reason:
            self.status = f"disabled: {reason}"
            log.info(f"[telegram-mtproto] disabled — {reason}")
            return
        try:
            TelegramClient, events = _load_telethon()
        except ImportError:
            self.status = "disabled: telethon not installed"
            log.warning("[telegram-mtproto] telethon not installed (pip install telethon) — disabled")
            return

        client = TelegramClient(str(session_file(self.session)), self.api_id, self.api_hash)
        self.client = client
        await client.connect()  # connect only: never start()/sign_in() here
        if not await client.is_user_authorized():
            self.status = "disabled: session not authorized"
            log.warning("[telegram-mtproto] session is not authorized — run "
                        "scripts/telegram_login.py; source disabled")
            await client.disconnect()
            return

        wanted = {c.lower() for c in self.channels}

        async def on_message(ev):
            chat = getattr(ev, "chat", None)
            if chat is None and hasattr(ev, "get_chat"):
                chat = await ev.get_chat()
            username = (getattr(chat, "username", "") or "").lower()
            if wanted and username and username not in wanted:
                return
            event = self.to_event(ev.message, username)
            if event is not None:
                await queue.put(event)

        client.add_event_handler(on_message, events.NewMessage(chats=self.channels))
        self.status = f"listening to {len(self.channels)} channels"
        log.info(f"[telegram-mtproto] listening to {', '.join('@' + c for c in self.channels)}")
        try:
            await client.run_until_disconnected()
        finally:
            self.status = "disconnected"
