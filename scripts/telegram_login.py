#!/usr/bin/env python3
"""One-time Telegram login for the Locus news listener.

Run this YOURSELF, once, in a terminal:

    python scripts/telegram_login.py            # create/authorize the session
    python scripts/telegram_login.py --join     # ...and join the news channels

It reads TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_SESSION /
TELEGRAM_NEWS_CHANNELS from .env, asks for your phone number, the login code
Telegram sends you, and your 2FA password if you have one (typed into this
terminal only — nothing is stored except the Telethon session file).

The session file (default telegram_news.session in the repo root) is a full
login to your Telegram account: keep it private, never commit it (it is in
.gitignore), and delete it + terminate the session in Telegram -> Settings ->
Devices to revoke access. A separate/secondary Telegram account is recommended.

The pipeline itself never logs in: without this file the source stays off.
`--join` subscribes the account to the configured channels; MTProto only
delivers new posts of channels the account has joined.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locus import config  # noqa: E402
from locus.sources.telegram_mtproto import session_file  # noqa: E402


async def main(join: bool) -> int:
    try:
        from telethon import TelegramClient
        from telethon.tl.functions.channels import JoinChannelRequest
    except ImportError:
        print("telethon is not installed: pip install telethon")
        return 1
    if not str(config.TELEGRAM_API_ID).isdigit() or not config.TELEGRAM_API_HASH:
        print("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first (https://my.telegram.org).")
        return 1
    path = session_file(config.TELEGRAM_SESSION)
    client = TelegramClient(str(path), int(config.TELEGRAM_API_ID), config.TELEGRAM_API_HASH)
    await client.start()  # interactive: phone -> code -> optional 2FA password
    me = await client.get_me()
    print(f"Authorized as {getattr(me, 'username', None) or me.id}. Session: {path}")
    for name in config.TELEGRAM_NEWS_CHANNELS:
        try:
            entity = await client.get_entity(name)
            if join:
                await client(JoinChannelRequest(entity))
                print(f"  joined  @{name}")
            else:
                print(f"  ok      @{name}  ({getattr(entity, 'title', '')})")
        except Exception as e:
            print(f"  FAILED  @{name}: {type(e).__name__}: {e}")
    if not join:
        print("Tip: run again with --join so new posts are delivered to the listener.")
    await client.disconnect()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--join", action="store_true", help="join the TELEGRAM_NEWS_CHANNELS")
    sys.exit(asyncio.run(main(ap.parse_args().join)))
