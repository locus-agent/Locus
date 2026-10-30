"""Per-source news latency metrics.

Two legs per news event:
  publish -> receive   (NewsEvent.latency_ms, set by the source)
  receive -> classify  (first classification of the event finished)
Kept in bounded in-memory windows; summary() feeds status.json and the CLI.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone

_WINDOW = 500
_lock = threading.Lock()
_pub_to_recv: dict[str, deque] = defaultdict(lambda: deque(maxlen=_WINDOW))
_recv_to_cls: dict[str, deque] = defaultdict(lambda: deque(maxlen=_WINDOW))


def record_receive(source: str, latency_ms: int) -> None:
    if latency_ms is None or latency_ms < 0:
        return
    with _lock:
        _pub_to_recv[source].append(int(latency_ms))


def record_classified(source: str, received_at: datetime, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    ms = int((now - received_at).total_seconds() * 1000)
    if ms < 0:
        return
    with _lock:
        _recv_to_cls[source].append(ms)


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summary() -> dict:
    with _lock:
        sources = set(_pub_to_recv) | set(_recv_to_cls)
        out = {}
        for s in sorted(sources):
            a, b = list(_pub_to_recv[s]), list(_recv_to_cls[s])
            out[s] = {
                "n_received": len(a),
                "publish_to_receive_ms_p50": _pct(a, 0.5),
                "publish_to_receive_ms_p90": _pct(a, 0.9),
                "n_classified": len(b),
                "receive_to_classify_ms_p50": _pct(b, 0.5),
                "receive_to_classify_ms_p90": _pct(b, 0.9),
            }
        return out


def reset() -> None:
    with _lock:
        _pub_to_recv.clear()
        _recv_to_cls.clear()
