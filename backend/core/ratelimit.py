"""In-process rate limiting.

A single-process research tool does not need Redis; it does need a ceiling on
participant creation and on scoring endpoints so a script cannot fill the tables
or spend the afternoon hashing pairs. Limits are per client IP per scope and are
configured through environment variables, e.g.::

    ACCOUNTANT_RL_PARTICIPANTS=30/60      # 30 requests per 60 seconds
    ACCOUNTANT_RL_SCORE=240/60
    ACCOUNTANT_RL_FEEDBACK=120/60

`0/60` disables a scope. Behind a reverse proxy set the limits there too; the
client address seen here is the proxy.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class SlidingWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(window - (now - hits[0])) + 1
                return False, max(1, retry)
            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


LIMITER = SlidingWindow()


def parse_spec(spec: str, fallback: tuple[int, float]) -> tuple[int, float]:
    try:
        limit, window = spec.split("/", 1)
        return max(0, int(limit)), max(1.0, float(window))
    except (ValueError, AttributeError):
        return fallback


def limit(scope: str, env_var: str, default: str):
    """Build a FastAPI dependency enforcing a per-IP sliding window."""

    def dependency(request: Request) -> None:
        limit_n, window = parse_spec(os.environ.get(env_var, default), (0, 60.0))
        if limit_n <= 0:
            return
        client = request.client.host if request.client else "unknown"
        allowed, retry = LIMITER.allow(f"{scope}:{client}", limit_n, window)
        if not allowed:
            raise HTTPException(status_code=429,
                                detail=f"rate limit exceeded for {scope}",
                                headers={"Retry-After": str(retry)})

    return dependency
