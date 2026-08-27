"""进程内滑动窗口限流，保护免费档 API 的每分钟额度。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimit:
    """某个 provider 在一个时间窗内允许的最大调用次数。"""

    max_calls: int
    period_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_calls <= 10_000:
            raise ValueError("rate limit max_calls must be between 1 and 10000")
        if not 0.1 <= self.period_seconds <= 3_600:
            raise ValueError("rate limit period must be between 0.1 and 3600 seconds")


class RateLimiter:
    """按 key 独立计数；等待时不持有锁，避免堵住其它 provider。"""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def acquire(self, key: str, limit: RateLimit, *, timeout_seconds: float = 30.0) -> None:
        if not key or len(key) > 200:
            raise ValueError("rate limit key is invalid")
        if not 0.0 <= timeout_seconds <= 120.0:
            raise ValueError("rate limit timeout must be between 0 and 120 seconds")
        deadline = self._clock() + timeout_seconds
        while True:
            with self._lock:
                now = self._clock()
                window_start = now - limit.period_seconds
                stamps = [stamp for stamp in self._events.get(key, []) if stamp > window_start]
                if len(stamps) < limit.max_calls:
                    stamps.append(now)
                    self._events[key] = stamps
                    return
                wait = stamps[0] + limit.period_seconds - now
            remaining = deadline - self._clock()
            if wait <= 0:
                continue
            if remaining <= 0 or wait > remaining:
                raise TimeoutError(f"rate limit exceeded for {key}")
            self._sleep(min(wait, remaining, 0.25))
