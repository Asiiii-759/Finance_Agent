from __future__ import annotations

import unittest

from mas_finance.rate_limit import RateLimit, RateLimiter


class RateLimiterTests(unittest.TestCase):
    def test_allows_calls_inside_the_window_then_times_out(self) -> None:
        clock = {"now": 0.0}
        sleeps: list[float] = []
        limiter = RateLimiter(clock=lambda: clock["now"], sleeper=lambda seconds: sleeps.append(seconds))
        limit = RateLimit(max_calls=2, period_seconds=10.0)
        limiter.acquire("bocha", limit, timeout_seconds=1.0)
        limiter.acquire("bocha", limit, timeout_seconds=1.0)
        with self.assertRaisesRegex(TimeoutError, "rate limit exceeded"):
            limiter.acquire("bocha", limit, timeout_seconds=0.0)
        self.assertEqual(sleeps, [])

    def test_isolates_keys(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit(max_calls=1, period_seconds=60.0)
        limiter.acquire("fred", limit, timeout_seconds=0.0)
        limiter.acquire("bocha", limit, timeout_seconds=0.0)
