from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from redis import Redis


@dataclass
class RedisQueueManager:
    redis_url: str
    queue_name: str

    def connection(self) -> Redis:
        return Redis.from_url(self.redis_url)

    def enqueue(self, payload: dict[str, Any]) -> None:
        self.connection().rpush(self.queue_name, json.dumps(payload, ensure_ascii=False))

    def dequeue(self, timeout_seconds: int = 5) -> dict[str, Any] | None:
        result = cast(
            tuple[bytes, bytes] | None,
            self.connection().blpop([self.queue_name], timeout=timeout_seconds),
        )
        if not result:
            return None
        _, raw_payload = result
        return json.loads(raw_payload.decode("utf-8"))
