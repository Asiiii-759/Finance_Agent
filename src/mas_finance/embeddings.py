"""Embedding boundary used by hybrid document retrieval."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

import httpx


class EmbeddingProvider(Protocol):
    """Deployment-injected embedding model with an explicit network boundary."""

    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def network_access(self) -> bool: ...

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class HTTPEmbeddingClient:
    """Bounded client for an OpenAI-compatible ``/embeddings`` endpoint.

    The endpoint and model are deployment configuration, never model-generated
    arguments. One method call performs one HTTP request so harness accounting
    remains aligned with the actual network boundary.
    """

    endpoint: str
    model_name: str
    api_key: str | None = field(default=None, repr=False, compare=False)
    timeout_seconds: float = 30.0
    max_inputs: int = 512
    max_total_characters: int = 1_000_000
    max_response_bytes: int = 20_000_000
    allow_insecure_http: bool = False
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)
    backend_name: str = field(default="http_embedding", init=False)
    network_access: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        loopback_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        allowed_scheme = parsed.scheme == "https" or loopback_http or (
            parsed.scheme == "http" and self.allow_insecure_http
        )
        if (
            not allowed_scheme
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "embedding endpoint must be a fixed credential-free HTTPS or loopback HTTP URL"
            )
        if not self.model_name.strip() or len(self.model_name) > 300:
            raise ValueError("embedding model name is invalid")
        object.__setattr__(self, "model_name", self.model_name.strip())
        if not 0.1 <= self.timeout_seconds <= 120:
            raise ValueError("embedding timeout must be between 0.1 and 120 seconds")
        if not 1 <= self.max_inputs <= 2_048:
            raise ValueError("embedding input count limit is invalid")
        if not 1_000 <= self.max_total_characters <= 10_000_000:
            raise ValueError("embedding character limit is invalid")
        if not 1_024 <= self.max_response_bytes <= 100_000_000:
            raise ValueError("embedding response limit is invalid")
        if self.api_key is not None and (
            not self.api_key
            or len(self.api_key) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in self.api_key)
        ):
            raise ValueError("embedding API key is invalid")

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not texts or len(texts) > self.max_inputs:
            raise ValueError(f"embedding input must contain between 1 and {self.max_inputs} texts")
        normalized = tuple(text.strip() if isinstance(text, str) else "" for text in texts)
        if any(not text or len(text) > 32_000 for text in normalized):
            raise ValueError("embedding texts must be non-empty strings of at most 32000 characters")
        if sum(len(text) for text in normalized) > self.max_total_characters:
            raise ValueError("embedding input exceeds the total character limit")

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with (
            httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers=headers,
            ) as client,
            client.stream(
                "POST",
                self.endpoint,
                json={"model": self.model_name, "input": list(normalized)},
            ) as response,
        ):
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").casefold()
            if content_type and "json" not in content_type:
                raise ValueError("embedding response must use a JSON content type")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise ValueError("embedding response exceeds the byte limit")
                chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("embedding endpoint returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("embedding response must contain a data list")
        data = payload["data"]
        if len(data) != len(normalized):
            raise ValueError("embedding response count does not match the request")

        indexed: dict[int, tuple[float, ...]] = {}
        dimension: int | None = None
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("embedding response item must be an object")
            index = item.get("index")
            raw_vector = item.get("embedding")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(normalized):
                raise ValueError("embedding response index is invalid")
            if index in indexed or not isinstance(raw_vector, list):
                raise ValueError("embedding response vector is invalid")
            vector = tuple(
                float(value)
                for value in raw_vector
                if not isinstance(value, bool) and isinstance(value, (int, float))
            )
            if len(vector) != len(raw_vector) or not 2 <= len(vector) <= 65_536:
                raise ValueError("embedding response vector dimension is invalid")
            if any(not math.isfinite(value) for value in vector) or not any(value != 0 for value in vector):
                raise ValueError("embedding response vector must contain finite non-zero data")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("embedding response vectors have inconsistent dimensions")
            indexed[index] = vector
        if set(indexed) != set(range(len(normalized))):
            raise ValueError("embedding response indices are incomplete")
        return tuple(indexed[index] for index in range(len(normalized)))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
