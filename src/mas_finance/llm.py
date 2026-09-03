from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("DeepSeek base URL must be a fixed credential-free HTTPS URL")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.model):
            raise ValueError("DeepSeek model name is invalid")
        if not 0.1 <= self.timeout_seconds <= 120:
            raise ValueError("DeepSeek timeout must be between 0.1 and 120 seconds")
        if self.api_key is not None and (
            not self.api_key
            or len(self.api_key) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in self.api_key)
        ):
            raise ValueError("DeepSeek API key is invalid")

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
            model=os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro",
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS") or "45"),
        )


class BaseLLMClient:
    backend_name = "unknown"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        raise NotImplementedError


class DeepSeekChatClient(BaseLLMClient):
    backend_name = "deepseek"

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.api_key:
            raise ValueError("an API key is required for the remote LLM client")
        self.settings = settings
        self.transport = transport

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("LLM prompts must be strings")
        if len(system_prompt) + len(user_prompt) > 4_000_000:
            raise ValueError("LLM prompt exceeds the character limit")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 16_384:
            raise ValueError("LLM max_tokens must be an integer between 1 and 16384")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0 <= float(temperature) <= 2
        ):
            raise ValueError("LLM temperature must be finite and between 0 and 2")
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Evidence synthesis values deterministic JSON more than hidden
            # reasoning. V4 defaults to thinking, which can consume a short
            # output budget without producing final content.
            "thinking": {"type": "disabled"},
        }
        try:
            with (
                httpx.Client(
                    timeout=self.settings.timeout_seconds,
                    follow_redirects=False,
                    transport=self.transport,
                ) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                if response.status_code == 429 or response.status_code >= 500:
                    raise ConnectionError(f"DeepSeek transient HTTP status: {response.status_code}")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").casefold()
                if content_type and "json" not in content_type:
                    raise ValueError("DeepSeek response must use a JSON content type")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > 2_000_000:
                        raise ValueError("DeepSeek response exceeds the byte limit")
                    chunks.append(chunk)
        except httpx.TransportError as exc:
            raise ConnectionError("DeepSeek transport failed") from exc
        try:
            data = json.loads(b"".join(chunks), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("DeepSeek returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("DeepSeek response must be a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("DeepSeek response has no choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek response has no final content")
        if len(content) > 200_000:
            raise ValueError("DeepSeek response exceeds the character limit")
        return content.strip()


def build_llm_client(settings: LLMSettings | None = None) -> DeepSeekChatClient | None:
    settings = settings or LLMSettings.from_env()
    return DeepSeekChatClient(settings) if settings.api_key else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
