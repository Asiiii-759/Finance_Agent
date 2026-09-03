"""Provider-neutral open-web search converted into bounded evidence cards."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType, utc_now
from .harness import (
    RetryPolicy,
    Tool,
    ToolArgumentContract,
    ToolExecutionError,
    ToolResultKind,
    ToolSpec,
    function_tool,
)
from .rate_limit import RateLimit, RateLimiter


class WebSearchProvider(Protocol):
    provider_name: str

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class BraveWebSearchClient:
    provider_name = "brave"
    _endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit: RateLimit | None = None,
    ) -> None:
        if not api_key or len(api_key) > 4_096 or any(ord(item) < 32 or ord(item) == 127 for item in api_key):
            raise ValueError("Brave Search API key is invalid")
        if not 0.1 <= timeout_seconds <= 60:
            raise ValueError("Brave Search timeout must be between 0.1 and 60 seconds")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.rate_limiter = rate_limiter
        self.rate_limit = rate_limit or RateLimit(6)

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire("brave", self.rate_limit, timeout_seconds=self.timeout_seconds)
        query, count, freshness, domains = _search_arguments(payload)
        effective_query = query
        if domains:
            effective_query += " " + " OR ".join(f"site:{domain}" for domain in domains)
        params: dict[str, str | int | bool] = {
            "q": effective_query,
            "count": count,
            "extra_snippets": True,
            "safesearch": "moderate",
        }
        if freshness:
            params["freshness"] = freshness
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            response = client.get(
                self._endpoint,
                params=params,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f"Brave Search transient HTTP status: {response.status_code}")
        if response.status_code in {401, 403}:
            raise ToolExecutionError(
                "provider_access_denied",
                f"Brave Search denied the request with HTTP {response.status_code}",
                details={"http_status": response.status_code, "model_action": "report_unavailable"},
            )
        if 400 <= response.status_code < 500:
            raise ToolExecutionError(
                "provider_request_rejected",
                f"Brave Search rejected the request with HTTP {response.status_code}",
                details={"http_status": response.status_code, "model_action": "choose_alternative_tool"},
            )
        response.raise_for_status()
        if "json" not in response.headers.get("content-type", "").casefold():
            raise ValueError("Brave Search response must be JSON")
        if len(response.content) > 5_000_000:
            raise ValueError("Brave Search response exceeds the byte limit")
        value = response.json()
        if not isinstance(value, Mapping):
            raise ValueError("Brave Search response must be an object")
        web = value.get("web")
        raw_results = web.get("results") if isinstance(web, Mapping) else None
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, list):
            raise ValueError("Brave Search results must be a list")
        results: list[dict[str, Any]] = []
        for item in raw_results[:count]:
            if not isinstance(item, Mapping):
                raise ValueError("Brave Search result must be an object")
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                    "extra_snippets": item.get("extra_snippets") or [],
                    "published_at": item.get("page_age"),
                }
            )
        return {"query": query, "results": results, "retrieved_at": utc_now()}


class BochaWebSearchClient:
    provider_name = "bocha"
    _endpoint = "https://api.bochaai.com/v1/web-search"
    _freshness = {"pd": "oneDay", "pw": "oneWeek", "pm": "oneMonth", "py": "oneYear"}

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit: RateLimit | None = None,
    ) -> None:
        if not api_key or len(api_key) > 4_096 or any(ord(item) < 32 or ord(item) == 127 for item in api_key):
            raise ValueError("Bocha Search API key is invalid")
        if not 0.1 <= timeout_seconds <= 60:
            raise ValueError("Bocha Search timeout must be between 0.1 and 60 seconds")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.rate_limiter = rate_limiter
        self.rate_limit = rate_limit or RateLimit(6)

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire("bocha", self.rate_limit, timeout_seconds=self.timeout_seconds)
        query, count, freshness, domains = _search_arguments(payload)
        effective_query = query
        if domains:
            effective_query = " OR ".join(f"site:{domain}" for domain in domains) + f" {query}"
        request_body: dict[str, Any] = {
            "query": effective_query,
            "summary": False,
            "count": count,
        }
        if freshness:
            request_body["freshness"] = self._freshness[freshness]
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            response = client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f"Bocha Search transient HTTP status: {response.status_code}")
        if response.status_code in {401, 403}:
            raise ToolExecutionError(
                "provider_access_denied",
                f"Bocha Search denied the request with HTTP {response.status_code}",
                details={"http_status": response.status_code, "model_action": "report_unavailable"},
            )
        if 400 <= response.status_code < 500:
            raise ToolExecutionError(
                "provider_request_rejected",
                f"Bocha Search rejected the request with HTTP {response.status_code}",
                details={"http_status": response.status_code, "model_action": "choose_alternative_tool"},
            )
        response.raise_for_status()
        if "json" not in response.headers.get("content-type", "").casefold():
            raise ValueError("Bocha Search response must be JSON")
        if len(response.content) > 5_000_000:
            raise ValueError("Bocha Search response exceeds the byte limit")
        value = response.json()
        if not isinstance(value, Mapping):
            raise ValueError("Bocha Search response must be an object")
        code = value.get("code")
        if code in {401, 403}:
            raise ToolExecutionError(
                "provider_access_denied",
                f"Bocha Search response indicates access denial code {code}",
                details={"provider_code": code, "model_action": "report_unavailable"},
            )
        if code == 429 or isinstance(code, int) and code >= 500:
            raise ConnectionError(f"Bocha Search transient provider code: {code}")
        if code != 200:
            raise ToolExecutionError(
                "provider_request_rejected",
                f"Bocha Search response indicates provider code {code}",
                details={"provider_code": code, "model_action": "choose_alternative_tool"},
            )
        data = value.get("data")
        web_pages = data.get("webPages") if isinstance(data, Mapping) else None
        raw_results = web_pages.get("value") if isinstance(web_pages, Mapping) else None
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, list):
            raise ValueError("Bocha Search results must be a list")
        results: list[dict[str, Any]] = []
        for item in raw_results[:count]:
            if not isinstance(item, Mapping):
                raise ValueError("Bocha Search result must be an object")
            results.append(
                {
                    "title": item.get("name"),
                    "url": item.get("url"),
                    "description": item.get("snippet"),
                    "extra_snippets": [],
                    "published_at": item.get("datePublished"),
                }
            )
        return {"query": query, "results": results, "retrieved_at": utc_now()}


class WebSearchEvidenceAdapter:
    def __init__(self, provider: WebSearchProvider) -> None:
        self.provider = provider

    def search(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        query, count, freshness, domains = _search_arguments(payload)
        raw = self.provider.search_json(
            {
                "query": query,
                "count": count,
                "freshness": freshness,
                "domains": list(domains),
            }
        )
        results = raw.get("results")
        if not isinstance(results, list):
            raise ValueError("web search provider results must be a list")
        bundle = EvidenceBundle()
        seen_urls: set[str] = set()
        seen_content: set[str] = set()
        for rank, item in enumerate(results[:count], start=1):
            if not isinstance(item, Mapping):
                raise ValueError("web search result must be an object")
            title = _bounded_text(item.get("title"), "web search title", 1_000)
            locator = _public_result_url(item.get("url"))
            hostname = (urlsplit(locator).hostname or "").casefold().rstrip(".")
            if domains and not any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains):
                continue
            description = _bounded_text(item.get("description"), "web search description", 4_000)
            snippets = item.get("extra_snippets") or []
            if not isinstance(snippets, Sequence) or isinstance(snippets, (str, bytes)):
                raise ValueError("web search extra_snippets must be a list")
            content_parts = [description]
            content_parts.extend(_bounded_text(value, "web search snippet", 2_000) for value in snippets[:5])
            content = "\n".join(dict.fromkeys(part for part in content_parts if part))[:8_000]
            if not content:
                continue
            content_key = re.sub(r"\s+", " ", content).casefold()
            if locator in seen_urls or content_key in seen_content:
                continue
            seen_urls.add(locator)
            seen_content.add(content_key)
            published_at = item.get("published_at")
            source = SourceRef.create(
                source_type=SourceType.WEB,
                title=title,
                locator=locator,
                provider=self.provider.provider_name,
                published_at=str(published_at) if published_at else None,
                metadata={
                    "query": query,
                    "freshness": freshness,
                    "domain": hostname,
                    "content_basis": "search_result_snippet",
                    "rank": rank,
                    "quality_tier": _quality_tier(urlsplit(locator).hostname or ""),
                },
            )
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content=content,
                    confidence=0.55,
                    tags=("web_search", "search_result_snippet"),
                )
            )
        gaps = []
        if not bundle.evidence:
            gaps.append({"code": "web_search_no_results", "message": "Open-web search returned no usable results."})
        return {"bundle": bundle.to_dict(), "gaps": gaps}


def web_search_harness_tool(
    adapter: WebSearchEvidenceAdapter,
    *,
    name: str = "web.search",
) -> Tool:
    return function_tool(
        ToolSpec(
            name=name,
            description=(
                "在公开网络搜索当前金融信息。设置 query、可选 freshness 窗口（pd/pw/pm/py）、可选域名白名单，"
                "最多返回 10 个结果及可引用摘要。"
            ),
            capability="web.search",
            network_access=True,
            timeout_seconds=30,
            retry=RetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.25,
                retryable_exceptions=(
                    TimeoutError,
                    ConnectionError,
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ),
            ),
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"query"}),
                optional=frozenset({"count", "freshness", "domains"}),
            ),
            input_schema={
                "type": "object",
                "required": ["query"],
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 400},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10},
                    "freshness": {"type": "string", "enum": ["pd", "pw", "pm", "py"]},
                    "domains": {
                        "type": "array",
                        "maxItems": 10,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 4, "maxLength": 253},
                    },
                },
            },
        ),
        lambda arguments, _context: adapter.search(arguments),
    )


def _search_arguments(payload: Mapping[str, Any]) -> tuple[str, int, str | None, tuple[str, ...]]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 400 or len(query.split()) > 50:
        raise ValueError("web search query must contain 1-400 characters and at most 50 words")
    count = payload.get("count", 5)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
        raise ValueError("web search count must be an integer between 1 and 10")
    freshness = payload.get("freshness")
    if freshness is not None and freshness not in {"pd", "pw", "pm", "py"}:
        raise ValueError("web search freshness must be one of pd, pw, pm or py")
    raw_domains = payload.get("domains") or []
    if not isinstance(raw_domains, list) or len(raw_domains) > 10:
        raise ValueError("web search domains must be a list of at most 10 domains")
    domains = tuple(str(item).strip().casefold() for item in raw_domains)
    if any(not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,63}", item) for item in domains):
        raise ValueError("web search domains contain an invalid public domain")
    return query.strip(), count, str(freshness) if freshness else None, domains


def _public_result_url(value: Any) -> str:
    text = _bounded_text(value, "web search URL", 4_000)
    parsed = urlsplit(text)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,63}", parsed.hostname.casefold())
    ):
        raise ValueError("web search result URL is invalid")
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in {"gclid", "fbclid"}
    ]
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path or "/", urlencode(query), ""))


def _bounded_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} is invalid")
    return value.strip()


def _quality_tier(hostname: str) -> str:
    domain = hostname.casefold().rstrip(".")
    if domain.endswith((".gov", ".gov.cn", ".gov.uk", ".europa.eu")) or domain in {
        "sec.gov",
        "federalreserve.gov",
    }:
        return "public_authority"
    return "open_web"
