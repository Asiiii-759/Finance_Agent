"""Adapters that convert retrieval-provider payloads into evidence contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from .harness import (
    RetryPolicy,
    Tool,
    ToolArgumentContract,
    ToolResultKind,
    ToolSpec,
    function_tool,
)


class RAGClient(Protocol):
    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HTTPJSONRAGClient:
    """Bounded client for a deployment-controlled canonical JSON search gateway.

    The endpoint is fixed when the service starts; it is never taken from an
    Agent request.  A gateway can front internal RAG, licensed news, or a web
    search provider while keeping provider-specific schemas out of the Agent.
    """

    endpoint: str
    api_key: str | None = field(default=None, repr=False, compare=False)
    timeout_seconds: float = 30.0
    max_response_bytes: int = 5_000_000
    allow_insecure_http: bool = False
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        allowed_schemes = {"https"} | ({"http"} if self.allow_insecure_http else set())
        if (
            parsed.scheme not in allowed_schemes
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("RAG endpoint must be a fixed credential-free HTTPS URL")
        if not 0.1 <= self.timeout_seconds <= 120:
            raise ValueError("RAG timeout must be between 0.1 and 120 seconds")
        if not 1_024 <= self.max_response_bytes <= 20_000_000:
            raise ValueError("RAG response limit is outside the supported range")
        if self.api_key is not None and (
            not self.api_key
            or len(self.api_key) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in self.api_key)
        ):
            raise ValueError("RAG API key is invalid")

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
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
            client.stream("POST", self.endpoint, json=dict(payload)) as response,
        ):
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").casefold()
            if content_type and "json" not in content_type:
                raise ValueError("RAG gateway response must use a JSON content type")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise ValueError("RAG gateway response exceeds the byte limit")
                chunks.append(chunk)
        try:
            value = json.loads(b"".join(chunks), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("RAG gateway returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("RAG gateway response must be a JSON object")
        return value


@dataclass(frozen=True)
class RetrievalSource:
    """A trusted, deployment-time RAG source exposed through the same harness contract."""

    name: str
    client: RAGClient
    provider: str
    network_access: bool = False
    description: str = "搜索已授权的金融研究语料库。"
    fixed_filters: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,63}", self.name):
            raise ValueError("retrieval source name must be a bounded namespaced tool name")
        if not self.provider.strip() or len(self.provider) > 200:
            raise ValueError("retrieval source provider is invalid")
        if not self.description.strip() or len(self.description) > 500:
            raise ValueError("retrieval source description is invalid")
        normalized_filters = _validated_filters(self.fixed_filters)
        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "fixed_filters", MappingProxyType(normalized_filters))

    def build_tool(self) -> Tool:
        return retrieval_harness_tool(
            RetrievalEvidenceAdapter(
                self.client,
                provider=self.provider,
                fixed_filters=self.fixed_filters,
            ),
            name=self.name,
            network_access=self.network_access,
            description=(
                f"{self.description} 只有当用户要求跨多个文档比较或综合时，才设置 diversify_documents=true。"
            ),
        )


@dataclass(frozen=True)
class RetrievalBatch:
    bundle: EvidenceBundle
    trace: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"bundle": self.bundle.to_dict(), "trace": dict(self.trace)}


class RetrievalEvidenceAdapter:
    """Anti-corruption layer between a retrieval backend and agent contracts."""

    def __init__(
        self,
        client: RAGClient,
        *,
        provider: str = "internal_corpus",
        fixed_filters: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.provider = provider
        self.fixed_filters = _validated_filters(fixed_filters or {})

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        search_mode: str = "rrf",
        rerank: bool = False,
        diversify_documents: bool = False,
    ) -> RetrievalBatch:
        if not isinstance(query, str) or not query.strip() or len(query) > 8_000:
            raise ValueError("retrieval query must be a non-empty bounded string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("retrieval top_k must be an integer between 1 and 20")
        request_filters = _validated_filters(filters or {})
        if search_mode not in {"lexical", "vector", "hybrid", "rrf"}:
            raise ValueError("retrieval search_mode must be lexical, vector, hybrid, or rrf")
        if not isinstance(rerank, bool):
            raise ValueError("retrieval rerank must be a boolean")
        if not isinstance(diversify_documents, bool):
            raise ValueError("retrieval diversify_documents must be a boolean")
        response = self.client.search_json(
            {
                "query": query,
                "top_k": top_k,
                # Deployment filters win over model/request filters.  They are
                # the ACL boundary and must never be weakened by an Agent call.
                "filters": {**request_filters, **self.fixed_filters},
                "search_mode": search_mode,
                "rerank": rerank,
                "diversify_documents": diversify_documents,
            }
        )
        if not isinstance(response, Mapping):
            raise ValueError("retrieval response must be an object")
        chunks = response.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("retrieval response must contain a chunks list")
        if len(chunks) > top_k:
            raise ValueError("retrieval provider returned more chunks than requested")
        trace = response.get("trace") or {}
        if not isinstance(trace, Mapping):
            raise ValueError("retrieval trace must be an object")
        trace_mode = trace.get("search_mode")
        if trace_mode is not None and not isinstance(trace_mode, str):
            raise ValueError("retrieval trace search_mode must be a string")
        effective_trace = {**dict(trace), "document_diversification": diversify_documents}

        bundle = EvidenceBundle()
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("retrieval chunk must be an object")
            metadata = chunk.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise ValueError("retrieval chunk metadata must be an object")
            chunk_id = chunk.get("id")
            if not isinstance(chunk_id, str) or not chunk_id.strip() or len(chunk_id) > 256:
                raise ValueError("retrieval chunk id must be a non-empty bounded string")
            raw_content = chunk.get("content")
            if not isinstance(raw_content, str):
                raise ValueError("retrieval chunk content must be a string")
            content = raw_content.strip()
            if not content:
                continue
            rank = _optional_rank(chunk.get("rank"))
            score = _optional_score(chunk.get("score"))
            scores = chunk.get("scores") or {}
            if not isinstance(scores, Mapping):
                raise ValueError("retrieval chunk scores must be an object")
            _validated_json_object(scores, field_name="retrieval chunk scores", max_characters=20_000)
            page = _optional_positive_int(metadata.get("source_page"), field_name="source_page")
            locator = _document_locator(metadata, chunk_id)
            source = SourceRef.create(
                source_type=SourceType.DOCUMENT,
                title=_first_text(
                    metadata,
                    "paragraph_title",
                    "document_title",
                    "title",
                    "file_name",
                    default="Untitled document",
                ),
                locator=locator,
                provider=self.provider,
                as_of=_optional_text(metadata.get("document_date") or metadata.get("publish_date")),
                published_at=_optional_text(metadata.get("publish_date")),
                metadata={
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": score,
                    "scores": dict(scores),
                    "kb_name": metadata.get("kb_name"),
                    "file_name": metadata.get("file_name"),
                    "source_url": _optional_text(metadata.get("source_url") or metadata.get("url")),
                    "publisher": _optional_text(metadata.get("publisher")),
                    "document_id": _optional_text(metadata.get("document_id")),
                    "corpus_record_id": _optional_text(metadata.get("corpus_record_id")),
                    "retrieval_trace": _safe_trace_metadata(effective_trace),
                },
            )
            span_start, span_end = _optional_span(metadata)
            evidence = Evidence.create(
                source=source,
                content=content,
                entity=_optional_text(metadata.get("company") or metadata.get("organization")),
                # Retrieval scores (especially RRF) are not calibrated
                # probabilities and must not be presented as confidence.
                confidence=_score_to_confidence(metadata.get("extraction_confidence")),
                page=page,
                span_start=span_start,
                span_end=span_end,
                # Retrieval mode belongs in provenance trace, not semantic
                # evidence tags. The same chunk can be returned by lexical and
                # hybrid tools in one run and must remain idempotent evidence.
                tags=("retrieved",),
            )
            bundle.add_evidence(evidence)
        return RetrievalBatch(bundle=bundle, trace=effective_trace)


def retrieval_harness_tool(
    adapter: RetrievalEvidenceAdapter,
    *,
    name: str = "corpus.search",
    network_access: bool = False,
    fixed_search_mode: str | None = None,
    description: str = (
        "搜索已授权的金融研究语料库并返回可引用证据。只有当用户要求跨多个文档比较或综合时，"
        "才设置 diversify_documents=true。"
    ),
) -> Tool:
    if fixed_search_mode is not None and fixed_search_mode not in {"lexical", "vector", "hybrid", "rrf"}:
        raise ValueError("fixed retrieval search mode is invalid")

    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        filters = arguments.get("filters") or {}
        search_mode = fixed_search_mode or arguments.get("search_mode", "rrf")
        rerank = False if fixed_search_mode is not None else arguments.get("rerank", False)
        diversify_documents = arguments.get("diversify_documents", False)
        if not isinstance(query, str):
            raise ValueError("retrieval query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("retrieval top_k must be an integer")
        if not isinstance(filters, Mapping):
            raise ValueError("retrieval filters must be an object")
        if not isinstance(search_mode, str):
            raise ValueError("retrieval search_mode must be a string")
        if not isinstance(rerank, bool):
            raise ValueError("retrieval rerank must be a boolean")
        if not isinstance(diversify_documents, bool):
            raise ValueError("retrieval diversify_documents must be a boolean")
        batch = adapter.search(
            query,
            top_k=top_k,
            filters=dict(filters),
            search_mode=search_mode,
            rerank=rerank,
            diversify_documents=diversify_documents,
        )
        return batch.to_dict()

    return function_tool(
        ToolSpec(
            name=name,
            description=description,
            capability="document.search",
            network_access=network_access,
            timeout_seconds=120,
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
                optional=frozenset({"top_k", "filters", "diversify_documents"})
                if fixed_search_mode is not None
                else frozenset({"top_k", "filters", "search_mode", "rerank", "diversify_documents"}),
            ),
            input_schema={
                "type": "object",
                "required": ["query"],
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    "filters": {"type": "object", "maxProperties": 50},
                    "diversify_documents": {"type": "boolean"},
                    **(
                        {}
                        if fixed_search_mode is not None
                        else {
                            "search_mode": {
                                "type": "string",
                                "enum": ["lexical", "vector", "hybrid", "rrf"],
                            },
                            "rerank": {"type": "boolean"},
                        }
                    ),
                },
            },
        ),
        invoke,
    )


def _document_locator(metadata: Mapping[str, Any], chunk_id: str) -> str:
    source_locator = _first_text(
        metadata,
        "file_name",
        "source_url",
        "url",
        "source_uri",
        default="unknown",
    )
    parts = [source_locator]
    page = _optional_positive_int(metadata.get("source_page"), field_name="source_page")
    if page is not None:
        parts.append(f"page={page}")
    parts.append(f"chunk={chunk_id}")
    return "#".join(parts)


def _score_to_confidence(score: Any) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.5
    return max(0.0, min(1.0, float(score)))


def _optional_span(metadata: Mapping[str, Any]) -> tuple[int | None, int | None]:
    start = _optional_int(metadata.get("global_start"))
    end = _optional_int(metadata.get("global_end"))
    if start is None or end is None or start < 0 or end < start:
        return None, None
    return start, end


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("retrieval text metadata must be a string")
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return None


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    parsed = _optional_int(value)
    if parsed is None or parsed < 1:
        raise ValueError(f"retrieval metadata {field_name} must be a positive integer")
    return parsed


def _optional_rank(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100_000:
        raise ValueError("retrieval chunk rank must be a positive integer")
    return value


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("retrieval chunk score must be finite numeric data")
    return float(value)


def _first_text(metadata: Mapping[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"retrieval metadata {key} must be a string")
        return value
    return default


def _validated_filters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) > 50:
        raise ValueError("retrieval filters must be a bounded object")
    normalized = dict(value)
    if any(
        not isinstance(key, str)
        or not key
        or len(key) > 100
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
        for key in normalized
    ):
        raise ValueError("retrieval filter keys are invalid")
    try:
        encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("retrieval filters must be finite JSON") from exc
    if len(encoded) > 20_000:
        raise ValueError("retrieval filters exceed the size limit")
    return normalized


def _validated_json_object(value: Mapping[str, Any], *, field_name: str, max_characters: int) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite JSON") from exc
    if len(encoded) > max_characters:
        raise ValueError(f"{field_name} exceeds the size limit")


def _safe_trace_metadata(trace: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "backend",
        "contract_version",
        "index_version",
        "request_id",
        "search_mode",
        "selected_chunk_count",
        "trace_id",
        "document_diversification",
        "embedding_backend",
        "embedding_batch_count",
        "embedding_model",
        "fusion",
        "minimum_vector_similarity",
        "vector_candidate_count",
        "vector_candidate_count_before_threshold",
    }
    result: dict[str, Any] = {}
    for key in allowed:
        value = trace.get(key)
        if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 500:
            result[key] = value
    # SourceRef performs the final finite-JSON validation.
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
