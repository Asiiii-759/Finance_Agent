"""Stable domain contracts shared by agents, tools, storage and APIs.

The orchestration layer must exchange these records instead of provider-specific
objects.  All records are JSON serialisable and intentionally contain provenance.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

_MAX_BUNDLE_SOURCES = 5_000
_MAX_BUNDLE_EVIDENCE = 5_000
_MAX_BUNDLE_CLAIMS = 1_000
_MAX_BUNDLE_CONTENT_CHARACTERS = 5_000_000


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return f"{prefix}_{sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


class SourceType(StrEnum):
    DOCUMENT = "document"
    WEB = "web"
    MARKET_DATA = "market_data"
    MACRO_DATA = "macro_data"
    REGULATORY_FILING = "regulatory_filing"
    CALCULATION = "calculation"
    USER_INPUT = "user_input"


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    source_type: SourceType
    title: str
    locator: str
    provider: str
    retrieved_at: str = field(default_factory=utc_now)
    as_of: str | None = None
    published_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source_type: SourceType,
        title: str,
        locator: str,
        provider: str,
        as_of: str | None = None,
        published_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SourceRef:
        normalized_title = title.strip()
        normalized_locator = locator.strip()
        normalized_provider = provider.strip()
        identity = {
            "source_type": source_type.value,
            "locator": normalized_locator,
            "provider": normalized_provider,
            "as_of": as_of,
        }
        return cls(
            source_id=stable_id("src", identity),
            source_type=source_type,
            title=normalized_title,
            locator=normalized_locator,
            provider=normalized_provider,
            as_of=as_of,
            published_at=published_at,
            metadata=dict(metadata or {}),
        )

    def __post_init__(self) -> None:
        if not self.title or not self.locator or not self.provider:
            raise ValueError("source title, locator and provider are required")
        if len(self.title) > 500 or len(self.locator) > 2_048 or len(self.provider) > 200:
            raise ValueError("source fields exceed length limits")
        if any(value is not None and len(value) > 100 for value in (self.retrieved_at, self.as_of, self.published_at)):
            raise ValueError("source timestamps exceed length limits")
        try:
            datetime.fromisoformat(self.retrieved_at)
        except ValueError as exc:
            raise ValueError("source retrieved_at must be an ISO-8601 timestamp") from exc
        _validate_json_size(self.metadata, field_name="source metadata", max_characters=50_000)
        expected_id = stable_id(
            "src",
            {
                "source_type": self.source_type.value,
                "locator": self.locator,
                "provider": self.provider,
                "as_of": self.as_of,
            },
        )
        if self.source_id != expected_id:
            raise ValueError("source_id does not match source identity")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_type"] = self.source_type.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceRef:
        return cls(
            source_id=str(value["source_id"]),
            source_type=SourceType(str(value["source_type"])),
            title=str(value["title"]),
            locator=str(value["locator"]),
            provider=str(value["provider"]),
            retrieved_at=str(value.get("retrieved_at") or utc_now()),
            as_of=str(value["as_of"]) if value.get("as_of") is not None else None,
            published_at=str(value["published_at"]) if value.get("published_at") is not None else None,
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: SourceRef
    content: str
    entity: str | None = None
    field_name: str | None = None
    value: str | int | float | bool | None = None
    unit: str | None = None
    period: str | None = None
    confidence: float = 1.0
    page: int | None = None
    span_start: int | None = None
    span_end: int | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        source: SourceRef,
        content: str,
        entity: str | None = None,
        field_name: str | None = None,
        value: str | int | float | bool | None = None,
        unit: str | None = None,
        period: str | None = None,
        confidence: float = 1.0,
        page: int | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        tags: tuple[str, ...] = (),
    ) -> Evidence:
        normalized_content = content.strip()
        identity = {
            "source_id": source.source_id,
            "content": normalized_content,
            "entity": entity,
            "field_name": field_name,
            "period": period,
            "page": page,
            "span_start": span_start,
            "span_end": span_end,
        }
        return cls(
            evidence_id=stable_id("ev", identity),
            source=source,
            content=normalized_content,
            entity=entity,
            field_name=field_name,
            value=value,
            unit=unit,
            period=period,
            confidence=confidence,
            page=page,
            span_start=span_start,
            span_end=span_end,
            tags=tuple(tags),
        )

    def __post_init__(self) -> None:
        if not self.content and self.value is None:
            raise ValueError("evidence needs content or a structured value")
        if len(self.content) > 500_000:
            raise ValueError("evidence content exceeds length limit")
        if self.entity is not None and len(self.entity) > 200:
            raise ValueError("evidence entity exceeds length limit")
        if self.field_name is not None and len(self.field_name) > 100:
            raise ValueError("evidence field name exceeds length limit")
        if self.unit is not None and len(self.unit) > 50:
            raise ValueError("evidence unit exceeds length limit")
        if self.period is not None and len(self.period) > 100:
            raise ValueError("evidence period exceeds length limit")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("evidence numeric value must be finite")
        if len(self.tags) > 32 or any(not item or len(item) > 100 for item in self.tags):
            raise ValueError("evidence tags exceed count or length limits")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("span_start and span_end must be provided together")
        if (
            self.span_start is not None
            and self.span_end is not None
            and (self.span_start < 0 or self.span_end < self.span_start)
        ):
            raise ValueError("invalid evidence span")
        expected_id = stable_id(
            "ev",
            {
                "source_id": self.source.source_id,
                "content": self.content,
                "entity": self.entity,
                "field_name": self.field_name,
                "period": self.period,
                "page": self.page,
                "span_start": self.span_start,
                "span_end": self.span_end,
            },
        )
        if self.evidence_id != expected_id:
            raise ValueError("evidence_id does not match evidence identity")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.to_dict()
        value["tags"] = list(self.tags)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Evidence:
        return cls(
            evidence_id=str(value["evidence_id"]),
            source=SourceRef.from_dict(value["source"]),
            content=str(value.get("content") or ""),
            entity=str(value["entity"]) if value.get("entity") is not None else None,
            field_name=str(value["field_name"]) if value.get("field_name") is not None else None,
            value=value.get("value"),
            unit=str(value["unit"]) if value.get("unit") is not None else None,
            period=str(value["period"]) if value.get("period") is not None else None,
            confidence=float(value.get("confidence", 1.0)),
            page=int(value["page"]) if value.get("page") is not None else None,
            span_start=int(value["span_start"]) if value.get("span_start") is not None else None,
            span_end=int(value["span_end"]) if value.get("span_end") is not None else None,
            tags=tuple(str(item) for item in value.get("tags") or ()),
        )


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"
    CONFLICTED = "conflicted"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    status: ClaimStatus
    evidence_ids: tuple[str, ...] = ()
    caveat: str | None = None

    @classmethod
    def create(
        cls,
        *,
        text: str,
        status: ClaimStatus,
        evidence_ids: tuple[str, ...] = (),
        caveat: str | None = None,
    ) -> Claim:
        normalized_text = text.strip()
        identity = {
            "text": normalized_text,
            "status": status.value,
            "evidence_ids": sorted(set(evidence_ids)),
            "caveat": caveat,
        }
        return cls(
            claim_id=stable_id("claim", identity),
            text=normalized_text,
            status=status,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            caveat=caveat,
        )

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("claim text is required")
        if len(self.text) > 8_000 or (self.caveat is not None and len(self.caveat) > 2_000):
            raise ValueError("claim fields exceed length limits")
        if len(self.evidence_ids) > 100:
            raise ValueError("claim cites too many evidence items")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("claim contains duplicate evidence ids")
        if self.status == ClaimStatus.SUPPORTED and not self.evidence_ids:
            raise ValueError("a supported claim must cite at least one evidence item")
        if self.status in {ClaimStatus.INFERRED, ClaimStatus.UNSUPPORTED, ClaimStatus.CONFLICTED} and not self.caveat:
            raise ValueError(f"{self.status.value} claims must state a caveat")
        expected_id = stable_id(
            "claim",
            {
                "text": self.text,
                "status": self.status.value,
                "evidence_ids": sorted(set(self.evidence_ids)),
                "caveat": self.caveat,
            },
        )
        if self.claim_id != expected_id:
            raise ValueError("claim_id does not match claim identity")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["evidence_ids"] = list(self.evidence_ids)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Claim:
        return cls(
            claim_id=str(value["claim_id"]),
            text=str(value["text"]),
            status=ClaimStatus(str(value["status"])),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids") or ()),
            caveat=str(value["caveat"]) if value.get("caveat") is not None else None,
        )


@dataclass
class EvidenceBundle:
    """Run-scoped evidence ledger with referential-integrity checks."""

    sources: dict[str, SourceRef] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    _content_characters: int = field(default=0, init=False, repr=False)

    def add_evidence(self, item: Evidence) -> None:
        existing_source = self.sources.get(item.source.source_id)
        if existing_source is not None and _source_semantic_identity(existing_source) != _source_semantic_identity(
            item.source
        ):
            raise ValueError(f"conflicting source id: {item.source.source_id}")
        existing_evidence = self.evidence.get(item.evidence_id)
        if existing_evidence is not None and _evidence_semantic_identity(
            existing_evidence
        ) != _evidence_semantic_identity(item):
            raise ValueError(f"conflicting evidence id: {item.evidence_id}")
        if existing_evidence is None:
            if len(self.evidence) >= _MAX_BUNDLE_EVIDENCE:
                raise ValueError("evidence bundle exceeds evidence count limit")
            if self._content_characters + len(item.content) > _MAX_BUNDLE_CONTENT_CHARACTERS:
                raise ValueError("evidence bundle exceeds content size limit")
        if existing_source is None and len(self.sources) >= _MAX_BUNDLE_SOURCES:
            raise ValueError("evidence bundle exceeds source count limit")
        self.sources.setdefault(item.source.source_id, item.source)
        if existing_evidence is None:
            self.evidence[item.evidence_id] = item
            self._content_characters += len(item.content)

    def add_claim(self, claim: Claim) -> None:
        missing = set(claim.evidence_ids).difference(self.evidence)
        if missing:
            raise ValueError(f"claim references unknown evidence: {sorted(missing)}")
        if claim.claim_id not in self.claims and len(self.claims) >= _MAX_BUNDLE_CLAIMS:
            raise ValueError("evidence bundle exceeds claim count limit")
        self.claims[claim.claim_id] = claim

    def merge(self, other: EvidenceBundle) -> None:
        for source_id, source in other.sources.items():
            existing_source = self.sources.get(source_id)
            if existing_source is not None and _source_semantic_identity(existing_source) != _source_semantic_identity(
                source
            ):
                raise ValueError(f"conflicting source id: {source_id}")
            if existing_source is None and len(self.sources) >= _MAX_BUNDLE_SOURCES:
                raise ValueError("evidence bundle exceeds source count limit")
            self.sources.setdefault(source_id, source)
        for item in other.evidence.values():
            self.add_evidence(item)
        for claim in other.claims.values():
            self.add_claim(claim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [item.to_dict() for item in self.sources.values()],
            "evidence": [item.to_dict() for item in self.evidence.values()],
            "claims": [item.to_dict() for item in self.claims.values()],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceBundle:
        bundle = cls()
        raw_evidence = value.get("evidence") or ()
        raw_sources = value.get("sources") or ()
        raw_claims = value.get("claims") or ()
        if not all(isinstance(items, (list, tuple)) for items in (raw_evidence, raw_sources, raw_claims)):
            raise ValueError("evidence bundle collections must be arrays")
        if (
            len(raw_evidence) > _MAX_BUNDLE_EVIDENCE
            or len(raw_sources) > _MAX_BUNDLE_SOURCES
            or len(raw_claims) > _MAX_BUNDLE_CLAIMS
        ):
            raise ValueError("evidence bundle collection count exceeds limit")
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                raise ValueError("evidence bundle contains a non-object evidence item")
            bundle.add_evidence(Evidence.from_dict(item))
        # Source-only records are uncommon but remain part of the contract.
        for item in raw_sources:
            if not isinstance(item, Mapping):
                raise ValueError("evidence bundle contains a non-object source item")
            source = SourceRef.from_dict(item)
            existing = bundle.sources.get(source.source_id)
            if existing is not None and _source_semantic_identity(existing) != _source_semantic_identity(source):
                raise ValueError(f"conflicting source id: {source.source_id}")
            if existing is None and len(bundle.sources) >= _MAX_BUNDLE_SOURCES:
                raise ValueError("evidence bundle exceeds source count limit")
            bundle.sources.setdefault(source.source_id, source)
        for item in raw_claims:
            if not isinstance(item, Mapping):
                raise ValueError("evidence bundle contains a non-object claim item")
            bundle.add_claim(Claim.from_dict(item))
        return bundle


def _source_semantic_identity(source: SourceRef) -> tuple[Any, ...]:
    """Exclude retrieval-time/ranking metadata from stable source identity checks."""
    return (
        source.source_id,
        source.source_type,
        source.title,
        source.locator,
        source.provider,
        source.as_of,
        source.published_at,
    )


def _evidence_semantic_identity(item: Evidence) -> tuple[Any, ...]:
    return (
        item.evidence_id,
        _source_semantic_identity(item.source),
        item.content,
        item.entity,
        item.field_name,
        item.value,
        item.unit,
        item.period,
        item.confidence,
        item.page,
        item.span_start,
        item.span_end,
        item.tags,
    )


def _validate_json_size(value: Any, *, field_name: str, max_characters: int) -> None:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    if len(payload) > max_characters:
        raise ValueError(f"{field_name} exceeds length limit")
