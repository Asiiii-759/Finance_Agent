"""Prompt context assembly with provenance, balancing and explicit trust zones."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .agent import ResearchRequest
from .contracts import Evidence, EvidenceBundle, SourceType


@dataclass(frozen=True)
class ContextManifest:
    included_evidence_ids: tuple[str, ...]
    omitted_evidence_count: int
    evidence_characters: int
    max_evidence_characters: int
    max_item_characters: int
    groups: tuple[str, ...]
    source_type_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "included_evidence_ids": list(self.included_evidence_ids),
            "omitted_evidence_count": self.omitted_evidence_count,
            "evidence_characters": self.evidence_characters,
            "max_evidence_characters": self.max_evidence_characters,
            "max_item_characters": self.max_item_characters,
            "groups": list(self.groups),
            "source_type_counts": dict(self.source_type_counts),
        }


class FinancialContextAssembler:
    """Select evidence fairly across entity/source groups within a hard character budget."""

    def __init__(self, *, max_evidence_chars: int = 48_000, max_item_chars: int = 2_400) -> None:
        if max_evidence_chars < 1_000 or max_item_chars < 200:
            raise ValueError("context budgets are too small")
        self.max_evidence_chars = max_evidence_chars
        self.max_item_chars = max_item_chars

    def build(
        self,
        request: ResearchRequest,
        bundle: EvidenceBundle,
        *,
        research_context: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ContextManifest]:
        ranked = sorted(
            bundle.evidence.values(),
            key=lambda item: self._score(request.query, item),
            reverse=True,
        )
        grouped: dict[str, deque[Evidence]] = defaultdict(deque)
        for item in ranked:
            group = self._group(item)
            grouped[group].append(item)
        selected: list[tuple[Evidence, dict[str, Any], int]] = []
        used = 0
        active = deque(sorted(grouped))
        while active:
            group = active.popleft()
            candidates = grouped[group]
            if not candidates:
                continue
            item = candidates.popleft()
            card = self._card(item, request.query)
            card_characters = len(json.dumps(card, ensure_ascii=False, separators=(",", ":")))
            if used + card_characters <= self.max_evidence_chars:
                selected.append((item, card, card_characters))
                used += card_characters
            if candidates:
                active.append(group)

        evidence_cards = [card for _item, card, _characters in selected]
        context = dict(research_context or {})
        thread_context = _safe_thread_context(request.thread_context)
        payload = {
            "prompt_version": "finance-evidence-synthesis-v3",
            "entities": list(request.entities),
            "task": {
                "query": request.query,
                "entities": list(request.entities),
                "response_language": "zh" if _contains_cjk(request.query) else "en",
            },
            "research": {
                "scope": context.get("scope"),
                "coverage": context.get("coverage"),
                "unresolved_gaps": context.get("unresolved_gaps", []),
                "stop_reason": context.get("stop_reason"),
            },
            "thread_context": thread_context,
            "personal_context": [dict(item) for item in request.personal_context],
            "evidence": evidence_cards,
        }
        manifest = ContextManifest(
            included_evidence_ids=tuple(item.evidence_id for item, _card, _characters in selected),
            omitted_evidence_count=len(bundle.evidence) - len(selected),
            evidence_characters=used,
            max_evidence_characters=self.max_evidence_chars,
            max_item_characters=self.max_item_chars,
            groups=tuple(sorted(grouped)),
            source_type_counts={
                source_type.value: sum(
                    item.source.source_type == source_type for item, _card, _characters in selected
                )
                for source_type in SourceType
            },
        )
        return payload, manifest

    def _card(self, item: Evidence, query: str) -> dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "content": _query_window(item.content, query, self.max_item_chars),
            "entity": item.entity,
            "field_name": item.field_name,
            "value": item.value,
            "unit": item.unit,
            "period": item.period,
            "confidence": item.confidence,
            "source": {
                "source_type": item.source.source_type.value,
                "title": item.source.title,
                "provider": item.source.provider,
                "locator": item.source.locator,
                "as_of": item.source.as_of,
                "published_at": item.source.published_at,
                "quality_tier": item.source.metadata.get("quality_tier"),
                "retrieval_rank": item.source.metadata.get("rank"),
                "content_basis": item.source.metadata.get("content_basis"),
            },
        }

    @staticmethod
    def _group(item: Evidence) -> str:
        source = item.source
        if source.source_type == SourceType.DOCUMENT:
            retrieval_trace = source.metadata.get("retrieval_trace")
            diversify_documents = (
                isinstance(retrieval_trace, Mapping)
                and retrieval_trace.get("document_diversification") is True
            )
            if diversify_documents:
                origin = (
                    source.metadata.get("document_id")
                    or source.metadata.get("corpus_record_id")
                    or source.source_id
                )
            else:
                origin = source.provider
        elif source.source_type == SourceType.WEB:
            origin = source.metadata.get("domain") or urlsplit(source.locator).hostname or source.source_id
        else:
            origin = source.provider
        return f"{item.entity or 'global'}::{source.source_type.value}::{origin}"

    @staticmethod
    def _score(query: str, item: Evidence) -> tuple[int, int, int, int, float, int, str]:
        terms = _tokens(query)
        haystack = " ".join(filter(None, (item.content, item.entity, item.field_name, item.source.title))).casefold()
        overlap = sum(term in haystack for term in terms)
        structured = int(item.value is not None and item.field_name is not None)
        source_priority = {
            SourceType.REGULATORY_FILING: 5,
            SourceType.CALCULATION: 4,
            SourceType.MARKET_DATA: 3,
            SourceType.MACRO_DATA: 3,
            SourceType.DOCUMENT: 2,
            SourceType.WEB: 1,
            SourceType.USER_INPUT: 1,
        }[item.source.source_type]
        quality_priority = {
            "public_authority": 3,
            "publisher_primary": 2,
            "open_web": 1,
        }.get(str(item.source.metadata.get("quality_tier") or ""), 0)
        rank = item.source.metadata.get("rank")
        rank_priority = -rank if isinstance(rank, int) and not isinstance(rank, bool) else -100_000
        return (
            overlap,
            source_priority,
            quality_priority,
            structured,
            item.confidence,
            rank_priority,
            item.evidence_id,
        )


def _safe_thread_context(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"summary", "recent_events", "entity_state", "relations", "focus_entities", "manifest"}
    return {str(key): item for key, item in value.items() if str(key) in allowed}


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
    bigrams = {value[index : index + 2] for value in cjk for index in range(max(0, len(value) - 1))}
    return latin | bigrams


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _query_window(content: str, query: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    lowered = content.casefold()
    positions = [lowered.find(term) for term in sorted(_tokens(query), key=len, reverse=True)]
    matches = [position for position in positions if position >= 0]
    if not matches:
        return content[:limit]
    center = min(matches)
    start = max(0, center - limit // 3)
    return content[start : start + limit]
