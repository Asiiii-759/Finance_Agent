"""Prompt context assembly with provenance, balancing and explicit trust zones."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .agent import AgentContext, ChatTurn
from .contracts import Evidence, EvidenceBundle, SourceType


@dataclass(frozen=True)
class ContextManifest:
    included_evidence_ids: tuple[str, ...]
    omitted_evidence_count: int
    evidence_tokens: int
    max_evidence_tokens: int
    groups: tuple[str, ...]
    source_type_counts: Mapping[str, int]
    thread_context_tokens: int
    personal_context_tokens: int
    omitted_personal_context_count: int
    reserved_tokens: int
    total_context_tokens: int
    max_context_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "included_evidence_ids": list(self.included_evidence_ids),
            "omitted_evidence_count": self.omitted_evidence_count,
            "evidence_tokens": self.evidence_tokens,
            "max_evidence_tokens": self.max_evidence_tokens,
            "groups": list(self.groups),
            "source_type_counts": dict(self.source_type_counts),
            "thread_context_tokens": self.thread_context_tokens,
            "personal_context_tokens": self.personal_context_tokens,
            "omitted_personal_context_count": self.omitted_personal_context_count,
            "reserved_tokens": self.reserved_tokens,
            "total_context_tokens": self.total_context_tokens,
            "max_context_tokens": self.max_context_tokens,
        }


class FinancialContextAssembler:
    """Select complete evidence passages fairly within a hard token budget."""

    def __init__(
        self,
        *,
        max_evidence_tokens: int,
        count_tokens: Callable[[str], int],
        max_context_tokens: int = 300_000,
    ) -> None:
        if max_evidence_tokens < 1_000:
            raise ValueError("context token budget is too small")
        self.max_evidence_tokens = max_evidence_tokens
        self.count_tokens = count_tokens
        if max_context_tokens < max_evidence_tokens + 1_000:
            raise ValueError("context budget cannot fit the evidence budget")
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        turn: ChatTurn,
        agent_context: AgentContext,
        bundle: EvidenceBundle,
        *,
        research_context: Mapping[str, Any] | None = None,
        reserved_tokens: int = 0,
    ) -> tuple[dict[str, Any], ContextManifest]:
        if reserved_tokens < 0 or reserved_tokens >= self.max_context_tokens:
            raise ValueError("reserved context tokens are invalid")
        context = dict(research_context or {})
        thread_context = _safe_thread_context(agent_context.thread_context)
        base_payload = {
            "prompt_version": "finance-evidence-synthesis-v3",
            "task": {
                "message": turn.message,
                "response_language": "zh" if _contains_cjk(turn.message) else "en",
            },
            "research": {
                "scope": context.get("scope"),
                "coverage": context.get("coverage"),
                "unresolved_gaps": context.get("unresolved_gaps", []),
                "stop_reason": context.get("stop_reason"),
            },
            "thread_context": thread_context,
            "personal_context": [],
            "evidence": [],
        }
        base_tokens = self.count_tokens(json.dumps(base_payload, ensure_ascii=False, separators=(",", ":")))
        remaining = self.max_context_tokens - reserved_tokens - base_tokens
        if remaining < 0:
            raise ValueError("thread and control context exceed the model input budget")
        ranked = sorted(
            bundle.evidence.values(),
            key=lambda item: self._score(turn.message, item),
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
            card = self._card(item)
            card_tokens = self.count_tokens(json.dumps(card, ensure_ascii=False, separators=(",", ":")))
            if used + card_tokens <= min(self.max_evidence_tokens, remaining):
                selected.append((item, card, card_tokens))
                used += card_tokens
            if candidates:
                active.append(group)

        evidence_cards = [card for _item, card, _tokens in selected]
        remaining -= used
        personal_ranked = sorted(
            (dict(item) for item in agent_context.personal_context),
            key=lambda item: self._personal_score(turn.message, item),
            reverse=True,
        )
        personal_context: list[dict[str, Any]] = []
        personal_tokens = 0
        for personal_item in personal_ranked:
            item_tokens = self.count_tokens(
                json.dumps(personal_item, ensure_ascii=False, separators=(",", ":"))
            )
            if personal_tokens + item_tokens <= remaining:
                personal_context.append(personal_item)
                personal_tokens += item_tokens
        payload = {
            "prompt_version": "finance-evidence-synthesis-v3",
            "task": {
                "message": turn.message,
                "response_language": "zh" if _contains_cjk(turn.message) else "en",
            },
            "research": {
                "scope": context.get("scope"),
                "coverage": context.get("coverage"),
                "unresolved_gaps": context.get("unresolved_gaps", []),
                "stop_reason": context.get("stop_reason"),
            },
            "thread_context": thread_context,
            "personal_context": personal_context,
            "evidence": evidence_cards,
        }
        total_context_tokens = self.count_tokens(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ) + reserved_tokens
        if total_context_tokens > self.max_context_tokens:
            raise ValueError("assembled context exceeds the model input budget")
        manifest = ContextManifest(
            included_evidence_ids=tuple(item.evidence_id for item, _card, _characters in selected),
            omitted_evidence_count=len(bundle.evidence) - len(selected),
            evidence_tokens=used,
            max_evidence_tokens=self.max_evidence_tokens,
            groups=tuple(sorted(grouped)),
            source_type_counts={
                source_type.value: sum(
                    item.source.source_type == source_type for item, _card, _characters in selected
                )
                for source_type in SourceType
            },
            thread_context_tokens=self.count_tokens(
                json.dumps(thread_context, ensure_ascii=False, separators=(",", ":"))
            ),
            personal_context_tokens=personal_tokens,
            omitted_personal_context_count=len(agent_context.personal_context) - len(personal_context),
            reserved_tokens=reserved_tokens,
            total_context_tokens=total_context_tokens,
            max_context_tokens=self.max_context_tokens,
        )
        return payload, manifest

    @staticmethod
    def _personal_score(query: str, item: Mapping[str, Any]) -> tuple[int, str]:
        terms = _tokens(query)
        text = json.dumps(item, ensure_ascii=False, sort_keys=True).casefold()
        return sum(term in text for term in terms), text

    def _card(self, item: Evidence) -> dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "content": item.content,
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
    allowed = {
        "summary",
        "recent_events",
        "run_state",
        "manifest",
    }
    return {str(key): item for key, item in value.items() if str(key) in allowed}


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
    bigrams = {value[index : index + 2] for value in cjk for index in range(max(0, len(value) - 1))}
    return latin | bigrams


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)
