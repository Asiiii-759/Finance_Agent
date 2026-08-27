"""Explicitly scoped memory primitives for finance-agent runs."""

from __future__ import annotations

import builtins
import json
import math
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class PersonalMemoryKind(StrEnum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    SKILL = "skill"


class ConversationEventKind(StrEnum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_EVENT = "tool_event"


@dataclass(frozen=True)
class PersonalMemory:
    kind: PersonalMemoryKind
    title: str
    content: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip() or len(self.title) > 200:
            raise ValueError("personal memory title is invalid")
        if not self.content.strip() or len(self.content) > 8_000:
            raise ValueError("personal memory content is invalid")
        if len(self.tags) > 20 or any(not item.strip() or len(item) > 100 for item in self.tags):
            raise ValueError("personal memory tags are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "title": self.title.strip(),
            "content": self.content.strip(),
            "tags": list(dict.fromkeys(item.strip() for item in self.tags)),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PersonalMemory:
        if not isinstance(value, dict) or set(value).difference({"kind", "title", "content", "tags"}):
            raise ValueError("personal memory must be a known object shape")
        tags = value.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(item, str) for item in tags):
            raise ValueError("personal memory tags must be strings")
        return cls(
            kind=PersonalMemoryKind(str(value.get("kind") or "")),
            title=str(value.get("title") or ""),
            content=str(value.get("content") or ""),
            tags=tuple(tags),
        )


@dataclass(frozen=True)
class MemoryNamespace:
    tenant_id: str
    user_id: str
    kind: str
    thread_id: str | None = None

    def __post_init__(self) -> None:
        values = (self.tenant_id, self.user_id, self.kind) + (
            (self.thread_id,) if self.thread_id is not None else ()
        )
        if any(
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or "/" in value
            or ".." in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            for value in values
        ):
            raise ValueError("invalid memory namespace")

    def key(self) -> tuple[str, ...]:
        base = (self.tenant_id, self.user_id, self.kind)
        return base + ((self.thread_id,) if self.thread_id else ())


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: Any
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityRelation:
    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        if any(not value.strip() or len(value) > 200 for value in (self.subject, self.object)):
            raise ValueError("conversation relation endpoints are invalid")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.predicate):
            raise ValueError("conversation relation predicate is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"subject": self.subject.strip(), "predicate": self.predicate, "object": self.object.strip()}

    @classmethod
    def from_dict(cls, value: Any) -> EntityRelation:
        if not isinstance(value, dict) or set(value) != {"subject", "predicate", "object"}:
            raise ValueError("conversation relation must use the known object shape")
        if any(not isinstance(value[name], str) for name in ("subject", "predicate", "object")):
            raise ValueError("conversation relation fields must be strings")
        return cls(value["subject"], value["predicate"], value["object"])


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    sequence: int
    kind: ConversationEventKind
    content: str
    occurred_at: str
    run_id: str
    entities: tuple[str, ...] = ()
    relations: tuple[EntityRelation, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_record_key(self.event_id)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("conversation event sequence is invalid")
        if not self.content.strip() or len(self.content) > 100_000:
            raise ValueError("conversation event content is invalid")
        if not self.run_id.strip() or len(self.run_id) > 200:
            raise ValueError("conversation event run_id is invalid")
        if len(self.entities) > 50 or any(not value.strip() or len(value) > 200 for value in self.entities):
            raise ValueError("conversation event entities are invalid")
        if len(set(self.entities)) != len(self.entities):
            raise ValueError("conversation event entities contain duplicates")
        if len(self.relations) > 100:
            raise ValueError("conversation event relations exceed the limit")
        _aware_datetime(self.occurred_at, "conversation event occurred_at")
        _validate_json_payload(self.payload, "conversation event payload", 30_000)

    def to_dict(self, *, content_limit: int | None = None) -> dict[str, Any]:
        content = self.content if content_limit is None else self.content[:content_limit]
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "content": content,
            "occurred_at": self.occurred_at,
            "run_id": self.run_id,
            "entities": list(self.entities),
            "relations": [item.to_dict() for item in self.relations],
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ConversationSummaryRecord:
    covered_through_sequence: int
    summary: dict[str, Any]
    updated_at: str


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        previous_summary: dict[str, Any],
        events: tuple[ConversationEvent, ...],
    ) -> dict[str, Any]: ...


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class DeepSeekV4TokenEstimator:
    """Conservative preflight estimate; API usage remains authoritative."""

    name = "deepseek_v4_estimate_v1"

    def count(self, text: str) -> int:
        cjk = sum("\u3400" <= character <= "\u9fff" for character in text)
        non_cjk = len(text) - cjk
        punctuation = sum(not character.isalnum() and not character.isspace() for character in text)
        estimate = cjk * 0.7 + non_cjk * 0.35 + punctuation * 0.65
        return max(1, math.ceil(estimate * 1.1))


def build_conversation_window(
    store: SQLiteMemoryStore,
    namespace: MemoryNamespace,
    *,
    max_tokens: int = 300_000,
    recent_tokens: int = 20_000,
    summarizer: ConversationSummarizer | None = None,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Build the bounded prompt projection without deleting the durable event ledger."""
    if not 16_000 <= max_tokens <= 300_000:
        raise ValueError("conversation context budget must be between 16000 and 300000 tokens")
    if not 4_000 <= recent_tokens <= 100_000:
        raise ValueError("recent conversation budget must be between 4000 and 100000 tokens")
    _validate_conversation_namespace(namespace)
    counter = token_counter or DeepSeekV4TokenEstimator()

    stored = store.get_conversation_summary(namespace)
    covered = stored.covered_through_sequence if stored is not None else 0
    summary = deepcopy(stored.summary) if stored is not None else {
        "semantic_summary": _empty_semantic_summary(),
        "entity_state": {},
        "entity_events": [],
        "focus_history": [],
        "run_state": {},
    }
    pending = store.list_conversation_events(namespace, after_sequence=covered)

    projection = _conversation_projection(summary, pending, covered)
    projection_tokens = _projection_tokens(projection, counter)
    compacted, recent = _split_recent_runs(pending, counter, recent_tokens)
    if projection_tokens >= int(max_tokens * 0.85) and compacted:
        if summarizer is None:
            raise RuntimeError("conversation compaction requires an LLM summarizer")
        summary = _merge_conversation_summary(summary, compacted, summarizer)
        covered = compacted[-1].sequence
        pending = recent
        projection = _conversation_projection(summary, pending, covered)
        projection_tokens = _projection_tokens(projection, counter)
        store.put_conversation_summary(
            namespace,
            covered_through_sequence=covered,
            summary=summary,
        )

    if projection_tokens > max_tokens:
        raise ValueError("conversation context cannot fit the configured budget")
    projection["manifest"]["estimated_token_count"] = projection_tokens
    projection["manifest"]["token_count_method"] = counter.name
    projection["manifest"]["max_context_tokens"] = max_tokens
    projection["manifest"]["recent_context_tokens"] = _events_tokens(pending, counter)
    projection["manifest"]["max_recent_context_tokens"] = recent_tokens
    return projection


class MemoryStore(Protocol):
    def put(self, namespace: MemoryNamespace, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None: ...

    def get(self, namespace: MemoryNamespace, key: str) -> MemoryRecord | None: ...

    def list(self, namespace: MemoryNamespace, limit: int = 100) -> list[MemoryRecord]: ...

    def delete(self, namespace: MemoryNamespace, key: str) -> bool: ...

    def delete_namespace(self, namespace: MemoryNamespace) -> int: ...


class InMemoryStore:
    """Thread-safe development store with defensive copies.

    It deliberately offers no cross-namespace search. Callers must construct the
    exact tenant/user/thread namespace, which makes accidental data leakage harder.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, ...], dict[str, MemoryRecord]] = {}
        self._lock = threading.RLock()

    def put(
        self,
        namespace: MemoryNamespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _validate_record(key, value, metadata)
        now = _utc_now()
        with self._lock:
            bucket = self._data.setdefault(namespace.key(), {})
            existing = bucket.get(key)
            bucket[key] = MemoryRecord(
                key=key,
                value=deepcopy(value),
                created_at=existing.created_at if existing else now,
                updated_at=now,
                metadata=deepcopy(metadata or {}),
            )

    def get(self, namespace: MemoryNamespace, key: str) -> MemoryRecord | None:
        with self._lock:
            record = self._data.get(namespace.key(), {}).get(key)
            return deepcopy(record)

    def list(self, namespace: MemoryNamespace, limit: int = 100) -> list[MemoryRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            records = list(self._data.get(namespace.key(), {}).values())
            records.sort(key=lambda item: item.updated_at, reverse=True)
            return deepcopy(records[:limit])

    def delete_namespace(self, namespace: MemoryNamespace) -> int:
        with self._lock:
            records = self._data.pop(namespace.key(), {})
            return len(records)

    def delete(self, namespace: MemoryNamespace, key: str) -> bool:
        _validate_record_key(key)
        with self._lock:
            bucket = self._data.get(namespace.key(), {})
            return bucket.pop(key, None) is not None


class SQLiteMemoryStore:
    """Durable JSON memory with exact namespace lookup and deletion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memories (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, kind, thread_id, record_key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_events (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    relations_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, kind, thread_id, sequence),
                    UNIQUE (tenant_id, user_id, kind, thread_id, event_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    covered_through_sequence INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, kind, thread_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(
        self,
        namespace: MemoryNamespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _validate_record(key, value, metadata)
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO agent_memories (
                    tenant_id, user_id, kind, thread_id, record_key,
                    value_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, user_id, kind, thread_id, record_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    namespace.tenant_id,
                    namespace.user_id,
                    namespace.kind,
                    namespace.thread_id or "",
                    key,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )

    def get(self, namespace: MemoryNamespace, key: str) -> MemoryRecord | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_memories WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? AND record_key = ?
                """,
                (*_namespace_values(namespace), key),
            ).fetchone()
        return _row_to_record(row) if row else None

    def list(self, namespace: MemoryNamespace, limit: int = 100) -> list[MemoryRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_memories WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (*_namespace_values(namespace), limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def delete_namespace(self, namespace: MemoryNamespace) -> int:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM agent_memories WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ?
                """,
                _namespace_values(namespace),
            )
            return cursor.rowcount

    def delete(self, namespace: MemoryNamespace, key: str) -> bool:
        _validate_record_key(key)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM agent_memories WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? AND record_key = ?
                """,
                (*_namespace_values(namespace), key),
            )
            return cursor.rowcount == 1

    def append_conversation_event(
        self,
        namespace: MemoryNamespace,
        *,
        event_id: str,
        kind: ConversationEventKind,
        content: str,
        run_id: str,
        entities: tuple[str, ...] = (),
        relations: tuple[EntityRelation, ...] = (),
        payload: dict[str, Any] | None = None,
        occurred_at: str | None = None,
    ) -> ConversationEvent:
        if namespace.kind != "conversation_history" or namespace.thread_id is None:
            raise ValueError("conversation events require a thread-scoped conversation_history namespace")
        timestamp = occurred_at or _utc_now()
        candidate = ConversationEvent(
            event_id=event_id,
            sequence=1,
            kind=kind,
            content=content,
            occurred_at=timestamp,
            run_id=run_id,
            entities=entities,
            relations=relations,
            payload=dict(payload or {}),
        )
        values = _namespace_values(namespace)
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM conversation_events WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? AND event_id = ?
                """,
                (*values, event_id),
            ).fetchone()
            if existing is not None:
                stored = _row_to_conversation_event(existing)
                if (
                    stored.kind != candidate.kind
                    or stored.content != candidate.content
                    or stored.run_id != candidate.run_id
                    or stored.entities != candidate.entities
                    or stored.relations != candidate.relations
                    or stored.payload != candidate.payload
                ):
                    raise ValueError("conversation event_id conflicts with an existing event")
                return stored
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM conversation_events WHERE tenant_id = ? AND user_id = ? AND kind = ? AND thread_id = ?
                """,
                values,
            ).fetchone()
            sequence = int(row["next_sequence"])
            connection.execute(
                """
                INSERT INTO conversation_events (
                    tenant_id, user_id, kind, thread_id, sequence, event_id, event_kind,
                    content, occurred_at, run_id, entities_json, relations_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *values,
                    sequence,
                    candidate.event_id,
                    candidate.kind.value,
                    candidate.content,
                    candidate.occurred_at,
                    candidate.run_id,
                    json.dumps(candidate.entities, ensure_ascii=False),
                    json.dumps([item.to_dict() for item in candidate.relations], ensure_ascii=False),
                    json.dumps(candidate.payload, ensure_ascii=False, sort_keys=True),
                ),
            )
        return ConversationEvent(
            event_id=candidate.event_id,
            sequence=sequence,
            kind=candidate.kind,
            content=candidate.content,
            occurred_at=candidate.occurred_at,
            run_id=candidate.run_id,
            entities=candidate.entities,
            relations=candidate.relations,
            payload=candidate.payload,
        )

    def list_conversation_events(
        self,
        namespace: MemoryNamespace,
        *,
        after_sequence: int = 0,
        limit: int = 10_000,
    ) -> builtins.list[ConversationEvent]:
        if namespace.kind != "conversation_history" or namespace.thread_id is None:
            raise ValueError("conversation events require a thread-scoped conversation_history namespace")
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise ValueError("conversation event cursor is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("conversation event limit must be between 1 and 10000")
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversation_events WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? AND sequence > ? ORDER BY sequence ASC LIMIT ?
                """,
                (*_namespace_values(namespace), after_sequence, limit),
            ).fetchall()
        return [_row_to_conversation_event(row) for row in rows]

    def get_conversation_summary(self, namespace: MemoryNamespace) -> ConversationSummaryRecord | None:
        _validate_conversation_namespace(namespace)
        with self._lock, self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM conversation_summaries WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ?
                """,
                _namespace_values(namespace),
            ).fetchone()
        if row is None:
            return None
        summary = json.loads(row["summary_json"])
        if not isinstance(summary, dict):
            raise ValueError("stored conversation summary must be an object")
        _validate_json_payload(summary, "stored conversation summary", 50_000)
        covered_through_sequence = row["covered_through_sequence"]
        if (
            isinstance(covered_through_sequence, bool)
            or not isinstance(covered_through_sequence, int)
            or covered_through_sequence < 1
        ):
            raise ValueError("stored conversation summary cursor is invalid")
        return ConversationSummaryRecord(
            covered_through_sequence=covered_through_sequence,
            summary=summary,
            updated_at=_aware_datetime(str(row["updated_at"]), "stored conversation summary updated_at").isoformat(),
        )

    def put_conversation_summary(
        self,
        namespace: MemoryNamespace,
        *,
        covered_through_sequence: int,
        summary: dict[str, Any],
    ) -> None:
        _validate_conversation_namespace(namespace)
        if (
            isinstance(covered_through_sequence, bool)
            or not isinstance(covered_through_sequence, int)
            or covered_through_sequence < 1
        ):
            raise ValueError("conversation summary cursor is invalid")
        _validate_json_payload(summary, "conversation summary", 50_000)
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_summaries (
                    tenant_id, user_id, kind, thread_id, covered_through_sequence, summary_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, user_id, kind, thread_id) DO UPDATE SET
                    covered_through_sequence = excluded.covered_through_sequence,
                    summary_json = excluded.summary_json,
                    updated_at = excluded.updated_at
                """,
                (
                    *_namespace_values(namespace),
                    covered_through_sequence,
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )

    def conversation_run_ids(self, namespace: MemoryNamespace) -> tuple[str, ...]:
        _validate_conversation_namespace(namespace)
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT run_id FROM conversation_events WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? ORDER BY run_id
                """,
                _namespace_values(namespace),
            ).fetchall()
        return tuple(str(row["run_id"]) for row in rows)

    def delete_conversation(self, namespace: MemoryNamespace) -> dict[str, int]:
        _validate_conversation_namespace(namespace)
        with self._lock, self._connection() as connection:
            events = connection.execute(
                """
                DELETE FROM conversation_events WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ?
                """,
                _namespace_values(namespace),
            ).rowcount
            summaries = connection.execute(
                """
                DELETE FROM conversation_summaries WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ?
                """,
                _namespace_values(namespace),
            ).rowcount
        return {"events": events, "summaries": summaries}


def _namespace_values(namespace: MemoryNamespace) -> tuple[str, str, str, str]:
    return (
        namespace.tenant_id,
        namespace.user_id,
        namespace.kind,
        namespace.thread_id or "",
    )


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    value = json.loads(row["value_json"])
    metadata = json.loads(row["metadata_json"])
    if not isinstance(metadata, dict):
        raise ValueError("stored memory metadata must be an object")
    _validate_record(str(row["record_key"]), value, metadata)
    return MemoryRecord(
        key=str(row["record_key"]),
        value=value,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata=metadata,
    )


def _row_to_conversation_event(row: sqlite3.Row) -> ConversationEvent:
    entities = json.loads(row["entities_json"])
    relations = json.loads(row["relations_json"])
    payload = json.loads(row["payload_json"])
    if not isinstance(entities, list) or any(not isinstance(item, str) for item in entities):
        raise ValueError("stored conversation entities must be a string list")
    if not isinstance(relations, list) or not isinstance(payload, dict):
        raise ValueError("stored conversation event JSON has an invalid shape")
    return ConversationEvent(
        event_id=str(row["event_id"]),
        sequence=int(row["sequence"]),
        kind=ConversationEventKind(str(row["event_kind"])),
        content=str(row["content"]),
        occurred_at=str(row["occurred_at"]),
        run_id=str(row["run_id"]),
        entities=tuple(entities),
        relations=tuple(EntityRelation.from_dict(item) for item in relations),
        payload=payload,
    )


def _conversation_projection(
    summary: dict[str, Any],
    events: list[ConversationEvent],
    covered_through_sequence: int,
) -> dict[str, Any]:
    entity_state = deepcopy(summary.get("entity_state") or {})
    entity_events = [dict(item) for item in summary.get("entity_events") or []]
    focus_history = [dict(item) for item in summary.get("focus_history") or []]
    run_state = deepcopy(summary.get("run_state") or {})
    for event in events:
        atom = _entity_event_atom(event)
        if atom is not None:
            entity_events.append(atom)
        run = run_state.setdefault(
            event.run_id,
            {
                "run_id": event.run_id,
                "status": "unfinished",
                "started_at": event.occurred_at,
                "last_sequence": event.sequence,
                "tools": [],
            },
        )
        run["last_sequence"] = event.sequence
        if event.kind is ConversationEventKind.USER_MESSAGE:
            run["request"] = event.content[:500]
            for entity in event.entities:
                state = entity_state.setdefault(
                    entity,
                    {
                        "first_seen_at": event.occurred_at,
                        "last_seen_at": event.occurred_at,
                        "last_sequence": event.sequence,
                        "mention_count": 0,
                    },
                )
                state["last_seen_at"] = event.occurred_at
                state["last_sequence"] = event.sequence
                state["mention_count"] = int(state["mention_count"]) + 1
                symbols = event.payload.get("entity_symbols") or {}
                if isinstance(symbols, dict) and isinstance(symbols.get(entity), str):
                    state["symbol"] = symbols[entity]
            for relation in event.relations:
                if relation.predicate == "has_symbol" and relation.subject in entity_state:
                    entity_state[relation.subject]["symbol"] = relation.object
            if event.entities:
                focus_history.append(
                    {
                        "sequence": event.sequence,
                        "occurred_at": event.occurred_at,
                        "entities": list(event.entities),
                    }
                )
        elif event.kind is ConversationEventKind.TOOL_EVENT:
            run["tools"].append(
                {
                    "tool_name": event.payload.get("tool_name"),
                    "result_status": event.payload.get("result_status"),
                    "error_code": event.payload.get("error_code"),
                    "attempts": event.payload.get("attempts"),
                }
            )
        else:
            run["status"] = "completed"
            run["outcome_status"] = event.payload.get("status")
    focus_history = focus_history[-50:]
    entity_events = entity_events[-100:]
    if focus_history:
        focus_entities = list(focus_history[-1]["entities"])
    else:
        ranked = sorted(
            entity_state.items(),
            key=lambda item: (int(item[1]["last_sequence"]), str(item[0])),
            reverse=True,
        )
        focus_entities = [name for name, _ in ranked[:10]]

    ordered_runs = sorted(run_state.values(), key=lambda item: int(item["last_sequence"]))[-20:]
    return {
        "summary": dict(summary.get("semantic_summary") or _empty_semantic_summary()),
        "recent_events": [_prompt_event(event) for event in events],
        "entity_state": entity_state,
        "entity_events": entity_events,
        "focus_history": focus_history,
        "focus_entities": focus_entities,
        "run_state": ordered_runs,
        "manifest": {
            "covered_through_sequence": covered_through_sequence,
            "recent_event_count": len(events),
            "latest_sequence": events[-1].sequence if events else covered_through_sequence,
            "full_history_persisted": True,
            "memory_is_evidence": False,
        },
    }


def _merge_conversation_summary(
    summary: dict[str, Any],
    events: list[ConversationEvent],
    summarizer: ConversationSummarizer,
) -> dict[str, Any]:
    projection = _conversation_projection(summary, events, 0)
    previous_summary = summary.get("semantic_summary")
    if not isinstance(previous_summary, dict):
        previous_summary = {
            "conversation_summary": json.dumps(
                {
                    key: summary.get(key) or []
                    for key in ("user_requests", "assistant_outcomes", "tool_activity")
                },
                ensure_ascii=False,
            )[:12_000],
            "user_goals": [],
            "decisions": [],
            "open_questions": [],
        }
    semantic_summary = summarizer.summarize(previous_summary, tuple(events))
    _validate_semantic_summary(semantic_summary)
    return {
        "semantic_summary": semantic_summary,
        "entity_state": dict(list(projection["entity_state"].items())[-50:]),
        "entity_events": projection["entity_events"][-100:],
        "focus_history": projection["focus_history"][-50:],
        "run_state": {item["run_id"]: item for item in projection["run_state"][-20:]},
    }


def _entity_event_atom(event: ConversationEvent) -> dict[str, Any] | None:
    if not event.entities:
        return None
    if event.kind is ConversationEventKind.USER_MESSAGE:
        if len(event.entities) > 1 and re.search(r"比较|对比|区别|孰优|versus|\bvs\.?\b|compare", event.content, re.I):
            action = "comparison_requested"
            fact = f"用户请求比较：{'、'.join(event.entities)}"
        elif re.search(r"计算|测算|回撤|收益率|cagr|指标|估值", event.content, re.I):
            action = "calculation_requested"
            fact = f"用户请求计算或分析：{'、'.join(event.entities)}"
        else:
            action = "mentioned"
            fact = f"用户提到：{'、'.join(event.entities)}"
        status = "requested"
    elif event.kind is ConversationEventKind.TOOL_EVENT:
        action = f"tool:{event.payload.get('tool_name') or 'unknown'}"
        status = str(event.payload.get("result_status") or "unknown")
        fact = f"{action} {status}"
    else:
        action = "response_completed"
        status = str(event.payload.get("status") or "completed")
        fact = "已生成本轮回答"
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "occurred_at": event.occurred_at,
        "run_id": event.run_id,
        "action": action,
        "entities": list(event.entities),
        "status": status,
        "fact": fact,
        "confidence": "explicit" if event.kind is ConversationEventKind.USER_MESSAGE else "observed",
    }


def _prompt_event(event: ConversationEvent) -> dict[str, Any]:
    safe_payload_keys = {
        "attempts",
        "capability",
        "claim_count",
        "error_code",
        "gap_codes",
        "network_attempts",
        "result_status",
        "source_count",
        "status",
        "tool_name",
    }
    value = event.to_dict()
    value.pop("relations", None)
    value["payload"] = {
        key: item for key, item in value["payload"].items() if key in safe_payload_keys
    }
    return value


def _events_tokens(events: list[ConversationEvent], counter: TokenCounter) -> int:
    return counter.count(json.dumps([_prompt_event(event) for event in events], ensure_ascii=False, sort_keys=True))


def _split_recent_runs(
    events: list[ConversationEvent],
    counter: TokenCounter,
    recent_tokens: int,
) -> tuple[list[ConversationEvent], list[ConversationEvent]]:
    if not events:
        return [], []
    runs: list[list[ConversationEvent]] = []
    for event in events:
        if not runs or runs[-1][-1].run_id != event.run_id:
            runs.append([])
        runs[-1].append(event)
    selected: list[list[ConversationEvent]] = []
    used = 0
    for run in reversed(runs):
        run_tokens = _events_tokens(run, counter)
        if selected and used + run_tokens > recent_tokens:
            break
        selected.append(run)
        used += run_tokens
    selected.reverse()
    recent = [event for run in selected for event in run]
    return events[: len(events) - len(recent)], recent


def _empty_semantic_summary() -> dict[str, Any]:
    return {
        "conversation_summary": "",
        "user_goals": [],
        "requirements": [],
        "decisions": [],
        "completed_work": [],
        "successful_tools": [],
        "failed_tools": [],
        "unfinished_work": [],
        "open_questions": [],
    }


def _validate_semantic_summary(value: dict[str, Any]) -> None:
    expected = {
        "conversation_summary",
        "user_goals",
        "requirements",
        "decisions",
        "completed_work",
        "successful_tools",
        "failed_tools",
        "unfinished_work",
        "open_questions",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("conversation summary must use the required schema")
    if not isinstance(value["conversation_summary"], str) or len(value["conversation_summary"]) > 20_000:
        raise ValueError("conversation summary text is invalid")
    for name, limit in (
        ("user_goals", 50),
        ("requirements", 50),
        ("decisions", 50),
        ("completed_work", 100),
        ("successful_tools", 100),
        ("failed_tools", 100),
        ("unfinished_work", 50),
        ("open_questions", 30),
    ):
        items = value[name]
        if (
            not isinstance(items, list)
            or len(items) > limit
            or any(not isinstance(item, str) or not item.strip() or len(item) > 1_000 for item in items)
        ):
            raise ValueError(f"conversation summary {name} is invalid")


def _projection_tokens(projection: dict[str, Any], counter: TokenCounter) -> int:
    return counter.count(json.dumps(projection, ensure_ascii=False, sort_keys=True))


def _validate_record(key: str, value: Any, metadata: dict[str, Any] | None) -> None:
    _validate_record_key(key)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("memory metadata must be an object")
    for field_name, payload, maximum in (
        ("memory value", value, 100_000),
        ("memory metadata", metadata or {}, 20_000),
    ):
        try:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be JSON serializable") from exc
        if len(encoded) > maximum:
            raise ValueError(f"{field_name} exceeds length limit")


def _validate_json_payload(value: Any, field_name: str, maximum: int) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    if len(encoded) > maximum:
        raise ValueError(f"{field_name} exceeds length limit")


def _aware_datetime(value: str, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be ISO-8601")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _validate_conversation_namespace(namespace: MemoryNamespace) -> None:
    if namespace.kind != "conversation_history" or namespace.thread_id is None:
        raise ValueError("conversation data requires a thread-scoped conversation_history namespace")


def _validate_record_key(key: str) -> None:
    if (
        not isinstance(key, str)
        or not key
        or len(key) > 128
        or "/" in key
        or ".." in key
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
    ):
        raise ValueError("invalid memory key")
