"""Explicitly scoped memory primitives for finance-agent runs."""

from __future__ import annotations

import builtins
import json
import re
import sqlite3
import threading
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


def build_conversation_window(
    store: SQLiteMemoryStore,
    namespace: MemoryNamespace,
    *,
    max_characters: int = 16_000,
    recent_event_count: int = 12,
) -> dict[str, Any]:
    """Build the bounded prompt projection without deleting the durable event ledger."""
    if not 4_000 <= max_characters <= 100_000:
        raise ValueError("conversation context budget must be between 4000 and 100000 characters")
    if not 4 <= recent_event_count <= 50:
        raise ValueError("recent conversation event count must be between 4 and 50")
    _validate_conversation_namespace(namespace)

    stored = store.get_conversation_summary(namespace)
    covered = stored.covered_through_sequence if stored is not None else 0
    summary = deepcopy(stored.summary) if stored is not None else {
        "user_requests": [],
        "assistant_outcomes": [],
        "tool_activity": [],
        "entity_state": {},
        "relations": [],
    }
    pending = store.list_conversation_events(namespace, after_sequence=covered)

    content_limit = max(200, max_characters // (recent_event_count * 2))
    projection = _conversation_projection(summary, pending, covered, content_limit=content_limit)
    if len(json.dumps(projection, ensure_ascii=False, sort_keys=True)) >= int(max_characters * 0.85) and len(
        pending
    ) > recent_event_count:
        compacted = pending[:-recent_event_count]
        summary = _merge_conversation_summary(summary, compacted)
        covered = compacted[-1].sequence
        pending = pending[-recent_event_count:]
        projection = _conversation_projection(summary, pending, covered, content_limit=content_limit)
        while len(json.dumps(projection, ensure_ascii=False, sort_keys=True)) > max_characters and any(
            summary[name] for name in ("user_requests", "assistant_outcomes", "tool_activity")
        ):
            populated = [
                name for name in ("user_requests", "assistant_outcomes", "tool_activity") if summary[name]
            ]
            oldest = min(populated, key=lambda name: int(summary[name][0]["sequence"]))
            summary[oldest].pop(0)
            projection = _conversation_projection(summary, pending, covered, content_limit=content_limit)
        store.put_conversation_summary(
            namespace,
            covered_through_sequence=covered,
            summary=summary,
        )

    while len(json.dumps(projection, ensure_ascii=False, sort_keys=True)) > max_characters and len(pending) > 4:
        pending = pending[1:]
        projection = _conversation_projection(summary, pending, covered, content_limit=content_limit)
    if len(json.dumps(projection, ensure_ascii=False, sort_keys=True)) > max_characters:
        raise ValueError("conversation context cannot fit the configured budget")
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
        with self._connect() as connection:
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

    def put(
        self,
        namespace: MemoryNamespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _validate_record(key, value, metadata)
        now = _utc_now()
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_memories WHERE tenant_id = ? AND user_id = ?
                AND kind = ? AND thread_id = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (*_namespace_values(namespace), limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def delete_namespace(self, namespace: MemoryNamespace) -> int:
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
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
    *,
    content_limit: int = 1_200,
) -> dict[str, Any]:
    entity_state = deepcopy(summary.get("entity_state") or {})
    relation_state: dict[tuple[str, str, str], dict[str, Any]] = {
        (str(item["subject"]), str(item["predicate"]), str(item["object"])): dict(item)
        for item in summary.get("relations") or []
    }
    for event in events:
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
        for relation in event.relations:
            key = (relation.subject, relation.predicate, relation.object)
            state = relation_state.setdefault(
                key,
                {
                    **relation.to_dict(),
                    "first_seen_at": event.occurred_at,
                    "last_seen_at": event.occurred_at,
                    "mention_count": 0,
                },
            )
            state["last_seen_at"] = event.occurred_at
            state["mention_count"] = int(state["mention_count"]) + 1

    latest_user = next(
        (event for event in reversed(events) if event.kind is ConversationEventKind.USER_MESSAGE and event.entities),
        None,
    )
    if latest_user is not None:
        focus_entities = list(latest_user.entities)
    else:
        ranked = sorted(
            entity_state.items(),
            key=lambda item: (int(item[1]["last_sequence"]), str(item[0])),
            reverse=True,
        )
        focus_entities = [name for name, _ in ranked[:10]]

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
    recent_events = []
    for event in events:
        value = event.to_dict(content_limit=content_limit)
        value["payload"] = {
            key: item for key, item in value["payload"].items() if key in safe_payload_keys
        }
        recent_events.append(value)
    return {
        "summary": {
            "user_requests": list(summary.get("user_requests") or []),
            "assistant_outcomes": list(summary.get("assistant_outcomes") or []),
            "tool_activity": list(summary.get("tool_activity") or []),
        },
        "recent_events": recent_events,
        "entity_state": entity_state,
        "relations": list(relation_state.values()),
        "focus_entities": focus_entities,
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
) -> dict[str, Any]:
    projection = _conversation_projection(summary, events, 0)
    user_requests = list(summary.get("user_requests") or [])
    assistant_outcomes = list(summary.get("assistant_outcomes") or [])
    tool_activity = list(summary.get("tool_activity") or [])
    for event in events:
        item = {
            "sequence": event.sequence,
            "occurred_at": event.occurred_at,
            "content": event.content[:600],
            "entities": list(event.entities),
        }
        if event.kind is ConversationEventKind.USER_MESSAGE:
            user_requests.append(item)
        elif event.kind is ConversationEventKind.ASSISTANT_MESSAGE:
            assistant_outcomes.append(item)
        else:
            item["tool_name"] = event.payload.get("tool_name")
            item["result_status"] = event.payload.get("result_status")
            tool_activity.append(item)
    return {
        "user_requests": user_requests[-20:],
        "assistant_outcomes": assistant_outcomes[-12:],
        "tool_activity": tool_activity[-30:],
        "entity_state": dict(list(projection["entity_state"].items())[-50:]),
        "relations": projection["relations"][-100:],
    }


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
