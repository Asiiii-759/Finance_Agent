"""Explicitly scoped memory primitives for finance-agent runs."""

from __future__ import annotations

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
        values = (self.tenant_id, self.user_id, self.kind) + ((self.thread_id,) if self.thread_id else ())
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
class ThreadContextMemory:
    previous_query: str
    entities: tuple[str, ...]
    symbols: dict[str, str]
    last_status: str
    unresolved_gap_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.previous_query.strip() or len(self.previous_query) > 500:
            raise ValueError("invalid remembered query")
        if len(self.entities) > 50 or any(not item.strip() or len(item) > 200 for item in self.entities):
            raise ValueError("invalid remembered entities")
        if len(set(self.entities)) != len(self.entities):
            raise ValueError("remembered entities contain duplicates")
        if set(self.symbols).difference(self.entities) or any(
            not value or len(value) > 64 for value in self.symbols.values()
        ):
            raise ValueError("invalid remembered symbols")
        if self.last_status not in {"succeeded", "degraded", "failed"}:
            raise ValueError("invalid remembered status")
        if len(self.unresolved_gap_codes) > 20 or any(
            not re.fullmatch(r"[a-z0-9_.:-]{1,100}", item) for item in self.unresolved_gap_codes
        ):
            raise ValueError("invalid remembered gap codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_query": self.previous_query,
            "entities": list(self.entities),
            "symbols": dict(self.symbols),
            "last_status": self.last_status,
            "unresolved_gap_codes": list(self.unresolved_gap_codes),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ThreadContextMemory:
        if not isinstance(value, dict):
            raise ValueError("thread context memory must be an object")
        allowed = {
            "previous_query",
            "entities",
            "symbols",
            "last_status",
            "unresolved_gap_codes",
        }
        if set(value).difference(allowed):
            raise ValueError("thread context memory contains unknown fields")
        entities = value.get("entities") or []
        symbols = value.get("symbols") or {}
        gaps = value.get("unresolved_gap_codes") or []
        if (
            not isinstance(value.get("previous_query"), str)
            or not isinstance(value.get("last_status"), str)
            or not isinstance(entities, list)
            or any(not isinstance(item, str) for item in entities)
            or not isinstance(symbols, dict)
            or any(not isinstance(key, str) or not isinstance(item, str) for key, item in symbols.items())
            or not isinstance(gaps, list)
            or any(not isinstance(item, str) for item in gaps)
        ):
            raise ValueError("thread context memory fields have invalid types")
        return cls(
            previous_query=value["previous_query"],
            entities=tuple(entities),
            symbols=dict(symbols),
            last_status=value["last_status"],
            unresolved_gap_codes=tuple(gaps),
        )


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
