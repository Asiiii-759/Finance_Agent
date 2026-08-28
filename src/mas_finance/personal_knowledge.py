"""Persistent, user-scoped personal document knowledge base."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .corpus import CorpusDocument, InMemoryCorpus
from .embeddings import EmbeddingProvider


class SQLitePersonalKnowledgeBase:
    """Store parsed page text; raw uploads remain outside this database."""

    def __init__(self, path: Path, *, max_documents_per_user: int = 100) -> None:
        if not 1 <= max_documents_per_user <= 10_000:
            raise ValueError("personal knowledge document limit is invalid")
        self.path = path
        self.max_documents_per_user = max_documents_per_user
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS personal_documents (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    text_characters INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS personal_document_pages (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, document_id, page_number),
                    FOREIGN KEY (tenant_id, user_id, document_id)
                        REFERENCES personal_documents (tenant_id, user_id, document_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS personal_document_acl (
                    tenant_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    principal_type TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    permission TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, document_id, principal_type, principal_id)
                );
                CREATE TABLE IF NOT EXISTS personal_index_manifests (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    embedding_model TEXT,
                    embedding_dimension INTEGER,
                    index_status TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS personal_chunk_embeddings (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, document_id, chunk_id)
                );
                """
            )
            self._backfill_legacy_documents(connection)

    @staticmethod
    def _backfill_legacy_documents(connection: sqlite3.Connection) -> None:
        documents = connection.execute(
            """
            SELECT d.tenant_id, d.user_id, d.document_id, d.filename, d.created_at
            FROM personal_documents d
            LEFT JOIN personal_index_manifests m USING (tenant_id, user_id, document_id)
            WHERE m.document_id IS NULL
            """
        ).fetchall()
        for document in documents:
            pages = connection.execute(
                """
                SELECT page_number, text, extraction_method FROM personal_document_pages
                WHERE tenant_id = ? AND user_id = ? AND document_id = ? ORDER BY page_number
                """,
                (document["tenant_id"], document["user_id"], document["document_id"]),
            ).fetchall()
            normalized_pages = tuple(dict(page) for page in pages)
            corpus = _document_corpus(
                str(document["document_id"]),
                str(document["filename"]),
                normalized_pages,
                None,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO personal_document_acl (
                    tenant_id, document_id, principal_type, principal_id, permission, created_at
                ) VALUES (?, ?, 'user', ?, 'read', ?)
                """,
                (
                    document["tenant_id"],
                    document["document_id"],
                    document["user_id"],
                    document["created_at"],
                ),
            )
            connection.execute(
                """
                INSERT INTO personal_index_manifests (
                    tenant_id, user_id, document_id, content_sha256, chunk_count,
                    embedding_model, embedding_dimension, index_status, indexed_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 'lexical_ready', ?)
                """,
                (
                    document["tenant_id"],
                    document["user_id"],
                    document["document_id"],
                    sha256("\0".join(str(page["text"]) for page in pages).encode()).hexdigest(),
                    len(corpus.index_records()),
                    document["created_at"],
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
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

    def add_document(
        self,
        tenant_id: str,
        user_id: str,
        document: Mapping[str, Any],
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> dict[str, Any]:
        document_id, filename, pages = _validated_document(document)
        text_characters = sum(len(page["text"]) for page in pages)
        now = datetime.now(UTC).isoformat()
        corpus = _document_corpus(document_id, filename, pages, embedding_provider)
        vector_count = corpus.index_embeddings()
        records = corpus.index_records()
        embedding_dimension = len(records[0]["embedding"]) if vector_count else None
        content_sha256 = sha256("\0".join(page["text"] for page in pages).encode()).hexdigest()
        with self._lock, self._connection() as connection:
            existing = connection.execute(
                """
                SELECT filename, page_count, text_characters, created_at FROM personal_documents
                WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                """,
                (tenant_id, user_id, document_id),
            ).fetchone()
            if existing is None:
                count = connection.execute(
                    "SELECT COUNT(*) FROM personal_documents WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).fetchone()[0]
                if int(count) >= self.max_documents_per_user:
                    raise ValueError("personal knowledge document limit has been reached")
                connection.execute(
                    """
                    INSERT INTO personal_documents (
                        tenant_id, user_id, document_id, filename, page_count, text_characters, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tenant_id, user_id, document_id, filename, len(pages), text_characters, now),
                )
                connection.executemany(
                    """
                    INSERT INTO personal_document_pages (
                        tenant_id, user_id, document_id, page_number, text, extraction_method
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            tenant_id,
                            user_id,
                            document_id,
                            page["page_number"],
                            page["text"],
                            page["extraction_method"],
                        )
                        for page in pages
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO personal_document_acl (
                        tenant_id, document_id, principal_type, principal_id, permission, created_at
                    ) VALUES (?, ?, 'user', ?, 'read', ?)
                    """,
                    (tenant_id, document_id, user_id, now),
                )
                connection.execute(
                    """
                    INSERT INTO personal_index_manifests (
                        tenant_id, user_id, document_id, content_sha256, chunk_count,
                        embedding_model, embedding_dimension, index_status, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        user_id,
                        document_id,
                        content_sha256,
                        len(records),
                        embedding_provider.model_name if embedding_provider else None,
                        embedding_dimension,
                        "vector_ready" if embedding_provider else "lexical_ready",
                        now,
                    ),
                )
                if vector_count:
                    connection.executemany(
                        """
                        INSERT INTO personal_chunk_embeddings (
                            tenant_id, user_id, document_id, chunk_id, vector_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                tenant_id,
                                user_id,
                                document_id,
                                record["chunk_id"],
                                json.dumps(record["embedding"], separators=(",", ":")),
                            )
                            for record in records
                        ],
                    )
                result = {
                    "document_id": document_id,
                    "filename": filename,
                    "page_count": len(pages),
                    "text_characters": text_characters,
                    "created_at": now,
                    "index_status": "vector_ready" if embedding_provider else "lexical_ready",
                }
            else:
                manifest = connection.execute(
                    """
                    SELECT index_status FROM personal_index_manifests
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                    """,
                    (tenant_id, user_id, document_id),
                ).fetchone()
                result = {
                    "document_id": document_id,
                    **dict(existing),
                    "index_status": str(manifest["index_status"]),
                }
        return result

    def list_documents(self, tenant_id: str, user_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.filename, d.page_count, d.text_characters, d.created_at,
                       m.content_sha256, m.chunk_count, m.embedding_model,
                       m.embedding_dimension, m.index_status, m.indexed_at
                FROM personal_documents d
                JOIN personal_document_acl a
                  ON a.tenant_id = d.tenant_id AND a.document_id = d.document_id
                 AND a.principal_type = 'user' AND a.principal_id = ? AND a.permission = 'read'
                JOIN personal_index_manifests m USING (tenant_id, user_id, document_id)
                WHERE d.tenant_id = ? AND d.user_id = ?
                ORDER BY d.created_at DESC
                """,
                (user_id, tenant_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, tenant_id: str, user_id: str, document_id: str) -> bool:
        _validate_identifier(document_id, "document_id")
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM personal_documents WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                """,
                (tenant_id, user_id, document_id),
            )
            connection.execute(
                "DELETE FROM personal_document_acl WHERE tenant_id = ? AND document_id = ? AND principal_id = ?",
                (tenant_id, document_id, user_id),
            )
            connection.execute(
                "DELETE FROM personal_index_manifests WHERE tenant_id = ? AND user_id = ? AND document_id = ?",
                (tenant_id, user_id, document_id),
            )
            connection.execute(
                "DELETE FROM personal_chunk_embeddings WHERE tenant_id = ? AND user_id = ? AND document_id = ?",
                (tenant_id, user_id, document_id),
            )
            return cursor.rowcount == 1

    def corpus(
        self,
        tenant_id: str,
        user_id: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> InMemoryCorpus:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.filename, p.page_number, p.text, p.extraction_method
                FROM personal_documents d
                JOIN personal_document_pages p USING (tenant_id, user_id, document_id)
                JOIN personal_document_acl a
                  ON a.tenant_id = d.tenant_id AND a.document_id = d.document_id
                 AND a.principal_type = 'user' AND a.principal_id = ? AND a.permission = 'read'
                WHERE d.tenant_id = ? AND d.user_id = ?
                ORDER BY d.created_at, p.page_number
                """,
                (user_id, tenant_id, user_id),
            ).fetchall()
            embeddings = []
            if embedding_provider is not None:
                embeddings = connection.execute(
                    """
                    SELECT e.chunk_id, e.vector_json FROM personal_chunk_embeddings e
                    JOIN personal_index_manifests m USING (tenant_id, user_id, document_id)
                    WHERE e.tenant_id = ? AND e.user_id = ? AND m.embedding_model = ?
                    """,
                    (tenant_id, user_id, embedding_provider.model_name),
                ).fetchall()
        corpus = InMemoryCorpus(embedding_provider=embedding_provider)
        for row in rows:
            corpus.ingest(
                CorpusDocument.create(
                    title=f"{row['filename']}#page={row['page_number']}",
                    text=str(row["text"]),
                    metadata={
                        "document_id": str(row["document_id"]),
                        "corpus_record_id": f"{row['document_id']}:{row['page_number']}",
                        "document_title": str(row["filename"]),
                        "file_name": str(row["filename"]),
                        "source_page": int(row["page_number"]),
                        "extraction_method": str(row["extraction_method"]),
                        "kb_name": "personal_knowledge",
                    },
                )
            )
        for row in embeddings:
            corpus.restore_embedding(str(row["chunk_id"]), json.loads(row["vector_json"]))
        return corpus


def _document_corpus(
    document_id: str,
    filename: str,
    pages: Sequence[Mapping[str, Any]],
    embedding_provider: EmbeddingProvider | None,
) -> InMemoryCorpus:
    corpus = InMemoryCorpus(embedding_provider=embedding_provider)
    for page in pages:
        corpus.ingest(
            CorpusDocument.create(
                title=f"{filename}#page={page['page_number']}",
                text=str(page["text"]),
                metadata={
                    "document_id": document_id,
                    "corpus_record_id": f"{document_id}:{page['page_number']}",
                    "document_title": filename,
                    "file_name": filename,
                    "source_page": int(page["page_number"]),
                    "extraction_method": str(page["extraction_method"]),
                    "kb_name": "personal_knowledge",
                },
            )
        )
    return corpus


class PersonalKnowledgeClient:
    def __init__(
        self,
        store: SQLitePersonalKnowledgeBase,
        tenant_id: str,
        user_id: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.embedding_provider = embedding_provider
        self._corpus: InMemoryCorpus | None = None

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._corpus is None:
            self._corpus = self.store.corpus(
                self.tenant_id,
                self.user_id,
                embedding_provider=self.embedding_provider,
            )
        result = self._corpus.search_json(payload)
        trace = dict(result["trace"])
        trace["backend"] = f"sqlite_personal_knowledge_{trace['backend']}"
        trace["contract_version"] = 1
        return {"chunks": result["chunks"], "trace": trace}


def _validated_document(document: Mapping[str, Any]) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    document_id = document.get("document_id")
    filename = document.get("filename")
    raw_pages = document.get("pages")
    _validate_identifier(document_id, "document_id")
    if not isinstance(filename, str) or not filename.strip() or len(filename) > 500:
        raise ValueError("personal knowledge filename is invalid")
    if not isinstance(raw_pages, Sequence) or isinstance(raw_pages, (str, bytes)) or not raw_pages:
        raise ValueError("personal knowledge document must contain pages")
    pages: list[dict[str, Any]] = []
    for value in raw_pages:
        if not isinstance(value, Mapping):
            raise ValueError("personal knowledge page must be an object")
        page_number = value.get("page_number")
        text = value.get("text")
        extraction_method = value.get("extraction_method")
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ValueError("personal knowledge page number is invalid")
        if not isinstance(text, str) or not text.strip() or len(text) > 500_000:
            raise ValueError("personal knowledge page text is invalid")
        if extraction_method not in {"paddleocr", "mcp"}:
            raise ValueError("personal knowledge extraction method is invalid")
        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
                "extraction_method": extraction_method,
            }
        )
    if len({page["page_number"] for page in pages}) != len(pages):
        raise ValueError("personal knowledge pages contain duplicate numbers")
    json.dumps(pages, ensure_ascii=False, allow_nan=False)
    return str(document_id), filename.strip(), tuple(pages)


def _validate_identifier(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or "/" in value
        or ".." in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"personal knowledge {name} is invalid")
