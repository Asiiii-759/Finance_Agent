"""Persistent, user-scoped personal document knowledge base."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .corpus import CorpusDocument, DocumentTokenizer, InMemoryCorpus
from .embeddings import EmbeddingProvider

CHUNKING_VERSION = "token-1024-overlap-256-structured-v1"


class SQLitePersonalKnowledgeBase:
    """Store parsed page text; raw uploads remain outside this database."""

    def __init__(
        self,
        path: Path,
        *,
        tokenizer: DocumentTokenizer,
        max_documents_per_user: int = 100,
    ) -> None:
        if not 1 <= max_documents_per_user <= 10_000:
            raise ValueError("personal knowledge document limit is invalid")
        self.path = path
        self.tokenizer = tokenizer
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
                    blocks_json TEXT NOT NULL DEFAULT '[]',
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
                    chunking_version TEXT NOT NULL DEFAULT 'legacy',
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
            page_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(personal_document_pages)").fetchall()
            }
            if "blocks_json" not in page_columns:
                connection.execute(
                    "ALTER TABLE personal_document_pages ADD COLUMN blocks_json TEXT NOT NULL DEFAULT '[]'"
                )
            manifest_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(personal_index_manifests)").fetchall()
            }
            if "chunking_version" not in manifest_columns:
                connection.execute(
                    "ALTER TABLE personal_index_manifests "
                    "ADD COLUMN chunking_version TEXT NOT NULL DEFAULT 'legacy'"
                )
            self._backfill_legacy_documents(connection)

    def _backfill_legacy_documents(self, connection: sqlite3.Connection) -> None:
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
                SELECT page_number, text, extraction_method, blocks_json FROM personal_document_pages
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
                self.tokenizer,
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
                    chunking_version, embedding_model, embedding_dimension, index_status, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'lexical_ready', ?)
                """,
                (
                    document["tenant_id"],
                    document["user_id"],
                    document["document_id"],
                    sha256("\0".join(str(page["text"]) for page in pages).encode()).hexdigest(),
                    len(corpus.index_records()),
                    CHUNKING_VERSION,
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
        corpus = _document_corpus(document_id, filename, pages, embedding_provider, self.tokenizer)
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
                        tenant_id, user_id, document_id, page_number, text, extraction_method, blocks_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            tenant_id,
                            user_id,
                            document_id,
                            page["page_number"],
                            page["text"],
                            page["extraction_method"],
                            json.dumps(page["blocks"], ensure_ascii=False, separators=(",", ":")),
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
                        chunking_version, embedding_model, embedding_dimension, index_status, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        user_id,
                        document_id,
                        content_sha256,
                        len(records),
                        CHUNKING_VERSION,
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
                       m.content_sha256, m.chunk_count, m.chunking_version, m.embedding_model,
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

    def vector_index_ready(self, tenant_id: str, user_id: str, embedding_model: str) -> bool:
        documents = self.list_documents(tenant_id, user_id)
        return bool(documents) and all(
            document["index_status"] == "vector_ready"
            and document["chunking_version"] == CHUNKING_VERSION
            and document["embedding_model"] == embedding_model
            for document in documents
        )

    def reindex_documents(
        self,
        tenant_id: str,
        user_id: str,
        embedding_provider: EmbeddingProvider,
    ) -> dict[str, int | str]:
        documents = self.list_documents(tenant_id, user_id)
        indexed_chunks = 0
        for document in documents:
            document_id = str(document["document_id"])
            with self._lock, self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT page_number, text, extraction_method, blocks_json FROM personal_document_pages
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ? ORDER BY page_number
                    """,
                    (tenant_id, user_id, document_id),
                ).fetchall()
            corpus = _document_corpus(
                document_id,
                str(document["filename"]),
                tuple(dict(row) for row in rows),
                embedding_provider,
                self.tokenizer,
            )
            corpus.index_embeddings()
            records = corpus.index_records()
            dimension = len(records[0]["embedding"]) if records else None
            if dimension is None or any(record["embedding"] is None for record in records):
                raise ValueError("personal knowledge vector indexing produced incomplete records")
            indexed_at = datetime.now(UTC).isoformat()
            with self._lock, self._connection() as connection:
                connection.execute(
                    """
                    DELETE FROM personal_chunk_embeddings
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                    """,
                    (tenant_id, user_id, document_id),
                )
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
                connection.execute(
                    """
                    UPDATE personal_index_manifests
                    SET chunk_count = ?, chunking_version = ?, embedding_model = ?, embedding_dimension = ?,
                        index_status = 'vector_ready', indexed_at = ?
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                    """,
                    (
                        len(records),
                        CHUNKING_VERSION,
                        embedding_provider.model_name,
                        dimension,
                        indexed_at,
                        tenant_id,
                        user_id,
                        document_id,
                    ),
                )
            indexed_chunks += len(records)
        return {
            "documents": len(documents),
            "chunks": indexed_chunks,
            "embedding_model": embedding_provider.model_name,
        }

    def corpus(
        self,
        tenant_id: str,
        user_id: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        minimum_vector_similarity: float = 0.5,
    ) -> InMemoryCorpus:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.filename, p.page_number, p.text, p.extraction_method,
                       p.blocks_json
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
        corpus = InMemoryCorpus(
            tokenizer=self.tokenizer,
            embedding_provider=embedding_provider,
            minimum_vector_similarity=minimum_vector_similarity,
        )
        documents: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            documents.setdefault((str(row["document_id"]), str(row["filename"])), []).append(
                {
                    "page_number": int(row["page_number"]),
                    "text": str(row["text"]),
                    "extraction_method": str(row["extraction_method"]),
                    "blocks_json": str(row["blocks_json"]),
                }
            )
        for (document_id, filename), pages in documents.items():
            blocks = [
                block
                for page in pages
                for block in json.loads(str(page.get("blocks_json") or "[]"))
            ]
            if blocks:
                corpus.ingest_blocks(
                    document_id=document_id,
                    title=filename,
                    blocks=blocks,
                    metadata={
                        "document_id": document_id,
                        "document_title": filename,
                        "file_name": filename,
                        "extraction_method": str(pages[0]["extraction_method"]),
                        "kb_name": "personal_knowledge",
                    },
                )
                continue
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
        current_chunk_ids = {str(record["chunk_id"]) for record in corpus.index_records()}
        for row in embeddings:
            chunk_id = str(row["chunk_id"])
            if chunk_id in current_chunk_ids:
                corpus.restore_embedding(chunk_id, json.loads(row["vector_json"]))
        if embedding_provider is not None and any(
            record["embedding"] is None for record in corpus.index_records()
        ):
            raise ValueError("personal knowledge vector index is not ready; run reindex first")
        return corpus


def _document_corpus(
    document_id: str,
    filename: str,
    pages: Sequence[Mapping[str, Any]],
    embedding_provider: EmbeddingProvider | None,
    tokenizer: DocumentTokenizer,
) -> InMemoryCorpus:
    corpus = InMemoryCorpus(tokenizer=tokenizer, embedding_provider=embedding_provider)
    blocks = [
        block
        for page in pages
        for block in (
            page.get("blocks")
            or json.loads(str(page.get("blocks_json") or "[]"))
        )
    ]
    if blocks:
        corpus.ingest_blocks(
            document_id=document_id,
            title=filename,
            blocks=blocks,
            metadata={
                "document_id": document_id,
                "document_title": filename,
                "file_name": filename,
                "extraction_method": str(pages[0]["extraction_method"]),
                "kb_name": "personal_knowledge",
            },
        )
        return corpus
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
        minimum_vector_similarity: float = 0.5,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.embedding_provider = embedding_provider
        self.minimum_vector_similarity = minimum_vector_similarity
        self._corpus: InMemoryCorpus | None = None

    def search_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._corpus is None:
            self._corpus = self.store.corpus(
                self.tenant_id,
                self.user_id,
                embedding_provider=self.embedding_provider,
                minimum_vector_similarity=self.minimum_vector_similarity,
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
        raw_blocks = value.get("blocks") or []
        if isinstance(raw_blocks, (str, bytes)) or not isinstance(raw_blocks, Sequence):
            raise ValueError("personal knowledge page blocks must be an array")
        blocks: list[dict[str, Any]] = []
        for raw_block in raw_blocks:
            if not isinstance(raw_block, Mapping):
                raise ValueError("personal knowledge block must be an object")
            label = raw_block.get("label")
            content = raw_block.get("content")
            block_page = raw_block.get("page_number")
            order = raw_block.get("order")
            paragraph_title = raw_block.get("paragraph_title")
            bbox = raw_block.get("bbox")
            if label not in {"text", "heading", "table", "chart"}:
                raise ValueError("personal knowledge block label is invalid")
            if not isinstance(content, str) or not content.strip() or len(content) > 500_000:
                raise ValueError("personal knowledge block content is invalid")
            if block_page != page_number:
                raise ValueError("personal knowledge block page does not match its parent page")
            if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                raise ValueError("personal knowledge block order is invalid")
            if paragraph_title is not None and (
                not isinstance(paragraph_title, str) or len(paragraph_title) > 1_000
            ):
                raise ValueError("personal knowledge block paragraph title is invalid")
            if bbox is not None and (
                isinstance(bbox, (str, bytes))
                or not isinstance(bbox, Sequence)
                or len(bbox) != 4
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bbox)
                or any(not math.isfinite(float(item)) for item in bbox)
            ):
                raise ValueError("personal knowledge block bbox is invalid")
            blocks.append(
                {
                    "label": label,
                    "content": content.strip(),
                    "page_number": block_page,
                    "order": order,
                    "paragraph_title": paragraph_title,
                    "bbox": list(bbox) if bbox is not None else None,
                }
            )
        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
                "extraction_method": extraction_method,
                "blocks": blocks,
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
