"""Small, dependency-light document corpus used by the canonical agent.

The index deliberately exposes the same JSON retrieval contract as remote RAG
providers. It is suitable for uploaded/internal documents and can be replaced by
an embedding or managed-search backend without changing the agent core.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .embeddings import EmbeddingProvider

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for item in _TOKEN.findall(text):
        if "\u4e00" <= item[0] <= "\u9fff":
            tokens.extend(item[index : index + 2] for index in range(max(1, len(item) - 1)))
        elif len(item) > 1:
            tokens.append(item.lower())
    return tokens


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    title: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, title: str, text: str, metadata: Mapping[str, Any] | None = None) -> CorpusDocument:
        normalized = text.strip()
        if not title.strip() or not normalized:
            raise ValueError("document title and text are required")
        identifier = sha256(f"{title}\0{normalized}".encode()).hexdigest()
        return cls(identifier, title.strip(), normalized, dict(metadata or {}))


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    content: str
    metadata: Mapping[str, Any]
    term_counts: Counter[str]


class InMemoryCorpus:
    """Tenant/run-local BM25 index with optional embedding/RRF retrieval."""

    def __init__(
        self,
        *,
        chunk_chars: int = 1600,
        overlap_chars: int = 200,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if chunk_chars < 200 or not 0 <= overlap_chars < chunk_chars:
            raise ValueError("invalid chunk configuration")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.embedding_provider = embedding_provider
        self._chunks: list[_Chunk] = []
        self._embeddings: dict[str, tuple[float, ...]] = {}

    def ingest(self, document: CorpusDocument) -> int:
        count = 0
        step = self.chunk_chars - self.overlap_chars
        for index, start in enumerate(range(0, len(document.text), step)):
            content = document.text[start : start + self.chunk_chars].strip()
            if not content:
                continue
            chunk_id = f"{document.document_id[:16]}-{index}"
            metadata = {
                **dict(document.metadata),
                "document_id": document.metadata.get("document_id", document.document_id),
                "corpus_record_id": document.document_id,
                "document_title": document.metadata.get("document_title", document.title),
                "file_name": document.metadata.get("file_name", document.title),
                "global_start": start,
                "global_end": start + len(content),
            }
            self._chunks.append(_Chunk(chunk_id, content, metadata, Counter(_tokens(content))))
            count += 1
            if start + self.chunk_chars >= len(document.text):
                break
        return count

    def index_embeddings(self) -> int:
        """Materialize document vectors so a persistence layer can save them."""
        provider = self.embedding_provider
        if provider is None or not self._chunks:
            return 0
        missing = [chunk for chunk in self._chunks if chunk.chunk_id not in self._embeddings]
        if not missing:
            return 0
        dimension: int | None = None
        for start in range(0, len(missing), 128):
            batch = missing[start : start + 128]
            vectors = provider.embed_texts([chunk.content for chunk in batch])
            if len(vectors) != len(batch):
                raise ValueError("embedding provider returned an unexpected vector count")
            normalized = [_normalized_vector(vector) for vector in vectors]
            batch_dimensions = {len(vector) for vector in normalized}
            if len(batch_dimensions) != 1 or (dimension is not None and dimension not in batch_dimensions):
                raise ValueError("embedding model returned inconsistent vector dimensions")
            dimension = len(normalized[0])
            for chunk, vector in zip(batch, normalized, strict=True):
                self._embeddings[chunk.chunk_id] = vector
        return len(missing)

    def index_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": dict(chunk.metadata),
                "embedding": list(self._embeddings[chunk.chunk_id]) if chunk.chunk_id in self._embeddings else None,
            }
            for chunk in self._chunks
        )

    def restore_embedding(self, chunk_id: str, vector: Sequence[float]) -> None:
        if chunk_id not in {chunk.chunk_id for chunk in self._chunks}:
            raise ValueError("embedding chunk does not exist in the corpus")
        self._embeddings[chunk_id] = _normalized_vector(vector)

    def search_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_query = payload.get("query")
        raw_top_k = payload.get("top_k", 5)
        raw_filters = payload.get("filters") or {}
        search_mode = payload.get("search_mode", "lexical")
        rerank = payload.get("rerank", False)
        diversify_documents = payload.get("diversify_documents", False)
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError("query is required")
        query = raw_query.strip()
        if len(query) > 8_000:
            raise ValueError("query exceeds length limit")
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
            raise ValueError("top_k must be an integer")
        top_k = raw_top_k
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not isinstance(raw_filters, Mapping) or len(raw_filters) > 50:
            raise ValueError("filters must be a bounded object")
        if search_mode not in {"lexical", "vector", "hybrid", "rrf"}:
            raise ValueError("search_mode must be lexical, vector, hybrid, or rrf")
        if not isinstance(rerank, bool):
            raise ValueError("rerank must be a boolean")
        if rerank:
            raise ValueError("rerank is not configured for the in-memory corpus")
        if not isinstance(diversify_documents, bool):
            raise ValueError("diversify_documents must be a boolean")
        if search_mode != "lexical" and self.embedding_provider is None:
            raise ValueError(f"{search_mode} search requires a configured embedding provider")
        filters = dict(raw_filters)
        candidates = [chunk for chunk in self._chunks if _matches(chunk.metadata, filters)]
        query_terms = Counter(_tokens(query))
        lexical_scores = self._score(candidates, query_terms)
        lexical_by_id = {
            chunk.chunk_id: score
            for chunk, score in zip(candidates, lexical_scores, strict=True)
            if score > 0
        }
        lexical_ranked = sorted(
            ((chunk, lexical_by_id[chunk.chunk_id]) for chunk in candidates if chunk.chunk_id in lexical_by_id),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        vector_by_id: dict[str, float] = {}
        if search_mode != "lexical" and candidates:
            vector_by_id = self._vector_scores(candidates, query)
        vector_ranked = sorted(
            ((chunk, vector_by_id[chunk.chunk_id]) for chunk in candidates if chunk.chunk_id in vector_by_id),
            key=lambda item: (-item[1], item[0].chunk_id),
        )

        ranked: list[tuple[_Chunk, float, Mapping[str, float | int]]] = []
        if search_mode == "lexical":
            for chunk, score in lexical_ranked:
                ranked.append((chunk, score, {"bm25": round(score, 6)}))
        elif search_mode == "vector":
            for chunk, score in vector_ranked:
                ranked.append((chunk, score, {"cosine": round(score, 6)}))
        else:
            lexical_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(lexical_ranked, start=1)}
            vector_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(vector_ranked, start=1)}
            fused: list[tuple[_Chunk, float, dict[str, float | int]]] = []
            for chunk in candidates:
                score = 0.0
                scores: dict[str, float | int] = {}
                lexical_rank = lexical_ranks.get(chunk.chunk_id)
                vector_rank = vector_ranks.get(chunk.chunk_id)
                if lexical_rank is not None:
                    score += 1 / (60 + lexical_rank)
                    scores["bm25"] = round(lexical_by_id[chunk.chunk_id], 6)
                    scores["lexical_rank"] = lexical_rank
                if vector_rank is not None:
                    score += 1 / (60 + vector_rank)
                    scores["cosine"] = round(vector_by_id[chunk.chunk_id], 6)
                    scores["vector_rank"] = vector_rank
                scores["rrf"] = round(score, 8)
                fused.append((chunk, score, scores))
            ranked = sorted(fused, key=lambda item: (-item[1], item[0].chunk_id))

        selected: list[tuple[_Chunk, float, Mapping[str, float | int]]] = []
        if diversify_documents:
            selected_chunk_ids: set[str] = set()
            selected_documents: set[str] = set()
            for chunk, score, score_parts in ranked:
                document_id = str(
                    chunk.metadata.get("document_id") or chunk.metadata.get("corpus_record_id") or chunk.chunk_id
                )
                if document_id in selected_documents:
                    continue
                selected.append((chunk, score, score_parts))
                selected_chunk_ids.add(chunk.chunk_id)
                selected_documents.add(document_id)
                if len(selected) == top_k:
                    break
            if len(selected) < top_k:
                for chunk, score, score_parts in ranked:
                    if chunk.chunk_id in selected_chunk_ids:
                        continue
                    selected.append((chunk, score, score_parts))
                    if len(selected) == top_k:
                        break
        else:
            selected = ranked[:top_k]
        chunks = []
        for rank, (chunk, score, score_parts) in enumerate(selected, start=1):
            chunks.append(
                {
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "rank": rank,
                    "score": round(score, 6),
                    "scores": dict(score_parts),
                    "metadata": dict(chunk.metadata),
                }
            )
        backend = {
            "lexical": "in_memory_bm25",
            "vector": "in_memory_vector",
            "hybrid": "in_memory_hybrid_rrf",
            "rrf": "in_memory_hybrid_rrf",
        }[search_mode]
        return {
            "chunks": chunks,
            "trace": {
                "backend": backend,
                "search_mode": search_mode,
                "fusion": "rrf" if search_mode in {"hybrid", "rrf"} else "none",
                "candidate_count": len(candidates),
                "lexical_candidate_count": len(lexical_ranked),
                "vector_candidate_count": len(vector_ranked),
                "returned_count": len(chunks),
                "document_diversification": diversify_documents,
                "embedding_backend": (
                    self.embedding_provider.backend_name if self.embedding_provider is not None else None
                ),
                "embedding_model": (
                    self.embedding_provider.model_name if self.embedding_provider is not None else None
                ),
            },
        }

    def _vector_scores(self, chunks: list[_Chunk], query: str) -> dict[str, float]:
        provider = self.embedding_provider
        if provider is None:
            raise ValueError("vector search requires a configured embedding provider")
        missing = [chunk for chunk in chunks if chunk.chunk_id not in self._embeddings]
        vectors = provider.embed_texts([chunk.content for chunk in missing] + [query])
        if len(vectors) != len(missing) + 1:
            raise ValueError("embedding provider returned an unexpected vector count")
        normalized_vectors = [_normalized_vector(vector) for vector in vectors]
        query_vector = normalized_vectors[-1]
        for chunk, vector in zip(missing, normalized_vectors[:-1], strict=True):
            self._embeddings[chunk.chunk_id] = vector
        if any(len(vector) != len(query_vector) for vector in self._embeddings.values()):
            raise ValueError("embedding model dimension changed within the corpus")
        return {
            chunk.chunk_id: sum(
                value * query_value
                for value, query_value in zip(self._embeddings[chunk.chunk_id], query_vector, strict=True)
            )
            for chunk in chunks
        }

    @staticmethod
    def _score(chunks: list[_Chunk], query: Counter[str]) -> list[float]:
        if not chunks or not query:
            return [0.0] * len(chunks)
        doc_count = len(chunks)
        frequencies = Counter(term for chunk in chunks for term in set(chunk.term_counts))
        avg_length = sum(sum(chunk.term_counts.values()) for chunk in chunks) / doc_count
        scores: list[float] = []
        for chunk in chunks:
            length = max(1, sum(chunk.term_counts.values()))
            score = 0.0
            for term, query_frequency in query.items():
                tf = chunk.term_counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (doc_count - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                denominator = tf + 1.2 * (0.25 + 0.75 * length / max(avg_length, 1))
                score += idf * (tf * 2.2 / denominator) * query_frequency
            scores.append(score)
        return scores


def _matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items() if value is not None)


def _normalized_vector(raw_vector: Any) -> tuple[float, ...]:
    if isinstance(raw_vector, (str, bytes)) or not isinstance(raw_vector, Sequence):
        raise ValueError("embedding provider vector must be a numeric sequence")
    if not 2 <= len(raw_vector) <= 65_536:
        raise ValueError("embedding provider vector dimension is invalid")
    vector = tuple(
        float(value)
        for value in raw_vector
        if not isinstance(value, bool) and isinstance(value, (int, float))
    )
    if len(vector) != len(raw_vector) or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding provider vector must contain finite numeric data")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("embedding provider returned a zero vector")
    return tuple(value / norm for value in vector)
