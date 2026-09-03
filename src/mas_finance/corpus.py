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
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

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


class DocumentTokenizer:
    """Tokenizer used to make document windows match the embedding model."""

    def __init__(self, tokenizer_path: Path) -> None:
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"document tokenizer does not exist: {tokenizer_path}")
        self.path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, end)
            for start, end in self._tokenizer.encode(text, add_special_tokens=False).offsets
            if end > start
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class InMemoryCorpus:
    """Tenant/run-local BM25 index with optional embedding/RRF retrieval."""

    def __init__(
        self,
        *,
        tokenizer: DocumentTokenizer,
        chunk_tokens: int = 1024,
        overlap_tokens: int = 256,
        embedding_provider: EmbeddingProvider | None = None,
        minimum_vector_similarity: float = 0.5,
    ) -> None:
        if chunk_tokens < 1 or not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("invalid chunk configuration")
        if not math.isfinite(minimum_vector_similarity) or not -1 <= minimum_vector_similarity <= 1:
            raise ValueError("minimum vector similarity must be between -1 and 1")
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_provider = embedding_provider
        self.minimum_vector_similarity = minimum_vector_similarity
        self._chunks: list[_Chunk] = []
        self._embeddings: dict[str, tuple[float, ...]] = {}

    def ingest(self, document: CorpusDocument) -> int:
        count = 0
        offsets = self.tokenizer.token_offsets(document.text)
        token_start = 0
        index = 0
        while token_start < len(offsets):
            token_end = min(token_start + self.chunk_tokens, len(offsets))
            token_end = _natural_token_end(document.text, offsets, token_start, token_end, self.chunk_tokens)
            start = offsets[token_start][0]
            end = offsets[token_end - 1][1]
            content = document.text[start:end]
            while self.tokenizer.count_tokens(content) > self.chunk_tokens:
                token_end -= 1
                end = offsets[token_end - 1][1]
                content = document.text[start:end]
            chunk_id = (
                f"{document.document_id[:16]}-t{self.chunk_tokens}o{self.overlap_tokens}-{index}"
            )
            metadata = {
                **dict(document.metadata),
                "document_id": document.metadata.get("document_id", document.document_id),
                "corpus_record_id": document.document_id,
                "document_title": document.metadata.get("document_title", document.title),
                "file_name": document.metadata.get("file_name", document.title),
                "global_start": start,
                "global_end": end,
                "token_start": token_start,
                "token_end": token_end,
            }
            self._chunks.append(_Chunk(chunk_id, content, metadata, Counter(_tokens(content))))
            count += 1
            if token_end >= len(offsets):
                break
            token_start = token_end - self.overlap_tokens
            index += 1
        return count

    def ingest_blocks(
        self,
        *,
        document_id: str,
        title: str,
        blocks: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        grouped: list[tuple[str, str, int, int, str | None]] = []
        text_parts: list[str] = []
        text_pages: list[int] = []
        paragraph_title: str | None = None
        previous_page: int | None = None

        def flush_text() -> None:
            if text_parts:
                grouped.append(
                    ("text", "\n\n".join(text_parts), min(text_pages), max(text_pages), paragraph_title)
                )
                text_parts.clear()
                text_pages.clear()

        for block in blocks:
            label = str(block["label"])
            content = str(block["content"]).strip()
            page_number = int(block["page_number"])
            next_title = str(block.get("paragraph_title") or "").strip() or None
            if label in {"table", "chart"}:
                flush_text()
                parts = _table_parts(content, self.tokenizer, self.chunk_tokens) if label == "table" else (content,)
                grouped.extend((label, part, page_number, page_number, next_title) for part in parts)
                paragraph_title = next_title
                previous_page = page_number
                continue
            if text_parts and (next_title != paragraph_title or page_number != previous_page):
                flush_text()
            paragraph_title = next_title
            text_parts.append(content)
            text_pages.append(page_number)
            previous_page = page_number
        flush_text()

        count = 0
        for index, (label, content, page_start, page_end, heading) in enumerate(grouped):
            count += self.ingest(
                CorpusDocument.create(
                    title=f"{title}#block={index}",
                    text=content,
                    metadata={
                        **dict(metadata or {}),
                        "document_id": document_id,
                        "document_title": title,
                        "file_name": title,
                        "source_page": page_start,
                        "source_page_end": page_end,
                        "block_label": label,
                        "paragraph_title": heading,
                    },
                )
            )
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
        raw_vector_candidate_count = 0
        embedding_batch_count = 0
        if search_mode != "lexical" and candidates:
            vector_by_id, embedding_batch_count = self._vector_scores(candidates, query)
            raw_vector_candidate_count = len(vector_by_id)
            vector_by_id = {
                chunk_id: score
                for chunk_id, score in vector_by_id.items()
                if score >= self.minimum_vector_similarity
            }
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
                if lexical_rank is None and vector_rank is None:
                    continue
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
        selected_chunk_count = len(selected)
        selected = _merge_overlapping_chunks(selected)
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
                "vector_candidate_count_before_threshold": raw_vector_candidate_count,
                "minimum_vector_similarity": self.minimum_vector_similarity,
                "returned_count": len(chunks),
                "selected_chunk_count": selected_chunk_count,
                "document_diversification": diversify_documents,
                "embedding_backend": (
                    self.embedding_provider.backend_name if self.embedding_provider is not None else None
                ),
                "embedding_model": (
                    self.embedding_provider.model_name if self.embedding_provider is not None else None
                ),
                "embedding_batch_count": embedding_batch_count,
            },
        }

    def _vector_scores(self, chunks: list[_Chunk], query: str) -> tuple[dict[str, float], int]:
        provider = self.embedding_provider
        if provider is None:
            raise ValueError("vector search requires a configured embedding provider")
        missing = [chunk for chunk in chunks if chunk.chunk_id not in self._embeddings]
        batch_count = 0
        for start in range(0, len(missing), 128):
            batch = missing[start : start + 128]
            vectors = provider.embed_texts([chunk.content for chunk in batch])
            batch_count += 1
            if len(vectors) != len(batch):
                raise ValueError("embedding provider returned an unexpected vector count")
            normalized_vectors = [_normalized_vector(vector) for vector in vectors]
            for chunk, vector in zip(batch, normalized_vectors, strict=True):
                self._embeddings[chunk.chunk_id] = vector
        query_vectors = provider.embed_texts([query])
        batch_count += 1
        if len(query_vectors) != 1:
            raise ValueError("embedding provider returned an unexpected query vector count")
        query_vector = _normalized_vector(query_vectors[0])
        if any(len(vector) != len(query_vector) for vector in self._embeddings.values()):
            raise ValueError("embedding model dimension changed within the corpus")
        return (
            {
                chunk.chunk_id: sum(
                    value * query_value
                    for value, query_value in zip(self._embeddings[chunk.chunk_id], query_vector, strict=True)
                )
                for chunk in chunks
            },
            batch_count,
        )

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


def _natural_token_end(
    text: str,
    offsets: Sequence[tuple[int, int]],
    token_start: int,
    hard_end: int,
    chunk_tokens: int,
) -> int:
    if hard_end >= len(offsets):
        return hard_end
    minimum_end = min(hard_end - 1, token_start + max(1, int(chunk_tokens * 0.8)))
    search_start = offsets[minimum_end][0]
    search_end = offsets[hard_end - 1][1]
    window = text[search_start:search_end]
    boundaries = list(re.finditer(r"\n{2,}|\n|[。！？.!?](?:[\"'’”」』])?\s*", window))
    if not boundaries:
        return hard_end
    boundary = search_start + boundaries[-1].end()
    for index in range(hard_end - 1, minimum_end - 1, -1):
        if offsets[index][1] <= boundary:
            return index + 1
    return hard_end


def _table_parts(content: str, tokenizer: DocumentTokenizer, chunk_tokens: int) -> tuple[str, ...]:
    if tokenizer.count_tokens(content) <= chunk_tokens:
        return (content,)
    html_rows = re.findall(r"<tr[^>]*>.*?</tr>", content, flags=re.IGNORECASE | re.DOTALL)
    if len(html_rows) > 1:
        first = content.find(html_rows[0])
        last = content.rfind(html_rows[-1]) + len(html_rows[-1])
        header = content[:first] + html_rows[0]
        return _pack_table_rows(header, html_rows[1:], content[last:], tokenizer, chunk_tokens)
    lines = content.splitlines(keepends=True)
    if len(lines) < 2:
        return (content,)
    header_count = 2 if re.fullmatch(r"\s*\|?(?:\s*:?-+:?\s*\|)+\s*", lines[1]) else 1
    return _pack_table_rows("".join(lines[:header_count]), lines[header_count:], "", tokenizer, chunk_tokens)


def _pack_table_rows(
    header: str,
    rows: Sequence[str],
    suffix: str,
    tokenizer: DocumentTokenizer,
    chunk_tokens: int,
) -> tuple[str, ...]:
    fixed_tokens = tokenizer.count_tokens(header + suffix)
    if fixed_tokens >= chunk_tokens:
        return _token_slices(header + "".join(rows) + suffix, tokenizer, chunk_tokens)
    parts: list[str] = []
    current = header
    for row in rows:
        candidate = current + row + suffix
        if tokenizer.count_tokens(candidate) <= chunk_tokens:
            current += row
            continue
        if current != header:
            parts.append(current + suffix)
            current = header
        for row_part in _token_slices(row, tokenizer, chunk_tokens - fixed_tokens):
            candidate = header + row_part + suffix
            if tokenizer.count_tokens(candidate) > chunk_tokens:
                raise ValueError("table row could not be split within the token limit")
            parts.append(candidate)
    if current != header:
        parts.append(current + suffix)
    return tuple(parts)


def _token_slices(text: str, tokenizer: DocumentTokenizer, limit: int) -> tuple[str, ...]:
    offsets = tokenizer.token_offsets(text)
    return tuple(
        text[offsets[start][0] : offsets[min(start + limit, len(offsets)) - 1][1]]
        for start in range(0, len(offsets), limit)
    )


def _merge_overlapping_chunks(
    selected: list[tuple[_Chunk, float, Mapping[str, float | int]]],
) -> list[tuple[_Chunk, float, Mapping[str, float | int]]]:
    grouped: dict[str, list[tuple[int, _Chunk, float, Mapping[str, float | int]]]] = {}
    for order, (chunk, score, score_parts) in enumerate(selected):
        origin = str(chunk.metadata.get("corpus_record_id") or chunk.chunk_id)
        grouped.setdefault(origin, []).append((order, chunk, score, score_parts))

    merged: list[tuple[int, _Chunk, float, Mapping[str, float | int]]] = []
    for items in grouped.values():
        items.sort(key=lambda item: (int(item[1].metadata.get("global_start") or 0), item[0]))
        current_order, current_chunk, current_score, current_parts = items[0]
        merged_ids = [current_chunk.chunk_id]
        for next_order, next_chunk, next_score, next_parts in items[1:]:
            current_end = int(current_chunk.metadata.get("global_end") or 0)
            next_start = int(next_chunk.metadata.get("global_start") or 0)
            next_end = int(next_chunk.metadata.get("global_end") or 0)
            if next_start >= current_end or next_end <= next_start:
                metadata = {**dict(current_chunk.metadata), "merged_chunk_ids": merged_ids}
                merged.append(
                    (
                        current_order,
                        _Chunk(current_chunk.chunk_id, current_chunk.content, metadata, current_chunk.term_counts),
                        current_score,
                        current_parts,
                    )
                )
                current_order, current_chunk, current_score, current_parts = (
                    next_order,
                    next_chunk,
                    next_score,
                    next_parts,
                )
                merged_ids = [next_chunk.chunk_id]
                continue
            overlap_characters = max(0, current_end - next_start)
            content = current_chunk.content + next_chunk.content[overlap_characters:]
            merged_ids.append(next_chunk.chunk_id)
            metadata = {
                **dict(current_chunk.metadata),
                "global_end": max(current_end, next_end),
                "token_end": max(
                    int(current_chunk.metadata.get("token_end") or 0),
                    int(next_chunk.metadata.get("token_end") or 0),
                ),
                "merged_chunk_ids": list(merged_ids),
            }
            current_chunk = _Chunk(
                sha256("\0".join(merged_ids).encode()).hexdigest()[:32],
                content,
                metadata,
                Counter(_tokens(content)),
            )
            current_order = min(current_order, next_order)
            if next_score > current_score:
                current_score, current_parts = next_score, next_parts
        metadata = {**dict(current_chunk.metadata), "merged_chunk_ids": merged_ids}
        merged.append(
            (
                current_order,
                _Chunk(current_chunk.chunk_id, current_chunk.content, metadata, current_chunk.term_counts),
                current_score,
                current_parts,
            )
        )
    merged.sort(key=lambda item: item[0])
    return [(chunk, score, score_parts) for _order, chunk, score, score_parts in merged]


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
