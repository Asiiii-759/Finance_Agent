from __future__ import annotations

import json
import unittest
from pathlib import Path

import httpx

from mas_finance.corpus import CorpusDocument, DocumentTokenizer, InMemoryCorpus
from mas_finance.embeddings import HTTPEmbeddingClient
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.retrieval import RetrievalEvidenceAdapter, retrieval_harness_tool

TOKENIZER = DocumentTokenizer(Path(".runtime/models/bge-m3/tokenizer.json"))


class SemanticEmbedding:
    backend_name = "semantic-fixture"
    model_name = "fixture-v1"

    def __init__(self, *, network_access: bool = False) -> None:
        self.network_access = network_access
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if any(term in normalized for term in ("liquidity", "ample cash", "资金充裕")):
                vectors.append((1.0, 0.0, 0.0))
            elif any(term in normalized for term in ("profit", "margin", "利润")):
                vectors.append((0.0, 1.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return tuple(vectors)


class EmbeddingAndHybridRetrievalTests(unittest.TestCase):
    def test_http_embedding_boundary_is_fixed_bounded_and_ordered(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, httpx.URL("https://embedding.example.test/v1/embeddings"))
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            self.assertEqual(
                json.loads(request.content),
                {"model": "bge-m3", "input": ["first", "second"]},
            )
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
                headers={"content-type": "application/json"},
            )

        client = HTTPEmbeddingClient(
            "https://embedding.example.test/v1/embeddings",
            "bge-m3",
            api_key="secret",
            transport=httpx.MockTransport(handler),
        )
        self.assertNotIn("secret", repr(client))
        self.assertEqual(client.embed_texts(("first", "second")), ((1.0, 0.0), (0.0, 1.0)))

        with self.assertRaisesRegex(ValueError, "HTTPS"):
            HTTPEmbeddingClient("http://embedding.example.test/v1/embeddings", "bge-m3")

        local = HTTPEmbeddingClient(
            "http://127.0.0.1:8001/v1/embeddings",
            "BAAI/bge-m3",
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual(local.endpoint, "http://127.0.0.1:8001/v1/embeddings")

    def test_http_embedding_rejects_malformed_vectors(self) -> None:
        responses = (
            ({"data": []}, "count"),
            ({"data": [{"index": 0, "embedding": [0.0, 0.0]}]}, "non-zero"),
            (b'{"data":[{"index":0,"embedding":[1.0,NaN]}]}', "finite"),
            ({"data": [{"index": 0, "embedding": [1.0]}]}, "dimension"),
        )
        for payload, error in responses:
            with self.subTest(error=error):
                client = HTTPEmbeddingClient(
                    "https://embedding.example.test/v1/embeddings",
                    "bge-m3",
                    transport=httpx.MockTransport(
                        lambda _request, value=payload: httpx.Response(
                            200,
                            content=value if isinstance(value, bytes) else json.dumps(value).encode(),
                            headers={"content-type": "application/json"},
                        )
                    ),
                )
                with self.assertRaisesRegex(ValueError, error):
                    client.embed_texts(("query",))

    def test_hybrid_recovers_semantic_match_missed_by_bm25_and_caches_documents(self) -> None:
        embedding = SemanticEmbedding()
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, embedding_provider=embedding)
        corpus.ingest(
            CorpusDocument.create(
                title="treasury.pdf",
                text="The company maintains ample cash reserves and short-term funding capacity.",
            )
        )
        corpus.ingest(
            CorpusDocument.create(
                title="earnings.pdf",
                text="The operating margin and profit outlook weakened during the quarter.",
            )
        )

        lexical = corpus.search_json({"query": "liquidity resilience", "search_mode": "lexical"})
        hybrid = corpus.search_json({"query": "liquidity resilience", "search_mode": "hybrid"})
        repeated = corpus.search_json({"query": "liquidity stress", "search_mode": "hybrid"})

        self.assertEqual(lexical["chunks"], [])
        self.assertEqual(hybrid["chunks"][0]["metadata"]["file_name"], "treasury.pdf")
        self.assertEqual(hybrid["trace"]["fusion"], "rrf")
        self.assertEqual(hybrid["trace"]["embedding_model"], "fixture-v1")
        self.assertIn("vector_rank", hybrid["chunks"][0]["scores"])
        self.assertEqual(len(embedding.calls[0]), 2)
        self.assertEqual(len(embedding.calls[1]), 1)
        self.assertEqual(hybrid["trace"]["embedding_batch_count"], 2)
        self.assertEqual(repeated["trace"]["embedding_batch_count"], 1)
        self.assertEqual(repeated["chunks"][0]["metadata"]["file_name"], "treasury.pdf")

    def test_vector_search_batches_large_missing_corpus_and_embeds_query_separately(self) -> None:
        embedding = SemanticEmbedding()
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, embedding_provider=embedding)
        for index in range(300):
            corpus.ingest(
                CorpusDocument.create(
                    title=f"document-{index}.pdf",
                    text=f"Document {index} reports ample cash and liquidity reserves.",
                )
            )

        result = corpus.search_json({"query": "liquidity", "search_mode": "hybrid"})

        self.assertEqual([len(call) for call in embedding.calls], [128, 128, 44, 1])
        self.assertEqual(result["trace"]["embedding_batch_count"], 4)

    def test_hybrid_abstains_when_bm25_misses_and_all_vector_scores_are_below_threshold(self) -> None:
        class OrthogonalEmbedding:
            backend_name = "orthogonal-fixture"
            model_name = "fixture-v1"
            network_access = False

            def embed_texts(self, texts):
                return tuple(
                    (0.0, 1.0) if text == "unrelated query" else (1.0, 0.0)
                    for text in texts
                )

        corpus = InMemoryCorpus(
            tokenizer=TOKENIZER,
            embedding_provider=OrthogonalEmbedding(),
            minimum_vector_similarity=0.5,
        )
        corpus.ingest(CorpusDocument.create(title="report.pdf", text="ACME reports quarterly revenue."))

        result = corpus.search_json({"query": "unrelated query", "search_mode": "hybrid"})

        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["trace"]["vector_candidate_count_before_threshold"], 1)
        self.assertEqual(result["trace"]["vector_candidate_count"], 0)
        self.assertEqual(result["trace"]["minimum_vector_similarity"], 0.5)

    def test_selected_overlapping_chunks_are_reconstructed_without_duplicate_text(self) -> None:
        text = " ".join(["liquidity"] * 1800)
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, chunk_tokens=1024, overlap_tokens=256)
        chunk_count = corpus.ingest(CorpusDocument.create(title="long-report.pdf", text=text))

        result = corpus.search_json(
            {"query": "liquidity", "search_mode": "lexical", "top_k": chunk_count}
        )

        self.assertGreater(chunk_count, 1)
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["content"], text)
        self.assertEqual(result["chunks"][0]["metadata"]["global_start"], 0)
        self.assertEqual(result["chunks"][0]["metadata"]["global_end"], len(text))
        self.assertEqual(result["trace"]["selected_chunk_count"], chunk_count)
        self.assertEqual(result["trace"]["returned_count"], 1)

    def test_structured_blocks_preserve_pages_and_repeat_table_header_within_token_limit(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, chunk_tokens=24, overlap_tokens=6)
        rows = "\n".join(f"| 2026-Q{index} | {index * 100} |" for index in range(1, 13))
        count = corpus.ingest_blocks(
            document_id="document-1",
            title="financials.pdf",
            blocks=(
                {
                    "label": "heading",
                    "content": "Liquidity",
                    "page_number": 1,
                    "order": 0,
                    "paragraph_title": "Liquidity",
                },
                {
                    "label": "text",
                    "content": "Cash reserves remained adequate.",
                    "page_number": 1,
                    "order": 1,
                    "paragraph_title": "Liquidity",
                },
                {
                    "label": "text",
                    "content": "Debt maturities increased.",
                    "page_number": 2,
                    "order": 0,
                    "paragraph_title": "Liquidity",
                },
                {
                    "label": "table",
                    "content": f"| Period | Revenue |\n| --- | --- |\n{rows}",
                    "page_number": 3,
                    "order": 0,
                    "paragraph_title": "Revenue",
                },
            ),
        )
        records = corpus.index_records()

        self.assertEqual(count, len(records))
        self.assertTrue(all(TOKENIZER.count_tokens(record["content"]) <= 24 for record in records))
        page_two = [record for record in records if record["metadata"]["source_page"] == 2]
        self.assertEqual(len(page_two), 1)
        self.assertNotIn("Cash reserves", page_two[0]["content"])
        table_records = [record for record in records if record["metadata"]["block_label"] == "table"]
        self.assertGreater(len(table_records), 1)
        self.assertTrue(all("| Period | Revenue |" in record["content"] for record in table_records))

    def test_html_table_chunks_are_individually_closed_and_token_bounded(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, chunk_tokens=50, overlap_tokens=10)
        rows = "".join(f"<tr><td>Q{index}</td><td>{index * 100}</td></tr>" for index in range(1, 10))
        corpus.ingest_blocks(
            document_id="document-1",
            title="table.pdf",
            blocks=(
                {
                    "label": "table",
                    "content": f"<table><tr><th>Period</th><th>Revenue</th></tr>{rows}</table>",
                    "page_number": 1,
                    "order": 0,
                },
            ),
        )
        records = corpus.index_records()

        self.assertGreater(len(records), 1)
        self.assertTrue(all(record["content"].startswith("<table>") for record in records))
        self.assertTrue(all(record["content"].endswith("</table>") for record in records))
        self.assertTrue(all(TOKENIZER.count_tokens(record["content"]) <= 50 for record in records))

    def test_hybrid_without_embedding_and_unconfigured_rerank_fail_fast(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER)
        corpus.ingest(CorpusDocument.create(title="report", text="ACME liquidity remained stable."))
        with self.assertRaisesRegex(ValueError, "configured embedding"):
            corpus.search_json({"query": "liquidity", "search_mode": "hybrid"})
        with self.assertRaisesRegex(ValueError, "rerank is not configured"):
            corpus.search_json({"query": "liquidity", "rerank": True})

    def test_same_chunk_from_lexical_and_hybrid_is_idempotent_evidence(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, embedding_provider=SemanticEmbedding())
        corpus.ingest(CorpusDocument.create(title="report", text="ACME maintains ample cash reserves."))
        adapter = RetrievalEvidenceAdapter(corpus)

        lexical = adapter.search("ACME liquidity", search_mode="lexical").bundle
        hybrid = adapter.search("ACME liquidity", search_mode="hybrid").bundle
        lexical.merge(hybrid)

        self.assertEqual(len(lexical.evidence), 1)
        self.assertEqual(next(iter(lexical.evidence.values())).tags, ("retrieved",))

    def test_harness_separates_local_lexical_from_networked_hybrid(self) -> None:
        embedding = SemanticEmbedding(network_access=True)
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, embedding_provider=embedding)
        corpus.ingest(CorpusDocument.create(title="report", text="ACME maintains ample cash reserves."))
        adapter = RetrievalEvidenceAdapter(corpus)
        harness = ToolHarness()
        harness.register(
            retrieval_harness_tool(adapter, name="corpus.search", fixed_search_mode="lexical")
        )
        harness.register(
            retrieval_harness_tool(
                adapter,
                name="corpus.hybrid_search",
                fixed_search_mode="hybrid",
                network_access=True,
            )
        )
        local_context = ToolContext(
            run_id="local",
            thread_id="thread",
            policy=ExecutionPolicy(allowed_capabilities=frozenset({"document.search"})),
        )
        lexical = harness.invoke("corpus.search", {"query": "ACME"}, local_context)
        smuggled = harness.invoke(
            "corpus.search",
            {"query": "liquidity", "search_mode": "hybrid"},
            local_context,
        )
        denied = harness.invoke("corpus.hybrid_search", {"query": "liquidity"}, local_context)

        self.assertTrue(lexical.ok)
        self.assertFalse(smuggled.ok)
        self.assertEqual(smuggled.error_code, "invalid_tool_arguments")
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error_code, "network_denied")
        self.assertEqual(embedding.calls, [])

        allowed = harness.invoke(
            "corpus.hybrid_search",
            {"query": "liquidity"},
            ToolContext(
                run_id="networked",
                thread_id="thread",
                policy=ExecutionPolicy(
                    allowed_capabilities=frozenset({"document.search"}),
                    allow_network=True,
                ),
            ),
        )
        self.assertTrue(allowed.ok)
        self.assertEqual(len(embedding.calls), 2)


if __name__ == "__main__":
    unittest.main()
