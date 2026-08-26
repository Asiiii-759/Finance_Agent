from __future__ import annotations

import json
import unittest

import httpx

from mas_finance.corpus import CorpusDocument, InMemoryCorpus
from mas_finance.embeddings import HTTPEmbeddingClient
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.retrieval import RetrievalEvidenceAdapter, retrieval_harness_tool


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
        corpus = InMemoryCorpus(embedding_provider=embedding)
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
        self.assertEqual(len(embedding.calls[0]), 3)
        self.assertEqual(len(embedding.calls[1]), 1)
        self.assertEqual(repeated["chunks"][0]["metadata"]["file_name"], "treasury.pdf")

    def test_hybrid_without_embedding_and_unconfigured_rerank_fail_fast(self) -> None:
        corpus = InMemoryCorpus()
        corpus.ingest(CorpusDocument.create(title="report", text="ACME liquidity remained stable."))
        with self.assertRaisesRegex(ValueError, "configured embedding"):
            corpus.search_json({"query": "liquidity", "search_mode": "hybrid"})
        with self.assertRaisesRegex(ValueError, "rerank is not configured"):
            corpus.search_json({"query": "liquidity", "rerank": True})

    def test_same_chunk_from_lexical_and_hybrid_is_idempotent_evidence(self) -> None:
        corpus = InMemoryCorpus(embedding_provider=SemanticEmbedding())
        corpus.ingest(CorpusDocument.create(title="report", text="ACME maintains ample cash reserves."))
        adapter = RetrievalEvidenceAdapter(corpus)

        lexical = adapter.search("ACME liquidity", search_mode="lexical").bundle
        hybrid = adapter.search("ACME liquidity", search_mode="hybrid").bundle
        lexical.merge(hybrid)

        self.assertEqual(len(lexical.evidence), 1)
        self.assertEqual(next(iter(lexical.evidence.values())).tags, ("retrieved",))

    def test_harness_separates_local_lexical_from_networked_hybrid(self) -> None:
        embedding = SemanticEmbedding(network_access=True)
        corpus = InMemoryCorpus(embedding_provider=embedding)
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
        self.assertEqual(len(embedding.calls), 1)


if __name__ == "__main__":
    unittest.main()
