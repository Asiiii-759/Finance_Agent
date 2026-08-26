from __future__ import annotations

import unittest

import httpx

from mas_finance import HTTPJSONRAGClient as PublicHTTPJSONRAGClient
from mas_finance import RetrievalSource as PublicRetrievalSource
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.retrieval import (
    HTTPJSONRAGClient,
    RetrievalEvidenceAdapter,
    retrieval_harness_tool,
)


class FakeRAGClient:
    def search_json(self, payload):
        return {
            "query": payload["query"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "content": "Revenue increased by 10% year over year.",
                    "rank": 1,
                    "score": 0.82,
                    "scores": {"vector": 0.7, "final": 0.82},
                    "metadata": {
                        "file_name": "annual-report.pdf",
                        "document_title": "Annual report",
                        "company": "ACME",
                        "source_page": 8,
                        "global_start": 101,
                        "global_end": 145,
                        "publish_date": "2026-03-01",
                        "kb_name": "Finance",
                    },
                }
            ],
            "trace": {"search_mode": payload["search_mode"], "contract_version": "1.0"},
        }


class RetrievalAdapterTests(unittest.TestCase):
    def test_deployment_extensions_are_public_package_exports(self) -> None:
        self.assertIs(PublicHTTPJSONRAGClient, HTTPJSONRAGClient)
        self.assertEqual(PublicRetrievalSource.__name__, "RetrievalSource")

    def test_fixed_acl_filters_override_request_filters(self) -> None:
        class CapturingClient(FakeRAGClient):
            payload = None

            def search_json(self, payload):
                self.payload = dict(payload)
                return super().search_json(payload)

        client = CapturingClient()
        RetrievalEvidenceAdapter(
            client,
            fixed_filters={"tenant_id": "trusted", "acl_group": "research"},
        ).search(
            "ACME",
            filters={"tenant_id": "attacker", "desk": "credit"},
        )
        self.assertEqual(
            client.payload["filters"],
            {"tenant_id": "trusted", "acl_group": "research", "desk": "credit"},
        )

    def test_http_json_gateway_is_fixed_bounded_and_contract_compatible(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, httpx.URL("https://rag.example.test/search"))
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            return httpx.Response(
                200,
                json=FakeRAGClient().search_json({"query": "ACME", "search_mode": "rrf"}),
                headers={"content-type": "application/json"},
            )

        client = HTTPJSONRAGClient(
            "https://rag.example.test/search",
            api_key="secret",
            transport=httpx.MockTransport(handler),
        )
        self.assertNotIn("secret", repr(client))
        batch = RetrievalEvidenceAdapter(client, provider="gateway").search("ACME")
        self.assertEqual(len(batch.bundle.evidence), 1)

        with self.assertRaisesRegex(ValueError, "HTTPS"):
            HTTPJSONRAGClient("http://rag.example.test/search")

        oversized = HTTPJSONRAGClient(
            "https://rag.example.test/search",
            max_response_bytes=1_024,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=b"x" * 1_025,
                    headers={"content-type": "application/json"},
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "byte limit"):
            oversized.search_json({"query": "ACME"})

        non_finite = HTTPJSONRAGClient(
            "https://rag.example.test/search",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=b'{"chunks": [], "trace": {"score": NaN}}',
                    headers={"content-type": "application/json"},
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            non_finite.search_json({"query": "ACME"})

        redirecting = HTTPJSONRAGClient(
            "https://rag.example.test/search",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    302,
                    headers={"location": "https://other.example.test/search"},
                )
            ),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            redirecting.search_json({"query": "ACME"})

    def test_maps_rag_payload_to_provenance_evidence(self) -> None:
        adapter = RetrievalEvidenceAdapter(FakeRAGClient(), provider="test_rag")
        batch = adapter.search("ACME revenue", top_k=3)
        payload = batch.to_dict()
        evidence = payload["bundle"]["evidence"][0]

        self.assertEqual(evidence["entity"], "ACME")
        self.assertEqual(evidence["source"]["provider"], "test_rag")
        self.assertIn("page=8", evidence["source"]["locator"])
        self.assertEqual(evidence["span_start"], 101)
        self.assertEqual(
            evidence["source"]["metadata"]["retrieval_trace"]["contract_version"],
            "1.0",
        )
        self.assertEqual(payload["trace"]["contract_version"], "1.0")

    def test_web_search_chunk_preserves_title_url_and_publish_date(self) -> None:
        class WebSearch:
            def search_json(self, payload):
                return {
                    "chunks": [
                        {
                            "id": "news-1",
                            "content": "ACME announced a refinancing plan.",
                            "rank": 1,
                            "score": 0.8,
                            "metadata": {
                                "title": "ACME refinancing announcement",
                                "source_url": "https://news.example.test/acme-refinancing",
                                "publisher": "Example News",
                                "publish_date": "2026-08-11",
                                "company": "ACME",
                            },
                        }
                    ],
                    "trace": {"search_mode": payload["search_mode"]},
                }

        evidence = next(
            iter(RetrievalEvidenceAdapter(WebSearch(), provider="web").search("ACME").bundle.evidence.values())
        )
        self.assertEqual(evidence.source.title, "ACME refinancing announcement")
        self.assertIn("https://news.example.test/acme-refinancing", evidence.source.locator)
        self.assertEqual(evidence.source.published_at, "2026-08-11")
        self.assertEqual(evidence.source.metadata["publisher"], "Example News")

    def test_harness_enforces_network_permission_around_rag(self) -> None:
        harness = ToolHarness()
        harness.register(
            retrieval_harness_tool(
                RetrievalEvidenceAdapter(FakeRAGClient()),
                name="remote.search",
                network_access=True,
            )
        )
        denied = harness.invoke(
            "remote.search",
            {"query": "ACME"},
            ToolContext(
                run_id="run-1",
                thread_id="thread-1",
                policy=ExecutionPolicy(allowed_capabilities=frozenset({"document.search"})),
            ),
        )
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error_code, "network_denied")

        allowed = harness.invoke(
            "remote.search",
            {"query": "ACME"},
            ToolContext(
                run_id="run-2",
                thread_id="thread-2",
                policy=ExecutionPolicy(
                    allowed_capabilities=frozenset({"document.search"}),
                    allow_network=True,
                ),
            ),
        )
        self.assertTrue(allowed.ok)
        self.assertEqual(len(allowed.data["bundle"]["evidence"]), 1)

    def test_malformed_provider_payload_fails_closed(self) -> None:
        class BrokenClient:
            def search_json(self, _payload):
                return {"chunks": "not-a-list"}

        with self.assertRaises(ValueError):
            RetrievalEvidenceAdapter(BrokenClient()).search("query")

        mutations = (
            (lambda value: value["chunks"][0].update({"rank": True}), "rank"),
            (lambda value: value["chunks"][0].update({"score": float("nan")}), "score"),
            (
                lambda value: value["chunks"][0]["metadata"].update({"source_page": "eight"}),
                "source_page",
            ),
            (lambda value: value["trace"].update({"search_mode": 7}), "search_mode"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):

                class MutatedClient(FakeRAGClient):
                    def __init__(self, mutation):
                        self.mutation = mutation

                    def search_json(self, payload):
                        response = super().search_json(payload)
                        self.mutation(response)
                        return response

                with self.assertRaisesRegex(ValueError, expected):
                    RetrievalEvidenceAdapter(MutatedClient(mutate)).search("query")

    def test_partial_span_is_omitted_instead_of_creating_invalid_evidence(self) -> None:
        class PartialSpanClient(FakeRAGClient):
            def search_json(self, payload):
                response = super().search_json(payload)
                response["chunks"][0]["metadata"].pop("global_end")
                return response

        evidence = RetrievalEvidenceAdapter(PartialSpanClient()).search("query").bundle.evidence
        item = next(iter(evidence.values()))
        self.assertIsNone(item.span_start)
        self.assertIsNone(item.span_end)


if __name__ == "__main__":
    unittest.main()
