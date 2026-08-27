from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pdf_fixtures import MCPPDFParserFixture, write_stub_pdf

from mas_finance.config import AppConfig
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.corpus import InMemoryCorpus
from mas_finance.harness import SideEffect, ToolArgumentContract, ToolResultKind, ToolSpec, function_tool
from mas_finance.llm import LLMSettings
from mas_finance.retrieval import RetrievalEvidenceAdapter, RetrievalSource
from mas_finance.service import FinanceAnalysisService


def make_config(root: Path, *, allow_network: bool = False) -> AppConfig:
    db_path = root / "finance.db"
    return AppConfig(
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        redis_url=None,
        redis_queue_name="test",
        market_data_provider="offline",
        alphavantage_api_key=None,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        llm=LLMSettings(None, "https://api.deepseek.com", "deepseek-v4-flash", 10),
        allow_network=allow_network,
        conversation_memory_enabled=False,
    )


class EvidenceRAGClient:
    def __init__(self, *, empty: bool = False, malformed_content: bool = False) -> None:
        self.empty = empty
        self.malformed_content = malformed_content
        self.calls = 0
        self.last_payload = None

    def search_json(self, payload):
        self.calls += 1
        self.last_payload = dict(payload)
        chunks = []
        if not self.empty:
            chunks.append(
                {
                    "id": "acme-risk-1",
                    "content": (
                        {"not": "text"}
                        if self.malformed_content
                        else "ACME disclosed that covenant headroom narrowed during the quarter."
                    ),
                    "rank": 1,
                    "score": 0.9,
                    "metadata": {
                        "file_name": "acme-credit-memo.pdf",
                        "document_title": "ACME credit memo",
                        "company": "ACME",
                        "source_page": 4,
                    },
                }
            )
        return {
            "chunks": chunks,
            "trace": {"search_mode": payload["search_mode"], "request_id": "trace-1"},
        }


class SemanticEmbedding:
    backend_name = "semantic-fixture"
    model_name = "fixture-v1"
    network_access = False

    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return tuple(
            (1.0, 0.0)
            if any(term in text.casefold() for term in ("liquidity", "ample cash", "流动性"))
            else (0.0, 1.0)
            for text in texts
        )


class RAGAndUploadIntegrationTests(unittest.TestCase):
    def test_uploaded_pdf_uses_real_hybrid_tool_for_semantic_only_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "treasury-policy.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {
                    pdf_path.name: {
                        1: "Treasury policy requires ample cash reserves and diversified short-term funding capacity."
                    }
                }
            )
            embedding = SemanticEmbedding()
            service = FinanceAnalysisService(
                make_config(root),
                embedding_provider=embedding,
                pdf_document_parser=parser,
                pdf_parser_network_access=False,
            )

            response = service.analyze(
                "Assess liquidity resilience using this PDF.",
                document_paths=[str(pdf_path)],
                require_documents=True,
                require_market_data=False,
                export_artifacts=False,
            )["result"]

            self.assertEqual(response["status"], "succeeded")
            self.assertEqual(response["observations"][0]["task"]["tool_name"], "corpus.hybrid_search")
            evidence = response["bundle"]["evidence"][0]
            self.assertEqual(evidence["source"]["metadata"]["retrieval_trace"]["fusion"], "rrf")
            self.assertEqual(
                evidence["source"]["metadata"]["retrieval_trace"]["embedding_model"],
                "fixture-v1",
            )
            self.assertEqual(embedding.calls, 1)
            catalog = {item["name"]: item for item in service.describe_tools()}
            self.assertEqual(catalog["corpus.search"]["search_mode"], "lexical")
            self.assertEqual(catalog["corpus.hybrid_search"]["search_mode"], "hybrid_rrf")
            self.assertNotIn(
                "search_mode",
                catalog["corpus.hybrid_search"]["input_contract"]["optional"],
            )

    def test_personal_documents_gain_hybrid_search_without_embedding_at_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "personal-treasury.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {pdf_path.name: {1: "My treasury rule requires ample cash reserves before any private investment."}}
            )
            embedding = SemanticEmbedding()
            service = FinanceAnalysisService(
                make_config(root),
                embedding_provider=embedding,
                pdf_document_parser=parser,
                pdf_parser_network_access=False,
            )

            service.ingest_personal_documents([str(pdf_path)], user_id="alice")
            self.assertEqual(embedding.calls, 0)
            result = service.analyze(
                "What does my library say about liquidity resilience?",
                require_documents=True,
                require_market_data=False,
                export_artifacts=False,
                user_id="alice",
            )["result"]

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["observations"][0]["task"]["tool_name"], "personal.hybrid_search")
            self.assertEqual(embedding.calls, 1)

    def test_deployment_extension_rejects_side_effecting_or_non_evidence_tools(self) -> None:
        side_effecting = function_tool(
            ToolSpec(
                name="portfolio.write",
                description="Mutate portfolio data.",
                capability="document.search",
                side_effect=SideEffect.EXTERNAL_WRITE,
                result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            ),
            lambda _arguments, _context: {"bundle": EvidenceBundle().to_dict()},
        )
        non_evidence = function_tool(
            ToolSpec(
                name="portfolio.raw",
                description="Return untyped provider data.",
                capability="document.search",
                result_kind=ToolResultKind.ANY,
            ),
            lambda _arguments, _context: {"raw": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            for tool in (side_effecting, non_evidence):
                with self.subTest(tool=tool.spec.name), self.assertRaisesRegex(
                    ValueError,
                    "read-only canonical evidence",
                ):
                    FinanceAnalysisService(config, evidence_tools=(tool,))

    def test_deployment_evidence_tool_can_inject_mcp_shaped_read_only_capability(self) -> None:
        source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="Portfolio policy",
            locator="mcp://portfolio/policy",
            provider="portfolio-mcp",
        )
        bundle = EvidenceBundle()
        bundle.add_evidence(Evidence.create(source=source, content="Maximum single issuer exposure is five percent."))
        tool = function_tool(
            ToolSpec(
                name="portfolio.policy_search",
                description="Search a user's authorized portfolio policy through an MCP gateway.",
                capability="document.search",
                result_kind=ToolResultKind.EVIDENCE_BUNDLE,
                arguments=ToolArgumentContract(
                    required=frozenset({"query"}),
                    optional=frozenset({"top_k", "filters"}),
                ),
            ),
            lambda _arguments, _context: {"bundle": bundle.to_dict(), "gaps": []},
        )
        with tempfile.TemporaryDirectory() as directory:
            service = FinanceAnalysisService(make_config(Path(directory)), evidence_tools=(tool,))
            result = service.analyze(
                "根据组合政策说明单一发行人限额",
                require_documents=True,
                export_artifacts=False,
            )["result"]
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["observations"][0]["task"]["tool_name"], "portfolio.policy_search")
            self.assertIn("portfolio.policy_search", {item["name"] for item in service.describe_tools()})

    def test_personal_knowledge_is_persistent_scoped_searchable_and_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "my-bond-notes.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {
                    pdf_path.name: {
                        1: "My bond policy requires checking duration, convexity, and credit spread before purchase."
                    }
                }
            )

            first = FinanceAnalysisService(
                make_config(root), pdf_document_parser=parser, pdf_parser_network_access=False
            )
            stored = first.ingest_personal_documents([str(pdf_path)], user_id="alice")
            first.close()

            second = FinanceAnalysisService(make_config(root))
            result = second.analyze(
                "根据我的知识库，duration、convexity 和 credit spread 要怎么检查？",
                require_documents=True,
                export_artifacts=False,
                user_id="alice",
            )["result"]
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["observations"][0]["task"]["tool_name"], "personal.search")
            evidence = result["bundle"]["evidence"][0]
            self.assertEqual(evidence["source"]["provider"], "personal_knowledge")
            self.assertIn("convexity", evidence["content"])

            hidden = second.analyze(
                "根据我的知识库，duration、convexity 和 credit spread 要怎么检查？",
                require_documents=True,
                export_artifacts=False,
                user_id="bob",
            )["result"]
            self.assertEqual(hidden["status"], "degraded")
            self.assertIn("document:query", hidden["coverage"]["missing"])
            self.assertNotIn(
                "personal_knowledge",
                {item["source"]["provider"] for item in hidden["bundle"]["evidence"]},
            )
            self.assertTrue(second.delete_personal_document(stored[0]["document_id"], user_id="alice"))
            self.assertEqual(second.list_personal_documents(user_id="alice"), [])

    def test_session_document_namespace_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "bounded.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture({pdf_path.name: {1: "Bounded session document text."}})
            service = FinanceAnalysisService(
                replace(make_config(root), max_session_document_sessions=1),
                pdf_document_parser=parser,
                pdf_parser_network_access=False,
            )
            service.analyze(
                "分析 bounded document",
                thread_id="first-session",
                document_paths=[str(pdf_path)],
                retain_documents_for_session=True,
                export_artifacts=False,
            )
            with self.assertRaisesRegex(ValueError, "namespace limit"):
                service.analyze(
                    "分析 bounded document",
                    thread_id="second-session",
                    document_paths=[str(pdf_path)],
                    retain_documents_for_session=True,
                    export_artifacts=False,
                )

    def test_retained_session_documents_support_followups_without_raw_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "acme-covenant.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {pdf_path.name: {1: "ACME covenant headroom narrowed to 18 million in the second quarter."}}
            )
            service = FinanceAnalysisService(
                make_config(root), pdf_document_parser=parser, pdf_parser_network_access=False
            )

            first = service.analyze(
                "分析 ACME covenant headroom",
                thread_id="credit-session",
                entities=["ACME"],
                document_paths=[str(pdf_path)],
                retain_documents_for_session=True,
                export_artifacts=False,
            )
            pdf_path.unlink()
            followup = service.analyze(
                "这份文档披露的 covenant headroom 是多少？",
                thread_id="credit-session",
                entities=["ACME"],
                use_session_documents=True,
                export_artifacts=False,
            )

            self.assertEqual(first["session_document_count"], 1)
            self.assertEqual(first["document_diagnostics"][0]["lifecycle"], "session_retained")
            self.assertEqual(followup["result"]["status"], "succeeded")
            self.assertEqual(followup["document_diagnostics"][0]["lifecycle"], "session")
            evidence_text = [item["content"] for item in followup["result"]["bundle"]["evidence"]]
            self.assertTrue(any("18 million" in text for text in evidence_text))
            self.assertEqual(service.delete_session_documents("credit-session"), 1)
            self.assertEqual(service.list_session_documents("credit-session"), [])

    def test_session_documents_require_explicit_thread_and_are_namespace_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = FinanceAnalysisService(make_config(root))
            with self.assertRaisesRegex(ValueError, "thread_id is required"):
                service.analyze(
                    "分析会话文档",
                    use_session_documents=True,
                    export_artifacts=False,
                )

            pdf_path = root / "private.pdf"
            write_stub_pdf(pdf_path)
            service = FinanceAnalysisService(
                make_config(root),
                pdf_document_parser=MCPPDFParserFixture(
                    {pdf_path.name: {1: "Tenant A private liquidity forecast."}}
                ),
                pdf_parser_network_access=False,
            )
            service.analyze(
                "分析 liquidity forecast",
                thread_id="shared-name",
                tenant_id="tenant-a",
                user_id="alice",
                document_paths=[str(pdf_path)],
                retain_documents_for_session=True,
                export_artifacts=False,
            )
            self.assertEqual(
                len(service.list_session_documents("shared-name", tenant_id="tenant-a", user_id="alice")),
                1,
            )
            self.assertEqual(
                service.list_session_documents("shared-name", tenant_id="tenant-a", user_id="bob"),
                [],
            )

    def test_image_only_upload_fails_closed_without_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "scan.pdf"
            write_stub_pdf(pdf_path)

            with self.assertRaisesRegex(ValueError, "PaddleOCR or MCP"):
                FinanceAnalysisService(make_config(root)).analyze(
                    "分析扫描件中的风险",
                    document_paths=[str(pdf_path)],
                    export_artifacts=False,
                )

    def test_remote_ocr_requires_server_and_request_network_consent(self) -> None:
        class OCR:
            parser_kind = "paddleocr"
            calls = 0

            def extract_document(self, _file_path: Path) -> dict[int, str]:
                self.calls += 1
                return {1: "ACME covenant headroom narrowed."}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "scan.pdf"
            write_stub_pdf(pdf_path)
            ocr = OCR()
            service = FinanceAnalysisService(
                make_config(root, allow_network=True),
                pdf_document_parser=ocr,
                pdf_parser_network_access=True,
            )

            with self.assertRaisesRegex(ValueError, "network authorization"):
                service.analyze(
                    "分析 ACME 扫描件中的风险",
                    entities=["ACME"],
                    document_paths=[str(pdf_path)],
                    allow_network=False,
                    export_artifacts=False,
                )
            self.assertEqual(ocr.calls, 0)

            response = service.analyze(
                "分析 ACME 扫描件中的风险",
                entities=["ACME"],
                document_paths=[str(pdf_path)],
                allow_network=True,
                export_artifacts=False,
            )
            self.assertEqual(ocr.calls, 1)
            self.assertEqual(response["document_diagnostics"][0]["parsed_page_count"], 1)
            self.assertEqual(response["document_diagnostics"][0]["parser_kind"], "paddleocr")
            self.assertEqual(response["result"]["status"], "succeeded")

    def test_uploaded_pdf_preserves_page_citation_and_explicit_unknown_entity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "credit-review.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {
                    pdf_path.name: {
                        1: "General introduction without company-specific findings.",
                        2: "ACME liquidity covenant headroom narrowed and refinancing risk increased.",
                    }
                }
            )

            response = FinanceAnalysisService(
                make_config(root), pdf_document_parser=parser, pdf_parser_network_access=False
            ).analyze(
                "根据这份 PDF，ACME 的主要风险是什么？",
                entities=["ACME"],
                document_paths=[str(pdf_path)],
                export_artifacts=False,
            )

            self.assertEqual(response["result"]["status"], "succeeded")
            evidence = response["result"]["bundle"]["evidence"]
            document_items = [item for item in evidence if item["source"]["source_type"] == "document"]
            self.assertEqual(len(document_items), 1)
            self.assertEqual(document_items[0]["entity"], "ACME")
            self.assertEqual(document_items[0]["page"], 2)
            self.assertIn("page=2", document_items[0]["source"]["locator"])
            self.assertEqual(len(document_items[0]["source"]["metadata"]["document_id"]), 64)
            self.assertEqual(response["document_diagnostics"][0]["text_page_count"], 2)
            self.assertEqual(response["document_diagnostics"][0]["parsed_page_count"], 2)
            self.assertEqual(response["document_diagnostics"][0]["parser_kind"], "mcp")

    def test_injected_internal_rag_is_planned_and_visible_in_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = EvidenceRAGClient()
            source = RetrievalSource(
                name="internal.credit_search",
                client=client,
                provider="enterprise_credit_corpus",
                fixed_filters={"tenant_id": "tenant-a", "acl_group": "research"},
            )
            service = FinanceAnalysisService(make_config(root), retrieval_sources=(source,))

            response = service.analyze(
                "根据内部文档说明 ACME 的 covenant 风险",
                entities=["ACME"],
                export_artifacts=False,
            )

            self.assertEqual(response["result"]["status"], "succeeded")
            self.assertEqual(
                [item["task"]["tool_name"] for item in response["result"]["observations"]],
                ["internal.credit_search"],
            )
            catalog = {item["name"]: item for item in service.describe_tools()}
            self.assertEqual(
                catalog["internal.credit_search"]["support_tier"],
                "deployment_injected_contract",
            )
            self.assertTrue(catalog["internal.credit_search"]["server_filters_enforced"])
            self.assertEqual(
                client.last_payload["filters"],
                {"tenant_id": "tenant-a", "acl_group": "research"},
            )
            self.assertEqual(client.calls, 1)

    def test_configured_rag_is_available_but_not_forced_on_unrelated_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = EvidenceRAGClient()
            service = FinanceAnalysisService(
                make_config(root),
                retrieval_sources=(RetrievalSource("internal.search", client, "internal"),),
            )

            calculation = service.analyze(
                "一项投资从100增长到121，用了2年，CAGR是多少？",
                export_artifacts=False,
            )["result"]
            self.assertEqual(calculation["status"], "succeeded")
            self.assertEqual(
                [item["task"]["tool_name"] for item in calculation["observations"]],
                ["finance.calculate"],
            )
            self.assertEqual(client.calls, 0)

            forced = service.analyze(
                "说明 ACME 的 covenant 风险",
                entities=["ACME"],
                require_documents=True,
                export_artifacts=False,
            )["result"]
            self.assertEqual(forced["status"], "succeeded")
            self.assertEqual(forced["observations"][0]["task"]["tool_name"], "internal.search")
            self.assertEqual(client.calls, 1)

            disabled = service.analyze(
                "根据内部文档说明 ACME 的 covenant 风险",
                entities=["ACME"],
                require_documents=False,
                export_artifacts=False,
            )["result"]
            self.assertEqual(disabled["status"], "failed")
            self.assertEqual(disabled["observations"], [])
            self.assertEqual(client.calls, 1)

    def test_rag_fallback_and_network_denial_are_bounded_and_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = EvidenceRAGClient(empty=True)
            useful = EvidenceRAGClient()
            service = FinanceAnalysisService(
                make_config(root),
                retrieval_sources=(
                    RetrievalSource("internal.primary", empty, "primary"),
                    RetrievalSource("internal.fallback", useful, "fallback"),
                ),
            )
            response = service.analyze(
                "根据内部文档说明 ACME 的 covenant 风险",
                entities=["ACME"],
                export_artifacts=False,
            )
            self.assertEqual(response["result"]["status"], "succeeded")
            self.assertEqual(
                [item["task"]["tool_name"] for item in response["result"]["observations"]],
                ["internal.primary", "internal.fallback"],
            )
            self.assertEqual(empty.calls, 1)
            self.assertEqual(useful.calls, 1)

            remote = EvidenceRAGClient()
            denied_service = FinanceAnalysisService(
                make_config(root, allow_network=True),
                retrieval_sources=(RetrievalSource("external.web_search", remote, "web", network_access=True),),
            )
            denied = denied_service.analyze(
                "根据外部资料说明 ACME 的 covenant 风险",
                entities=["ACME"],
                allow_network=False,
                export_artifacts=False,
            )
            self.assertEqual(denied["result"]["status"], "failed")
            self.assertIn("network_denied", {item["code"] for item in denied["result"]["gaps"]})
            self.assertEqual(denied["result"]["budget_usage"]["network_attempts"], 0)
            self.assertEqual(remote.calls, 0)

    def test_retrieval_contract_rejects_type_coercion(self) -> None:
        malformed = EvidenceRAGClient(malformed_content=True)
        with self.assertRaisesRegex(ValueError, "content must be a string"):
            RetrievalEvidenceAdapter(malformed).search("ACME")

        corpus = InMemoryCorpus()
        with self.assertRaisesRegex(ValueError, "query is required"):
            corpus.search_json({"query": ["not", "text"]})
        with self.assertRaisesRegex(ValueError, "top_k must be an integer"):
            corpus.search_json({"query": "ACME", "top_k": True})

    def test_retrieval_source_names_are_unique_and_upload_name_is_reserved(self) -> None:
        client = EvidenceRAGClient()
        source = RetrievalSource("internal.search", client, "internal")
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "unique"):
                FinanceAnalysisService(config, retrieval_sources=(source, source))
            with self.assertRaisesRegex(ValueError, "reserved"):
                FinanceAnalysisService(
                    config,
                    retrieval_sources=(RetrievalSource("corpus.search", client, "bad"),),
                )
            with self.assertRaisesRegex(ValueError, "at most four"):
                FinanceAnalysisService(
                    config,
                    retrieval_sources=tuple(
                        RetrievalSource(f"internal.search_{index}", client, f"provider-{index}") for index in range(5)
                    ),
                )


if __name__ == "__main__":
    unittest.main()
