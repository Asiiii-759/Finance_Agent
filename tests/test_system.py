from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from llm_fixtures import research_app, research_service
from pdf_fixtures import MCPPDFParserFixture, write_stub_pdf

from mas_finance.api.app import create_app
from mas_finance.config import AppConfig
from mas_finance.llm import LLMSettings
from mas_finance.memory_store import PersonalMemoryKind
from mas_finance.service import FinanceAnalysisService

ROOT = Path(__file__).resolve().parents[1]


def build_test_config(root: Path, api_key: str | None = None) -> AppConfig:
    db_path = root / "data" / "mas_finance.db"
    return AppConfig(
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        market_data_provider="offline",
        alphavantage_api_key=None,
        host="127.0.0.1",
        port=8000,
        api_key=api_key,
        llm=LLMSettings(
            api_key=None,
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            timeout_seconds=45,
        ),
        allow_network=False,
    )


class FinanceSystemTestCase(unittest.TestCase):
    def test_personal_memory_capacity_is_enforced_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = research_service(build_test_config(Path(directory)))
            for index in range(12):
                service.save_personal_memory(
                    kind=PersonalMemoryKind.EXPERIENCE,
                    title=f"长期经验 {index}",
                    content="x" * 7_800,
                )
            with self.assertRaisesRegex(ValueError, "100000 字符"):
                service.save_personal_memory(
                    kind=PersonalMemoryKind.EXPERIENCE,
                    title="超限经验",
                    content="x" * 7_800,
                )
            self.assertEqual(len(service.list_personal_memories()), 12)
            service.close()

    def test_personal_memory_is_explicit_scoped_fully_injected_and_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = research_service(build_test_config(Path(directory)))
            preference = service.save_personal_memory(
                kind=PersonalMemoryKind.PREFERENCE,
                title="回答风格",
                content="使用中文并先解释风险，再展示计算。",
                tags=["中文", "风险"],
                user_id="alice",
            )
            service.save_personal_memory(
                kind=PersonalMemoryKind.EXPERIENCE,
                title="债券分析步骤",
                content="分析久期、凸性与信用利差。",
                tags=["债券"],
                user_id="alice",
            )
            replacement = service.save_personal_memory(
                kind=PersonalMemoryKind.PREFERENCE,
                title="回答风格",
                content="使用中文，先展示结论，再解释风险。",
                tags=["中文", "风险"],
                user_id="alice",
            )
            self.assertEqual(replacement["memory_id"], preference["memory_id"])
            self.assertEqual(
                len(service.list_personal_memories(kind=PersonalMemoryKind.PREFERENCE, user_id="alice")),
                1,
            )
            result = service.analyze(
                "什么是市盈率？",
                export_artifacts=False,
                user_id="alice",
            )["result"]
            personal = result["request"]["personal_context"]
            self.assertEqual([item["kind"] for item in personal], ["preference", "experience"])
            self.assertIn("先展示结论", personal[0]["content"])
            self.assertIn("久期", personal[1]["content"])
            self.assertNotIn("personal_context", result["bundle"])
            self.assertEqual(service.list_personal_memories(user_id="bob"), [])
            self.assertTrue(service.delete_personal_memory(preference["memory_id"], user_id="alice"))

    def test_personal_memory_api_requires_explicit_write_and_supports_crud(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"personal-memory-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = create_app(build_test_config(tmp_root, api_key=None))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                created = await client.post(
                    "/api/v1/memories",
                    json={
                        "kind": "profile",
                        "title": "投资期限",
                        "content": "我的分析期限通常是五年以上。",
                        "tags": ["长期"],
                    },
                )
                listed = await client.get("/api/v1/memories")
                deleted = await client.delete(f"/api/v1/memories/{created.json()['memory_id']}")
                empty = await client.get("/api/v1/memories")
                return created, listed, deleted, empty

        created, listed, deleted, empty = asyncio.run(scenario())
        self.assertEqual(created.status_code, 201)
        self.assertEqual(listed.json()[0]["kind"], "profile")
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(empty.json(), [])

    def test_api_principal_is_server_owned_and_isolates_personal_data(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"principal-isolation-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        base = build_test_config(tmp_root, api_key=None)
        alice_app = create_app(replace(base, local_user_id="alice"))
        bob_app = create_app(replace(base, local_user_id="bob"))

        async def scenario():
            async with (
                AsyncClient(transport=ASGITransport(app=alice_app), base_url="http://alice") as alice,
                AsyncClient(transport=ASGITransport(app=bob_app), base_url="http://bob") as bob,
            ):
                created = await alice.post(
                    "/api/v1/memories",
                    json={
                        "kind": "preference",
                        "title": "输出语言",
                        "content": "长期使用中文。",
                        "tags": [],
                    },
                )
                alice_memories = await alice.get("/api/v1/memories")
                bob_memories = await bob.get("/api/v1/memories")
                alice_config = await alice.get("/api/v1/config")
                bob_config = await bob.get("/api/v1/config")
                return created, alice_memories, bob_memories, alice_config, bob_config

        created, alice_memories, bob_memories, alice_config, bob_config = asyncio.run(scenario())
        self.assertEqual(created.status_code, 201)
        self.assertEqual(len(alice_memories.json()), 1)
        self.assertEqual(bob_memories.json(), [])
        self.assertEqual(alice_config.json()["principal"]["user_id"], "alice")
        self.assertEqual(bob_config.json()["principal"]["user_id"], "bob")

    def test_personal_knowledge_api_persists_parsed_pages_and_deletes_document(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"personal-knowledge-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        parser = MCPPDFParserFixture(
            {"policy.pdf": {1: "Personal policy: review duration and credit spread."}}
        )
        app = create_app(
            build_test_config(tmp_root, api_key=None),
            pdf_document_parser=parser,
            pdf_parser_network_access=False,
        )
        pdf_bytes = b"%PDF-1.7\n%%EOF\n"

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                created = await client.post(
                    "/api/v1/knowledge/documents",
                    files={"files": ("policy.pdf", pdf_bytes, "application/pdf")},
                )
                listed = await client.get("/api/v1/knowledge/documents")
                document_id = created.json()["documents"][0]["document_id"]
                deleted = await client.delete(f"/api/v1/knowledge/documents/{document_id}")
                empty = await client.get("/api/v1/knowledge/documents")
                return created, listed, deleted, empty

        created, listed, deleted, empty = asyncio.run(scenario())
        self.assertEqual(created.status_code, 201)
        self.assertEqual(listed.json()["documents"][0]["filename"], "policy.pdf")
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(empty.json()["documents"], [])

    def test_config_repr_does_not_expose_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                build_test_config(Path(directory), api_key="api-secret"),
                paddleocr_access_token="ocr-secret",
                fred_api_key="fred-secret",
                brave_search_api_key="search-secret",
                bocha_search_api_key="bocha-secret",
            )
            rendered = repr(config)
            self.assertNotIn("api-secret", rendered)
            self.assertNotIn("ocr-secret", rendered)
            self.assertNotIn("fred-secret", rendered)
            self.assertNotIn("search-secret", rendered)
            self.assertNotIn("bocha-secret", rendered)

    def test_web_workspace_is_served_with_the_api(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"web-workspace-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = create_app(build_test_config(tmp_root, api_key=None))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                page = await client.get("/")
                script = await client.get("/static/app.js")
                return page, script

        page, script = asyncio.run(scenario())
        self.assertEqual(page.status_code, 200)
        self.assertIn("MAS Finance · 研究工作台", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("submitResearch", script.text)

    def test_bocha_is_the_explicit_web_search_preference_when_both_keys_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                build_test_config(Path(directory)),
                brave_search_api_key="brave-secret",
                bocha_search_api_key="bocha-secret",
            )
            tool = next(
                item for item in FinanceAnalysisService(config).describe_tools() if item["name"] == "web.search"
            )
            self.assertEqual(tool["availability"], "configured")
            self.assertEqual(tool["provider"], "bocha")

    def test_background_job_does_not_persist_provider_exception_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = FinanceAnalysisService(build_test_config(Path(directory)))
            created = service.submit_job("test failure")
            with (
                patch.object(
                    service,
                    "analyze",
                    side_effect=RuntimeError("Bearer should-never-be-persisted"),
                ),
                self.assertRaises(RuntimeError),
            ):
                service.run_job(
                    created["job_id"],
                    "test failure",
                    created["thread_id"],
                    export_artifacts=False,
                )
            job = service.get_job(created["job_id"])
            self.assertEqual(job["error_message"], "Analysis failed (RuntimeError).")

    def test_end_to_end_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = build_test_config(root)
            pdf_path = root / "acme.pdf"
            write_stub_pdf(pdf_path)
            parser = MCPPDFParserFixture(
                {pdf_path.name: {1: "ACME revenue demand and operating cash flow remained resilient in 2026"}}
            )
            response = research_service(
                config, pdf_document_parser=parser, pdf_parser_network_access=False
            ).analyze(
                "Analyze ACME demand and cash flow",
                thread_id="test-e2e",
                entities=["ACME"],
                document_paths=[str(pdf_path)],
            )
            result = response["result"]
            self.assertIn(result["status"], {"succeeded", "degraded"})
            self.assertIn("ACME", result["report"])
            self.assertGreaterEqual(len(result["bundle"]["evidence"]), 1)
            for artifact_path in response["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists())
            state_payload = json.loads(Path(response["artifacts"]["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state_payload["request"]["thread_id"], "test-e2e")

    def test_analyze_fails_fast_without_llm_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = FinanceAnalysisService(build_test_config(Path(directory)))
            with self.assertRaisesRegex(RuntimeError, "LLM configuration is required"):
                service.analyze("什么是市盈率？", export_artifacts=False)

    def test_api_endpoint(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"api-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = research_app(build_test_config(tmp_root, api_key=None))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                valid = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "对比分析 Apple 与 Microsoft 的供应链风险和研发投入。",
                        "thread_id": "test-api",
                        "export_artifacts": False,
                    },
                )
                invalid = await client.post(
                    "/api/v1/analyze",
                    json={"query": "   ", "export_artifacts": False},
                )
                return valid, invalid

        response, invalid = asyncio.run(scenario())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["thread_id"], "test-api")
        self.assertTrue(payload["run_id"].startswith("run-"))
        self.assertIn("report", payload)
        self.assertTrue(payload["report"])
        self.assertEqual(payload["llm_backend"], "fixture")
        self.assertNotIn("state", payload)
        self.assertEqual(invalid.status_code, 422)

    def test_api_structured_financial_calculation(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"calculation-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = research_app(build_test_config(tmp_root, api_key=None))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                valid = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "计算三年 CAGR",
                        "calculations": [
                            {
                                "operation": "cagr",
                                "inputs": {
                                    "beginning_value": 100,
                                    "ending_value": 150,
                                    "years": 3,
                                },
                            }
                        ],
                        "export_artifacts": False,
                    },
                )
                invalid = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "bad calculation",
                        "calculations": [{"operation": "eval", "inputs": {"expression": "1+1"}}],
                    },
                )
                return valid, invalid

        valid, invalid = asyncio.run(scenario())
        self.assertEqual(valid.status_code, 200)
        payload = valid.json()
        fields = {item.get("field_name") for item in payload["evidence_bundle"]["evidence"]}
        self.assertIn("cagr", fields)
        self.assertEqual(invalid.status_code, 422)

    def test_tool_catalog_and_thread_memory_deletion(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"tool-catalog-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = research_app(build_test_config(tmp_root, api_key=None))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                analysis = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "什么是市盈率？",
                        "thread_id": "memory-delete-test",
                        "export_artifacts": False,
                    },
                )
                run_id = analysis.json()["run_id"]
                messages = await client.get("/api/v1/conversations/memory-delete-test/messages")
                conversations = await client.get("/api/v1/conversations")
                runs = await client.get("/api/v1/conversations/memory-delete-test/runs")
                run = await client.get(f"/api/v1/conversations/memory-delete-test/runs/{run_id}")
                tools = await client.get("/api/v1/tools")
                config = await client.get("/api/v1/config")
                deleted = await client.delete("/api/v1/conversations/memory-delete-test")
                return analysis, messages, conversations, runs, run, tools, config, deleted

        analysis, messages, conversations, runs, run, tools, config, deleted = asyncio.run(scenario())
        self.assertEqual(analysis.status_code, 200)
        self.assertEqual(analysis.json()["status"], "succeeded")
        self.assertEqual([item["role"] for item in messages.json()["messages"]], ["user", "assistant"])
        assistant = messages.json()["messages"][1]
        self.assertEqual(assistant["content"], analysis.json()["evidence_bundle"]["claims"][0]["text"])
        self.assertEqual(conversations.json()["conversations"][0]["thread_id"], "memory-delete-test")
        self.assertNotIn("result", runs.json()["runs"][0])
        self.assertEqual(run.json()["result"]["report"], analysis.json()["report"])
        self.assertEqual(tools.status_code, 200)
        catalog = {item["name"]: item for item in tools.json()}
        names = set(catalog)
        self.assertIn("finance.calculate", names)
        self.assertNotIn("finance.knowledge", names)
        cagr_contract = catalog["finance.calculate"]["operation_contract"]["cagr"]
        self.assertEqual(
            cagr_contract["required_inputs"],
            ["beginning_value", "ending_value", "years"],
        )
        self.assertEqual(cagr_contract["default_unit"], "ratio_per_year")
        self.assertIn("market.history", names)
        self.assertEqual(
            catalog["llm.synthesize"]["availability"],
            "required",
        )
        self.assertEqual(catalog["llm.task_frame"]["availability"], "required")
        config_payload = config.json()
        self.assertEqual(config_payload["database_backend"], "sqlite")
        self.assertNotIn("database_url", config_payload)
        self.assertNotIn("db_path", config_payload)
        self.assertNotIn("output_dir", config_payload)
        self.assertFalse(config_payload["embedding_enabled"])
        self.assertIsNone(config_payload["embedding_model"])
        self.assertEqual(config_payload["conversation_context_tokens"], 300_000)
        self.assertEqual(config_payload["conversation_recent_tokens"], 20_000)
        self.assertEqual(config_payload["principal"], {"tenant_id": "local", "user_id": "owner"})
        self.assertEqual(deleted.status_code, 200)
        self.assertGreaterEqual(deleted.json()["events"], 2)
        self.assertEqual(deleted.json()["summaries"], 0)
        self.assertEqual(deleted.json()["runs"], 1)
        self.assertEqual(deleted.json()["checkpoints"], 1)

    def test_embedding_config_requires_endpoint_and_model_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_test_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "configured together"):
                replace(config, embedding_endpoint="https://embedding.example.test/v1/embeddings")
            with self.assertRaisesRegex(ValueError, "configured together"):
                replace(config, embedding_model="bge-m3")
            app = create_app(
                replace(
                    config,
                    embedding_endpoint="https://embedding.example.test/v1/embeddings",
                    embedding_model="bge-m3",
                    embedding_api_key="embedding-secret",
                )
            )

            async def scenario():
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    return await client.get("/api/v1/config"), await client.get("/api/v1/tools")

            response, tools = asyncio.run(scenario())
            self.assertTrue(response.json()["embedding_enabled"])
            self.assertEqual(response.json()["embedding_model"], "bge-m3")
            self.assertNotIn("embedding_api_key", response.json())
            catalog = {item["name"]: item for item in tools.json()}
            self.assertTrue(catalog["corpus.hybrid_search"]["network_access"])

    def test_upload_endpoint(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"upload-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        parser = MCPPDFParserFixture({"Apple_report.pdf": {1: "Apple revenue 400 EBITDA 120 risk warning"}})
        app = research_app(
            build_test_config(tmp_root, api_key=None),
            pdf_document_parser=parser,
            pdf_parser_network_access=False,
        )
        pdf_bytes = b"%PDF-1.7\n%%EOF\n"

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(
                    "/api/v1/analyze-upload",
                    data={
                        "query": "请分析这份 Apple 财报 PDF 的核心风险。",
                        "thread_id": "upload-test",
                        "export_artifacts": "false",
                    },
                    files={"files": ("Apple_report.pdf", pdf_bytes, "application/pdf")},
                )

        response = asyncio.run(scenario())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["thread_id"], "upload-test")
        self.assertIn("Apple", payload["report"])
        document_sources = [item for item in payload["evidence_bundle"]["sources"] if item["source_type"] == "document"]
        self.assertTrue(document_sources)
        self.assertTrue(all(item["title"] == "Apple_report.pdf" for item in document_sources))
        self.assertTrue(all("Apple_report.pdf" in item["locator"] for item in document_sources))
        self.assertEqual(payload["document_diagnostics"][0]["text_page_count"], 1)
        self.assertEqual(payload["document_diagnostics"][0]["parsed_page_count"], 1)
        self.assertEqual(payload["document_diagnostics"][0]["parser_kind"], "mcp")
        self.assertEqual(list((tmp_root / "uploads").glob("*")), [])

    def test_upload_can_be_retained_for_explicit_session_recall_and_deleted(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"session-upload-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        parser = MCPPDFParserFixture({"maturity.pdf": {1: "ACME refinancing maturity is September 2027."}})
        app = research_app(
            build_test_config(tmp_root, api_key=None),
            pdf_document_parser=parser,
            pdf_parser_network_access=False,
        )
        pdf_bytes = b"%PDF-1.7\n%%EOF\n"

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                uploaded = await client.post(
                    "/api/v1/analyze-upload",
                    data={
                        "query": "分析 ACME refinancing maturity",
                        "thread_id": "retained-upload",
                        "retain_for_session": "true",
                        "export_artifacts": "false",
                    },
                    files={"files": ("maturity.pdf", pdf_bytes, "application/pdf")},
                )
                listing = await client.get("/api/v1/session-documents/retained-upload")
                recalled = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "这份文档中的到期时间是什么？",
                        "thread_id": "retained-upload",
                        "entities": ["ACME"],
                        "use_session_documents": True,
                        "export_artifacts": False,
                    },
                )
                deleted = await client.delete("/api/v1/session-documents/retained-upload")
                empty = await client.get("/api/v1/session-documents/retained-upload")
                invalid = await client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "使用会话文档",
                        "use_session_documents": True,
                        "export_artifacts": False,
                    },
                )
                return uploaded, listing, recalled, deleted, empty, invalid

        uploaded, listing, recalled, deleted, empty, invalid = asyncio.run(scenario())
        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()["session_document_count"], 1)
        self.assertEqual(
            uploaded.json()["document_diagnostics"][0]["lifecycle"],
            "session_retained",
        )
        self.assertEqual(len(listing.json()["documents"]), 1)
        self.assertEqual(recalled.status_code, 200)
        self.assertIn("September 2027", recalled.json()["report"])
        self.assertEqual(deleted.json()["deleted_documents"], 1)
        self.assertEqual(empty.json()["documents"], [])
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(list((tmp_root / "uploads").glob("*")), [])

    def test_job_endpoints_and_auth(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"job-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = research_app(build_test_config(tmp_root, api_key="secret-key"))

        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                unauthorized = await client.get("/api/v1/config")
                response = await client.post(
                    "/api/v1/jobs",
                    headers={"X-API-Key": "secret-key"},
                    json={
                        "query": "分析 Apple 2025 财报供应链附录风险，并输出报告。",
                        "thread_id": "job-test",
                        "export_artifacts": False,
                    },
                )
                created = response.json()
                detail = await client.get(
                    f"/api/v1/jobs/{created['job_id']}",
                    headers={"X-API-Key": "secret-key"},
                )
                job_payload = detail.json()
                for _ in range(20):
                    if job_payload["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.05)
                    detail = await client.get(
                        f"/api/v1/jobs/{created['job_id']}",
                        headers={"X-API-Key": "secret-key"},
                    )
                    job_payload = detail.json()
                listing = await client.get("/api/v1/jobs", headers={"X-API-Key": "secret-key"})
                return unauthorized, response, created, detail, job_payload, listing

        unauthorized, response, created, detail, job_payload, listing = asyncio.run(scenario())
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 202)
        self.assertIn(created["status"], {"pending", "running", "completed"})
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(job_payload["job_id"], created["job_id"])
        self.assertIn(job_payload["status"], {"completed", "failed"})
        self.assertEqual(listing.status_code, 200)
        self.assertGreaterEqual(len(listing.json()), 1)


if __name__ == "__main__":
    unittest.main()
