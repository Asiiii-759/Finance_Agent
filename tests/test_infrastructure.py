from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from mas_finance.database import JobRepository
from mas_finance.harness import (
    ExecutionPolicy,
    ToolArgumentContract,
    ToolContext,
    ToolHarness,
    ToolResultKind,
    ToolSpec,
    function_tool,
)
from mas_finance.memory_store import MemoryNamespace, SQLiteMemoryStore
from mas_finance.personal_knowledge import SQLitePersonalKnowledgeBase
from mas_finance.queueing import ReliableJobQueue


class EmbeddingFixture:
    backend_name = "fixture"
    model_name = "fixture-v1"
    network_access = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        return tuple((1.0, 0.0) if "cash" in text.casefold() else (0.0, 1.0) for text in texts)


class InfrastructureTests(unittest.TestCase):
    def test_reliable_queue_is_idempotent_and_lease_tokens_fence_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.db"
            repository = JobRepository(f"sqlite:///{path.as_posix()}", db_path=path)
            payload = {"query": "test", "thread_id": "thread"}
            first, created = repository.submit_job(
                job_id="job-1",
                thread_id="thread",
                query="test",
                payload=payload,
                idempotency_key="same-request",
                max_attempts=2,
            )
            duplicate, duplicate_created = repository.submit_job(
                job_id="job-2",
                thread_id="thread",
                query="test",
                payload=payload,
                idempotency_key="same-request",
                max_attempts=2,
            )
            self.assertTrue(created)
            self.assertFalse(duplicate_created)
            self.assertEqual(first["job_id"], duplicate["job_id"])

            queue = ReliableJobQueue(repository, lease_seconds=10, retry_delay_seconds=0)
            leased = queue.claim("worker-a")
            self.assertIsNotNone(leased)
            assert leased is not None
            self.assertIsNone(queue.claim("worker-b"))
            self.assertFalse(queue.complete("job-1", "stale-token"))
            self.assertEqual(queue.fail("job-1", leased["lease_token"], "TimeoutError"), "pending")
            retry = queue.claim("worker-b")
            self.assertEqual(retry["attempt_count"], 2)
            self.assertEqual(queue.fail("job-1", retry["lease_token"], "TimeoutError"), "dead")

            repository.submit_job(
                job_id="job-cancel",
                thread_id="thread",
                query="cancel",
                payload={"query": "cancel", "thread_id": "thread"},
                idempotency_key="cancel-request",
                max_attempts=2,
            )
            self.assertEqual(queue.request_cancellation("job-cancel"), "cancelled")
            self.assertIsNone(queue.claim("worker-c"))

            repository.submit_job(
                job_id="job-complete",
                thread_id="thread",
                query="complete",
                payload={"query": "complete", "thread_id": "thread"},
                idempotency_key="complete-request",
                max_attempts=2,
            )
            completed_lease = queue.claim("worker-c")
            self.assertTrue(queue.renew("job-complete", completed_lease["lease_token"]))
            self.assertTrue(queue.complete("job-complete", completed_lease["lease_token"]))

            repository.submit_job(
                job_id="job-cancel-running",
                thread_id="thread",
                query="cancel running",
                payload={"query": "cancel running", "thread_id": "thread"},
                idempotency_key="cancel-running-request",
                max_attempts=2,
            )
            cancelled_lease = queue.claim("worker-d")
            self.assertEqual(queue.request_cancellation("job-cancel-running"), "cancel_requested")
            self.assertTrue(
                queue.complete_cancellation("job-cancel-running", cancelled_lease["lease_token"])
            )
            self.assertGreaterEqual(
                repository.delete_terminal_jobs_before("2100-01-01T00:00:00+00:00"),
                2,
            )
            other_user, other_created = repository.submit_job(
                job_id="job-other-user",
                tenant_id="default",
                user_id="bob",
                thread_id="thread",
                query="test",
                payload=payload,
                idempotency_key="same-request",
                max_attempts=2,
            )
            self.assertTrue(other_created)
            self.assertEqual(other_user["job_id"], "job-other-user")
            self.assertEqual(
                [item["job_id"] for item in repository.list_jobs("default", "bob")],
                ["job-other-user"],
            )
            self.assertIsNone(repository.get_job_for_principal("job-other-user", "default", "anonymous"))

    def test_personal_corpus_persists_acl_manifest_and_document_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.db"
            embedding = EmbeddingFixture()
            store = SQLitePersonalKnowledgeBase(path)
            stored = store.add_document(
                "tenant",
                "alice",
                {
                    "document_id": "doc-1",
                    "filename": "policy.pdf",
                    "pages": [
                        {"page_number": 1, "text": "Keep ample cash reserves.", "extraction_method": "mcp"}
                    ],
                },
                embedding_provider=embedding,
            )
            self.assertEqual(stored["index_status"], "vector_ready")
            manifest = store.list_documents("tenant", "alice")[0]
            self.assertEqual(manifest["embedding_model"], "fixture-v1")
            self.assertEqual(store.list_documents("tenant", "bob"), [])

            restored = SQLitePersonalKnowledgeBase(path).corpus(
                "tenant", "alice", embedding_provider=embedding
            )
            result = restored.search_json({"query": "cash", "search_mode": "vector"})
            self.assertEqual(result["chunks"][0]["metadata"]["document_id"], "doc-1")
            self.assertEqual(len(embedding.calls), 2)
            self.assertEqual(len(embedding.calls[-1]), 1)

    def test_audit_ledger_is_append_only_and_model_token_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            store = SQLiteMemoryStore(path)
            namespace = MemoryNamespace("tenant", "user", "conversation_history", "thread")
            event = {
                "call_id": "run:1",
                "run_id": "run",
                "timestamp": "2026-08-28T00:00:00+00:00",
                "tool_name": "llm.plan",
            }
            self.assertTrue(store.append_audit_event(namespace, event))
            self.assertFalse(store.append_audit_event(namespace, event))
            store.record_run_usage(
                namespace,
                "run",
                {
                    "tool_calls": 1,
                    "network_attempts": 0,
                    "model_calls": 1,
                    "model_input_tokens": 10,
                    "model_output_tokens": 2,
                },
            )
            deleted = store.delete_operational_history_before("2100-01-01T00:00:00+00:00")
            self.assertEqual(deleted["run_usage"], 1)
            connection = sqlite3.connect(path)
            try:
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    connection.execute("DELETE FROM audit_ledger WHERE event_id = 'run:1'")
            finally:
                connection.close()

            harness = ToolHarness()
            harness.register(
                function_tool(
                    ToolSpec(
                        "llm.test",
                        "fixture",
                        "model.generate",
                        result_kind=ToolResultKind.MODEL_RESPONSE,
                        arguments=ToolArgumentContract(
                            required=frozenset({"system_prompt", "user_prompt"}),
                            optional=frozenset({"max_tokens"}),
                        ),
                    ),
                    lambda _arguments, _context: {"content": "ok"},
                )
            )
            context = ToolContext(
                run_id="budget",
                thread_id="thread",
                policy=ExecutionPolicy(
                    allowed_capabilities=frozenset({"model.generate"}),
                    max_model_input_tokens=1_000,
                    max_model_output_tokens=256,
                ),
            )
            denied = harness.invoke(
                "llm.test",
                {"system_prompt": "x" * 3_000, "user_prompt": "y", "max_tokens": 100},
                context,
            )
            self.assertEqual(denied.error_code, "model_input_token_budget_exhausted")

    def test_conversation_thread_directory_is_owner_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            store.upsert_conversation_thread(
                "tenant",
                "alice",
                "thread-a",
                title="比较两家公司",
                run_id="run-a",
                status="running",
            )
            store.upsert_conversation_thread(
                "tenant",
                "alice",
                "thread-a",
                title="不会覆盖首轮标题",
                run_id="run-b",
                status="succeeded",
            )
            self.assertEqual(store.list_conversation_threads("tenant", "bob"), [])
            thread = store.list_conversation_threads("tenant", "alice")[0]
            self.assertEqual(thread["title"], "比较两家公司")
            self.assertEqual(thread["last_run_id"], "run-b")
            self.assertEqual(thread["last_status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
