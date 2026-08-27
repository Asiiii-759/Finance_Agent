from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mas_finance.contracts import Claim, ClaimStatus, Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.harness import (
    ExecutionPolicy,
    RetryPolicy,
    SideEffect,
    ToolContext,
    ToolHarness,
    ToolSpec,
    ToolStatus,
    function_tool,
)
from mas_finance.memory_store import (
    ConversationEventKind,
    EntityRelation,
    InMemoryStore,
    MemoryNamespace,
    SQLiteMemoryStore,
    build_conversation_window,
)


class CharacterTokenCounter:
    name = "character-test-counter"

    def count(self, text: str) -> int:
        return len(text)


class SummaryFixture:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, previous_summary, events):
        self.calls += 1
        return {
            "conversation_summary": f"Summarized through event {events[-1].sequence}.",
            "user_goals": ["Compare Apple and Microsoft."],
            "requirements": ["Use a consistent valuation basis."],
            "decisions": [],
            "completed_work": [],
            "successful_tools": [],
            "failed_tools": [],
            "unfinished_work": [],
            "open_questions": ["Which valuation basis should be used?"],
        }


class ContractTests(unittest.TestCase):
    def test_evidence_ids_are_stable_and_claims_have_referential_integrity(self) -> None:
        source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="Annual report",
            locator="file:///report.pdf#page=3",
            provider="finance-rag",
            as_of="2025-12-31",
        )
        first = Evidence.create(source=source, content="Revenue was 100.", page=3, entity="ACME")
        second = Evidence.create(source=source, content="Revenue was 100.", page=3, entity="ACME")
        self.assertEqual(first.evidence_id, second.evidence_id)

        bundle = EvidenceBundle()
        bundle.add_evidence(first)
        claim = Claim.create(
            text="ACME revenue was 100.",
            status=ClaimStatus.SUPPORTED,
            evidence_ids=(first.evidence_id,),
        )
        bundle.add_claim(claim)
        self.assertEqual(bundle.to_dict()["claims"][0]["status"], "supported")

        unknown = Claim.create(
            text="Unknown claim",
            status=ClaimStatus.SUPPORTED,
            evidence_ids=("ev_missing",),
        )
        with self.assertRaises(ValueError):
            bundle.add_claim(unknown)

    def test_non_supported_claim_needs_visible_caveat(self) -> None:
        with self.assertRaises(ValueError):
            Claim.create(text="An inference", status=ClaimStatus.INFERRED)

    def test_merge_tolerates_volatile_source_metadata_but_not_identity_conflicts(self) -> None:
        source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="Report",
            locator="report.pdf#chunk=1",
            provider="internal",
        )
        first = EvidenceBundle()
        first.add_evidence(Evidence.create(source=source, content="Revenue increased."))
        refreshed_source = replace(source, retrieved_at="2027-01-01T00:00:00Z", metadata={"rank": 2})
        refreshed = EvidenceBundle()
        refreshed.add_evidence(Evidence.create(source=refreshed_source, content="A second excerpt."))
        first.merge(refreshed)
        self.assertEqual(len(first.evidence), 2)

        tampered = source.to_dict()
        tampered["provider"] = "different-provider"
        with self.assertRaisesRegex(ValueError, "source_id does not match"):
            SourceRef.from_dict(tampered)


class HarnessTests(unittest.TestCase):
    def build_context(self, **overrides):
        values = {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "policy": ExecutionPolicy(
                allowed_capabilities=frozenset({"document.search", "market.read"}),
                allow_network=True,
                max_tool_calls=2,
                max_network_calls=2,
            ),
        }
        values.update(overrides)
        return ToolContext(**values)

    def test_policy_budget_retry_and_secret_redaction(self) -> None:
        attempts = {"count": 0}

        def flaky(arguments, _context):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("temporary")
            return {"query": arguments["query"]}

        harness = ToolHarness(sleeper=lambda _seconds: None)
        harness.register(
            function_tool(
                ToolSpec(
                    name="rag.search",
                    description="Search documents",
                    capability="document.search",
                    network_access=True,
                    retry=RetryPolicy(max_attempts=2),
                ),
                flaky,
            )
        )
        context = self.build_context()
        result = harness.invoke(
            "rag.search",
            {"query": "risk", "api_key": "secret", "user_prompt": "private document text"},
            context,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(harness.audit_events()[0]["arguments"]["api_key"], "***REDACTED***")
        self.assertEqual(harness.audit_events()[0]["arguments"]["query"]["length"], 4)
        self.assertEqual(
            harness.audit_events()[0]["arguments"]["user_prompt"],
            "***CONTENT_OMITTED***",
        )

        budget_result = harness.invoke("rag.search", {"query": "again"}, context)
        self.assertEqual(budget_result.status, ToolStatus.BUDGET_EXHAUSTED)
        self.assertEqual(budget_result.error_code, "network_call_budget_exhausted")

    def test_network_and_transaction_tools_are_default_denied(self) -> None:
        harness = ToolHarness()
        harness.register(
            function_tool(
                ToolSpec(
                    name="broker.order",
                    description="Place an order",
                    capability="broker.trade",
                    side_effect=SideEffect.FINANCIAL_TRANSACTION,
                    network_access=True,
                ),
                lambda _arguments, _context: {"placed": True},
            )
        )
        context = ToolContext(
            run_id="run-safe",
            thread_id="thread-safe",
            policy=ExecutionPolicy(allowed_capabilities=frozenset({"broker.trade"})),
        )
        result = harness.invoke("broker.order", {"symbol": "ACME"}, context)
        self.assertEqual(result.status, ToolStatus.DENIED)
        self.assertEqual(result.error_code, "side_effect_denied")

    def test_side_effecting_tool_cannot_have_automatic_retries(self) -> None:
        with self.assertRaises(ValueError):
            ToolSpec(
                name="writer",
                description="Writes externally",
                capability="external.write",
                side_effect=SideEffect.EXTERNAL_WRITE,
                retry=RetryPolicy(max_attempts=2),
            )


class MemoryTests(unittest.TestCase):
    def test_memory_record_can_be_deleted_without_deleting_namespace(self) -> None:
        namespace = MemoryNamespace("tenant", "user", "personal_memory")
        store = InMemoryStore()
        store.put(namespace, "first", {"value": 1})
        store.put(namespace, "second", {"value": 2})
        self.assertTrue(store.delete(namespace, "first"))
        self.assertFalse(store.delete(namespace, "first"))
        self.assertEqual([item.key for item in store.list(namespace)], ["second"])

    def test_entity_relation_deserialization_does_not_coerce_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be strings"):
            EntityRelation.from_dict({"subject": 123, "predicate": "has_symbol", "object": "AAPL"})

    def test_namespace_isolation_and_defensive_copy(self) -> None:
        store = InMemoryStore()
        alice = MemoryNamespace("tenant-a", "alice", "thread_state", "thread-1")
        bob = MemoryNamespace("tenant-a", "bob", "thread_state", "thread-1")
        value = {"watchlist": ["AAPL"]}
        store.put(alice, "state", value)
        value["watchlist"].append("MSFT")

        self.assertEqual(store.get(alice, "state").value, {"watchlist": ["AAPL"]})

        with self.assertRaisesRegex(ValueError, "invalid memory namespace"):
            MemoryNamespace("tenant-a", "alice", "conversation_history", "")
        self.assertIsNone(store.get(bob, "state"))

        fetched = store.get(alice, "state")
        fetched.value["watchlist"].append("NVDA")
        self.assertEqual(store.get(alice, "state").value, {"watchlist": ["AAPL"]})

    def test_memory_rejects_unbounded_or_non_json_records(self) -> None:
        store = InMemoryStore()
        namespace = MemoryNamespace("tenant-a", "alice", "thread_context", "thread-1")
        with self.assertRaises(ValueError):
            store.put(namespace, "oversized", {"text": "x" * 100_001})
        with self.assertRaises(ValueError):
            store.put(namespace, "invalid", {"value": object()})

    def test_sqlite_memory_is_durable_scoped_and_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            namespace = MemoryNamespace("tenant-a", "alice", "thread_context", "thread-1")
            store.put(namespace, "latest", {"entities": ["Apple", "Microsoft"]})
            reopened = SQLiteMemoryStore(Path(directory) / "memory.db")
            self.assertEqual(
                reopened.get(namespace, "latest").value["entities"],
                ["Apple", "Microsoft"],
            )
            self.assertIsNone(
                reopened.get(
                    MemoryNamespace("tenant-b", "alice", "thread_context", "thread-1"),
                    "latest",
                )
            )
            self.assertEqual(reopened.delete_namespace(namespace), 1)

    def test_conversation_history_is_durable_compacted_and_explicitly_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            namespace = MemoryNamespace("tenant-a", "alice", "conversation_history", "thread-1")
            store = SQLiteMemoryStore(path)
            summarizer = SummaryFixture()
            for index in range(16):
                kind = ConversationEventKind.USER_MESSAGE if index % 2 == 0 else ConversationEventKind.ASSISTANT_MESSAGE
                store.append_conversation_event(
                    namespace,
                    event_id=f"event-{index}",
                    kind=kind,
                    content=f"message {index} " + "x" * 1_200,
                    occurred_at=f"2026-08-26T12:{index:02d}:00+08:00",
                    run_id=f"run-{index // 2}",
                    entities=("Apple", "Microsoft") if kind is ConversationEventKind.USER_MESSAGE else (),
                    payload=(
                        {"entity_symbols": {"Apple": "AAPL", "Microsoft": "MSFT"}}
                        if kind is ConversationEventKind.USER_MESSAGE
                        else None
                    ),
                )
            store.append_conversation_event(
                namespace,
                event_id="event-16",
                kind=ConversationEventKind.USER_MESSAGE,
                content="continue the comparison " + "x" * 1_200,
                occurred_at="2026-08-26T12:16:00+08:00",
                run_id="run-pending",
                entities=("Apple",),
                payload={"entity_symbols": {"Apple": "AAPL"}},
            )
            store.append_conversation_event(
                namespace,
                event_id="event-17",
                kind=ConversationEventKind.TOOL_EVENT,
                content="market.history: failed",
                occurred_at="2026-08-26T12:17:00+08:00",
                run_id="run-pending",
                payload={
                    "tool_name": "market.history",
                    "result_status": "failed",
                    "error_code": "provider_timeout",
                    "attempts": 2,
                },
            )
            store.append_conversation_event(
                namespace,
                event_id="fact-early",
                kind=ConversationEventKind.ATOMIC_FACT,
                content="用户要求比较 Apple 与 Microsoft。",
                occurred_at="2026-08-26T12:18:00+08:00",
                run_id="run-pending",
                entities=("Apple", "Microsoft"),
                payload={"source_event_ids": ["event-16"], "status": "requested"},
            )

            with self.assertRaisesRegex(RuntimeError, "requires an LLM summarizer"):
                build_conversation_window(
                    store,
                    namespace,
                    max_tokens=16_000,
                    recent_tokens=4_000,
                    token_counter=CharacterTokenCounter(),
                )
            context = build_conversation_window(
                store,
                namespace,
                max_tokens=16_000,
                recent_tokens=4_000,
                summarizer=summarizer,
                token_counter=CharacterTokenCounter(),
            )
            self.assertLessEqual(context["manifest"]["estimated_token_count"], 16_000)
            self.assertEqual(context["manifest"]["token_count_method"], "character-test-counter")
            self.assertEqual(summarizer.calls, 1)
            self.assertIn("Summarized through event", context["summary"]["conversation_summary"])
            self.assertEqual(context["focus_entities"], ["Apple"])
            self.assertEqual(context["focus_history"][-1]["entities"], ["Apple"])
            self.assertEqual(context["entity_state"]["Apple"]["mention_count"], 9)
            self.assertEqual(context["entity_state"]["Apple"]["symbol"], "AAPL")
            self.assertEqual(context["atomic_facts"][0]["content"], "用户要求比较 Apple 与 Microsoft。")
            self.assertLessEqual(context["manifest"]["recent_context_tokens"], 4_000)
            self.assertEqual(context["manifest"]["max_recent_context_tokens"], 4_000)
            self.assertEqual({event["run_id"] for event in context["recent_events"]}, {"run-pending"})
            self.assertEqual(context["run_state"][-1]["status"], "unfinished")
            self.assertEqual(context["run_state"][-1]["tools"][0]["error_code"], "provider_timeout")
            self.assertGreater(context["manifest"]["covered_through_sequence"], 0)
            self.assertEqual(len(store.list_conversation_events(namespace)), 19)

            reopened = SQLiteMemoryStore(path)
            self.assertEqual(len(reopened.list_conversation_events(namespace)), 19)
            self.assertIsNotNone(reopened.get_conversation_summary(namespace))
            self.assertEqual(
                reopened.list_conversation_events(namespace)[0],
                reopened.append_conversation_event(
                    namespace,
                    event_id="event-0",
                    kind=ConversationEventKind.USER_MESSAGE,
                    content="message 0 " + "x" * 1_200,
                    occurred_at="2026-08-26T12:00:00+08:00",
                    run_id="run-0",
                    entities=("Apple", "Microsoft"),
                    payload={"entity_symbols": {"Apple": "AAPL", "Microsoft": "MSFT"}},
                ),
            )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                reopened.append_conversation_event(
                    namespace,
                    event_id="event-0",
                    kind=ConversationEventKind.USER_MESSAGE,
                    content="different content",
                    run_id="run-0",
                )
            self.assertEqual(
                reopened.delete_conversation(namespace),
                {"events": 19, "summaries": 1, "run_logs": 0},
            )
            self.assertEqual(reopened.list_conversation_events(namespace), [])


if __name__ == "__main__":
    unittest.main()
