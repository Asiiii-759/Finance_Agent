from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from llm_fixtures import NullPlanner, agent_run_input, llm_backed_agent

from mas_finance.corpus import CorpusDocument, DocumentTokenizer, InMemoryCorpus
from mas_finance.harness import ToolHarness
from mas_finance.market import MarketEvidenceAdapter, market_data_harness_tool
from mas_finance.retrieval import RetrievalEvidenceAdapter, retrieval_harness_tool

TOKENIZER = DocumentTokenizer(Path(".runtime/models/bge-m3/tokenizer.json"))


class FakeMarket:
    def fetch_company_snapshot(self, company, symbol=None):
        return {
            "provider": "fake",
            "company": company,
            "symbol": symbol or company,
            "current_price": 10.0,
            "monthly_return": 0.05,
            "market_cap": 1000.0,
            "trailing_pe": 15.0,
            "fifty_two_week_high": 12.0,
            "fifty_two_week_low": 6.0,
            "currency": "USD",
            "as_of": "2026-07-30",
            "retrieved_at": "2026-07-31",
        }


class EmptyRetrieval:
    def search_json(self, payload):
        return {"chunks": [], "trace": {"backend": "empty"}}


def populated_corpus() -> InMemoryCorpus:
    corpus = InMemoryCorpus(tokenizer=TOKENIZER)
    corpus.ingest(
        CorpusDocument.create(
            title="acme-annual-report.txt",
            text="ACME reported resilient demand and improving operating cash flow in 2026.",
            metadata={"company": "ACME", "file_name": "acme-annual-report.txt"},
        )
    )
    return corpus


class AgentLoopTests(unittest.TestCase):
    def test_corpus_uses_bge_token_windows_with_hard_limit_and_overlap(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER)
        corpus.ingest(CorpusDocument.create(title="long.pdf", text="现金流风险分析。" * 400))

        records = corpus.index_records()
        self.assertGreater(len(records), 1)
        for record in records:
            self.assertLessEqual(TOKENIZER.count_tokens(record["content"]), 1024)
        for previous, current in zip(records[:-1], records[1:], strict=True):
            self.assertEqual(
                previous["metadata"]["token_end"] - current["metadata"]["token_start"],
                256,
            )

    def test_corpus_top_k_diversifies_relevant_documents_before_extra_chunks(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER, chunk_tokens=20, overlap_tokens=0)
        corpus.ingest(
            CorpusDocument.create(
                title="first.pdf",
                text=("ACME liquidity covenant details. " * 20),
                metadata={"document_id": "first", "file_name": "first.pdf"},
            )
        )
        corpus.ingest(
            CorpusDocument.create(
                title="second.pdf",
                text="ACME was reviewed by the credit committee.",
                metadata={"document_id": "second", "file_name": "second.pdf"},
            )
        )
        focused = corpus.search_json({"query": "ACME liquidity covenant", "top_k": 2})
        self.assertEqual(
            {item["metadata"]["document_id"] for item in focused["chunks"]},
            {"first"},
        )
        result = corpus.search_json(
            {"query": "ACME liquidity covenant", "top_k": 2, "diversify_documents": True}
        )
        self.assertEqual(
            {item["metadata"]["document_id"] for item in result["chunks"]},
            {"first", "second"},
        )

    def test_corpus_retrieval_supports_chinese_bigrams(self) -> None:
        corpus = InMemoryCorpus(tokenizer=TOKENIZER)
        corpus.ingest(CorpusDocument.create(title="苹果财报", text="苹果公司收入增长，经营现金流保持稳定。"))
        result = corpus.search_json({"query": "分析苹果收入和现金流", "top_k": 3})
        self.assertEqual(len(result["chunks"]), 1)

    def test_loop_completes_with_citations_and_bounded_controls(self) -> None:
        harness = ToolHarness()
        harness.register(
            retrieval_harness_tool(
                RetrievalEvidenceAdapter(populated_corpus()),
                fixed_search_mode="lexical",
            )
        )
        harness.register(market_data_harness_tool(MarketEvidenceAdapter(FakeMarket())))
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="Analyze the supplied ACME report for demand, cash flow, and valuation",
                run_id="loop-success",
                allow_network=True,
            )
        )
        self.assertIn(outcome.status, {"succeeded", "degraded"})
        self.assertIn("corpus.search", [item.task.tool_name for item in outcome.state.observations])
        self.assertIn("## Run controls", outcome.state.report)
        self.assertFalse(outcome.state.validation_issues)

    def test_sqlite_checkpoint_is_principal_scoped_and_terminal_run_resumes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            SqliteSaver.from_conn_string(f"{directory}/checkpoints.db") as checkpointer,
        ):
            harness = ToolHarness()
            harness.register(
                retrieval_harness_tool(
                    RetrievalEvidenceAdapter(populated_corpus()),
                    fixed_search_mode="lexical",
                )
            )
            turn, policy, context = agent_run_input(
                query="Analyze ACME demand",
                run_id="resume-run",
            )
            first = llm_backed_agent(harness, checkpointer=checkpointer).run(turn, policy, context)
            second = llm_backed_agent(ToolHarness(), checkpointer=checkpointer).run(
                turn, policy, context, resume=True
            )
            self.assertEqual(first.state.to_dict(), second.state.to_dict())
            other_tenant = replace(turn, tenant_id="other-tenant")
            self.assertIsNone(
                llm_backed_agent(ToolHarness(), planner=NullPlanner(), checkpointer=checkpointer).get_state(
                    other_tenant, policy
                )
            )
            other_user = replace(turn, user_id="other-user")
            self.assertIsNone(
                llm_backed_agent(ToolHarness(), planner=NullPlanner(), checkpointer=checkpointer).get_state(
                    other_user, policy
                )
            )

    def test_no_evidence_fails_without_fabricated_claims(self) -> None:
        harness = ToolHarness()
        harness.register(retrieval_harness_tool(RetrievalEvidenceAdapter(EmptyRetrieval())))
        outcome = llm_backed_agent(harness, checkpointer=InMemorySaver()).run(
            *agent_run_input(
                query="根据文档分析 Unknown topic",
                run_id="no-evidence",
                max_iterations=2,
            )
        )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.state.bundle.claims, {})
        self.assertIn("当前证据不足以完整回答", outcome.state.report)
        self.assertIn("No structured finding could be supported", outcome.state.report)


if __name__ == "__main__":
    unittest.main()
