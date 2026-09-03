from __future__ import annotations

import math
import statistics
import unittest

from llm_fixtures import agent_run_input, llm_backed_agent

from mas_finance.agent import AgentContext, ChatTurn, CoverageAssessor
from mas_finance.context import FinancialContextAssembler
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.harness import ToolHarness
from mas_finance.macro import FREDEvidenceAdapter
from mas_finance.market import MarketHistoryEvidenceAdapter
from mas_finance.metrics import (
    MetricError,
    MetricOperation,
    MetricRequest,
    calculate_metric,
    financial_calculation_harness_tool,
)
from mas_finance.research import FinancialIntent, ResearchRequirement, ResearchScope
from mas_finance.sec import SECRecentFilingsAdapter


class MetricToolTests(unittest.TestCase):
    def test_allowlisted_financial_formulas(self) -> None:
        ratio = calculate_metric(
            MetricRequest(
                operation=MetricOperation.RATIO,
                inputs={"numerator": 25, "denominator": 100},
            )
        )
        self.assertEqual(ratio.value, 0.25)

        percentage_change = calculate_metric(
            MetricRequest(
                operation=MetricOperation.PERCENTAGE_CHANGE,
                inputs={"beginning_value": 100, "ending_value": 125},
            )
        )
        self.assertEqual(percentage_change.value, 0.25)

        cagr = calculate_metric(
            MetricRequest(
                operation=MetricOperation.CAGR,
                inputs={"beginning_value": 100, "ending_value": 150, "years": 3},
            )
        )
        self.assertAlmostEqual(cagr.value, (1.5 ** (1 / 3)) - 1, places=9)

        future = calculate_metric(
            MetricRequest(
                operation=MetricOperation.FUTURE_VALUE,
                inputs={"present_value": 1000, "rate": 0.05, "periods": 3},
            )
        )
        self.assertAlmostEqual(future.value, 1157.625)

        present = calculate_metric(
            MetricRequest(
                operation=MetricOperation.PRESENT_VALUE,
                inputs={"future_value": 1157.625, "rate": 0.05, "periods": 3},
                unit="USD",
            )
        )
        self.assertAlmostEqual(present.value, 1000)
        self.assertEqual(present.unit, "USD")

        payment = calculate_metric(
            MetricRequest(
                operation=MetricOperation.LOAN_PAYMENT,
                inputs={"principal": 1200, "rate": 0, "periods": 12},
            )
        )
        self.assertEqual(payment.value, 100)

        drawdown = calculate_metric(
            MetricRequest(
                operation=MetricOperation.MAX_DRAWDOWN,
                inputs={"values": [100, 90, 120, 96]},
            )
        )
        self.assertEqual(drawdown.value, -0.2)

        returns = [0.01, -0.02, 0.03, 0.01]
        annualized_return = calculate_metric(
            MetricRequest(
                operation=MetricOperation.ANNUALIZED_RETURN,
                inputs={"returns": returns, "annualization_factor": 12},
            )
        )
        expected_return = math.prod(1 + item for item in returns) ** 3 - 1
        self.assertAlmostEqual(annualized_return.value, expected_return)

        annualized_volatility = calculate_metric(
            MetricRequest(
                operation=MetricOperation.ANNUALIZED_VOLATILITY,
                inputs={"returns": returns, "annualization_factor": 12},
            )
        )
        expected_volatility = statistics.stdev(returns) * math.sqrt(12)
        self.assertAlmostEqual(annualized_volatility.value, expected_volatility)

        sharpe = calculate_metric(
            MetricRequest(
                operation=MetricOperation.SHARPE_RATIO,
                inputs={
                    "returns": returns,
                    "annualization_factor": 12,
                    "annual_risk_free_rate": 0.02,
                },
            )
        )
        self.assertAlmostEqual(sharpe.value, (expected_return - 0.02) / expected_volatility)

        with self.assertRaises(MetricError):
            calculate_metric(
                MetricRequest(
                    operation=MetricOperation.RATIO,
                    inputs={"numerator": 1, "denominator": 0},
                )
            )

    def test_calculation_contract_rejects_coercion_extra_fields_and_false_units(self) -> None:
        with self.assertRaisesRegex(MetricError, "operation must be a string"):
            MetricRequest.from_dict({"operation": True, "inputs": {"numerator": 1, "denominator": 2}})
        with self.assertRaisesRegex(MetricError, "unsupported fields"):
            MetricRequest.from_dict(
                {
                    "operation": "ratio",
                    "inputs": {"numerator": 1, "denominator": 2},
                    "expression": "arbitrary code",
                }
            )
        with self.assertRaisesRegex(MetricError, "text fields must be strings"):
            MetricRequest.from_dict(
                {
                    "operation": "ratio",
                    "inputs": {"numerator": 1, "denominator": 2},
                    "label": 42,
                }
            )
        with self.assertRaisesRegex(MetricError, "incompatible"):
            MetricRequest(
                operation=MetricOperation.CAGR,
                inputs={"beginning_value": 100, "ending_value": 121, "years": 2},
                unit="USD",
            )
        with self.assertRaisesRegex(MetricError, "supported range"):
            calculate_metric(
                MetricRequest(
                    operation=MetricOperation.FUTURE_VALUE,
                    inputs={"present_value": 1, "rate": 0.1, "periods": 1_000_001},
                )
            )

    def test_natural_language_calculation_is_planned_as_a_tool(self) -> None:
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="计算 CAGR，beginning=100, ending=150, years=3",
                run_id="natural-cagr",
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual([item.task.tool_name for item in outcome.state.observations], ["finance.calculate"])
        calculated = [item for item in outcome.state.bundle.evidence.values() if item.field_name == "cagr"]
        self.assertEqual(len(calculated), 1)
        self.assertIn("input_evidence_ids", calculated[0].source.metadata)


class AdaptiveScopeTests(unittest.TestCase):
    def test_finance_definition_answers_without_curated_knowledge_tool(self) -> None:
        outcome = llm_backed_agent(ToolHarness()).run(
            *agent_run_input(
                query="什么是市盈率，应该如何理解？",
                run_id="pe-concept",
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(outcome.state.observations, [])
        self.assertFalse(outcome.state.bundle.evidence)
        self.assertTrue(outcome.state.bundle.claims)
        claim = next(iter(outcome.state.bundle.claims.values()))
        self.assertEqual(claim.status.value, "inferred")
        self.assertEqual(claim.evidence_ids, ())


class DataAdapterTests(unittest.TestCase):
    def test_market_history_generates_provenance_bound_risk_metrics(self) -> None:
        class FakeHistory:
            def fetch_price_history(self, company, symbol=None, *, range_name="1y", interval="1d"):
                return {
                    "provider": "fake-market",
                    "company": company,
                    "symbol": symbol,
                    "currency": "USD",
                    "points": [
                        {"date": "2026-01-01", "close": 100},
                        {"date": "2026-01-02", "close": 90},
                        {"date": "2026-01-03", "close": 120},
                        {"date": "2026-01-04", "close": 110},
                    ],
                }

        batch = MarketHistoryEvidenceAdapter(FakeHistory()).fetch("ACME", "ACME")
        by_field = {item.field_name: item for item in batch.bundle.evidence.values()}
        self.assertAlmostEqual(float(by_field["total_return"].value), 0.1)
        self.assertAlmostEqual(float(by_field["max_drawdown"].value), -0.1)
        self.assertTrue(math.isfinite(float(by_field["annualized_volatility"].value)))
        for name in ("total_return", "annualized_return", "annualized_volatility", "max_drawdown"):
            inputs = by_field[name].source.metadata["input_evidence_ids"]
            self.assertTrue(all(item in batch.bundle.evidence for item in inputs))

    def test_fred_adapter_preserves_series_metadata_and_skips_missing_values(self) -> None:
        class FakeFRED:
            def fetch_series(self, series_id, *, observation_start=None, observation_end=None, limit=120):
                return {
                    "series": {
                        "id": series_id,
                        "title": "Test unemployment rate",
                        "units": "Percent",
                        "frequency": "Monthly",
                        "last_updated": "2026-07-01",
                    },
                    "observations": [
                        {"date": "2026-05-01", "value": "4.0"},
                        {"date": "2026-06-01", "value": "."},
                        {"date": "2026-07-01", "value": "4.2"},
                    ],
                    "retrieved_at": "2026-07-31T00:00:00Z",
                }

        batch = FREDEvidenceAdapter(FakeFRED()).fetch("UNRATE")
        by_field = {item.field_name: item for item in batch.bundle.evidence.values()}
        self.assertEqual(by_field["latest_value"].value, 4.2)
        self.assertAlmostEqual(float(by_field["change_from_previous"].value), 0.2)
        self.assertEqual(by_field["latest_value"].source.metadata["frequency"], "Monthly")

    def test_recent_sec_filings_preserve_primary_document_locators(self) -> None:
        class FakeFilings:
            def fetch_recent_filings(self, symbol):
                return {
                    "cik": "1234",
                    "name": "ACME Corp",
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-Q", "4"],
                            "accessionNumber": ["0001-26-000003", "0001-26-000002", "0001-26-000001"],
                            "filingDate": ["2026-07-03", "2026-06-01", "2026-05-01"],
                            "reportDate": ["2026-07-02", "2026-03-31", ""],
                            "primaryDocument": ["event.htm", "quarter.htm", "ownership.xml"],
                            "primaryDocDescription": ["Current report", "Quarterly report", "Form 4"],
                        }
                    },
                }

        batch = SECRecentFilingsAdapter(FakeFilings()).fetch("ACME", "ACME", forms=("10-Q", "8-K"), limit=5)
        self.assertEqual(len(batch.bundle.evidence), 2)
        locators = [item.source.locator for item in batch.bundle.evidence.values()]
        self.assertTrue(any(locator.endswith("/event.htm") for locator in locators))
        self.assertTrue(all("filing_metadata" in item.tags for item in batch.bundle.evidence.values()))


class ContextAndMemoryTests(unittest.TestCase):
    def test_calculation_coverage_uses_canonical_operation_not_display_label(self) -> None:
        bundle = EvidenceBundle()
        source = SourceRef.create(
            source_type=SourceType.CALCULATION,
            title="Calculated metric: annual growth",
            locator="formula://cagr/v1/request-1",
            provider="test",
            metadata={
                "request_id": "request-1",
                "operation": "cagr",
                "input_evidence_ids": ["input-1"],
            },
        )
        bundle.add_evidence(
            Evidence.create(
                source=source,
                content="annual growth = 0.1447",
                field_name="annual growth",
                value=0.1447,
                tags=("calculation", "cagr", "request-1"),
            )
        )
        scope = ResearchScope(
            intents=(FinancialIntent.CALCULATION,),
            requirements=(
                ResearchRequirement(
                    "calculation:query:1",
                    "calculation",
                    "Calculate CAGR.",
                    fields=("CAGR",),
                ),
            ),
        )
        self.assertTrue(CoverageAssessor().assess(ChatTurn(message="计算 CAGR"), bundle, scope).complete)

    def test_multi_document_intent_requires_at_least_two_documents_but_focused_question_does_not(self) -> None:
        bundle = EvidenceBundle()
        first_source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="first.pdf",
            locator="first.pdf#page=1",
            provider="test",
            metadata={"document_id": "first"},
        )
        bundle.add_evidence(Evidence.create(source=first_source, content="ACME covenant headroom is 20%."))
        focused = ChatTurn(message="根据文档回答 covenant headroom 是多少？")
        focused_scope = ResearchScope(
            intents=(FinancialIntent.DOCUMENT_RESEARCH,),
            requirements=(ResearchRequirement("document:query:1", "document", "Search supplied documents."),),
        )
        self.assertTrue(CoverageAssessor().assess(focused, bundle, focused_scope).complete)

        comparison = ChatTurn(message="对比这些 PDF 的 covenant 风险。")
        comparison_scope = ResearchScope(
            intents=(FinancialIntent.DOCUMENT_RESEARCH,),
            requirements=(
                ResearchRequirement(
                    "document:query:1",
                    "document",
                    "Compare supplied documents.",
                    parameters={"minimum_documents": 2},
                ),
            ),
        )
        self.assertFalse(CoverageAssessor().assess(comparison, bundle, comparison_scope).complete)
        second_source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="second.pdf",
            locator="second.pdf#page=1",
            provider="test",
            metadata={"document_id": "second"},
        )
        bundle.add_evidence(Evidence.create(source=second_source, content="ACME covenant risk increased."))
        self.assertTrue(CoverageAssessor().assess(comparison, bundle, comparison_scope).complete)

    def test_context_balances_documents_not_only_entities(self) -> None:
        bundle = EvidenceBundle()
        for document_id, suffix in (("document-a", "one"), ("document-a", "two"), ("document-b", "three")):
            source = SourceRef.create(
                source_type=SourceType.DOCUMENT,
                title=f"ACME {document_id}",
                locator=f"{document_id}.pdf#{suffix}",
                provider="test",
                metadata={
                    "document_id": document_id,
                    "retrieval_trace": {"document_diversification": True},
                },
            )
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content=f"ACME liquidity {suffix}. " + ("x" * 300),
                    entity="ACME",
                )
            )
        payload, manifest = FinancialContextAssembler(
            max_evidence_tokens=1_400,
            count_tokens=len,
        ).build(ChatTurn(message="ACME liquidity"), AgentContext(), bundle)
        locators = {item["source"]["locator"].split(".pdf", maxsplit=1)[0] for item in payload["evidence"]}
        self.assertEqual(locators, {"document-a", "document-b"})
        self.assertEqual(len(manifest.groups), 2)

    def test_context_balances_entities_and_includes_source_provenance(self) -> None:
        bundle = EvidenceBundle()
        for entity in ("Alpha", "Beta"):
            source = SourceRef.create(
                source_type=SourceType.DOCUMENT,
                title=f"{entity} filing",
                locator=f"{entity}.pdf#page=1",
                provider="test",
            )
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content=f"{entity} reported evidence. " + ("x" * 350),
                    entity=entity,
                )
            )
        turn = ChatTurn(message="Compare Alpha and Beta")
        agent_context = AgentContext(
            thread_context={
                "manifest": {"memory_is_evidence": False},
                "forbidden": "drop me",
            },
        )
        payload, manifest = FinancialContextAssembler(
            max_evidence_tokens=1_600,
            count_tokens=len,
        ).build(turn, agent_context, bundle)
        self.assertEqual({item["entity"] for item in payload["evidence"]}, {"Alpha", "Beta"})
        self.assertTrue(all("provider" in item["source"] for item in payload["evidence"]))
        self.assertNotIn("forbidden", payload["thread_context"])
        self.assertEqual(manifest.omitted_evidence_count, 0)

    def test_context_keeps_complete_retrieved_passage_without_per_item_character_window(self) -> None:
        content = "prefix " + ("long financial passage " * 200) + "semantic conclusion at the end"
        source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="Long report",
            locator="long.pdf#page=1",
            provider="test",
        )
        bundle = EvidenceBundle()
        bundle.add_evidence(Evidence.create(source=source, content=content))

        payload, _manifest = FinancialContextAssembler(
            max_evidence_tokens=10_000,
            count_tokens=len,
        ).build(ChatTurn(message="What is the conclusion?"), AgentContext(), bundle)

        self.assertEqual(payload["evidence"][0]["content"], content)
        self.assertTrue(payload["evidence"][0]["content"].endswith("semantic conclusion at the end"))


if __name__ == "__main__":
    unittest.main()
