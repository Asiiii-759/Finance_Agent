from __future__ import annotations

import unittest

from mas_finance.contracts import SourceType
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.sec import SECCompanyFactsAdapter, sec_company_facts_harness_tool


class FakeSECClient:
    def fetch_company_facts(self, symbol):
        return {
            "cik": 1234,
            "entityName": "ACME Corp",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100,
                                    "end": "2025-12-31",
                                    "filed": "2026-02-01",
                                    "form": "10-K",
                                    "accn": "0001",
                                    "fy": 2025,
                                    "fp": "FY",
                                },
                                {
                                    "val": 120,
                                    "start": "2026-01-01",
                                    "end": "2026-03-31",
                                    "filed": "2026-05-01",
                                    "form": "10-Q",
                                    "accn": "0002",
                                    "fy": 2026,
                                    "fp": "Q1",
                                },
                            ]
                        }
                    }
                }
            },
        }


class SECAdapterTests(unittest.TestCase):
    def test_latest_filed_fact_becomes_regulatory_evidence(self) -> None:
        batch = SECCompanyFactsAdapter(FakeSECClient()).fetch("ACME", "ACME")
        evidence = next(iter(batch.bundle.evidence.values()))
        self.assertEqual(evidence.source.source_type, SourceType.REGULATORY_FILING)
        self.assertEqual(evidence.value, 120)
        self.assertEqual(evidence.period, "2026-01-01/2026-03-31")
        self.assertIn("accn=0002", evidence.source.locator)

    def test_sec_tool_is_network_gated(self) -> None:
        harness = ToolHarness()
        harness.register(sec_company_facts_harness_tool(SECCompanyFactsAdapter(FakeSECClient())))
        denied = harness.invoke(
            "sec.company_facts",
            {"company": "ACME", "symbol": "ACME"},
            ToolContext(
                run_id="sec-denied",
                thread_id="thread",
                policy=ExecutionPolicy(allowed_capabilities=frozenset({"regulatory.read"})),
            ),
        )
        self.assertEqual(denied.error_code, "network_denied")


if __name__ == "__main__":
    unittest.main()
