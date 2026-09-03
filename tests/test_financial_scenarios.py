from __future__ import annotations

import unittest

from llm_fixtures import agent_run_input, llm_backed_agent

from mas_finance.harness import ToolHarness
from mas_finance.macro import FREDEvidenceAdapter, fred_series_harness_tool
from mas_finance.market import MarketHistoryEvidenceAdapter, market_history_harness_tool
from mas_finance.sec import SECCompanyFactsAdapter, sec_company_facts_harness_tool


class FixtureSEC:
    def fetch_company_facts(self, symbol):
        revenue, income = {
            "AAPL": (400.0, 100.0),
            "MSFT": (300.0, 90.0),
        }[symbol]

        def fact(value, accession):
            return {
                "val": value,
                "form": "10-K",
                "start": "2025-01-01",
                "end": "2025-12-31",
                "filed": "2026-02-01",
                "accn": accession,
                "fy": 2025,
                "fp": "FY",
                "frame": "CY2025",
            }

        return {
            "cik": 1 if symbol == "AAPL" else 2,
            "entityName": symbol,
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {"USD": [fact(revenue, f"{symbol}-revenue")]}
                    },
                    "NetIncomeLoss": {"units": {"USD": [fact(income, f"{symbol}-income")]}},
                }
            },
        }


class FixtureHistory:
    def fetch_price_history(self, company, symbol=None, *, range_name="1y", interval="1d"):
        return {
            "provider": "fixture-market",
            "company": company,
            "symbol": symbol,
            "currency": "USD",
            "price_basis": "adjusted_close",
            "points": [
                {"date": "2026-01-02", "close": 100.0},
                {"date": "2026-01-03", "close": 80.0},
                {"date": "2026-01-04", "close": 120.0},
                {"date": "2026-01-05", "close": 110.0},
            ],
        }


class FixtureFRED:
    def fetch_series(self, series_id, *, observation_start=None, observation_end=None, limit=120):
        del observation_start, observation_end, limit
        return {
            "series": {
                "id": series_id,
                "title": "US unemployment rate",
                "units": "Percent",
                "frequency": "Monthly",
                "last_updated": "2026-08-01",
            },
            "observations": [
                {"date": "2026-06-01", "value": "4.1"},
                {"date": "2026-07-01", "value": "4.2"},
            ],
            "retrieved_at": "2026-08-09T00:00:00Z",
        }


class FinancialScenarioTests(unittest.TestCase):
    def test_comparative_profitability_uses_sec_then_aligned_ratios(self) -> None:
        harness = ToolHarness()
        harness.register(sec_company_facts_harness_tool(SECCompanyFactsAdapter(FixtureSEC())))
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="比较 Apple 和 Microsoft 的净利率和盈利能力",
                allow_network=True,
                run_id="scenario-profitability",
            )
        )
        margins = {
            item.entity: item.value
            for item in outcome.state.bundle.evidence.values()
            if item.field_name == "net_margin"
        }
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(margins, {"Apple": 0.25, "Microsoft": 0.3})
        self.assertEqual(
            [item.task.tool_name for item in outcome.state.observations],
            ["sec.company_facts", "sec.company_facts"],
        )

    def test_market_risk_question_uses_adjusted_series_with_lineage(self) -> None:
        harness = ToolHarness()
        harness.register(market_history_harness_tool(MarketHistoryEvidenceAdapter(FixtureHistory())))
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="Alpha过去1年的收益率、波动率和最大回撤是多少？",
                allow_network=True,
                run_id="scenario-risk",
            )
        )
        fields = {item.field_name: item for item in outcome.state.bundle.evidence.values()}
        self.assertEqual(outcome.status, "succeeded")
        self.assertAlmostEqual(float(fields["total_return"].value), 0.1)
        self.assertAlmostEqual(float(fields["max_drawdown"].value), -0.2)
        self.assertIn("history_observations", fields)
        self.assertIn(
            fields["history_observations"].evidence_id,
            fields["annualized_volatility"].source.metadata["input_evidence_ids"],
        )

    def test_unadjusted_history_is_usable_but_visibly_degraded(self) -> None:
        class UnadjustedHistory(FixtureHistory):
            def fetch_price_history(self, *args, **kwargs):
                result = super().fetch_price_history(*args, **kwargs)
                result["price_basis"] = "close"
                return result

        harness = ToolHarness()
        harness.register(market_history_harness_tool(MarketHistoryEvidenceAdapter(UnadjustedHistory())))
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="Alpha过去1年的收益率和最大回撤是多少？",
                allow_network=True,
                run_id="scenario-unadjusted-risk",
            )
        )
        self.assertEqual(outcome.status, "degraded")
        self.assertIn("unadjusted_price_history", {item.code for item in outcome.state.gaps})

    def test_macro_question_uses_latest_fred_observation_and_cited_change(self) -> None:
        harness = ToolHarness()
        harness.register(fred_series_harness_tool(FREDEvidenceAdapter(FixtureFRED())))
        outcome = llm_backed_agent(harness).run(
            *agent_run_input(
                query="美国失业率目前是多少？",
                allow_network=True,
                run_id="scenario-macro",
            )
        )
        fields = {item.field_name: item for item in outcome.state.bundle.evidence.values()}
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(fields["latest_value"].value, 4.2)
        self.assertAlmostEqual(float(fields["change_from_previous"].value), 0.1)
        self.assertEqual(fields["change_from_previous"].source.source_type.value, "calculation")
        self.assertTrue(fields["change_from_previous"].source.metadata["input_evidence_ids"])


if __name__ == "__main__":
    unittest.main()
