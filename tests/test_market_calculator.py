from __future__ import annotations

import unittest

import httpx

from mas_finance.agent import reconcile_conflicts
from mas_finance.calculator import CalculationError, calculate_ratio, derive_standard_ratios
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.market import MarketEvidenceAdapter
from mas_finance.market_data import MarketDataClient
from mas_finance.mcp_servers.market import ExternalMarketClient, _tools
from mas_finance.metrics import financial_calculation_harness_tool


class FakeMarketClient:
    def fetch_company_snapshot(self, company, symbol=None):
        return {
            "provider": "fake-exchange",
            "symbol": symbol or "ACME",
            "company": company,
            "current_price": 25.0,
            "market_cap": 1_000_000.0,
            "monthly_return": 0.1,
            "currency": "USD",
            "as_of": "2026-07-30T20:00:00Z",
            "retrieved_at": "2026-07-31T01:00:00Z",
        }


class MarketAdapterTests(unittest.TestCase):
    def test_external_market_mcp_requires_explicit_symbol_and_propagates_network_failure(self) -> None:
        schemas = {tool["name"]: tool["inputSchema"] for tool in _tools()}
        self.assertEqual(schemas["snapshot"]["required"], ["company", "symbol"])
        self.assertEqual(schemas["history"]["required"], ["company", "symbol"])

        client = ExternalMarketClient(
            alltick_token="configured",
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request))
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit market symbol"):
            client.fetch_company_snapshot("ACME")
        with self.assertRaises(httpx.ConnectError):
            client.fetch_company_snapshot("ACME", "ACME")

    def test_calculation_tool_exposes_operation_specific_nested_schema(self) -> None:
        schema = financial_calculation_harness_tool().spec.input_schema
        self.assertIsNotNone(schema)
        variants = schema["properties"]["requests"]["items"]["oneOf"]
        cagr = next(item for item in variants if item["properties"]["operation"]["const"] == "cagr")
        self.assertEqual(
            cagr["properties"]["inputs"]["required"],
            ["beginning_value", "ending_value", "years"],
        )

    def test_alphavantage_configuration_never_silently_falls_back_to_yahoo(self) -> None:
        class SpyClient(MarketDataClient):
            yahoo_called = False

            def _fetch_from_yahoo(self, company, ticker):
                self.yahoo_called = True
                raise AssertionError("Yahoo must not be used for an AlphaVantage configuration")

        missing_key = SpyClient(provider="alphavantage")
        snapshot = missing_key.fetch_company_snapshot("ACME", "ACME")
        history = missing_key.fetch_price_history("ACME", "ACME")
        self.assertFalse(missing_key.yahoo_called)
        self.assertEqual(snapshot["provider"], "alphavantage")
        self.assertEqual(snapshot["error_codes"], ["alphavantage_api_key_missing"])
        self.assertEqual(history["error_codes"], ["alphavantage_api_key_missing"])

        class FailingAlphaVantage(SpyClient):
            def _fetch_from_alphavantage(self, company, ticker):
                raise ConnectionError("provider unavailable")

            def _history_from_alphavantage(self, company, ticker, range_name, interval):
                raise ConnectionError("provider unavailable")

        failed = FailingAlphaVantage(provider="alphavantage", alphavantage_api_key="configured")
        with self.assertRaises(ConnectionError):
            failed.fetch_company_snapshot("ACME", "ACME")
        with self.assertRaises(ConnectionError):
            failed.fetch_price_history("ACME", "ACME")
        self.assertFalse(failed.yahoo_called)

    def test_market_fields_become_individual_evidence(self) -> None:
        batch = MarketEvidenceAdapter(FakeMarketClient()).fetch("ACME", "ACME")
        by_field = {item.field_name: item for item in batch.bundle.evidence.values()}
        self.assertEqual(by_field["current_price"].value, 25.0)
        self.assertEqual(by_field["current_price"].unit, "USD")
        self.assertEqual(by_field["current_price"].source.as_of, "2026-07-30T20:00:00Z")
        self.assertEqual(batch.gaps[0].code, "market_fields_missing")

    def test_empty_provider_result_is_a_gap_not_fake_data(self) -> None:
        class EmptyClient:
            def fetch_company_snapshot(self, company, symbol=None):
                return {"provider": "offline", "symbol": symbol, "company": company}

        batch = MarketEvidenceAdapter(EmptyClient()).fetch("ACME", "ACME")
        self.assertEqual(batch.bundle.evidence, {})
        self.assertEqual(
            [gap.code for gap in batch.gaps],
            ["market_provider_unavailable", "market_as_of_missing"],
        )


class CalculatorTests(unittest.TestCase):
    def make_input(self, bundle, field, value, unit="USD", period="FY2025", entity="ACME"):
        source = SourceRef.create(
            source_type=SourceType.DOCUMENT,
            title="Annual report",
            locator=f"report.pdf#{field}",
            provider="test",
            as_of=period,
        )
        evidence = Evidence.create(
            source=source,
            content=f"{field}: {value}",
            entity=entity,
            field_name=field,
            value=value,
            unit=unit,
            period=period,
        )
        bundle.add_evidence(evidence)
        return evidence

    def test_ratio_preserves_inputs_and_adds_calculation_evidence(self) -> None:
        bundle = EvidenceBundle()
        profit = self.make_input(bundle, "profit", 20.0)
        revenue = self.make_input(bundle, "revenue", 100.0)
        result = calculate_ratio(
            bundle,
            metric_name="profit_margin",
            numerator_id=profit.evidence_id,
            denominator_id=revenue.evidence_id,
        )
        self.assertEqual(result.value, 0.2)
        self.assertEqual(result.unit, "ratio")
        self.assertEqual(
            result.source.metadata["input_evidence_ids"],
            [profit.evidence_id, revenue.evidence_id],
        )

    def test_ratio_rejects_zero_unit_and_period_mismatch(self) -> None:
        bundle = EvidenceBundle()
        numerator = self.make_input(bundle, "profit", 20.0)
        zero = self.make_input(bundle, "revenue", 0.0)
        with self.assertRaisesRegex(CalculationError, "denominator is zero"):
            calculate_ratio(
                bundle,
                metric_name="margin",
                numerator_id=numerator.evidence_id,
                denominator_id=zero.evidence_id,
            )

        other = self.make_input(bundle, "assets", 10.0, unit="CNY")
        with self.assertRaisesRegex(CalculationError, "different units"):
            calculate_ratio(
                bundle,
                metric_name="bad_ratio",
                numerator_id=numerator.evidence_id,
                denominator_id=other.evidence_id,
            )

    def test_standard_ratios_are_derived_only_from_aligned_facts(self) -> None:
        bundle = EvidenceBundle()
        self.make_input(bundle, "net_income", 25.0)
        self.make_input(bundle, "revenue", 100.0)
        derived = derive_standard_ratios(bundle)
        self.assertEqual([(item.field_name, item.value) for item in derived], [("net_margin", 0.25)])
        self.assertEqual(derived[0].source.source_type, SourceType.CALCULATION)
        evidence_count = len(bundle.evidence)
        self.assertEqual(derive_standard_ratios(bundle), [])
        self.assertEqual(len(bundle.evidence), evidence_count)

    def test_conflicting_facts_suppress_calculation_and_become_conflicted_claim(self) -> None:
        bundle = EvidenceBundle()
        self.make_input(bundle, "net_income", 25.0)
        self.make_input(bundle, "net_income", 30.0)
        self.make_input(bundle, "revenue", 100.0)
        self.assertEqual(derive_standard_ratios(bundle), [])
        reconciled = reconcile_conflicts(bundle, ())
        conflict = next(claim for claim in reconciled if claim.status.value == "conflicted")
        self.assertEqual(len(conflict.evidence_ids), 2)


if __name__ == "__main__":
    unittest.main()
