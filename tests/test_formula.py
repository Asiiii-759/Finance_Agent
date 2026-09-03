from __future__ import annotations

import json
import unittest

from llm_fixtures import agent_run_input, llm_backed_agent

from mas_finance.formula import evaluate_formula, formula_harness_tool
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.llm import BaseLLMClient
from mas_finance.planning import ModelPlanner, llm_planning_harness_tool


class FormulaPlanningLLM(BaseLLMClient):
    backend_name = "scripted"

    def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1200):
        return json.dumps(
            {
                "action": "call_tool",
                "tool_name": "finance.formula",
                "arguments": {
                    "expression": "(revenue - cost) / revenue",
                    "inputs": {"revenue": 125, "cost": 80},
                    "label": "gross_margin",
                    "unit": "ratio",
                    "entity": "ACME",
                },
                "reason": "Evaluate the requested declarative formula with deterministic arithmetic.",
            }
        )


class DeclarativeFormulaTests(unittest.TestCase):
    def test_model_can_select_formula_but_final_claim_is_semantically_qualified(self) -> None:
        harness = ToolHarness()
        harness.register(formula_harness_tool())
        harness.register(llm_planning_harness_tool(FormulaPlanningLLM(), network_access=False))
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="Calculate ACME gross margin using (revenue-cost)/revenue.",
                max_iterations=1,
                max_model_calls=4,
                run_id="formula-planning",
            )
        )
        self.assertEqual(outcome.state.observations[0].task.tool_name, "finance.formula")
        claim = next(iter(outcome.state.bundle.claims.values()))
        self.assertEqual(claim.status.value, "inferred")
        self.assertIn("formula semantics", claim.caveat)

    def test_model_designed_formula_is_reproducible_and_has_lineage(self) -> None:
        harness = ToolHarness()
        harness.register(formula_harness_tool())
        result = harness.invoke(
            "finance.formula",
            {
                "expression": "(revenue - cost) / revenue",
                "inputs": {"revenue": 125, "cost": 80},
                "label": "gross_margin",
                "unit": "ratio",
                "entity": "ACME",
            },
            ToolContext(
                run_id="formula-run",
                thread_id="formula-thread",
                policy=ExecutionPolicy(allowed_capabilities=frozenset({"calculation"})),
            ),
        )
        self.assertTrue(result.ok)
        evidence = result.data["bundle"]["evidence"]
        calculated = next(item for item in evidence if item["field_name"] == "gross_margin")
        self.assertEqual(calculated["value"], 0.36)
        self.assertEqual(len(calculated["source"]["metadata"]["input_evidence_ids"]), 1)

    def test_formula_rejects_code_attributes_unknown_variables_and_unbounded_power(self) -> None:
        for expression in (
            "__import__('os').system('id')",
            "portfolio.value",
            "missing + 1",
            "10 ** 1000",
            "[value for value in values]",
        ):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                evaluate_formula(expression, {"portfolio": 1, "values": 2})

    def test_formula_domain_and_numeric_checks_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "real finite domain"):
            evaluate_formula("1 / denominator", {"denominator": 0})
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            evaluate_formula("value", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
