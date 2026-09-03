from __future__ import annotations

import os
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from mas_finance.config import AppConfig
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.harness import (
    Tool,
    ToolArgumentContract,
    ToolContext,
    ToolExecutionError,
    ToolResultKind,
    ToolSpec,
    function_tool,
)
from mas_finance.service import FinanceAnalysisService


@unittest.skipUnless(
    os.getenv("MAS_RUN_LIVE_LLM_TESTS") == "1",
    "set MAS_RUN_LIVE_LLM_TESTS=1 to spend DeepSeek quota on live contract tests",
)
class LiveDeepSeekTests(unittest.TestCase):
    def _service(
        self,
        directory: str,
        *,
        evidence_tools: tuple[Tool, ...] = (),
    ) -> FinanceAnalysisService:
        root = Path(directory)
        config = replace(
            AppConfig.from_env(),
            output_dir=root / "outputs",
            upload_dir=root / "uploads",
            db_path=root / "live.db",
            database_url=f"sqlite:///{root / 'live.db'}",
            market_data_provider="offline",
            sec_user_agent=None,
            fred_api_key=None,
            bocha_search_api_key=None,
            brave_search_api_key=None,
            embedding_endpoint=None,
            embedding_model=None,
            embedding_api_key=None,
            conversation_memory_enabled=False,
            personal_memory_enabled=False,
            personal_knowledge_enabled=False,
            automatic_memory_consolidation_enabled=False,
            automatic_skill_learning_enabled=False,
            mcp_servers=(),
        )
        return FinanceAnalysisService(config, evidence_tools=evidence_tools)

    def test_concept_question_uses_all_three_model_stages_without_retrieval(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = self._service(directory)
            response = service.analyze(
                "请用两句话解释债券久期的含义；这是概念题，不需要检索。",
                thread_id="live-concept",
                run_id="live-concept-run",
                allow_network=False,
                export_artifacts=False,
                use_personal_memory=False,
                use_personal_knowledge=False,
            )
            result = response["result"]
            diagnostic = {
                "stop_reason": result["stop_reason"],
                "coverage": result["coverage"],
                "gaps": result["gaps"],
                "tools": [
                    (item["tool_name"], item["result_status"], item.get("error_code"))
                    for item in result["audit_events"]
                ],
                "claims": [item["status"] for item in result["bundle"]["claims"]],
                "validation_issues": result["validation_issues"],
            }
            self.assertEqual(result["status"], "succeeded", diagnostic)
            self.assertEqual(result["stop_reason"], "coverage_satisfied")
            self.assertEqual((result["scope"] or {})["requirements"], [])
            self.assertEqual(result["bundle"]["evidence"], [])
            self.assertGreaterEqual(len(result["bundle"]["claims"]), 1)
            self.assertEqual(
                [item["tool_name"] for item in result["audit_events"]],
                ["llm.task_frame", "llm.plan", "llm.synthesize"],
            )
            self.assertFalse(
                any(item["severity"] == "error" for item in result["validation_issues"])
            )
            service.close()

    def test_calculation_question_selects_deterministic_function_tool(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = self._service(directory)
            response = service.analyze(
                "期初价值 100，期末价值 150，期限 3 年。请调用确定性金融计算工具计算 CAGR。",
                thread_id="live-calculation",
                run_id="live-calculation-run",
                allow_network=False,
                export_artifacts=False,
                use_personal_memory=False,
                use_personal_knowledge=False,
            )
            result = response["result"]
            diagnostic = {
                "stop_reason": result["stop_reason"],
                "coverage": result["coverage"],
                "gaps": result["gaps"],
                "tools": [
                    (item["tool_name"], item["result_status"], item.get("error_code"))
                    for item in result["audit_events"]
                ],
                "claims": [item["status"] for item in result["bundle"]["claims"]],
                "validation_issues": result["validation_issues"],
            }
            self.assertEqual(result["status"], "succeeded", diagnostic)
            self.assertIn(
                "calculation",
                {item["category"] for item in (result["scope"] or {})["requirements"]},
            )
            self.assertIn("finance.calculate", [item["tool_name"] for item in result["audit_events"]])
            self.assertTrue(
                any(item["source"]["source_type"] == "calculation" for item in result["bundle"]["evidence"])
            )
            self.assertGreaterEqual(len(result["bundle"]["claims"]), 1)
            self.assertFalse(
                any(item["severity"] == "error" for item in result["validation_issues"])
            )
            service.close()

    def test_actionable_tool_error_is_returned_to_model_and_corrected(self) -> None:
        attempts: list[str] = []

        def quote(arguments: Mapping[str, Any], _context: ToolContext) -> Mapping[str, Any]:
            symbol = str(arguments["symbol"])
            attempts.append(symbol)
            if len(attempts) == 1 or symbol != "BRK-B":
                raise ToolExecutionError(
                    "unknown_symbol",
                    f"provider does not recognize symbol {symbol!r}",
                    details={
                        "field": "symbol",
                        "received": symbol,
                        "suggested_symbol": "BRK-B",
                        "model_action": "change_arguments",
                    },
                )
            source = SourceRef.create(
                source_type=SourceType.MARKET_DATA,
                title="Controlled live quote fixture",
                locator="fixture://live-quote/BRK-B",
                provider="controlled-live-provider",
                as_of="2026-09-02",
            )
            bundle = EvidenceBundle()
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content="BRK-B current price is 500 USD in the controlled live fixture.",
                    entity="BRK.B",
                    field_name="current_price",
                    value=500,
                    unit="USD",
                )
            )
            return {"bundle": bundle.to_dict(), "gaps": []}

        tool = function_tool(
            ToolSpec(
                name="live.quote",
                description=(
                    "Read a controlled current quote. Pass the user's symbol first; when unknown_symbol "
                    "is returned, call again using details.suggested_symbol."
                ),
                capability="market.read",
                result_kind=ToolResultKind.EVIDENCE_BUNDLE,
                arguments=ToolArgumentContract(required=frozenset({"symbol"})),
                input_schema={
                    "type": "object",
                    "required": ["symbol"],
                    "additionalProperties": False,
                    "properties": {"symbol": {"type": "string", "minLength": 1, "maxLength": 32}},
                },
            ),
            quote,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = self._service(directory, evidence_tools=(tool,))
            response = service.analyze(
                "请使用 live.quote 查询 BRK.B 当前价格；如果工具返回符号建议，请修正参数后重试。",
                thread_id="live-tool-correction",
                run_id="live-tool-correction-run",
                allow_network=False,
                export_artifacts=False,
                use_personal_memory=False,
                use_personal_knowledge=False,
            )
            result = response["result"]
            quote_audit = [item for item in result["audit_events"] if item["tool_name"] == "live.quote"]
            self.assertEqual([item["result_status"] for item in quote_audit], ["error", "success"])
            self.assertEqual(quote_audit[0]["error_code"], "unknown_symbol")
            self.assertEqual(attempts[-1], "BRK-B")
            self.assertEqual(result["status"], "succeeded")
            service.close()


if __name__ == "__main__":
    unittest.main()
