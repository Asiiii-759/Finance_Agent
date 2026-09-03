"""Deterministic black-box acceptance scenarios for the finance agent."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig
from .llm import BaseLLMClient, LLMSettings, build_llm_client
from .retrieval import RetrievalSource
from .service import FinanceAnalysisService


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    query: str
    expected_status: str
    expected_tools: tuple[str, ...] = ()
    expected_gaps: tuple[str, ...] = ()
    market_provider: str = "offline"
    request_network: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    passed: bool
    status: str
    tools: tuple[str, ...]
    gaps: tuple[str, ...]
    budget_usage: dict[str, int]
    failures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "tools": list(self.tools),
            "gaps": list(self.gaps),
            "budget_usage": dict(self.budget_usage),
            "failures": list(self.failures),
        }


ENTERPRISE_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        name="natural_chinese_cagr",
        query="一项投资从100增长到121，用了2年，CAGR是多少？",
        expected_status="succeeded",
        expected_tools=("finance.calculate",),
    ),
    EvaluationCase(
        name="model_finance_definition",
        query="什么是市盈率，使用时有哪些局限？",
        expected_status="succeeded",
        expected_tools=(),
    ),
    EvaluationCase(
        name="structured_sharpe_ratio",
        query="根据收益率 [0.01,-0.005,0.02,0.004]、年化因子252和年化无风险利率0.03计算夏普比率",
        expected_status="succeeded",
        expected_tools=("finance.calculate",),
    ),
    EvaluationCase(
        name="model_interest_rate_transmission",
        query="利率上升如何影响银行股？",
        expected_status="succeeded",
        expected_tools=(),
    ),
    EvaluationCase(
        name="unsupported_precise_forecast",
        query="请预测明天一只未指定股票的精确收盘价。",
        expected_status="failed",
        expected_gaps=("unsupported_research_scope",),
    ),
    EvaluationCase(
        name="offline_market_fails_closed",
        query="Apple 当前股价是多少？",
        expected_status="failed",
        expected_tools=("market.snapshot",),
        expected_gaps=("market_provider_unavailable",),
    ),
    EvaluationCase(
        name="network_requires_server_and_request_consent",
        query="Apple 当前股价是多少？",
        market_provider="yahoo",
        request_network=False,
        expected_status="failed",
        expected_tools=("market.snapshot",),
        expected_gaps=("network_denied",),
    ),
)


def run_enterprise_evaluation(*, llm_client: BaseLLMClient | None = None) -> dict[str, Any]:
    client = llm_client or build_llm_client()
    if client is None:
        raise RuntimeError("an LLM configuration is required for financial research")
    results: list[EvaluationResult] = []
    with tempfile.TemporaryDirectory(prefix="mas-finance-eval-") as directory:
        root = Path(directory)
        for index, case in enumerate(ENTERPRISE_CASES):
            service = FinanceAnalysisService(
                _evaluation_config(root / str(index), case.market_provider),
                llm_client=client,
            )
            response = service.analyze(
                case.query,
                allow_network=case.request_network,
                export_artifacts=False,
            )
            result = response["result"]
            status = str(result["status"])
            tools = _research_tools(result)
            gaps = tuple(
                dict.fromkeys(str(item["code"]) for item in result.get("gaps", []) if not item.get("resolved", False))
            )
            failures: list[str] = []
            if status != case.expected_status:
                failures.append(f"status expected {case.expected_status}, received {status}")
            missing_tools = set(case.expected_tools).difference(tools)
            if missing_tools:
                failures.append(f"missing tools: {sorted(missing_tools)}")
            unexpected_tools = set(tools).difference(case.expected_tools)
            if unexpected_tools:
                failures.append(f"unexpected tools: {sorted(unexpected_tools)}")
            missing_gaps = set(case.expected_gaps).difference(gaps)
            if missing_gaps:
                failures.append(f"missing gaps: {sorted(missing_gaps)}")
            results.append(
                EvaluationResult(
                    name=case.name,
                    passed=not failures,
                    status=status,
                    tools=tools,
                    gaps=gaps,
                    budget_usage=dict(result.get("budget_usage") or {}),
                    failures=tuple(failures),
                )
            )

        results.append(_evaluate_thread_entity_switch(root / "thread-memory", client))
        results.append(_evaluate_injected_rag(root / "injected-rag", client))

    return {
        "suite": "mas-finance-enterprise-black-box-v2",
        "passed": all(item.passed for item in results),
        "case_count": len(results),
        "passed_count": sum(item.passed for item in results),
        "results": [item.to_dict() for item in results],
    }


def _evaluate_thread_entity_switch(root: Path, llm_client: BaseLLMClient) -> EvaluationResult:
    service = FinanceAnalysisService(_evaluation_config(root, "offline"), llm_client=llm_client)
    service.analyze(
        "解释 Apple 的市盈率",
        thread_id="entity-switch",
        export_artifacts=False,
    )
    response = service.analyze(
        "Microsoft 的最大回撤呢？",
        thread_id="entity-switch",
        export_artifacts=False,
    )["result"]
    entities = tuple(
        (str(item.get("name")), str(item.get("origin")))
        for item in (response.get("task_frame") or {}).get("entities") or ()
    )
    expected = (("Microsoft", "current_request"),)
    failures = () if entities == expected else (f"entity bleed detected: {entities}",)
    return EvaluationResult(
        name="thread_memory_does_not_override_current_entity",
        passed=not failures,
        status=str(response["status"]),
        tools=_research_tools(response),
        gaps=tuple(
            dict.fromkeys(str(item["code"]) for item in response.get("gaps", []) if not item.get("resolved", False))
        ),
        budget_usage=dict(response.get("budget_usage") or {}),
        failures=failures,
    )


def _evaluate_injected_rag(root: Path, llm_client: BaseLLMClient) -> EvaluationResult:
    class EvaluationRAG:
        filters: dict[str, Any] = {}

        def search_json(self, payload):
            self.filters = dict(payload.get("filters") or {})
            return {
                "chunks": [
                    {
                        "id": "risk-1",
                        "content": "ACME covenant headroom narrowed during the quarter.",
                        "metadata": {
                            "company": "ACME",
                            "file_name": "credit-review.pdf",
                            "source_page": 7,
                        },
                    }
                ],
                "trace": {"request_id": "evaluation-rag-1", "search_mode": "rrf"},
            }

    client = EvaluationRAG()
    service = FinanceAnalysisService(
        _evaluation_config(root, "offline"),
        llm_client=llm_client,
        retrieval_sources=(
            RetrievalSource(
                "internal.credit_search",
                client,
                "evaluation_corpus",
                fixed_filters={"tenant_id": "evaluation"},
            ),
        ),
    )
    response = service.analyze(
        "根据内部文档说明 ACME 的 covenant 风险",
        export_artifacts=False,
    )["result"]
    tools = _research_tools(response)
    document_evidence = [item for item in response["bundle"]["evidence"] if item["source"]["source_type"] == "document"]
    failures: list[str] = []
    if response["status"] != "succeeded":
        failures.append(f"status expected succeeded, received {response['status']}")
    if tools != ("internal.credit_search",):
        failures.append(f"unexpected tools: {tools}")
    if len(document_evidence) != 1 or document_evidence[0]["page"] != 7:
        failures.append("RAG page provenance was not preserved")
    if client.filters != {"tenant_id": "evaluation"}:
        failures.append(f"server filters were not enforced: {client.filters}")
    return EvaluationResult(
        name="injected_rag_preserves_acl_and_provenance",
        passed=not failures,
        status=str(response["status"]),
        tools=tools,
        gaps=tuple(
            dict.fromkeys(str(item["code"]) for item in response.get("gaps", []) if not item.get("resolved", False))
        ),
        budget_usage=dict(response.get("budget_usage") or {}),
        failures=tuple(failures),
    )


def _research_tools(result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item["tool_name"])
            for item in result.get("audit_events", [])
            if not str(item.get("tool_name") or "").startswith("llm.")
        )
    )


def _evaluation_config(root: Path, market_provider: str) -> AppConfig:
    db_path = root / "finance.db"
    return AppConfig(
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        market_data_provider=market_provider,
        alphavantage_api_key=None,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        llm=LLMSettings(None, "https://api.deepseek.com", "deepseek-v4-flash", 10),
        allow_network=False,
    )


def main() -> None:
    report = run_enterprise_evaluation()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
