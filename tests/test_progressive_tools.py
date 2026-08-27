from __future__ import annotations

import json
import sys
import unittest

from mas_finance.agent import ResearchRequest
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.graph import FinancialResearchAgent
from mas_finance.harness import ToolHarness
from mas_finance.llm import BaseLLMClient
from mas_finance.mcp import (
    MCPHost,
    McpServerConfig,
    builtin_extmarket_server_config,
    mcp_discovery_tools,
)
from mas_finance.mcp_servers.market import (
    parse_alltick_kline,
    parse_alltick_tick,
    parse_biying_history,
    parse_biying_realtime,
)
from mas_finance.metrics import financial_calculation_harness_tool
from mas_finance.planning import ModelPlanner, llm_planning_harness_tool


class ScriptedLLM(BaseLLMClient):
    backend_name = "scripted"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.user_prompts: list[str] = []

    def chat(self, system_prompt, user_prompt, temperature=0.2, max_tokens=600):
        self.user_prompts.append(user_prompt)
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


def _policy_bundle(query: str = "单一发行人") -> dict:
    source = SourceRef.create(
        source_type=SourceType.DOCUMENT,
        title="Portfolio policy",
        locator="mcp://memory/portfolio/policy",
        provider="memory-mcp",
    )
    bundle = EvidenceBundle()
    bundle.add_evidence(
        Evidence.create(
            source=source,
            content=f"单一发行人限额为百分之五。检索词：{query}",
        )
    )
    return {"bundle": bundle.to_dict(), "gaps": []}


class FakeMCPClient:
    def __init__(self, listed: tuple[dict, ...], results: dict[str, dict]) -> None:
        self.listed = listed
        self.results = results
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def list_tools(self) -> tuple[dict, ...]:
        return self.listed

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name not in self.results:
            raise RuntimeError(f"unknown MCP tool: {name}")
        return self.results[name]

    def close(self) -> None:
        self.closed = True


def _read_only_tool(name: str = "policy_search") -> dict:
    return {
        "name": name,
        "description": "Search authorized portfolio policy.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
        "_meta": {"mas_finance": {"capability": "document.search", "side_effect": "read_only"}},
    }


class ProgressiveDiscoveryTests(unittest.TestCase):
    def test_reserved_server_name_mcp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved"):
            McpServerConfig(
                name="mcp",
                transport="stdio",
                default_capability="document.search",
                command=sys.executable,
                args=("-c", "pass"),
            )

    def test_search_describe_and_call_do_not_dump_full_catalog_into_planner(self) -> None:
        client = FakeMCPClient((_read_only_tool(),), {"policy_search": _policy_bundle("限额")})
        host = MCPHost(
            (
                McpServerConfig(
                    name="fixture",
                    transport="stdio",
                    default_capability="document.search",
                    command=sys.executable,
                    args=("-c", "pass"),
                ),
            ),
            client_factory=lambda _config: client,
        )
        host.connect()
        try:
            harness = ToolHarness()
            harness.register(financial_calculation_harness_tool())
            for tool in host.tools():
                harness.register(tool)
            for tool in mcp_discovery_tools(host):
                harness.register(tool)
            llm = ScriptedLLM(
                [
                    {
                        "action": "call_tools",
                        "reason": "先检索 MCP 目录再调用政策工具。",
                        "tools": [
                            {
                                "tool_name": "mcp.search_tools",
                                "arguments": {"query": "policy 限额", "limit": 5},
                                "reason": "缩小 MCP 候选。",
                            }
                        ],
                    },
                    {
                        "action": "call_tool",
                        "tool_name": "mcp.call_tool",
                        "arguments": {
                            "name": "fixture.policy_search",
                            "arguments": {"query": "单一发行人限额"},
                        },
                        "reason": "按发现结果调用政策检索。",
                    },
                    {"action": "finish", "reason": "政策限额已有证据。"},
                ]
            )
            harness.register(llm_planning_harness_tool(llm, network_access=False))
            outcome = FinancialResearchAgent(
                harness,
                planner=ModelPlanner(
                    harness,
                    mcp_tool_index=(
                        {
                            "name": "fixture.policy_search",
                            "capability": "document.search",
                            "network_access": False,
                            "description": "Search authorized portfolio policy.",
                            "planner_category": "document",
                        },
                    ),
                ),
                planner_hidden_tool_names=frozenset({"fixture.policy_search"}),
            ).run(
                ResearchRequest(
                    query="根据组合政策说明单一发行人限额",
                    require_documents=True,
                    require_market_data=False,
                    require_regulatory_data=False,
                    run_id="progressive-discovery",
                    max_model_calls=3,
                    max_iterations=4,
                )
            )
            self.assertEqual(outcome.status, "succeeded")
            catalog = {item["name"] for item in json.loads(llm.user_prompts[0])["available_tools"]}
            self.assertIn("mcp.search_tools", catalog)
            self.assertIn("mcp.call_tool", catalog)
            self.assertNotIn("fixture.policy_search", catalog)
            self.assertEqual(
                [item.task.tool_name for item in outcome.state.observations],
                ["mcp.search_tools", "mcp.call_tool"],
            )
            self.assertEqual(client.calls[0][0], "policy_search")
            self.assertTrue(outcome.state.bundle.evidence)
        finally:
            host.close()

    def test_builtin_market_server_is_skipped_without_keys(self) -> None:
        self.assertIsNone(
            builtin_extmarket_server_config(
                alltick_token=None,
                biying_licence=None,
                existing_names=(),
                existing_count=0,
            )
        )
        config = builtin_extmarket_server_config(
            alltick_token="token",
            biying_licence=None,
            existing_names=(),
            existing_count=0,
        )
        assert config is not None
        self.assertEqual(config.name, "extmarket")
        self.assertEqual(config.allowed_tools, ("snapshot", "history"))


class MarketParserTests(unittest.TestCase):
    def test_alltick_and_biying_parsers(self) -> None:
        tick = parse_alltick_tick(
            {"data": {"tick_list": [{"code": "AAPL.US", "price": "189.5", "tick_time": "1710000000000"}]}},
            company="Apple",
            symbol="AAPL.US",
        )
        self.assertEqual(tick["current_price"], 189.5)
        points = parse_alltick_kline(
            {
                "data": {
                    "kline_list": [
                        {
                            "timestamp": "1710000000",
                            "open_price": "1",
                            "close_price": "2",
                            "high_price": "3",
                            "low_price": "0.5",
                            "volume": "10",
                        }
                    ]
                }
            }
        )
        self.assertEqual(points[0]["close"], 2.0)
        realtime = parse_biying_realtime({"price": 10.2, "pe": 8.1}, company="平安", symbol="000001.SZ")
        self.assertEqual(realtime["current_price"], 10.2)
        history = parse_biying_history([{"date": "2026-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.5}])
        self.assertEqual(history[0]["close"], 1.5)
