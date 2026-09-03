from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import httpx
from llm_fixtures import research_service

from mas_finance import MCPHost as PublicMCPHost
from mas_finance import McpServerConfig as PublicMcpServerConfig
from mas_finance.config import AppConfig
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolExecutionError, ToolHarness
from mas_finance.llm import LLMSettings
from mas_finance.mcp import (
    HttpMCPClient,
    MCPHost,
    McpServerConfig,
    _jsonrpc_execution_error,
    _parse_call_payload,
    mcp_discovery_tools,
    parse_mcp_servers_json,
)
from mas_finance.service import FinanceAnalysisService

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "mcp_stdio_fixture.py"


def _config(root: Path) -> AppConfig:
    db_path = root / "finance.db"
    return AppConfig(
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        market_data_provider="offline",
        alphavantage_api_key=None,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        llm=LLMSettings(None, "https://api.deepseek.com", "deepseek-v4-flash", 10),
        allow_network=False,
        conversation_memory_enabled=False,
    )


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


def _read_only_tool(name: str = "policy_search") -> dict:
    return {
        "name": name,
        "description": "Search authorized portfolio policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "filters": {"type": "object"},
                "diversify_documents": {"type": "boolean"},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
        "_meta": {"mas_finance": {"capability": "document.search", "side_effect": "read_only"}},
    }


@dataclass
class FakeMCPClient:
    listed: tuple[dict, ...]
    results: dict[str, dict] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)
    closed: bool = False

    def list_tools(self) -> tuple[dict, ...]:
        return self.listed

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name not in self.results:
            raise RuntimeError(f"unknown MCP tool: {name}")
        return self.results[name]

    def close(self) -> None:
        self.closed = True


class MCPHostTests(unittest.TestCase):
    def test_progressive_mcp_call_validates_remote_nested_schema_before_provider_call(self) -> None:
        client = FakeMCPClient(listed=(_read_only_tool(),), results={"policy_search": _policy_bundle()})
        host = MCPHost(
            (
                McpServerConfig(
                    name="portfolio",
                    transport="stdio",
                    default_capability="document.search",
                    network_access=False,
                    command=sys.executable,
                    args=("-c", "pass"),
                ),
            ),
            client_factory=lambda _config: client,
        )
        host.connect()
        try:
            harness = ToolHarness()
            for tool in mcp_discovery_tools(host):
                harness.register(tool)
            result = harness.invoke(
                "mcp.call_tool",
                {
                    "name": "portfolio.policy_search",
                    "arguments": {"query": "policy", "top_k": "five"},
                },
                ToolContext(
                    run_id="nested-schema",
                    thread_id="thread",
                    policy=ExecutionPolicy(allowed_capabilities=frozenset({"mcp.invoke"})),
                ),
            )
        finally:
            host.close()

        self.assertEqual(result.error_code, "mcp_invalid_arguments")
        self.assertEqual(result.error_details["field"], "top_k")
        self.assertEqual(result.error_details["expected_type"], "integer")
        self.assertEqual(result.error_details["model_action"], "change_arguments")
        self.assertEqual(client.calls, [])

    def test_jsonrpc_invalid_params_preserves_code_data_and_replan_action(self) -> None:
        error = _jsonrpc_execution_error(
            {
                "code": -32602,
                "message": "interval is unsupported",
                "data": {"field": "interval", "allowed_values": ["1d", "1wk"]},
            }
        )
        self.assertEqual(error.code, "mcp_invalid_arguments")
        self.assertEqual(error.details["jsonrpc_code"], -32602)
        self.assertEqual(error.details["data"]["field"], "interval")
        self.assertEqual(error.details["model_action"], "change_arguments")

    def test_structured_tool_error_preserves_retry_guidance(self) -> None:
        with self.assertRaises(ToolExecutionError) as raised:
            _parse_call_payload(
                {
                    "isError": True,
                    "structuredContent": {
                        "error_code": "unknown_symbol",
                        "message": "供应商无法识别代码 BRK.B",
                        "field": "symbol",
                        "received": "BRK.B",
                        "candidates": [{"symbol": "BRK-B", "market": "US"}],
                        "retryable": True,
                    },
                }
            )
        self.assertEqual(raised.exception.code, "unknown_symbol")
        self.assertEqual(raised.exception.details["field"], "symbol")
        self.assertTrue(raised.exception.details["retryable"])
        self.assertEqual(raised.exception.details["model_action"], "change_arguments")

    def test_verified_mcp_arguments_are_stored_outside_personal_memory(self) -> None:
        client = FakeMCPClient(listed=(_read_only_tool("snapshot"),))
        host = MCPHost(
            (
                McpServerConfig(
                    name="provider",
                    transport="stdio",
                    default_capability="market.read",
                    command=sys.executable,
                    args=("-c", "pass"),
                ),
            ),
            client_factory=lambda _config: client,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = FinanceAnalysisService(_config(Path(directory)), mcp_host=host)
            try:
                service._record_tool_usage_memory(
                    "local",
                    "alice",
                    (
                        {
                            "tool_name": "provider.snapshot",
                            "arguments": {"query": "Berkshire BRK-B"},
                            "result_status": "success",
                            "timestamp": "2026-08-27T12:00:00+08:00",
                        },
                    ),
                )
                recalled = service._recall_tool_usage_memory("local", "alice", "Berkshire")
                self.assertEqual(recalled[0]["tool_name"], "provider.snapshot")
                self.assertEqual(recalled[0]["success_count"], 1)
                self.assertEqual(service.list_personal_memories(user_id="alice"), [])
                self.assertEqual(service._recall_tool_usage_memory("default", "bob", "Berkshire"), ())
            finally:
                service.close()

    def test_public_exports_exist(self) -> None:
        self.assertIs(PublicMCPHost, MCPHost)
        self.assertIs(PublicMcpServerConfig, McpServerConfig)

    def test_host_rejects_side_effecting_and_undeclared_tools(self) -> None:
        client = FakeMCPClient(
            listed=(
                _read_only_tool(),
                {
                    "name": "write_note",
                    "description": "Write a note",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                    "annotations": {"readOnlyHint": False, "destructiveHint": True},
                },
                {
                    "name": "mystery",
                    "description": "No read-only declaration",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ),
            results={"policy_search": _policy_bundle()},
        )
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
            names = [tool.spec.name for tool in host.tools()]
            self.assertEqual(names, ["fixture.policy_search"])
            reasons = {item.tool_name: item.reason for item in host.rejections()}
            self.assertIn("write_note", reasons)
            self.assertIn("mystery", reasons)
        finally:
            host.close()
        self.assertTrue(client.closed)

    def test_host_rejects_unknown_capability_even_if_read_only(self) -> None:
        client = FakeMCPClient(
            listed=(
                {
                    **_read_only_tool("orders"),
                    "_meta": {"mas_finance": {"capability": "broker.trade", "side_effect": "read_only"}},
                },
            )
        )
        host = MCPHost(
            (
                McpServerConfig(
                    name="broker",
                    transport="http",
                    default_capability="market.read",
                    url="https://mcp.example.test/mcp",
                ),
            ),
            client_factory=lambda _config: client,
        )
        host.connect()
        try:
            self.assertEqual(host.tools(), ())
            self.assertEqual(host.rejections()[0].reason, "capability is not an allowed evidence capability")
        finally:
            host.close()

    def test_service_uses_filtered_mcp_tool_and_covers_document_requirement(self) -> None:
        client = FakeMCPClient(
            listed=(_read_only_tool(),),
            results={"policy_search": _policy_bundle("组合政策")},
        )
        host = MCPHost(
            (
                McpServerConfig(
                    name="portfolio",
                    transport="http",
                    default_capability="document.search",
                    url="https://mcp.example.test/mcp",
                ),
            ),
            client_factory=lambda _config: client,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(_config(Path(directory)), mcp_host=host)
            try:
                catalog = {item["name"]: item for item in service.describe_tools()}
                self.assertEqual(catalog["portfolio.policy_search"]["availability"], "mcp_connected")
                result = service.analyze(
                    "根据组合政策说明单一发行人限额",
                    export_artifacts=False,
                )["result"]
                self.assertEqual(result["status"], "succeeded")
                self.assertIn(
                    "mcp.call_tool",
                    [item["task"]["tool_name"] for item in result["observations"]],
                )
                self.assertTrue(client.calls)
            finally:
                service.close()

    def test_raw_mcp_json_is_not_accepted_as_evidence(self) -> None:
        client = FakeMCPClient(
            listed=(_read_only_tool(),),
            results={"policy_search": {"text": "not a bundle"}},
        )
        host = MCPHost(
            (
                McpServerConfig(
                    name="portfolio",
                    transport="http",
                    default_capability="document.search",
                    url="https://mcp.example.test/mcp",
                ),
            ),
            client_factory=lambda _config: client,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(_config(Path(directory)), mcp_host=host)
            try:
                result = service.analyze(
                    "根据组合政策说明单一发行人限额",
                    export_artifacts=False,
                )["result"]
                self.assertNotEqual(result["status"], "succeeded")
                self.assertTrue(any(gap["code"] == "tool_internal_error" for gap in result["gaps"]))
            finally:
                service.close()

    def test_reserved_mcp_tool_name_is_rejected_at_service_startup(self) -> None:
        client = FakeMCPClient(listed=(_read_only_tool("snapshot"),))
        host = MCPHost(
            (
                McpServerConfig(
                    name="market",
                    transport="http",
                    default_capability="market.read",
                    url="https://mcp.example.test/mcp",
                ),
            ),
            client_factory=lambda _config: client,
        )
        with (
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory,
            self.assertRaisesRegex(ValueError, "collides"),
        ):
            FinanceAnalysisService(_config(Path(directory)), mcp_host=host)
        self.assertTrue(client.closed)

    def test_stdio_fixture_server_is_filtered_and_callable(self) -> None:
        server = McpServerConfig(
            name="localpolicy",
            transport="stdio",
            default_capability="document.search",
            command=sys.executable,
            args=("-u", str(FIXTURE)),
            env={"PYTHONPATH": str(ROOT / "src")},
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(_config(Path(directory)), mcp_host=MCPHost((server,)))
            try:
                self.assertEqual(
                    [item.spec.name for item in service.mcp_tools],
                    ["localpolicy.policy_search"],
                )
                self.assertTrue(any(item.tool_name == "write_note" for item in service.mcp_host.rejections()))
                result = service.analyze(
                    "根据组合政策说明单一发行人限额",
                    export_artifacts=False,
                )["result"]
                self.assertEqual(result["status"], "succeeded")
                self.assertIn(
                    "mcp.call_tool",
                    [item["task"]["tool_name"] for item in result["observations"]],
                )
            finally:
                service.close()

    def test_http_client_lists_and_calls_json_rpc(self) -> None:
        bundle = _policy_bundle("http")

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            method = payload.get("method")
            if method == "initialize":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "http-fixture", "version": "1"},
                        },
                    },
                    headers={"content-type": "application/json", "mcp-session-id": "sess-1"},
                )
            if method == "notifications/initialized":
                self.assertEqual(request.headers.get("mcp-session-id"), "sess-1")
                return httpx.Response(200, json={}, headers={"content-type": "application/json"})
            if method == "tools/list":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": [_read_only_tool()]}},
                    headers={"content-type": "application/json"},
                )
            if method == "tools/call":
                self.assertEqual(payload["params"]["name"], "policy_search")
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {"structuredContent": bundle, "content": []},
                    },
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(400, json={"error": "unexpected"})

        client = HttpMCPClient(
            McpServerConfig(
                name="httppolicy",
                transport="http",
                default_capability="document.search",
                url="https://mcp.example.test/mcp",
                api_key="secret",
            ),
            transport=httpx.MockTransport(handler),
        )
        host = MCPHost(
            (
                McpServerConfig(
                    name="httppolicy",
                    transport="http",
                    default_capability="document.search",
                    url="https://mcp.example.test/mcp",
                ),
            ),
            client_factory=lambda _config: client,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(_config(Path(directory)), mcp_host=host)
            try:
                result = service.analyze(
                    "根据组合政策说明单一发行人限额",
                    export_artifacts=False,
                )["result"]
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(result["observations"][0]["task"]["tool_name"], "mcp.call_tool")
            finally:
                service.close()

    def test_http_url_must_be_fixed_https_without_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            McpServerConfig(
                name="bad",
                transport="http",
                default_capability="document.search",
                url="http://mcp.example.test/mcp",
            )
        with self.assertRaisesRegex(ValueError, "credential-free"):
            McpServerConfig(
                name="bad",
                transport="http",
                default_capability="document.search",
                url="https://user:pass@mcp.example.test/mcp",
            )

    def test_env_json_parses_stdio_allowlist(self) -> None:
        raw = json.dumps(
            [
                {
                    "name": "localpolicy",
                    "transport": "stdio",
                    "default_capability": "document.search",
                    "command": sys.executable,
                    "args": ["-u", "fixture.py"],
                    "network_access": False,
                }
            ]
        )
        servers = parse_mcp_servers_json(raw)
        self.assertEqual(servers[0].name, "localpolicy")
        self.assertEqual(servers[0].args, ("-u", "fixture.py"))
        with patch.dict(os.environ, {"MAS_MCP_SERVERS": raw}, clear=False):
            config = AppConfig.from_env()
            self.assertEqual(config.mcp_servers[0].name, "localpolicy")
        self.assertEqual(parse_mcp_servers_json(""), ())
        self.assertEqual(parse_mcp_servers_json(None), ())
