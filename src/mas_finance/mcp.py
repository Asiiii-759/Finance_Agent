"""MCP Host 与 Client：部署 allowlist 连接外部 server，再收成 Harness 工具。

协议仍是 initialize / tools/list / tools/call。Host 决定哪些工具能进目录。
模型默认只看到 builtins + 渐进发现元工具；具体 MCP schema 不进 LLM `tools` 字段，也不整表塞进 planner prompt。
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from .contracts import EvidenceBundle
from .harness import (
    SideEffect,
    Tool,
    ToolArgumentContract,
    ToolContext,
    ToolExecutionError,
    ToolResultKind,
    ToolSpec,
    function_tool,
)
from .rate_limit import RateLimit, RateLimiter

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "mas-finance"
CLIENT_VERSION = "2.2.0"
MAX_MCP_SERVERS = 4
MAX_MCP_TOOLS = 20
MAX_TOOLS_PER_SERVER = 32
DISCOVERY_TOOL_NAMES = frozenset({"mcp.search_tools", "mcp.describe_tool", "mcp.call_tool"})
MAX_DISCOVERY_MATCHES = 20
_ALLOWED_CAPABILITIES = frozenset(
    {
        "document.search",
        "market.read",
        "regulatory.read",
        "macro.read",
        "calculation",
        "knowledge.read",
        "web.search",
    }
)
_PLANNER_CATEGORIES = frozenset(
    {
        "document",
        "market",
        "market_history",
        "regulatory",
        "filings",
        "macro",
        "calculation",
        "knowledge",
        "web",
    }
)
_CAPABILITY_TO_PLANNER = {
    "document.search": "document",
    "market.read": "market",
    "regulatory.read": "regulatory",
    "macro.read": "macro",
    "calculation": "calculation",
    "knowledge.read": "knowledge",
    "web.search": "web",
}
_NAME_RE = re.compile(r"[a-z][a-z0-9_.-]{0,99}")
_ARG_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SERVER_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ENV_KEY_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_STDIO_ENV_PASSTHROUGH = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "HOME",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "PYTHONPATH",
    "PYTHONHOME",
    "LANG",
    "LC_ALL",
)


class MCPClient(Protocol):
    def list_tools(self) -> tuple[Mapping[str, Any], ...]: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class McpServerConfig:
    """部署期 MCP server allowlist 条目。endpoint/command 不能来自模型。"""

    name: str
    transport: str
    default_capability: str
    network_access: bool = False
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    allowed_tools: tuple[str, ...] = ()
    timeout_seconds: float = 15.0
    max_response_bytes: int = 1_000_000
    max_calls_per_minute: int = 6

    def __post_init__(self) -> None:
        if self.name == "mcp":
            raise ValueError("MCP server name 'mcp' is reserved")
        if not _SERVER_NAME_RE.fullmatch(self.name):
            raise ValueError("MCP server name is invalid")
        if self.transport not in {"stdio", "http"}:
            raise ValueError("MCP transport must be stdio or http")
        if self.default_capability not in _ALLOWED_CAPABILITIES:
            raise ValueError("MCP default_capability is not an allowed evidence capability")
        if not 0.5 <= self.timeout_seconds <= 120:
            raise ValueError("MCP timeout must be between 0.5 and 120 seconds")
        if not 1_024 <= self.max_response_bytes <= 5_000_000:
            raise ValueError("MCP response limit is outside the supported range")
        if not 1 <= self.max_calls_per_minute <= 60:
            raise ValueError("MCP max_calls_per_minute must be between 1 and 60")
        if len(self.allowed_tools) > MAX_TOOLS_PER_SERVER:
            raise ValueError("MCP allowed_tools exceeds the per-server limit")
        if any(not isinstance(item, str) or not item or len(item) > 100 for item in self.allowed_tools):
            raise ValueError("MCP allowed_tools contains an invalid remote tool name")
        normalized_env = _validated_env(self.env)
        object.__setattr__(self, "env", MappingProxyType(normalized_env))
        if self.api_key is not None and (
            not self.api_key
            or len(self.api_key) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in self.api_key)
        ):
            raise ValueError("MCP API key is invalid")
        if self.transport == "stdio":
            if self.url:
                raise ValueError("stdio MCP servers cannot declare a URL")
            _validate_stdio_command(self.command, self.args)
        else:
            if self.command or self.args:
                raise ValueError("HTTP MCP servers cannot declare a stdio command")
            _validate_http_url(self.url)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> McpServerConfig:
        if not isinstance(value, Mapping):
            raise ValueError("MCP server config must be an object")
        raw_args = value.get("args") or ()
        raw_allowed = value.get("allowed_tools") or ()
        raw_env = value.get("env") or {}
        if not isinstance(raw_args, (list, tuple)):
            raise ValueError("MCP args must be a list")
        if not isinstance(raw_allowed, (list, tuple)):
            raise ValueError("MCP allowed_tools must be a list")
        if not isinstance(raw_env, Mapping):
            raise ValueError("MCP env must be an object")
        return cls(
            name=str(value.get("name") or ""),
            transport=str(value.get("transport") or ""),
            default_capability=str(value.get("default_capability") or ""),
            network_access=bool(value.get("network_access", False)),
            command=str(value["command"]) if value.get("command") is not None else None,
            args=tuple(str(item) for item in raw_args),
            env={str(key): str(item) for key, item in raw_env.items()},
            url=str(value["url"]) if value.get("url") is not None else None,
            api_key=str(value["api_key"]) if value.get("api_key") is not None else None,
            allowed_tools=tuple(str(item) for item in raw_allowed),
            timeout_seconds=float(value.get("timeout_seconds", 15.0)),
            max_response_bytes=int(value.get("max_response_bytes", 1_000_000)),
            max_calls_per_minute=int(value.get("max_calls_per_minute", 6)),
        )


@dataclass(frozen=True)
class McpRejection:
    server_name: str
    tool_name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"server_name": self.server_name, "tool_name": self.tool_name, "reason": self.reason}


@dataclass(frozen=True)
class _AcceptedRemoteTool:
    server_name: str
    remote_name: str
    local_name: str
    description: str
    capability: str
    planner_category: str
    network_access: bool
    arguments: ToolArgumentContract
    input_schema: Mapping[str, Any]


def parse_mcp_servers_json(raw: str | None) -> tuple[McpServerConfig, ...]:
    if raw is None or not str(raw).strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MAS_MCP_SERVERS must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("MAS_MCP_SERVERS must be a JSON array")
    if len(payload) > MAX_MCP_SERVERS:
        raise ValueError(f"at most {MAX_MCP_SERVERS} MCP servers are supported")
    servers = tuple(McpServerConfig.from_dict(item) for item in payload)
    names = [item.name for item in servers]
    if len(names) != len(set(names)):
        raise ValueError("MCP server names must be unique")
    return servers


class StdioMCPClient:
    """Newline-delimited JSON-RPC client for a local stdio MCP server."""

    def __init__(self, config: McpServerConfig) -> None:
        if config.transport != "stdio" or not config.command:
            raise ValueError("StdioMCPClient requires a stdio server command")
        self.config = config
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, Queue[Mapping[str, Any]]] = {}
        self._closed = False
        self._stderr_tail = ""
        env = _stdio_env(config)
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": env,
            "bufsize": 0,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        self._process = subprocess.Popen(  # noqa: S603
            [config.command, *config.args],
            **popen_kwargs,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._err_reader.start()
        try:
            self._handshake()
        except Exception:
            self.close()
            raise

    def list_tools(self) -> tuple[Mapping[str, Any], ...]:
        result = self._request("tools/list", {})
        return _parse_tool_list(result)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        return _parse_call_payload(result)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        process = self._process
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=2)

    def _handshake(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        self._notify("notifications/initialized")

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        waiter: Queue[Mapping[str, Any]] = Queue(maxsize=1)
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP stdio client is closed")
            call_id = self._next_id
            self._next_id += 1
            self._pending[call_id] = waiter
            payload = {"jsonrpc": "2.0", "id": call_id, "method": method, "params": dict(params)}
            self._write(payload)
        try:
            message = waiter.get(timeout=self.config.timeout_seconds)
        except Empty as exc:
            with self._lock:
                self._pending.pop(call_id, None)
            detail = self._stderr_tail.strip()
            suffix = f" stderr={detail}" if detail else ""
            raise TimeoutError(f"MCP stdio request timed out: {method}{suffix}") from exc
        if "error" in message:
            raise RuntimeError(_jsonrpc_error_message(message["error"]))
        result = message.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"MCP {method} result must be an object")
        return result

    def _notify(self, method: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP stdio client is closed")
            self._write({"jsonrpc": "2.0", "method": method})

    def _write(self, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if b"\n" in encoded:
            raise ValueError("MCP stdio payload cannot contain newlines")
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("MCP stdio stdin is unavailable")
        stdin.write(encoded + b"\n")
        stdin.flush()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        while True:
            line = stdout.readline()
            if not line:
                return
            if len(line) > self.config.max_response_bytes:
                continue
            try:
                message = json.loads(line.decode("utf-8"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(message, dict) or "id" not in message:
                continue
            call_id = message.get("id")
            if not isinstance(call_id, int):
                continue
            with self._lock:
                waiter = self._pending.pop(call_id, None)
            if waiter is not None:
                waiter.put(message)

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        chunks: list[bytes] = []
        total = 0
        while True:
            piece = stderr.read(256)
            if not piece:
                break
            total += len(piece)
            if total <= 2_000:
                chunks.append(piece)
        self._stderr_tail = b"".join(chunks).decode("utf-8", errors="replace")


class HttpMCPClient:
    """Bounded JSON-RPC HTTP client for a deployment-fixed MCP endpoint.

    First slice accepts JSON responses only, not SSE Streamable HTTP.
    """

    def __init__(
        self,
        config: McpServerConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config.transport != "http" or not config.url:
            raise ValueError("HttpMCPClient requires a fixed HTTPS MCP URL")
        self.config = config
        self.transport = transport
        self._lock = threading.Lock()
        self._next_id = 1
        self._session_id: str | None = None
        self._initialized = False

    def list_tools(self) -> tuple[Mapping[str, Any], ...]:
        self._ensure_initialized()
        result = self._request("tools/list", {})
        return _parse_tool_list(result)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure_initialized()
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        return _parse_call_payload(result)

    def close(self) -> None:
        return

    def _ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        self._notify("notifications/initialized")
        with self._lock:
            self._initialized = True

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            call_id = self._next_id
            self._next_id += 1
            session_id = self._session_id
        message = self._post(
            {"jsonrpc": "2.0", "id": call_id, "method": method, "params": dict(params)},
            session_id=session_id,
        )
        if "error" in message:
            raise RuntimeError(_jsonrpc_error_message(message["error"]))
        result = message.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"MCP {method} result must be an object")
        return result

    def _notify(self, method: str) -> None:
        with self._lock:
            session_id = self._session_id
        try:
            self._post({"jsonrpc": "2.0", "method": method}, session_id=session_id)
        except (httpx.HTTPError, ValueError, RuntimeError):
            # 通知失败不阻断后续 list/call；握手仍以 initialize 成功为准。
            return

    def _post(self, payload: Mapping[str, Any], *, session_id: str | None) -> Mapping[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        endpoint = self.config.url
        if not endpoint:
            raise ValueError("HTTP MCP URL is missing")
        with (
            httpx.Client(
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client,
            client.stream("POST", endpoint, headers=headers, json=dict(payload)) as response,
        ):
            response.raise_for_status()
            incoming_session = response.headers.get("mcp-session-id")
            if incoming_session:
                with self._lock:
                    self._session_id = incoming_session.strip()[:200] or self._session_id
            content_type = response.headers.get("content-type", "").casefold()
            if content_type and "json" not in content_type:
                raise ValueError("MCP HTTP response must use a JSON content type")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.config.max_response_bytes:
                    raise ValueError("MCP HTTP response exceeds the byte limit")
                chunks.append(chunk)
        if not chunks:
            return {}
        try:
            data = json.loads(b"".join(chunks), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("MCP HTTP response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("MCP HTTP response must be a JSON object")
        return data


class MCPHost:
    """Agent 侧 Host：连接 allowlist 中的 server，过滤后交给 ToolHarness。"""

    def __init__(
        self,
        servers: Sequence[McpServerConfig] = (),
        *,
        client_factory: Callable[[McpServerConfig], MCPClient] | None = None,
    ) -> None:
        if len(servers) > MAX_MCP_SERVERS:
            raise ValueError(f"at most {MAX_MCP_SERVERS} MCP servers are supported")
        names = [item.name for item in servers]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")
        self.servers = tuple(servers)
        self._client_factory = client_factory or create_mcp_client
        self._clients: list[MCPClient] = []
        self._tools: list[Tool] = []
        self._tools_by_name: dict[str, Tool] = {}
        self._planner_categories: dict[str, str] = {}
        self._input_schemas: dict[str, Mapping[str, Any]] = {}
        self._rejections: list[McpRejection] = []
        self._connected = False
        self._limiter = RateLimiter()

    def connect(self) -> None:
        if self._connected:
            return
        try:
            for server in self.servers:
                inner = self._client_factory(server)
                client: MCPClient = _RateLimitedMCPClient(
                    inner,
                    self._limiter,
                    key=f"mcp:{server.name}",
                    limit=RateLimit(server.max_calls_per_minute, 60.0),
                    timeout_seconds=server.timeout_seconds,
                )
                self._clients.append(client)
                listed = client.list_tools()
                if len(listed) > MAX_TOOLS_PER_SERVER:
                    listed = listed[:MAX_TOOLS_PER_SERVER]
                for remote in listed:
                    accepted = _accept_remote_tool(server, remote)
                    if isinstance(accepted, McpRejection):
                        self._rejections.append(accepted)
                        continue
                    if len(self._tools) >= MAX_MCP_TOOLS:
                        self._rejections.append(
                            McpRejection(server.name, accepted.remote_name, "mcp tool budget exhausted")
                        )
                        continue
                    if any(tool.spec.name == accepted.local_name for tool in self._tools):
                        self._rejections.append(
                            McpRejection(server.name, accepted.remote_name, "duplicate local tool name")
                        )
                        continue
                    self._tools.append(_bind_harness_tool(client, accepted))
                    self._tools_by_name[accepted.local_name] = self._tools[-1]
                    self._planner_categories[accepted.local_name] = accepted.planner_category
                    self._input_schemas[accepted.local_name] = accepted.input_schema
        except Exception:
            self.close()
            raise
        self._connected = True

    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools)

    def rejections(self) -> tuple[McpRejection, ...]:
        return tuple(self._rejections)

    def planner_category_for(self, tool_name: str) -> str | None:
        return self._planner_categories.get(tool_name)

    def tool_by_name(self, name: str) -> Tool | None:
        return self._tools_by_name.get(name)

    def catalog_index(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": tool.spec.name,
                "capability": tool.spec.capability,
                "network_access": tool.spec.network_access,
                "description": tool.spec.description,
                "planner_category": self._planner_categories.get(tool.spec.name),
                "input_contract": tool.spec.arguments.to_dict(),
                "input_schema": dict(self._input_schemas[tool.spec.name]),
            }
            for tool in self._tools
        )

    def close(self) -> None:
        while self._clients:
            client = self._clients.pop()
            try:
                client.close()
            except Exception:
                continue
        self._connected = False


def create_mcp_client(config: McpServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return HttpMCPClient(config)


class _RateLimitedMCPClient:
    """对 tools/call 做每分钟上限；list/handshake 不计额度。"""

    def __init__(
        self,
        inner: MCPClient,
        limiter: RateLimiter,
        *,
        key: str,
        limit: RateLimit,
        timeout_seconds: float,
    ) -> None:
        self._inner = inner
        self._limiter = limiter
        self._key = key
        self._limit = limit
        self._timeout_seconds = timeout_seconds

    def list_tools(self) -> tuple[Mapping[str, Any], ...]:
        return self._inner.list_tools()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        self._limiter.acquire(self._key, self._limit, timeout_seconds=self._timeout_seconds)
        return self._inner.call_tool(name, arguments)

    def close(self) -> None:
        self._inner.close()


def mcp_discovery_tools(host: MCPHost) -> tuple[Tool, ...]:
    """渐进发现元工具：短索引 + describe + 受控 invoke，不把全部 MCP schema 塞进 planner。"""

    if not host.tools():
        return ()
    any_network = any(tool.spec.network_access for tool in host.tools())

    def search(arguments: Mapping[str, Any], _context: ToolContext) -> dict[str, Any]:
        query = str(arguments["query"]).strip()
        if not query or len(query) > 500:
            raise ValueError("MCP search query is invalid")
        limit = int(arguments.get("limit") or 8)
        if not 1 <= limit <= MAX_DISCOVERY_MATCHES:
            raise ValueError(f"MCP search limit must be between 1 and {MAX_DISCOVERY_MATCHES}")
        matches = _rank_mcp_catalog(host.catalog_index(), query)[:limit]
        return {"query": query, "matches": matches}

    def describe(arguments: Mapping[str, Any], _context: ToolContext) -> dict[str, Any]:
        name = str(arguments["name"])
        for item in host.catalog_index():
            if item["name"] == name:
                return dict(item)
        raise ValueError("unknown or unauthorized MCP tool")

    def call(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
        name = str(arguments["name"])
        inner_arguments = arguments.get("arguments")
        if not isinstance(inner_arguments, Mapping):
            raise ValueError("MCP call arguments must be an object")
        tool = host.tool_by_name(name)
        if tool is None:
            raise ValueError("unknown or unauthorized MCP tool")
        if tool.spec.network_access and not context.policy.allow_network:
            raise RuntimeError("network access is not allowed for this run")
        tool.spec.arguments.validate(inner_arguments)
        payload = tool(inner_arguments, context)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("bundle"), Mapping):
            raise ValueError("MCP tool did not return an evidence bundle")
        EvidenceBundle.from_dict(payload["bundle"])
        return {"bundle": dict(payload["bundle"]), "gaps": list(payload.get("gaps") or [])}

    return (
        function_tool(
            ToolSpec(
                name="mcp.search_tools",
                description="按关键词检索已连接 MCP 工具的短描述，不返回完整参数契约。",
                capability="mcp.discover",
                side_effect=SideEffect.READ_ONLY,
                network_access=False,
                timeout_seconds=5.0,
                result_kind=ToolResultKind.CATALOG,
                arguments=ToolArgumentContract(
                    required=frozenset({"query"}),
                    optional=frozenset({"limit"}),
                ),
            ),
            search,
        ),
        function_tool(
            ToolSpec(
                name="mcp.describe_tool",
                description="读取单个已授权 MCP 工具的完整输入契约，执行前按需调用。",
                capability="mcp.discover",
                side_effect=SideEffect.READ_ONLY,
                network_access=False,
                timeout_seconds=5.0,
                result_kind=ToolResultKind.CATALOG,
                arguments=ToolArgumentContract(required=frozenset({"name"})),
            ),
            describe,
        ),
        function_tool(
            ToolSpec(
                name="mcp.call_tool",
                description="调用 mcp_tool_index 中的本地 MCP 工具名，并要求返回 EvidenceBundle。",
                capability="mcp.invoke",
                side_effect=SideEffect.READ_ONLY,
                network_access=any_network,
                timeout_seconds=35.0,
                result_kind=ToolResultKind.EVIDENCE_BUNDLE,
                arguments=ToolArgumentContract(
                    required=frozenset({"name", "arguments"}),
                ),
            ),
            call,
        ),
    )


def builtin_extmarket_server_config(
    *,
    alltick_token: str | None,
    biying_licence: str | None,
    existing_names: Sequence[str],
    existing_count: int,
    enable_yfinance: bool = False,
    enable_akshare: bool = False,
    max_calls_per_minute: int = 6,
) -> McpServerConfig | None:
    """在配置了 AllTick 或必盈许可时，自动挂上本地行情 MCP server。"""

    token = (alltick_token or "").strip()
    licence = (biying_licence or "").strip()
    if not token and not licence:
        return None
    if "extmarket" in existing_names or existing_count >= MAX_MCP_SERVERS:
        return None
    src_dir = str(Path(__file__).resolve().parents[1])
    env = {"PYTHONPATH": src_dir}
    if token:
        env["ALLTICK_TOKEN"] = token
    if licence:
        env["BIYING_LICENCE"] = licence
    if enable_yfinance:
        env["MAS_ENABLE_YFINANCE"] = "true"
    if enable_akshare:
        env["MAS_ENABLE_AKSHARE"] = "true"
    env["MAS_MARKET_MAX_CALLS_PER_MINUTE"] = str(max_calls_per_minute)
    return McpServerConfig(
        name="extmarket",
        transport="stdio",
        default_capability="market.read",
        network_access=True,
        command=sys.executable,
        args=("-u", "-m", "mas_finance.mcp_servers.market"),
        env=env,
        timeout_seconds=25.0,
        max_calls_per_minute=max_calls_per_minute,
        allowed_tools=("snapshot", "history"),
    )


def _accept_remote_tool(server: McpServerConfig, remote: Mapping[str, Any]) -> _AcceptedRemoteTool | McpRejection:
    remote_name = str(remote.get("name") or "")
    if not remote_name:
        return McpRejection(server.name, "unknown", "missing tool name")
    if server.allowed_tools and remote_name not in server.allowed_tools:
        return McpRejection(server.name, remote_name, "tool is not in the server allowlist")
    annotations_raw = remote.get("annotations")
    annotations: Mapping[str, Any] = annotations_raw if isinstance(annotations_raw, Mapping) else {}
    meta_raw = remote.get("_meta")
    meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}
    finance_raw = meta.get("mas_finance")
    finance_meta: Mapping[str, Any] = finance_raw if isinstance(finance_raw, Mapping) else {}
    if annotations.get("destructiveHint") is True:
        return McpRejection(server.name, remote_name, "destructive tools are not allowed")
    if annotations.get("readOnlyHint") is False:
        return McpRejection(server.name, remote_name, "non-read-only tools are not allowed")
    side_effect = str(finance_meta.get("side_effect") or "")
    if side_effect and side_effect != SideEffect.READ_ONLY.value:
        return McpRejection(server.name, remote_name, "only read_only MCP tools are accepted")
    if annotations.get("readOnlyHint") is not True and side_effect != SideEffect.READ_ONLY.value:
        return McpRejection(server.name, remote_name, "MCP tool must declare read-only")
    capability = str(finance_meta.get("capability") or server.default_capability)
    if capability not in _ALLOWED_CAPABILITIES:
        return McpRejection(server.name, remote_name, "capability is not an allowed evidence capability")
    planner_category = str(finance_meta.get("planner_category") or _CAPABILITY_TO_PLANNER[capability])
    if planner_category not in _PLANNER_CATEGORIES:
        return McpRejection(server.name, remote_name, "planner_category is invalid")
    local_name = f"{server.name}.{_sanitize_remote_name(remote_name)}"
    if not _NAME_RE.fullmatch(local_name):
        return McpRejection(server.name, remote_name, "local tool name is invalid")
    description = str(remote.get("description") or f"MCP tool {remote_name}").strip()
    if not description:
        return McpRejection(server.name, remote_name, "tool description is empty")
    if len(description) > 1_000:
        description = description[:997] + "..."
    try:
        input_schema = remote.get("inputSchema") or {"type": "object", "properties": {}}
        arguments = _contract_from_json_schema(input_schema)
    except ValueError as exc:
        return McpRejection(server.name, remote_name, str(exc))
    return _AcceptedRemoteTool(
        server_name=server.name,
        remote_name=remote_name,
        local_name=local_name,
        description=description,
        capability=capability,
        planner_category=planner_category,
        network_access=server.network_access,
        arguments=arguments,
        input_schema=dict(input_schema),
    )


def _bind_harness_tool(client: MCPClient, accepted: _AcceptedRemoteTool) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: ToolContext) -> dict[str, Any]:
        payload = client.call_tool(accepted.remote_name, arguments)
        bundle = payload.get("bundle")
        if not isinstance(bundle, Mapping):
            raise ValueError("MCP tool did not return an evidence bundle")
        EvidenceBundle.from_dict(bundle)
        gaps = payload.get("gaps", [])
        if gaps is None:
            gaps = []
        if not isinstance(gaps, list) or any(not isinstance(item, Mapping) for item in gaps):
            raise ValueError("MCP tool returned malformed gaps")
        return {"bundle": dict(bundle), "gaps": [dict(item) for item in gaps]}

    return function_tool(
        ToolSpec(
            name=accepted.local_name,
            description=accepted.description,
            capability=accepted.capability,
            side_effect=SideEffect.READ_ONLY,
            network_access=accepted.network_access,
            timeout_seconds=30.0,
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=accepted.arguments,
        ),
        invoke,
    )


def _contract_from_json_schema(schema: Any) -> ToolArgumentContract:
    if not isinstance(schema, Mapping):
        raise ValueError("inputSchema must be an object")
    schema_type = schema.get("type", "object")
    if schema_type != "object":
        raise ValueError("inputSchema type must be object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        raise ValueError("inputSchema properties must be an object")
    required_raw = schema.get("required") or []
    if not isinstance(required_raw, list) or any(not isinstance(item, str) for item in required_raw):
        raise ValueError("inputSchema required must be a list of strings")
    keys = [str(key) for key in properties]
    required = [str(item) for item in required_raw]
    if any(key not in keys for key in required):
        raise ValueError("inputSchema required key is missing from properties")
    if any(not _ARG_KEY_RE.fullmatch(key) for key in (*keys, *required)):
        raise ValueError("inputSchema property names must be snake_case")
    extra = schema.get("additionalProperties", False)
    if extra not in {True, False}:
        extra = False
    return ToolArgumentContract(
        required=frozenset(required),
        optional=frozenset(keys) - frozenset(required),
        allow_extra=bool(extra),
    )


def _parse_tool_list(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    tools = result.get("tools")
    if not isinstance(tools, list):
        raise ValueError("MCP tools/list must return a tools array")
    parsed: list[Mapping[str, Any]] = []
    for item in tools:
        if not isinstance(item, Mapping):
            raise ValueError("MCP tool descriptor must be an object")
        parsed.append(item)
    return tuple(parsed)


def _parse_call_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("isError") is True:
        details = _call_error_details(result)
        raise ToolExecutionError(
            str(details.pop("error_code", "mcp_tool_error")),
            str(details.pop("message", _call_error_message(result))),
            details=details,
        )
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        return structured
    content = result.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("MCP tool result must include structuredContent or text content")
    texts = [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    if len(texts) != 1 or not texts[0].strip():
        raise ValueError("MCP tool result must contain exactly one JSON text part")
    try:
        payload = json.loads(texts[0], parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP tool text result is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("MCP tool JSON result must be an object")
    return payload


def _call_error_message(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        texts = [
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if texts and texts[0]:
            return texts[0][:1_000]
    return "MCP tool returned isError=true"


def _call_error_details(result: Mapping[str, Any]) -> dict[str, Any]:
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        payload = dict(structured)
    else:
        message = _call_error_message(result)
        try:
            decoded = json.loads(message)
        except json.JSONDecodeError:
            decoded = None
        payload = dict(decoded) if isinstance(decoded, Mapping) else {"message": message}
    raw_code = str(payload.get("error_code") or "mcp_tool_error").casefold()
    code = re.sub(r"[^a-z0-9_]", "_", raw_code).strip("_")
    if not code or not code[0].isalpha():
        code = "mcp_tool_error"
    payload["error_code"] = code[:64]
    payload["message"] = str(payload.get("message") or _call_error_message(result))[:1_000]
    return payload


def _jsonrpc_error_message(error: Any) -> str:
    if isinstance(error, Mapping):
        message = str(error.get("message") or "MCP JSON-RPC error")
        return message[:1_000]
    return "MCP JSON-RPC error"


def _sanitize_remote_name(name: str) -> str:
    lowered = name.strip().replace("/", ".").replace(" ", "_").casefold()
    cleaned = re.sub(r"[^a-z0-9_.-]", "_", lowered).strip("._-")
    return cleaned or "tool"


def _validate_stdio_command(command: str | None, args: Sequence[str]) -> None:
    if not command or len(command) > 512 or any(ord(character) < 32 for character in command):
        raise ValueError("MCP stdio command is invalid")
    if len(args) > 32:
        raise ValueError("MCP stdio args exceed the supported limit")
    if any(not isinstance(item, str) or not item or len(item) > 4_096 or "\x00" in item for item in args):
        raise ValueError("MCP stdio args are invalid")


def _validate_http_url(url: str | None) -> None:
    if not url:
        raise ValueError("HTTP MCP servers require a URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MCP HTTP URL must be a fixed credential-free HTTPS URL")


def _validated_env(env: Mapping[str, str]) -> dict[str, str]:
    if len(env) > 32:
        raise ValueError("MCP env exceeds the supported limit")
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not _ENV_KEY_RE.fullmatch(str(key)) or not isinstance(value, str):
            raise ValueError("MCP env contains an invalid key or value")
        if len(value) > 4_096 or any(ord(character) < 32 for character in value if character not in {"\t"}):
            raise ValueError("MCP env value is invalid")
        normalized[str(key)] = value
    return normalized


def _stdio_env(config: McpServerConfig) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key in _STDIO_ENV_PASSTHROUGH:
        value = os.environ.get(key)
        if value:
            merged[key] = value
    merged.update(dict(config.env))
    merged["PYTHONUNBUFFERED"] = "1"
    merged["PYTHONIOENCODING"] = "utf-8"
    return merged


def _rank_mcp_catalog(catalog: Sequence[Mapping[str, Any]], query: str) -> list[dict[str, Any]]:
    tokens = _discovery_tokens(query)
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in catalog:
        haystack = " ".join(
            str(item.get(key) or "")
            for key in ("name", "description", "capability", "planner_category")
        )
        score = len(tokens.intersection(_discovery_tokens(haystack)))
        name = str(item.get("name") or "")
        if query.strip().casefold() in name.casefold() or query.strip() in str(item.get("description") or ""):
            score += 3
        if score <= 0:
            continue
        ranked.append(
            (
                score,
                {
                    "name": name,
                    "capability": item.get("capability"),
                    "network_access": bool(item.get("network_access")),
                    "planner_category": item.get("planner_category"),
                    "description": item.get("description"),
                },
            )
        )
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1]["name"])))
    return [item for _, item in ranked]


def _discovery_tokens(text: str) -> set[str]:
    lowered = text.casefold()
    latin = set(re.findall(r"[a-z0-9_.-]{2,}", lowered))
    cjk = set(re.findall(r"[\u4e00-\u9fff]+", text))
    return latin | cjk


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
