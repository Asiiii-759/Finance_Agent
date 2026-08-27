"""本地 stdio MCP server，供 Host/Client 集成测试使用。"""

from __future__ import annotations

import json
import sys
from typing import Any

from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType


def _read_message() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    payload = json.loads(line.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stdio MCP message must be an object")
    return payload


def _write_message(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _policy_result(query: str) -> dict[str, Any]:
    source = SourceRef.create(
        source_type=SourceType.DOCUMENT,
        title="Portfolio policy",
        locator="mcp://fixture/portfolio/policy",
        provider="stdio-mcp-fixture",
    )
    bundle = EvidenceBundle()
    bundle.add_evidence(
        Evidence.create(
            source=source,
            content=f"单一发行人限额为百分之五。检索词：{query}",
        )
    )
    return {"bundle": bundle.to_dict(), "gaps": []}


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "policy_search",
            "description": "Search authorized portfolio policy through a local MCP fixture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "filters": {"type": "object"},
                    "diversify_documents": {"type": "boolean"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "destructiveHint": False},
            "_meta": {"mas_finance": {"capability": "document.search", "side_effect": "read_only"}},
        },
        {
            "name": "write_note",
            "description": "Write a note. This side-effecting tool must be rejected by the host.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
        },
    ]


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            return
        method = str(message.get("method") or "")
        call_id = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "initialize":
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "stdio-mcp-fixture", "version": "1.0.0"},
                    },
                }
            )
            continue
        if method == "notifications/initialized":
            continue
        if method == "tools/list":
            _write_message({"jsonrpc": "2.0", "id": call_id, "result": {"tools": _tools()}})
            continue
        if method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if name != "policy_search":
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "result": {
                            "isError": True,
                            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                        },
                    }
                )
                continue
            payload = _policy_result(str(arguments.get("query") or ""))
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                    },
                }
            )
            continue
        if call_id is not None:
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
