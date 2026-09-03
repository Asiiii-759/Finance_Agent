"""Evidence-first financial research agent package."""

from __future__ import annotations

from typing import Any

__version__ = "2.2.0"
__all__ = [
    "EmbeddingProvider",
    "FinancialResearchAgent",
    "HTTPEmbeddingClient",
    "HTTPJSONRAGClient",
    "MCPHost",
    "McpServerConfig",
    "PDFDocumentParser",
    "PaddleOCRClient",
    "AgentContext",
    "ChatAttachment",
    "ChatTurn",
    "RuntimePolicy",
    "RetrievalSource",
    "create_app",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentContext", "ChatAttachment", "ChatTurn", "FinancialResearchAgent", "RuntimePolicy"}:
        from .agent import AgentContext, ChatAttachment, ChatTurn, RuntimePolicy
        from .graph import FinancialResearchAgent

        return {
            "AgentContext": AgentContext,
            "ChatAttachment": ChatAttachment,
            "ChatTurn": ChatTurn,
            "FinancialResearchAgent": FinancialResearchAgent,
            "RuntimePolicy": RuntimePolicy,
        }[name]
    if name == "create_app":
        from .api.app import create_app

        return create_app
    if name in {"HTTPJSONRAGClient", "RetrievalSource"}:
        from .retrieval import HTTPJSONRAGClient, RetrievalSource

        return {
            "HTTPJSONRAGClient": HTTPJSONRAGClient,
            "RetrievalSource": RetrievalSource,
        }[name]
    if name in {"MCPHost", "McpServerConfig"}:
        from .mcp import MCPHost, McpServerConfig

        return {"MCPHost": MCPHost, "McpServerConfig": McpServerConfig}[name]
    if name == "PaddleOCRClient":
        from .ocr import PaddleOCRClient

        return PaddleOCRClient
    if name == "PDFDocumentParser":
        from .documents import PDFDocumentParser

        return PDFDocumentParser
    if name in {"EmbeddingProvider", "HTTPEmbeddingClient"}:
        from .embeddings import EmbeddingProvider, HTTPEmbeddingClient

        return {
            "EmbeddingProvider": EmbeddingProvider,
            "HTTPEmbeddingClient": HTTPEmbeddingClient,
        }[name]
    raise AttributeError(name)
