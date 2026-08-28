from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import ResearchRequest
from .atomic_facts import AtomicFactExtractor, LLMAtomicFactExtractor
from .config import AppConfig
from .contracts import stable_id
from .conversation import LLMConversationSummarizer
from .corpus import CorpusDocument, InMemoryCorpus
from .documents import PDFDocumentParser, parse_pdf_document
from .embeddings import EmbeddingProvider, HTTPEmbeddingClient
from .formula import formula_harness_tool
from .graph import FinancialResearchAgent
from .harness import SideEffect, Tool, ToolHarness, ToolResultKind
from .llm import BaseLLMClient, build_llm_client
from .macro import FREDClient, FREDEvidenceAdapter, fred_series_harness_tool
from .market import (
    MarketEvidenceAdapter,
    MarketHistoryEvidenceAdapter,
    market_data_harness_tool,
    market_history_harness_tool,
)
from .market_data import DEFAULT_TICKER_MAP, MarketDataClient
from .mcp import MCPHost, builtin_extmarket_server_config, mcp_discovery_tools
from .memory_consolidation import (
    LLMLongTermMemoryExtractor,
    LongTermMemoryCandidate,
    LongTermMemoryExtractor,
)
from .memory_store import (
    ConversationEvent,
    ConversationEventKind,
    ConversationSummarizer,
    MemoryNamespace,
    PersonalMemory,
    PersonalMemoryKind,
    SQLiteMemoryStore,
    TokenCounter,
    build_conversation_window,
)
from .metrics import describe_metric_operations, financial_calculation_harness_tool
from .ocr import PaddleOCRClient
from .personal_knowledge import PersonalKnowledgeClient, SQLitePersonalKnowledgeBase
from .planning import ModelPlanner, llm_planning_harness_tool
from .rate_limit import RateLimit, RateLimiter
from .reporting import export_run_artifacts
from .retrieval import RetrievalEvidenceAdapter, RetrievalSource, retrieval_harness_tool
from .sec import (
    SECCompanyFactsAdapter,
    SECCompanyFactsClient,
    SECRecentFilingsAdapter,
    sec_company_facts_harness_tool,
    sec_recent_filings_harness_tool,
)
from .security import safe_child, safe_upload_name
from .skill_learning import LearnedSkill, LLMSkillExtractor, SkillExtractor, skill_run_context
from .synthesis import EvidenceBoundLLMSynthesizer, llm_synthesis_harness_tool
from .task_frame import LLMTaskInterpreter, llm_task_frame_harness_tool
from .web_search import BochaWebSearchClient, BraveWebSearchClient, WebSearchEvidenceAdapter, web_search_harness_tool

if TYPE_CHECKING:
    from .database import JobRepository

_MAX_PERSONAL_MEMORY_CHARACTERS = 100_000


class FinanceAnalysisService:
    def __init__(
        self,
        config: AppConfig,
        *,
        retrieval_sources: Sequence[RetrievalSource] = (),
        evidence_tools: Sequence[Tool] = (),
        pdf_document_parser: PDFDocumentParser | None = None,
        pdf_parser_network_access: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
        conversation_summarizer: ConversationSummarizer | None = None,
        conversation_token_counter: TokenCounter | None = None,
        long_term_memory_extractor: LongTermMemoryExtractor | None = None,
        skill_extractor: SkillExtractor | None = None,
        atomic_fact_extractor: AtomicFactExtractor | None = None,
        mcp_host: MCPHost | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.retrieval_sources = tuple(retrieval_sources)
        self.evidence_tools = tuple(evidence_tools)
        self.pdf_document_parser = pdf_document_parser or (
            PaddleOCRClient(
                config.paddleocr_access_token,
                job_url=config.paddleocr_job_url,
                model=config.paddleocr_model,
                max_file_bytes=config.max_upload_bytes,
                max_pages=config.max_pdf_pages,
            )
            if config.paddleocr_access_token
            else None
        )
        self.pdf_parser_network_access = pdf_parser_network_access
        self.conversation_summarizer = conversation_summarizer
        self.conversation_token_counter = conversation_token_counter
        self.long_term_memory_extractor = long_term_memory_extractor
        self.skill_extractor = skill_extractor
        self.atomic_fact_extractor = atomic_fact_extractor
        if embedding_provider is not None and config.embedding_endpoint:
            raise ValueError("inject either an embedding provider or configured embedding endpoint, not both")
        self.embedding_provider = embedding_provider or (
            HTTPEmbeddingClient(
                config.embedding_endpoint,
                config.embedding_model,
                api_key=config.embedding_api_key,
                timeout_seconds=config.embedding_timeout_seconds,
            )
            if config.embedding_endpoint and config.embedding_model
            else None
        )
        if len(self.retrieval_sources) > 4:
            raise ValueError("at most four deployment retrieval sources are supported")
        if len(self.evidence_tools) > 20:
            raise ValueError("at most twenty deployment evidence tools are supported")
        retrieval_names = [item.name for item in self.retrieval_sources]
        if len(retrieval_names) != len(set(retrieval_names)):
            raise ValueError("retrieval source names must be unique")
        if set(retrieval_names).intersection({"corpus.search", "corpus.hybrid_search"}):
            raise ValueError("corpus search names are reserved for request/session-scoped uploads")
        if set(retrieval_names).intersection({"personal.search", "personal.hybrid_search"}):
            raise ValueError("personal search names are reserved for the personal knowledge base")
        extension_names = [tool.spec.name for tool in self.evidence_tools]
        if len(extension_names) != len(set(extension_names)):
            raise ValueError("deployment evidence tool names must be unique")
        reserved_names = {
            "finance.calculate",
            "finance.formula",
            "corpus.search",
            "corpus.hybrid_search",
            "personal.search",
            "personal.hybrid_search",
            "market.snapshot",
            "market.history",
            "sec.company_facts",
            "sec.recent_filings",
            "macro.fred_series",
            "web.search",
            "llm.task_frame",
            "llm.plan",
            "llm.synthesize",
            "mcp.search_tools",
            "mcp.describe_tool",
            "mcp.call_tool",
        }
        if set(extension_names).intersection(reserved_names | set(retrieval_names)):
            raise ValueError("deployment evidence tool name collides with a built-in tool")
        allowed_extension_capabilities = {
            "document.search",
            "market.read",
            "regulatory.read",
            "macro.read",
            "calculation",
            "knowledge.read",
            "web.search",
        }
        if any(
            tool.spec.side_effect != SideEffect.READ_ONLY
            or tool.spec.result_kind != ToolResultKind.EVIDENCE_BUNDLE
            or tool.spec.capability not in allowed_extension_capabilities
            for tool in self.evidence_tools
        ):
            raise ValueError("deployment evidence tools must be read-only canonical evidence tools")
        extra_mcp = builtin_extmarket_server_config(
            alltick_token=config.alltick_token,
            biying_licence=config.biying_licence,
            existing_names=[item.name for item in config.mcp_servers],
            existing_count=len(config.mcp_servers),
            enable_yfinance=config.enable_yfinance_fallback,
            enable_akshare=config.enable_akshare_fallback,
            max_calls_per_minute=config.market_max_calls_per_minute,
        )
        mcp_servers = (*config.mcp_servers, extra_mcp) if extra_mcp is not None else config.mcp_servers
        self.mcp_host = mcp_host or MCPHost(mcp_servers)
        self._rate_limiter = RateLimiter()
        try:
            self.mcp_host.connect()
            self.mcp_tools = self.mcp_host.tools()
            if len(self.mcp_tools) > 20:
                raise ValueError("at most twenty MCP tools are supported")
            mcp_names = [tool.spec.name for tool in self.mcp_tools]
            if len(mcp_names) != len(set(mcp_names)):
                raise ValueError("MCP tool names must be unique")
            if set(mcp_names).intersection(reserved_names | set(retrieval_names) | set(extension_names)):
                raise ValueError("MCP tool name collides with a built-in or injected tool")
            if any(
                tool.spec.side_effect != SideEffect.READ_ONLY
                or tool.spec.result_kind != ToolResultKind.EVIDENCE_BUNDLE
                or tool.spec.capability not in allowed_extension_capabilities
                for tool in self.mcp_tools
            ):
                raise ValueError("MCP tools must be read-only canonical evidence tools")
        except Exception:
            self.mcp_host.close()
            raise
        self._repository: JobRepository | None = None
        self._memory_store: SQLiteMemoryStore | None = None
        self._personal_knowledge_store: SQLitePersonalKnowledgeBase | None = None
        # Session documents are deliberately process-local: unlike thread
        # metadata they must not silently become a persistent knowledge base.
        self._session_documents: dict[tuple[str, str, str], tuple[datetime, tuple[dict, ...]]] = {}
        self._session_documents_lock = RLock()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph_checkpoint_connection: sqlite3.Connection | None = None
        self._graph_checkpointer: SqliteSaver | None = None

    def close(self) -> None:
        try:
            self.mcp_host.close()
        finally:
            self._close_graph_checkpointer()

    def _open_graph_checkpointer(self) -> SqliteSaver:
        if self._graph_checkpointer is None:
            self._graph_checkpoint_connection = sqlite3.connect(
                self.config.db_path,
                timeout=15,
                check_same_thread=False,
            )
            self._graph_checkpointer = SqliteSaver(self._graph_checkpoint_connection)
        return self._graph_checkpointer

    def _close_graph_checkpointer(self) -> None:
        if self._graph_checkpoint_connection is not None:
            self._graph_checkpoint_connection.close()
        self._graph_checkpoint_connection = None
        self._graph_checkpointer = None

    @property
    def repository(self) -> JobRepository:
        if self._repository is None:
            from .database import JobRepository

            self._repository = JobRepository(self.config.database_url, db_path=self.config.db_path)
        return self._repository

    @property
    def memory_store(self) -> SQLiteMemoryStore:
        if self._memory_store is None:
            self._memory_store = SQLiteMemoryStore(self.config.db_path)
        return self._memory_store

    def _llm_configured(self) -> bool:
        return self.llm_client is not None or bool(self.config.llm.api_key)

    def _require_llm_client(self) -> BaseLLMClient:
        client = self.llm_client or build_llm_client(self.config.llm)
        if client is None:
            raise RuntimeError("an LLM configuration is required for financial research")
        return client

    @property
    def personal_knowledge_store(self) -> SQLitePersonalKnowledgeBase:
        if self._personal_knowledge_store is None:
            self._personal_knowledge_store = SQLitePersonalKnowledgeBase(
                self.config.db_path,
                max_documents_per_user=self.config.max_personal_knowledge_documents,
            )
        return self._personal_knowledge_store

    def describe_tools(self) -> list[dict]:
        """Return the public capability catalog without exposing callables or credentials."""

        def interface(required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
            return {
                "input_contract": {
                    "required": list(required),
                    "optional": list(optional),
                    "allow_extra": False,
                },
                "result_contract": "evidence_bundle",
                "visibility": "agent",
            }

        market_network = self.config.market_data_provider not in {"offline", "disabled", "none"}
        market_availability = "when_entity_present" if market_network else "disabled_by_provider"
        tools = [
            {
                "name": "finance.calculate",
                "capability": "calculation",
                "description": "Allowlisted deterministic financial formulas with input provenance.",
                "network_access": False,
                "availability": "always",
                "operation_contract": describe_metric_operations(),
                **interface(("requests",)),
            },
            {
                "name": "finance.formula",
                "capability": "calculation",
                "description": "Evaluate a bounded declarative formula without executing model-generated code.",
                "network_access": False,
                "availability": "always",
                **interface(("expression", "inputs"), ("label", "unit", "entity", "period")),
            },
            {
                "name": "corpus.search",
                "capability": "document.search",
                "description": "Run deterministic BM25 search over current request/session PDFs.",
                "network_access": False,
                "availability": "when_pdf_uploaded",
                "search_mode": "lexical",
                **interface(("query",), ("top_k", "filters", "diversify_documents")),
            },
            {
                "name": "corpus.hybrid_search",
                "capability": "document.search",
                "description": "Run BM25 and embedding retrieval over current/session PDFs, fused with RRF.",
                "network_access": bool(self.embedding_provider and self.embedding_provider.network_access),
                "availability": "when_pdf_uploaded" if self.embedding_provider else "missing_embedding_provider",
                "search_mode": "hybrid_rrf",
                "embedding_model": self.embedding_provider.model_name if self.embedding_provider else None,
                **interface(("query",), ("top_k", "filters", "diversify_documents")),
            },
            {
                "name": "personal.search",
                "capability": "document.search",
                "description": "Run deterministic BM25 search over the user's persistent document library.",
                "network_access": False,
                "availability": (
                    "when_personal_documents_exist" if self.config.personal_knowledge_enabled else "disabled"
                ),
                "provider": "personal_knowledge",
                "search_mode": "lexical",
                **interface(("query",), ("top_k", "filters", "diversify_documents")),
            },
            {
                "name": "personal.hybrid_search",
                "capability": "document.search",
                "description": "Run BM25 and embedding retrieval over the personal library, fused with RRF.",
                "network_access": bool(self.embedding_provider and self.embedding_provider.network_access),
                "availability": (
                    "when_personal_documents_exist"
                    if self.config.personal_knowledge_enabled and self.embedding_provider
                    else "missing_embedding_provider"
                    if self.config.personal_knowledge_enabled
                    else "disabled"
                ),
                "provider": "personal_knowledge",
                "search_mode": "hybrid_rrf",
                "embedding_model": self.embedding_provider.model_name if self.embedding_provider else None,
                **interface(("query",), ("top_k", "filters", "diversify_documents")),
            },
            {
                "name": "market.snapshot",
                "capability": "market.read",
                "description": "Read current price, valuation and market snapshot fields.",
                "network_access": market_network,
                "availability": market_availability,
                "provider": self.config.market_data_provider,
                "support_tier": (
                    "experimental_non_contractual"
                    if self.config.market_data_provider == "yahoo"
                    else "configured_contract"
                    if self.config.market_data_provider == "alphavantage"
                    else "disabled"
                ),
                **interface(("company",), ("symbol", "required_fields")),
            },
            {
                "name": "market.history",
                "capability": "market.read",
                "description": (
                    "Read price history with an explicit adjusted/raw basis and derive return, volatility and drawdown."
                ),
                "network_access": market_network,
                "availability": market_availability,
                "provider": self.config.market_data_provider,
                "support_tier": (
                    "experimental_non_contractual"
                    if self.config.market_data_provider == "yahoo"
                    else "configured_contract"
                    if self.config.market_data_provider == "alphavantage"
                    else "disabled"
                ),
                **interface(("company",), ("symbol", "range", "interval")),
            },
            {
                "name": "sec.company_facts",
                "capability": "regulatory.read",
                "description": "Read SEC XBRL company facts and derive aligned financial ratios.",
                "network_access": True,
                "availability": "configured" if self.config.sec_user_agent else "missing_sec_user_agent",
                **interface(("company", "symbol"), ("required_fields",)),
            },
            {
                "name": "sec.recent_filings",
                "capability": "regulatory.read",
                "description": "Read recent SEC forms and primary-document locators.",
                "network_access": True,
                "availability": "configured" if self.config.sec_user_agent else "missing_sec_user_agent",
                **interface(("company", "symbol"), ("forms", "limit")),
            },
            {
                "name": "macro.fred_series",
                "capability": "macro.read",
                "description": "Read official FRED macroeconomic observations and metadata.",
                "network_access": True,
                "availability": "configured" if self.config.fred_api_key else "missing_fred_api_key",
                **interface(
                    ("series_id",),
                    ("observation_start", "observation_end", "limit"),
                ),
            },
            {
                "name": "web.search",
                "capability": "web.search",
                "description": "Model-directed open-web search returning cited result snippets.",
                "network_access": True,
                "availability": (
                    "configured"
                    if self.config.bocha_search_api_key or self.config.brave_search_api_key
                    else "missing_web_search_api_key"
                ),
                "provider": "bocha" if self.config.bocha_search_api_key else "brave",
                "support_tier": (
                    "configured_contract"
                    if self.config.bocha_search_api_key or self.config.brave_search_api_key
                    else "disabled"
                ),
                **interface(("query",), ("count", "freshness", "domains")),
            },
            {
                "name": "llm.task_frame",
                "capability": "model.generate",
                "description": "Interpret the current request and visible conversation memory into a TaskFrame.",
                "network_access": self._llm_configured(),
                "availability": "required" if self._llm_configured() else "missing_llm_configuration",
                "input_contract": {
                    "required": ["system_prompt", "user_prompt"],
                    "optional": ["temperature", "max_tokens"],
                    "allow_extra": False,
                },
                "result_contract": "model_response",
                "visibility": "internal_task_frame_only",
            },
            {
                "name": "llm.plan",
                "capability": "model.generate",
                "description": "Choose one next authorized evidence-gathering action.",
                "network_access": self._llm_configured(),
                "availability": "required" if self._llm_configured() else "missing_llm_configuration",
                "input_contract": {
                    "required": ["system_prompt", "user_prompt"],
                    "optional": ["temperature", "max_tokens"],
                    "allow_extra": False,
                },
                "result_contract": "model_response",
                "visibility": "internal_planning_only",
            },
            {
                "name": "llm.synthesize",
                "capability": "model.generate",
                "description": "Generate claims; cited evidence must pass literal quote validation.",
                "network_access": self._llm_configured(),
                "availability": "required" if self._llm_configured() else "missing_llm_configuration",
                "input_contract": {
                    "required": ["system_prompt", "user_prompt"],
                    "optional": ["temperature", "max_tokens"],
                    "allow_extra": False,
                },
                "result_contract": "model_response",
                "visibility": "internal_synthesis_only",
            },
        ]
        tools.extend(
            {
                "name": source.name,
                "capability": "document.search",
                "description": source.description,
                "network_access": source.network_access,
                "availability": "configured",
                "provider": source.provider,
                "support_tier": "deployment_injected_contract",
                "server_filters_enforced": bool(source.fixed_filters),
                **interface(
                    ("query",),
                    ("top_k", "filters", "search_mode", "rerank", "diversify_documents"),
                ),
            }
            for source in self.retrieval_sources
        )
        tools.extend(
            {
                "name": tool.spec.name,
                "capability": tool.spec.capability,
                "description": tool.spec.description,
                "network_access": tool.spec.network_access,
                "availability": "deployment_injected",
                "support_tier": "deployment_injected_contract",
                "input_contract": tool.spec.arguments.to_dict(),
                "result_contract": tool.spec.result_kind.value,
                "visibility": "agent",
            }
            for tool in self.evidence_tools
        )
        tools.extend(
            {
                "name": tool.spec.name,
                "capability": tool.spec.capability,
                "description": tool.spec.description,
                "network_access": tool.spec.network_access,
                "availability": "mcp_connected",
                "support_tier": "mcp_host_filtered",
                "input_contract": tool.spec.arguments.to_dict(),
                "result_contract": tool.spec.result_kind.value,
                "visibility": "mcp_index",
            }
            for tool in self.mcp_tools
        )
        tools.extend(
            {
                "name": tool.spec.name,
                "capability": tool.spec.capability,
                "description": tool.spec.description,
                "network_access": tool.spec.network_access,
                "availability": "mcp_connected",
                "support_tier": "mcp_progressive_discovery",
                "input_contract": tool.spec.arguments.to_dict(),
                "result_contract": tool.spec.result_kind.value,
                "visibility": "planner_meta",
            }
            for tool in mcp_discovery_tools(self.mcp_host)
        )
        return tools

    def analyze(
        self,
        query: str,
        thread_id: str | None = None,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        entities: list[str] | None = None,
        symbols: dict[str, str] | None = None,
        allow_network: bool | None = None,
        macro_series: list[str] | None = None,
        calculations: list[dict] | None = None,
        require_documents: bool | None = None,
        require_market_data: bool | None = None,
        require_market_history: bool | None = None,
        require_regulatory_data: bool | None = None,
        market_history_range: str = "1y",
        market_history_interval: str = "1d",
        use_session_documents: bool = False,
        use_personal_memory: bool = True,
        use_personal_knowledge: bool = True,
        retain_documents_for_session: bool = False,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        run_id: str | None = None,
        resume: bool = False,
    ) -> dict:
        if thread_id is not None:
            _validate_thread_id(thread_id)
        if use_session_documents and thread_id is None:
            raise ValueError("thread_id is required when using session documents")
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        actual_run_id = run_id or f"run-{uuid4().hex[:12]}"
        conversation_namespace = self._conversation_namespace(tenant_id, user_id, actual_thread_id)
        self.memory_store.upsert_conversation_thread(
            conversation_namespace.tenant_id,
            conversation_namespace.user_id,
            actual_thread_id,
            title=query.strip()[:200],
            run_id=actual_run_id,
            status="running",
        )
        self.memory_store.append_run_log(
            conversation_namespace,
            run_id=actual_run_id,
            event_type="run.started",
            level="info",
            message="研究运行已启动。",
            details={"resume": resume, "requested_network": allow_network is True},
        )
        # Network access requires both deployment authorization and explicit
        # per-request consent. None never means implicit consent.
        network_allowed = self.config.allow_network and allow_network is True
        llm_client = self._require_llm_client()
        conversation_summarizer = self.conversation_summarizer or LLMConversationSummarizer(llm_client)
        memory_extractor = self.long_term_memory_extractor or (
            LLMLongTermMemoryExtractor(llm_client)
            if self.config.automatic_memory_consolidation_enabled
            else None
        )
        skill_extractor = self.skill_extractor or (
            LLMSkillExtractor(llm_client) if self.config.automatic_skill_learning_enabled else None
        )
        atomic_fact_extractor = self.atomic_fact_extractor or LLMAtomicFactExtractor(llm_client)
        thread_context = self._load_conversation_context(
            tenant_id,
            user_id,
            actual_thread_id,
            summarizer=conversation_summarizer,
        )
        self.memory_store.append_run_log(
            conversation_namespace,
            run_id=actual_run_id,
            event_type="context.loaded",
            level="info",
            message="会话、个人记忆与 Skill 索引已装载。",
            details={
                "atomic_fact_count": len(thread_context.get("atomic_facts") or ()),
                "recent_event_count": len(thread_context.get("recent_events") or ()),
            },
        )
        personal_context = (
            (*self._user_profile_context(), *self._personal_memory_context(tenant_id, user_id))
            if use_personal_memory and self.config.personal_memory_enabled
            else ()
        )
        tool_usage_context = self._recall_tool_usage_memory(tenant_id, user_id, query)
        learned_skills = self.list_learned_skills(tenant_id=tenant_id, user_id=user_id)
        skill_index = tuple(
            {
                "skill_id": item["skill_id"],
                "name": item["name"],
                "description": item["description"],
                "applicability": item["applicability"],
            }
            for item in learned_skills
        )
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        harness.register(formula_harness_tool())
        for tool in self.evidence_tools:
            harness.register(tool)
        for tool in self.mcp_tools:
            harness.register(tool)
        for tool in mcp_discovery_tools(self.mcp_host):
            harness.register(tool)
        harness.register(llm_task_frame_harness_tool(llm_client, network_access=True))
        harness.register(llm_planning_harness_tool(llm_client, network_access=True))
        harness.register(llm_synthesis_harness_tool(llm_client, network_access=True))
        if self.config.bocha_search_api_key:
            harness.register(
                web_search_harness_tool(
                    WebSearchEvidenceAdapter(
                        BochaWebSearchClient(
                            self.config.bocha_search_api_key,
                            rate_limiter=self._rate_limiter,
                            rate_limit=RateLimit(self.config.bocha_max_calls_per_minute),
                        )
                    )
                )
            )
        elif self.config.brave_search_api_key:
            harness.register(
                web_search_harness_tool(
                    WebSearchEvidenceAdapter(
                        BraveWebSearchClient(
                            self.config.brave_search_api_key,
                            rate_limiter=self._rate_limiter,
                            rate_limit=RateLimit(self.config.brave_max_calls_per_minute),
                        )
                    )
                )
            )

        explicit_document_entities = _normalized_entities(entities or [])
        if document_paths and self.pdf_document_parser is None:
            raise ValueError("a PaddleOCR or MCP PDF document parser is required for PDF analysis")
        if document_paths and self.pdf_parser_network_access and not network_allowed:
            raise ValueError("PDF parsing requires server and request network authorization")
        current_document_contexts = [
            parse_pdf_document(
                Path(path),
                include_pages=True,
                max_pages=self.config.max_pdf_pages,
                max_file_bytes=self.config.max_upload_bytes,
                max_text_characters=self.config.max_pdf_text_characters,
                display_name=self._upload_display_name(Path(path)),
                document_parser=self.pdf_document_parser,
            )
            for path in (document_paths or [])
        ]
        if retain_documents_for_session and not current_document_contexts:
            raise ValueError("at least one uploaded document is required for session retention")
        if retain_documents_for_session:
            self._retain_session_documents(
                tenant_id,
                user_id,
                actual_thread_id,
                current_document_contexts,
            )
        session_document_contexts = (
            self._load_session_documents(tenant_id, user_id, actual_thread_id) if use_session_documents else []
        )
        merged_documents = {
            str(document["document_id"]): {**document, "lifecycle": "session"} for document in session_document_contexts
        }
        merged_documents.update(
            {
                str(document["document_id"]): {
                    **document,
                    "lifecycle": ("session_retained" if retain_documents_for_session else "request"),
                }
                for document in current_document_contexts
            }
        )
        document_contexts = list(merged_documents.values())
        document_tool_names: list[str] = []
        document_tool_names.extend(
            tool.spec.name for tool in self.evidence_tools if tool.spec.capability == "document.search"
        )
        document_tool_names.extend(self._mcp_planner_names("document"))
        if document_contexts:
            corpus = InMemoryCorpus(embedding_provider=self.embedding_provider)
            ingested_chunks = 0
            for document in document_contexts:
                detected = document.get("detected_companies") or []
                base_metadata = {
                    "file_name": document["filename"],
                    "document_title": document["filename"],
                    "document_id": document["document_id"],
                    "page_count": document["page_count"],
                }
                if len(detected) == 1:
                    base_metadata["company"] = detected[0]
                elif not detected and len(explicit_document_entities) == 1:
                    # An explicit single entity is a user assertion about the
                    # uploaded document's scope.  It is safer and more useful
                    # than guessing arbitrary companies from an allowlist.
                    base_metadata["company"] = explicit_document_entities[0]
                for page in document.get("pages") or ():
                    ingested_chunks += corpus.ingest(
                        CorpusDocument.create(
                            title=f"{document['filename']}#page={page['page_number']}",
                            text=page["text"],
                            metadata={
                                **base_metadata,
                                "source_page": page["page_number"],
                                "span_basis": "page",
                                "extraction_method": page["extraction_method"],
                                "page_text_characters": page["text_characters"],
                            },
                        )
                    )
            if not ingested_chunks:
                raise ValueError("PDF document parser returned no extractable text")
            corpus_adapter = RetrievalEvidenceAdapter(corpus)
            harness.register(
                retrieval_harness_tool(
                    corpus_adapter,
                    fixed_search_mode="lexical",
                    description=(
                        "对当前请求/会话 PDF 执行确定性 BM25 搜索。适用于精确词项、名称、标识符，以及 embedding "
                        "搜索不可用时。"
                    ),
                )
            )
            if self.embedding_provider is not None:
                harness.register(
                    retrieval_harness_tool(
                        corpus_adapter,
                        name="corpus.hybrid_search",
                        network_access=self.embedding_provider.network_access,
                        fixed_search_mode="hybrid",
                        description=(
                            "对当前请求/会话 PDF 执行 BM25 与语义 embedding 检索，并用 RRF 融合排名。优先用于改写、"
                            "同义词和跨语言查询。"
                        ),
                    )
                )
                if not self.embedding_provider.network_access or network_allowed:
                    document_tool_names.extend(("corpus.hybrid_search", "corpus.search"))
                else:
                    document_tool_names.extend(("corpus.search", "corpus.hybrid_search"))
            else:
                document_tool_names.append("corpus.search")
        for source in self.retrieval_sources:
            harness.register(source.build_tool())
            document_tool_names.append(source.name)
        personal_tenant, personal_user = _personal_principal_ids(tenant_id, user_id)
        personal_documents = (
            self.personal_knowledge_store.list_documents(personal_tenant, personal_user)
            if use_personal_knowledge and self.config.personal_knowledge_enabled
            else []
        )
        if use_personal_knowledge and self.config.personal_knowledge_enabled and personal_documents:
            personal_client = PersonalKnowledgeClient(
                self.personal_knowledge_store,
                personal_tenant,
                personal_user,
                embedding_provider=self.embedding_provider,
            )
            personal_adapter = RetrievalEvidenceAdapter(
                personal_client,
                provider="personal_knowledge",
            )
            harness.register(
                retrieval_harness_tool(
                    personal_adapter,
                    name="personal.search",
                    fixed_search_mode="lexical",
                    description=(
                        "对用户持久金融文档执行确定性 BM25 搜索。只有在明确跨文档比较或综合时，"
                        "才设置 diversify_documents=true。"
                    ),
                )
            )
            if self.embedding_provider is not None:
                harness.register(
                    retrieval_harness_tool(
                        personal_adapter,
                        name="personal.hybrid_search",
                        network_access=self.embedding_provider.network_access,
                        fixed_search_mode="hybrid",
                        description=(
                            "对用户持久文档执行 BM25 与语义 embedding 检索，并用 RRF 融合排名。优先用于改写、"
                            "同义词和跨语言查询。"
                        ),
                    )
                )
                if not self.embedding_provider.network_access or network_allowed:
                    document_tool_names.extend(("personal.hybrid_search", "personal.search"))
                else:
                    document_tool_names.extend(("personal.search", "personal.hybrid_search"))
            else:
                document_tool_names.append("personal.search")

        # Explicit API entities remain request parameters. Natural-language entity
        # extraction and reference resolution belong to the LLM TaskFrame.
        requested_entities = _normalized_entities(entities or [])
        request_thread_context = dict(thread_context)
        market_client = MarketDataClient(
            provider=self.config.market_data_provider,
            alphavantage_api_key=self.config.alphavantage_api_key,
            rate_limiter=self._rate_limiter,
            rate_limit=RateLimit(self.config.market_max_calls_per_minute),
        )
        harness.register(
            market_data_harness_tool(
                MarketEvidenceAdapter(market_client),
                network_access=self.config.market_data_provider not in {"offline", "disabled", "none"},
            )
        )
        harness.register(
            market_history_harness_tool(
                MarketHistoryEvidenceAdapter(market_client),
                network_access=self.config.market_data_provider not in {"offline", "disabled", "none"},
            )
        )

        resolved_symbols: dict[str, str] = {}
        for entity in requested_entities:
            candidate = (symbols or {}).get(entity) or DEFAULT_TICKER_MAP.get(entity)
            if candidate:
                resolved_symbols[entity] = str(candidate).strip()
            elif re.fullmatch(r"[A-Za-z0-9.^=_:-]{1,32}", entity):
                resolved_symbols[entity] = entity
        if self.config.sec_user_agent:
            sec_client = SECCompanyFactsClient(self.config.sec_user_agent)
            harness.register(sec_company_facts_harness_tool(SECCompanyFactsAdapter(sec_client)))
            harness.register(sec_recent_filings_harness_tool(SECRecentFilingsAdapter(sec_client)))
        if self.config.fred_api_key:
            harness.register(
                fred_series_harness_tool(
                    FREDEvidenceAdapter(
                        FREDClient(
                            self.config.fred_api_key,
                            base_url=self.config.fred_base_url,
                            rate_limiter=self._rate_limiter,
                            rate_limit=RateLimit(self.config.fred_max_calls_per_minute),
                        )
                    )
                )
            )

        document_research_required = (
            bool(document_contexts)
            or require_documents is True
            or (bool(document_tool_names) and require_documents is None and _requests_document_research(query))
        )
        request = ResearchRequest(
            query=query,
            entities=requested_entities,
            symbols=resolved_symbols,
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            allow_network=network_allowed,
            max_iterations=6,
            max_model_calls=8,
            max_model_input_tokens=self.config.model_input_token_budget,
            max_model_output_tokens=self.config.model_output_token_budget,
            max_parallel_tool_calls=4,
            require_documents=document_research_required,
            require_market_data=require_market_data,
            require_market_history=require_market_history,
            require_regulatory_data=require_regulatory_data,
            market_history_range=market_history_range,
            market_history_interval=market_history_interval,
            macro_series=tuple(macro_series or ()),
            calculations=tuple(dict(item) for item in (calculations or ())),
            thread_context=request_thread_context,
            personal_context=personal_context,
            skill_index=skill_index,
            available_document_count=len(document_contexts) + len(personal_documents),
        )
        if self.config.conversation_memory_enabled:
            self.memory_store.append_conversation_event(
                self._conversation_namespace(tenant_id, user_id, actual_thread_id),
                event_id=stable_id("event", {"run_id": actual_run_id, "kind": "user"}),
                kind=ConversationEventKind.USER_MESSAGE,
                content=query,
                run_id=actual_run_id,
                entities=requested_entities,
                payload={"entity_symbols": resolved_symbols},
            )
        agent_failure: Exception | None = None
        try:
            outcome = FinancialResearchAgent(
                harness,
                planner=ModelPlanner(
                    harness,
                    max_evidence_chars=self.config.planning_evidence_characters,
                    mcp_tool_index=self._mcp_tool_index(),
                    tool_usage_context=tool_usage_context,
                    learned_skills=learned_skills,
                ),
                synthesizer=EvidenceBoundLLMSynthesizer(
                    llm_client,
                    harness=harness,
                    max_evidence_chars=self.config.synthesis_evidence_characters,
                    max_output_tokens=self.config.synthesis_output_tokens,
                ),
                checkpointer=self._open_graph_checkpointer(),
                task_interpreter=LLMTaskInterpreter(harness),
                planner_hidden_tool_names=frozenset(tool.spec.name for tool in self.mcp_tools),
            ).run(request, resume=resume)
        except Exception as exc:
            agent_failure = exc
            raise
        finally:
            self._close_graph_checkpointer()
            for audit in harness.audit_events(actual_run_id):
                self.memory_store.append_audit_event(conversation_namespace, audit)
                if self.config.conversation_memory_enabled:
                    self.memory_store.append_conversation_event(
                        conversation_namespace,
                        event_id=stable_id("event", {"run_id": actual_run_id, "call_id": audit["call_id"]}),
                        kind=ConversationEventKind.TOOL_EVENT,
                        content=f"{audit['tool_name']}: {audit['result_status']}",
                        occurred_at=str(audit["timestamp"]),
                        run_id=actual_run_id,
                        entities=requested_entities,
                        payload={
                            key: audit.get(key)
                            for key in (
                                "tool_name",
                                "capability",
                                "result_status",
                                "attempts",
                                "network_attempts",
                                "error_code",
                            )
                        },
                    )
                self.memory_store.append_run_log(
                    conversation_namespace,
                    run_id=actual_run_id,
                    event_type="tool.completed",
                    level="info" if audit["result_status"] == "success" else "warning",
                    message=f"工具 {audit['tool_name']} 调用结束。",
                    occurred_at=str(audit["timestamp"]),
                    details={
                        key: audit.get(key)
                        for key in (
                            "call_id",
                            "tool_name",
                            "capability",
                            "result_status",
                            "attempts",
                            "network_attempts",
                            "model_input_tokens",
                            "model_output_tokens",
                            "duration_ms",
                            "error_code",
                            "error_message",
                            "result_summary",
                        )
                    },
                )
            if agent_failure is not None:
                self.memory_store.append_run_log(
                    conversation_namespace,
                    run_id=actual_run_id,
                    event_type="run.failed",
                    level="error",
                    message="研究运行在 Agent 执行阶段失败。",
                    details={"phase": "agent_execution", "error_type": type(agent_failure).__name__},
                )
        result = outcome.to_dict()
        self.memory_store.record_run_usage(
            conversation_namespace,
            actual_run_id,
            result.get("budget_usage") or {},
        )
        assistant_reply = _assistant_reply(result)
        self.memory_store.append_run_log(
            conversation_namespace,
            run_id=actual_run_id,
            event_type="run.completed",
            level="info" if result["status"] == "succeeded" else "warning",
            message="Agent 已生成研究终态。",
            details={
                "status": result["status"],
                "stop_reason": result.get("stop_reason"),
                "claim_count": len(result.get("claims") or ()),
                "source_count": len(result.get("sources") or ()),
                "unresolved_gap_codes": [
                    str(item.get("code") or "data_gap")
                    for item in result.get("gaps") or ()
                    if not item.get("resolved", False)
                ][:20],
                "budget": result.get("budget"),
            },
        )
        if self.config.conversation_memory_enabled:
            namespace = self._conversation_namespace(tenant_id, user_id, actual_thread_id)
            self.memory_store.put_conversation_run(
                namespace,
                run_id=actual_run_id,
                status=str(result["status"]),
                stop_reason=str(result.get("stop_reason") or "unknown"),
                assistant_reply=assistant_reply,
                result=result,
            )
            self.memory_store.upsert_conversation_thread(
                namespace.tenant_id,
                namespace.user_id,
                actual_thread_id,
                title=query.strip()[:200],
                run_id=actual_run_id,
                status=str(result["status"]),
            )
            self.memory_store.append_conversation_event(
                namespace,
                event_id=stable_id("event", {"run_id": actual_run_id, "kind": "assistant"}),
                kind=ConversationEventKind.ASSISTANT_MESSAGE,
                content=assistant_reply,
                run_id=actual_run_id,
                entities=requested_entities,
                payload={
                    "status": result["status"],
                    "claim_count": len(result.get("claims") or ()),
                    "source_count": len(result.get("sources") or ()),
                    "gap_codes": [
                        str(item.get("code") or "data_gap")
                        for item in result.get("gaps") or ()
                        if not item.get("resolved", False)
                    ][:20],
                },
            )
            run_events = tuple(
                event
                for event in self.memory_store.list_conversation_events(namespace)
                if event.run_id == actual_run_id and event.kind is not ConversationEventKind.ATOMIC_FACT
            )
            try:
                atomic_facts = atomic_fact_extractor.extract(run_events)
                for fact in atomic_facts:
                    self.memory_store.append_conversation_event(
                        namespace,
                        event_id=stable_id(
                            "fact",
                            {"run_id": actual_run_id, "text": fact.text, "sources": fact.source_event_ids},
                        ),
                        kind=ConversationEventKind.ATOMIC_FACT,
                        content=fact.text,
                        run_id=actual_run_id,
                        entities=fact.entities,
                        payload={"source_event_ids": list(fact.source_event_ids), "status": fact.status},
                    )
                self.memory_store.append_run_log(
                    conversation_namespace,
                    run_id=actual_run_id,
                    event_type="memory.atomic_facts_completed",
                    level="info",
                    message="原子事实已写入独立账本。",
                    details={"fact_count": len(atomic_facts)},
                )
            except Exception as exc:
                self.memory_store.append_run_log(
                    conversation_namespace,
                    run_id=actual_run_id,
                    event_type="memory.atomic_facts_failed",
                    level="error",
                    message="原子事实提取或写入失败。",
                    details={"phase": "atomic_fact_persistence", "error_type": type(exc).__name__},
                )
            try:
                build_conversation_window(
                    self.memory_store,
                    namespace,
                    max_tokens=self.config.conversation_context_tokens,
                    recent_tokens=self.config.conversation_recent_tokens,
                    summarizer=conversation_summarizer,
                    token_counter=self.conversation_token_counter,
                )
            except Exception as exc:
                self.memory_store.append_run_log(
                    conversation_namespace,
                    run_id=actual_run_id,
                    event_type="memory.compaction_failed",
                    level="error",
                    message="对话上下文投影或压缩失败。",
                    details={"phase": "conversation_compaction", "error_type": type(exc).__name__},
                )
            if memory_extractor is not None and use_personal_memory and self.config.personal_memory_enabled:
                run_events = tuple(
                    event
                    for event in self.memory_store.list_conversation_events(namespace)
                    if event.run_id == actual_run_id and event.kind is ConversationEventKind.USER_MESSAGE
                )
                try:
                    self._consolidate_long_term_memory(
                        tenant_id,
                        user_id,
                        actual_thread_id,
                        actual_run_id,
                        run_events,
                        memory_extractor,
                    )
                except Exception as exc:
                    self.memory_store.append_run_log(
                        conversation_namespace,
                        run_id=actual_run_id,
                        event_type="memory.long_term_failed",
                        level="error",
                        message="长期记忆候选处理失败。",
                        details={"phase": "long_term_memory", "error_type": type(exc).__name__},
                    )
                else:
                    self.memory_store.append_run_log(
                        conversation_namespace,
                        run_id=actual_run_id,
                        event_type="memory.long_term_completed",
                        level="info",
                        message="长期记忆候选已处理。",
                        details={},
                    )
        try:
            self._record_tool_usage_memory(
                tenant_id,
                user_id,
                harness.audit_events(actual_run_id),
            )
        except Exception as exc:
            self.memory_store.append_run_log(
                conversation_namespace,
                run_id=actual_run_id,
                event_type="tool_usage.learning_failed",
                level="error",
                message="成功工具参数沉淀失败。",
                details={"phase": "tool_usage_learning", "error_type": type(exc).__name__},
            )
        if skill_extractor is not None:
            context = skill_run_context(result)
            if context is not None:
                try:
                    skill = skill_extractor.extract(context)
                    if skill is not None:
                        self._save_learned_skill(tenant_id, user_id, actual_run_id, skill)
                except Exception as exc:
                    self.memory_store.append_run_log(
                        conversation_namespace,
                        run_id=actual_run_id,
                        event_type="skill.learning_failed",
                        level="error",
                        message="成功工作路径学习失败。",
                        details={"phase": "skill_learning", "error_type": type(exc).__name__},
                    )
                else:
                    self.memory_store.append_run_log(
                        conversation_namespace,
                        run_id=actual_run_id,
                        event_type="skill.learning_completed",
                        level="info",
                        message="成功工作路径学习已处理。",
                        details={"skill_created": skill is not None},
                    )

        artifacts: dict[str, str] = {}
        if export_artifacts:
            artifacts = export_run_artifacts(
                result=result,
                output_dir=self.config.output_dir,
                thread_id=actual_thread_id,
                tenant_id=stable_id("tenant", {"value": tenant_id}),
                user_id=stable_id("user", {"value": user_id}),
            )

        return {
            "thread_id": actual_thread_id,
            "llm_backend": llm_client.backend_name,
            "result": result,
            "artifacts": artifacts,
            "document_diagnostics": [
                {
                    "filename": item["filename"],
                    "page_count": item["page_count"],
                    "text_page_count": item["text_page_count"],
                    "parsed_page_count": item["parsed_page_count"],
                    "parser_kind": item["parser_kind"],
                    "lifecycle": item["lifecycle"],
                    "warnings": list(item["warnings"]),
                }
                for item in document_contexts
            ],
            "session_document_count": len(
                self.list_session_documents(actual_thread_id, tenant_id=tenant_id, user_id=user_id)
            ),
        }

    def _session_document_namespace(self, tenant_id: str, user_id: str, thread_id: str) -> tuple[str, str, str]:
        return (
            stable_id("tenant", {"value": tenant_id}),
            stable_id("user", {"value": user_id}),
            stable_id("thread", {"value": thread_id}),
        )

    def _retain_session_documents(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        documents: list[dict],
    ) -> None:
        namespace = self._session_document_namespace(tenant_id, user_id, thread_id)
        now = datetime.now(UTC)
        with self._session_documents_lock:
            expired_namespaces = [key for key, record in self._session_documents.items() if record[0] <= now]
            for key in expired_namespaces:
                del self._session_documents[key]
            if (
                namespace not in self._session_documents
                and len(self._session_documents) >= self.config.max_session_document_sessions
            ):
                raise ValueError("session document namespace limit has been reached")
            existing_record = self._session_documents.get(namespace)
            existing = list(existing_record[1]) if existing_record is not None and existing_record[0] > now else []
            merged = {str(item["document_id"]): item for item in existing}
            merged.update({str(item["document_id"]): item for item in documents})
            retained = tuple(merged.values())
            if len(retained) > self.config.max_upload_files:
                raise ValueError(f"session document count exceeds {self.config.max_upload_files} documents")
            total_characters = sum(
                len(str(page["text"])) for document in retained for page in document.get("pages") or ()
            )
            if total_characters > self.config.max_pdf_text_characters:
                raise ValueError("session document text exceeds the configured character limit")
            self._session_documents[namespace] = (
                now + timedelta(seconds=self.config.session_document_ttl_seconds),
                retained,
            )

    def _load_session_documents(self, tenant_id: str, user_id: str, thread_id: str) -> list[dict]:
        namespace = self._session_document_namespace(tenant_id, user_id, thread_id)
        with self._session_documents_lock:
            record = self._session_documents.get(namespace)
            if record is None:
                return []
            if record[0] <= datetime.now(UTC):
                del self._session_documents[namespace]
                return []
            return list(record[1])

    def list_session_documents(
        self,
        thread_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        _validate_thread_id(thread_id)
        documents = self._load_session_documents(tenant_id, user_id, thread_id)
        namespace = self._session_document_namespace(tenant_id, user_id, thread_id)
        with self._session_documents_lock:
            record = self._session_documents.get(namespace)
            expires_at = record[0].isoformat() if record is not None else None
        return [
            {
                "document_id": item["document_id"],
                "filename": item["filename"],
                "page_count": item["page_count"],
                "text_page_count": item["text_page_count"],
                "parsed_page_count": item["parsed_page_count"],
                "parser_kind": item["parser_kind"],
                "expires_at": expires_at,
            }
            for item in documents
        ]

    def delete_session_documents(
        self,
        thread_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> int:
        _validate_thread_id(thread_id)
        namespace = self._session_document_namespace(tenant_id, user_id, thread_id)
        with self._session_documents_lock:
            record = self._session_documents.pop(namespace, None)
        return len(record[1]) if record is not None else 0

    def _upload_display_name(self, path: Path) -> str | None:
        """Hide the server-side random upload prefix from evidence citations."""
        try:
            is_managed_upload = path.resolve().parent == self.config.upload_dir.resolve()
        except OSError:
            return None
        if not is_managed_upload:
            return None
        match = re.fullmatch(r"[0-9a-f]{8}_(.+)", path.name)
        return match.group(1) if match else path.name

    def _conversation_namespace(self, tenant_id: str, user_id: str, thread_id: str) -> MemoryNamespace:
        return MemoryNamespace(
            tenant_id=stable_id("tenant", {"value": tenant_id}),
            user_id=stable_id("user", {"value": user_id}),
            kind="conversation_history",
            thread_id=stable_id("thread", {"value": thread_id}),
        )

    def _mcp_tool_index(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "name": tool.spec.name,
                "capability": tool.spec.capability,
                "network_access": tool.spec.network_access,
                "description": tool.spec.description[:200],
                "planner_category": self.mcp_host.planner_category_for(tool.spec.name),
            }
            for tool in self.mcp_tools
        )

    def _mcp_planner_names(self, category: str) -> tuple[str, ...]:
        return tuple(
            tool.spec.name for tool in self.mcp_tools if self.mcp_host.planner_category_for(tool.spec.name) == category
        )

    def _load_conversation_context(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        *,
        summarizer: ConversationSummarizer | None = None,
    ) -> dict:
        if not self.config.conversation_memory_enabled:
            return {}
        return build_conversation_window(
            self.memory_store,
            self._conversation_namespace(tenant_id, user_id, thread_id),
            max_tokens=self.config.conversation_context_tokens,
            recent_tokens=self.config.conversation_recent_tokens,
            summarizer=summarizer,
            token_counter=self.conversation_token_counter,
        )

    def delete_conversation(
        self,
        thread_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> dict[str, int]:
        _validate_thread_id(thread_id)
        namespace = self._conversation_namespace(tenant_id, user_id, thread_id)
        run_ids = self.memory_store.conversation_run_ids(namespace)
        deleted = self.memory_store.delete_conversation(namespace)
        deleted["threads"] = self.memory_store.delete_conversation_thread(
            namespace.tenant_id,
            namespace.user_id,
            thread_id,
        )
        checkpointer = self._open_graph_checkpointer()
        try:
            for stored_run_id in run_ids:
                checkpoint_thread_id = stable_id(
                    "run",
                    {"tenant_id": tenant_id, "thread_id": thread_id, "run_id": stored_run_id},
                )
                checkpointer.delete_thread(checkpoint_thread_id)
        finally:
            self._close_graph_checkpointer()
        return {**deleted, "checkpoints": len(run_ids)}

    def list_run_logs(
        self,
        thread_id: str,
        run_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        _validate_thread_id(thread_id)
        return [
            event.to_dict()
            for event in self.memory_store.list_run_logs(
                self._conversation_namespace(tenant_id, user_id, thread_id),
                run_id,
            )
        ]

    def list_conversation_messages(
        self,
        thread_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 200,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        _validate_thread_id(thread_id)
        events = self.memory_store.list_conversation_messages(
            self._conversation_namespace(tenant_id, user_id, thread_id),
            after_sequence=after_sequence,
            limit=limit,
        )
        return [
            {
                "sequence": event.sequence,
                "event_id": event.event_id,
                "role": "user" if event.kind is ConversationEventKind.USER_MESSAGE else "assistant",
                "content": event.content,
                "run_id": event.run_id,
                "occurred_at": event.occurred_at,
                "status": event.payload.get("status"),
            }
            for event in events
        ]

    def list_conversations(
        self,
        *,
        limit: int = 100,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict[str, str]]:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return self.memory_store.list_conversation_threads(tenant_key, user_key, limit=limit)

    def list_conversation_runs(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        _validate_thread_id(thread_id)
        return self.memory_store.list_conversation_runs(
            self._conversation_namespace(tenant_id, user_id, thread_id),
            limit=limit,
        )

    def get_conversation_run(
        self,
        thread_id: str,
        run_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> dict | None:
        _validate_thread_id(thread_id)
        return self.memory_store.get_conversation_run(
            self._conversation_namespace(tenant_id, user_id, thread_id),
            run_id,
        )

    def _personal_memory_namespace(self, tenant_id: str, user_id: str) -> MemoryNamespace:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return MemoryNamespace(
            tenant_id=tenant_key,
            user_id=user_key,
            kind="personal_memory",
        )

    def _personal_memory_candidate_namespace(self, tenant_id: str, user_id: str) -> MemoryNamespace:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return MemoryNamespace(tenant_id=tenant_key, user_id=user_key, kind="personal_memory_candidates")

    def _tool_usage_memory_namespace(self, tenant_id: str, user_id: str) -> MemoryNamespace:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return MemoryNamespace(tenant_id=tenant_key, user_id=user_key, kind="tool_usage_memory")

    def _learned_skill_namespace(self, tenant_id: str, user_id: str) -> MemoryNamespace:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return MemoryNamespace(tenant_id=tenant_key, user_id=user_key, kind="learned_skills")

    def list_learned_skills(
        self,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        return [
            {
                "skill_id": record.key,
                **LearnedSkill.from_dict(record.value).to_dict(),
                "success_count": int(record.metadata["success_count"]),
                "updated_at": record.updated_at,
            }
            for record in self.memory_store.list(self._learned_skill_namespace(tenant_id, user_id), limit=100)
        ]

    def delete_learned_skill(
        self,
        skill_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> bool:
        return self.memory_store.delete(self._learned_skill_namespace(tenant_id, user_id), skill_id)

    def _save_learned_skill(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str,
        skill: LearnedSkill,
    ) -> None:
        namespace = self._learned_skill_namespace(tenant_id, user_id)
        skill_id = stable_id("skill", {"name": skill.name.casefold()})
        stored = self.memory_store.get(namespace, skill_id)
        run_ids = list(dict.fromkeys([*((stored.metadata.get("run_ids") or []) if stored else []), run_id]))
        self.memory_store.put(
            namespace,
            skill_id,
            skill.to_dict(),
            metadata={
                "schema_version": 1,
                "source": "successful_run",
                "success_count": len(run_ids),
                "run_ids": run_ids,
                "untrusted_guidance": True,
            },
        )

    def save_personal_memory(
        self,
        *,
        kind: PersonalMemoryKind,
        title: str,
        content: str,
        tags: Sequence[str] = (),
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> dict:
        if not self.config.personal_memory_enabled:
            raise ValueError("personal memory is disabled")
        memory = PersonalMemory(kind=kind, title=title, content=content, tags=tuple(tags))
        value = memory.to_dict()
        # An explicit write with the same kind/title replaces the prior value.
        # Conflicting preferences therefore have one visible latest value
        # instead of leaving the model to guess between contradictory records.
        memory_id = stable_id(
            "mem",
            {"kind": memory.kind.value, "title": memory.title.strip().casefold()},
        )
        self._store_personal_memory(
            tenant_id,
            user_id,
            memory_id,
            value,
            {
                "schema_version": 1,
                "write_policy": "explicit_user_action",
                "source": "user",
                "contains_financial_evidence": False,
            },
        )
        record = self.memory_store.get(self._personal_memory_namespace(tenant_id, user_id), memory_id)
        if record is None:
            raise RuntimeError("personal memory write was not durable")
        return {
            "memory_id": record.key,
            **memory.to_dict(),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def list_personal_memories(
        self,
        *,
        kind: PersonalMemoryKind | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        records = self.memory_store.list(self._personal_memory_namespace(tenant_id, user_id), limit=500)
        result = []
        for record in records:
            memory = PersonalMemory.from_dict(record.value)
            if kind is None or memory.kind == kind:
                result.append(
                    {
                        "memory_id": record.key,
                        **memory.to_dict(),
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "metadata": dict(record.metadata),
                    }
                )
        return result

    def delete_personal_memory(
        self,
        memory_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> bool:
        return self.memory_store.delete(self._personal_memory_namespace(tenant_id, user_id), memory_id)

    def _personal_memory_context(self, tenant_id: str, user_id: str) -> tuple[dict, ...]:
        memories = tuple(
            {
                "memory_id": item["memory_id"],
                "kind": item["kind"],
                "title": item["title"],
                "content": item["content"],
                "tags": item["tags"],
                "authority": "低于系统规则和用户当前明确要求",
            }
            for item in self.list_personal_memories(tenant_id=tenant_id, user_id=user_id)
        )
        if len(json.dumps(memories, ensure_ascii=False)) > _MAX_PERSONAL_MEMORY_CHARACTERS:
            raise ValueError("个人长期记忆总量超过 100000 字符，必须先合并或删除后才能完整注入")
        return memories

    def _store_personal_memory(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        value: dict,
        metadata: dict,
    ) -> None:
        namespace = self._personal_memory_namespace(tenant_id, user_id)
        proposed = {
            record.key: PersonalMemory.from_dict(record.value).to_dict()
            for record in self.memory_store.list(namespace, limit=500)
        }
        proposed[memory_id] = PersonalMemory.from_dict(value).to_dict()
        if len(proposed) > 500:
            raise ValueError("个人长期记忆不能超过 500 条")
        if len(json.dumps(list(proposed.values()), ensure_ascii=False)) > _MAX_PERSONAL_MEMORY_CHARACTERS:
            raise ValueError("个人长期记忆总量不能超过 100000 字符")
        self.memory_store.put(namespace, memory_id, value, metadata=metadata)

    def _user_profile_context(self) -> tuple[dict, ...]:
        path = self.config.user_profile_path
        if path is None:
            return ()
        content = path.read_text(encoding="utf-8").strip()
        if not content or len(content) > 8_000:
            raise ValueError("用户长期指令文件必须包含 1 到 8000 个字符")
        return (
            {
                "kind": "user_instructions",
                "title": path.name,
                "content": content,
                "source": "user_managed_file",
                "authority": "低于系统规则，高于系统推断记忆",
            },
        )

    def _consolidate_long_term_memory(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        run_id: str,
        events: Sequence[ConversationEvent],
        extractor: LongTermMemoryExtractor,
    ) -> None:
        existing = self.list_personal_memories(tenant_id=tenant_id, user_id=user_id)
        candidates = extractor.extract(events, existing)
        for candidate in candidates:
            if candidate.confidence < 0.75 or candidate.operation == "ignore":
                continue
            self._merge_long_term_memory_candidate(
                tenant_id,
                user_id,
                thread_id,
                run_id,
                candidate,
                existing,
            )

    def _merge_long_term_memory_candidate(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        run_id: str,
        candidate: LongTermMemoryCandidate,
        existing: list[dict],
    ) -> None:
        matched = _matching_memory(candidate, existing)
        if candidate.operation == "update" and matched is None:
            raise ValueError("长期记忆 update 未匹配到已有同槽位记忆")
        if matched is not None:
            metadata = dict(matched.get("metadata") or {})
            evidence_runs = list(dict.fromkeys([*(metadata.get("evidence_run_ids") or []), run_id]))
            metadata.update(
                {
                    "schema_version": 2,
                    "evidence_run_ids": evidence_runs,
                    "last_evidence_thread_id": stable_id("thread", {"value": thread_id}),
                    "confidence": max(float(metadata.get("confidence") or 0), candidate.confidence),
                    "contains_financial_evidence": False,
                }
            )
            value = PersonalMemory.from_dict(
                {key: matched[key] for key in ("kind", "title", "content", "tags")}
            ).to_dict()
            if (
                candidate.explicitness == "explicit"
                and candidate.operation == "update"
            ):
                value = PersonalMemory(
                    candidate.kind,
                    candidate.title,
                    candidate.content,
                    candidate.tags,
                ).to_dict()
                metadata["write_policy"] = "automatic_llm_consolidation"
                metadata["scope"] = candidate.scope
                metadata["replaces_prior_memory"] = True
            self._store_personal_memory(
                tenant_id,
                user_id,
                matched["memory_id"],
                value,
                metadata,
            )
            return

        candidate_key = stable_id(
            "candidate",
            {"kind": candidate.kind.value, "title": candidate.title.casefold()},
        )
        namespace = self._personal_memory_candidate_namespace(tenant_id, user_id)
        stored = self.memory_store.get(namespace, candidate_key)
        prior_runs = (stored.metadata.get("evidence_run_ids") or []) if stored else []
        evidence_runs = list(dict.fromkeys([*prior_runs, run_id]))
        value = {
            "kind": candidate.kind.value,
            "title": candidate.title,
            "content": candidate.content,
            "scope": candidate.scope,
            "explicitness": candidate.explicitness,
            "confidence": candidate.confidence,
            "operation": candidate.operation,
            "tags": list(candidate.tags),
        }
        metadata = {
            "schema_version": 1,
            "evidence_run_ids": evidence_runs,
            "last_evidence_thread_id": stable_id("thread", {"value": thread_id}),
        }
        self.memory_store.put(namespace, candidate_key, value, metadata=metadata)
        should_promote = candidate.explicitness == "explicit" or len(evidence_runs) >= 2
        if not should_promote:
            return
        memory = PersonalMemory(candidate.kind, candidate.title, candidate.content, candidate.tags)
        memory_id = stable_id(
            "mem",
            {"kind": candidate.kind.value, "title": candidate.title.casefold()},
        )
        self._store_personal_memory(
            tenant_id,
            user_id,
            memory_id,
            memory.to_dict(),
            {
                "schema_version": 2,
                "write_policy": "automatic_llm_consolidation",
                "source": "conversation_inference",
                "explicitness": candidate.explicitness,
                "scope": candidate.scope,
                "confidence": candidate.confidence,
                "evidence_run_ids": evidence_runs,
                "contains_financial_evidence": False,
            },
        )
        self.memory_store.delete(namespace, candidate_key)
        existing.append({"memory_id": memory_id, **memory.to_dict()})

    def _record_tool_usage_memory(
        self,
        tenant_id: str,
        user_id: str,
        audits: Sequence[dict],
    ) -> None:
        mcp_names = {tool.spec.name for tool in self.mcp_tools}
        namespace = self._tool_usage_memory_namespace(tenant_id, user_id)
        catalog = {str(item["name"]): item for item in self.mcp_host.catalog_index()}
        for audit in audits:
            if audit.get("result_status") != "success":
                continue
            invoked_name = str(audit.get("tool_name") or "")
            arguments = dict(audit.get("arguments") or {})
            if invoked_name == "mcp.call_tool":
                local_name = str(arguments.get("name") or "")
                successful_arguments = arguments.get("arguments")
            elif invoked_name in mcp_names:
                local_name = invoked_name
                successful_arguments = arguments
            else:
                continue
            if local_name not in catalog or not isinstance(successful_arguments, dict):
                continue
            schema = catalog[local_name].get("input_schema")
            schema_fingerprint = stable_id("schema", {"value": schema})
            key = stable_id(
                "tooluse",
                {"tool": local_name, "schema": schema_fingerprint, "arguments": successful_arguments},
            )
            stored = self.memory_store.get(namespace, key)
            success_count = int(stored.value.get("success_count") or 0) + 1 if stored else 1
            company = successful_arguments.get("company")
            symbol = successful_arguments.get("symbol")
            entity_alias = (
                {"canonical_name": company, "provider_identifier": symbol}
                if isinstance(company, str) and isinstance(symbol, str)
                else None
            )
            self.memory_store.put(
                namespace,
                key,
                {
                    "tool_name": local_name,
                    "server_id": local_name.partition(".")[0],
                    "schema_fingerprint": schema_fingerprint,
                    "arguments": successful_arguments,
                    "entity_alias": entity_alias,
                    "success_count": success_count,
                    "last_verified_at": str(audit.get("timestamp") or ""),
                },
                metadata={"schema_version": 1, "source": "verified_success", "contains_credentials": False},
            )

    def _recall_tool_usage_memory(self, tenant_id: str, user_id: str, query: str) -> tuple[dict, ...]:
        terms = _memory_terms(query)
        current_schemas = {
            str(item["name"]): stable_id("schema", {"value": item.get("input_schema")})
            for item in self.mcp_host.catalog_index()
        }
        ranked = []
        for record in self.memory_store.list(self._tool_usage_memory_namespace(tenant_id, user_id), limit=200):
            value = dict(record.value)
            if current_schemas.get(str(value.get("tool_name") or "")) != value.get("schema_fingerprint"):
                continue
            haystack = f"{value.get('tool_name', '')} {json.dumps(value.get('arguments') or {}, ensure_ascii=False)}"
            overlap = len(terms.intersection(_memory_terms(haystack)))
            ranked.append((overlap, value.get("success_count", 0), record.updated_at, value))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return tuple(item[3] for item in ranked[:5] if item[0] > 0)

    def ingest_personal_documents(
        self,
        document_paths: Sequence[str],
        *,
        allow_network: bool = False,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        if not self.config.personal_knowledge_enabled:
            raise ValueError("personal knowledge is disabled")
        if not 1 <= len(document_paths) <= self.config.max_upload_files:
            raise ValueError("personal knowledge upload count is invalid")
        network_allowed = self.config.allow_network and allow_network
        if self.pdf_document_parser is None:
            raise ValueError("a PaddleOCR or MCP PDF document parser is required for personal PDF ingestion")
        if self.pdf_parser_network_access and not network_allowed:
            raise ValueError("personal PDF parsing requires server and request network authorization")
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        results = []
        for value in document_paths:
            path = Path(value)
            parsed = parse_pdf_document(
                path,
                include_pages=True,
                max_pages=self.config.max_pdf_pages,
                max_file_bytes=self.config.max_upload_bytes,
                max_text_characters=self.config.max_pdf_text_characters,
                display_name=self._upload_display_name(path),
                document_parser=self.pdf_document_parser,
            )
            if not parsed.get("pages"):
                raise ValueError("PDF document parser returned no extractable personal PDF text")
            results.append(
                self.personal_knowledge_store.add_document(
                    tenant_key,
                    user_key,
                    parsed,
                    embedding_provider=self.embedding_provider,
                )
            )
        return results

    def list_personal_documents(
        self,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return self.personal_knowledge_store.list_documents(tenant_key, user_key)

    def delete_personal_document(
        self,
        document_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> bool:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return self.personal_knowledge_store.delete_document(tenant_key, user_key, document_id)

    def submit_job(
        self,
        query: str,
        thread_id: str | None = None,
        *,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        cleanup_documents: bool = False,
        idempotency_key: str | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        allow_network: bool = False,
        use_session_documents: bool = False,
        use_personal_memory: bool = True,
        use_personal_knowledge: bool = True,
        retain_documents_for_session: bool = False,
    ) -> dict:
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        job_id = f"job-{uuid4().hex[:10]}"
        key = idempotency_key or job_id
        job, created = self.repository.submit_job(
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=actual_thread_id,
            query=query,
            payload={
                "query": query,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": actual_thread_id,
                "export_artifacts": export_artifacts,
                "document_paths": document_paths or [],
                "cleanup_documents": cleanup_documents,
                "allow_network": allow_network,
                "use_session_documents": use_session_documents,
                "use_personal_memory": use_personal_memory,
                "use_personal_knowledge": use_personal_knowledge,
                "retain_documents_for_session": retain_documents_for_session,
            },
            idempotency_key=key,
            max_attempts=self.config.job_max_attempts,
        )
        return {
            "job_id": job["job_id"],
            "thread_id": job["thread_id"],
            "status": job["status"],
            "created": created,
        }

    def process_queued_job(self, worker_id: str, *, job_id: str | None = None) -> bool:
        from .queueing import ReliableJobQueue

        queue = ReliableJobQueue(
            self.repository,
            lease_seconds=self.config.job_lease_seconds,
            retry_delay_seconds=self.config.job_retry_delay_seconds,
        )
        payload = queue.claim(worker_id, job_id=job_id)
        if payload is None:
            return False
        heartbeat_stop = Event()

        def renew_lease() -> None:
            interval = max(1, self.config.job_lease_seconds // 3)
            while not heartbeat_stop.wait(interval):
                if not queue.renew(payload["job_id"], payload["lease_token"]):
                    return

        heartbeat = Thread(target=renew_lease, name=f"lease-{payload['job_id']}", daemon=True)
        heartbeat.start()
        try:
            self.run_job(
                payload["job_id"],
                payload["query"],
                payload["thread_id"],
                payload.get("export_artifacts", True),
                payload.get("document_paths") or [],
                tenant_id=payload["tenant_id"],
                user_id=payload["user_id"],
                allow_network=payload.get("allow_network", False),
                use_session_documents=payload.get("use_session_documents", False),
                use_personal_memory=payload.get("use_personal_memory", True),
                use_personal_knowledge=payload.get("use_personal_knowledge", True),
                retain_documents_for_session=payload.get("retain_documents_for_session", False),
                resume=payload["attempt_count"] > 1,
            )
        except Exception as exc:
            heartbeat_stop.set()
            heartbeat.join(timeout=1)
            status = queue.fail(payload["job_id"], payload["lease_token"], type(exc).__name__)
            if status == "dead" and payload.get("cleanup_documents"):
                for document_path in payload.get("document_paths") or []:
                    Path(document_path).unlink(missing_ok=True)
            return True
        heartbeat_stop.set()
        heartbeat.join(timeout=1)
        if not queue.complete(payload["job_id"], payload["lease_token"]):
            raise RuntimeError("analysis completed after its job lease expired")
        if payload.get("cleanup_documents"):
            for document_path in payload.get("document_paths") or []:
                Path(document_path).unlink(missing_ok=True)
        return True

    def run_job(
        self,
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        allow_network: bool = False,
        use_session_documents: bool = False,
        use_personal_memory: bool = True,
        use_personal_knowledge: bool = True,
        retain_documents_for_session: bool = False,
        resume: bool = False,
    ) -> None:
        existing = self.repository.get_job(job_id)
        if existing is None:
            raise ValueError("analysis job does not exist")
        self.repository.update_job_status(job_id=job_id, status="running")
        try:
            try:
                response = self.analyze(
                    query=query,
                    thread_id=thread_id,
                    export_artifacts=export_artifacts,
                    document_paths=document_paths,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    allow_network=allow_network,
                    use_session_documents=use_session_documents,
                    use_personal_memory=use_personal_memory,
                    use_personal_knowledge=use_personal_knowledge,
                    retain_documents_for_session=retain_documents_for_session,
                    run_id=job_id,
                    resume=resume,
                )
            except ValueError as exc:
                if not resume or str(exc) != "no LangGraph checkpoint exists for this run":
                    raise
                response = self.analyze(
                    query=query,
                    thread_id=thread_id,
                    export_artifacts=export_artifacts,
                    document_paths=document_paths,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    allow_network=allow_network,
                    use_session_documents=use_session_documents,
                    use_personal_memory=use_personal_memory,
                    use_personal_knowledge=use_personal_knowledge,
                    retain_documents_for_session=retain_documents_for_session,
                    run_id=job_id,
                    resume=False,
                )
            self.repository.update_job_status(
                job_id=job_id,
                status="completed",
                llm_backend=response["llm_backend"],
                result=response["result"],
                artifacts=response["artifacts"],
            )
        except Exception as exc:
            self.repository.update_job_status(
                job_id=job_id,
                status="failed",
                # Provider exception strings are untrusted and may contain a
                # credential-bearing URL or response fragment.  Job status is
                # public API data; detailed diagnostics belong in protected
                # telemetry, not the persisted response row.
                error_message=f"Analysis failed ({type(exc).__name__}).",
            )
            raise

    def get_job(
        self,
        job_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> dict | None:
        return self.repository.get_job_for_principal(job_id, tenant_id, user_id)

    def cancel_job(
        self,
        job_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> str | None:
        from .queueing import ReliableJobQueue

        if self.repository.get_job_for_principal(job_id, tenant_id, user_id) is None:
            return None
        return ReliableJobQueue(self.repository).request_cancellation(job_id)

    def list_jobs(
        self,
        limit: int = 20,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> list[dict]:
        return self.repository.list_jobs(tenant_id, user_id, limit=limit)

    def run_retention(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(UTC)
        operational_cutoff = current - timedelta(days=self.config.operational_retention_days)
        job_cutoff = current - timedelta(days=self.config.completed_job_retention_days)
        deleted = self.memory_store.delete_operational_history_before(operational_cutoff.isoformat())
        deleted["analysis_jobs"] = self.repository.delete_terminal_jobs_before(job_cutoff.isoformat())
        with self._session_documents_lock:
            expired = [key for key, record in self._session_documents.items() if record[0] <= current]
            for key in expired:
                del self._session_documents[key]
        deleted["session_document_namespaces"] = len(expired)
        for name, directory, cutoff in (
            ("upload_files", self.config.upload_dir, operational_cutoff),
            ("artifact_files", self.config.output_dir, job_cutoff),
        ):
            count = 0
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime, UTC) < cutoff:
                        path.unlink()
                        count += 1
            deleted[name] = count
        return deleted

    def save_uploaded_files(self, files: list[tuple[str, bytes]]) -> list[str]:
        if not 1 <= len(files) <= self.config.max_upload_files:
            raise ValueError(f"upload count must be between 1 and {self.config.max_upload_files}")
        self.config.upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        try:
            for filename, content in files:
                if not content.startswith(b"%PDF-"):
                    raise ValueError("uploaded content is not a PDF")
                if len(content) > self.config.max_upload_bytes:
                    raise ValueError(f"uploaded file exceeds {self.config.max_upload_bytes} bytes")
                normalized = safe_upload_name(filename)
                unique_name = f"{uuid4().hex[:8]}_{normalized}"
                target_path = safe_child(self.config.upload_dir, unique_name)
                target_path.write_bytes(content)
                saved_paths.append(str(target_path))
        except Exception:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
            raise
        return saved_paths


def _validate_thread_id(thread_id: str) -> None:
    if not thread_id.strip() or len(thread_id) > 200 or any(ord(item) < 32 or ord(item) == 127 for item in thread_id):
        raise ValueError("thread_id is invalid")


def _normalized_entities(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for item in values if (text := str(item).strip()) and len(text) <= 200))[:50]


def _assistant_reply(result: dict) -> str:
    if result.get("status") == "needs_clarification":
        return str(result["report"]).strip()
    claims = [
        str(item.get("text") or "").strip()
        for item in (result.get("bundle") or {}).get("claims") or ()
        if str(item.get("text") or "").strip()
    ]
    return "\n\n".join(claims) if claims else str(result["report"]).strip()


def _requests_document_research(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(
        marker in normalized
        for marker in (
            " document ",
            " report ",
            " filing text ",
            " internal ",
            " knowledge base ",
            " news ",
            " web search ",
            " search for ",
            " source material ",
            "文档",
            "报告",
            "财报原文",
            "内部资料",
            "外部资料",
            "知识库",
            "新闻",
            "网页",
            "联网搜索",
            "检索",
            "搜索",
            "查找资料",
        )
    )


def _memory_terms(text: str) -> set[str]:
    normalized = text.casefold()
    latin = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
    return latin | {value[index : index + 2] for value in cjk for index in range(max(0, len(value) - 1))}


def _matching_memory(candidate: LongTermMemoryCandidate, memories: Sequence[dict]) -> dict | None:
    candidate_terms = _memory_terms(f"{candidate.title} {candidate.content}")
    matches = []
    for memory in memories:
        if memory.get("kind") != candidate.kind.value:
            continue
        if str(memory.get("title") or "").strip().casefold() == candidate.title.casefold():
            return memory
        memory_terms = _memory_terms(f"{memory.get('title', '')} {memory.get('content', '')}")
        union = candidate_terms.union(memory_terms)
        similarity = len(candidate_terms.intersection(memory_terms)) / len(union) if union else 0.0
        if similarity >= 0.6:
            matches.append((similarity, str(memory.get("updated_at") or ""), memory))
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return matches[0][2] if matches else None


def _personal_principal_ids(tenant_id: str, user_id: str) -> tuple[str, str]:
    return stable_id("tenant", {"value": tenant_id}), stable_id("user", {"value": user_id})
