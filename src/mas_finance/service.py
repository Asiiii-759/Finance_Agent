from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import AdaptivePlanner, DeterministicSynthesizer, ResearchRequest
from .config import AppConfig
from .contracts import stable_id
from .corpus import CorpusDocument, InMemoryCorpus
from .documents import PDFOCRProvider, detect_companies, parse_pdf_document
from .embeddings import EmbeddingProvider, HTTPEmbeddingClient
from .formula import formula_harness_tool
from .graph import FinancialResearchAgent
from .harness import SideEffect, Tool, ToolHarness, ToolResultKind
from .knowledge import finance_knowledge_harness_tool
from .llm import build_llm_client
from .macro import FREDClient, FREDEvidenceAdapter, fred_series_harness_tool
from .market import (
    MarketEvidenceAdapter,
    MarketHistoryEvidenceAdapter,
    market_data_harness_tool,
    market_history_harness_tool,
)
from .market_data import DEFAULT_TICKER_MAP, MarketDataClient
from .memory_store import (
    ConversationEventKind,
    EntityRelation,
    MemoryNamespace,
    PersonalMemory,
    PersonalMemoryKind,
    SQLiteMemoryStore,
    build_conversation_window,
)
from .metrics import describe_metric_operations, financial_calculation_harness_tool
from .ocr import PaddleOCRClient
from .personal_knowledge import PersonalKnowledgeClient, SQLitePersonalKnowledgeBase
from .planning import ModelPlanner, llm_planning_harness_tool
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
from .synthesis import EvidenceBoundLLMSynthesizer, llm_synthesis_harness_tool
from .web_search import BochaWebSearchClient, BraveWebSearchClient, WebSearchEvidenceAdapter, web_search_harness_tool

if TYPE_CHECKING:
    from .database import JobRepository


class FinanceAnalysisService:
    def __init__(
        self,
        config: AppConfig,
        *,
        retrieval_sources: Sequence[RetrievalSource] = (),
        evidence_tools: Sequence[Tool] = (),
        pdf_ocr_provider: PDFOCRProvider | None = None,
        pdf_ocr_network_access: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config
        self.retrieval_sources = tuple(retrieval_sources)
        self.evidence_tools = tuple(evidence_tools)
        self.pdf_ocr_provider = pdf_ocr_provider or (
            PaddleOCRClient(
                config.paddleocr_access_token,
                job_url=config.paddleocr_job_url,
                model=config.paddleocr_model,
                max_file_bytes=config.max_upload_bytes,
            )
            if config.paddleocr_access_token
            else None
        )
        self.pdf_ocr_network_access = pdf_ocr_network_access
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
            "finance.knowledge",
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
            "llm.plan",
            "llm.synthesize",
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
        self._repository: JobRepository | None = None
        self._memory_store: SQLiteMemoryStore | None = None
        self._personal_knowledge_store: SQLitePersonalKnowledgeBase | None = None
        # Session documents are deliberately process-local: unlike thread
        # metadata they must not silently become a persistent knowledge base.
        self._session_documents: dict[tuple[str, str, str], tuple[datetime, tuple[dict, ...]]] = {}
        self._session_documents_lock = RLock()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph_checkpoint_connection = sqlite3.connect(
            self.config.db_path,
            timeout=15,
            check_same_thread=False,
        )
        self._graph_checkpointer = SqliteSaver(self._graph_checkpoint_connection)

    def close(self) -> None:
        self._graph_checkpoint_connection.close()

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
                "name": "finance.knowledge",
                "capability": "knowledge.read",
                "description": "Versioned finance definitions, formulas and interpretation caveats.",
                "network_access": False,
                "availability": "always",
                **interface(("query",), ("concepts", "top_k")),
            },
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
                "name": "llm.plan",
                "capability": "model.generate",
                "description": "Choose one next authorized evidence-gathering action.",
                "network_access": bool(self.config.llm.api_key),
                "availability": "remote" if self.config.llm.api_key else "not_registered_deterministic_planner",
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
                "description": "Generate evidence-bound claims with literal quote validation.",
                "network_access": bool(self.config.llm.api_key),
                "availability": ("remote" if self.config.llm.api_key else "not_registered_deterministic_synthesis"),
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
        # Network access requires both deployment authorization and explicit
        # per-request consent. None never means implicit consent.
        network_allowed = self.config.allow_network and allow_network is True
        thread_context = self._load_conversation_context(tenant_id, user_id, actual_thread_id)
        personal_context = (
            self._recall_personal_memories(tenant_id, user_id, query)
            if use_personal_memory and self.config.personal_memory_enabled
            else ()
        )
        llm_client = build_llm_client(self.config.llm)
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        harness.register(formula_harness_tool())
        harness.register(finance_knowledge_harness_tool())
        for tool in self.evidence_tools:
            harness.register(tool)
        if llm_client is not None:
            harness.register(
                llm_planning_harness_tool(
                    llm_client,
                    network_access=True,
                )
            )
            harness.register(
                llm_synthesis_harness_tool(
                    llm_client,
                    network_access=True,
                )
            )
        if self.config.bocha_search_api_key:
            harness.register(
                web_search_harness_tool(
                    WebSearchEvidenceAdapter(BochaWebSearchClient(self.config.bocha_search_api_key))
                )
            )
        elif self.config.brave_search_api_key:
            harness.register(
                web_search_harness_tool(
                    WebSearchEvidenceAdapter(BraveWebSearchClient(self.config.brave_search_api_key))
                )
            )

        explicit_document_entities = _normalized_entities(entities or [])
        current_document_contexts = [
            parse_pdf_document(
                Path(path),
                include_pages=True,
                max_pages=self.config.max_pdf_pages,
                max_file_bytes=self.config.max_upload_bytes,
                max_text_characters=self.config.max_pdf_text_characters,
                display_name=self._upload_display_name(Path(path)),
                ocr_provider=(self.pdf_ocr_provider if not self.pdf_ocr_network_access or network_allowed else None),
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
                                "page_image_count": page["image_count"],
                            },
                        )
                    )
            if not ingested_chunks:
                if self.pdf_ocr_provider and self.pdf_ocr_network_access and not network_allowed:
                    raise ValueError("uploaded PDF requires OCR; server and request network authorization are required")
                if self.pdf_ocr_provider:
                    raise ValueError("uploaded PDF has no extractable text after OCR")
                raise ValueError("uploaded PDF has no extractable text; configure a trusted OCR provider")
            corpus_adapter = RetrievalEvidenceAdapter(corpus)
            harness.register(
                retrieval_harness_tool(
                    corpus_adapter,
                    fixed_search_mode="lexical",
                    description=(
                        "Run deterministic BM25 search over current request/session PDFs. Use for exact terms, "
                        "names, identifiers and when embedding search is unavailable."
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
                            "Run BM25 and semantic embedding retrieval over current request/session PDFs and "
                            "fuse both rankings with RRF. Prefer for paraphrases, synonyms and cross-language queries."
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
        if (
            use_personal_knowledge
            and self.config.personal_knowledge_enabled
            and personal_documents
        ):
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
                        "Run deterministic BM25 search over the user's persistent financial documents. Set "
                        "diversify_documents=true only for explicit cross-document comparison or synthesis."
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
                            "Run BM25 and semantic embedding retrieval over the user's persistent documents and "
                            "fuse both rankings with RRF. Prefer for paraphrases, synonyms and cross-language queries."
                        ),
                    )
                )
                if not self.embedding_provider.network_access or network_allowed:
                    document_tool_names.extend(("personal.hybrid_search", "personal.search"))
                else:
                    document_tool_names.extend(("personal.search", "personal.hybrid_search"))
            else:
                document_tool_names.append("personal.search")

        detected_entities = detect_companies(
            query
            + " "
            + " ".join(
                company for document in document_contexts for company in document.get("detected_companies") or []
            )
        )
        requested_entities, use_thread_context = _resolve_request_entities(
            query=query,
            explicit_entities=entities or [],
            detected_entities=detected_entities,
            conversation_context=thread_context,
        )
        if requested_entities:
            market_client = MarketDataClient(
                provider=self.config.market_data_provider,
                alphavantage_api_key=self.config.alphavantage_api_key,
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

        remembered_symbols = {
            str(item["subject"]): str(item["object"])
            for item in thread_context.get("relations") or []
            if item.get("predicate") == "has_symbol"
        }
        resolved_symbols: dict[str, str] = {}
        for entity in requested_entities:
            candidate = (symbols or {}).get(entity) or remembered_symbols.get(entity) or DEFAULT_TICKER_MAP.get(entity)
            if candidate:
                resolved_symbols[entity] = str(candidate).strip()
            elif re.fullmatch(r"[A-Za-z0-9.^=_:-]{1,32}", entity):
                resolved_symbols[entity] = entity
        if requested_entities and self.config.sec_user_agent:
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
                        )
                    )
                )
            )

        document_research_required = (
            bool(document_contexts)
            or require_documents is True
            or (
                bool(document_tool_names)
                and require_documents is None
                and _requests_document_research(query)
            )
        )
        relations = tuple(
            [
                EntityRelation(entity, "has_symbol", resolved_symbols[entity])
                for entity in requested_entities
                if entity in resolved_symbols
            ]
            + [
                EntityRelation(requested_entities[index], "co_mentioned", requested_entities[index + 1])
                for index in range(len(requested_entities) - 1)
            ]
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
            max_model_calls=7 if llm_client is not None else 1,
            require_documents=document_research_required,
            require_market_data=require_market_data,
            require_market_history=require_market_history,
            require_regulatory_data=require_regulatory_data,
            market_history_range=market_history_range,
            market_history_interval=market_history_interval,
            macro_series=tuple(macro_series or ()),
            calculations=tuple(dict(item) for item in (calculations or ())),
            thread_context=thread_context if use_thread_context else {},
            personal_context=personal_context,
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
                relations=relations,
            )
        try:
            outcome = FinancialResearchAgent(
                harness,
                planner=(
                    ModelPlanner(
                        harness,
                        fallback=AdaptivePlanner(document_tools=tuple(document_tool_names)),
                        max_evidence_chars=self.config.planning_evidence_characters,
                    )
                    if llm_client is not None
                    else AdaptivePlanner(document_tools=tuple(document_tool_names))
                ),
                synthesizer=(
                    EvidenceBoundLLMSynthesizer(
                        llm_client,
                        harness=harness,
                        max_evidence_chars=self.config.synthesis_evidence_characters,
                        max_output_tokens=self.config.synthesis_output_tokens,
                    )
                    if llm_client is not None
                    else DeterministicSynthesizer()
                ),
                checkpointer=self._graph_checkpointer,
            ).run(request, resume=resume)
        finally:
            if self.config.conversation_memory_enabled:
                namespace = self._conversation_namespace(tenant_id, user_id, actual_thread_id)
                for audit in harness.audit_events(actual_run_id):
                    self.memory_store.append_conversation_event(
                        namespace,
                        event_id=stable_id("event", {"run_id": actual_run_id, "call_id": audit["call_id"]}),
                        kind=ConversationEventKind.TOOL_EVENT,
                        content=f"{audit['tool_name']}: {audit['result_status']}",
                        occurred_at=str(audit["timestamp"]),
                        run_id=actual_run_id,
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
        result = outcome.to_dict()
        if self.config.conversation_memory_enabled:
            namespace = self._conversation_namespace(tenant_id, user_id, actual_thread_id)
            self.memory_store.append_conversation_event(
                namespace,
                event_id=stable_id("event", {"run_id": actual_run_id, "kind": "assistant"}),
                kind=ConversationEventKind.ASSISTANT_MESSAGE,
                content=str(result["report"]),
                run_id=actual_run_id,
                entities=requested_entities,
                relations=relations,
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
            build_conversation_window(
                self.memory_store,
                namespace,
                max_characters=self.config.conversation_context_characters,
                recent_event_count=self.config.conversation_recent_events,
            )

        artifacts: dict[str, str] = {}
        if export_artifacts:
            artifacts = export_run_artifacts(
                result=result,
                output_dir=self.config.output_dir,
                thread_id=actual_thread_id,
            )

        return {
            "thread_id": actual_thread_id,
            "llm_backend": (llm_client.backend_name if llm_client is not None else "deterministic"),
            "result": result,
            "artifacts": artifacts,
            "document_diagnostics": [
                {
                    "filename": item["filename"],
                    "page_count": item["page_count"],
                    "text_page_count": item["text_page_count"],
                    "ocr_page_count": item["ocr_page_count"],
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
                "ocr_page_count": item["ocr_page_count"],
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

    def _load_conversation_context(self, tenant_id: str, user_id: str, thread_id: str) -> dict:
        if not self.config.conversation_memory_enabled:
            return {}
        return build_conversation_window(
            self.memory_store,
            self._conversation_namespace(tenant_id, user_id, thread_id),
            max_characters=self.config.conversation_context_characters,
            recent_event_count=self.config.conversation_recent_events,
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
        for stored_run_id in run_ids:
            checkpoint_thread_id = stable_id(
                "run",
                {"tenant_id": tenant_id, "thread_id": thread_id, "run_id": stored_run_id},
            )
            self._graph_checkpointer.delete_thread(checkpoint_thread_id)
        return {**deleted, "checkpoints": len(run_ids)}

    def _personal_memory_namespace(self, tenant_id: str, user_id: str) -> MemoryNamespace:
        tenant_key, user_key = _personal_principal_ids(tenant_id, user_id)
        return MemoryNamespace(
            tenant_id=tenant_key,
            user_id=user_key,
            kind="personal_memory",
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
        self.memory_store.put(
            self._personal_memory_namespace(tenant_id, user_id),
            memory_id,
            value,
            metadata={
                "schema_version": 1,
                "write_policy": "explicit_user_action",
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

    def _recall_personal_memories(self, tenant_id: str, user_id: str, query: str) -> tuple[dict, ...]:
        query_terms = _memory_terms(query)
        ranked: list[tuple[int, str, dict]] = []
        for item in self.list_personal_memories(tenant_id=tenant_id, user_id=user_id):
            memory_terms = _memory_terms(
                " ".join((item["title"], item["content"], " ".join(item["tags"])))
            )
            overlap = len(query_terms.intersection(memory_terms))
            always_relevant = item["kind"] in {
                PersonalMemoryKind.PROFILE.value,
                PersonalMemoryKind.PREFERENCE.value,
            }
            if not always_relevant and overlap == 0:
                continue
            payload = {
                "memory_id": item["memory_id"],
                "kind": item["kind"],
                "title": item["title"],
                "content": item["content"][:2_000],
                "tags": item["tags"],
            }
            ranked.append((overlap + int(always_relevant), item["updated_at"], payload))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        selected: list[dict] = []
        characters = 0
        for _score, _updated_at, payload in ranked:
            size = len(json.dumps(payload, ensure_ascii=False))
            if characters + size > 12_000:
                continue
            selected.append(payload)
            characters += size
            if len(selected) == 8:
                break
        return tuple(selected)

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
                ocr_provider=(self.pdf_ocr_provider if not self.pdf_ocr_network_access or network_allowed else None),
            )
            if not parsed.get("pages"):
                if self.pdf_ocr_provider and self.pdf_ocr_network_access and not network_allowed:
                    raise ValueError("personal PDF requires OCR; server and request network authorization are required")
                raise ValueError("personal PDF has no extractable text")
            results.append(self.personal_knowledge_store.add_document(tenant_key, user_key, parsed))
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

    def submit_job(self, query: str, thread_id: str | None = None) -> dict:
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        job_id = f"job-{uuid4().hex[:10]}"
        self.repository.create_job(job_id=job_id, thread_id=actual_thread_id, query=query)
        return {"job_id": job_id, "thread_id": actual_thread_id, "status": "pending"}

    def enqueue_job(
        self,
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        cleanup_documents: bool = False,
    ) -> bool:
        if not self.config.redis_url:
            return False
        from .queueing import RedisQueueManager

        queue = RedisQueueManager(self.config.redis_url, self.config.redis_queue_name)
        queue.enqueue(
            {
                "job_id": job_id,
                "query": query,
                "thread_id": thread_id,
                "export_artifacts": export_artifacts,
                "document_paths": document_paths or [],
                "cleanup_documents": cleanup_documents,
            }
        )
        return True

    def run_job(
        self,
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        cleanup_documents: bool = False,
    ) -> None:
        existing = self.repository.get_job(job_id)
        if existing is None:
            raise ValueError("analysis job does not exist")
        resume = existing["status"] == "running"
        self.repository.update_job_status(job_id=job_id, status="running")
        try:
            response = self.analyze(
                query=query,
                thread_id=thread_id,
                export_artifacts=export_artifacts,
                document_paths=document_paths,
                run_id=job_id,
                resume=resume,
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
        finally:
            if cleanup_documents:
                for document_path in document_paths or []:
                    Path(document_path).unlink(missing_ok=True)

    def get_job(self, job_id: str) -> dict | None:
        return self.repository.get_job(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        return self.repository.list_jobs(limit=limit)

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


def _resolve_request_entities(
    *,
    query: str,
    explicit_entities: list[str],
    detected_entities: list[str],
    conversation_context: dict,
) -> tuple[tuple[str, ...], bool]:
    explicit = _normalized_entities(explicit_entities)
    detected = _normalized_entities(detected_entities)
    remembered = _normalized_entities(conversation_context.get("focus_entities") or [])
    contextual = _is_contextual_followup(query)
    has_history = int(conversation_context.get("manifest", {}).get("latest_sequence") or 0) > 0
    if explicit:
        return explicit, has_history
    if detected:
        if _references_previous_entity(query):
            return tuple(dict.fromkeys((*detected, *remembered))), True
        return detected, has_history
    if _references_first_entity(query):
        return remembered[:1], True
    if _references_last_entity(query):
        return remembered[-1:], True
    if _references_entity_group(query):
        return remembered, True
    if _references_previous_entity(query):
        # A singular pronoun after a multi-entity turn is ambiguous.  Keep the
        # context visible to the planner but do not guess which entity it means.
        return (remembered if len(remembered) == 1 else ()), True
    if contextual:
        return remembered, True
    return (), False


def _validate_thread_id(thread_id: str) -> None:
    if not thread_id.strip() or len(thread_id) > 200 or any(ord(item) < 32 or ord(item) == 127 for item in thread_id):
        raise ValueError("thread_id is invalid")


def _normalized_entities(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for item in values if (text := str(item).strip()) and len(text) <= 200))[:50]


def _references_previous_entity(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(
        marker in normalized
        for marker in (" 它", "其", "该公司", "这家公司", "上述公司", " it ", " its ", " that company ")
    )


def _references_first_entity(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(marker in normalized for marker in ("前者", "第一个", "第一家", " former ", " first one "))


def _references_last_entity(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(marker in normalized for marker in ("后者", "最后一个", "最后一家", " latter ", " last one "))


def _references_entity_group(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(
        marker in normalized
        for marker in ("它们", "这些公司", "上述公司们", "两者", " they ", " them ", " those companies ", " both ")
    )


def _is_contextual_followup(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return _references_previous_entity(query) or any(
        marker in normalized for marker in ("呢", "那么", "那 ", "继续", "what about", "how about", "and what")
    )


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


def _personal_principal_ids(tenant_id: str, user_id: str) -> tuple[str, str]:
    return stable_id("tenant", {"value": tenant_id}), stable_id("user", {"value": user_id})
