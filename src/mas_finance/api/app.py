from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile

from ..config import AppConfig
from ..documents import PDFDocumentParser
from ..embeddings import EmbeddingProvider
from ..harness import Tool
from ..logging_utils import configure_logging, request_logging_middleware
from ..memory_store import ConversationSummarizer, PersonalMemoryKind, TokenCounter
from ..retrieval import RetrievalSource
from ..service import FinanceAnalysisService
from .auth import build_api_key_dependency
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    JobResponse,
    PersonalMemoryCreate,
    PersonalMemoryResponse,
    SubmitJobRequest,
    SubmitJobResponse,
)


def create_app(
    config: AppConfig | None = None,
    *,
    retrieval_sources: Sequence[RetrievalSource] = (),
    evidence_tools: Sequence[Tool] = (),
    pdf_document_parser: PDFDocumentParser | None = None,
    pdf_parser_network_access: bool = True,
    embedding_provider: EmbeddingProvider | None = None,
    conversation_summarizer: ConversationSummarizer | None = None,
    conversation_token_counter: TokenCounter | None = None,
) -> FastAPI:
    configure_logging()
    app_config = config or AppConfig.from_env()
    service = FinanceAnalysisService(
        app_config,
        retrieval_sources=retrieval_sources,
        evidence_tools=evidence_tools,
        pdf_document_parser=pdf_document_parser,
        pdf_parser_network_access=pdf_parser_network_access,
        embedding_provider=embedding_provider,
        conversation_summarizer=conversation_summarizer,
        conversation_token_counter=conversation_token_counter,
    )
    auth_dependency = build_api_key_dependency(app_config.api_key)

    async def run_job_in_thread(
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        cleanup_documents: bool = False,
    ) -> None:
        service.run_job(
            job_id,
            query,
            thread_id,
            export_artifacts,
            document_paths,
            cleanup_documents,
        )

    async def persist_uploads(files: list[UploadFile]) -> list[str]:
        if not 1 <= len(files) <= app_config.max_upload_files:
            raise HTTPException(status_code=413, detail="Too many uploaded files.")
        payloads: list[tuple[str, bytes]] = []
        for upload in files:
            content = await upload.read(app_config.max_upload_bytes + 1)
            if len(content) > app_config.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            payloads.append((upload.filename or "document.pdf", content))
        try:
            return service.save_uploaded_files(payloads)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def analyze_response(response: dict) -> AnalyzeResponse:
        result = response["result"]
        return AnalyzeResponse(
            thread_id=response["thread_id"],
            run_id=result["request"]["run_id"],
            llm_backend=response["llm_backend"],
            status=result["status"],
            stop_reason=result["stop_reason"],
            research_scope=result.get("scope"),
            coverage=result.get("coverage"),
            report=result["report"],
            evidence_bundle=result["bundle"],
            gaps=result.get("gaps", []),
            validation_issues=result.get("validation_issues", []),
            context_manifests=result.get("context_manifests", []),
            audit_events=result.get("audit_events", []),
            budget_usage=result.get("budget_usage", {}),
            artifacts=response["artifacts"],
            document_diagnostics=response.get("document_diagnostics", []),
            session_document_count=response.get("session_document_count", 0),
        )

    app = FastAPI(
        title="MAS Finance API",
        version="2.1.0",
        description="Evidence-first financial research API with a LangGraph planning agent.",
    )
    app.middleware("http")(request_logging_middleware)
    app.router.add_event_handler("shutdown", service.close)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        backend = "deepseek" if app_config.llm.api_key else "deterministic"
        return HealthResponse(status="ok", llm_backend=backend)

    @app.get("/api/v1/config")
    async def get_config(_: None = Depends(auth_dependency)) -> dict:
        return {
            # Never reflect filesystem layout or a database DSN: a DSN can
            # contain credentials and this endpoint is capability discovery,
            # not a raw settings dump.
            "database_backend": urlsplit(app_config.database_url).scheme or "unknown",
            "deepseek_model": app_config.llm.model,
            "deepseek_enabled": bool(app_config.llm.api_key),
            "api_key_enabled": bool(app_config.api_key),
            "redis_enabled": bool(app_config.redis_url),
            "market_data_provider": app_config.market_data_provider,
            "network_allowed": app_config.allow_network,
            "sec_enabled": bool(app_config.sec_user_agent),
            "fred_enabled": bool(app_config.fred_api_key),
            "conversation_memory_enabled": app_config.conversation_memory_enabled,
            "personal_memory_enabled": app_config.personal_memory_enabled,
            "personal_knowledge_enabled": app_config.personal_knowledge_enabled,
            "conversation_context_tokens": app_config.conversation_context_tokens,
            "conversation_recent_events": app_config.conversation_recent_events,
            "session_document_ttl_seconds": app_config.session_document_ttl_seconds,
            "max_session_document_sessions": app_config.max_session_document_sessions,
            "max_pdf_pages": app_config.max_pdf_pages,
            "max_pdf_text_characters": app_config.max_pdf_text_characters,
            "paddleocr_enabled": bool(app_config.paddleocr_access_token),
            "paddleocr_model": app_config.paddleocr_model,
            "web_search_enabled": bool(app_config.bocha_search_api_key or app_config.brave_search_api_key),
            "embedding_enabled": service.embedding_provider is not None,
            "embedding_model": service.embedding_provider.model_name if service.embedding_provider else None,
            "planning_evidence_characters": app_config.planning_evidence_characters,
            "synthesis_evidence_characters": app_config.synthesis_evidence_characters,
            "synthesis_output_tokens": app_config.synthesis_output_tokens,
        }

    @app.get("/api/v1/tools")
    async def get_tools(_: None = Depends(auth_dependency)) -> list[dict]:
        return service.describe_tools()

    @app.delete("/api/v1/conversations/{thread_id}")
    async def delete_conversation(thread_id: str, _: None = Depends(auth_dependency)) -> dict[str, int | str]:
        try:
            deleted = service.delete_conversation(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, **deleted}

    @app.post("/api/v1/memories", response_model=PersonalMemoryResponse, status_code=201)
    async def save_personal_memory(
        payload: PersonalMemoryCreate,
        _: None = Depends(auth_dependency),
    ) -> PersonalMemoryResponse:
        try:
            value = service.save_personal_memory(
                kind=payload.kind,
                title=payload.title,
                content=payload.content,
                tags=payload.tags,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PersonalMemoryResponse(**value)

    @app.get("/api/v1/memories", response_model=list[PersonalMemoryResponse])
    async def list_personal_memories(
        kind: PersonalMemoryKind | None = None,
        _: None = Depends(auth_dependency),
    ) -> list[PersonalMemoryResponse]:
        return [PersonalMemoryResponse(**item) for item in service.list_personal_memories(kind=kind)]

    @app.delete("/api/v1/memories/{memory_id}")
    async def delete_personal_memory(memory_id: str, _: None = Depends(auth_dependency)) -> dict[str, str | bool]:
        try:
            deleted = service.delete_personal_memory(memory_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"memory_id": memory_id, "deleted": deleted}

    @app.post("/api/v1/knowledge/documents", status_code=201)
    async def save_personal_documents(
        allow_network: bool = Form(default=False),
        files: list[UploadFile] = File(...),  # noqa: B008
        _: None = Depends(auth_dependency),
    ) -> dict:
        saved_paths = await persist_uploads(files)
        try:
            try:
                documents = service.ingest_personal_documents(saved_paths, allow_network=allow_network)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
        return {"documents": documents}

    @app.get("/api/v1/knowledge/documents")
    async def list_personal_documents(_: None = Depends(auth_dependency)) -> dict:
        return {"documents": service.list_personal_documents()}

    @app.delete("/api/v1/knowledge/documents/{document_id}")
    async def delete_personal_document(document_id: str, _: None = Depends(auth_dependency)) -> dict[str, str | bool]:
        try:
            deleted = service.delete_personal_document(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document_id": document_id, "deleted": deleted}

    @app.get("/api/v1/session-documents/{thread_id}")
    async def list_session_documents(thread_id: str, _: None = Depends(auth_dependency)) -> dict:
        try:
            documents = service.list_session_documents(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "thread_id": thread_id,
            "documents": documents,
        }

    @app.delete("/api/v1/session-documents/{thread_id}")
    async def delete_session_documents(thread_id: str, _: None = Depends(auth_dependency)) -> dict[str, int | str]:
        try:
            deleted = service.delete_session_documents(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, "deleted_documents": deleted}

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    async def analyze(payload: AnalyzeRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        try:
            response = service.analyze(
                query=payload.query,
                thread_id=payload.thread_id,
                export_artifacts=payload.export_artifacts,
                entities=payload.entities,
                symbols=payload.symbols,
                allow_network=payload.allow_network,
                macro_series=payload.macro_series,
                calculations=[item.model_dump(mode="json", exclude_none=True) for item in payload.calculations],
                require_documents=payload.require_documents,
                require_market_data=payload.require_market_data,
                require_market_history=payload.require_market_history,
                require_regulatory_data=payload.require_regulatory_data,
                market_history_range=payload.market_history_range,
                market_history_interval=payload.market_history_interval,
                use_session_documents=payload.use_session_documents,
                use_personal_memory=payload.use_personal_memory,
                use_personal_knowledge=payload.use_personal_knowledge,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return analyze_response(response)

    @app.post("/api/v1/analyze-upload", response_model=AnalyzeResponse)
    async def analyze_upload(
        query: str = Form(..., min_length=1, max_length=8_000, pattern=r".*\S.*"),
        thread_id: str | None = Form(
            default=None,
            min_length=1,
            max_length=200,
            pattern=r"^[^\x00-\x1f\x7f]+$",
        ),
        export_artifacts: bool = Form(default=True),
        entities: str = Form(default="", max_length=5_000),
        allow_network: bool = Form(default=False),
        use_session_documents: bool = Form(default=False),
        retain_for_session: bool = Form(default=False),
        use_personal_memory: bool = Form(default=True),
        use_personal_knowledge: bool = Form(default=True),
        files: list[UploadFile] = File(...),  # noqa: B008 - required FastAPI dependency declaration
        _: None = Depends(auth_dependency),
    ) -> AnalyzeResponse:
        saved_paths = await persist_uploads(files)
        try:
            try:
                response = service.analyze(
                    query=query,
                    thread_id=thread_id,
                    export_artifacts=export_artifacts,
                    document_paths=saved_paths,
                    entities=[item.strip() for item in entities.split(",") if item.strip()],
                    allow_network=allow_network,
                    use_session_documents=use_session_documents,
                    retain_documents_for_session=retain_for_session,
                    use_personal_memory=use_personal_memory,
                    use_personal_knowledge=use_personal_knowledge,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
        return analyze_response(response)

    @app.post("/api/v1/jobs", response_model=SubmitJobResponse, status_code=202)
    async def submit_job(
        payload: SubmitJobRequest,
        background_tasks: BackgroundTasks,
        _: None = Depends(auth_dependency),
    ) -> SubmitJobResponse:
        created = service.submit_job(payload.query, payload.thread_id)
        queued = service.enqueue_job(created["job_id"], payload.query, created["thread_id"], payload.export_artifacts)
        if not queued:
            background_tasks.add_task(
                run_job_in_thread,
                created["job_id"],
                payload.query,
                created["thread_id"],
                payload.export_artifacts,
            )
        return SubmitJobResponse(**created, queue_backend="redis" if queued else "background-task")

    @app.post("/api/v1/jobs/upload", response_model=SubmitJobResponse, status_code=202)
    async def submit_upload_job(
        background_tasks: BackgroundTasks,
        query: str = Form(..., min_length=1, max_length=8_000, pattern=r".*\S.*"),
        thread_id: str | None = Form(
            default=None,
            min_length=1,
            max_length=200,
            pattern=r"^[^\x00-\x1f\x7f]+$",
        ),
        export_artifacts: bool = Form(default=True),
        files: list[UploadFile] = File(...),  # noqa: B008 - required FastAPI dependency declaration
        _: None = Depends(auth_dependency),
    ) -> SubmitJobResponse:
        saved_paths = await persist_uploads(files)
        try:
            created = service.submit_job(query, thread_id)
            queued = service.enqueue_job(
                created["job_id"],
                query,
                created["thread_id"],
                export_artifacts,
                saved_paths,
                True,
            )
        except Exception:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
            raise
        if not queued:
            background_tasks.add_task(
                run_job_in_thread,
                created["job_id"],
                query,
                created["thread_id"],
                export_artifacts,
                saved_paths,
                True,
            )
        return SubmitJobResponse(**created, queue_backend="redis" if queued else "background-task")

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str, _: None = Depends(auth_dependency)) -> JobResponse:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JobResponse(**job)

    @app.get("/api/v1/jobs", response_model=list[JobResponse])
    async def list_jobs(
        limit: int = Query(default=20, ge=1, le=100),
        _: None = Depends(auth_dependency),
    ) -> list[JobResponse]:
        jobs = service.list_jobs(limit)
        return [JobResponse(**job) for job in jobs]

    return app


app = create_app()
