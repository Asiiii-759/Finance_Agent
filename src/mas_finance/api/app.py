from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..config import AppConfig
from ..documents import PDFDocumentParser
from ..embeddings import EmbeddingProvider
from ..harness import Tool
from ..llm import BaseLLMClient
from ..logging_utils import configure_logging, request_logging_middleware
from ..memory_store import ConversationSummarizer, PersonalMemoryKind, TokenCounter
from ..retrieval import RetrievalSource
from ..service import FinanceAnalysisService
from .auth import Principal, build_api_key_dependency
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
    llm_client: BaseLLMClient | None = None,
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
        llm_client=llm_client,
    )
    auth_dependency = build_api_key_dependency(
        app_config.api_key,
        Principal(app_config.local_tenant_id, app_config.local_user_id),
    )

    def run_job_in_process(job_id: str) -> None:
        from ..worker import process_one_isolated_job

        process_one_isolated_job(service, f"api-background-{uuid4().hex[:12]}", job_id=job_id)

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
    web_dir = Path(__file__).resolve().parents[1] / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def web_app() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        backend = "deepseek" if app_config.llm.api_key else "missing"
        return HealthResponse(status="ok", llm_backend=backend)

    @app.get("/api/v1/config")
    async def get_config(principal: Principal = Depends(auth_dependency)) -> dict:  # noqa: B008
        return {
            # Never reflect filesystem layout or a database DSN: a DSN can
            # contain credentials and this endpoint is capability discovery,
            # not a raw settings dump.
            "database_backend": urlsplit(app_config.database_url).scheme or "unknown",
            "principal": {"tenant_id": principal.tenant_id, "user_id": principal.user_id},
            "deepseek_model": app_config.llm.model,
            "deepseek_enabled": bool(app_config.llm.api_key),
            "api_key_enabled": bool(app_config.api_key),
            "market_data_provider": app_config.market_data_provider,
            "network_allowed": app_config.allow_network,
            "sec_enabled": bool(app_config.sec_user_agent),
            "fred_enabled": bool(app_config.fred_api_key),
            "conversation_memory_enabled": app_config.conversation_memory_enabled,
            "personal_memory_enabled": app_config.personal_memory_enabled,
            "automatic_memory_consolidation_enabled": app_config.automatic_memory_consolidation_enabled,
            "automatic_skill_learning_enabled": app_config.automatic_skill_learning_enabled,
            "user_profile_configured": app_config.user_profile_path is not None,
            "personal_knowledge_enabled": app_config.personal_knowledge_enabled,
            "conversation_context_tokens": app_config.conversation_context_tokens,
            "conversation_recent_tokens": app_config.conversation_recent_tokens,
            "model_input_token_budget": app_config.model_input_token_budget,
            "model_output_token_budget": app_config.model_output_token_budget,
            "job_queue_backend": "database_lease",
            "job_lease_seconds": app_config.job_lease_seconds,
            "job_max_attempts": app_config.job_max_attempts,
            "job_execution_boundary": "isolated_process",
            "operational_retention_days": app_config.operational_retention_days,
            "completed_job_retention_days": app_config.completed_job_retention_days,
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
    async def get_tools(_principal: Principal = Depends(auth_dependency)) -> list[dict]:  # noqa: B008
        return service.describe_tools()

    @app.get("/api/v1/conversations")
    async def get_conversations(
        limit: int = Query(default=100, ge=1, le=500),
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        return {
            "conversations": service.list_conversations(
                limit=limit,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        }

    @app.get("/api/v1/conversations/{thread_id}/messages")
    async def get_conversation_messages(
        thread_id: str,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=500),
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        try:
            messages = service.list_conversation_messages(
                thread_id,
                after_sequence=after_sequence,
                limit=limit,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "thread_id": thread_id,
            "messages": messages,
            "next_sequence": messages[-1]["sequence"] if messages else after_sequence,
        }

    @app.get("/api/v1/conversations/{thread_id}/runs")
    async def get_conversation_runs(
        thread_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        try:
            runs = service.list_conversation_runs(
                thread_id,
                limit=limit,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, "runs": runs}

    @app.get("/api/v1/conversations/{thread_id}/runs/{run_id}/logs")
    async def get_run_logs(
        thread_id: str,
        run_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        try:
            events = service.list_run_logs(
                thread_id,
                run_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, "run_id": run_id, "events": events}

    @app.get("/api/v1/conversations/{thread_id}/runs/{run_id}")
    async def get_conversation_run(
        thread_id: str,
        run_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        try:
            run = service.get_conversation_run(
                thread_id,
                run_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Conversation run not found.")
        return {"thread_id": thread_id, **run}

    @app.get("/api/v1/skills")
    async def get_learned_skills(principal: Principal = Depends(auth_dependency)) -> dict:  # noqa: B008
        return {
            "skills": service.list_learned_skills(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        }

    @app.delete("/api/v1/skills/{skill_id}")
    async def delete_learned_skill(
        skill_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, str | bool]:
        try:
            deleted = service.delete_learned_skill(
                skill_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"skill_id": skill_id, "deleted": deleted}

    @app.delete("/api/v1/conversations/{thread_id}")
    async def delete_conversation(
        thread_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, int | str]:
        try:
            deleted = service.delete_conversation(
                thread_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, **deleted}

    @app.post("/api/v1/memories", response_model=PersonalMemoryResponse, status_code=201)
    async def save_personal_memory(
        payload: PersonalMemoryCreate,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> PersonalMemoryResponse:
        try:
            value = service.save_personal_memory(
                kind=payload.kind,
                title=payload.title,
                content=payload.content,
                tags=payload.tags,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PersonalMemoryResponse(**value)

    @app.get("/api/v1/memories", response_model=list[PersonalMemoryResponse])
    async def list_personal_memories(
        kind: PersonalMemoryKind | None = None,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> list[PersonalMemoryResponse]:
        return [
            PersonalMemoryResponse(**item)
            for item in service.list_personal_memories(
                kind=kind,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        ]

    @app.delete("/api/v1/memories/{memory_id}")
    async def delete_personal_memory(
        memory_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, str | bool]:
        try:
            deleted = service.delete_personal_memory(
                memory_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"memory_id": memory_id, "deleted": deleted}

    @app.post("/api/v1/knowledge/documents", status_code=201)
    async def save_personal_documents(
        allow_network: bool = Form(default=False),
        files: list[UploadFile] = File(...),  # noqa: B008
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        saved_paths = await persist_uploads(files)
        try:
            try:
                documents = service.ingest_personal_documents(
                    saved_paths,
                    allow_network=allow_network,
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
        return {"documents": documents}

    @app.get("/api/v1/knowledge/documents")
    async def list_personal_documents(principal: Principal = Depends(auth_dependency)) -> dict:  # noqa: B008
        return {
            "documents": service.list_personal_documents(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        }

    @app.delete("/api/v1/knowledge/documents/{document_id}")
    async def delete_personal_document(
        document_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, str | bool]:
        try:
            deleted = service.delete_personal_document(
                document_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document_id": document_id, "deleted": deleted}

    @app.get("/api/v1/session-documents/{thread_id}")
    async def list_session_documents(
        thread_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict:
        try:
            documents = service.list_session_documents(
                thread_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "thread_id": thread_id,
            "documents": documents,
        }

    @app.delete("/api/v1/session-documents/{thread_id}")
    async def delete_session_documents(
        thread_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, int | str]:
        try:
            deleted = service.delete_session_documents(
                thread_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, "deleted_documents": deleted}

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    async def analyze(
        payload: AnalyzeRequest,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> AnalyzeResponse:
        try:
            response = await run_in_threadpool(
                service.analyze,
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
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> AnalyzeResponse:
        saved_paths = await persist_uploads(files)
        try:
            try:
                response = await run_in_threadpool(
                    service.analyze,
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
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
        return analyze_response(response)

    @app.post("/api/v1/jobs", response_model=SubmitJobResponse, status_code=202)
    async def submit_job(
        payload: SubmitJobRequest,
        background_tasks: BackgroundTasks,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> SubmitJobResponse:
        created = service.submit_job(
            payload.query,
            payload.thread_id,
            export_artifacts=payload.export_artifacts,
            idempotency_key=payload.idempotency_key,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            allow_network=payload.allow_network,
            use_session_documents=payload.use_session_documents,
            use_personal_memory=payload.use_personal_memory,
            use_personal_knowledge=payload.use_personal_knowledge,
        )
        if created["created"]:
            background_tasks.add_task(run_job_in_process, created["job_id"])
        return SubmitJobResponse(**created, queue_backend="database-lease")

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
        allow_network: bool = Form(default=False),
        retain_for_session: bool = Form(default=False),
        idempotency_key: str | None = Form(default=None, min_length=1, max_length=200),
        files: list[UploadFile] = File(...),  # noqa: B008 - required FastAPI dependency declaration
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> SubmitJobResponse:
        saved_paths = await persist_uploads(files)
        try:
            created = service.submit_job(
                query,
                thread_id,
                export_artifacts=export_artifacts,
                document_paths=saved_paths,
                cleanup_documents=True,
                idempotency_key=idempotency_key,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                allow_network=allow_network,
                use_personal_memory=True,
                use_personal_knowledge=True,
                retain_documents_for_session=retain_for_session,
            )
        except Exception:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
            raise
        if created["created"]:
            background_tasks.add_task(run_job_in_process, created["job_id"])
        else:
            for saved_path in saved_paths:
                Path(saved_path).unlink(missing_ok=True)
        return SubmitJobResponse(**created, queue_backend="database-lease")

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    async def get_job(
        job_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> JobResponse:
        job = service.get_job(
            job_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JobResponse(**job)

    @app.delete("/api/v1/jobs/{job_id}")
    async def cancel_job(
        job_id: str,
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> dict[str, str]:
        status = service.cancel_job(
            job_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"job_id": job_id, "status": status}

    @app.get("/api/v1/jobs", response_model=list[JobResponse])
    async def list_jobs(
        limit: int = Query(default=20, ge=1, le=100),
        principal: Principal = Depends(auth_dependency),  # noqa: B008
    ) -> list[JobResponse]:
        jobs = service.list_jobs(
            limit,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        return [JobResponse(**job) for job in jobs]

    return app


app = create_app()
