from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from ..memory_store import PersonalMemoryKind

QueryText = Annotated[str, Field(min_length=1, max_length=8_000, pattern=r".*\S.*")]
RunIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$"),
]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText = Field(description="Financial research question.")
    allow_network: bool = Field(default=False, description="Request network tools; server policy still applies.")
    thread_id: RunIdentifier | None = Field(default=None, description="Optional conversation thread id.")
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    use_session_documents: bool = Field(
        default=False,
        description="Retrieve PDFs explicitly retained for this thread; thread_id is required.",
    )
    use_personal_memory: bool = Field(
        default=True,
        description="Inject all active personal profile/preference/experience context; never financial evidence.",
    )
    use_personal_knowledge: bool = Field(
        default=True,
        description="Make explicitly persisted personal documents available to the planner.",
    )


class AnalyzeResponse(BaseModel):
    thread_id: str
    run_id: str
    llm_backend: str
    status: str
    stop_reason: str
    research_scope: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    report: str
    evidence_bundle: dict[str, Any]
    gaps: list[dict[str, Any]]
    validation_issues: list[dict[str, Any]]
    context_manifests: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    budget_usage: dict[str, int]
    artifacts: dict[str, str]
    document_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    session_document_count: int = 0


class HealthResponse(BaseModel):
    status: str
    llm_backend: str


class PersonalMemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PersonalMemoryKind
    title: str = Field(min_length=1, max_length=200, pattern=r".*\S.*")
    content: str = Field(min_length=1, max_length=8_000, pattern=r".*\S.*")
    tags: list[str] = Field(default_factory=list, max_length=20)


class PersonalMemoryResponse(BaseModel):
    memory_id: str
    kind: PersonalMemoryKind
    title: str
    content: str
    tags: list[str]
    created_at: str
    updated_at: str


class SubmitJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText = Field(description="User query for asynchronous financial analysis.")
    thread_id: RunIdentifier | None = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether the background job should export files.")
    allow_network: bool = Field(default=False, description="Request network tools; server policy still applies.")
    use_session_documents: bool = Field(default=False)
    use_personal_memory: bool = Field(default=True)
    use_personal_knowledge: bool = Field(default=True)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200, pattern=r".*\S.*")


class SubmitJobResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str
    queue_backend: str | None = None


class JobResponse(BaseModel):
    job_id: str
    thread_id: str
    query: str
    status: str
    llm_backend: str | None = None
    result: dict[str, Any] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str
