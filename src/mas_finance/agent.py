"""Financial research state, planning and reporting domain objects."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .contracts import Claim, ClaimStatus, EvidenceBundle, SourceType, stable_id
from .harness import ToolSpec
from .research import ResearchRequirement, ResearchScope


class AgentPhase(StrEnum):
    INTENT = "intent"
    PLANNING = "planning"
    VALIDATING = "validating"
    FINAL_GENERATION = "final_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(StrEnum):
    CLARIFICATION_REQUIRED = "clarification_required"
    COVERAGE_SATISFIED = "coverage_satisfied"
    MAX_ITERATIONS = "max_iterations"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    NO_AVAILABLE_ACTION = "no_available_action"
    VALIDATION_FAILED = "validation_failed"
    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ChatAttachment:
    document_id: str
    title: str
    media_type: str = "application/pdf"

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("document_id", self.document_id, 256),
            ("title", self.title, 255),
            ("media_type", self.media_type, 100),
        ):
            if not value.strip() or len(value) > limit or any(ord(item) < 32 or ord(item) == 127 for item in value):
                raise ValueError(f"attachment {name} is invalid")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChatAttachment:
        return cls(
            document_id=str(value["document_id"]),
            title=str(value["title"]),
            media_type=str(value.get("media_type") or "application/pdf"),
        )


@dataclass(frozen=True)
class ChatTurn:
    message: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    thread_id: str = "thread"
    run_id: str = "run"
    allow_network: bool = False
    attachments: tuple[ChatAttachment, ...] = ()

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("message is required")
        if len(self.message) > 8_000:
            raise ValueError("message cannot exceed 8000 characters")
        if len(self.attachments) > 20 or any(not isinstance(item, ChatAttachment) for item in self.attachments):
            raise ValueError("attachments must contain at most 20 attachment references")
        for name, value in (
            ("tenant_id", self.tenant_id),
            ("user_id", self.user_id),
            ("thread_id", self.thread_id),
            ("run_id", self.run_id),
        ):
            if not value.strip() or len(value) > 200 or any(ord(item) < 32 or ord(item) == 127 for item in value):
                raise ValueError(f"{name} is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChatTurn:
        return cls(
            message=str(value["message"]),
            tenant_id=str(value["tenant_id"]),
            user_id=str(value["user_id"]),
            thread_id=str(value["thread_id"]),
            run_id=str(value["run_id"]),
            allow_network=bool(value["allow_network"]),
            attachments=tuple(ChatAttachment.from_dict(item) for item in value.get("attachments") or ()),
        )


@dataclass(frozen=True)
class RuntimePolicy:
    max_iterations: int = 3
    max_tool_calls: int = 12
    max_network_calls: int = 8
    max_model_calls: int = 8
    max_model_input_tokens: int = 300_000
    max_model_output_tokens: int = 32_768
    max_parallel_tool_calls: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 8:
            raise ValueError("max_iterations must be between 1 and 8")
        if not 1 <= self.max_tool_calls <= 100:
            raise ValueError("max_tool_calls must be between 1 and 100")
        if not 0 <= self.max_network_calls <= self.max_tool_calls:
            raise ValueError("max_network_calls must be between zero and max_tool_calls")
        if not 0 <= self.max_model_calls <= 20:
            raise ValueError("max_model_calls must be between zero and twenty")
        if not 16_000 <= self.max_model_input_tokens <= 1_000_000:
            raise ValueError("max_model_input_tokens must be between 16000 and 1000000")
        if not 1_024 <= self.max_model_output_tokens <= 200_000:
            raise ValueError("max_model_output_tokens must be between 1024 and 200000")
        if not 1 <= self.max_parallel_tool_calls <= 8:
            raise ValueError("max_parallel_tool_calls must be between 1 and 8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimePolicy:
        return cls(
            max_iterations=int(value["max_iterations"]),
            max_tool_calls=int(value["max_tool_calls"]),
            max_network_calls=int(value["max_network_calls"]),
            max_model_calls=int(value["max_model_calls"]),
            max_model_input_tokens=int(value["max_model_input_tokens"]),
            max_model_output_tokens=int(value["max_model_output_tokens"]),
            max_parallel_tool_calls=int(value["max_parallel_tool_calls"]),
        )


@dataclass(frozen=True)
class AgentContext:
    thread_context: Mapping[str, Any] = field(default_factory=dict)
    personal_context: tuple[Mapping[str, Any], ...] = ()
    skill_index: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if len(self.thread_context) > 20:
            raise ValueError("thread_context contains too many fields")
        try:
            thread_payload = json.dumps(
                dict(self.thread_context),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("thread_context must be JSON serializable") from exc
        if len(thread_payload) > 2_000_000:
            raise ValueError("thread_context exceeds length limit")
        if len(self.personal_context) > 501 or any(not isinstance(item, Mapping) for item in self.personal_context):
            raise ValueError("personal_context must contain at most 501 objects")
        try:
            personal_payload = json.dumps(
                [dict(item) for item in self.personal_context],
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("personal_context must be JSON serializable") from exc
        if len(personal_payload) > 110_000:
            raise ValueError("personal_context exceeds length limit")
        if len(self.skill_index) > 100 or any(not isinstance(item, Mapping) for item in self.skill_index):
            raise ValueError("skill_index must contain at most 100 objects")
        skill_payload = json.dumps([dict(item) for item in self.skill_index], ensure_ascii=False, allow_nan=False)
        if len(skill_payload) > 20_000:
            raise ValueError("skill_index exceeds length limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_context": dict(self.thread_context),
            "personal_context": [dict(item) for item in self.personal_context],
            "skill_index": [dict(item) for item in self.skill_index],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentContext:
        return cls(
            thread_context=dict(value["thread_context"]),
            personal_context=tuple(dict(item) for item in value["personal_context"]),
            skill_index=tuple(dict(item) for item in value["skill_index"]),
        )


@dataclass(frozen=True)
class ToolTask:
    task_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    reason: str
    category: str
    entity: str | None = None
    requirement_key: str | None = None

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        reason: str,
        category: str,
        entity: str | None = None,
        requirement_key: str | None = None,
    ) -> ToolTask:
        task_id = stable_id("task", {"tool": tool_name, "arguments": dict(arguments)})
        return cls(task_id, tool_name, dict(arguments), reason, category, entity, requirement_key)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolTask:
        return cls(
            task_id=str(value["task_id"]),
            tool_name=str(value["tool_name"]),
            arguments=dict(value.get("arguments") or {}),
            reason=str(value.get("reason") or ""),
            category=str(value.get("category") or "other"),
            entity=str(value["entity"]) if value.get("entity") is not None else None,
            requirement_key=(str(value["requirement_key"]) if value.get("requirement_key") is not None else None),
        )


@dataclass(frozen=True)
class ResearchPlan:
    iteration: int
    rationale: str
    tasks: tuple[ToolTask, ...]
    ready_for_validation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "rationale": self.rationale,
            "tasks": [task.to_dict() for task in self.tasks],
            "ready_for_validation": self.ready_for_validation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchPlan:
        return cls(
            iteration=int(value["iteration"]),
            rationale=str(value.get("rationale") or ""),
            tasks=tuple(ToolTask.from_dict(item) for item in value.get("tasks") or ()),
            ready_for_validation=bool(value.get("ready_for_validation", False)),
        )


@dataclass(frozen=True)
class ToolObservation:
    task: ToolTask
    iteration: int
    result: Mapping[str, Any]
    network_access: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "iteration": self.iteration,
            "result": dict(self.result),
            "network_access": self.network_access,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolObservation:
        return cls(
            task=ToolTask.from_dict(value["task"]),
            iteration=int(value["iteration"]),
            result=dict(value.get("result") or {}),
            network_access=bool(value.get("network_access", False)),
        )


@dataclass(frozen=True)
class ResearchGap:
    code: str
    message: str
    entity: str | None = None
    requirement_key: str | None = None
    tool_name: str | None = None
    task_id: str | None = None
    resolvable: bool = False
    resolved: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_.:-]{1,100}", self.code):
            raise ValueError("invalid research gap code")
        if not self.message or len(self.message) > 2_000:
            raise ValueError("invalid research gap message")
        for name, value, limit in (
            ("entity", self.entity, 200),
            ("requirement_key", self.requirement_key, 300),
            ("tool_name", self.tool_name, 200),
            ("task_id", self.task_id, 200),
        ):
            if value is not None and (not value or len(value) > limit):
                raise ValueError(f"invalid research gap {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchGap:
        return cls(
            code=str(value.get("code") or "data_gap"),
            message=str(value.get("message") or "Data is incomplete."),
            entity=str(value["entity"]) if value.get("entity") is not None else None,
            requirement_key=(str(value["requirement_key"]) if value.get("requirement_key") is not None else None),
            tool_name=str(value["tool_name"]) if value.get("tool_name") is not None else None,
            task_id=str(value["task_id"]) if value.get("task_id") is not None else None,
            resolvable=bool(value.get("resolvable", False)),
            resolved=bool(value.get("resolved", False)),
        )


@dataclass(frozen=True)
class CoverageDecision:
    complete: bool
    missing: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"complete": self.complete, "missing": list(self.missing), "rationale": self.rationale}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoverageDecision:
        return cls(
            complete=bool(value.get("complete", False)),
            missing=tuple(str(item) for item in value.get("missing") or ()),
            rationale=str(value.get("rationale") or ""),
        )


@dataclass
class ResearchState:
    turn: ChatTurn
    runtime_policy: RuntimePolicy
    context: AgentContext = field(default_factory=AgentContext)
    scope: ResearchScope | None = None
    task_frame: dict[str, Any] | None = None
    phase: AgentPhase = AgentPhase.INTENT
    iteration: int = 0
    plans: list[ResearchPlan] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    gaps: list[ResearchGap] = field(default_factory=list)
    coverage: CoverageDecision | None = None
    report: str = ""
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    context_manifests: list[dict[str, Any]] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: StopReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "turn": self.turn.to_dict(),
            "runtime_policy": self.runtime_policy.to_dict(),
            "context": self.context.to_dict(),
            "scope": self.scope.to_dict() if self.scope else None,
            "task_frame": dict(self.task_frame) if self.task_frame else None,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "plans": [plan.to_dict() for plan in self.plans],
            "observations": [observation.to_dict() for observation in self.observations],
            "bundle": self.bundle.to_dict(),
            "gaps": [gap.to_dict() for gap in self.gaps],
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "report": self.report,
            "validation_issues": list(self.validation_issues),
            "context_manifests": list(self.context_manifests),
            "audit_events": list(self.audit_events),
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchState:
        if int(value.get("schema_version", 0)) != 6:
            raise ValueError("unsupported checkpoint schema")
        return cls(
            turn=ChatTurn.from_dict(value["turn"]),
            runtime_policy=RuntimePolicy.from_dict(value["runtime_policy"]),
            context=AgentContext.from_dict(value["context"]),
            scope=ResearchScope.from_dict(value["scope"]) if value.get("scope") else None,
            task_frame=dict(value["task_frame"]) if value.get("task_frame") else None,
            phase=AgentPhase(str(value["phase"])),
            iteration=int(value.get("iteration", 0)),
            plans=[ResearchPlan.from_dict(item) for item in value.get("plans") or ()],
            observations=[ToolObservation.from_dict(item) for item in value.get("observations") or ()],
            bundle=EvidenceBundle.from_dict(value.get("bundle") or {}),
            gaps=[ResearchGap.from_dict(item) for item in value.get("gaps") or ()],
            coverage=(CoverageDecision.from_dict(value["coverage"]) if value.get("coverage") else None),
            report=str(value.get("report") or ""),
            validation_issues=list(value.get("validation_issues") or ()),
            context_manifests=[dict(item) for item in value.get("context_manifests") or ()],
            audit_events=[dict(item) for item in value.get("audit_events") or ()],
            stop_reason=StopReason(str(value["stop_reason"])) if value.get("stop_reason") else None,
        )


class Planner(Protocol):
    def plan(self, state: ResearchState, available_tools: Mapping[str, ToolSpec]) -> ResearchPlan: ...


class Synthesizer(Protocol):
    def synthesize(
        self,
        turn: ChatTurn,
        runtime_policy: RuntimePolicy,
        context: AgentContext,
        bundle: EvidenceBundle,
        *,
        research_context: Mapping[str, Any] | None = None,
    ) -> Sequence[Claim]: ...


class CoverageAssessor:
    def assess(
        self,
        turn: ChatTurn,
        bundle: EvidenceBundle,
        scope: ResearchScope | None = None,
    ) -> CoverageDecision:
        if scope is None:
            raise ValueError("coverage assessment requires a model-created research scope")
        missing: list[str] = []
        for requirement in scope.requirements:
            if requirement.category == "knowledge":
                # 概念定义由模型自身判断，不强制检索词条。
                continue
            candidates = self._candidates(requirement, bundle)
            fields = {item.field_name.casefold() for item in candidates if item.field_name}
            if requirement.category == "calculation":
                fields.update(
                    str(item.source.metadata["operation"]).casefold()
                    for item in candidates
                    if item.source.metadata.get("operation")
                )
            distinct_documents = {
                item.source.metadata.get("document_id")
                or item.source.metadata.get("corpus_record_id")
                or item.source.metadata.get("file_name")
                or item.source.title
                for item in candidates
            }
            minimum_documents = int(requirement.parameters.get("minimum_documents", 1))
            multi_document_floor_missing = (
                requirement.category == "document" and len(distinct_documents) < minimum_documents
            )
            if (
                not candidates
                or any(field.casefold() not in fields for field in requirement.fields)
                or multi_document_floor_missing
            ):
                missing.append(requirement.key)
        return CoverageDecision(
            complete=not missing,
            missing=tuple(missing),
            rationale=(
                "All required evidence classes are covered." if not missing else "Required evidence remains missing."
            ),
        )

    @staticmethod
    def _candidates(requirement: ResearchRequirement, bundle: EvidenceBundle) -> list[Any]:
        entity = (requirement.entity or "").casefold()
        candidates = [
            item for item in bundle.evidence.values() if not entity or (item.entity or "").casefold() == entity
        ]
        if requirement.category == "document":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.DOCUMENT
            ]
        if requirement.category == "market":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.MARKET_DATA
                and "market_history" not in item.tags
                and item.source.as_of
            ]
        if requirement.category == "market_history":
            return [item for item in candidates if "market_history" in item.tags and item.source.as_of]
        if requirement.category == "regulatory":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.REGULATORY_FILING and item.source.as_of
            ]
        if requirement.category == "filings":
            return [item for item in candidates if "filing_metadata" in item.tags and item.source.published_at]
        if requirement.category == "macro":
            return [
                item for item in candidates if item.source.source_type == SourceType.MACRO_DATA and item.source.as_of
            ]
        if requirement.category == "calculation":
            request_id = str(requirement.parameters.get("request_id") or "")
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.CALCULATION
                and (not request_id or request_id in item.tags)
            ]
        if requirement.category == "derived_metric":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.CALCULATION and item.field_name in requirement.fields
            ]
        if requirement.category == "web":
            web = [item for item in candidates if item.source.source_type == SourceType.WEB]
            if any(item.source.metadata.get("quality_tier") == "public_authority" for item in web):
                return web
            domains = {item.source.metadata.get("domain") for item in web if item.source.metadata.get("domain")}
            return web if len(domains) >= 2 else []
        return []


def _claim_requires_degraded_status(claim: Claim) -> bool:
    if claim.status == ClaimStatus.SUPPORTED:
        return False
    return not (claim.status == ClaimStatus.INFERRED and not claim.evidence_ids)


@dataclass(frozen=True)
class ResearchOutcome:
    state: ResearchState
    audit_events: tuple[Mapping[str, Any], ...]
    budget_usage: Mapping[str, int]

    @property
    def status(self) -> str:
        if self.state.stop_reason is StopReason.CLARIFICATION_REQUIRED:
            return "needs_clarification"
        if self.state.phase == AgentPhase.FAILED or self.state.stop_reason is StopReason.NO_EVIDENCE:
            return "failed"
        if not self.state.bundle.claims and not self.state.bundle.evidence:
            return "failed"
        has_degrading_claims = any(
            _claim_requires_degraded_status(claim) for claim in self.state.bundle.claims.values()
        )
        if (
            any(not gap.resolved for gap in self.state.gaps)
            or has_degrading_claims
            or (self.state.coverage and not self.state.coverage.complete)
        ):
            return "degraded"
        return "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            **self.state.to_dict(),
            "audit_events": list(self.audit_events),
            "budget_usage": dict(self.budget_usage),
        }


def render_report(state: ResearchState) -> str:
    turn = state.turn
    runtime_policy = state.runtime_policy
    incomplete = bool(state.coverage and state.coverage.missing) or state.stop_reason in {
        StopReason.NO_EVIDENCE,
        StopReason.INSUFFICIENT_EVIDENCE,
        StopReason.MAX_ITERATIONS,
        StopReason.NO_AVAILABLE_ACTION,
        StopReason.TOOL_BUDGET_EXHAUSTED,
    }
    lines = [
        "# Evidence-first financial research report / 证据优先金融研究报告",
        "",
        f"Query: {_report_text(turn.message)}",
        "",
        "## Answerability / 可回答性",
        (
            "- 当前证据不足以完整回答该问题；以下只保留已取得的材料、明确缺口和可核验内容。"
            if incomplete
            else "- 当前答案已达到本轮声明的证据要求。"
        ),
        "",
        "## Research strategy / 研究策略",
    ]
    if state.scope:
        lines.append(f"- Intents: {', '.join(item.value for item in state.scope.intents)}")
        lines.append(
            "- Requirements: " + (", ".join(_report_text(item.key) for item in state.scope.requirements) or "none")
        )
        lines.append(f"- Decision summary: {_report_text(state.scope.rationale)}")
    else:
        lines.append("- No structured research scope was recorded.")
    lines.extend(
        [
            "",
            "## Supported findings / 已证实结论",
        ]
    )
    supported = [claim for claim in state.bundle.claims.values() if claim.status == ClaimStatus.SUPPORTED]
    if supported:
        for claim in supported:
            citations = " ".join(f"[^{item}]" for item in claim.evidence_ids)
            lines.append(f"- {_report_text(claim.text)} {citations}")
    else:
        lines.append("- No structured finding could be supported by the available evidence.")

    lines.extend(["", "## Conflicts and caveats / 冲突与限定"])
    qualified = [claim for claim in state.bundle.claims.values() if claim.status != ClaimStatus.SUPPORTED]
    if qualified:
        for claim in qualified:
            citations = " ".join(f"[^{item}]" for item in claim.evidence_ids)
            lines.append(
                f"- [{claim.status.value}] {_report_text(claim.text)} "
                f"Caveat: {_report_text(claim.caveat or '')} {citations}"
            )
    else:
        lines.append("- No conflicting, inferred, or unsupported claims were emitted.")

    lines.extend(["", "## Retrieved document evidence / 文档证据"])
    documents = [item for item in state.bundle.evidence.values() if item.source.source_type == SourceType.DOCUMENT]
    if documents:
        for item in documents:
            lines.append(f"- {_report_text(item.content)[:280]} [^{item.evidence_id}]")
    else:
        lines.append("- No document evidence was retrieved.")

    lines.extend(["", "## Data gaps / 数据缺口"])
    if state.gaps:
        for gap in _unique_gaps(state.gaps):
            resolution = " (resolved by fallback evidence)" if gap.resolved else ""
            subject = gap.entity or gap.requirement_key or "run"
            lines.append(f"- [{gap.code}] {_report_text(subject)}: {_report_text(gap.message)}{resolution}")
    else:
        lines.append("- No tool or data gaps were reported.")
    if state.coverage and state.coverage.missing:
        lines.append(
            "- [coverage_incomplete] Missing: " + ", ".join(_report_text(item) for item in state.coverage.missing)
        )

    lines.extend(["", "## Run controls / 运行控制"])
    lines.append(f"- Iterations: {state.iteration}/{runtime_policy.max_iterations}")
    research_tool_calls = sum(
        1
        for item in state.audit_events
        if item.get("capability") != "model.generate" and int(item.get("attempts") or 0) > 0
    )
    network_attempts = sum(
        int(item.get("network_attempts", item.get("attempts")) or 0)
        for item in state.audit_events
        if item.get("network_access") and item.get("capability") != "model.generate"
    )
    model_calls = sum(
        1
        for item in state.audit_events
        if item.get("capability") == "model.generate" and int(item.get("attempts") or 0) > 0
    )
    lines.append(f"- Research tool calls: {research_tool_calls}/{runtime_policy.max_tool_calls}")
    lines.append(f"- Data-network attempts: {network_attempts}/{runtime_policy.max_network_calls}")
    lines.append(f"- Model calls: {model_calls}/{runtime_policy.max_model_calls}")
    lines.append(f"- Parallel tool calls per plan: {runtime_policy.max_parallel_tool_calls}")
    lines.append(f"- Harness audit events: {len(state.audit_events)}")
    lines.append(f"- Stop reason: {(state.stop_reason or StopReason.NO_AVAILABLE_ACTION).value}")

    lines.extend(["", "## Sources / 来源"])
    for item in state.bundle.evidence.values():
        input_ids = item.source.metadata.get("input_evidence_ids") or []
        input_suffix = f"; inputs={_report_text(','.join(str(value) for value in input_ids))}" if input_ids else ""
        lines.append(
            f"[^{item.evidence_id}]: {_report_text(item.source.title)}; "
            f"{_report_text(item.source.provider)}; {_report_text(item.source.locator)}; "
            f"as_of={_report_text(item.source.as_of or 'unknown')}{input_suffix}"
        )
    lines.extend(
        [
            "",
            "## Risk notice / 风险提示",
            (
                "This report is generated for research-system evaluation and does not constitute investment advice. "
                "本报告用于研究系统评估，不构成投资建议。"
            ),
        ]
    )
    return "\n".join(lines)


def _unique_gaps(gaps: Sequence[ResearchGap]) -> list[ResearchGap]:
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    result: list[ResearchGap] = []
    for gap in gaps:
        identity = (gap.code, gap.entity, gap.requirement_key, gap.tool_name)
        if identity not in seen:
            seen.add(identity)
            result.append(gap)
    return result


def reconcile_conflicts(bundle: EvidenceBundle, claims: Sequence[Claim]) -> list[Claim]:
    """Replace affirmative claims over conflicting structured facts with explicit conflicts."""
    groups: dict[tuple[str | None, str, str | None, str | None, str | None], list[str]] = {}
    for item in bundle.evidence.values():
        if item.field_name and item.value is not None:
            calculation_request = (
                str(item.source.metadata.get("request_id"))
                if item.source.source_type == SourceType.CALCULATION and item.source.metadata.get("request_id")
                else None
            )
            key = (item.entity, item.field_name, item.period, item.unit, calculation_request)
            groups.setdefault(key, []).append(item.evidence_id)
    conflicting = {
        key: evidence_ids
        for key, evidence_ids in groups.items()
        if len({bundle.evidence[item].value for item in evidence_ids}) > 1
    }
    conflicted_ids = {item for evidence_ids in conflicting.values() for item in evidence_ids}
    reconciled = [claim for claim in claims if not conflicted_ids.intersection(claim.evidence_ids)]
    for (entity, field_name, period, unit, _calculation_request), evidence_ids in conflicting.items():
        values = [bundle.evidence[item].value for item in evidence_ids]
        reconciled.append(
            Claim.create(
                text=(
                    f"Conflicting values for {entity or 'unknown entity'} {field_name} "
                    f"({period or 'unknown period'}): {values} {unit or ''}."
                ),
                status=ClaimStatus.CONFLICTED,
                evidence_ids=tuple(evidence_ids),
                caveat="Sources disagree; no single value was selected and derived calculations were suppressed.",
            )
        )
    return reconciled


def _report_text(value: str) -> str:
    normalized = " ".join(str(value).split())
    return (
        normalized.replace("\\", "\\\\")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _safe_gap_code(value: Any) -> str:
    normalized = str(value or "data_gap").casefold()
    return normalized if re.fullmatch(r"[a-z0-9_.:-]{1,100}", normalized) else "data_gap"


def _merge_audit_events(
    existing: Sequence[Mapping[str, Any]],
    live: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in (*existing, *live):
        call_id = str(item.get("call_id") or "")
        if call_id:
            merged[call_id] = dict(item)
    return sorted(
        merged.values(),
        key=lambda item: int(str(item["call_id"]).rsplit(":", 1)[-1]),
    )
