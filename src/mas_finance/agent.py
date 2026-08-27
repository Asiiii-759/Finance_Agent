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
from .research import FinancialQueryAnalyzer, ResearchRequirement, ResearchScope, validate_macro_series


class AgentPhase(StrEnum):
    INTENT = "intent"
    PLANNING = "planning"
    VALIDATING = "validating"
    FINAL_GENERATION = "final_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(StrEnum):
    COVERAGE_SATISFIED = "coverage_satisfied"
    MAX_ITERATIONS = "max_iterations"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    NO_AVAILABLE_ACTION = "no_available_action"
    VALIDATION_FAILED = "validation_failed"
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class ResearchRequest:
    query: str
    entities: tuple[str, ...] = ()
    symbols: Mapping[str, str] = field(default_factory=dict)
    tenant_id: str = "default"
    user_id: str = "anonymous"
    thread_id: str = "thread"
    run_id: str = "run"
    allow_network: bool = False
    top_k: int = 5
    max_iterations: int = 3
    max_tool_calls: int = 12
    max_network_calls: int = 8
    max_model_calls: int = 1
    require_documents: bool = True
    require_market_data: bool | None = None
    require_market_history: bool | None = None
    require_regulatory_data: bool | None = None
    market_history_range: str = "1y"
    market_history_interval: str = "1d"
    macro_series: tuple[str, ...] = ()
    calculations: tuple[Mapping[str, Any], ...] = ()
    thread_context: Mapping[str, Any] = field(default_factory=dict)
    personal_context: tuple[Mapping[str, Any], ...] = ()
    available_document_count: int = 0

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query is required")
        if len(self.query) > 8_000:
            raise ValueError("query cannot exceed 8000 characters")
        if len(self.entities) > 50 or any(
            len(item) > 200 or any(ord(character) < 32 or ord(character) == 127 for character in item)
            for item in self.entities
        ):
            raise ValueError("entities exceed count or length limits")
        if any(not item.strip() for item in self.entities):
            raise ValueError("entities cannot contain empty values")
        if len(set(self.entities)) != len(self.entities):
            raise ValueError("entities cannot contain duplicates")
        if len(self.symbols) > 50 or any(
            not str(key).strip()
            or len(str(key)) > 200
            or not str(value).strip()
            or len(str(value)) > 64
            or not re.fullmatch(r"[A-Za-z0-9.^=_:-]{1,64}", str(value))
            for key, value in self.symbols.items()
        ):
            raise ValueError("symbols exceed count or length limits")
        if set(self.symbols).difference(self.entities):
            raise ValueError("symbol keys must refer to requested entities")
        for name, value in (
            ("tenant_id", self.tenant_id),
            ("user_id", self.user_id),
            ("thread_id", self.thread_id),
            ("run_id", self.run_id),
        ):
            if not value.strip() or len(value) > 200 or any(ord(item) < 32 or ord(item) == 127 for item in value):
                raise ValueError(f"{name} is invalid")
        if not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 1 <= self.max_iterations <= 8:
            raise ValueError("max_iterations must be between 1 and 8")
        if not 1 <= self.max_tool_calls <= 100:
            raise ValueError("max_tool_calls must be between 1 and 100")
        if not 0 <= self.max_network_calls <= self.max_tool_calls:
            raise ValueError("max_network_calls must be between zero and max_tool_calls")
        if not 0 <= self.max_model_calls <= 20:
            raise ValueError("max_model_calls must be between zero and twenty")
        object.__setattr__(self, "macro_series", validate_macro_series(self.macro_series))
        if len(self.calculations) > 20 or any(not isinstance(item, Mapping) for item in self.calculations):
            raise ValueError("calculations must contain at most 20 objects")
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
        if len(self.personal_context) > 20 or any(not isinstance(item, Mapping) for item in self.personal_context):
            raise ValueError("personal_context must contain at most 20 objects")
        try:
            personal_payload = json.dumps(
                [dict(item) for item in self.personal_context],
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("personal_context must be JSON serializable") from exc
        if len(personal_payload) > 20_000:
            raise ValueError("personal_context exceeds length limit")
        if (
            isinstance(self.available_document_count, bool)
            or not isinstance(self.available_document_count, int)
            or not 0 <= self.available_document_count <= 10_000
        ):
            raise ValueError("available_document_count must be between zero and 10000")
        if self.market_history_range not in {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"}:
            raise ValueError("unsupported market_history_range")
        if self.market_history_interval not in {"1d", "1wk", "1mo"}:
            raise ValueError("unsupported market_history_interval")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["entities"] = list(self.entities)
        value["symbols"] = dict(self.symbols)
        value["macro_series"] = list(self.macro_series)
        value["calculations"] = [dict(item) for item in self.calculations]
        value["thread_context"] = dict(self.thread_context)
        value["personal_context"] = [dict(item) for item in self.personal_context]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchRequest:
        return cls(
            query=str(value.get("query") or ""),
            entities=tuple(str(item) for item in value.get("entities") or ()),
            symbols={str(key): str(item) for key, item in dict(value.get("symbols") or {}).items()},
            tenant_id=str(value.get("tenant_id") or "default"),
            user_id=str(value.get("user_id") or "anonymous"),
            thread_id=str(value.get("thread_id") or "thread"),
            run_id=str(value.get("run_id") or "run"),
            allow_network=bool(value.get("allow_network", False)),
            top_k=int(value.get("top_k", 5)),
            max_iterations=int(value.get("max_iterations", 3)),
            max_tool_calls=int(value.get("max_tool_calls", 12)),
            max_network_calls=int(value.get("max_network_calls", 8)),
            max_model_calls=int(value.get("max_model_calls", 1)),
            require_documents=bool(value.get("require_documents", True)),
            require_market_data=value.get("require_market_data"),
            require_market_history=value.get("require_market_history"),
            require_regulatory_data=value.get("require_regulatory_data"),
            market_history_range=str(value.get("market_history_range") or "1y"),
            market_history_interval=str(value.get("market_history_interval") or "1d"),
            macro_series=tuple(str(item) for item in value.get("macro_series") or ()),
            calculations=tuple(dict(item) for item in value.get("calculations") or ()),
            thread_context=dict(value.get("thread_context") or {}),
            personal_context=tuple(dict(item) for item in value.get("personal_context") or ()),
            available_document_count=int(value.get("available_document_count", 0)),
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
    request: ResearchRequest
    scope: ResearchScope | None = None
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
            "schema_version": 3,
            "request": self.request.to_dict(),
            "scope": self.scope.to_dict() if self.scope else None,
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
        if int(value.get("schema_version", 0)) != 3:
            raise ValueError("unsupported checkpoint schema")
        return cls(
            request=ResearchRequest.from_dict(value["request"]),
            scope=ResearchScope.from_dict(value["scope"]) if value.get("scope") else None,
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
        request: ResearchRequest,
        bundle: EvidenceBundle,
        *,
        research_context: Mapping[str, Any] | None = None,
    ) -> Sequence[Claim]: ...


class AdaptivePlanner:
    """Requirement-driven planner with ordered fallback providers."""

    def __init__(
        self,
        *,
        document_tools: Sequence[str] = ("corpus.search",),
        market_tools: Sequence[str] = ("market.snapshot",),
        market_history_tools: Sequence[str] = ("market.history",),
        regulatory_tools: Sequence[str] = ("sec.company_facts",),
        filing_tools: Sequence[str] = ("sec.recent_filings",),
        macro_tools: Sequence[str] = ("macro.fred_series",),
        calculation_tools: Sequence[str] = ("finance.calculate",),
        web_tools: Sequence[str] = ("web.search",),
    ) -> None:
        self.document_tools = tuple(document_tools)
        self.market_tools = tuple(market_tools)
        self.market_history_tools = tuple(market_history_tools)
        self.regulatory_tools = tuple(regulatory_tools)
        self.filing_tools = tuple(filing_tools)
        self.macro_tools = tuple(macro_tools)
        self.calculation_tools = tuple(calculation_tools)
        self.web_tools = tuple(web_tools)

    def plan(self, state: ResearchState, available_tools: Mapping[str, ToolSpec]) -> ResearchPlan:
        attempted = {observation.task.task_id for observation in state.observations}
        tasks: list[ToolTask] = []
        scope = state.scope or FinancialQueryAnalyzer().analyze(state.request)
        missing = set(state.coverage.missing if state.coverage else (item.key for item in scope.requirements))
        for requirement in scope.requirements:
            if requirement.key not in missing:
                continue
            task = self._first_untried_requirement(
                state.request,
                scope,
                requirement,
                available_tools,
                attempted,
            )
            if task:
                tasks.append(task)

        return ResearchPlan(
            iteration=state.iteration + 1,
            rationale=(
                "Select the next untried authorized provider for uncovered requirements: " + ", ".join(sorted(missing))
            ),
            tasks=tuple(tasks),
        )

    def _first_untried_requirement(
        self,
        request: ResearchRequest,
        scope: ResearchScope,
        requirement: ResearchRequirement,
        available: Mapping[str, ToolSpec],
        attempted: set[str],
    ) -> ToolTask | None:
        candidates = {
            "document": self.document_tools,
            "market": self.market_tools,
            "market_history": self.market_history_tools,
            "regulatory": self.regulatory_tools,
            "filings": self.filing_tools,
            "macro": self.macro_tools,
            "calculation": self.calculation_tools,
            "knowledge": ("finance.knowledge",),
            "web": self.web_tools,
        }.get(requirement.category, ())
        for name in candidates:
            if name not in available:
                continue
            arguments: dict[str, Any]
            if requirement.category == "document":
                query = f"{requirement.entity}: {request.query}" if requirement.entity else request.query
                diversify_documents = _requests_multi_document_synthesis(request.query)
                arguments = {
                    "query": query,
                    "top_k": (
                        min(20, max(request.top_k, request.available_document_count))
                        if diversify_documents
                        else request.top_k
                    ),
                }
                if diversify_documents:
                    arguments["diversify_documents"] = True
            elif requirement.category in {"market", "regulatory"}:
                arguments = {
                    "company": requirement.entity,
                    "symbol": request.symbols.get(requirement.entity or ""),
                    "required_fields": list(requirement.fields),
                }
            elif requirement.category in {"filings", "market_history"}:
                arguments = {
                    "company": requirement.entity,
                    "symbol": request.symbols.get(requirement.entity or ""),
                    **dict(requirement.parameters),
                }
            elif requirement.category == "macro":
                arguments = dict(requirement.parameters)
            elif requirement.category == "calculation":
                calculation = next(
                    item for item in scope.calculations if item.request_id == requirement.parameters.get("request_id")
                )
                arguments = {"requests": [calculation.to_dict()]}
            elif requirement.category == "knowledge":
                arguments = {"query": request.query, **dict(requirement.parameters)}
            elif requirement.category == "web":
                arguments = {"query": request.query, "count": min(request.top_k, 10)}
            else:
                continue
            task = ToolTask.create(
                tool_name=name,
                arguments=arguments,
                reason=requirement.reason,
                category=requirement.category,
                entity=requirement.entity,
                requirement_key=requirement.key,
            )
            if task.task_id not in attempted:
                return task
        return None


def _requests_multi_document_synthesis(query: str) -> bool:
    normalized = query.casefold()
    return bool(
        re.search(
            r"(?:综合|对比|比较|分别|逐份).{0,12}(?:文档|材料|pdf|报告)|"
            r"(?:所有|全部|这些|几份).{0,12}(?:文档|材料|pdf|报告)|"
            r"\b(?:compare|synthesize|across|both|all)\b.{0,30}\b(?:documents?|files?|pdfs?)\b",
            normalized,
        )
    )


class DeterministicSynthesizer:
    """Safe baseline: only restates evidence and never invents unsupported facts."""

    def synthesize(
        self,
        request: ResearchRequest,
        bundle: EvidenceBundle,
        *,
        research_context: Mapping[str, Any] | None = None,
    ) -> Sequence[Claim]:
        claims: list[Claim] = []
        chinese = _contains_cjk(request.query)
        for item in bundle.evidence.values():
            if item.value is not None and item.field_name:
                suffix = f" {item.unit}" if item.unit else ""
                period = f" ({item.period})" if item.period else ""
                if item.entity:
                    entity = item.entity
                elif item.source.source_type == SourceType.CALCULATION:
                    entity = "计算结果" if chinese else "Calculation result"
                else:
                    entity = "未知实体" if chinese else "Unknown entity"
                text = f"{entity} {item.field_name}: {item.value}{suffix}{period}."
                if "declarative_formula" in item.tags:
                    claims.append(
                        Claim.create(
                            text=text,
                            status=ClaimStatus.INFERRED,
                            evidence_ids=(item.evidence_id,),
                            caveat=(
                                "数值可复算且执行安全，但公式语义由用户或模型提供，尚未经过内置金融公式验证。"
                                if chinese
                                else (
                                    "The value is reproducible and safely evaluated, but the user/model-supplied "
                                    "formula semantics are not a built-in verified financial formula."
                                )
                            ),
                        )
                    )
                    continue
            elif item.source.source_type == SourceType.DOCUMENT:
                excerpt = " ".join(item.content.split())[:360]
                prefix = "与问题相关的文档证据表明：" if chinese else "Document evidence relevant to the query states: "
                text = f"{prefix}{excerpt}"
            elif item.source.source_type == SourceType.WEB:
                excerpt = " ".join(item.content.split())[:360]
                prefix = "搜索结果摘要显示：" if chinese else "An open-web search snippet states: "
                text = f"{prefix}{excerpt}"
                claims.append(
                    Claim.create(
                        text=text,
                        status=ClaimStatus.INFERRED,
                        evidence_ids=(item.evidence_id,),
                        caveat=(
                            "该结论仅基于搜索结果摘要，需打开原始页面或结构化一手来源复核。"
                            if chinese
                            else (
                                "This is based only on a search-result snippet and requires verification against "
                                "the original page or a structured primary source."
                            )
                        ),
                    )
                )
                continue
            else:
                continue
            claims.append(Claim.create(text=text, status=ClaimStatus.SUPPORTED, evidence_ids=(item.evidence_id,)))
        return claims


class CoverageAssessor:
    def assess(
        self,
        request: ResearchRequest,
        bundle: EvidenceBundle,
        scope: ResearchScope | None = None,
    ) -> CoverageDecision:
        actual_scope = scope or FinancialQueryAnalyzer().analyze(request)
        missing: list[str] = []
        for requirement in actual_scope.requirements:
            candidates = self._candidates(requirement, bundle)
            fields = {item.field_name for item in candidates if item.field_name}
            distinct_documents = {
                item.source.metadata.get("document_id")
                or item.source.metadata.get("corpus_record_id")
                or item.source.metadata.get("file_name")
                or item.source.title
                for item in candidates
            }
            multi_document_floor_missing = (
                requirement.category == "document"
                and _requests_multi_document_synthesis(request.query)
                and len(distinct_documents) < 2
            )
            if (
                not candidates
                or any(field not in fields for field in requirement.fields)
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
                and "curated_finance_knowledge" not in item.tags
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
                if item.source.source_type == SourceType.CALCULATION and request_id in item.tags
            ]
        if requirement.category == "derived_metric":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.CALCULATION and item.field_name in requirement.fields
            ]
        if requirement.category == "knowledge":
            return [
                item
                for item in candidates
                if item.source.source_type == SourceType.DOCUMENT and "curated_finance_knowledge" in item.tags
            ]
        if requirement.category == "web":
            web = [item for item in candidates if item.source.source_type == SourceType.WEB]
            if any(item.source.metadata.get("quality_tier") == "public_authority" for item in web):
                return web
            domains = {item.source.metadata.get("domain") for item in web if item.source.metadata.get("domain")}
            return web if len(domains) >= 2 else []
        return []


@dataclass(frozen=True)
class ResearchOutcome:
    state: ResearchState
    audit_events: tuple[Mapping[str, Any], ...]
    budget_usage: Mapping[str, int]

    @property
    def status(self) -> str:
        if self.state.phase == AgentPhase.FAILED or not self.state.bundle.evidence:
            return "failed"
        has_qualified_claims = any(claim.status != ClaimStatus.SUPPORTED for claim in self.state.bundle.claims.values())
        if (
            any(not gap.resolved for gap in self.state.gaps)
            or has_qualified_claims
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
    request = state.request
    lines = [
        "# Evidence-first financial research report / 证据优先金融研究报告",
        "",
        f"Query: {_report_text(request.query)}",
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
    lines.append(f"- Iterations: {state.iteration}/{request.max_iterations}")
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
    lines.append(f"- Research tool calls: {research_tool_calls}/{request.max_tool_calls}")
    lines.append(f"- Data-network attempts: {network_attempts}/{request.max_network_calls}")
    lines.append(f"- Model calls: {model_calls}/{request.max_model_calls}")
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


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


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
