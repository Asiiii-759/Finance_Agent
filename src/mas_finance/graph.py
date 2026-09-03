"""LangGraph runtime for the financial research agent.

The graph contains only business stages. Tool execution is an operation of the
planning stage and is always paired with the ToolHarness policy boundary; the
harness is infrastructure, not a graph node.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .adequacy import EvidenceAdequacyChecker, EvidenceAdequacyGap
from .agent import (
    AgentContext,
    AgentPhase,
    ChatTurn,
    CoverageAssessor,
    CoverageDecision,
    Planner,
    ResearchGap,
    ResearchOutcome,
    ResearchPlan,
    ResearchState,
    RuntimePolicy,
    StopReason,
    Synthesizer,
    ToolObservation,
    ToolTask,
    _merge_audit_events,
    _safe_gap_code,
    reconcile_conflicts,
    render_report,
)
from .calculator import derive_standard_ratios
from .contracts import EvidenceBundle, stable_id
from .harness import ExecutionPolicy, SideEffect, ToolContext, ToolHarness, ToolSpec
from .task_frame import LLMTaskInterpreter
from .validators import Severity, validate_research_output

_GROUNDING_CATEGORIES = frozenset(
    {
        "document",
        "market",
        "market_history",
        "regulatory",
        "filings",
        "macro",
        "calculation",
        "derived_metric",
        "web",
        "unsupported",
    }
)


def _requires_grounding(state: ResearchState) -> bool:
    if state.scope is None:
        return False
    return any(item.category in _GROUNDING_CATEGORIES for item in state.scope.requirements)


class FinancialGraphState(TypedDict, total=False):
    research: dict[str, Any]


class FinancialResearchAgent:
    """Four-stage financial agent backed by one LangGraph runtime."""

    def __init__(
        self,
        harness: ToolHarness,
        *,
        planner: Planner,
        task_interpreter: LLMTaskInterpreter,
        synthesizer: Synthesizer,
        assessor: CoverageAssessor | None = None,
        adequacy_checker: EvidenceAdequacyChecker | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        allowed_capabilities: frozenset[str] = frozenset(
            {
                "document.search",
                "market.read",
                "regulatory.read",
                "macro.read",
                "calculation",
                "knowledge.read",
                "web.search",
                "mcp.discover",
                "mcp.invoke",
            }
        ),
        planner_hidden_tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self.harness = harness
        self.planner = planner
        self.task_interpreter = task_interpreter
        self.synthesizer = synthesizer
        self.assessor = assessor or CoverageAssessor()
        self.adequacy_checker = adequacy_checker
        self.allowed_capabilities = allowed_capabilities
        self.planner_hidden_tool_names = planner_hidden_tool_names
        self.checkpointer = checkpointer or InMemorySaver()
        self.available_tools = {
            spec.name: spec for spec in self.harness.tool_specs() if spec.capability in self.allowed_capabilities
        }
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(FinancialGraphState)
        # LangGraph 1.2's contravariant node Protocol does not accept bound
        # methods under mypy even though their runtime signatures match.
        builder.add_node("intent", cast(Any, self._intent_node))
        builder.add_node("planning", cast(Any, self._planning_node))
        builder.add_node("validation", cast(Any, self._validation_node))
        builder.add_node("final_generation", cast(Any, self._final_generation_node))
        builder.add_edge(START, "intent")
        builder.add_edge("intent", "planning")
        builder.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {"planning": "planning", "validation": "validation"},
        )
        builder.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {"planning": "planning", "final_generation": "final_generation", "end": END},
        )
        builder.add_edge("final_generation", "validation")
        return builder.compile(checkpointer=self.checkpointer)

    def run(
        self,
        turn: ChatTurn,
        runtime_policy: RuntimePolicy,
        context: AgentContext,
        *,
        resume: bool = False,
    ) -> ResearchOutcome:
        config = self._config(turn, runtime_policy)
        snapshot = self.graph.get_state(config)
        if resume:
            if not snapshot.values:
                raise ValueError("no LangGraph checkpoint exists for this run")
            persisted = self._state(snapshot.values)
            if (
                persisted.turn != turn
                or persisted.runtime_policy != runtime_policy
                or persisted.context != context
            ):
                raise ValueError("checkpoint input does not match resume input")
            self._prime_harness(persisted)
            result = self.graph.invoke(None, config)
        else:
            if snapshot.values:
                raise ValueError("a LangGraph checkpoint already exists for this run")
            initial = ResearchState(turn=turn, runtime_policy=runtime_policy, context=context)
            result = self.graph.invoke({"research": initial.to_dict()}, config)
        state = self._state(result)
        return self._outcome(state)

    def get_state(self, turn: ChatTurn, runtime_policy: RuntimePolicy) -> ResearchState | None:
        values = self.graph.get_state(self._config(turn, runtime_policy)).values
        return self._state(values) if values else None

    def state_history(self, turn: ChatTurn, runtime_policy: RuntimePolicy) -> tuple[ResearchState, ...]:
        return tuple(
            self._state(snapshot.values)
            for snapshot in self.graph.get_state_history(self._config(turn, runtime_policy))
            if snapshot.values and "research" in snapshot.values
        )

    def _intent_node(self, graph_state: FinancialGraphState) -> dict[str, Any]:
        state = self._state(graph_state)
        self.harness.clear_run(state.turn.run_id)
        frame = self.task_interpreter.interpret(
            state.turn,
            state.runtime_policy,
            state.context,
            self._planner_catalog(),
        )
        state.scope = frame.scope
        state.task_frame = frame.to_dict()
        context_manifest = self.task_interpreter.context_manifest()
        if context_manifest:
            state.context_manifests.append({"phase": "intent", "iteration": 0, **context_manifest})
        state.phase = AgentPhase.INTENT
        if frame.requires_clarification:
            state.stop_reason = StopReason.CLARIFICATION_REQUIRED
            state.report = f"需要澄清后才能继续：{frame.clarification_question}"
        return {"research": state.to_dict()}

    def _planning_node(self, graph_state: FinancialGraphState) -> dict[str, Any]:
        state = self._state(graph_state)
        state.phase = AgentPhase.PLANNING
        if state.stop_reason is not None:
            return self._update(state)
        if len(state.observations) >= state.runtime_policy.max_tool_calls:
            self._mark_tool_budget_exhausted(state)
            return self._update(state)

        observed = {item.task.task_id for item in state.observations}
        plan = self._unfinished_plan(state, observed)
        if plan is None:
            if state.iteration >= state.runtime_policy.max_iterations:
                state.stop_reason = StopReason.MAX_ITERATIONS
                return self._update(state)
            plan = self._validated_plan(self.planner.plan(state, self._planner_catalog()), state)
            context_manifest = getattr(self.planner, "context_manifest", lambda: None)()
            if context_manifest:
                state.context_manifests.append(
                    {"phase": "planning", "iteration": state.iteration + 1, **context_manifest}
                )
            state.audit_events = _merge_audit_events(
                state.audit_events,
                self.harness.audit_events(state.turn.run_id),
            )
            self._consume_planner_diagnostics(state)
            if not plan.tasks:
                state.plans.append(plan)
                state.iteration = plan.iteration
                if not plan.ready_for_validation:
                    self._mark_no_available_action(state)
                return self._update(state)
            state.plans.append(plan)
            state.iteration = plan.iteration
            observed = {item.task.task_id for item in state.observations}

        pending = [item for item in plan.tasks if item.task_id not in observed]
        budget_left = state.runtime_policy.max_tool_calls - len(state.observations)
        if budget_left <= 0 or not pending:
            self._mark_tool_budget_exhausted(state)
            return self._update(state)
        self._execute_tasks(state, pending[:budget_left])
        remaining = {item.task_id for item in plan.tasks} - {item.task.task_id for item in state.observations}
        if remaining and len(state.observations) >= state.runtime_policy.max_tool_calls:
            self._mark_tool_budget_exhausted(state)
        return self._update(state)

    def _planner_catalog(self) -> Mapping[str, ToolSpec]:
        if not self.planner_hidden_tool_names or not getattr(self.planner, "requires_explicit_finish", False):
            return self.available_tools
        return {
            name: spec
            for name, spec in self.available_tools.items()
            if name not in self.planner_hidden_tool_names
        }

    def _execute_tasks(self, state: ResearchState, tasks: Sequence[ToolTask]) -> None:
        context = self._tool_context(state.turn, state.runtime_policy, self.available_tools)
        workers = max(1, min(state.runtime_policy.max_parallel_tool_calls, len(tasks)))

        def run(task: ToolTask) -> tuple[ToolTask, Any]:
            return task, self.harness.invoke(task.tool_name, task.arguments, context)

        if workers == 1 or len(tasks) == 1:
            pairs = [run(task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pairs = list(pool.map(run, tasks))
        exhausted = False
        for task, result in pairs:
            observation = ToolObservation(
                task=task,
                iteration=state.iteration,
                result=result.to_dict(),
                network_access=self.available_tools[task.tool_name].network_access,
            )
            state.observations.append(observation)
            self._consume(observation, state)
            if result.status.value == "budget_exhausted":
                exhausted = True
        state.audit_events = _merge_audit_events(
            state.audit_events,
            self.harness.audit_events(state.turn.run_id),
        )
        if exhausted:
            self._mark_tool_budget_exhausted(state)

    def _validation_node(self, graph_state: FinancialGraphState) -> dict[str, Any]:
        state = self._state(graph_state)
        state.phase = AgentPhase.VALIDATING
        if state.stop_reason is StopReason.CLARIFICATION_REQUIRED:
            state.phase = AgentPhase.COMPLETED
            return self._update(state)
        if state.report:
            validation = validate_research_output(
                bundle=state.bundle,
                report=state.report,
                gaps=(gap.to_dict() for gap in state.gaps),
            )
            state.validation_issues = [issue.to_dict() for issue in validation.issues]
            if any(issue.severity == Severity.ERROR for issue in validation.issues):
                state.phase = AgentPhase.FAILED
                state.stop_reason = StopReason.VALIDATION_FAILED
                state.report = render_report(state)
            elif not state.bundle.evidence and not state.bundle.claims:
                state.phase = AgentPhase.FAILED
                state.stop_reason = StopReason.NO_EVIDENCE
            else:
                state.phase = AgentPhase.COMPLETED
            return self._update(state)

        derive_standard_ratios(state.bundle)
        state.coverage = self.assessor.assess(state.turn, state.bundle, state.scope)
        self._resolve_recovered_gaps(state)
        if self._only_unsupported_requirements_remain(state):
            self._mark_no_available_action(state)
            return self._update(state)
        planner_finish_required = bool(getattr(self.planner, "requires_explicit_finish", False))
        model_declared_finish = bool(state.plans and state.plans[-1].ready_for_validation)
        if state.coverage.complete and model_declared_finish and self.adequacy_checker is not None:
            adequacy_gaps = tuple(self.adequacy_checker.check(state))
            state.audit_events = _merge_audit_events(
                state.audit_events,
                self.harness.audit_events(state.turn.run_id),
            )
            manifest = getattr(self.adequacy_checker, "context_manifest", lambda: None)()
            if manifest:
                state.context_manifests.append(
                    {"phase": "validation", "iteration": state.iteration, **manifest}
                )
            self._apply_adequacy(state, adequacy_gaps)
            if adequacy_gaps and state.iteration >= state.runtime_policy.max_iterations:
                state.stop_reason = StopReason.INSUFFICIENT_EVIDENCE
        if (
            state.coverage.complete
            and (state.bundle.evidence or not _requires_grounding(state))
            and (
                not planner_finish_required
                or model_declared_finish
                or state.iteration >= state.runtime_policy.max_iterations
            )
        ):
            state.stop_reason = StopReason.COVERAGE_SATISFIED
        elif state.stop_reason is None and state.iteration >= state.runtime_policy.max_iterations:
            state.stop_reason = StopReason.MAX_ITERATIONS
        return self._update(state)

    def _final_generation_node(self, graph_state: FinancialGraphState) -> dict[str, Any]:
        state = self._state(graph_state)
        state.phase = AgentPhase.FINAL_GENERATION
        derive_standard_ratios(state.bundle)
        state.bundle.claims.clear()
        if not state.bundle.evidence and _requires_grounding(state):
            state.stop_reason = StopReason.NO_EVIDENCE
            state.report = render_report(state)
            return self._update(state)
        claims = reconcile_conflicts(
            state.bundle,
            self.synthesizer.synthesize(
                state.turn,
                state.runtime_policy,
                state.context,
                state.bundle,
                research_context={
                    "task_frame": state.task_frame,
                    "scope": state.scope.to_dict() if state.scope else None,
                    "coverage": state.coverage.to_dict() if state.coverage else None,
                    "unresolved_gaps": [gap.to_dict() for gap in state.gaps if not gap.resolved],
                    "stop_reason": state.stop_reason.value if state.stop_reason else None,
                },
            ),
        )
        diagnostics = getattr(self.synthesizer, "diagnostics", lambda: ())()
        context_manifest = getattr(self.synthesizer, "context_manifest", lambda: None)()
        if context_manifest:
            state.context_manifests.append(
                {"phase": "final_generation", "iteration": state.iteration, **context_manifest}
            )
        for item in diagnostics:
            state.gaps.append(
                ResearchGap(
                    code=_safe_gap_code(item.get("code") or "synthesis_degraded"),
                    message=str(item.get("message") or "Synthesis degraded.")[:2_000],
                    resolvable=False,
                )
            )
        for claim in claims:
            state.bundle.add_claim(claim)
        state.audit_events = _merge_audit_events(
            state.audit_events,
            self.harness.audit_events(state.turn.run_id),
        )
        state.report = render_report(state)
        return self._update(state)

    @staticmethod
    def _apply_adequacy(state: ResearchState, decisions: Sequence[EvidenceAdequacyGap]) -> None:
        missing = {item.requirement_key for item in decisions}
        updated: list[ResearchGap] = []
        for gap in state.gaps:
            if gap.code not in {"evidence_semantically_insufficient", "evidence_conflict"}:
                updated.append(gap)
                continue
            updated.append(
                ResearchGap(
                    code=gap.code,
                    message=gap.message,
                    entity=gap.entity,
                    requirement_key=gap.requirement_key,
                    tool_name=gap.tool_name,
                    task_id=gap.task_id,
                    resolvable=gap.resolvable,
                    resolved=gap.requirement_key not in missing,
                )
            )
        for decision in decisions:
            code = "evidence_conflict" if decision.status == "conflicting" else "evidence_semantically_insufficient"
            parts = [*decision.missing_information]
            if decision.retrieval_hint:
                parts.append(f"下一步：{decision.retrieval_hint}")
            message = "；".join(parts)
            if not any(
                gap.code == code
                and gap.requirement_key == decision.requirement_key
                and not gap.resolved
                and gap.message == message
                for gap in updated
            ):
                updated.append(
                    ResearchGap(
                        code=code,
                        message=message,
                        requirement_key=decision.requirement_key,
                        resolvable=True,
                    )
                )
        state.gaps = updated
        if missing:
            state.coverage = CoverageDecision(
                complete=False,
                missing=tuple(sorted(missing)),
                rationale="Retrieved evidence is not semantically sufficient for every requirement.",
            )

    def _route_after_planning(self, graph_state: FinancialGraphState) -> Literal["planning", "validation"]:
        state = self._state(graph_state)
        if state.stop_reason is None and state.plans:
            observed = {item.task.task_id for item in state.observations}
            if any(task.task_id not in observed for task in state.plans[-1].tasks):
                return "planning"
        return "validation"

    def _route_after_validation(
        self,
        graph_state: FinancialGraphState,
    ) -> Literal["planning", "final_generation", "end"]:
        state = self._state(graph_state)
        if state.phase in {AgentPhase.COMPLETED, AgentPhase.FAILED}:
            return "end"
        return "final_generation" if state.stop_reason is not None else "planning"

    def _validated_plan(self, plan: ResearchPlan, state: ResearchState) -> ResearchPlan:
        invalid = [task for task in plan.tasks if task.tool_name not in self.available_tools]
        for task in invalid:
            state.gaps.append(
                ResearchGap(
                    code="planner_tool_denied",
                    message="Planner selected a tool outside the agent capability allowlist.",
                    entity=task.entity,
                    tool_name=task.tool_name,
                    task_id=task.task_id,
                    requirement_key=task.requirement_key,
                )
            )
        attempted = {item.task.task_id for item in state.observations}
        seen: set[str] = set()
        tasks: list[ToolTask] = []
        for task in plan.tasks:
            if task.tool_name not in self.available_tools:
                continue
            if task.task_id in attempted:
                state.gaps.append(
                    ResearchGap(
                        code="repeated_planner_action",
                        message="Planner repeated an already observed tool action; it was not executed again.",
                        entity=task.entity,
                        requirement_key=task.requirement_key,
                        tool_name=task.tool_name,
                        task_id=task.task_id,
                        resolvable=True,
                    )
                )
                continue
            if task.task_id in seen:
                state.gaps.append(
                    ResearchGap(
                        code="duplicate_planner_task",
                        message="Planner emitted a duplicate task; it was not executed.",
                        entity=task.entity,
                        requirement_key=task.requirement_key,
                        tool_name=task.tool_name,
                        task_id=task.task_id,
                        resolved=True,
                    )
                )
                continue
            seen.add(task.task_id)
            tasks.append(task)
        return ResearchPlan(
            iteration=plan.iteration,
            rationale=plan.rationale,
            tasks=tuple(tasks),
            ready_for_validation=plan.ready_for_validation or bool(plan.tasks and not tasks and not invalid),
        )

    def _consume_planner_diagnostics(self, state: ResearchState) -> None:
        diagnostics = getattr(self.planner, "diagnostics", lambda: ())()
        existing = {(gap.code, gap.message) for gap in state.gaps}
        for item in diagnostics:
            code = _safe_gap_code(item.get("code") or "planner_degraded")
            message = str(item.get("message") or "Planner degraded.")[:2_000]
            if (code, message) not in existing:
                state.gaps.append(ResearchGap(code=code, message=message, resolvable=False))

    def _mark_no_available_action(self, state: ResearchState) -> None:
        state.coverage = self.assessor.assess(state.turn, state.bundle, state.scope)
        if state.coverage.complete and (state.bundle.evidence or not _requires_grounding(state)):
            state.stop_reason = StopReason.COVERAGE_SATISFIED
            return
        requirements = {item.key: item for item in state.scope.requirements} if state.scope else {}
        unsupported = {
            key: requirements[key]
            for key in state.coverage.missing
            if key in requirements and requirements[key].category == "unsupported"
        }
        for key, requirement in unsupported.items():
            state.gaps.append(
                ResearchGap(
                    code=_safe_gap_code(requirement.parameters.get("gap_code", "unsupported_research_scope")),
                    message=requirement.reason,
                    entity=requirement.entity,
                    requirement_key=key,
                    resolvable=False,
                )
            )
        if not requirements:
            state.gaps.append(
                ResearchGap(
                    code="unsupported_research_scope",
                    message="The question did not map to a supported evidence requirement.",
                    resolvable=False,
                )
            )
        for key in state.coverage.missing:
            if key not in unsupported:
                state.gaps.append(
                    ResearchGap(
                        code="requirement_provider_unavailable",
                        message=f"No authorized untried tool can satisfy requirement {key}.",
                        requirement_key=key,
                        resolvable=False,
                    )
                )
        state.stop_reason = StopReason.NO_AVAILABLE_ACTION

    def _only_unsupported_requirements_remain(self, state: ResearchState) -> bool:
        if not state.scope or not state.coverage or not state.coverage.missing:
            return False
        requirements = {item.key: item for item in state.scope.requirements}
        return all(
            key in requirements and requirements[key].category == "unsupported" for key in state.coverage.missing
        )

    @staticmethod
    def _unfinished_plan(state: ResearchState, observed: set[str]) -> ResearchPlan | None:
        if state.plans and any(task.task_id not in observed for task in state.plans[-1].tasks):
            return state.plans[-1]
        return None

    @staticmethod
    def _mark_tool_budget_exhausted(state: ResearchState) -> None:
        if not any(gap.code == "tool_call_budget_exhausted" for gap in state.gaps):
            state.gaps.append(
                ResearchGap(
                    code="tool_call_budget_exhausted",
                    message="The research tool-call budget was exhausted before evidence coverage completed.",
                    resolvable=False,
                )
            )
        state.stop_reason = StopReason.TOOL_BUDGET_EXHAUSTED

    def _consume(self, observation: ToolObservation, state: ResearchState) -> None:
        result = observation.result
        if not result.get("ok"):
            state.gaps.append(
                ResearchGap(
                    code=_safe_gap_code(result.get("error_code") or "tool_error"),
                    message=str(result.get("error_message") or "Tool call failed.")[:2_000],
                    entity=observation.task.entity,
                    requirement_key=observation.task.requirement_key,
                    tool_name=observation.task.tool_name,
                    task_id=observation.task.task_id,
                    resolvable=True,
                )
            )
            return
        data = result.get("data")
        if not isinstance(data, Mapping):
            state.gaps.append(
                ResearchGap(
                    code="invalid_tool_contract",
                    message="Tool did not return an object result.",
                    entity=observation.task.entity,
                    requirement_key=observation.task.requirement_key,
                    tool_name=observation.task.tool_name,
                    task_id=observation.task.task_id,
                    resolvable=True,
                )
            )
            return
        spec = self.available_tools.get(observation.task.tool_name)
        if spec is not None and spec.capability == "mcp.discover":
            return
        if not isinstance(data.get("bundle"), Mapping):
            state.gaps.append(
                ResearchGap(
                    code="invalid_tool_contract",
                    message="Tool did not return an evidence bundle.",
                    entity=observation.task.entity,
                    requirement_key=observation.task.requirement_key,
                    tool_name=observation.task.tool_name,
                    task_id=observation.task.task_id,
                    resolvable=True,
                )
            )
            return
        incoming = EvidenceBundle.from_dict(data["bundle"])
        state.bundle.merge(incoming)
        for value in data.get("gaps") or ():
            if isinstance(value, Mapping):
                state.gaps.append(
                    ResearchGap(
                        code=_safe_gap_code(value.get("code")),
                        message=str(value.get("message") or "Data is incomplete.")[:2_000],
                        entity=observation.task.entity,
                        requirement_key=observation.task.requirement_key,
                        tool_name=observation.task.tool_name,
                        task_id=observation.task.task_id,
                        resolvable=bool(value.get("recoverable_by_coverage", False)),
                    )
                )

    @staticmethod
    def _resolve_recovered_gaps(state: ResearchState) -> None:
        if state.coverage is None:
            return
        missing = set(state.coverage.missing)
        updated: list[ResearchGap] = []
        for gap in state.gaps:
            if gap.resolved or not gap.resolvable:
                updated.append(gap)
                continue
            # 只有明确对应到某个 requirement 且该需求已覆盖时，才把可恢复缺口标成已解决。
            # 模型计划通常不带 requirement_key；此时不能用 category:entity 去猜，
            # 否则 `market:Apple` 对不上 `market:Apple:1`，未覆盖的 network_denied 也会被误标 resolved。
            recovered = gap.requirement_key not in missing if gap.requirement_key is not None else not missing
            updated.append(
                ResearchGap(
                    code=gap.code,
                    message=gap.message,
                    entity=gap.entity,
                    requirement_key=gap.requirement_key,
                    tool_name=gap.tool_name,
                    task_id=gap.task_id,
                    resolvable=gap.resolvable,
                    resolved=recovered,
                )
            )
        state.gaps = updated

    @staticmethod
    def _tool_context(
        turn: ChatTurn,
        runtime_policy: RuntimePolicy,
        available: Mapping[str, ToolSpec],
    ) -> ToolContext:
        return ToolContext(
            run_id=turn.run_id,
            thread_id=turn.thread_id,
            tenant_id=turn.tenant_id,
            user_id=turn.user_id,
            policy=ExecutionPolicy(
                allowed_capabilities=frozenset(spec.capability for spec in available.values()),
                allowed_side_effects=frozenset({SideEffect.READ_ONLY}),
                allow_network=turn.allow_network,
                max_tool_calls=runtime_policy.max_tool_calls,
                max_network_calls=runtime_policy.max_network_calls,
                max_model_calls=runtime_policy.max_model_calls,
                max_model_input_tokens=runtime_policy.max_model_input_tokens,
                max_model_output_tokens=runtime_policy.max_model_output_tokens,
            ),
        )

    def _prime_harness(self, state: ResearchState) -> None:
        tool_calls, network_calls, model_calls, model_input_tokens, model_output_tokens, sequence = self._audit_usage(
            state.audit_events
        )
        self.harness.prime_run(
            state.turn.run_id,
            tool_calls=tool_calls,
            network_calls=network_calls,
            model_calls=model_calls,
            model_input_tokens=model_input_tokens,
            model_output_tokens=model_output_tokens,
            sequence=sequence,
        )

    def _outcome(self, state: ResearchState) -> ResearchOutcome:
        state.audit_events = _merge_audit_events(
            state.audit_events,
            self.harness.audit_events(state.turn.run_id),
        )
        tool_calls, network_calls, model_calls, model_input_tokens, model_output_tokens, _ = self._audit_usage(
            state.audit_events
        )
        return ResearchOutcome(
            state=state,
            audit_events=tuple(state.audit_events),
            budget_usage={
                "tool_calls": tool_calls,
                "network_attempts": network_calls,
                "model_calls": model_calls,
                "model_input_tokens": model_input_tokens,
                "model_output_tokens": model_output_tokens,
            },
        )

    @staticmethod
    def _audit_usage(events: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int, int, int]:
        tool_calls = sum(
            1 for item in events if item.get("capability") != "model.generate" and bool(item.get("budget_consumed"))
        )
        model_calls = sum(
            1 for item in events if item.get("capability") == "model.generate" and bool(item.get("budget_consumed"))
        )
        network_calls = sum(
            int(item.get("network_attempts") or 0) for item in events if item.get("capability") != "model.generate"
        )
        model_input_tokens = sum(int(item.get("model_input_tokens") or 0) for item in events)
        model_output_tokens = sum(int(item.get("model_output_tokens") or 0) for item in events)
        sequence = max(
            (int(str(item["call_id"]).rsplit(":", 1)[-1]) for item in events if item.get("call_id")),
            default=0,
        )
        return tool_calls, network_calls, model_calls, model_input_tokens, model_output_tokens, sequence

    @staticmethod
    def _state(graph_state: Mapping[str, Any]) -> ResearchState:
        return ResearchState.from_dict(graph_state["research"])

    @staticmethod
    def _update(state: ResearchState) -> dict[str, Any]:
        return {"research": state.to_dict()}

    @staticmethod
    def _config(turn: ChatTurn, runtime_policy: RuntimePolicy) -> dict[str, Any]:
        thread_id = stable_id(
            "run",
            {
                "tenant_id": turn.tenant_id,
                "user_id": turn.user_id,
                "thread_id": turn.thread_id,
                "run_id": turn.run_id,
            },
        )
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": runtime_policy.max_tool_calls * 2 + runtime_policy.max_iterations * 2 + 8,
        }
