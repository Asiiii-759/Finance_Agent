"""Policy-enforcing execution harness for all agent tool calls."""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from opentelemetry import trace

from .contracts import EvidenceBundle, utc_now

_TRACER = trace.get_tracer("mas_finance.tools")


class SideEffect(StrEnum):
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    FINANCIAL_TRANSACTION = "financial_transaction"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ToolResultKind(StrEnum):
    ANY = "any"
    EVIDENCE_BUNDLE = "evidence_bundle"
    MODEL_RESPONSE = "model_response"
    CATALOG = "catalog"


class ToolContractError(ValueError):
    pass


class ToolExecutionError(RuntimeError):
    """工具向规划模型返回的结构化、可采取行动的执行错误。"""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            raise ValueError("tool execution error code is invalid")
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ToolArgumentContract:
    """Small, dependency-free boundary contract for tool arguments.

    Domain-specific type/range checks remain inside adapters.  This contract
    centrally rejects missing/unknown keys and unbounded or non-JSON payloads
    before a provider call consumes budget.
    """

    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()
    allow_extra: bool = False
    max_serialized_characters: int = 100_000

    def __post_init__(self) -> None:
        if self.required.intersection(self.optional):
            raise ValueError("tool argument contract keys overlap")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item) for item in (*self.required, *self.optional)):
            raise ValueError("tool argument contract contains an invalid key")
        if self.max_serialized_characters < 100:
            raise ValueError("tool argument size limit is too small")

    def validate(self, arguments: Mapping[str, Any]) -> None:
        if not isinstance(arguments, Mapping):
            raise ToolContractError("tool arguments must be an object")
        if any(not isinstance(key, str) for key in arguments):
            raise ToolContractError("tool argument keys must be strings")
        keys = set(arguments)
        missing = self.required.difference(keys)
        if missing:
            raise ToolContractError(f"missing required tool arguments: {sorted(missing)}")
        unknown = keys.difference(self.required, self.optional)
        if unknown and not self.allow_extra:
            raise ToolContractError(f"unknown tool arguments: {sorted(unknown)}")
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolContractError("tool arguments must be finite JSON values") from exc
        if len(encoded) > self.max_serialized_characters:
            raise ToolContractError("tool arguments exceed the serialized size limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": sorted(self.required),
            "optional": sorted(self.optional),
            "allow_extra": self.allow_extra,
            "max_serialized_characters": self.max_serialized_characters,
        }


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    initial_backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if (
            not math.isfinite(self.initial_backoff_seconds)
            or not math.isfinite(self.backoff_multiplier)
            or self.initial_backoff_seconds < 0
            or self.initial_backoff_seconds > 60
            or not 1 <= self.backoff_multiplier <= 10
        ):
            raise ValueError("invalid retry backoff")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    capability: str
    side_effect: SideEffect = SideEffect.READ_ONLY
    network_access: bool = False
    timeout_seconds: float = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    result_kind: ToolResultKind = ToolResultKind.ANY
    arguments: ToolArgumentContract = field(default_factory=lambda: ToolArgumentContract(allow_extra=True))
    input_schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", self.name):
            raise ValueError("tool name is invalid")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", self.capability):
            raise ValueError("tool capability is invalid")
        if not self.description.strip() or len(self.description) > 1_000:
            raise ValueError("tool description is invalid")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 600:
            raise ValueError("tool timeout must be between 0 and 600 seconds")
        if self.side_effect != SideEffect.READ_ONLY and self.retry.max_attempts > 1:
            raise ValueError("side-effecting tools cannot be retried automatically")
        if self.input_schema is not None:
            try:
                encoded_schema = json.dumps(self.input_schema, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("tool input schema must be finite JSON") from exc
            if len(encoded_schema) > 100_000:
                raise ValueError("tool input schema exceeds length limit")
            try:
                Draft202012Validator.check_schema(self.input_schema)
            except SchemaError as exc:
                raise ValueError("tool input schema is not valid Draft 2020-12 JSON Schema") from exc
            if self.input_schema.get("type") != "object":
                raise ValueError("tool input schema root type must be object")
            properties = self.input_schema.get("properties")
            required = self.input_schema.get("required", [])
            if not isinstance(properties, Mapping) or not isinstance(required, list):
                raise ValueError("tool input schema must declare properties and required keys")
            schema_keys = frozenset(str(key) for key in properties)
            schema_required = frozenset(str(key) for key in required)
            if schema_required != self.arguments.required:
                raise ValueError("tool input schema required keys disagree with the argument contract")
            if not self.arguments.allow_extra and schema_keys != self.arguments.required | self.arguments.optional:
                raise ValueError("tool input schema properties disagree with the argument contract")


def model_tool_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["system_prompt", "user_prompt"],
        "additionalProperties": False,
        "properties": {
            "system_prompt": {"type": "string", "minLength": 1, "maxLength": 1_000_000},
            "user_prompt": {"type": "string", "minLength": 1, "maxLength": 1_000_000},
            "temperature": {"type": "number", "minimum": 0, "maximum": 2},
            "max_tokens": {"type": "integer", "minimum": 1, "maximum": 200_000},
        },
    }


def model_retry_policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=2, initial_backoff_seconds=0.25)


def validate_tool_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> None:
    try:
        spec.arguments.validate(arguments)
    except ToolContractError as exc:
        raise ToolExecutionError(
            "invalid_tool_arguments",
            str(exc),
            details={"model_action": "change_arguments"},
        ) from exc
    if spec.input_schema is None:
        return
    try:
        Draft202012Validator(spec.input_schema).validate(dict(arguments))
    except ValidationError as exc:
        message, details = _json_schema_error(exc)
        raise ToolExecutionError("invalid_tool_arguments", message, details=details) from exc


@dataclass(frozen=True)
class ExecutionPolicy:
    allowed_capabilities: frozenset[str]
    allowed_side_effects: frozenset[SideEffect] = frozenset({SideEffect.READ_ONLY})
    allow_network: bool = False
    max_tool_calls: int = 20
    max_network_calls: int = 8
    max_model_calls: int = 1
    max_model_input_tokens: int = 300_000
    max_model_output_tokens: int = 32_768

    def __post_init__(self) -> None:
        if not 0 <= self.max_tool_calls <= 1_000:
            raise ValueError("research tool-call budget must be between 0 and 1000")
        if not 0 <= self.max_network_calls <= 10_000:
            raise ValueError("network-attempt budget must be between 0 and 10000")
        if not 0 <= self.max_model_calls <= 100:
            raise ValueError("model-call budget must be between 0 and 100")
        if not 1_000 <= self.max_model_input_tokens <= 1_000_000:
            raise ValueError("model input-token budget must be between 1000 and 1000000")
        if not 256 <= self.max_model_output_tokens <= 200_000:
            raise ValueError("model output-token budget must be between 256 and 200000")
        if min(self.max_tool_calls, self.max_network_calls, self.max_model_calls) < 0:
            raise ValueError("execution budgets cannot be negative")


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    thread_id: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    policy: ExecutionPolicy = field(default_factory=lambda: ExecutionPolicy(frozenset()))

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("thread_id", self.thread_id),
            ("tenant_id", self.tenant_id),
            ("user_id", self.user_id),
        ):
            if (
                not value
                or len(value) > 200
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ValueError(f"tool context {name} is invalid")


class Tool(Protocol):
    spec: ToolSpec

    def __call__(self, arguments: Mapping[str, Any], context: ToolContext) -> Any: ...


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolStatus
    started_at: str
    duration_ms: float
    attempts: int
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["ok"] = self.ok
        return value


@dataclass(frozen=True)
class ToolAuditEvent:
    call_id: str
    run_id: str
    thread_id: str
    tenant_id: str
    tool_name: str
    capability: str
    side_effect: str
    network_access: bool
    arguments: Mapping[str, Any]
    result_status: str
    attempts: int
    budget_consumed: bool
    network_attempts: int
    model_input_tokens: int
    model_output_tokens: int
    duration_ms: float
    timestamp: str
    error_code: str | None = None
    error_message: str | None = None
    result_summary: Mapping[str, Any] | None = None


@dataclass
class _Budget:
    tool_calls: int = 0
    network_calls: int = 0
    model_calls: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0


@dataclass(frozen=True)
class _RunBoundary:
    thread_id: str
    tenant_id: str
    user_id: str
    allow_network: bool
    max_tool_calls: int
    max_network_calls: int
    max_model_calls: int
    max_model_input_tokens: int
    max_model_output_tokens: int


@dataclass(frozen=True)
class ToolBudgetUsage:
    tool_calls: int
    network_attempts: int
    model_calls: int
    model_input_tokens: int
    model_output_tokens: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ToolHarness:
    """Central registry, policy gate, retry budget and audit boundary.

    Timeouts are *observed* for synchronous tools. Providers must also receive
    and enforce their own I/O timeout; Python cannot safely kill a running
    thread. This avoids pretending that a leaked worker is a hard timeout.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._budgets: dict[str, _Budget] = {}
        self._run_boundaries: dict[str, _RunBoundary] = {}
        self._sequences: dict[str, int] = {}
        self._events: list[ToolAuditEvent] = []
        self._lock = threading.RLock()
        self._clock = clock
        self._sleep = sleeper
        self._random = random_source or random.Random()

    def register(self, tool: Tool) -> None:
        with self._lock:
            if tool.spec.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.spec.name}")
            self._tools[tool.spec.name] = tool

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        """Return immutable tool metadata for planning without exposing callables."""
        with self._lock:
            return tuple(tool.spec for tool in self._tools.values())

    def has_tool(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def prime_run(
        self,
        run_id: str,
        *,
        tool_calls: int,
        network_calls: int,
        model_calls: int,
        model_input_tokens: int = 0,
        model_output_tokens: int = 0,
        sequence: int,
    ) -> None:
        """Restore counters before resuming a persisted run.

        This prevents a resumed process from silently receiving a fresh budget or
        reusing call identifiers. It intentionally restores counters, not audit
        payloads; durable observations live in the agent checkpoint.
        """
        if min(tool_calls, network_calls, model_calls, model_input_tokens, model_output_tokens, sequence) < 0:
            raise ValueError("invalid restored harness counters")
        with self._lock:
            self._budgets[run_id] = _Budget(
                tool_calls=tool_calls,
                network_calls=network_calls,
                model_calls=model_calls,
                model_input_tokens=model_input_tokens,
                model_output_tokens=model_output_tokens,
            )
            self._sequences[run_id] = sequence

    def invoke(self, name: str, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        with _TRACER.start_as_current_span(
            "agent.tool",
            attributes={"agent.tool.name": name, "agent.run.id": context.run_id},
        ) as span:
            result = self._invoke(name, arguments, context)
            span.set_attribute("agent.tool.status", result.status.value)
            span.set_attribute("agent.tool.duration_ms", result.duration_ms)
            return result

    def _invoke(self, name: str, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        started_at = utc_now()
        start = self._clock()
        valid_name = isinstance(name, str) and bool(re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", name))
        tool = self._tools.get(name) if valid_name else None
        call_id = f"{context.run_id}:{self._reserve_call_number(context.run_id)}"
        boundary_error = self._bind_or_check_run_boundary(context)
        if boundary_error is not None:
            return self._finish_denied(
                call_id,
                name,
                arguments,
                started_at,
                start,
                context,
                "run_context_mismatch",
                boundary_error,
                tool.spec if tool else None,
            )
        if not valid_name:
            return self._finish_denied(
                call_id,
                "invalid.tool",
                arguments,
                started_at,
                start,
                context,
                "invalid_tool_name",
                "requested tool name is invalid",
            )
        if tool is None:
            return self._finish_denied(
                call_id,
                name,
                arguments,
                started_at,
                start,
                context,
                "tool_not_registered",
                "tool is not registered",
            )

        denial = self._check_policy(tool.spec, context)
        if denial:
            return self._finish_denied(
                call_id, name, arguments, started_at, start, context, denial[0], denial[1], tool.spec
            )

        try:
            validate_tool_arguments(tool.spec, arguments)
        except ToolExecutionError as exc:
            return self._finish(
                call_id=call_id,
                tool=tool,
                context=context,
                arguments=arguments,
                status=ToolStatus.ERROR,
                started_at=started_at,
                start=start,
                attempts=0,
                error_code=exc.code,
                error_message=str(exc),
                error_details=exc.details,
            )

        with self._lock:
            budget = self._budgets.setdefault(context.run_id, _Budget())
            is_model = tool.spec.capability == "model.generate"
            input_tokens = _model_input_tokens(arguments) if is_model else 0
            requested_output_tokens = int(arguments.get("max_tokens", 600)) if is_model else 0
            if is_model and budget.model_calls >= context.policy.max_model_calls:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.BUDGET_EXHAUSTED,
                    started_at=started_at,
                    start=start,
                    attempts=0,
                    error_code="model_call_budget_exhausted",
                    error_message="run model-call budget exhausted",
                )
            if is_model and budget.model_input_tokens + input_tokens > context.policy.max_model_input_tokens:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.BUDGET_EXHAUSTED,
                    started_at=started_at,
                    start=start,
                    attempts=0,
                    error_code="model_input_token_budget_exhausted",
                    error_message="run model input-token budget exhausted",
                )
            if (
                is_model
                and budget.model_output_tokens + requested_output_tokens
                > context.policy.max_model_output_tokens
            ):
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.BUDGET_EXHAUSTED,
                    started_at=started_at,
                    start=start,
                    attempts=0,
                    error_code="model_output_token_budget_exhausted",
                    error_message="run model output-token budget exhausted",
                )
            if not is_model and budget.tool_calls >= context.policy.max_tool_calls:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.BUDGET_EXHAUSTED,
                    started_at=started_at,
                    start=start,
                    attempts=0,
                    error_code="tool_call_budget_exhausted",
                    error_message="run tool-call budget exhausted",
                )
            if not is_model and tool.spec.network_access and budget.network_calls >= context.policy.max_network_calls:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.BUDGET_EXHAUSTED,
                    started_at=started_at,
                    start=start,
                    attempts=0,
                    error_code="network_call_budget_exhausted",
                    error_message="run network-call budget exhausted",
                )
            if is_model:
                budget.model_calls += 1
                budget.model_input_tokens += input_tokens
            else:
                budget.tool_calls += 1
                budget.network_calls += int(tool.spec.network_access)
        attempts = 0
        while attempts < tool.spec.retry.max_attempts:
            attempts += 1
            try:
                data = tool(arguments, context)
                _validate_tool_output(tool.spec, data)
                if is_model:
                    with self._lock:
                        content = data.get("content") if isinstance(data, Mapping) else ""
                        self._budgets[context.run_id].model_output_tokens += _estimated_tokens(str(content))
                elapsed = self._clock() - start
                if elapsed > tool.spec.timeout_seconds:
                    return self._finish(
                        call_id=call_id,
                        tool=tool,
                        context=context,
                        arguments=arguments,
                        status=ToolStatus.TIMEOUT,
                        started_at=started_at,
                        start=start,
                        attempts=attempts,
                        error_code="observed_timeout",
                        error_message=f"tool exceeded {tool.spec.timeout_seconds:g}s timeout",
                    )
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.SUCCESS,
                    started_at=started_at,
                    start=start,
                    attempts=attempts,
                    data=data,
                )
            except ToolContractError as exc:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.ERROR,
                    started_at=started_at,
                    start=start,
                    attempts=attempts,
                    error_code="invalid_tool_result",
                    error_message=str(exc),
                )
            except ToolExecutionError as exc:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.ERROR,
                    started_at=started_at,
                    start=start,
                    attempts=attempts,
                    error_code=exc.code,
                    error_message=str(exc),
                    error_details=exc.details,
                )
            except tool.spec.retry.retryable_exceptions as exc:
                if attempts >= tool.spec.retry.max_attempts:
                    return self._finish(
                        call_id=call_id,
                        tool=tool,
                        context=context,
                        arguments=arguments,
                        status=ToolStatus.ERROR,
                        started_at=started_at,
                        start=start,
                        attempts=attempts,
                        error_code="provider_temporarily_unavailable",
                        error_message=str(exc),
                        error_details={
                            "exception_type": type(exc).__name__,
                            "model_action": "choose_alternative_tool",
                        },
                    )
                if tool.spec.network_access and tool.spec.capability != "model.generate":
                    with self._lock:
                        budget = self._budgets.setdefault(context.run_id, _Budget())
                        if budget.network_calls >= context.policy.max_network_calls:
                            return self._finish(
                                call_id=call_id,
                                tool=tool,
                                context=context,
                                arguments=arguments,
                                status=ToolStatus.BUDGET_EXHAUSTED,
                                started_at=started_at,
                                start=start,
                                attempts=attempts,
                                error_code="network_retry_budget_exhausted",
                                error_message="run network-call budget exhausted before retry",
                            )
                        budget.network_calls += 1
                backoff = tool.spec.retry.initial_backoff_seconds * (
                    tool.spec.retry.backoff_multiplier ** (attempts - 1)
                )
                jitter = backoff * self._random.uniform(0.0, 0.1)
                self._sleep(backoff + jitter)
            except Exception as exc:
                return self._finish(
                    call_id=call_id,
                    tool=tool,
                    context=context,
                    arguments=arguments,
                    status=ToolStatus.ERROR,
                    started_at=started_at,
                    start=start,
                    attempts=attempts,
                    error_code="tool_internal_error",
                    error_message=str(exc),
                    error_details={
                        "exception_type": type(exc).__name__,
                        "model_action": "choose_alternative_tool",
                    },
                )

        raise AssertionError("unreachable")

    def audit_events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events if run_id is None else [event for event in self._events if event.run_id == run_id]
            return [asdict(event) for event in events]

    def budget_usage(self, run_id: str) -> ToolBudgetUsage:
        with self._lock:
            budget = self._budgets.get(run_id, _Budget())
            return ToolBudgetUsage(
                tool_calls=budget.tool_calls,
                network_attempts=budget.network_calls,
                model_calls=budget.model_calls,
                model_input_tokens=budget.model_input_tokens,
                model_output_tokens=budget.model_output_tokens,
            )

    def clear_run(self, run_id: str) -> None:
        with self._lock:
            self._budgets.pop(run_id, None)
            self._run_boundaries.pop(run_id, None)
            self._sequences.pop(run_id, None)
            self._events = [event for event in self._events if event.run_id != run_id]

    def _bind_or_check_run_boundary(self, context: ToolContext) -> str | None:
        proposed = _RunBoundary(
            thread_id=context.thread_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            allow_network=context.policy.allow_network,
            max_tool_calls=context.policy.max_tool_calls,
            max_network_calls=context.policy.max_network_calls,
            max_model_calls=context.policy.max_model_calls,
            max_model_input_tokens=context.policy.max_model_input_tokens,
            max_model_output_tokens=context.policy.max_model_output_tokens,
        )
        with self._lock:
            existing = self._run_boundaries.get(context.run_id)
            if existing is None:
                self._run_boundaries[context.run_id] = proposed
                return None
            if existing != proposed:
                return "run identity, network policy, or budget ceilings changed after the first call"
        return None

    def _reserve_call_number(self, run_id: str) -> int:
        with self._lock:
            next_number = self._sequences.get(run_id, 0) + 1
            self._sequences[run_id] = next_number
            return next_number

    @staticmethod
    def _check_policy(spec: ToolSpec, context: ToolContext) -> tuple[str, str] | None:
        policy = context.policy
        if spec.capability not in policy.allowed_capabilities:
            return "capability_denied", f"capability is not allowed: {spec.capability}"
        if spec.side_effect not in policy.allowed_side_effects:
            return "side_effect_denied", f"side effect is not allowed: {spec.side_effect.value}"
        # 模型调用是研究链路的部署依赖，由 API key / 模型预算约束；
        # allow_network 只约束行情、检索、监管、宏观等数据 provider。
        if spec.network_access and spec.capability != "model.generate" and not policy.allow_network:
            return "network_denied", "network access is not allowed for this run"
        return None

    def _finish_denied(
        self,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        started_at: str,
        start: float,
        context: ToolContext,
        error_code: str,
        error_message: str,
        spec: ToolSpec | None = None,
    ) -> ToolResult:
        placeholder = _CallableTool(
            spec or ToolSpec(name=name, description="unknown", capability="unknown"), lambda _a, _c: None
        )
        return self._finish(
            call_id=call_id,
            tool=placeholder,
            context=context,
            arguments=arguments,
            status=ToolStatus.DENIED,
            started_at=started_at,
            start=start,
            attempts=0,
            error_code=error_code,
            error_message=_redact_error(error_message),
        )

    def _finish(
        self,
        *,
        call_id: str,
        tool: Tool,
        context: ToolContext,
        arguments: Mapping[str, Any],
        status: ToolStatus,
        started_at: str,
        start: float,
        attempts: int,
        data: Any = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_details: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        duration_ms = round((self._clock() - start) * 1000, 3)
        result = ToolResult(
            call_id=call_id,
            tool_name=tool.spec.name,
            status=status,
            started_at=started_at,
            duration_ms=duration_ms,
            attempts=attempts,
            data=data,
            error_code=error_code,
            error_message=_redact_error(error_message),
            error_details=_redact(error_details) if error_details is not None else None,
        )
        event = ToolAuditEvent(
            call_id=call_id,
            run_id=context.run_id,
            thread_id=context.thread_id,
            tenant_id=context.tenant_id,
            tool_name=tool.spec.name,
            capability=tool.spec.capability,
            side_effect=tool.spec.side_effect.value,
            network_access=tool.spec.network_access,
            arguments=_redact(arguments),
            result_status=status.value,
            attempts=attempts,
            budget_consumed=attempts > 0,
            network_attempts=(attempts if tool.spec.network_access and tool.spec.capability != "model.generate" else 0),
            model_input_tokens=(
                _model_input_tokens(arguments)
                if tool.spec.capability == "model.generate" and attempts
                else 0
            ),
            model_output_tokens=(
                _estimated_tokens(str(data.get("content", "")))
                if tool.spec.capability == "model.generate"
                and status is ToolStatus.SUCCESS
                and isinstance(data, Mapping)
                else 0
            ),
            duration_ms=duration_ms,
            timestamp=started_at,
            error_code=error_code,
            error_message=result.error_message,
            result_summary=_result_summary(data) if status is ToolStatus.SUCCESS else None,
        )
        with self._lock:
            self._events.append(event)
        return result


class _CallableTool:
    def __init__(self, spec: ToolSpec, function: Callable[[Mapping[str, Any], ToolContext], Any]) -> None:
        self.spec = spec
        self._function = function

    def __call__(self, arguments: Mapping[str, Any], context: ToolContext) -> Any:
        return self._function(arguments, context)


def function_tool(spec: ToolSpec, function: Callable[[Mapping[str, Any], ToolContext], Any]) -> Tool:
    return _CallableTool(spec, function)


def _json_schema_error(
    error: ValidationError,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any]]:
    path = (*prefix, *(str(item) for item in error.absolute_path))
    field = ".".join(path) or "$"
    rule = str(error.validator or "schema")
    schema: Mapping[str, Any] = error.schema if isinstance(error.schema, Mapping) else {}
    constraint = schema.get(rule)
    if rule == "oneOf" and isinstance(error.instance, Mapping) and isinstance(constraint, list):
        for branch in constraint:
            if not isinstance(branch, Mapping):
                continue
            properties = branch.get("properties")
            if not isinstance(properties, Mapping):
                continue
            discriminators = [
                (str(key), value.get("const"))
                for key, value in properties.items()
                if isinstance(value, Mapping) and "const" in value
            ]
            if not discriminators or not all(error.instance.get(key) == value for key, value in discriminators):
                continue
            nested = next(Draft202012Validator(branch).iter_errors(error.instance), None)
            if nested is not None:
                return _json_schema_error(nested, prefix=path)
    details: dict[str, Any] = {
        "field": field,
        "rule": rule,
        "model_action": "change_arguments",
    }
    if rule == "required" and isinstance(error.instance, Mapping):
        required = [str(item) for item in constraint] if isinstance(constraint, list) else []
        missing = [item for item in required if item not in error.instance]
        details["missing_fields"] = missing
        return f"{field} is missing required fields: {missing}", details
    if rule == "additionalProperties" and isinstance(error.instance, Mapping):
        properties = schema.get("properties")
        known = set(properties) if isinstance(properties, Mapping) else set()
        unknown = sorted(str(item) for item in error.instance if item not in known)
        details["unknown_fields"] = unknown
        return f"{field} contains unknown fields: {unknown}", details
    if rule == "type":
        details["expected_type"] = constraint
        return f"{field} must match JSON type {constraint}", details
    if rule in {"enum", "const"}:
        allowed = constraint if rule == "enum" and isinstance(constraint, list) else [constraint]
        details["allowed_values"] = allowed[:50]
        return f"{field} must use an allowed value", details
    if rule in {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }:
        details["constraint"] = constraint
        return f"{field} violates the {rule} constraint", details
    if rule == "pattern":
        details["pattern"] = str(constraint)
        return f"{field} does not match the required pattern", details
    if rule in {"oneOf", "anyOf", "allOf"}:
        return f"{field} does not match an allowed argument shape", details
    return f"{field} violates the {rule} input constraint", details


_SENSITIVE_CONTENT_KEYS = {"document_content", "raw_document", "system_prompt", "user_prompt"}
_HASHED_CONTENT_KEYS = {"query"}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "***REDACTED***"
                if _is_secret_key(str(key))
                else "***CONTENT_OMITTED***"
                if str(key).lower() in _SENSITIVE_CONTENT_KEYS
                else {
                    "sha256": sha256(str(item).encode("utf-8")).hexdigest(),
                    "length": len(str(item)),
                }
                if str(key).lower() in _HASHED_CONTENT_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return {
                "sha256": sha256(str(value).encode("utf-8")).hexdigest(),
                "length": len(value),
                "content": "***SEQUENCE_OMITTED***",
            }
        return [_redact(item) for item in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…[truncated]"
    return value


def _redact_error(message: str | None) -> str | None:
    if message is None:
        return None
    # Exception strings are provider controlled. Keep the API useful without
    # returning arbitrary multi-kilobyte payloads or common credential formats.
    cleaned = message[:1000]
    cleaned = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer ***REDACTED***", cleaned)
    cleaned = re.sub(r"(?i)\bsk-[a-z0-9_-]+", "sk-***REDACTED***", cleaned)
    cleaned = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|password|secret|authorization)"
        r"\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=***REDACTED***",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)(https?://[^:/@\s]+:)[^@\s]+@",
        r"\1***REDACTED***@",
        cleaned,
    )
    return cleaned


def _result_summary(value: Any) -> dict[str, Any]:
    """Describe a tool return for operations logs without persisting its content."""
    if value is None:
        return {"type": "null"}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        summary: dict[str, Any] = {"type": "object", "keys": sorted(str(key) for key in value)[:50]}
        for key in ("status", "ok", "source_count", "evidence_count", "claim_count"):
            item = value.get(key)
            if isinstance(item, (str, int, float, bool)) or item is None:
                summary[key] = item
        for key in ("sources", "evidence", "claims", "items", "results", "gaps"):
            item = value.get(key)
            if isinstance(item, (list, tuple)):
                summary[f"{key}_count"] = len(item)
        return summary
    if isinstance(value, (list, tuple)):
        return {"type": "array", "item_count": len(value)}
    return {"type": type(value).__name__}


def _model_input_tokens(arguments: Mapping[str, Any]) -> int:
    return _estimated_tokens(str(arguments.get("system_prompt", ""))) + _estimated_tokens(
        str(arguments.get("user_prompt", ""))
    )


def _estimated_tokens(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


def _is_secret_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    return any(marker in normalized for marker in ("apikey", "authorization", "password", "secret", "token"))


def _validate_tool_output(spec: ToolSpec, data: Any) -> None:
    if spec.result_kind == ToolResultKind.ANY:
        return
    if not isinstance(data, Mapping):
        raise ToolContractError(f"{spec.name} returned a non-object result")
    try:
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolContractError(f"{spec.name} returned non-JSON data") from exc
    if len(serialized) > 5_000_000:
        raise ToolContractError(f"{spec.name} result exceeds the serialized size limit")
    if spec.result_kind == ToolResultKind.EVIDENCE_BUNDLE:
        bundle = data.get("bundle")
        if not isinstance(bundle, Mapping):
            raise ToolContractError(f"{spec.name} result has no evidence bundle")
        try:
            EvidenceBundle.from_dict(bundle)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolContractError(f"{spec.name} returned an invalid evidence bundle") from exc
        gaps = data.get("gaps", [])
        if not isinstance(gaps, (list, tuple)) or any(not isinstance(item, Mapping) for item in gaps):
            raise ToolContractError(f"{spec.name} result has malformed gaps")
        return
    if spec.result_kind == ToolResultKind.MODEL_RESPONSE:
        content = data.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 200_000:
            raise ToolContractError(f"{spec.name} returned invalid model content")
        return
    if spec.result_kind == ToolResultKind.CATALOG:
        if len(serialized) > 100_000:
            raise ToolContractError(f"{spec.name} catalog result exceeds the size limit")
        return
