"""Deterministic financial-math tools with explicit formulas and provenance."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType, stable_id
from .harness import Tool, ToolArgumentContract, ToolResultKind, ToolSpec, function_tool


class MetricOperation(StrEnum):
    RATIO = "ratio"
    PERCENTAGE_CHANGE = "percentage_change"
    CAGR = "cagr"
    FUTURE_VALUE = "future_value"
    PRESENT_VALUE = "present_value"
    LOAN_PAYMENT = "loan_payment"
    ANNUALIZED_RETURN = "annualized_return"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"


_FORMULAS: dict[MetricOperation, str] = {
    MetricOperation.RATIO: "numerator / denominator",
    MetricOperation.PERCENTAGE_CHANGE: "ending_value / beginning_value - 1",
    MetricOperation.CAGR: "(ending_value / beginning_value) ** (1 / years) - 1",
    MetricOperation.FUTURE_VALUE: "present_value * (1 + rate) ** periods",
    MetricOperation.PRESENT_VALUE: "future_value / (1 + rate) ** periods",
    MetricOperation.LOAN_PAYMENT: "principal * rate * (1 + rate) ** periods / ((1 + rate) ** periods - 1)",
    MetricOperation.ANNUALIZED_RETURN: "product(1 + periodic_return) ** (annualization_factor / n) - 1",
    MetricOperation.ANNUALIZED_VOLATILITY: "sample_stdev(periodic_returns) * sqrt(annualization_factor)",
    MetricOperation.SHARPE_RATIO: "(annualized_return - annual_risk_free_rate) / annualized_volatility",
    MetricOperation.MAX_DRAWDOWN: "min(value / running_peak - 1)",
}

_INPUT_KEYS: dict[MetricOperation, frozenset[str]] = {
    MetricOperation.RATIO: frozenset({"numerator", "denominator"}),
    MetricOperation.PERCENTAGE_CHANGE: frozenset({"beginning_value", "ending_value"}),
    MetricOperation.CAGR: frozenset({"beginning_value", "ending_value", "years"}),
    MetricOperation.FUTURE_VALUE: frozenset({"present_value", "rate", "periods"}),
    MetricOperation.PRESENT_VALUE: frozenset({"future_value", "rate", "periods"}),
    MetricOperation.LOAN_PAYMENT: frozenset({"principal", "rate", "periods"}),
    MetricOperation.ANNUALIZED_RETURN: frozenset({"returns", "annualization_factor"}),
    MetricOperation.ANNUALIZED_VOLATILITY: frozenset({"returns", "annualization_factor"}),
    MetricOperation.SHARPE_RATIO: frozenset({"returns", "annualization_factor", "annual_risk_free_rate"}),
    MetricOperation.MAX_DRAWDOWN: frozenset({"values"}),
}

_REQUIRED_INPUT_KEYS: dict[MetricOperation, frozenset[str]] = {
    operation: keys for operation, keys in _INPUT_KEYS.items()
}
_REQUIRED_INPUT_KEYS[MetricOperation.SHARPE_RATIO] = frozenset({"returns", "annualization_factor"})

_DEFAULT_UNITS: dict[MetricOperation, str] = {
    MetricOperation.RATIO: "ratio",
    MetricOperation.PERCENTAGE_CHANGE: "ratio",
    MetricOperation.CAGR: "ratio_per_year",
    MetricOperation.FUTURE_VALUE: "currency",
    MetricOperation.PRESENT_VALUE: "currency",
    MetricOperation.LOAN_PAYMENT: "currency_per_period",
    MetricOperation.ANNUALIZED_RETURN: "ratio_per_year",
    MetricOperation.ANNUALIZED_VOLATILITY: "ratio_per_year",
    MetricOperation.SHARPE_RATIO: "ratio",
    MetricOperation.MAX_DRAWDOWN: "ratio",
}

_REQUEST_KEYS = frozenset({"operation", "inputs", "label", "entity", "unit", "period", "request_id"})


class MetricError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def describe_metric_operations() -> dict[str, dict[str, Any]]:
    """Public, JSON-safe function contract for callers and model tool schemas."""
    return {
        operation.value: {
            "required_inputs": sorted(_REQUIRED_INPUT_KEYS[operation]),
            "optional_inputs": sorted(_INPUT_KEYS[operation] - _REQUIRED_INPUT_KEYS[operation]),
            "formula": _FORMULAS[operation],
            "default_unit": _DEFAULT_UNITS[operation],
        }
        for operation in MetricOperation
    }


@dataclass(frozen=True)
class MetricRequest:
    operation: MetricOperation
    inputs: Mapping[str, Any]
    label: str | None = None
    entity: str | None = None
    unit: str | None = None
    period: str | None = None
    request_id: str = ""

    def __post_init__(self) -> None:
        if len(self.inputs) > 20:
            raise MetricError("too_many_inputs", "a calculation accepts at most 20 named inputs")
        unknown = set(self.inputs).difference(_INPUT_KEYS[self.operation])
        if unknown:
            raise MetricError(
                "unexpected_input",
                f"calculation contains unsupported inputs: {sorted(str(item) for item in unknown)}",
            )
        missing = _REQUIRED_INPUT_KEYS[self.operation].difference(self.inputs)
        if missing:
            raise MetricError(
                "missing_input",
                f"calculation is missing required inputs: {sorted(missing)}",
            )
        for name, value, limit in (
            ("label", self.label, 100),
            ("entity", self.entity, 200),
            ("unit", self.unit, 50),
            ("period", self.period, 100),
        ):
            if value is not None and (not value.strip() or len(value) > limit):
                raise MetricError("invalid_text_field", f"calculation {name} is invalid")
        _validate_output_unit(self.operation, self.unit)
        try:
            serialized_inputs = json.dumps(
                dict(self.inputs),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise MetricError("invalid_inputs", "calculation inputs must be finite JSON values") from exc
        if len(serialized_inputs) > 200_000:
            raise MetricError("inputs_too_large", "serialized calculation inputs exceed 200000 characters")
        generated = stable_id(
            "metric",
            {
                "operation": self.operation.value,
                "inputs": dict(self.inputs),
                "label": self.label,
                "entity": self.entity,
                "unit": self.unit,
                "period": self.period,
            },
        )
        if not self.request_id:
            object.__setattr__(self, "request_id", generated)
        elif not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", self.request_id):
            raise MetricError("invalid_request_id", "calculation request_id is invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetricRequest:
        unknown = set(value).difference(_REQUEST_KEYS)
        if unknown:
            raise MetricError(
                "unexpected_request_field",
                f"calculation request contains unsupported fields: {sorted(str(item) for item in unknown)}",
            )
        inputs = value.get("inputs") or {}
        if not isinstance(inputs, Mapping):
            raise MetricError("invalid_inputs", "calculation inputs must be an object")
        raw_operation = value.get("operation")
        if not isinstance(raw_operation, str):
            raise MetricError("unsupported_operation", "calculation operation must be a string")
        try:
            operation = MetricOperation(raw_operation)
        except ValueError as exc:
            raise MetricError("unsupported_operation", "calculation operation is unsupported") from exc
        request_id = value.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise MetricError("invalid_request_id", "calculation request_id must be a string")
        return cls(
            operation=operation,
            inputs=dict(inputs),
            label=_optional_text(value.get("label")),
            entity=_optional_text(value.get("entity")),
            unit=_optional_text(value.get("unit")),
            period=_optional_text(value.get("period")),
            request_id=request_id or "",
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        value["inputs"] = dict(self.inputs)
        return value


@dataclass(frozen=True)
class MetricResult:
    request_id: str
    operation: MetricOperation
    value: float
    formula: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        return value


@dataclass(frozen=True)
class MetricBatch:
    bundle: EvidenceBundle
    results: tuple[MetricResult, ...]
    gaps: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "results": [item.to_dict() for item in self.results],
            "gaps": list(self.gaps),
        }


def calculate_metric(request: MetricRequest) -> MetricResult:
    """Evaluate one allowlisted financial formula without arbitrary expressions."""
    operation = request.operation
    inputs = request.inputs
    if operation == MetricOperation.RATIO:
        result = _scalar(inputs, "numerator") / _nonzero(inputs, "denominator")
        unit = request.unit or "ratio"
    elif operation == MetricOperation.PERCENTAGE_CHANGE:
        result = _scalar(inputs, "ending_value") / _nonzero(inputs, "beginning_value") - 1
        unit = request.unit or "ratio"
    elif operation == MetricOperation.CAGR:
        beginning = _scalar(inputs, "beginning_value")
        ending = _scalar(inputs, "ending_value")
        years = _positive(inputs, "years")
        if beginning <= 0 or ending < 0:
            raise MetricError("invalid_growth_domain", "CAGR requires beginning > 0 and ending >= 0")
        result = _safe_power(ending / beginning, 1 / years) - 1
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.FUTURE_VALUE:
        present = _scalar(inputs, "present_value")
        rate = _rate(inputs, "rate")
        periods = _nonnegative(inputs, "periods")
        result = present * _safe_power(1 + rate, periods)
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.PRESENT_VALUE:
        future = _scalar(inputs, "future_value")
        rate = _rate(inputs, "rate")
        periods = _nonnegative(inputs, "periods")
        result = future / _safe_power(1 + rate, periods)
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.LOAN_PAYMENT:
        principal = _nonnegative(inputs, "principal")
        rate = _rate(inputs, "rate")
        periods = _positive(inputs, "periods")
        if rate == 0:
            result = principal / periods
        else:
            growth = _safe_power(1 + rate, periods)
            result = principal * rate * growth / (growth - 1)
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.ANNUALIZED_RETURN:
        returns = _series(inputs, "returns", minimum=1)
        _validate_returns(returns)
        factor = _positive(inputs, "annualization_factor")
        compounded = math.prod(1 + item for item in returns)
        if compounded < 0:
            raise MetricError("invalid_return_domain", "compounded return is outside the real-valued domain")
        result = _safe_power(compounded, factor / len(returns)) - 1
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.ANNUALIZED_VOLATILITY:
        returns = _series(inputs, "returns", minimum=2)
        _validate_returns(returns)
        factor = _positive(inputs, "annualization_factor")
        result = statistics.stdev(returns) * math.sqrt(factor)
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.SHARPE_RATIO:
        returns = _series(inputs, "returns", minimum=2)
        _validate_returns(returns)
        factor = _positive(inputs, "annualization_factor")
        risk_free = _scalar(inputs, "annual_risk_free_rate", default=0.0)
        compounded = math.prod(1 + item for item in returns)
        annual_return = _safe_power(compounded, factor / len(returns)) - 1
        volatility = statistics.stdev(returns) * math.sqrt(factor)
        if volatility == 0:
            raise MetricError("division_by_zero", "Sharpe ratio requires non-zero volatility")
        result = (annual_return - risk_free) / volatility
        unit = request.unit or _DEFAULT_UNITS[operation]
    elif operation == MetricOperation.MAX_DRAWDOWN:
        values = _series(inputs, "values", minimum=1)
        if any(item <= 0 for item in values):
            raise MetricError("invalid_price_series", "max drawdown values must be positive")
        peak = values[0]
        result = 0.0
        for item in values:
            peak = max(peak, item)
            result = min(result, item / peak - 1)
        unit = request.unit or _DEFAULT_UNITS[operation]
    else:  # pragma: no cover - exhaustive enum guard
        raise MetricError("unsupported_operation", f"unsupported operation: {operation}")
    if not math.isfinite(result):
        raise MetricError("non_finite_result", "calculation produced a non-finite result")
    return MetricResult(
        request_id=request.request_id,
        operation=operation,
        value=round(result, 10),
        formula=_FORMULAS[operation],
        unit=unit,
    )


class MetricEvidenceAdapter:
    """Turn explicit numeric inputs and deterministic results into cited evidence."""

    def calculate(self, requests: Sequence[MetricRequest]) -> MetricBatch:
        if not 1 <= len(requests) <= 20:
            raise MetricError("invalid_request_count", "provide between 1 and 20 calculations")
        bundle = EvidenceBundle()
        results: list[MetricResult] = []
        gaps: list[dict[str, Any]] = []
        for request in requests:
            try:
                result = calculate_metric(request)
            except MetricError as exc:
                gaps.append(
                    {
                        "code": exc.code,
                        "message": str(exc),
                        "request_id": request.request_id,
                    }
                )
                continue
            input_source = SourceRef.create(
                source_type=SourceType.USER_INPUT,
                title=f"User-supplied inputs for {request.operation.value}",
                locator=f"request://calculation/{request.request_id}/inputs",
                provider="user_request",
                as_of=request.period,
                metadata={"request_id": request.request_id, "operation": request.operation.value},
            )
            input_evidence = Evidence.create(
                source=input_source,
                content=json.dumps(dict(request.inputs), ensure_ascii=False, sort_keys=True),
                entity=request.entity,
                field_name=f"{request.operation.value}_inputs",
                period=request.period,
                confidence=1.0,
                tags=("user_input", "calculation_input", request.request_id),
            )
            bundle.add_evidence(input_evidence)
            source = SourceRef.create(
                source_type=SourceType.CALCULATION,
                title=f"Calculated metric: {request.label or request.operation.value}",
                locator=f"formula://{request.operation.value}/v1/{request.request_id}",
                provider="mas_finance.metrics",
                as_of=request.period,
                metadata={
                    "request_id": request.request_id,
                    "operation": request.operation.value,
                    "formula": result.formula,
                    "input_evidence_ids": [input_evidence.evidence_id],
                    "formula_version": "1",
                },
            )
            evidence = Evidence.create(
                source=source,
                content=(
                    f"{request.label or request.operation.value} = {result.value} {result.unit}; "
                    f"formula: {result.formula}; input: {input_evidence.evidence_id}."
                ),
                entity=request.entity,
                field_name=request.label or request.operation.value,
                value=result.value,
                unit=result.unit,
                period=request.period,
                confidence=1.0,
                tags=("calculation", request.operation.value, request.request_id),
            )
            bundle.add_evidence(evidence)
            results.append(result)
        return MetricBatch(bundle=bundle, results=tuple(results), gaps=tuple(gaps))


def financial_calculation_harness_tool(adapter: MetricEvidenceAdapter | None = None) -> Tool:
    calculator = adapter or MetricEvidenceAdapter()

    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        values = arguments.get("requests")
        if not isinstance(values, list):
            raise MetricError("invalid_requests", "requests must be a list")
        requests = [MetricRequest.from_dict(item) for item in values if isinstance(item, Mapping)]
        if len(requests) != len(values):
            raise MetricError("invalid_requests", "every calculation request must be an object")
        return calculator.calculate(requests).to_dict()

    return function_tool(
        ToolSpec(
            name="finance.calculate",
            description=(
                "计算白名单金融公式，包括 CAGR、百分比变化、现值/终值、贷款还款额、年化收益/波动率、"
                "夏普比率和最大回撤。"
            ),
            capability="calculation",
            network_access=False,
            timeout_seconds=5,
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(required=frozenset({"requests"})),
        ),
        invoke,
    )


def infer_metric_requests(query: str) -> tuple[MetricRequest, ...]:
    """Conservatively infer only unambiguous calculations from natural language."""
    normalized = query.casefold().replace("，", ",")
    values = _named_numbers(normalized)
    inferred: list[MetricRequest] = []
    from_to = _from_to_numbers(normalized)
    cagr_requested = any(item in normalized for item in ("cagr", "复合增长率", "年复合增长率"))
    if cagr_requested:
        beginning = _first_named(values, "beginning", "begin", "start", "initial", "期初", "初始")
        ending = _first_named(values, "ending", "end", "final", "期末", "最终")
        years = _first_named(values, "years", "year", "年数", "年")
        if from_to is not None:
            beginning = beginning if beginning is not None else from_to[0]
            ending = ending if ending is not None else from_to[1]
            years = years if years is not None else from_to[2]
        if beginning is not None and ending is not None and years is not None:
            inferred.append(
                MetricRequest(
                    operation=MetricOperation.CAGR,
                    inputs={"beginning_value": beginning, "ending_value": ending, "years": years},
                    label="cagr",
                )
            )
    if not cagr_requested and any(
        item in normalized for item in ("percentage change", "percent change", "增长率", "变化率")
    ):
        beginning = _first_named(values, "beginning", "begin", "start", "initial", "期初", "初始")
        ending = _first_named(values, "ending", "end", "final", "期末", "最终")
        if from_to is not None:
            beginning = beginning if beginning is not None else from_to[0]
            ending = ending if ending is not None else from_to[1]
        if beginning is not None and ending is not None:
            inferred.append(
                MetricRequest(
                    operation=MetricOperation.PERCENTAGE_CHANGE,
                    inputs={"beginning_value": beginning, "ending_value": ending},
                    label="percentage_change",
                )
            )
    return tuple(inferred)


def _from_to_numbers(text: str) -> tuple[float, float, float | None] | None:
    number = r"([-+]?\d+(?:\.\d+)?)"
    pair_patterns = (
        rf"from\s*{number}\s*to\s*{number}",
        rf"从\s*{number}\s*(?:增长|增加|上升|变为|涨)?\s*到\s*{number}",
        rf"{number}\s*(?:增长|增加|上升|变为|涨)\s*到\s*{number}",
    )
    match = next((value for pattern in pair_patterns if (value := re.search(pattern, text))), None)
    if match is None:
        return None
    years_match = re.search(
        rf"(?:over|during|in|用了|经过|历时)?\s*{number}\s*(?:years?|年)",
        text,
        flags=re.IGNORECASE,
    )
    return (
        float(match.group(1)),
        float(match.group(2)),
        float(years_match.group(1)) if years_match else None,
    )


def _named_numbers(text: str) -> dict[str, float]:
    pattern = re.compile(
        r"([a-z_\u4e00-\u9fff]{1,24})\s*(?:=|:|：)\s*([-+]?\d+(?:\.\d+)?)\s*(%)?",
        re.IGNORECASE,
    )
    result: dict[str, float] = {}
    for name, raw, percentage in pattern.findall(text):
        value = float(raw)
        result[name.casefold()] = value / 100 if percentage else value
    return result


def _first_named(values: Mapping[str, float], *names: str) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def _scalar(inputs: Mapping[str, Any], name: str, *, default: float | None = None) -> float:
    if name not in inputs:
        if default is not None:
            return default
        raise MetricError("missing_input", f"missing calculation input: {name}")
    value = inputs[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise MetricError("invalid_input", f"calculation input must be a finite number: {name}")
    parsed = float(value)
    if abs(parsed) > 1e100:
        raise MetricError("invalid_input", f"calculation input magnitude is too large: {name}")
    return parsed


def _positive(inputs: Mapping[str, Any], name: str) -> float:
    value = _scalar(inputs, name)
    if value <= 0:
        raise MetricError("invalid_input", f"calculation input must be positive: {name}")
    if value > 1_000_000:
        raise MetricError("invalid_input", f"calculation input exceeds the supported range: {name}")
    return value


def _nonnegative(inputs: Mapping[str, Any], name: str) -> float:
    value = _scalar(inputs, name)
    if value < 0:
        raise MetricError("invalid_input", f"calculation input must be non-negative: {name}")
    if name == "periods" and value > 1_000_000:
        raise MetricError("invalid_input", "calculation periods exceed the supported range")
    return value


def _nonzero(inputs: Mapping[str, Any], name: str) -> float:
    value = _scalar(inputs, name)
    if value == 0:
        raise MetricError("division_by_zero", f"calculation input must be non-zero: {name}")
    return value


def _rate(inputs: Mapping[str, Any], name: str) -> float:
    value = _scalar(inputs, name)
    if value <= -1:
        raise MetricError("invalid_rate", f"rate must be greater than -1: {name}")
    return value


def _series(inputs: Mapping[str, Any], name: str, *, minimum: int) -> list[float]:
    raw = inputs.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MetricError("invalid_series", f"calculation input must be a numeric series: {name}")
    if not minimum <= len(raw) <= 10_000:
        raise MetricError("invalid_series_length", f"{name} must contain between {minimum} and 10000 values")
    values: list[float] = []
    for item in raw:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
            raise MetricError("invalid_series", f"{name} contains a non-finite value")
        parsed = float(item)
        if abs(parsed) > 1e100:
            raise MetricError("invalid_series", f"{name} contains an out-of-range value")
        values.append(parsed)
    return values


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise MetricError("invalid_text_field", "calculation text fields must be strings")
    return value.strip()


def _validate_returns(values: Sequence[float]) -> None:
    if any(item <= -1 for item in values):
        raise MetricError("invalid_return", "periodic returns must be greater than -1")


def _safe_power(base: float, exponent: float) -> float:
    try:
        value = base**exponent
    except (OverflowError, ValueError) as exc:
        raise MetricError("numeric_overflow", "calculation exponent is outside the supported range") from exc
    if not math.isfinite(value):
        raise MetricError("numeric_overflow", "calculation exponent produced a non-finite value")
    return value


def _validate_output_unit(operation: MetricOperation, unit: str | None) -> None:
    if unit is None:
        return
    expected = _DEFAULT_UNITS[operation]
    valid = unit == expected
    if operation in {MetricOperation.FUTURE_VALUE, MetricOperation.PRESENT_VALUE}:
        valid = valid or bool(re.fullmatch(r"[A-Z]{3}", unit))
    elif operation == MetricOperation.LOAN_PAYMENT:
        valid = valid or bool(re.fullmatch(r"[A-Z]{3}_per_period", unit))
    if not valid:
        raise MetricError(
            "unit_mismatch",
            f"unit {unit!r} is incompatible with calculation operation {operation.value}",
        )
