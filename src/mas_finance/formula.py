"""Safe declarative formulas proposed by a model or supplied by a user."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType, stable_id
from .harness import Tool, ToolArgumentContract, ToolExecutionError, ToolResultKind, ToolSpec, function_tool

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "min": min,
    "max": max,
}


def evaluate_formula(expression: str, inputs: Mapping[str, Any]) -> float:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        raise ValueError("formula expression must contain 1-500 characters")
    if not isinstance(inputs, Mapping) or not 1 <= len(inputs) <= 30:
        raise ValueError("formula inputs must contain 1-30 variables")
    variables: dict[str, float] = {}
    for name, value in inputs.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError("formula variable name is invalid")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"formula input must be finite numeric data: {name}")
        variables[name] = float(value)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("formula expression has invalid syntax") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 100:
        raise ValueError("formula expression is too complex")
    value = _evaluate_node(tree.body, variables, depth=0)
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("formula result is non-finite or outside the supported range")
    return round(value, 10)


def formula_harness_tool() -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        try:
            expression = arguments.get("expression")
            inputs = arguments.get("inputs")
            label = arguments.get("label", "custom_formula")
            unit = arguments.get("unit", "unspecified")
            entity = arguments.get("entity")
            period = arguments.get("period")
            if not isinstance(expression, str) or not isinstance(inputs, Mapping):
                raise ValueError("formula expression and inputs have invalid types")
            for name, value, limit in (
                ("label", label, 100),
                ("unit", unit, 50),
                ("entity", entity, 200),
                ("period", period, 100),
            ):
                if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > limit):
                    raise ValueError(f"formula {name} is invalid")
            result = evaluate_formula(expression, inputs)
        except ValueError as exc:
            raise ToolExecutionError(
                "invalid_formula",
                str(exc),
                details={"model_action": "change_arguments"},
            ) from exc
        request_id = stable_id(
            "formula",
            {"expression": expression, "inputs": dict(inputs), "label": label, "unit": unit, "entity": entity},
        )
        bundle = EvidenceBundle()
        input_source = SourceRef.create(
            source_type=SourceType.USER_INPUT,
            title=f"User/model supplied inputs for {label}",
            locator=f"request://formula/{request_id}/inputs",
            provider="user_or_model_parameters",
            as_of=period,
        )
        input_evidence = Evidence.create(
            source=input_source,
            content=json.dumps(dict(inputs), ensure_ascii=False, sort_keys=True),
            entity=entity,
            field_name=f"{label}_inputs",
            period=period,
            tags=("user_input", "formula_input", request_id),
        )
        bundle.add_evidence(input_evidence)
        result_source = SourceRef.create(
            source_type=SourceType.CALCULATION,
            title=f"Declarative formula: {label}",
            locator=f"formula://declarative/v1/{request_id}",
            provider="mas_finance.formula",
            as_of=period,
            metadata={
                "expression": expression,
                "input_evidence_ids": [input_evidence.evidence_id],
                "formula_version": "declarative-v1",
            },
        )
        bundle.add_evidence(
            Evidence.create(
                source=result_source,
                content=f"{label} = {result} {unit}; formula: {expression}; input: {input_evidence.evidence_id}.",
                entity=entity,
                field_name=label,
                value=result,
                unit=unit,
                period=period,
                tags=("calculation", "declarative_formula", request_id),
            )
        )
        return {"bundle": bundle.to_dict(), "gaps": []}

    return function_tool(
        ToolSpec(
            name="finance.formula",
            description=(
                "使用具名有限数值输入计算模型设计的声明式数值公式。仅支持 +、-、*、/、有界幂、括号、"
                "abs、sqrt、log、exp、min 和 max；绝不执行代码。"
            ),
            capability="calculation",
            timeout_seconds=2,
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"expression", "inputs"}),
                optional=frozenset({"label", "unit", "entity", "period"}),
                max_serialized_characters=20_000,
            ),
            input_schema={
                "type": "object",
                "required": ["expression", "inputs"],
                "additionalProperties": False,
                "properties": {
                    "expression": {"type": "string", "minLength": 1, "maxLength": 500},
                    "inputs": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 30,
                        "propertyNames": {"pattern": "^[A-Za-z][A-Za-z0-9_]{0,63}$"},
                        "additionalProperties": {"type": "number"},
                    },
                    "label": {"type": "string", "minLength": 1, "maxLength": 100},
                    "unit": {"type": "string", "minLength": 1, "maxLength": 50},
                    "entity": {"type": "string", "minLength": 1, "maxLength": 200},
                    "period": {"type": "string", "minLength": 1, "maxLength": 100},
                },
            },
        ),
        invoke,
    )


def _evaluate_node(node: ast.AST, variables: Mapping[str, float], *, depth: int) -> float:
    if depth > 20:
        raise ValueError("formula expression is too deeply nested")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("formula constants must be numeric")
        value = float(node.value)
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("formula constant is outside the supported range")
        return value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"formula references an unknown variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return float(_UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, variables, depth=depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, variables, depth=depth + 1)
        right = _evaluate_node(node.right, variables, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e50):
            raise ValueError("formula exponentiation is outside the supported range")
        try:
            return float(_BINARY_OPERATORS[type(node.op)](left, right))
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("formula cannot be evaluated in the real finite domain") from exc
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if node.keywords or not 1 <= len(node.args) <= 20:
            raise ValueError("formula function arguments are invalid")
        values = [_evaluate_node(item, variables, depth=depth + 1) for item in node.args]
        try:
            return float(_FUNCTIONS[node.func.id](*values))
        except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("formula function cannot be evaluated in the real finite domain") from exc
    raise ValueError(f"formula contains a forbidden construct: {type(node).__name__}")
