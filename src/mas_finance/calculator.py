"""Deterministic financial calculations that preserve input provenance."""

from __future__ import annotations

import math
from typing import cast

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType

_DERIVED_RATIOS: tuple[tuple[str, str, str], ...] = (
    ("net_margin", "net_income", "revenue"),
    ("gross_margin", "gross_profit", "revenue"),
    ("operating_margin", "operating_income", "revenue"),
    ("liabilities_to_assets", "total_liabilities", "total_assets"),
    ("debt_to_equity", "total_liabilities", "stockholders_equity"),
    ("equity_to_assets", "stockholders_equity", "total_assets"),
    ("cash_to_assets", "cash_and_cash_equivalents", "total_assets"),
    ("current_ratio", "current_assets", "current_liabilities"),
)


class CalculationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def calculate_ratio(
    bundle: EvidenceBundle,
    *,
    metric_name: str,
    numerator_id: str,
    denominator_id: str,
    rounding_digits: int = 6,
) -> Evidence:
    if not 0 <= rounding_digits <= 12:
        raise CalculationError("invalid_rounding", "rounding_digits must be between 0 and 12")
    numerator = _get_numeric_evidence(bundle, numerator_id)
    denominator = _get_numeric_evidence(bundle, denominator_id)
    if numerator.entity != denominator.entity:
        raise CalculationError("entity_mismatch", "ratio inputs refer to different entities")
    if numerator.period != denominator.period:
        raise CalculationError("period_mismatch", "ratio inputs refer to different periods")
    if numerator.unit != denominator.unit:
        raise CalculationError("unit_mismatch", "ratio inputs use different units")
    denominator_value = float(cast(int | float, denominator.value))
    if denominator_value == 0:
        raise CalculationError("division_by_zero", "ratio denominator is zero")
    raw_result = float(cast(int | float, numerator.value)) / denominator_value
    if not math.isfinite(raw_result):
        raise CalculationError("non_finite_result", "calculation produced a non-finite result")
    result = round(raw_result, rounding_digits)

    source = SourceRef.create(
        source_type=SourceType.CALCULATION,
        title=f"Calculated metric: {metric_name}",
        locator=f"formula://ratio/{metric_name}/v1",
        provider="mas_finance.calculator",
        as_of=numerator.period,
        metadata={
            "formula": "numerator / denominator",
            "formula_version": "1",
            "input_evidence_ids": [numerator_id, denominator_id],
            "rounding_digits": rounding_digits,
        },
    )
    evidence = Evidence.create(
        source=source,
        content=(
            f"{metric_name} = {numerator.value} / {denominator.value} = {result}; "
            f"inputs: {numerator_id}, {denominator_id}"
        ),
        entity=numerator.entity,
        field_name=metric_name,
        value=result,
        unit="ratio",
        period=numerator.period,
        confidence=min(numerator.confidence, denominator.confidence),
        tags=("calculation", "ratio"),
    )
    bundle.add_evidence(evidence)
    return evidence


def derive_standard_ratios(bundle: EvidenceBundle) -> list[Evidence]:
    """Derive conservative same-entity/same-period ratios from normalized facts."""
    candidates: dict[tuple[str | None, str | None, str], list[Evidence]] = {}
    for item in list(bundle.evidence.values()):
        if item.field_name and isinstance(item.value, (int, float)) and not isinstance(item.value, bool):
            candidates.setdefault((item.entity, item.period, item.field_name), []).append(item)
    indexed = {
        key: values[0] for key, values in candidates.items() if len({(item.value, item.unit) for item in values}) == 1
    }

    derived: list[Evidence] = []
    entity_periods = {(entity, period) for entity, period, _field in indexed}
    for entity, period in entity_periods:
        for metric_name, numerator_field, denominator_field in _DERIVED_RATIOS:
            if indexed.get((entity, period, metric_name)) is not None:
                continue
            numerator = indexed.get((entity, period, numerator_field))
            denominator = indexed.get((entity, period, denominator_field))
            if numerator is None or denominator is None:
                continue
            try:
                result = calculate_ratio(
                    bundle,
                    metric_name=metric_name,
                    numerator_id=numerator.evidence_id,
                    denominator_id=denominator.evidence_id,
                )
            except CalculationError:
                continue
            derived.append(result)
    return derived


def _get_numeric_evidence(bundle: EvidenceBundle, evidence_id: str) -> Evidence:
    try:
        evidence = bundle.evidence[evidence_id]
    except KeyError as exc:
        raise CalculationError("input_not_found", f"unknown evidence: {evidence_id}") from exc
    value = evidence.value
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise CalculationError("input_not_numeric", f"evidence is not a finite number: {evidence_id}")
    return evidence
