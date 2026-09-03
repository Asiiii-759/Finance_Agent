"""Typed research scope produced by the mandatory LLM task interpreter."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FinancialIntent(StrEnum):
    DOCUMENT_RESEARCH = "document_research"
    MARKET_SNAPSHOT = "market_snapshot"
    MARKET_PERFORMANCE = "market_performance"
    FUNDAMENTALS = "fundamentals"
    VALUATION = "valuation"
    PROFITABILITY = "profitability"
    SOLVENCY = "solvency"
    LIQUIDITY = "liquidity"
    MACROECONOMICS = "macroeconomics"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    RISK = "risk"
    GENERAL_RESEARCH = "general_research"
    FINANCIAL_EDUCATION = "financial_education"
    REGULATORY_FILINGS = "regulatory_filings"


@dataclass(frozen=True)
class ResearchRequirement:
    key: str
    category: str
    reason: str
    entity: str | None = None
    fields: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fields": list(self.fields),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchRequirement:
        return cls(
            key=str(value["key"]),
            category=str(value["category"]),
            reason=str(value.get("reason") or ""),
            entity=str(value["entity"]) if value.get("entity") is not None else None,
            fields=tuple(str(item) for item in value.get("fields") or ()),
            parameters=dict(value.get("parameters") or {}),
        )


@dataclass(frozen=True)
class ResearchScope:
    intents: tuple[FinancialIntent, ...]
    requirements: tuple[ResearchRequirement, ...]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intents": [item.value for item in self.intents],
            "requirements": [item.to_dict() for item in self.requirements],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchScope:
        return cls(
            intents=tuple(FinancialIntent(str(item)) for item in value.get("intents") or ()),
            requirements=tuple(ResearchRequirement.from_dict(item) for item in value.get("requirements") or ()),
            rationale=str(value.get("rationale") or ""),
        )


def validate_macro_series(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(item.strip().upper() for item in values if item.strip()))
    if len(normalized) > 20:
        raise ValueError("at most 20 macro series may be requested")
    if any(not re.fullmatch(r"[A-Z0-9_.-]{1,64}", item) for item in normalized):
        raise ValueError("macro series identifiers contain invalid characters")
    return normalized
