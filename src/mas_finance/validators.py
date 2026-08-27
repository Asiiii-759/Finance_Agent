"""Deterministic validation gates for evidence bundles and rendered reports."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .contracts import ClaimStatus, EvidenceBundle, SourceType


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    claim_id: str | None = None
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [issue.to_dict() for issue in self.issues]}


_REQUIRED_SECTIONS = (
    "## Supported findings",
    "## Conflicts and caveats",
    "## Retrieved document evidence",
    "## Data gaps",
    "## Sources",
    "## Risk notice",
)
_CITATION_PATTERN = re.compile(r"\[\^([^\]]+)\]")


def validate_research_output(
    *,
    bundle: EvidenceBundle,
    report: str,
    gaps: Iterable[Mapping[str, Any]] = (),
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for section in _REQUIRED_SECTIONS:
        if section not in report:
            issues.append(
                ValidationIssue(
                    code="report_section_missing",
                    severity=Severity.ERROR,
                    message=f"Required report section is missing: {section}",
                )
            )

    report_citations = set(_CITATION_PATTERN.findall(report))
    known_evidence = set(bundle.evidence)
    unknown_citations = report_citations.difference(known_evidence)
    for evidence_id in sorted(unknown_citations):
        issues.append(
            ValidationIssue(
                code="unknown_report_citation",
                severity=Severity.ERROR,
                message="Report cites evidence that is not present in the ledger.",
                evidence_id=evidence_id,
            )
        )

    for claim in bundle.claims.values():
        cited_items = [bundle.evidence[item] for item in claim.evidence_ids if item in bundle.evidence]
        if claim.status == ClaimStatus.SUPPORTED and not cited_items:
            issues.append(
                ValidationIssue(
                    code="supported_claim_without_evidence",
                    severity=Severity.ERROR,
                    message="Supported claim has no available evidence.",
                    claim_id=claim.claim_id,
                )
            )
            continue
        for evidence_id in claim.evidence_ids:
            if evidence_id not in report_citations:
                issues.append(
                    ValidationIssue(
                        code="claim_citation_missing",
                        severity=Severity.ERROR,
                        message="A supported claim is not cited in the rendered report.",
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                    )
                )
            footnote = f"[^{evidence_id}]:"
            if footnote not in report:
                issues.append(
                    ValidationIssue(
                        code="source_footnote_missing",
                        severity=Severity.ERROR,
                        message="Cited evidence has no source footnote.",
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                    )
                )

    for evidence in bundle.evidence.values():
        if evidence.source.source_type != SourceType.CALCULATION:
            continue
        input_ids = evidence.source.metadata.get("input_evidence_ids") or ()
        if not isinstance(input_ids, (list, tuple)) or not input_ids:
            issues.append(
                ValidationIssue(
                    code="calculation_inputs_missing",
                    severity=Severity.ERROR,
                    message="Calculation evidence does not declare input evidence IDs.",
                    evidence_id=evidence.evidence_id,
                )
            )
            continue
        for input_id in input_ids:
            if str(input_id) not in bundle.evidence:
                issues.append(
                    ValidationIssue(
                        code="calculation_input_unknown",
                        severity=Severity.ERROR,
                        message="Calculation evidence references an unknown input.",
                        evidence_id=evidence.evidence_id,
                    )
                )

    gap_list = list(gaps)
    if gap_list and "## Data gaps" in report:
        for gap in gap_list:
            code = str(gap.get("code") or "data_gap")
            if f"[{code}]" not in report:
                issues.append(
                    ValidationIssue(
                        code="data_gap_not_rendered",
                        severity=Severity.ERROR,
                        message=f"Data gap is missing from report: {code}",
                    )
                )

    if "does not constitute investment advice" not in report.lower():
        issues.append(
            ValidationIssue(
                code="risk_notice_missing",
                severity=Severity.ERROR,
                message="Investment-advice risk notice is missing.",
            )
        )
    if not bundle.evidence:
        issues.append(
            ValidationIssue(
                code="no_evidence",
                severity=Severity.WARNING,
                message="No retrieved evidence was available; conceptual claims were not retrieval-checked.",
            )
        )
    return ValidationResult(tuple(issues))
