"""SEC EDGAR company-facts provider and evidence adapter."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from .harness import (
    RetryPolicy,
    Tool,
    ToolArgumentContract,
    ToolResultKind,
    ToolSpec,
    function_tool,
)

_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
    ("Revenues", "Revenue"),
    ("GrossProfit", "Gross profit"),
    ("OperatingIncomeLoss", "Operating income"),
    ("NetIncomeLoss", "Net income"),
    ("Assets", "Total assets"),
    ("AssetsCurrent", "Current assets"),
    ("Liabilities", "Total liabilities"),
    ("LiabilitiesCurrent", "Current liabilities"),
    ("StockholdersEquity", "Stockholders' equity"),
    ("CashAndCashEquivalentsAtCarryingValue", "Cash and cash equivalents"),
    ("NetCashProvidedByUsedInOperatingActivities", "Operating cash flow"),
    ("PaymentsToAcquirePropertyPlantAndEquipment", "Capital expenditure"),
    ("EarningsPerShareDiluted", "Diluted earnings per share"),
)


class SECCompanyFactsClient:
    """Fixed-endpoint read-only client compliant with SEC declared-agent access."""

    def __init__(
        self,
        user_agent: str,
        *,
        timeout_seconds: float = 30.0,
        minimum_request_interval: float = 0.12,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError("SEC user agent must identify an organization and contact email")
        self.user_agent = user_agent.strip()
        self.timeout_seconds = timeout_seconds
        self.minimum_request_interval = minimum_request_interval
        self._ticker_ciks: dict[str, int] | None = None
        self._last_request = 0.0
        self._lock = threading.RLock()

    def fetch_company_facts(self, symbol: str) -> Mapping[str, Any]:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
            cik = self._resolve_cik(client, symbol)
            response = self._get(client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            return response.json()

    def fetch_recent_filings(self, symbol: str) -> Mapping[str, Any]:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
            cik = self._resolve_cik(client, symbol)
            response = self._get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            return response.json()

    def _resolve_cik(self, client: httpx.Client, symbol: str) -> int:
        if self._ticker_ciks is None:
            mapping_response = self._get(client, "https://www.sec.gov/files/company_tickers.json")
            self._ticker_ciks = _ticker_cik_map(mapping_response.json())
        normalized = symbol.strip().upper()
        try:
            return self._ticker_ciks[normalized]
        except KeyError as exc:
            raise ValueError(f"SEC CIK not found for symbol: {normalized}") from exc

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        with self._lock:
            remaining = self.minimum_request_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            response = client.get(url)
            self._last_request = time.monotonic()
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f"SEC transient HTTP status: {response.status_code}")
        response.raise_for_status()
        return response


@dataclass(frozen=True)
class SECBatch:
    bundle: EvidenceBundle
    gaps: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"bundle": self.bundle.to_dict(), "gaps": list(self.gaps)}


@dataclass(frozen=True)
class SECFilingsBatch:
    bundle: EvidenceBundle
    gaps: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"bundle": self.bundle.to_dict(), "gaps": list(self.gaps)}


class SECCompanyFactsAdapter:
    def __init__(self, client: SECCompanyFactsClient) -> None:
        self.client = client

    def fetch(self, company: str, symbol: str) -> SECBatch:
        if not company.strip() or not symbol.strip():
            raise ValueError("company and US-listed symbol are required")
        payload = self.client.fetch_company_facts(symbol)
        cik = int(payload.get("cik") or 0)
        entity_name = str(payload.get("entityName") or company)
        facts = payload.get("facts", {}).get("us-gaap", {})
        if not isinstance(facts, Mapping):
            raise ValueError("SEC response has no us-gaap facts object")

        bundle = EvidenceBundle()
        seen_labels: set[str] = set()
        for concept, label in _CONCEPTS:
            if label in seen_labels:
                continue
            concept_payload = facts.get(concept)
            if not isinstance(concept_payload, Mapping):
                continue
            latest = _latest_fact(concept_payload.get("units"))
            if latest is None:
                continue
            unit, fact = latest
            source = SourceRef.create(
                source_type=SourceType.REGULATORY_FILING,
                title=f"{entity_name} {fact.get('form', 'filing')} {concept}",
                locator=(
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
                    f"#concept={concept}&accn={fact.get('accn', 'unknown')}"
                ),
                provider="SEC EDGAR",
                as_of=str(fact.get("end") or "") or None,
                published_at=str(fact.get("filed") or "") or None,
                metadata={
                    "cik": cik,
                    "symbol": symbol.upper(),
                    "taxonomy": "us-gaap",
                    "concept": concept,
                    "accession": fact.get("accn"),
                    "form": fact.get("form"),
                    "fiscal_year": fact.get("fy"),
                    "fiscal_period": fact.get("fp"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "frame": fact.get("frame"),
                },
            )
            evidence = Evidence.create(
                source=source,
                content=(
                    f"{label} for {entity_name}: {fact['val']} {unit}; "
                    f"period ended {fact.get('end')}; filed {fact.get('filed')} on {fact.get('form')}."
                ),
                entity=company,
                field_name=_field_name(label),
                value=fact["val"],
                unit=unit,
                period=_fact_period(fact),
                confidence=1.0,
                tags=("regulatory", "sec", "xbrl", str(fact.get("form") or "unknown")),
            )
            bundle.add_evidence(evidence)
            seen_labels.add(label)

        gaps: list[dict[str, Any]] = []
        if not bundle.evidence:
            gaps.append(
                {
                    "code": "sec_facts_unavailable",
                    "message": f"No supported SEC company facts were found for {symbol.upper()}.",
                    "recoverable_by_coverage": True,
                }
            )
        return SECBatch(bundle, tuple(gaps))


class SECRecentFilingsAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def fetch(
        self,
        company: str,
        symbol: str,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q", "8-K", "20-F", "40-F"),
        limit: int = 10,
    ) -> SECFilingsBatch:
        if not company.strip() or not symbol.strip():
            raise ValueError("company and US-listed symbol are required")
        if not 1 <= limit <= 20:
            raise ValueError("SEC filing limit must be between 1 and 20")
        allowed_forms = {"10-K", "10-Q", "8-K", "20-F", "40-F", "6-K"}
        normalized_forms = tuple(dict.fromkeys(item.upper() for item in forms))
        if not normalized_forms or any(item not in allowed_forms for item in normalized_forms):
            raise ValueError("unsupported SEC filing form filter")
        payload = self.client.fetch_recent_filings(symbol)
        cik = int(payload.get("cik") or 0)
        entity_name = str(payload.get("name") or company)
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, Mapping):
            raise ValueError("SEC submissions response has no recent filings object")
        forms_values = recent.get("form") or []
        accessions = recent.get("accessionNumber") or []
        filed_dates = recent.get("filingDate") or []
        report_dates = recent.get("reportDate") or []
        documents = recent.get("primaryDocument") or []
        descriptions = recent.get("primaryDocDescription") or []
        bundle = EvidenceBundle()
        count = 0
        for index, form_value in enumerate(forms_values):
            form = str(form_value).upper()
            if form not in normalized_forms or count >= limit:
                continue
            accession = _at(accessions, index, "unknown")
            filed = _at(filed_dates, index, "")
            report_date = _at(report_dates, index, "")
            primary_document = _at(documents, index, "")
            description = _at(descriptions, index, "")
            accession_path = accession.replace("-", "")
            locator = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{primary_document}"
                if cik and primary_document
                else f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/"
            )
            source = SourceRef.create(
                source_type=SourceType.REGULATORY_FILING,
                title=f"{entity_name} {form} filed {filed}",
                locator=locator,
                provider="SEC EDGAR submissions",
                as_of=report_date or filed or None,
                published_at=filed or None,
                metadata={
                    "cik": cik,
                    "symbol": symbol.upper(),
                    "form": form,
                    "accession": accession,
                    "primary_document": primary_document,
                    "description": description,
                },
            )
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content=(
                        f"{entity_name} filed {form} on {filed}; report date {report_date or 'not supplied'}; "
                        f"accession {accession}; description {description or 'not supplied'}."
                    ),
                    entity=company,
                    field_name="filing",
                    value=form,
                    period=report_date or filed or None,
                    confidence=1.0,
                    tags=("regulatory", "sec", "filing_metadata", form),
                )
            )
            count += 1
        gaps = (
            ()
            if bundle.evidence
            else (
                {
                    "code": "sec_filings_unavailable",
                    "message": f"No recent requested SEC forms were found for {symbol.upper()}.",
                    "recoverable_by_coverage": True,
                },
            )
        )
        return SECFilingsBatch(bundle, gaps)


def sec_company_facts_harness_tool(adapter: SECCompanyFactsAdapter) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        return adapter.fetch(
            str(arguments.get("company") or ""),
            str(arguments.get("symbol") or ""),
        ).to_dict()

    return function_tool(
        ToolSpec(
            name="sec.company_facts",
            description="从 SEC EDGAR XBRL API 读取带时间戳的公司事实数据。",
            capability="regulatory.read",
            network_access=True,
            timeout_seconds=45,
            retry=RetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.5,
                retryable_exceptions=(
                    TimeoutError,
                    ConnectionError,
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ),
            ),
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"company", "symbol"}),
                optional=frozenset({"required_fields"}),
            ),
        ),
        invoke,
    )


def sec_recent_filings_harness_tool(adapter: SECRecentFilingsAdapter) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        raw_forms = arguments.get("forms") or ["10-K", "10-Q", "8-K", "20-F", "40-F"]
        if not isinstance(raw_forms, list):
            raise ValueError("SEC forms must be a list")
        return adapter.fetch(
            str(arguments.get("company") or ""),
            str(arguments.get("symbol") or ""),
            forms=tuple(str(item) for item in raw_forms),
            limit=int(arguments.get("limit", 10)),
        ).to_dict()

    return function_tool(
        ToolSpec(
            name="sec.recent_filings",
            description="读取发行人近期 SEC 申报元数据和主文档定位信息。",
            capability="regulatory.read",
            network_access=True,
            timeout_seconds=45,
            retry=RetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.5,
                retryable_exceptions=(
                    TimeoutError,
                    ConnectionError,
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ),
            ),
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"company", "symbol"}),
                optional=frozenset({"forms", "limit"}),
            ),
        ),
        invoke,
    )


def _ticker_cik_map(payload: Any) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise ValueError("SEC ticker mapping is malformed")
    result: dict[str, int] = {}
    for item in payload.values():
        if isinstance(item, Mapping) and item.get("ticker") and item.get("cik_str") is not None:
            result[str(item["ticker"]).upper()] = int(item["cik_str"])
    if not result:
        raise ValueError("SEC ticker mapping is empty")
    return result


def _latest_fact(units: Any) -> tuple[str, Mapping[str, Any]] | None:
    if not isinstance(units, Mapping):
        return None
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for unit, values in units.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if (
                isinstance(value, Mapping)
                and value.get("form") in {"10-K", "10-Q", "20-F", "40-F"}
                and isinstance(value.get("val"), (int, float))
                and not isinstance(value.get("val"), bool)
            ):
                candidates.append((str(unit), value))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            str(item[1].get("end") or ""),
            str(item[1].get("filed") or ""),
            int(bool(item[1].get("frame"))),
            str(item[1].get("start") or ""),
        ),
    )


def _field_name(label: str) -> str:
    return label.casefold().replace("'", "").replace(" ", "_")


def _fact_period(fact: Mapping[str, Any]) -> str | None:
    end = str(fact.get("end") or "")
    start = str(fact.get("start") or "")
    if start and end:
        return f"{start}/{end}"
    return end or None


def _at(values: Any, index: int, default: str) -> str:
    if isinstance(values, list) and index < len(values):
        return str(values[index] or default)
    return default
