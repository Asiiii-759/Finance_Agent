"""FRED macroeconomic series client and evidence adapter."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as date_type
from typing import Any, Protocol

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
from .rate_limit import RateLimit, RateLimiter


class MacroSeriesClient(Protocol):
    def fetch_series(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 120,
    ) -> Mapping[str, Any]: ...


class FREDClient:
    """Fixed-endpoint FRED v1 client; the API key never enters tool arguments."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.stlouisfed.org",
        timeout_seconds: float = 30.0,
        rate_limiter: RateLimiter | None = None,
        rate_limit: RateLimit | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("FRED API key is required")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = rate_limiter
        self.rate_limit = rate_limit or RateLimit(8)

    def fetch_series(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 120,
    ) -> Mapping[str, Any]:
        normalized = _validate_series_id(series_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("FRED observation limit must be between 1 and 10000")
        if self.rate_limiter is not None:
            self.rate_limiter.acquire("fred", self.rate_limit, timeout_seconds=self.timeout_seconds)
        common = {"series_id": normalized, "api_key": self.api_key, "file_type": "json"}
        observation_params: dict[str, Any] = {
            **common,
            "limit": limit,
            "sort_order": "desc",
        }
        if observation_start:
            observation_params["observation_start"] = _validate_date(observation_start)
        if observation_end:
            observation_params["observation_end"] = _validate_date(observation_end)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            series_response = client.get(f"{self.base_url}/fred/series", params=common)
            if series_response.status_code == 429 or series_response.status_code >= 500:
                raise ConnectionError(f"FRED transient HTTP status: {series_response.status_code}")
            series_response.raise_for_status()
            observations_response = client.get(f"{self.base_url}/fred/series/observations", params=observation_params)
            if observations_response.status_code == 429 or observations_response.status_code >= 500:
                raise ConnectionError(f"FRED transient HTTP status: {observations_response.status_code}")
            observations_response.raise_for_status()
        series_payload = series_response.json()
        observations_payload = observations_response.json()
        series_values = series_payload.get("seriess") or []
        if not isinstance(series_values, list) or not series_values:
            raise ValueError(f"FRED returned no metadata for series {normalized}")
        return {
            "provider": "FRED",
            "series": dict(series_values[0]),
            "observations": list(observations_payload.get("observations") or []),
            "retrieved_at": datetime.now(UTC).isoformat(),
        }


@dataclass(frozen=True)
class MacroBatch:
    bundle: EvidenceBundle
    observations: tuple[dict[str, Any], ...]
    gaps: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "observations": list(self.observations),
            "gaps": list(self.gaps),
        }


class FREDEvidenceAdapter:
    def __init__(self, client: MacroSeriesClient) -> None:
        self.client = client

    def fetch(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 120,
    ) -> MacroBatch:
        normalized = _validate_series_id(series_id)
        raw = self.client.fetch_series(
            normalized,
            observation_start=observation_start,
            observation_end=observation_end,
            limit=limit,
        )
        metadata = raw.get("series") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("macro series metadata must be an object")
        observations_by_date: dict[str, dict[str, Any]] = {}
        for item in raw.get("observations") or ():
            if not isinstance(item, Mapping):
                continue
            raw_value = item.get("value")
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            try:
                observation_date = date_type.fromisoformat(str(item.get("date") or "")).isoformat()
            except ValueError:
                continue
            observations_by_date[observation_date] = {
                "date": observation_date,
                "value": value,
            }
        observations = [observations_by_date[key] for key in sorted(observations_by_date)]
        bundle = EvidenceBundle()
        gaps: list[dict[str, Any]] = []
        if not observations:
            gaps.append(
                {
                    "code": "macro_observations_unavailable",
                    "message": f"FRED returned no numeric observations for {normalized}.",
                    "recoverable_by_coverage": True,
                }
            )
            return MacroBatch(bundle, (), tuple(gaps))

        latest = observations[-1]
        title = str(metadata.get("title") or normalized)
        units = str(metadata.get("units_short") or metadata.get("units") or "unknown")
        source = SourceRef.create(
            source_type=SourceType.MACRO_DATA,
            title=title,
            locator=f"https://fred.stlouisfed.org/series/{normalized}",
            provider="Federal Reserve Bank of St. Louis (FRED)",
            as_of=latest["date"],
            published_at=str(metadata.get("last_updated") or "") or None,
            metadata={
                "series_id": normalized,
                "frequency": metadata.get("frequency"),
                "units": metadata.get("units"),
                "seasonal_adjustment": metadata.get("seasonal_adjustment"),
                "observation_count": len(observations),
                "retrieved_at": raw.get("retrieved_at"),
            },
        )
        series_input = Evidence.create(
            source=source,
            content=json.dumps(observations, ensure_ascii=False, separators=(",", ":")),
            entity=normalized,
            field_name="observations",
            period=f"{observations[0]['date']}/{latest['date']}",
            confidence=1.0,
            tags=("macro", "fred", "calculation_input", normalized),
        )
        bundle.add_evidence(series_input)
        latest_evidence = Evidence.create(
            source=source,
            content=f"{title} ({normalized}) latest observation: {latest['value']} {units} on {latest['date']}.",
            entity=normalized,
            field_name="latest_value",
            value=latest["value"],
            unit=units,
            period=latest["date"],
            confidence=1.0,
            tags=("macro", "fred", normalized),
        )
        bundle.add_evidence(latest_evidence)
        if len(observations) >= 2:
            previous = observations[-2]
            change = latest["value"] - previous["value"]
            previous_evidence = Evidence.create(
                source=source,
                content=(
                    f"{title} ({normalized}) previous observation: {previous['value']} {units} on {previous['date']}."
                ),
                entity=normalized,
                field_name="previous_value",
                value=previous["value"],
                unit=units,
                period=previous["date"],
                confidence=1.0,
                tags=("macro", "fred", normalized),
            )
            bundle.add_evidence(previous_evidence)
            calculation_source = SourceRef.create(
                source_type=SourceType.CALCULATION,
                title=f"{normalized} change from previous observation",
                locator=f"formula://macro/absolute_change/v1/{normalized}",
                provider="mas_finance.macro",
                as_of=latest["date"],
                metadata={
                    "formula": "ending_value - beginning_value",
                    "formula_version": "1",
                    "input_evidence_ids": [
                        series_input.evidence_id,
                        previous_evidence.evidence_id,
                        latest_evidence.evidence_id,
                    ],
                },
            )
            bundle.add_evidence(
                Evidence.create(
                    source=calculation_source,
                    content=(
                        f"{title} ({normalized}) changed by {change} {units} from "
                        f"{previous['date']} to {latest['date']}."
                    ),
                    entity=normalized,
                    field_name="change_from_previous",
                    value=round(change, 10),
                    unit=units,
                    period=f"{previous['date']}/{latest['date']}",
                    confidence=1.0,
                    tags=("macro", "fred", "calculation", "calculated_change", normalized),
                )
            )
        return MacroBatch(bundle, tuple(observations), tuple(gaps))


def fred_series_harness_tool(adapter: FREDEvidenceAdapter) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        return adapter.fetch(
            str(arguments.get("series_id") or ""),
            observation_start=_optional_text(arguments.get("observation_start")),
            observation_end=_optional_text(arguments.get("observation_end")),
            limit=int(arguments.get("limit", 120)),
        ).to_dict()

    return function_tool(
        ToolSpec(
            name="macro.fred_series",
            description="读取白名单 FRED 序列及其元数据和带时间戳观测值。",
            capability="macro.read",
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
                required=frozenset({"series_id"}),
                optional=frozenset({"observation_start", "observation_end", "limit"}),
            ),
            input_schema={
                "type": "object",
                "required": ["series_id"],
                "additionalProperties": False,
                "properties": {
                    "series_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "pattern": "^[A-Za-z0-9_.-]+$",
                    },
                    "observation_start": {"type": "string", "format": "date"},
                    "observation_end": {"type": "string", "format": "date"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
                },
            },
        ),
        invoke,
    )


def _validate_series_id(value: str) -> str:
    normalized = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_.-]{1,64}", normalized):
        raise ValueError("invalid FRED series id")
    return normalized


def _validate_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("macro observation date must use YYYY-MM-DD") from exc
    return value


def _optional_text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
