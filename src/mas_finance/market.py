"""Market-data anti-corruption layer with field-level provenance."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as date_type
from hashlib import sha256
from typing import Any, Protocol

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from .harness import (
    RetryPolicy,
    Tool,
    ToolArgumentContract,
    ToolResultKind,
    ToolSpec,
    function_tool,
)
from .metrics import MetricOperation, MetricRequest, calculate_metric


class MarketClient(Protocol):
    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> Mapping[str, Any]: ...


class MarketHistoryClient(Protocol):
    def fetch_price_history(
        self,
        company: str,
        symbol: str | None = None,
        *,
        range_name: str = "1y",
        interval: str = "1d",
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DataGap:
    code: str
    message: str
    fields: tuple[str, ...] = ()
    recoverable_by_coverage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fields": list(self.fields),
            "recoverable_by_coverage": self.recoverable_by_coverage,
        }


@dataclass(frozen=True)
class MarketBatch:
    bundle: EvidenceBundle
    snapshot: Mapping[str, Any]
    gaps: tuple[DataGap, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "snapshot": dict(self.snapshot),
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


@dataclass(frozen=True)
class MarketHistoryBatch:
    bundle: EvidenceBundle
    summary: Mapping[str, Any]
    gaps: tuple[DataGap, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "summary": dict(self.summary),
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


_MARKET_FIELDS: dict[str, tuple[str, str]] = {
    "current_price": ("Current price", "currency"),
    "monthly_return": ("One-month return", "ratio"),
    "market_cap": ("Market capitalization", "currency"),
    "trailing_pe": ("Trailing price-to-earnings ratio", "ratio"),
    "price_to_book": ("Price-to-book ratio", "ratio"),
    "price_to_sales": ("Price-to-sales ratio", "ratio"),
    "enterprise_to_ebitda": ("Enterprise-value-to-EBITDA ratio", "ratio"),
    "fifty_two_week_high": ("52-week high", "currency"),
    "fifty_two_week_low": ("52-week low", "currency"),
}


class MarketEvidenceAdapter:
    def __init__(self, client: MarketClient) -> None:
        self.client = client

    def fetch(
        self,
        company: str,
        symbol: str | None = None,
        *,
        required_fields: tuple[str, ...] = (),
    ) -> MarketBatch:
        if not company.strip():
            raise ValueError("company is required")
        expected_fields = set(required_fields or _MARKET_FIELDS)
        if not expected_fields.issubset(_MARKET_FIELDS):
            raise ValueError("unsupported required market field")
        raw = dict(self.client.fetch_company_snapshot(company, symbol))
        provider = str(raw.get("provider") or "unknown")
        actual_symbol = str(raw.get("symbol") or symbol or company)
        as_of = _optional_text(raw.get("as_of"))
        retrieved_at = _optional_text(raw.get("retrieved_at"))
        currency = _optional_text(raw.get("currency"))
        source = SourceRef.create(
            source_type=SourceType.MARKET_DATA,
            title=f"{actual_symbol} market snapshot",
            locator=f"market://{provider}/{actual_symbol}",
            provider=provider,
            as_of=as_of,
            metadata={
                "symbol": actual_symbol,
                "company": company,
                "exchange": raw.get("exchange"),
                "retrieved_at": retrieved_at,
                "raw_status": raw.get("status", "ok"),
            },
        )
        bundle = EvidenceBundle()
        missing: list[str] = []
        for field_name, (label, unit_kind) in _MARKET_FIELDS.items():
            value = raw.get(field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                missing.append(field_name)
                continue
            unit = currency if unit_kind == "currency" else unit_kind
            evidence = Evidence.create(
                source=source,
                content=f"{label} for {actual_symbol}: {value}{f' {unit}' if unit else ''}",
                entity=company,
                field_name=field_name,
                value=float(value),
                unit=unit,
                period=as_of,
                confidence=1.0,
                tags=("market_data", "structured"),
            )
            bundle.add_evidence(evidence)

        gaps: list[DataGap] = []
        required_missing = tuple(item for item in missing if item in expected_fields)
        if len(required_missing) == len(expected_fields):
            error_codes = ", ".join(str(item) for item in raw.get("error_codes") or ())
            error_suffix = f" Provider errors: {error_codes}." if error_codes else ""
            gaps.append(
                DataGap(
                    code="market_provider_unavailable",
                    message=f"No usable market fields returned for {actual_symbol}.{error_suffix}",
                    fields=required_missing,
                    recoverable_by_coverage=True,
                )
            )
        elif required_missing:
            gaps.append(
                DataGap(
                    code="market_fields_missing",
                    message=f"Some market fields are unavailable for {actual_symbol}.",
                    fields=required_missing,
                    recoverable_by_coverage=True,
                )
            )
        if as_of is None:
            gaps.append(
                DataGap(
                    code="market_as_of_missing",
                    message=f"Provider did not supply an as-of timestamp for {actual_symbol}.",
                )
            )
        return MarketBatch(bundle=bundle, snapshot=raw, gaps=tuple(gaps))


class MarketHistoryEvidenceAdapter:
    def __init__(self, client: MarketHistoryClient) -> None:
        self.client = client

    def fetch(
        self,
        company: str,
        symbol: str | None = None,
        *,
        range_name: str = "1y",
        interval: str = "1d",
    ) -> MarketHistoryBatch:
        if not company.strip():
            raise ValueError("company is required")
        raw = dict(
            self.client.fetch_price_history(
                company,
                symbol,
                range_name=range_name,
                interval=interval,
            )
        )
        provider = str(raw.get("provider") or "unknown")
        actual_symbol = str(raw.get("symbol") or symbol or company)
        points = _valid_points(raw.get("points"))
        if len(points) < 3:
            errors = ", ".join(str(item) for item in raw.get("error_codes") or ())
            suffix = f" Provider errors: {errors}." if errors else ""
            return MarketHistoryBatch(
                EvidenceBundle(),
                {"symbol": actual_symbol, "observation_count": len(points)},
                (
                    DataGap(
                        code="market_history_unavailable",
                        message=f"Fewer than three price observations were returned for {actual_symbol}.{suffix}",
                        recoverable_by_coverage=True,
                    ),
                ),
            )
        currency = _optional_text(raw.get("currency"))
        price_basis = str(raw.get("price_basis") or "close")
        as_of = str(points[-1]["date"])
        point_hash = sha256(json.dumps(points, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        market_source = SourceRef.create(
            source_type=SourceType.MARKET_DATA,
            title=f"{actual_symbol} {price_basis.replace('_', ' ')} history",
            locator=f"market://{provider}/{actual_symbol}/history?range={range_name}&interval={interval}",
            provider=provider,
            as_of=as_of,
            metadata={
                "symbol": actual_symbol,
                "company": company,
                "range": range_name,
                "interval": interval,
                "price_basis": price_basis,
                "observation_count": len(points),
                "points_sha256": point_hash,
                "retrieved_at": raw.get("retrieved_at"),
            },
        )
        bundle = EvidenceBundle()
        start = Evidence.create(
            source=market_source,
            content=f"{actual_symbol} adjusted start price: {points[0]['close']} on {points[0]['date']}.",
            entity=company,
            field_name="history_start_price",
            value=points[0]["close"],
            unit=currency,
            period=str(points[0]["date"]),
            tags=("market_data", "market_history", "structured"),
        )
        end = Evidence.create(
            source=market_source,
            content=f"{actual_symbol} adjusted end price: {points[-1]['close']} on {points[-1]['date']}.",
            entity=company,
            field_name="history_end_price",
            value=points[-1]["close"],
            unit=currency,
            period=as_of,
            tags=("market_data", "market_history", "structured"),
        )
        bundle.add_evidence(start)
        bundle.add_evidence(end)
        normalized_points = json.dumps(points, ensure_ascii=False, separators=(",", ":"))
        series_input = Evidence.create(
            source=market_source,
            content=normalized_points,
            entity=company,
            field_name="history_observations",
            period=f"{points[0]['date']}/{as_of}",
            tags=("market_data", "market_history", "calculation_input"),
        )
        bundle.add_evidence(series_input)
        quality_gaps: list[DataGap] = []
        if price_basis != "adjusted_close":
            quality_gaps.append(
                DataGap(
                    code="unadjusted_price_history",
                    message=(
                        f"{actual_symbol} history uses unadjusted close; calculated returns exclude "
                        "cash dividends and may not equal total shareholder return."
                    ),
                )
            )

        closes = [float(item["close"]) for item in points]
        returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
        annualization_factor = {"1d": 252.0, "1wk": 52.0, "1mo": 12.0}[interval]
        requests = (
            MetricRequest(
                operation=MetricOperation.PERCENTAGE_CHANGE,
                inputs={"beginning_value": closes[0], "ending_value": closes[-1]},
                label="total_return",
                entity=company,
                period=f"{points[0]['date']}/{as_of}",
            ),
            MetricRequest(
                operation=MetricOperation.ANNUALIZED_RETURN,
                inputs={"returns": returns, "annualization_factor": annualization_factor},
                label="annualized_return",
                entity=company,
                period=f"{points[0]['date']}/{as_of}",
            ),
            MetricRequest(
                operation=MetricOperation.ANNUALIZED_VOLATILITY,
                inputs={"returns": returns, "annualization_factor": annualization_factor},
                label="annualized_volatility",
                entity=company,
                period=f"{points[0]['date']}/{as_of}",
            ),
            MetricRequest(
                operation=MetricOperation.MAX_DRAWDOWN,
                inputs={"values": closes},
                label="max_drawdown",
                entity=company,
                period=f"{points[0]['date']}/{as_of}",
            ),
        )
        summary: dict[str, Any] = {
            "symbol": actual_symbol,
            "start": points[0],
            "end": points[-1],
            "observation_count": len(points),
        }
        for request in requests:
            result = calculate_metric(request)
            calculation_source = SourceRef.create(
                source_type=SourceType.CALCULATION,
                title=f"{actual_symbol} {request.label}",
                locator=f"formula://market_history/{request.label}/v1/{point_hash[:16]}",
                provider="mas_finance.metrics",
                as_of=as_of,
                metadata={
                    "formula": result.formula,
                    "formula_version": "1",
                    "market_source_id": market_source.source_id,
                    "input_evidence_ids": [
                        series_input.evidence_id,
                        start.evidence_id,
                        end.evidence_id,
                    ],
                    "observation_count": len(points),
                    "points_sha256": point_hash,
                },
            )
            evidence = Evidence.create(
                source=calculation_source,
                content=(
                    f"{request.label} for {actual_symbol} from {points[0]['date']} to {as_of}: "
                    f"{result.value} {result.unit}; formula: {result.formula}."
                ),
                entity=company,
                field_name=request.label,
                value=result.value,
                unit=result.unit,
                period=f"{points[0]['date']}/{as_of}",
                tags=("calculation", "market_history", request.label or request.operation.value),
            )
            bundle.add_evidence(evidence)
            summary[str(request.label)] = result.value
        return MarketHistoryBatch(bundle=bundle, summary=summary, gaps=tuple(quality_gaps))


def market_data_harness_tool(
    adapter: MarketEvidenceAdapter,
    *,
    network_access: bool = True,
) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        raw_required_fields = arguments.get("required_fields") or []
        if not isinstance(raw_required_fields, list):
            raise ValueError("required_fields must be a list")
        return adapter.fetch(
            company=str(arguments.get("company") or ""),
            symbol=_optional_text(arguments.get("symbol")),
            required_fields=tuple(str(item) for item in raw_required_fields),
        ).to_dict()

    return function_tool(
        ToolSpec(
            name="market.snapshot",
            description="Read a point-in-time market snapshot with field-level provenance.",
            capability="market.read",
            network_access=network_access,
            timeout_seconds=35,
            retry=RetryPolicy(max_attempts=1),
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"company"}),
                optional=frozenset({"symbol", "required_fields"}),
            ),
        ),
        invoke,
    )


def market_history_harness_tool(
    adapter: MarketHistoryEvidenceAdapter,
    *,
    network_access: bool = True,
) -> Tool:
    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        return adapter.fetch(
            company=str(arguments.get("company") or ""),
            symbol=_optional_text(arguments.get("symbol")),
            range_name=str(arguments.get("range") or "1y"),
            interval=str(arguments.get("interval") or "1d"),
        ).to_dict()

    return function_tool(
        ToolSpec(
            name="market.history",
            description=(
                "Read price history with an explicit adjusted/raw basis and calculate return, "
                "volatility and max drawdown."
            ),
            capability="market.read",
            network_access=network_access,
            timeout_seconds=35,
            retry=RetryPolicy(max_attempts=1),
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"company"}),
                optional=frozenset({"symbol", "range", "interval"}),
            ),
        ),
        invoke,
    )


def _optional_text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _valid_points(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    points_by_date: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        close = item.get("close")
        date = str(item.get("date") or "")
        if (
            isinstance(close, (int, float))
            and not isinstance(close, bool)
            and math.isfinite(float(close))
            and float(close) > 0
            and date
        ):
            try:
                normalized_date = date_type.fromisoformat(date).isoformat()
            except ValueError:
                continue
            points_by_date[normalized_date] = {
                "date": normalized_date,
                "close": float(close),
            }
    return [points_by_date[key] for key in sorted(points_by_date)]
