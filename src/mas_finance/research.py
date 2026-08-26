"""Question scoping for adaptive, auditable financial research planning."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .knowledge import detect_finance_concepts
from .metrics import MetricRequest, infer_metric_requests


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
    calculations: tuple[MetricRequest, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intents": [item.value for item in self.intents],
            "requirements": [item.to_dict() for item in self.requirements],
            "calculations": [item.to_dict() for item in self.calculations],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchScope:
        return cls(
            intents=tuple(FinancialIntent(str(item)) for item in value.get("intents") or ()),
            requirements=tuple(ResearchRequirement.from_dict(item) for item in value.get("requirements") or ()),
            calculations=tuple(MetricRequest.from_dict(item) for item in value.get("calculations") or ()),
            rationale=str(value.get("rationale") or ""),
        )


_KEYWORDS: dict[FinancialIntent, tuple[str, ...]] = {
    FinancialIntent.MARKET_SNAPSHOT: (
        "price",
        "quote",
        "market cap",
        "股价",
        "市值",
        "行情",
        "当前价格",
    ),
    FinancialIntent.MARKET_PERFORMANCE: (
        "return",
        "performance",
        "volatility",
        "drawdown",
        "sharpe",
        "走势",
        "收益率",
        "波动率",
        "回撤",
        "夏普",
    ),
    FinancialIntent.FUNDAMENTALS: (
        "fundamental",
        "revenue",
        "income",
        "profit",
        "assets",
        "balance sheet",
        "cash flow",
        "基本面",
        "收入",
        "利润",
        "资产",
        "负债",
        "现金流",
        "财务指标",
    ),
    FinancialIntent.VALUATION: (
        "valuation",
        "p/e",
        "pe ratio",
        "price-to-earnings",
        "估值",
        "市盈率",
        "市销率",
        "市净率",
    ),
    FinancialIntent.PROFITABILITY: (
        "margin",
        "roe",
        "roa",
        "profitability",
        "盈利能力",
        "利润率",
        "净利率",
        "毛利率",
        "资产回报",
        "净资产收益",
    ),
    FinancialIntent.SOLVENCY: (
        "leverage",
        "solvency",
        "debt ratio",
        "debt-to-equity",
        "杠杆",
        "偿债",
        "负债率",
        "资产负债率",
        "债务权益",
    ),
    FinancialIntent.LIQUIDITY: (
        "liquidity",
        "current ratio",
        "quick ratio",
        "流动性",
        "流动比率",
        "速动比率",
    ),
    FinancialIntent.RISK: (
        "risk",
        "uncertainty",
        "风险",
        "不确定性",
    ),
    FinancialIntent.COMPARISON: (
        "compare",
        "comparison",
        "versus",
        " vs ",
        "对比",
        "比较",
    ),
}

_MACRO_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CPIAUCSL", ("cpi", "inflation", "consumer price", "通胀", "消费者价格")),
    ("UNRATE", ("unemployment", "jobless rate", "失业率")),
    ("GDP", ("gross domestic product", " gdp", "gdp ", "国内生产总值")),
    ("FEDFUNDS", ("federal funds", "fed funds", "联邦基金利率", "美联储利率")),
    ("DGS10", ("10-year treasury", "10 year treasury", "十年期美债", "10年期美债")),
    ("DGS2", ("2-year treasury", "2 year treasury", "两年期美债", "2年期美债")),
    ("PAYEMS", ("nonfarm payroll", "non-farm payroll", "非农就业", "非农数据")),
    ("PCEPI", ("pce price", "pce inflation", "pce物价", "pce通胀")),
    ("MORTGAGE30US", ("mortgage rate", "30-year mortgage", "房贷利率", "抵押贷款利率")),
)


class FinancialQueryAnalyzer:
    """Build a deterministic research scope; this is a decision trace, not hidden CoT."""

    def analyze(self, request: Any) -> ResearchScope:
        query = str(request.query)
        normalized = f" {query.casefold()} "
        entities = tuple(str(item) for item in request.entities)
        intents: set[FinancialIntent] = set()
        for intent, keywords in _KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                intents.add(intent)
        if len(entities) > 1:
            intents.add(FinancialIntent.COMPARISON)
        if any(
            item in normalized
            for item in (
                "10-k",
                "10-q",
                "8-k",
                "20-f",
                "40-f",
                "sec filing",
                "recent filing",
                "监管文件",
                "最新披露",
                "最新公告",
            )
        ):
            intents.add(FinancialIntent.REGULATORY_FILINGS)

        calculations = tuple(
            MetricRequest.from_dict(item) for item in getattr(request, "calculations", ())
        ) or infer_metric_requests(query)
        if calculations:
            intents.add(FinancialIntent.CALCULATION)

        explicit_macro = tuple(str(item).upper() for item in getattr(request, "macro_series", ()))
        macro_series = list(explicit_macro)
        for series_id, keywords in _MACRO_ALIASES:
            if any(keyword in normalized for keyword in keywords) and series_id not in macro_series:
                macro_series.append(series_id)
        if macro_series:
            intents.add(FinancialIntent.MACROECONOMICS)

        concepts = detect_finance_concepts(query)
        education_markers = (
            "what is",
            "what does",
            "explain",
            "formula",
            "interpret",
            "是什么",
            "什么意思",
            "解释",
            "公式",
            "如何理解",
            "怎么算",
        )
        needs_knowledge = bool(concepts) and (
            (not entities and not calculations) or any(item in normalized for item in education_markers)
        )
        if needs_knowledge:
            intents.add(FinancialIntent.FINANCIAL_EDUCATION)

        require_documents = bool(request.require_documents)
        if require_documents:
            intents.add(FinancialIntent.DOCUMENT_RESEARCH)

        broad_analysis = any(
            item in normalized for item in (" analyze ", " analysis ", " overview ", "研究", "分析", "概况")
        )
        document_deictic = require_documents and any(
            item in normalized for item in ("this report", "this document", "这份", "文档", "pdf")
        )
        market_explicit = getattr(request, "require_market_data", None)
        history_explicit = getattr(request, "require_market_history", None)
        regulatory_explicit = getattr(request, "require_regulatory_data", None)

        needs_market = bool(entities) and (
            market_explicit is True
            or (
                market_explicit is None
                and (
                    FinancialIntent.MARKET_SNAPSHOT in intents
                    or FinancialIntent.VALUATION in intents
                    or (broad_analysis and not document_deictic and not require_documents)
                )
            )
        )
        needs_history = bool(entities) and (
            history_explicit is True or (history_explicit is None and FinancialIntent.MARKET_PERFORMANCE in intents)
        )
        fundamental_intents = {
            FinancialIntent.FUNDAMENTALS,
            FinancialIntent.PROFITABILITY,
            FinancialIntent.SOLVENCY,
            FinancialIntent.LIQUIDITY,
        }
        needs_regulatory = bool(entities) and (
            regulatory_explicit is True
            or (
                regulatory_explicit is None
                and not require_documents
                and (bool(intents.intersection(fundamental_intents)) or (broad_analysis and not document_deictic))
            )
        )
        if market_explicit is False:
            needs_market = False
        if history_explicit is False:
            needs_history = False
        if regulatory_explicit is False:
            needs_regulatory = False

        requirements: list[ResearchRequirement] = []
        if require_documents:
            for entity in entities or (None,):
                requirements.append(
                    ResearchRequirement(
                        key=f"document:{entity or 'query'}",
                        category="document",
                        entity=entity,
                        reason="The request includes an authorized document corpus.",
                    )
                )
        if needs_market:
            fields = _market_fields(intents, concepts)
            for entity in entities:
                requirements.append(
                    ResearchRequirement(
                        key=f"market:{entity}",
                        category="market",
                        entity=entity,
                        fields=fields,
                        reason="The question asks for current market or valuation evidence.",
                    )
                )
        if needs_history:
            for entity in entities:
                requirements.append(
                    ResearchRequirement(
                        key=f"market_history:{entity}",
                        category="market_history",
                        entity=entity,
                        fields=("total_return", "annualized_volatility", "max_drawdown"),
                        parameters={
                            "range": _history_range(normalized, request.market_history_range),
                            "interval": request.market_history_interval,
                        },
                        reason="The question asks for return, volatility or drawdown analysis.",
                    )
                )
        if needs_regulatory:
            fields = _regulatory_fields(intents, concepts)
            for entity in entities:
                requirements.append(
                    ResearchRequirement(
                        key=f"regulatory:{entity}",
                        category="regulatory",
                        entity=entity,
                        fields=fields,
                        reason="The question needs filed fundamental evidence.",
                    )
                )
            for metric in _derived_metrics(intents, concepts):
                for entity in entities:
                    requirements.append(
                        ResearchRequirement(
                            key=f"metric:{entity}:{metric}",
                            category="derived_metric",
                            entity=entity,
                            fields=(metric,),
                            reason="The question asks for a deterministic metric derived from aligned evidence.",
                        )
                    )
        if entities and FinancialIntent.REGULATORY_FILINGS in intents:
            forms = _filing_forms(normalized)
            for entity in entities:
                requirements.append(
                    ResearchRequirement(
                        key=f"filings:{entity}",
                        category="filings",
                        entity=entity,
                        fields=("filing",),
                        parameters={"forms": list(forms), "limit": 10},
                        reason="The question asks for recent regulatory filing metadata.",
                    )
                )
        for series_id in macro_series:
            requirements.append(
                ResearchRequirement(
                    key=f"macro:{series_id}",
                    category="macro",
                    entity=series_id,
                    fields=("latest_value",),
                    parameters={"series_id": series_id, "limit": 120},
                    reason="The question references a known macroeconomic series.",
                )
            )
        for concept in concepts if needs_knowledge else ():
            requirements.append(
                ResearchRequirement(
                    key=f"knowledge:{concept}",
                    category="knowledge",
                    entity=concept,
                    fields=("definition",),
                    parameters={"concepts": [concept], "top_k": 1},
                    reason="The question asks for a finance definition, formula or interpretation caveat.",
                )
            )
        for calculation in calculations:
            requirements.append(
                ResearchRequirement(
                    key=f"calculation:{calculation.request_id}",
                    category="calculation",
                    entity=calculation.entity,
                    fields=(calculation.label or calculation.operation.value,),
                    parameters={"request_id": calculation.request_id},
                    reason="The request contains explicit or unambiguous numeric inputs.",
                )
            )
        unsupported_metrics: list[ResearchRequirement] = []
        if needs_regulatory:
            for metric in ("roe", "roa"):
                if metric in concepts:
                    unsupported_metrics.append(
                        ResearchRequirement(
                            key=f"unsupported:{metric}:average_balance",
                            category="unsupported",
                            entity=entities[0] if len(entities) == 1 else None,
                            parameters={"gap_code": "metric_requires_average_balance"},
                            reason=(
                                f"Reliable {metric.upper()} requires beginning and ending balance-sheet values "
                                "to calculate an average balance; the current SEC facts adapter only selects "
                                "one balance-sheet instant."
                            ),
                        )
                    )
            if "quick_ratio" in concepts:
                unsupported_metrics.append(
                    ResearchRequirement(
                        key="unsupported:quick_ratio:liquid_assets",
                        category="unsupported",
                        entity=entities[0] if len(entities) == 1 else None,
                        parameters={"gap_code": "metric_requires_liquid_asset_breakdown"},
                        reason=(
                            "Quick ratio requires inventory and other less-liquid current assets to be "
                            "excluded; those inputs are not normalized by the current SEC adapter."
                        ),
                    )
                )
        if entities and "dcf" in concepts and not needs_knowledge:
            unsupported_metrics.append(
                ResearchRequirement(
                    key="unsupported:dcf:forecast_assumptions",
                    category="unsupported",
                    entity=entities[0] if len(entities) == 1 else None,
                    parameters={"gap_code": "valuation_model_inputs_required"},
                    reason=(
                        "A DCF valuation requires explicit cash-flow forecasts, discount-rate assumptions "
                        "and a terminal-value method; the agent will not invent them."
                    ),
                )
            )
        requirements.extend(unsupported_metrics)
        has_supported_requirement = any(item.category != "unsupported" for item in requirements)
        precise_forecast = any(
            item in normalized
            for item in (
                "exact closing price tomorrow",
                "tomorrow's exact close",
                "明天的精确收盘价",
                "预测明天一只未指定股票的精确收盘价",
            )
        )
        if not has_supported_requirement and precise_forecast:
            intents.add(FinancialIntent.GENERAL_RESEARCH)
            requirements.append(
                ResearchRequirement(
                    key="unsupported:general_research",
                    category="unsupported",
                    parameters={"gap_code": "unsupported_research_scope"},
                    reason=(
                        "The question requires an evidence provider outside the currently supported "
                        "market, filing, macro, document, knowledge, or calculation tools."
                    ),
                )
            )
        elif not has_supported_requirement:
            intents.add(FinancialIntent.GENERAL_RESEARCH)
            requirements.append(
                ResearchRequirement(
                    key="web:general_research",
                    category="web",
                    reason="The question needs current or open-web financial research evidence.",
                )
            )
        ordered_intents = tuple(sorted(intents, key=lambda item: item.value))
        return ResearchScope(
            intents=ordered_intents,
            requirements=tuple(requirements),
            calculations=calculations,
            rationale=(
                f"Detected {len(ordered_intents)} intent(s) and {len(requirements)} evidence requirement(s) "
                "using deterministic bilingual finance rules."
            ),
        )


def _market_fields(intents: set[FinancialIntent], concepts: Sequence[str]) -> tuple[str, ...]:
    fields = {"current_price"}
    if FinancialIntent.VALUATION in intents:
        valuation_fields = {
            "pe_ratio": "trailing_pe",
            "pb_ratio": "price_to_book",
            "ps_ratio": "price_to_sales",
            "ev_ebitda": "enterprise_to_ebitda",
        }
        selected = {field for concept, field in valuation_fields.items() if concept in concepts}
        fields.update(selected or {"market_cap", "trailing_pe"})
    return tuple(sorted(fields))


def _regulatory_fields(intents: set[FinancialIntent], concepts: Sequence[str]) -> tuple[str, ...]:
    fields: set[str] = set()
    concept_set = set(concepts)
    if FinancialIntent.FUNDAMENTALS in intents or (
        FinancialIntent.PROFITABILITY in intents
        and not concept_set.intersection({"roe", "roa", "gross_margin", "operating_margin", "net_margin"})
    ):
        fields.update({"revenue", "net_income"})
    if FinancialIntent.SOLVENCY in intents:
        if "debt_to_equity" in concept_set:
            fields.update({"total_liabilities", "stockholders_equity"})
        else:
            fields.update({"total_assets", "total_liabilities"})
    if FinancialIntent.LIQUIDITY in intents:
        fields.update({"current_assets", "current_liabilities"})
    if "roe" in concepts:
        fields.update({"net_income", "stockholders_equity"})
    if "roa" in concepts:
        fields.update({"net_income", "total_assets"})
    if "net_margin" in concepts:
        fields.update({"net_income", "revenue"})
    if "gross_margin" in concepts:
        fields.update({"gross_profit", "revenue"})
    if "operating_margin" in concepts:
        fields.update({"operating_income", "revenue"})
    if not fields:
        fields.update({"revenue", "net_income", "total_assets", "total_liabilities"})
    return tuple(sorted(fields))


def _derived_metrics(intents: set[FinancialIntent], concepts: Sequence[str]) -> tuple[str, ...]:
    metrics: set[str] = set()
    concept_set = set(concepts)
    if FinancialIntent.PROFITABILITY in intents and not concept_set.intersection(
        {"roe", "roa", "gross_margin", "operating_margin"}
    ):
        metrics.add("net_margin")
    if "net_margin" in concept_set:
        metrics.add("net_margin")
    if "gross_margin" in concept_set:
        metrics.add("gross_margin")
    if "operating_margin" in concept_set:
        metrics.add("operating_margin")
    if "debt_to_equity" in concept_set:
        metrics.add("debt_to_equity")
    elif FinancialIntent.SOLVENCY in intents:
        metrics.add("liabilities_to_assets")
    if "current_ratio" in concept_set or (FinancialIntent.LIQUIDITY in intents and "quick_ratio" not in concept_set):
        metrics.add("current_ratio")
    return tuple(sorted(metrics))


def validate_macro_series(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(item.strip().upper() for item in values if item.strip()))
    if len(normalized) > 20:
        raise ValueError("at most 20 macro series may be requested")
    if any(not re.fullmatch(r"[A-Z0-9_.-]{1,64}", item) for item in normalized):
        raise ValueError("macro series identifiers contain invalid characters")
    return normalized


def _filing_forms(query: str) -> tuple[str, ...]:
    forms = tuple(item for item in ("10-K", "10-Q", "8-K", "20-F", "40-F", "6-K") if item.casefold() in query)
    return forms or ("10-K", "10-Q", "8-K", "20-F", "40-F")


def _history_range(query: str, configured: str) -> str:
    aliases = (
        ("10y", ("10 year", "10-year", "10年", "十年")),
        ("5y", ("5 year", "5-year", "5年", "五年")),
        ("2y", ("2 year", "2-year", "2年", "两年")),
        ("6mo", ("6 month", "6-month", "6个月", "半年")),
        ("3mo", ("3 month", "3-month", "3个月", "三个月")),
        ("1mo", ("1 month", "1-month", "1个月", "一个月")),
        ("1y", ("1 year", "1-year", "1年", "一年")),
    )
    for range_name, keywords in aliases:
        if any(item in query for item in keywords):
            return range_name
    return configured
