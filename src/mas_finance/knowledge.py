"""Versioned local finance knowledge for definitions, formulas and interpretation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from .harness import Tool, ToolArgumentContract, ToolResultKind, ToolSpec, function_tool


@dataclass(frozen=True)
class KnowledgeEntry:
    concept_id: str
    title: str
    aliases: tuple[str, ...]
    content: str


_ENTRIES: tuple[KnowledgeEntry, ...] = (
    KnowledgeEntry(
        "interest_rate_transmission",
        "Interest-rate transmission to companies and bank equities",
        (
            "interest rate impact",
            "interest rates",
            "rate hike",
            "rate hikes",
            "利率上升",
            "利率变化",
            "加息",
        ),
        "Interest-rate changes affect equity values through discount rates, financing costs, demand and credit "
        "conditions; the direction is not mechanically uniform. For banks, higher rates may support net interest "
        "income when earning assets reprice faster than funding, but higher deposit betas and wholesale funding "
        "costs can reverse that benefit. Yield-curve shape, securities valuation losses, liquidity, loan demand, "
        "borrower stress and resulting credit losses must be assessed together. This mechanism is not a forecast "
        "for a specific bank or share price.",
    ),
    KnowledgeEntry(
        "pe_ratio",
        "Price-to-earnings ratio (P/E)",
        ("p/e", "pe ratio", "price to earnings", "市盈率"),
        "P/E equals market price per share divided by earnings per share, or market capitalization divided "
        "by net income when the periods and share basis align. A high or low P/E is not independently a buy "
        "or sell signal; growth, cyclicality, accounting quality and negative earnings affect interpretation.",
    ),
    KnowledgeEntry(
        "pb_ratio",
        "Price-to-book ratio (P/B)",
        ("p/b", "pb ratio", "price to book", "市净率"),
        "P/B equals market capitalization divided by common shareholders' equity. Comparability is strongest "
        "within similar business models; intangible assets, write-downs and negative equity can make it misleading.",
    ),
    KnowledgeEntry(
        "ps_ratio",
        "Price-to-sales ratio (P/S)",
        ("p/s", "ps ratio", "price to sales", "市销率"),
        "P/S equals market capitalization divided by revenue for a consistent period. It does not account for "
        "profitability, capital intensity or leverage, so comparisons should use similar industries and revenue bases.",
    ),
    KnowledgeEntry(
        "ev_ebitda",
        "Enterprise value to EBITDA",
        ("ev/ebitda", "enterprise value", "企业价值倍数"),
        "EV/EBITDA equals enterprise value divided by EBITDA. Enterprise value commonly includes equity value "
        "plus debt and preferred interests minus cash. EBITDA is not cash flow and the exact adjustments must "
        "be consistent across companies.",
    ),
    KnowledgeEntry(
        "net_margin",
        "Net profit margin",
        ("net margin", "profit margin", "净利率", "净利润率"),
        "Net margin equals net income divided by revenue for the same entity and reporting period. It reflects "
        "the share of revenue remaining after all reported expenses but can be affected by one-off items and taxes.",
    ),
    KnowledgeEntry(
        "gross_margin",
        "Gross margin",
        ("gross margin", "gross profit margin", "毛利率"),
        "Gross margin equals gross profit divided by revenue for the same reporting period. Classification of cost "
        "of revenue differs by industry and accounting policy, so peer comparisons require consistent definitions.",
    ),
    KnowledgeEntry(
        "operating_margin",
        "Operating margin",
        ("operating margin", "operating profit margin", "营业利润率", "经营利润率"),
        "Operating margin equals operating income divided by revenue for the same reporting period. It excludes "
        "financing and tax effects but remains sensitive to segment mix and non-recurring operating items.",
    ),
    KnowledgeEntry(
        "roe",
        "Return on equity (ROE)",
        ("roe", "return on equity", "净资产收益率"),
        "ROE equals net income divided by average common shareholders' equity for a period. Using only period-end "
        "equity is an approximation; leverage, buybacks and negative equity can materially distort comparisons.",
    ),
    KnowledgeEntry(
        "roa",
        "Return on assets (ROA)",
        ("roa", "return on assets", "总资产收益率", "资产回报率"),
        "ROA equals net income divided by average total assets for a period. Average assets are preferred because "
        "income is measured over a duration while the balance sheet is measured at points in time.",
    ),
    KnowledgeEntry(
        "current_ratio",
        "Current ratio",
        ("current ratio", "流动比率"),
        "Current ratio equals current assets divided by current liabilities at the same balance-sheet date. It is "
        "a liquidity indicator, but inventory quality, receivable collectability and seasonal working capital matter.",
    ),
    KnowledgeEntry(
        "quick_ratio",
        "Quick ratio",
        ("quick ratio", "acid-test ratio", "速动比率"),
        "The quick ratio generally equals liquid current assets divided by current liabilities. Exact definitions "
        "vary, but inventory and other less-liquid current assets are normally excluded; current assets alone are "
        "not enough to calculate it reliably.",
    ),
    KnowledgeEntry(
        "debt_to_equity",
        "Debt-to-equity ratio",
        ("debt to equity", "debt-to-equity", "d/e", "产权比率", "债务权益比"),
        "Debt-to-equity compares a defined debt measure with shareholders' equity. Analysts must state whether "
        "debt means interest-bearing debt or total liabilities; negative equity makes the ratio hard to interpret.",
    ),
    KnowledgeEntry(
        "cagr",
        "Compound annual growth rate (CAGR)",
        ("cagr", "compound annual growth", "复合增长率", "年复合增长率"),
        "CAGR equals (ending value / beginning value) raised to 1 / years, minus 1. It is a smoothed rate between "
        "two endpoints and does not show interim volatility or cash-flow timing.",
    ),
    KnowledgeEntry(
        "volatility",
        "Annualized volatility",
        ("volatility", "annualized volatility", "波动率", "年化波动率"),
        "Annualized historical volatility is commonly sample standard deviation of periodic returns multiplied "
        "by the square root of periods per year. The scaling assumes a stable return process and is not a loss bound.",
    ),
    KnowledgeEntry(
        "sharpe_ratio",
        "Sharpe ratio",
        ("sharpe", "sharpe ratio", "夏普比率"),
        "Sharpe ratio equals excess return over a consistently measured risk-free rate divided by return volatility. "
        "Results depend on period, sampling frequency, annualization and the treatment of non-normal returns.",
    ),
    KnowledgeEntry(
        "max_drawdown",
        "Maximum drawdown",
        ("maximum drawdown", "max drawdown", "最大回撤", "回撤"),
        "Maximum drawdown is the largest peak-to-trough percentage decline in an observed value series. It is "
        "path-dependent and backward-looking; the chosen frequency and sample window affect the result.",
    ),
    KnowledgeEntry(
        "dcf",
        "Discounted cash flow (DCF)",
        ("dcf", "discounted cash flow", "现金流折现", "折现现金流"),
        "DCF estimates value by discounting forecast cash flows and a terminal value at a rate consistent with "
        "their risk and capital claim. Forecast horizon, terminal assumptions and discount rate usually drive "
        "sensitivity.",
    ),
    KnowledgeEntry(
        "npv",
        "Net present value (NPV)",
        ("npv", "net present value", "净现值"),
        "NPV is the sum of cash flows discounted to a common date, including the initial investment. It requires "
        "cash-flow timing and a discount rate aligned with the currency, nominal/real basis and risk.",
    ),
    KnowledgeEntry(
        "irr",
        "Internal rate of return (IRR)",
        ("irr", "internal rate of return", "内部收益率"),
        "IRR is a discount rate that sets NPV to zero. Non-conventional cash-flow patterns can produce multiple or "
        "no IRRs, and IRR can rank mutually exclusive projects differently from NPV.",
    ),
    KnowledgeEntry(
        "bond_duration",
        "Bond duration",
        ("duration", "modified duration", "bond duration", "久期", "修正久期"),
        "Modified duration approximates the percentage price change of a bond for a small yield change. Convexity "
        "becomes important for larger moves, and embedded options can make effective duration more appropriate.",
    ),
)


class FinancialKnowledgeBase:
    def search(
        self,
        query: str,
        *,
        concepts: Sequence[str] = (),
        top_k: int = 5,
    ) -> EvidenceBundle:
        if not 1 <= top_k <= 20:
            raise ValueError("knowledge top_k must be between 1 and 20")
        requested = {item.casefold() for item in concepts}
        query_terms = _tokens(query)
        scored: list[tuple[int, KnowledgeEntry]] = []
        for entry in _ENTRIES:
            exact = int(entry.concept_id.casefold() in requested)
            alias_score = sum(alias.casefold() in query.casefold() for alias in entry.aliases)
            overlap = len(query_terms.intersection(_tokens(" ".join((entry.title, *entry.aliases)))))
            score = exact * 100 + alias_score * 10 + overlap
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].concept_id))
        bundle = EvidenceBundle()
        for _score, entry in scored[:top_k]:
            source = SourceRef.create(
                source_type=SourceType.DOCUMENT,
                title=entry.title,
                locator=f"knowledge://finance/{entry.concept_id}/v1",
                provider="MAS Finance curated knowledge",
                as_of="v1",
                metadata={"concept_id": entry.concept_id, "content_version": "1"},
            )
            bundle.add_evidence(
                Evidence.create(
                    source=source,
                    content=entry.content,
                    entity=entry.concept_id,
                    field_name="definition",
                    confidence=1.0,
                    tags=("curated_finance_knowledge", entry.concept_id),
                )
            )
        return bundle


def finance_knowledge_harness_tool(knowledge: FinancialKnowledgeBase | None = None) -> Tool:
    store = knowledge or FinancialKnowledgeBase()

    def invoke(arguments: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        concepts = arguments.get("concepts") or []
        if not isinstance(concepts, list):
            raise ValueError("knowledge concepts must be a list")
        bundle = store.search(
            str(arguments.get("query") or ""),
            concepts=[str(item) for item in concepts],
            top_k=int(arguments.get("top_k", 5)),
        )
        gaps = (
            []
            if bundle.evidence
            else [
                {
                    "code": "finance_knowledge_not_found",
                    "message": "No curated finance concept matched the question.",
                }
            ]
        )
        return {"bundle": bundle.to_dict(), "gaps": gaps}

    return function_tool(
        ToolSpec(
            name="finance.knowledge",
            description="Retrieve versioned definitions, formulas and interpretation caveats for finance concepts.",
            capability="knowledge.read",
            network_access=False,
            timeout_seconds=5,
            result_kind=ToolResultKind.EVIDENCE_BUNDLE,
            arguments=ToolArgumentContract(
                required=frozenset({"query"}),
                optional=frozenset({"concepts", "top_k"}),
            ),
        ),
        invoke,
    )


def detect_finance_concepts(query: str) -> tuple[str, ...]:
    normalized = query.casefold()
    return tuple(
        entry.concept_id for entry in _ENTRIES if any(alias.casefold() in normalized for alias in entry.aliases)
    )


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    tokens = set(re.findall(r"[a-z0-9/.-]{2,}", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
    tokens.update(value[index : index + 2] for value in cjk for index in range(max(0, len(value) - 1)))
    return tokens
