from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from Finance_RAG.schemas.document import ParsedDocument, parsed_document_from_legacy


@dataclass
class MetadataCandidate:
    name: str
    candidate_type: str
    confidence: float
    evidence: str
    source: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetadataExtractionReport:
    document_title: Optional[str] = None
    document_date: Optional[str] = None
    publish_date: Optional[str] = None
    report_type: str = "unknown"
    organizations: List[MetadataCandidate] = field(default_factory=list)
    authors: List[MetadataCandidate] = field(default_factory=list)
    companies: List[MetadataCandidate] = field(default_factory=list)
    industries: List[MetadataCandidate] = field(default_factory=list)
    tickers: List[MetadataCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinanceMetadataExtractor:
    """Conservative rule-based metadata extraction for finance research docs."""

    DATE_PATTERNS = [
        re.compile(r"(?P<year>20\d{2})[-/.年](?P<month>\d{1,2})[-/.月](?P<day>\d{1,2})日?"),
        re.compile(r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})"),
    ]
    TICKER_PATTERN = re.compile(r"(?<!\d)(?P<ticker>\d{6})(?:\.(?P<exchange>SH|SZ|BJ))?(?!\d)", re.IGNORECASE)
    AUTHOR_PATTERN = re.compile(r"(?:分析师|研究员|联系人)[：:\s]+(?P<name>[\u4e00-\u9fa5]{2,4})")
    REPORT_TYPE_RULES = [
        ("industry", re.compile(r"行业|板块|产业链|赛道")),
        ("macro", re.compile(r"宏观|利率|通胀|出口|社融|PMI|GDP|CPI|PPI")),
        ("strategy", re.compile(r"策略|配置|市场周报|投资周报")),
        ("company", re.compile(r"公司|个股|深度|点评|首次覆盖|季报|年报|中报")),
    ]
    INDUSTRY_HINTS = [
        "低空经济",
        "大飞机",
        "半导体",
        "汽车",
        "电力设备",
        "新能源",
        "电子",
        "军工",
        "公用事业",
        "银行",
        "保险",
        "券商",
        "医药",
        "地产",
        "计算机",
        "通信",
        "消费",
        "传媒",
        "机械",
        "化工",
        "有色",
        "煤炭",
        "钢铁",
    ]
    BAD_SOURCE_PATTERNS = [
        re.compile(r"too blurry|recognize|image|无法识别|看不清", re.IGNORECASE),
        re.compile(r"^未知机构$"),
    ]
    BAD_TITLE_PATTERNS = [
        re.compile(r"^未命名文档$"),
        re.compile(r"^untitled$", re.IGNORECASE),
        re.compile(r"^图表?\s*\d+"),
        re.compile(r"^表\s*\d+"),
    ]
    GENERIC_COMPANY_PREFIX = re.compile(r"公司|简评|点评|深度|业绩|年报|中报|季报|事件|首次覆盖|新股覆盖|研究")

    def extract_from_legacy(self, data: Dict[str, Any]) -> MetadataExtractionReport:
        parser_name = data.get("document_info", {}).get("parser_name", "legacy_json")
        return self.extract(parsed_document_from_legacy(data, parser_name=parser_name))

    def extract(self, document: ParsedDocument) -> MetadataExtractionReport:
        info = document.document_info or {}
        file_name = str(info.get("file_name") or "")
        doc_title = self._clean_title(str(info.get("document_title") or info.get("doc_title") or ""))
        first_text = self._first_text(document)
        first_titles = self._first_titles(document)
        search_text = "\n".join([file_name, doc_title, *first_titles[:5], first_text[:1500]])

        report = MetadataExtractionReport(
            document_title=self._guess_title(file_name, doc_title, first_titles),
            document_date=self._guess_date(search_text),
            publish_date=self._guess_date(search_text),
            report_type=self._guess_report_type(search_text),
        )

        self._append_organization(report, str(info.get("doc_source") or ""))
        report.authors.extend(self._extract_authors(search_text))
        report.tickers.extend(self._extract_tickers(search_text))
        report.industries.extend(self._extract_industries(search_text))
        report.companies.extend(self._extract_company_candidates(file_name, doc_title, search_text, report.report_type))

        if not report.companies and report.report_type == "company":
            report.warnings.append("report looks company-oriented, but no reliable company candidate was found")
        if not report.industries and report.report_type == "industry":
            report.warnings.append("report looks industry-oriented, but no rule-based industry candidate was found")

        return report

    def enrich_legacy_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(data)
        document_info = dict(enriched.get("document_info", {}) or {})
        report = self.extract_from_legacy(enriched)
        document_info["metadata_extraction"] = report.to_dict()
        document_info.setdefault("document_title", report.document_title)
        document_info.setdefault("document_date", report.document_date)
        document_info.setdefault("publish_date", report.publish_date)
        document_info.setdefault("report_type", report.report_type)
        enriched["document_info"] = document_info
        return enriched

    def _append_organization(self, report: MetadataExtractionReport, doc_source: str) -> None:
        source = doc_source.strip()
        if source and not any(pattern.search(source) for pattern in self.BAD_SOURCE_PATTERNS):
            report.organizations.append(
                MetadataCandidate(
                    name=source,
                    candidate_type="organization",
                    confidence=0.75,
                    evidence="document_info.doc_source",
                    source="parser_metadata",
                )
            )
            return
        report.warnings.append("parser did not provide a reliable organization/doc_source")

    def _first_text(self, document: ParsedDocument) -> str:
        pieces = []
        for block in document.parsed_blocks[:40]:
            content = block.block_content.strip()
            if content:
                pieces.append(content)
        return "\n".join(pieces)

    def _first_titles(self, document: ParsedDocument) -> List[str]:
        titles = []
        for block in document.parsed_blocks[:40]:
            if block.block_label in {"paragraph_title", "figure_title"} and block.block_content.strip():
                titles.append(block.block_content.strip())
        return titles

    def _clean_title(self, title: str) -> str:
        text = title.strip()
        if any(pattern.search(text) for pattern in self.BAD_TITLE_PATTERNS):
            return ""
        return text

    def _file_title(self, file_name: str) -> str:
        text = file_name.strip()
        for suffix in (".json", ".pdf", ".docx", ".doc", ".xlsx", ".xls"):
            if text.lower().endswith(suffix):
                return text[: -len(suffix)]
        return text

    def _guess_title(self, file_name: str, doc_title: str, titles: List[str]) -> Optional[str]:
        if doc_title:
            return doc_title
        for title in titles:
            cleaned_title = self._clean_title(title)
            if cleaned_title:
                return cleaned_title
        stem = self._file_title(file_name)
        return stem or None

    def _guess_date(self, text: str) -> Optional[str]:
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                year = int(match.group("year"))
                month = int(match.group("month"))
                day = int(match.group("day"))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    def _guess_report_type(self, text: str) -> str:
        for report_type, pattern in self.REPORT_TYPE_RULES:
            if pattern.search(text):
                return report_type
        return "unknown"

    def _extract_authors(self, text: str) -> List[MetadataCandidate]:
        return self._dedupe_candidates(
            MetadataCandidate(
                name=match.group("name"),
                candidate_type="author",
                confidence=0.65,
                evidence=match.group(0),
                source="regex:first_pages",
            )
            for match in self.AUTHOR_PATTERN.finditer(text)
        )

    def _extract_tickers(self, text: str) -> List[MetadataCandidate]:
        candidates = []
        for match in self.TICKER_PATTERN.finditer(text):
            ticker = match.group("ticker")
            exchange = (match.group("exchange") or "").upper()
            candidates.append(
                MetadataCandidate(
                    name=f"{ticker}.{exchange}" if exchange else ticker,
                    candidate_type="ticker",
                    confidence=0.75 if exchange else 0.55,
                    evidence=match.group(0),
                    source="regex:first_pages",
                    extra={"exchange": exchange} if exchange else {},
                )
            )
        return self._dedupe_candidates(candidates)

    def _extract_industries(self, text: str) -> List[MetadataCandidate]:
        candidates = []
        for hint in self.INDUSTRY_HINTS:
            if hint in text:
                candidates.append(
                    MetadataCandidate(
                        name=hint,
                        candidate_type="industry",
                        confidence=0.55 if hint in text[:120] else 0.45,
                        evidence=hint,
                        source="keyword:first_pages",
                    )
                )
        return self._dedupe_candidates(candidates)

    def _extract_company_candidates(
        self,
        file_name: str,
        doc_title: str,
        text: str,
        report_type: str,
    ) -> List[MetadataCandidate]:
        candidates = []
        title = doc_title or self._file_title(file_name)
        title_parts = [part.strip() for part in re.split(r"[：:丨|]", title) if part.strip()]
        for index, part in enumerate(title_parts[:3]):
            if self._looks_like_company_name(part):
                candidates.append(
                    MetadataCandidate(
                        name=part,
                        candidate_type="company",
                        confidence=0.65 if report_type == "company" else 0.45,
                        evidence=title,
                        source="title_segment" if index else "title_prefix",
                    )
                )

        for ticker_candidate in self._extract_tickers(text):
            candidates.append(
                MetadataCandidate(
                    name=ticker_candidate.name,
                    candidate_type="company_or_security",
                    confidence=max(0.55, ticker_candidate.confidence - 0.1),
                    evidence=ticker_candidate.evidence,
                    source="ticker_proxy",
                )
            )
        return self._dedupe_candidates(candidates)

    def _looks_like_company_name(self, text: str) -> bool:
        if any(pattern.search(text) for pattern in self.BAD_TITLE_PATTERNS):
            return False
        if not 2 <= len(text) <= 12:
            return False
        if self.GENERIC_COMPANY_PREFIX.search(text):
            return False
        if re.search(r"行业|策略|周报|月报|宏观|专题|白皮书|报告|研究|产业|经济", text):
            return False
        return bool(re.search(r"[\u4e00-\u9fa5A-Za-z]", text))

    def _dedupe_candidates(self, candidates: Iterable[MetadataCandidate]) -> List[MetadataCandidate]:
        deduped: Dict[tuple[str, str], MetadataCandidate] = {}
        for candidate in candidates:
            key = (candidate.candidate_type, candidate.name)
            existing = deduped.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                deduped[key] = candidate
        return list(deduped.values())
