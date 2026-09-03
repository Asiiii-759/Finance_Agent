from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ParsedBlock:
    label: Literal["text", "heading", "table", "chart"]
    content: str
    page_number: int
    order: int
    paragraph_title: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.label not in {"text", "heading", "table", "chart"}:
            raise ValueError("parsed block label is invalid")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("parsed block content is invalid")
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or self.page_number < 1:
            raise ValueError("parsed block page number is invalid")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("parsed block order is invalid")
        if self.paragraph_title is not None and (
            not isinstance(self.paragraph_title, str) or len(self.paragraph_title) > 1_000
        ):
            raise ValueError("parsed block paragraph title is invalid")
        if self.bbox is not None and (
            len(self.bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.bbox)
            or any(not math.isfinite(float(value)) for value in self.bbox)
        ):
            raise ValueError("parsed block bbox is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedDocument:
    pages: Mapping[int, str]
    blocks: tuple[ParsedBlock, ...]
    parser_name: str
    parser_version: str


class PDFDocumentParser(Protocol):
    """Managed PDF parser seam; returned page numbers are one-based."""

    @property
    def parser_kind(self) -> Literal["paddleocr", "mcp"]: ...

    def extract_document(self, file_path: Path) -> ParsedDocument: ...


def parse_pdf_document(
    file_path: Path,
    *,
    include_pages: bool = False,
    max_pages: int = 500,
    max_file_bytes: int = 25 * 1024 * 1024,
    max_text_characters: int = 5_000_000,
    display_name: str | None = None,
    document_parser: PDFDocumentParser | None = None,
) -> dict[str, Any]:
    if max_pages < 1 or max_file_bytes < 1 or max_text_characters < 1:
        raise ValueError("PDF processing limits must be positive")
    if file_path.stat().st_size > max_file_bytes:
        raise ValueError(f"PDF exceeds the {max_file_bytes}-byte processing limit")
    if not file_path.read_bytes()[:5] == b"%PDF-":
        raise ValueError("document input must be a PDF")
    if document_parser is None:
        raise ValueError("a PaddleOCR or MCP PDF document parser is required")
    if document_parser.parser_kind not in {"paddleocr", "mcp"}:
        raise ValueError("PDF document parser must be PaddleOCR or MCP")
    extracted = document_parser.extract_document(file_path)
    if not isinstance(extracted, ParsedDocument):
        raise ValueError("PDF parser must return ParsedDocument")
    parsed_pages = extracted.pages
    if not isinstance(parsed_pages, Mapping) or not parsed_pages:
        raise ValueError("PDF document parser must return a non-empty page-to-text mapping")
    if len(parsed_pages) > max_pages:
        raise ValueError(f"PDF exceeds the {max_pages}-page processing limit")
    expected_page_numbers = set(range(1, len(parsed_pages) + 1))
    if set(parsed_pages) != expected_page_numbers:
        raise ValueError("PDF document parser returned non-contiguous page numbers")
    pages: list[dict[str, Any]] = []
    extracted_characters = 0
    for page_number in range(1, len(parsed_pages) + 1):
        text = parsed_pages[page_number]
        if not isinstance(text, str):
            raise ValueError("PDF document parser must return text for every page")
        normalized_text = _normalize_extracted_text(text)
        extracted_characters += len(normalized_text)
        if extracted_characters > max_text_characters:
            raise ValueError(f"PDF extracted text exceeds the {max_text_characters}-character limit")
        pages.append(
            {
                "page_number": page_number,
                "text": normalized_text,
                "extraction_method": document_parser.parser_kind,
                "text_characters": len(normalized_text),
                "blocks": [
                    {
                        **block.to_dict(),
                        "content": _normalize_extracted_text(block.content),
                    }
                    for block in extracted.blocks
                    if block.page_number == page_number and _normalize_extracted_text(block.content)
                ],
            }
        )
    if not extracted.blocks:
        raise ValueError("PDF parser returned no structured blocks")
    if any(block.page_number not in expected_page_numbers for block in extracted.blocks):
        raise ValueError("PDF parser returned a block outside the document page range")
    visible_name = Path(display_name).name if display_name else file_path.name
    if not visible_name or len(visible_name) > 255:
        raise ValueError("document display name is invalid")
    result: dict[str, Any] = {
        "document_id": sha256(file_path.read_bytes()).hexdigest(),
        "filename": visible_name,
        "page_count": len(pages),
        "text_page_count": sum(bool(page["text"]) for page in pages),
        "parsed_page_count": len(pages),
        "parser_kind": document_parser.parser_kind,
        "parser_name": extracted.parser_name,
        "parser_version": extracted.parser_version,
        "warnings": [],
    }
    if include_pages:
        # Page records are an ingestion contract, not a second copy persisted in
        # memory.  Keeping page boundaries here lets downstream RAG citations
        # point back to the exact page instead of an opaque whole-document span.
        result["pages"] = [dict(page) for page in pages if page["text"]]
    return result


def _normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character for character in normalized if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
