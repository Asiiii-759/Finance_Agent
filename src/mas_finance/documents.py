from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol

COMPANY_HINTS = {
    "apple": "Apple",
    "microsoft": "Microsoft",
    "tesla": "Tesla",
    "amazon": "Amazon",
    "google": "Alphabet",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "nvidia": "NVIDIA",
}


class PDFDocumentParser(Protocol):
    """Managed PDF parser seam; returned page numbers are one-based."""

    @property
    def parser_kind(self) -> Literal["paddleocr", "mcp"]: ...

    def extract_document(self, file_path: Path) -> Mapping[int, str]: ...


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
    parsed_pages = document_parser.extract_document(file_path)
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
            }
        )
    visible_name = Path(display_name).name if display_name else file_path.name
    if not visible_name or len(visible_name) > 255:
        raise ValueError("document display name is invalid")
    full_text = "\n".join(str(page["text"]) for page in pages).strip()
    lowered = full_text.lower()
    detected_companies = detect_companies(f"{visible_name}\n{lowered}")
    result: dict[str, Any] = {
        "document_id": sha256(file_path.read_bytes()).hexdigest(),
        "filename": visible_name,
        "page_count": len(pages),
        "text_page_count": sum(bool(page["text"]) for page in pages),
        "parsed_page_count": len(pages),
        "parser_kind": document_parser.parser_kind,
        "warnings": [],
        "detected_companies": detected_companies,
    }
    if include_pages:
        # Page records are an ingestion contract, not a second copy persisted in
        # memory.  Keeping page boundaries here lets downstream RAG citations
        # point back to the exact page instead of an opaque whole-document span.
        result["pages"] = [dict(page) for page in pages if page["text"]]
    return result


def detect_companies(text: str) -> list[str]:
    lowered = text.casefold()
    return sorted({name for key, name in COMPANY_HINTS.items() if key in lowered})


def _normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character for character in normalized if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
