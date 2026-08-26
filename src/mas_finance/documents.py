from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

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


class PDFOCRProvider(Protocol):
    """Optional bounded OCR seam; returned page numbers are one-based."""

    def extract_document(self, file_path: Path, expected_pages: int) -> Mapping[int, str]: ...


def parse_pdf_document(
    file_path: Path,
    *,
    include_pages: bool = False,
    max_pages: int = 500,
    max_file_bytes: int = 25 * 1024 * 1024,
    max_text_characters: int = 5_000_000,
    display_name: str | None = None,
    ocr_provider: PDFOCRProvider | None = None,
    min_native_text_characters: int = 24,
) -> dict[str, Any]:
    # Keep PyMuPDF optional for core-only installations.
    import fitz  # type: ignore[import-untyped]

    if max_pages < 1 or max_file_bytes < 1 or max_text_characters < 1 or min_native_text_characters < 0:
        raise ValueError("PDF processing limits must be positive")
    if file_path.stat().st_size > max_file_bytes:
        raise ValueError(f"PDF exceeds the {max_file_bytes}-byte processing limit")
    doc = fitz.open(file_path)
    try:
        if doc.page_count > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages}-page processing limit")
        pages: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        extracted_characters = 0
        for page_number, page in enumerate(doc, start=1):
            # sort=True follows visual reading order instead of PDF object
            # insertion order, which is materially better for multi-column
            # filings while preserving a small extraction surface.
            text = _normalize_extracted_text(page.get_text("text", sort=True))
            image_count = len(page.get_images(full=True))
            extraction_method = "native"
            native_characters = len(text)
            extracted_characters += native_characters
            if extracted_characters > max_text_characters:
                raise ValueError(f"PDF extracted text exceeds the {max_text_characters}-character limit")
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "extraction_method": extraction_method,
                    "text_characters": len(text),
                    "image_count": image_count,
                }
            )
        ocr_candidates = {
            int(page["page_number"])
            for page in pages
            if int(page["text_characters"]) < min_native_text_characters and int(page["image_count"]) > 0
        }
        if ocr_candidates and ocr_provider:
            ocr_pages = ocr_provider.extract_document(file_path, len(pages))
            if not isinstance(ocr_pages, Mapping):
                raise ValueError("PDF OCR provider must return a page-to-text mapping")
            unknown_pages = set(ocr_pages).difference(range(1, len(pages) + 1))
            if unknown_pages:
                raise ValueError("PDF OCR provider returned an invalid page number")
            for page_number in ocr_candidates:
                ocr_text = ocr_pages.get(page_number, "")
                if not isinstance(ocr_text, str):
                    raise ValueError("PDF OCR provider must return text")
                normalized_ocr = _normalize_extracted_text(ocr_text)
                page = pages[page_number - 1]
                if len(normalized_ocr) > int(page["text_characters"]):
                    page["text"] = normalized_ocr
                    page["text_characters"] = len(normalized_ocr)
                    page["extraction_method"] = "ocr"
        extracted_characters = 0
        for page in pages:
            text = str(page["text"])
            if not text and int(page["image_count"]) > 0:
                warnings.append(
                    {
                        "code": "ocr_required",
                        "page_number": page["page_number"],
                        "message": "Page contains images but no extractable text.",
                    }
                )
            extracted_characters += len(text)
            if extracted_characters > max_text_characters:
                raise ValueError(f"PDF extracted text exceeds the {max_text_characters}-character limit")
    finally:
        doc.close()
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
        "ocr_page_count": sum(page["extraction_method"] == "ocr" for page in pages),
        "warnings": warnings,
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
