from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ParsedBlock:
    block_label: str
    block_content: str
    source_page: Optional[int] = None
    block_bbox: Optional[List[float]] = None
    global_start: Optional[int] = None
    global_end: Optional[int] = None
    paragraph_title: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, Any]:
        data = {
            "block_label": self.block_label,
            "block_content": self.block_content,
        }
        optional_values = {
            "source_page": self.source_page,
            "block_bbox": self.block_bbox,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "paragraph_title": self.paragraph_title,
        }
        data.update({key: value for key, value in optional_values.items() if value is not None and value != ""})
        data.update(self.extra)
        return data


@dataclass
class ParsedDocument:
    document_info: Dict[str, Any]
    parsed_blocks: List[ParsedBlock]
    parser_name: str
    parser_version: str = ""
    raw_payload: Optional[Any] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        info = dict(self.document_info)
        info.setdefault("parser_name", self.parser_name)
        if self.parser_version:
            info.setdefault("parser_version", self.parser_version)
        return {
            "document_info": info,
            "parsed_blocks": [block.to_legacy_dict() for block in self.parsed_blocks],
        }


class DocumentParser(Protocol):
    parser_name: str
    parser_version: str

    def parse_pdf(self, pdf_path: str, save_json: bool = False) -> Dict[str, Any]:
        """Parse a PDF and return the legacy structured dict used by chunker."""
        ...


def parsed_document_from_legacy(
    data: Dict[str, Any],
    parser_name: str = "resolved_json",
    parser_version: str = "legacy",
) -> ParsedDocument:
    blocks = []
    for block in data.get("parsed_blocks", []) or []:
        known_keys = {
            "block_label",
            "block_content",
            "source_page",
            "block_bbox",
            "global_start",
            "global_end",
            "paragraph_title",
        }
        blocks.append(
            ParsedBlock(
                block_label=block.get("block_label", "text") or "text",
                block_content=block.get("block_content", "") or "",
                source_page=block.get("source_page"),
                block_bbox=block.get("block_bbox"),
                global_start=block.get("global_start"),
                global_end=block.get("global_end"),
                paragraph_title=block.get("paragraph_title"),
                extra={key: value for key, value in block.items() if key not in known_keys},
            )
        )
    return ParsedDocument(
        document_info=dict(data.get("document_info", {}) or {}),
        parsed_blocks=blocks,
        parser_name=parser_name,
        parser_version=parser_version,
        raw_payload=data,
    )
