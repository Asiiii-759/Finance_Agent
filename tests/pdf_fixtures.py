from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from mas_finance.documents import ParsedBlock, ParsedDocument


class MCPPDFParserFixture:
    parser_kind: Literal["mcp"] = "mcp"

    def __init__(self, pages_by_filename: Mapping[str, Mapping[int, str]]) -> None:
        self.pages_by_filename = pages_by_filename
        self.calls: list[Path] = []

    def extract_document(self, file_path: Path) -> ParsedDocument:
        self.calls.append(file_path)
        if file_path.name in self.pages_by_filename:
            pages = self.pages_by_filename[file_path.name]
        elif len(self.pages_by_filename) == 1:
            pages = next(iter(self.pages_by_filename.values()))
        else:
            raise KeyError(file_path.name)
        return ParsedDocument(
            pages=pages,
            blocks=tuple(
                ParsedBlock("text", text, page_number, 0)
                for page_number, text in pages.items()
            ),
            parser_name="mcp_fixture",
            parser_version="1",
        )


def write_stub_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    return path
