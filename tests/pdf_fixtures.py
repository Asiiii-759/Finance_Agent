from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal


class MCPPDFParserFixture:
    parser_kind: Literal["mcp"] = "mcp"

    def __init__(self, pages_by_filename: Mapping[str, Mapping[int, str]]) -> None:
        self.pages_by_filename = pages_by_filename
        self.calls: list[Path] = []

    def extract_document(self, file_path: Path) -> Mapping[int, str]:
        self.calls.append(file_path)
        if file_path.name in self.pages_by_filename:
            return self.pages_by_filename[file_path.name]
        if len(self.pages_by_filename) == 1:
            return next(iter(self.pages_by_filename.values()))
        raise KeyError(file_path.name)


def write_stub_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    return path
