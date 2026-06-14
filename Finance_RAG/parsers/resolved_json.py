from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from Finance_RAG.schemas.metadata import FinanceMetadataExtractor


class ResolvedJsonParser:
    parser_name = "resolved_json"
    parser_version = "legacy"

    def __init__(self, enrich_metadata: bool = True):
        self.enrich_metadata = enrich_metadata
        self.metadata_extractor = FinanceMetadataExtractor()

    def parse_json(self, json_path: str | Path) -> Dict[str, Any]:
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        document_info = dict(data.get("document_info", {}) or {})
        document_info.setdefault("parser_name", self.parser_name)
        document_info.setdefault("parser_version", self.parser_version)
        document_info.setdefault("file_name", path.stem)
        data["document_info"] = document_info

        if self.enrich_metadata:
            data = self.metadata_extractor.enrich_legacy_document(data)
        return data
