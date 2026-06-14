from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from Finance_RAG.parsers.paddle_ocr_api import PaddleOcrApiParser
from Finance_RAG.schemas.metadata import FinanceMetadataExtractor


DEFAULT_CONTENT_DIR = Path("Finance_RAG/Data/knowledge_base/Finance/content")
DEFAULT_OUTPUT_DIR = Path("Finance_RAG/Data/calibration/paddleocr_api")


def select_small_pdfs(limit: int) -> List[Path]:
    files = sorted(DEFAULT_CONTENT_DIR.glob("*.pdf"), key=lambda path: path.stat().st_size)
    return files[:limit]


def summarize_legacy_document(data: Dict[str, Any]) -> Dict[str, Any]:
    info = data.get("document_info", {}) or {}
    blocks = data.get("parsed_blocks", []) or []
    labels = Counter(block.get("block_label", "unknown") for block in blocks)
    extraction = FinanceMetadataExtractor().extract_from_legacy(data)
    return {
        "document_info_keys": sorted(info.keys()),
        "document_info_preview": {
            key: info.get(key)
            for key in [
                "file_name",
                "doc_source",
                "doc_title",
                "parser_name",
                "parser_version",
                "page_count",
                "document_title",
                "document_date",
                "publish_date",
                "report_type",
            ]
            if key in info
        },
        "block_count": len(blocks),
        "block_labels": dict(labels),
        "first_block_shapes": [
            {
                "block_label": block.get("block_label"),
                "source_page": block.get("source_page"),
                "global_start": block.get("global_start"),
                "global_end": block.get("global_end"),
                "paragraph_title": block.get("paragraph_title"),
                "content_len": len(block.get("block_content", "") or ""),
            }
            for block in blocks[:8]
        ],
        "metadata_extraction": extraction.to_dict(),
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_probe(pdf_paths: List[Path], output_dir: Path) -> List[Dict[str, Any]]:
    parser = PaddleOcrApiParser(artifact_dir=output_dir)
    summaries = []
    for pdf_path in pdf_paths:
        data = parser.parse_pdf(str(pdf_path), save_json=False)
        file_dir = output_dir / pdf_path.stem
        write_json(file_dir / "legacy.json", data)
        summary = summarize_legacy_document(data)
        summary["pdf_file"] = str(pdf_path)
        write_json(file_dir / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summaries


def main() -> None:
    cli = argparse.ArgumentParser(description="Run small-sample PaddleOCR API parser calibration.")
    cli.add_argument("pdf", nargs="*", type=Path, help="PDF files to parse. Defaults to small PDFs in the Finance KB.")
    cli.add_argument("--limit", type=int, default=2, help="Number of default small PDFs to parse when no path is given.")
    cli.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = cli.parse_args()

    pdf_paths = args.pdf or select_small_pdfs(limit=args.limit)
    run_probe(pdf_paths=pdf_paths, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
