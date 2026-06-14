from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from Finance_RAG.schemas.metadata import FinanceMetadataExtractor
from Finance_RAG.utils import build_logger


class PaddleOcrApiParser:
    parser_name = "paddleocr_api"
    parser_version = "PaddleOCR-VL-1.6"
    ignored_labels = {"header_image", "header", "footer", "footer_image", "number", "aside_text"}
    useless_section_pattern = re.compile(r"免责|重要声明|法律声明|评级说明|分析师声明|投资咨询业务|经济研究所")

    def __init__(
        self,
        token: Optional[str] = None,
        job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        model: str = "PaddleOCR-VL-1.6",
        poll_interval: float = 5.0,
        timeout: int = 300,
        optional_payload: Optional[Dict[str, Any]] = None,
        enrich_metadata: bool = True,
        artifact_dir: Optional[str | Path] = None,
    ):
        self.token = token or os.getenv("PADDLEOCR_API_TOKEN")
        self.job_url = job_url
        self.model = model
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.optional_payload = optional_payload or {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        self.enrich_metadata = enrich_metadata
        self.metadata_extractor = FinanceMetadataExtractor()
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = build_logger()
        return self._logger
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None

    def parse_pdf(self, pdf_path: str, save_json: bool = False) -> Dict[str, Any]:
        path = Path(pdf_path)
        if not self.token:
            raise RuntimeError("未配置 PADDLEOCR_API_TOKEN，无法调用 PaddleOCR API")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        job_id = self._submit_local_file(path)
        result_url = self._wait_job_done(job_id)
        jsonl_text = self._download_jsonl_text(result_url)
        if self.artifact_dir:
            self._save_text_artifact(path.stem, "raw.jsonl", jsonl_text)
        data = self._jsonl_text_to_legacy_document(jsonl_text, file_name=path.stem)

        if self.enrich_metadata:
            data = self.metadata_extractor.enrich_legacy_document(data)

        if save_json:
            json_path = path.parent.parent / "raw_resolve" / f"{path.stem}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            self.logger.info(f"已保存 PaddleOCR API 解析结果: {json_path}")

        return data

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"bearer {self.token}"}

    def _submit_local_file(self, path: Path) -> str:
        self.logger.info(f"提交 PaddleOCR API 任务: {path.name}")
        data = {
            "model": self.model,
            "optionalPayload": json.dumps(self.optional_payload, ensure_ascii=False),
        }
        with path.open("rb") as file:
            response = requests.post(
                self.job_url,
                headers=self._headers(),
                data=data,
                files={"file": file},
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        return payload["data"]["jobId"]

    def _wait_job_done(self, job_id: str) -> str:
        while True:
            response = requests.get(f"{self.job_url}/{job_id}", headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            data = response.json()["data"]
            state = data.get("state")
            if state == "done":
                return data["resultUrl"]["jsonUrl"]
            if state == "failed":
                raise RuntimeError(f"PaddleOCR API 任务失败: {data.get('errorMsg', 'unknown error')}")

            progress = data.get("extractProgress") or {}
            if progress:
                self.logger.info(
                    "PaddleOCR API running: {}/{} pages",
                    progress.get("extractedPages"),
                    progress.get("totalPages"),
                )
            else:
                self.logger.info(f"PaddleOCR API state: {state}")
            time.sleep(self.poll_interval)

    def _download_jsonl_text(self, jsonl_url: str) -> str:
        response = requests.get(jsonl_url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _save_text_artifact(self, file_stem: str, name: str, text: str) -> None:
        if not self.artifact_dir:
            return
        target_dir = self.artifact_dir / file_stem
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / name).write_text(text, encoding="utf-8")

    def _jsonl_text_to_legacy_document(self, jsonl_text: str, file_name: str) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        doc_title_parts: List[str] = []
        page_num = 0
        for raw_line in jsonl_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            result = payload.get("result", {})
            for page_result in result.get("layoutParsingResults", []) or []:
                page_blocks, page_doc_title_parts = self._page_result_to_blocks(page_result, page_num)
                if page_num == 0:
                    doc_title_parts.extend(page_doc_title_parts)
                blocks.extend(page_blocks)
                page_num += 1

        doc_title = "".join(part.strip() for part in doc_title_parts if part.strip())

        return {
            "document_info": {
                "file_name": file_name,
                "doc_source": "未知机构",
                "doc_title": doc_title,
                "parser_name": self.parser_name,
                "parser_version": self.parser_version,
                "parse_status": "parsed",
                "page_count": page_num,
            },
            "parsed_blocks": blocks,
        }

    def _page_result_to_blocks(self, page_result: Dict[str, Any], page_num: int) -> tuple[List[Dict[str, Any]], List[str]]:
        pruned = page_result.get("prunedResult", {}) or {}
        raw_items = pruned.get("parsing_res_list", []) or []
        items = sorted(
            enumerate(raw_items),
            key=lambda pair: (
                pair[1].get("block_order") if pair[1].get("block_order") is not None else pair[0],
                pair[1].get("block_id") if pair[1].get("block_id") is not None else pair[0],
            ),
        )

        blocks: List[Dict[str, Any]] = []
        doc_title_parts: List[str] = []
        current_paragraph_title = ""
        in_useless_section = False

        for _, item in items:
            label = item.get("block_label") or "text"
            content = (item.get("block_content") or "").strip()

            if label == "doc_title":
                if content:
                    doc_title_parts.append(content)
                continue
            if label in self.ignored_labels or not content:
                continue
            if label in {"paragraph_title", "figure_title"}:
                current_paragraph_title = content
                in_useless_section = bool(self.useless_section_pattern.search(content))
                continue
            if in_useless_section:
                continue

            normalized_label = "table&chart" if label in {"table", "chart"} else "text"
            block = {
                "block_label": normalized_label,
                "block_content": content,
                "block_bbox": item.get("block_bbox"),
                "source_page": page_num,
                "parser_block_label": label,
                "parser_block_id": item.get("block_id"),
                "parser_block_order": item.get("block_order"),
                "parser_group_id": item.get("group_id"),
            }
            if current_paragraph_title:
                block["paragraph_title"] = current_paragraph_title
            blocks.append(block)

        return blocks, doc_title_parts
