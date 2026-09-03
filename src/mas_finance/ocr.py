"""Bounded adapter for PaddleOCR document parsing.

Page Markdown and structured layout blocks are consumed. Remote image assets
are deliberately ignored.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from .documents import ParsedBlock, ParsedDocument


@dataclass(frozen=True)
class PaddleOCRClient:
    parser_kind: Literal["paddleocr"] = field(default="paddleocr", init=False)
    access_token: str = field(repr=False)
    job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    model: str = "PaddleOCR-VL-1.6"
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 2.0
    max_poll_requests: int = 120
    max_file_bytes: int = 25 * 1024 * 1024
    max_pages: int = 500
    max_result_bytes: int = 10_000_000
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_https_url(self.job_url, field_name="PaddleOCR job URL", allow_query=False)
        if (
            not self.access_token
            or len(self.access_token) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in self.access_token)
        ):
            raise ValueError("PaddleOCR access token is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.model):
            raise ValueError("PaddleOCR model name is invalid")
        if not 0.1 <= self.request_timeout_seconds <= 120:
            raise ValueError("PaddleOCR request timeout is outside the supported range")
        if not 0 <= self.poll_interval_seconds <= 30 or not 1 <= self.max_poll_requests <= 600:
            raise ValueError("PaddleOCR polling limits are invalid")
        if not 1_024 <= self.max_file_bytes <= 100_000_000:
            raise ValueError("PaddleOCR file limit is outside the supported range")
        if not 1 <= self.max_pages <= 10_000:
            raise ValueError("PaddleOCR page limit is outside the supported range")
        if not 1_024 <= self.max_result_bytes <= 50_000_000:
            raise ValueError("PaddleOCR result limit is outside the supported range")

    def extract_document(self, file_path: Path) -> ParsedDocument:
        if file_path.stat().st_size > self.max_file_bytes:
            raise ValueError("PaddleOCR input exceeds the file limit")
        content = file_path.read_bytes()
        if not content.startswith(b"%PDF-"):
            raise ValueError("PaddleOCR input must be a PDF")
        headers = {"Authorization": f"bearer {self.access_token}"}
        with httpx.Client(
            timeout=self.request_timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
            headers=headers,
        ) as client:
            response = client.post(
                self.job_url,
                data={
                    "model": self.model,
                    "optionalPayload": json.dumps(
                        {
                            "useDocOrientationClassify": False,
                            "useDocUnwarping": False,
                            "useChartRecognition": False,
                        }
                    ),
                },
                files={"file": (file_path.name, content, "application/pdf")},
            )
            response.raise_for_status()
            job_id = _job_id(response)
            result_url = self._poll(client, job_id)
        # Result objects commonly live on a different signed storage host.
        # Use a fresh client so the PaddleOCR bearer token is never forwarded.
        with httpx.Client(
            timeout=self.request_timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        ) as result_client:
            raw_result = _bounded_get(result_client, result_url, self.max_result_bytes)
        return _parse_document(raw_result, self.max_pages, self.model)

    def _poll(self, client: httpx.Client, job_id: str) -> str:
        status_url = f"{self.job_url.rstrip('/')}/{job_id}"
        for attempt in range(self.max_poll_requests):
            response = client.get(status_url)
            response.raise_for_status()
            data = _response_data(response, "PaddleOCR status")
            state = data.get("state")
            if state == "done":
                result = data.get("resultUrl")
                result_url = result.get("jsonUrl") if isinstance(result, Mapping) else None
                if not isinstance(result_url, str):
                    raise ValueError("PaddleOCR completed job has no JSON result URL")
                _validate_https_url(result_url, field_name="PaddleOCR result URL")
                return result_url
            if state == "failed":
                raise RuntimeError("PaddleOCR document parsing failed")
            if state not in {"pending", "running"}:
                raise ValueError("PaddleOCR returned an unknown job state")
            if attempt + 1 < self.max_poll_requests and self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)
        raise TimeoutError("PaddleOCR polling limit was exhausted")


def _job_id(response: httpx.Response) -> str:
    data = _response_data(response, "PaddleOCR submission")
    job_id = data.get("jobId")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", job_id):
        raise ValueError("PaddleOCR submission has no valid job id")
    return job_id


def _response_data(response: httpx.Response, label: str) -> Mapping[str, Any]:
    if len(response.content) > 1_000_000:
        raise ValueError(f"{label} response exceeds the byte limit")
    try:
        payload = json.loads(response.content, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} returned invalid JSON") from exc
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} response has no data object")
    return data


def _bounded_get(client: httpx.Client, url: str, limit: int) -> bytes:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise ValueError("PaddleOCR JSONL result exceeds the byte limit")
            chunks.append(chunk)
    return b"".join(chunks)


def _parse_document(raw: bytes, max_pages: int, model: str) -> ParsedDocument:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("PaddleOCR JSONL result is not UTF-8") from exc
    pages: dict[int, str] = {}
    blocks: list[ParsedBlock] = []
    page_number = 1
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError("PaddleOCR JSONL contains invalid JSON") from exc
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping):
            raise ValueError("PaddleOCR JSONL line has no result object")
        layouts = result.get("layoutParsingResults")
        if not isinstance(layouts, list):
            raise ValueError("PaddleOCR result has no layoutParsingResults list")
        for layout in layouts:
            markdown = layout.get("markdown") if isinstance(layout, Mapping) else None
            text = markdown.get("text") if isinstance(markdown, Mapping) else None
            if not isinstance(text, str):
                raise ValueError("PaddleOCR page has no Markdown text")
            if page_number > max_pages:
                raise ValueError("PaddleOCR returned more pages than the configured limit")
            pages[page_number] = text
            blocks.extend(_layout_blocks(layout, page_number, text))
            page_number += 1
    return ParsedDocument(
        pages=pages,
        blocks=tuple(blocks),
        parser_name="paddleocr_api",
        parser_version=model,
    )


def _layout_blocks(layout: Mapping[str, Any], page_number: int, markdown_text: str) -> list[ParsedBlock]:
    pruned = layout.get("prunedResult")
    raw_items = pruned.get("parsing_res_list") if isinstance(pruned, Mapping) else None
    if not isinstance(raw_items, list) or not raw_items:
        return [ParsedBlock("text", markdown_text, page_number, 0)]
    ordered = sorted(
        (item for item in raw_items if isinstance(item, Mapping)),
        key=lambda item: (
            _integer_order(item.get("block_order")),
            _integer_order(item.get("block_id")),
        ),
    )
    blocks: list[ParsedBlock] = []
    paragraph_title: str | None = None
    for order, item in enumerate(ordered):
        raw_label = str(item.get("block_label") or "text").casefold()
        content = str(item.get("block_content") or "").strip()
        if not content or raw_label in {"header", "header_image", "footer", "footer_image", "number"}:
            continue
        label: Literal["text", "heading", "table", "chart"]
        if raw_label in {"doc_title", "paragraph_title", "figure_title"}:
            label = "heading"
            paragraph_title = content
        elif raw_label == "table":
            label = "table"
        elif raw_label == "chart":
            label = "chart"
        else:
            label = "text"
        raw_bbox = item.get("block_bbox")
        bbox = (
            (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
            if isinstance(raw_bbox, list)
            and len(raw_bbox) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw_bbox)
            else None
        )
        blocks.append(
            ParsedBlock(
                label=label,
                content=content,
                page_number=page_number,
                order=order,
                paragraph_title=paragraph_title,
                bbox=bbox,
            )
        )
    return blocks or [ParsedBlock("text", markdown_text, page_number, 0)]


def _integer_order(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _validate_https_url(url: str, *, field_name: str, allow_query: bool = True) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (parsed.query and not allow_query)
    ):
        raise ValueError(f"{field_name} must be a credential-free HTTPS URL")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
