from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from pdf_fixtures import write_stub_pdf

from mas_finance import PaddleOCRClient as PublicPaddleOCRClient
from mas_finance.ocr import PaddleOCRClient


class PaddleOCRClientTests(unittest.TestCase):
    def test_client_is_a_public_package_export(self) -> None:
        self.assertIs(PublicPaddleOCRClient, PaddleOCRClient)

    def test_bounded_job_flow_returns_page_markdown_without_leaking_token(self) -> None:
        status_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.method == "POST":
                self.assertEqual(
                    request.url,
                    httpx.URL("https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"),
                )
                self.assertEqual(request.headers["authorization"], "bearer secret")
                self.assertIn(b"PaddleOCR-VL-1.6", request.content)
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            if request.url.host == "paddleocr.aistudio-app.com":
                status_calls += 1
                self.assertEqual(request.headers["authorization"], "bearer secret")
                if status_calls == 1:
                    return httpx.Response(200, json={"data": {"state": "pending"}})
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://results.example.test/job-1.jsonl?sig=x"},
                        }
                    },
                )
            self.assertEqual(request.url.host, "results.example.test")
            self.assertNotIn("authorization", request.headers)
            body = {"result": {"layoutParsingResults": [{"markdown": {"text": "# ACME\nCovenant headroom narrowed."}}]}}
            return httpx.Response(200, content=(json.dumps(body) + "\n").encode())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            write_stub_pdf(path)
            client = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                max_poll_requests=3,
                transport=httpx.MockTransport(handler),
            )
            self.assertNotIn("secret", repr(client))
            document = client.extract_document(path)
            self.assertEqual(document.pages, {1: "# ACME\nCovenant headroom narrowed."})
            self.assertEqual(document.blocks[0].content, "# ACME\nCovenant headroom narrowed.")
            self.assertEqual(document.parser_version, "PaddleOCR-VL-1.6")

    def test_polling_limit_fails_closed(self) -> None:
        def pending(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            return httpx.Response(200, json={"data": {"state": "pending"}})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            write_stub_pdf(path)
            client = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                max_poll_requests=1,
                transport=httpx.MockTransport(pending),
            )
            with self.assertRaisesRegex(TimeoutError, "polling limit"):
                client.extract_document(path)

    def test_transient_poll_status_is_retried_within_poll_budget(self) -> None:
        status_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.method == "POST":
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            if request.url.host == "paddleocr.aistudio-app.com":
                status_calls += 1
                if status_calls == 1:
                    return httpx.Response(503)
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://results.example.test/job.jsonl"},
                        }
                    },
                )
            body = {"result": {"layoutParsingResults": [{"markdown": {"text": "Recovered."}}]}}
            return httpx.Response(200, content=(json.dumps(body) + "\n").encode())

        with tempfile.TemporaryDirectory() as directory:
            path = write_stub_pdf(Path(directory) / "scan.pdf")
            document = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                max_poll_requests=2,
                transport=httpx.MockTransport(handler),
            ).extract_document(path)
        self.assertEqual(document.pages, {1: "Recovered."})
        self.assertEqual(status_calls, 2)

    def test_poll_transport_failure_is_retried_within_poll_budget(self) -> None:
        status_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.method == "POST":
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            if request.url.host == "paddleocr.aistudio-app.com":
                status_calls += 1
                if status_calls == 1:
                    raise httpx.ConnectError("temporary", request=request)
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://results.example.test/job.jsonl"},
                        }
                    },
                )
            body = {"result": {"layoutParsingResults": [{"markdown": {"text": "Recovered."}}]}}
            return httpx.Response(200, content=(json.dumps(body) + "\n").encode())

        with tempfile.TemporaryDirectory() as directory:
            path = write_stub_pdf(Path(directory) / "scan.pdf")
            document = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                max_poll_requests=2,
                transport=httpx.MockTransport(handler),
            ).extract_document(path)
        self.assertEqual(document.pages, {1: "Recovered."})
        self.assertEqual(status_calls, 2)

    def test_transient_submission_is_not_retried_without_idempotency(self) -> None:
        submissions = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal submissions
            submissions += 1
            return httpx.Response(503)

        with tempfile.TemporaryDirectory() as directory:
            path = write_stub_pdf(Path(directory) / "scan.pdf")
            client = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            )
            with self.assertRaisesRegex(ConnectionError, "not idempotent"):
                client.extract_document(path)
        self.assertEqual(submissions, 1)

    def test_submission_transport_failure_is_not_retried_without_idempotency(self) -> None:
        submissions = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal submissions
            submissions += 1
            raise httpx.ConnectError("temporary", request=request)

        with tempfile.TemporaryDirectory() as directory:
            path = write_stub_pdf(Path(directory) / "scan.pdf")
            client = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            )
            with self.assertRaisesRegex(ConnectionError, "not idempotent"):
                client.extract_document(path)
        self.assertEqual(submissions, 1)

    def test_transient_result_download_is_retried_once(self) -> None:
        result_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal result_calls
            if request.method == "POST":
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            if request.url.host == "paddleocr.aistudio-app.com":
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://results.example.test/job.jsonl"},
                        }
                    },
                )
            result_calls += 1
            if result_calls == 1:
                return httpx.Response(503)
            body = {"result": {"layoutParsingResults": [{"markdown": {"text": "Recovered."}}]}}
            return httpx.Response(200, content=(json.dumps(body) + "\n").encode())

        with tempfile.TemporaryDirectory() as directory:
            path = write_stub_pdf(Path(directory) / "scan.pdf")
            document = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            ).extract_document(path)
        self.assertEqual(document.pages, {1: "Recovered."})
        self.assertEqual(result_calls, 2)

    def test_structured_layout_preserves_heading_table_page_and_bbox(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"data": {"jobId": "job-1"}})
            if request.url.host == "paddleocr.aistudio-app.com":
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://results.example.test/job.jsonl"},
                        }
                    },
                )
            layout = {
                "markdown": {"text": "## Liquidity\n\n| Period | Cash |\n|---|---|\n| Q1 | 10 |"},
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "paragraph_title",
                            "block_content": "Liquidity",
                            "block_order": 1,
                            "block_bbox": [1, 2, 30, 12],
                        },
                        {
                            "block_label": "table",
                            "block_content": "| Period | Cash |\n|---|---|\n| Q1 | 10 |",
                            "block_order": 2,
                            "block_bbox": [1, 14, 80, 50],
                        },
                    ]
                },
            }
            return httpx.Response(
                200,
                content=(json.dumps({"result": {"layoutParsingResults": [layout]}}) + "\n").encode(),
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured.pdf"
            write_stub_pdf(path)
            document = PaddleOCRClient(
                "secret",
                poll_interval_seconds=0,
                max_poll_requests=1,
                transport=httpx.MockTransport(handler),
            ).extract_document(path)

        self.assertEqual([block.label for block in document.blocks], ["heading", "table"])
        self.assertEqual(document.blocks[1].paragraph_title, "Liquidity")
        self.assertEqual(document.blocks[1].page_number, 1)
        self.assertEqual(document.blocks[1].bbox, (1.0, 14.0, 80.0, 50.0))


if __name__ == "__main__":
    unittest.main()
