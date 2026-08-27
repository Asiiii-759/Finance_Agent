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
            self.assertEqual(
                client.extract_document(path),
                {1: "# ACME\nCovenant headroom narrowed."},
            )

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


if __name__ == "__main__":
    unittest.main()
