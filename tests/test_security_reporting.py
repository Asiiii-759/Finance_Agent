from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pdf_fixtures import MCPPDFParserFixture, write_stub_pdf

from mas_finance import PDFDocumentParser as PublicPDFDocumentParser
from mas_finance.documents import parse_pdf_document
from mas_finance.reporting import export_run_artifacts
from mas_finance.security import safe_child, safe_identifier, safe_upload_name


class SecurityBoundaryTests(unittest.TestCase):
    def test_pdf_document_parser_is_a_public_package_export(self) -> None:
        from mas_finance.documents import PDFDocumentParser

        self.assertIs(PublicPDFDocumentParser, PDFDocumentParser)

    def test_untrusted_names_cannot_escape_root(self) -> None:
        self.assertEqual(safe_upload_name("../../report.PDF"), "report.pdf")
        self.assertEqual(safe_upload_name("C:\\temp\\report.pdf"), "report.pdf")
        self.assertEqual(safe_identifier("../../run one"), "run-one")
        with self.assertRaises(ValueError):
            safe_upload_name("payload.exe")

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            safe_child(Path(tmp), "../escape.txt")

    def test_artifacts_stay_under_output_directory_and_do_not_collide(self) -> None:
        result = {"report": "report", "audit_events": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = export_run_artifacts(result, root, "../../outside")
            second = export_run_artifacts(result, root, "../../outside")
            self.assertNotEqual(first["state_path"], second["state_path"])
            for artifact in list(first.values()) + list(second.values()):
                path = Path(artifact).resolve()
                self.assertEqual(path.parent, root.resolve())
                self.assertTrue(path.exists())
            self.assertEqual(json.loads(Path(first["audit_path"]).read_text(encoding="utf-8")), [])

    def test_pdf_page_limit_fails_before_text_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.pdf"
            write_stub_pdf(path)
            parser = MCPPDFParserFixture({path.name: {1: "first", 2: "second"}})
            with self.assertRaisesRegex(ValueError, "page processing limit"):
                parse_pdf_document(path, max_pages=1, document_parser=parser)

    def test_pdf_file_size_limit_fails_before_parser_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.pdf"
            path.write_bytes(b"%PDF-" + b"x" * 32)
            with self.assertRaisesRegex(ValueError, "byte processing limit"):
                parse_pdf_document(path, max_file_bytes=16)

    def test_pdf_extracted_text_is_bounded_before_rag_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text-heavy.pdf"
            write_stub_pdf(path)
            parser = MCPPDFParserFixture({path.name: {1: "ACME " + ("risk " * 100)}})
            with self.assertRaisesRegex(ValueError, "extracted text exceeds"):
                parse_pdf_document(path, max_text_characters=32, document_parser=parser)

    def test_pdf_requires_managed_parser_and_accepts_mcp_parser(self) -> None:
        class LocalParser:
            parser_kind = "local"

            def extract_document(self, _file_path: Path) -> dict[int, str]:
                return {1: "local text"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.pdf"
            write_stub_pdf(path)

            with self.assertRaisesRegex(ValueError, "PaddleOCR or MCP"):
                parse_pdf_document(path, include_pages=True)
            with self.assertRaisesRegex(ValueError, "must be PaddleOCR or MCP"):
                parse_pdf_document(path, document_parser=LocalParser())  # type: ignore[arg-type]

            parser = MCPPDFParserFixture({path.name: {1: "ACME covenant headroom narrowed."}})
            parsed = parse_pdf_document(path, include_pages=True, document_parser=parser)
            self.assertEqual(parser.calls, [path])
            self.assertEqual(parsed["parsed_page_count"], 1)
            self.assertEqual(parsed["parser_kind"], "mcp")
            self.assertEqual(parsed["warnings"], [])
            self.assertEqual(parsed["pages"][0]["extraction_method"], "mcp")
            self.assertIn("covenant headroom", parsed["pages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
