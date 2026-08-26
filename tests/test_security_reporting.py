from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from mas_finance.documents import parse_pdf_document
from mas_finance.reporting import export_run_artifacts
from mas_finance.security import safe_child, safe_identifier, safe_upload_name


class SecurityBoundaryTests(unittest.TestCase):
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
            pdf = fitz.open()
            pdf.new_page()
            pdf.new_page()
            pdf.save(path)
            pdf.close()
            with self.assertRaisesRegex(ValueError, "page processing limit"):
                parse_pdf_document(path, max_pages=1)

    def test_pdf_file_size_limit_fails_before_parser_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.pdf"
            path.write_bytes(b"%PDF-" + b"x" * 32)
            with self.assertRaisesRegex(ValueError, "byte processing limit"):
                parse_pdf_document(path, max_file_bytes=16)

    def test_pdf_extracted_text_is_bounded_before_rag_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text-heavy.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "ACME " + ("risk " * 100))
            pdf.save(path)
            pdf.close()
            with self.assertRaisesRegex(ValueError, "extracted text exceeds"):
                parse_pdf_document(path, max_text_characters=32)

    def test_image_only_pdf_has_diagnostics_and_optional_ocr(self) -> None:
        class OCR:
            calls: list[tuple[Path, int]] = []

            def extract_document(self, file_path: Path, expected_pages: int) -> dict[int, str]:
                self.calls.append((file_path, expected_pages))
                return {1: "ACME covenant headroom narrowed."}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 0)
            pixmap.clear_with(255)
            page.insert_image(fitz.Rect(72, 72, 172, 172), pixmap=pixmap)
            pdf.save(path)
            pdf.close()

            without_ocr = parse_pdf_document(path, include_pages=True)
            self.assertEqual(without_ocr["text_page_count"], 0)
            self.assertEqual(without_ocr["warnings"][0]["code"], "ocr_required")

            ocr = OCR()
            parsed = parse_pdf_document(path, include_pages=True, ocr_provider=ocr)
            self.assertEqual(ocr.calls, [(path, 1)])
            self.assertEqual(parsed["ocr_page_count"], 1)
            self.assertEqual(parsed["warnings"], [])
            self.assertEqual(parsed["pages"][0]["extraction_method"], "ocr")
            self.assertIn("covenant headroom", parsed["pages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
