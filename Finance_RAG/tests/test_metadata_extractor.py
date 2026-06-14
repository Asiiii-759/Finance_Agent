import json
import tempfile
import unittest
from pathlib import Path

from Finance_RAG.parsers.resolved_json import ResolvedJsonParser
from Finance_RAG.schemas.metadata import FinanceMetadataExtractor


class MetadataExtractorTests(unittest.TestCase):
    def test_extracts_candidates_with_evidence_from_legacy_document(self):
        data = {
            "document_info": {
                "file_name": "中信证券：半导体行业深度报告2026-01-03.json",
                "doc_source": "中信证券",
            },
            "parsed_blocks": [
                {
                    "block_label": "text",
                    "block_content": "分析师：张三 证券代码：600000.SH 半导体设备景气度提升。",
                    "source_page": 0,
                }
            ],
        }

        report = FinanceMetadataExtractor().extract_from_legacy(data)

        self.assertEqual(report.document_date, "2026-01-03")
        self.assertEqual(report.report_type, "industry")
        self.assertEqual(report.organizations[0].name, "中信证券")
        self.assertTrue(any(candidate.name == "张三" for candidate in report.authors))
        self.assertTrue(any(candidate.name == "600000.SH" for candidate in report.tickers))
        self.assertTrue(any(candidate.name == "半导体" for candidate in report.industries))

    def test_resolved_json_parser_enriches_document_info(self):
        data = {
            "document_info": {"file_name": "汽车行业周报2026-02-04", "doc_source": "未知机构"},
            "parsed_blocks": [{"block_label": "text", "block_content": "汽车行业需求改善。"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sample.json"
            json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            parsed = ResolvedJsonParser(enrich_metadata=True).parse_json(json_path)

        info = parsed["document_info"]
        self.assertEqual(info["parser_name"], "resolved_json")
        self.assertIn("metadata_extraction", info)
        self.assertEqual(info["report_type"], "industry")

    def test_bad_doc_title_falls_back_to_filename(self):
        data = {
            "document_info": {
                "file_name": "电力设备行业周报：利用率94.8%",
                "doc_title": "未命名文档",
                "doc_source": "The image is too blurry to recognize any text content.",
            },
            "parsed_blocks": [{"block_label": "figure_title", "block_content": "图表4：涨跌top 5"}],
        }

        report = FinanceMetadataExtractor().extract_from_legacy(data)

        self.assertEqual(report.document_title, "电力设备行业周报：利用率94.8%")
        self.assertEqual(report.organizations, [])
        self.assertTrue(any("organization" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
