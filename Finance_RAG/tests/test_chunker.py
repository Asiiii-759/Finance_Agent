import json
from pathlib import Path
import unittest

from Finance_RAG.parser_chunk_search.chunker import RecursiveChineseBlockSplitter


class ChunkerTests(unittest.TestCase):
    def test_existing_json_chunk_coordinates_match_original_text(self):
        json_path = next(Path("Finance_RAG/Data/knowledge_base/Finance/raw_resolve").glob("*.json"))
        data = json.loads(json_path.read_text(encoding="utf-8"))
        full_text = "".join(block.get("block_content", "") for block in data.get("parsed_blocks", []))

        chunks = RecursiveChineseBlockSplitter(
            chunk_size=512,
            chunk_overlap=128,
            spm_model_path="",
        ).chunk(data)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            start = chunk["metadata"]["global_start"]
            end = chunk["metadata"]["global_end"]
            self.assertEqual(chunk["content"], full_text[start - 1:end])

    def test_oversized_html_table_row_is_split_without_coordinate_drift(self):
        long_cell = "X" * 1200
        table = f"<table><tr><td>header</td></tr><tr><td>{long_cell}</td></tr></table>"
        data = {
            "document_info": {"file_name": "synthetic"},
            "parsed_blocks": [{"block_label": "table&chart", "block_content": table}],
        }

        chunks = RecursiveChineseBlockSplitter(
            chunk_size=128,
            chunk_overlap=32,
            spm_model_path="",
        ).chunk(data)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            start = chunk["metadata"]["global_start"]
            end = chunk["metadata"]["global_end"]
            self.assertEqual(chunk["content"], table[start - 1:end])


if __name__ == "__main__":
    unittest.main()
