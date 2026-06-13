import unittest

from Finance_RAG.rag import retrieve_documents, update_docs


class RagSmokeTests(unittest.TestCase):
    def test_mock_embedding_faiss_ingest_and_retrieve(self):
        file_name = "1Q26汽车电子收入预计同比翻倍，海外IDM布局渐完整.pdf"
        exp_name = "test_mock_embedding_faiss"

        result = update_docs(
            [file_name],
            exp_name=exp_name,
            embed_model="mock",
            chunk_size=512,
            chunk_overlap=128,
        )
        self.assertEqual(result["failed_files"], {})
        self.assertIn(file_name, result["success_files"])

        docs = retrieve_documents(
            "汽车电子业务增长情况如何？",
            exp_name=exp_name,
            embed_model="mock",
            top_k=3,
        )
        self.assertGreater(len(docs), 0)
        self.assertIn("metadata", docs[0])
        self.assertIn("global_start", docs[0]["metadata"])
        self.assertIn("global_end", docs[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
