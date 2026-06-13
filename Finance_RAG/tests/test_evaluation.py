import unittest

from Finance_RAG.evaluation import evaluate_span_recall, merge_spans


class EvaluationTests(unittest.TestCase):
    def test_merge_spans(self):
        spans = [("a", 1, 10), ("a", 5, 20), ("b", 1, 2)]
        self.assertEqual(merge_spans(spans), [("a", 1, 20), ("b", 1, 2)])

    def test_evaluate_span_recall(self):
        result = evaluate_span_recall(
            gold_spans=[("a", 10, 20), ("a", 30, 40)],
            retrieved_spans=[("a", 15, 35)],
        )
        self.assertGreater(result["coverage"], 0)
        self.assertGreater(result["iou"], 0)
        self.assertTrue(result["hit"])


if __name__ == "__main__":
    unittest.main()
