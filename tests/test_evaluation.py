from __future__ import annotations

import unittest

from llm_fixtures import FixtureResearchLLM

from mas_finance.evaluation import ENTERPRISE_CASES, run_enterprise_evaluation


class EnterpriseEvaluationTests(unittest.TestCase):
    def test_black_box_acceptance_matrix(self) -> None:
        report = run_enterprise_evaluation(llm_client=FixtureResearchLLM())
        self.assertEqual(report["case_count"], len(ENTERPRISE_CASES) + 2)
        self.assertTrue(report["passed"], report["results"])


if __name__ == "__main__":
    unittest.main()
