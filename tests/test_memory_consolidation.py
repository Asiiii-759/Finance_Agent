from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from llm_fixtures import research_service
from test_system import build_test_config

from mas_finance.memory_consolidation import LongTermMemoryCandidate
from mas_finance.memory_store import PersonalMemoryKind


class StaticMemoryExtractor:
    def __init__(self, candidate: LongTermMemoryCandidate) -> None:
        self.candidate = candidate
        self.calls = 0

    def extract(self, events, existing_memories):
        self.calls += 1
        self.asserted_event_kinds = [event.kind.value for event in events]
        self.existing_counts = len(existing_memories)
        return (self.candidate,)


class MemoryConsolidationTests(unittest.TestCase):
    def test_inferred_memory_requires_two_distinct_completed_runs(self) -> None:
        candidate = LongTermMemoryCandidate(
            PersonalMemoryKind.PREFERENCE,
            "回答结构",
            "用户长期偏好先给结论，再解释风险。",
            "金融分析",
            "inferred",
            0.9,
            "add",
            ("结论优先",),
        )
        extractor = StaticMemoryExtractor(candidate)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(
                build_test_config(Path(directory)),
                long_term_memory_extractor=extractor,
            )
            try:
                service.analyze(
                    "什么是市盈率？", thread_id="memory-thread", run_id="memory-run-1", export_artifacts=False
                )
                self.assertEqual(service.list_personal_memories(), [])
                service.analyze(
                    "解释一下市净率。", thread_id="memory-thread", run_id="memory-run-2", export_artifacts=False
                )
                memories = service.list_personal_memories()
                self.assertEqual(len(memories), 1)
                self.assertEqual(memories[0]["title"], "回答结构")
                self.assertEqual(memories[0]["metadata"]["evidence_run_ids"], ["memory-run-1", "memory-run-2"])
                self.assertNotIn("tool_event", extractor.asserted_event_kinds)
            finally:
                service.close()

    def test_explicit_memory_promotes_once_and_user_managed_profile_is_separate(self) -> None:
        candidate = LongTermMemoryCandidate(
            PersonalMemoryKind.PREFERENCE,
            "回答语言",
            "用户长期偏好使用中文。",
            "全局",
            "explicit",
            0.98,
            "add",
            ("中文",),
        )
        extractor = StaticMemoryExtractor(candidate)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            profile = root / "USER_PROFILE.md"
            profile.write_text("# 长期指令\n所有报告先给结论。", encoding="utf-8")
            config = replace(build_test_config(root), user_profile_path=profile)
            service = research_service(config, long_term_memory_extractor=extractor)
            try:
                result = service.analyze(
                    "什么是市盈率？",
                    thread_id="profile-thread",
                    run_id="profile-run",
                    export_artifacts=False,
                )["result"]
                context = result["request"]["personal_context"]
                self.assertEqual(context[0]["kind"], "user_instructions")
                self.assertIn("所有报告先给结论", context[0]["content"])
                memories = service.list_personal_memories()
                self.assertEqual(memories[0]["metadata"]["write_policy"], "automatic_llm_consolidation")
            finally:
                service.close()
