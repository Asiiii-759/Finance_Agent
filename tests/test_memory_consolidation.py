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
    def test_temporary_candidate_must_be_ignore_and_update_must_be_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "临时偏好候选必须标记为 ignore"):
            LongTermMemoryCandidate.from_dict(
                {
                    "kind": "preference",
                    "title": "回答语言",
                    "content": "用户本轮要求使用日文。",
                    "scope": "本轮",
                    "explicitness": "explicit",
                    "confidence": 0.99,
                    "operation": "add",
                    "tags": [],
                }
            )
        with self.assertRaisesRegex(ValueError, "只有用户明确陈述"):
            LongTermMemoryCandidate.from_dict(
                {
                    "kind": "preference",
                    "title": "回答语言",
                    "content": "用户长期偏好使用英文。",
                    "scope": "全局",
                    "explicitness": "inferred",
                    "confidence": 0.9,
                    "operation": "update",
                    "tags": [],
                }
            )

    def test_explicit_long_term_update_replaces_prior_preference_but_temporary_ignore_does_not(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(build_test_config(Path(directory)))
            original = service.save_personal_memory(
                kind=PersonalMemoryKind.PREFERENCE,
                title="回答语言",
                content="用户长期偏好使用中文。",
            )
            service._merge_long_term_memory_candidate(
                "default",
                "anonymous",
                "memory-update-thread",
                "memory-update-run",
                LongTermMemoryCandidate(
                    PersonalMemoryKind.PREFERENCE,
                    "回答语言",
                    "用户今后长期偏好使用英文。",
                    "全局",
                    "explicit",
                    0.99,
                    "update",
                    ("英文",),
                ),
                service.list_personal_memories(),
            )
            updated = service.list_personal_memories()[0]
            self.assertEqual(updated["memory_id"], original["memory_id"])
            self.assertEqual(updated["content"], "用户今后长期偏好使用英文。")
            self.assertTrue(updated["metadata"]["replaces_prior_memory"])

            service._consolidate_long_term_memory(
                "default",
                "anonymous",
                "temporary-thread",
                "temporary-run",
                (),
                StaticMemoryExtractor(
                    LongTermMemoryCandidate(
                        PersonalMemoryKind.PREFERENCE,
                        "回答语言",
                        "用户本轮临时要求使用日文。",
                        "本轮",
                        "explicit",
                        1.0,
                        "ignore",
                        ("日文",),
                    )
                ),
            )
            self.assertEqual(service.list_personal_memories()[0]["content"], "用户今后长期偏好使用英文。")
            service.close()

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
