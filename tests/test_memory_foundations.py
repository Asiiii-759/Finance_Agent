from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_fixtures import ScriptedLLM, research_service
from test_system import build_test_config

from mas_finance.agent import AgentContext, ChatAttachment, ChatTurn, ResearchState, RuntimePolicy
from mas_finance.atomic_facts import LLMAtomicFactExtractor
from mas_finance.harness import ToolHarness
from mas_finance.memory_store import ConversationEvent, ConversationEventKind
from mas_finance.planning import ModelPlanner
from mas_finance.skill_learning import LearnedSkill, LLMSkillExtractor, skill_run_context
from mas_finance.task_frame import LLMTaskInterpreter


class FailingAtomicFactExtractor:
    def extract(self, events):
        del events
        raise ValueError("invalid atomic fact response")


class MemoryFoundationTests(unittest.TestCase):
    def test_task_frame_receives_the_complete_personal_context(self) -> None:
        personal_context = tuple(
            {
                "memory_id": f"memory-{index}",
                "kind": "experience",
                "title": f"长期经验 {index}",
                "content": f"稳定经验内容 {index}",
                "tags": [],
            }
            for index in range(10)
        )
        turn = ChatTurn(message="解释市盈率")
        agent_context = AgentContext(
            personal_context=personal_context,
        )
        context = LLMTaskInterpreter._context(turn, agent_context, {})
        self.assertEqual(context["personal_context"], list(personal_context))

    def test_task_frame_projects_relevant_personal_context_within_total_budget(self) -> None:
        personal_context = tuple(
            {
                "memory_id": f"memory-{index}",
                "kind": "experience",
                "title": f"主题 {index}",
                "content": ("稳定经验" * 250) + ("Alpha" if index == 0 else ""),
                "tags": [],
            }
            for index in range(8)
        )
        interpreter = LLMTaskInterpreter(ToolHarness(), max_context_tokens=5_000, count_tokens=len)
        raw = interpreter._context(
            ChatTurn(message="继续分析 Alpha"),
            AgentContext(personal_context=personal_context),
            {},
        )
        bounded = interpreter._bounded_context(raw, "继续分析 Alpha")
        manifest = interpreter.context_manifest()
        assert manifest is not None
        self.assertLessEqual(manifest["total_context_tokens"], 5_000)
        self.assertGreater(manifest["omitted_personal_context_count"], 0)
        self.assertEqual(bounded["personal_context"][0]["memory_id"], "memory-0")

    def test_task_frame_sees_only_current_attachments_not_a_document_catalog(self) -> None:
        turn = ChatTurn(
            message="比较我刚上传的两份 PDF",
            attachments=(
                ChatAttachment("doc-a", "alpha.pdf"),
                ChatAttachment("doc-b", "beta.pdf"),
            ),
        )
        context = LLMTaskInterpreter._context(turn, AgentContext(), {})

        self.assertEqual(
            [item["title"] for item in context["current_request"]["attachments"]],
            ["alpha.pdf", "beta.pdf"],
        )
        self.assertNotIn("available_documents", context["current_request"])

    def test_checkpoint_round_trip_keeps_current_attachments_outside_agent_context(self) -> None:
        state = ResearchState(
            turn=ChatTurn(
                message="分析附件",
                attachments=(ChatAttachment("doc-a", "alpha.pdf"),),
            ),
            runtime_policy=RuntimePolicy(),
        )

        payload = state.to_dict()
        restored = ResearchState.from_dict(payload)

        self.assertEqual(payload["schema_version"], 6)
        self.assertEqual(restored.turn.attachments[0].title, "alpha.pdf")
        self.assertNotIn("document_index", payload["context"])

    def test_skill_context_uses_successful_audit_capabilities_not_planner_categories(self) -> None:
        context = skill_run_context(
            {
                "status": "succeeded",
                "task_frame": {"goal": "核验资料", "entities": []},
                "coverage": {"complete": True},
                "observations": [
                    {
                        "task": {"tool_name": "source.one", "category": "document", "reason": "先核验"},
                        "result": {"call_id": "call-1", "ok": True},
                    },
                    {
                        "task": {"tool_name": "source.two", "category": "document", "reason": "再复核"},
                        "result": {"call_id": "call-2", "ok": True},
                    },
                ],
                "audit_events": [
                    {"call_id": "call-1", "capability": "document.search"},
                    {"call_id": "call-2", "capability": "web.search"},
                ],
            }
        )
        assert context is not None
        self.assertEqual(context["successful_capabilities"], ["document.search", "web.search"])

    def test_skill_learning_rejects_run_specific_details_and_unobserved_capabilities(self) -> None:
        context = {
            "successful_capabilities": ["web.search"],
            "entities": [{"name": "Apple", "symbol": "AAPL"}],
        }
        extractor = LLMSkillExtractor(
            ScriptedLLM(
                [
                    {
                        "skill": {
                            "name": "Apple 核验",
                            "description": "访问 https://example.com 核验。",
                            "applicability": "公司研究",
                            "steps": ["查询 Apple", "计算 2026 年指标"],
                            "required_capabilities": ["regulatory.read"],
                        }
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "capability not observed"):
            extractor.extract(context)

        entity_extractor = LLMSkillExtractor(
            ScriptedLLM(
                [
                    {
                        "skill": {
                            "name": "Apple 核验",
                            "description": "核验指定公司。",
                            "applicability": "公司研究",
                            "steps": ["查询一手来源", "交叉核验来源"],
                            "required_capabilities": ["web.search"],
                        }
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "entity-specific"):
            entity_extractor.extract(context)

    def test_atomic_fact_learning_failure_is_logged_without_failing_completed_answer(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(
                build_test_config(Path(directory)),
                atomic_fact_extractor=FailingAtomicFactExtractor(),
            )
            response = service.analyze(
                "什么是市盈率？",
                thread_id="atomic-failure-thread",
                run_id="atomic-failure-run",
                export_artifacts=False,
            )
            self.assertEqual(response["result"]["status"], "succeeded")
            logs = service.list_run_logs("atomic-failure-thread", "atomic-failure-run")
            event_types = [item["event_type"] for item in logs]
            self.assertIn("memory.atomic_facts_failed", event_types)
            self.assertNotIn("memory.atomic_facts_completed", event_types)
            completed = next(item for item in logs if item["event_type"] == "run.completed")
            self.assertEqual(completed["details"]["claim_count"], 1)
            self.assertEqual(completed["details"]["source_count"], 0)
            self.assertEqual(completed["details"]["budget"]["model_calls"], 3)
            service.close()

    def test_run_log_storage_redacts_secrets_and_omits_raw_results(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(build_test_config(Path(directory)))
            namespace = service._conversation_namespace("local", "owner", "log-redaction-thread")
            service.memory_store.append_run_log(
                namespace,
                run_id="log-redaction-run",
                event_type="tool.completed",
                level="info",
                message="工具完成。",
                details={
                    "api_key": "secret-value",
                    "nested": {"access_token": "token-value"},
                    "result": {"raw": "large provider response"},
                    "result_summary": {"type": "object", "evidence_count": 2},
                },
            )
            details = service.list_run_logs("log-redaction-thread", "log-redaction-run")[0]["details"]
            self.assertEqual(details["api_key"], "***REDACTED***")
            self.assertEqual(details["nested"]["access_token"], "***REDACTED***")
            self.assertEqual(details["result"], "***CONTENT_OMITTED***")
            self.assertEqual(details["result_summary"]["evidence_count"], 2)
            service.close()

    def test_atomic_fact_extractor_preserves_sentence_and_validates_only_source_boundary(self) -> None:
        event = ConversationEvent(
            event_id="event-user",
            sequence=1,
            kind=ConversationEventKind.USER_MESSAGE,
            content="比较 Apple 和 Microsoft 的最大回撤。",
            occurred_at="2026-08-27T12:00:00+08:00",
            run_id="run-1",
            entities=("Apple", "Microsoft"),
        )
        extractor = LLMAtomicFactExtractor(
            ScriptedLLM(
                [
                    {
                        "facts": [
                            {
                                "text": "用户要求比较 Apple 与 Microsoft 的最大回撤。",
                                "source_event_ids": ["event-user"],
                                "entities": ["Apple", "Microsoft"],
                            }
                        ]
                    }
                ]
            )
        )
        facts = extractor.extract((event,))
        self.assertEqual(facts[0].text, "用户要求比较 Apple 与 Microsoft 的最大回撤。")
        self.assertEqual(facts[0].source_event_ids, ("event-user",))

    def test_atomic_fact_extractor_never_discloses_tool_or_assistant_content(self) -> None:
        events = (
            ConversationEvent(
                event_id="event-user",
                sequence=1,
                kind=ConversationEventKind.USER_MESSAGE,
                content="它和 Microsoft 相比怎么样？",
                occurred_at="2026-08-27T12:00:00+08:00",
                run_id="run-1",
            ),
            ConversationEvent(
                event_id="event-tool",
                sequence=2,
                kind=ConversationEventKind.TOOL_EVENT,
                content="internal provider payload must not enter atomic extraction",
                occurred_at="2026-08-27T12:00:01+08:00",
                run_id="run-1",
                entities=("Apple",),
            ),
            ConversationEvent(
                event_id="event-assistant",
                sequence=3,
                kind=ConversationEventKind.ASSISTANT_MESSAGE,
                content="internal assistant conclusion must not enter atomic extraction",
                occurred_at="2026-08-27T12:00:02+08:00",
                run_id="run-1",
                entities=("Apple", "Microsoft"),
            ),
        )
        client = ScriptedLLM(
            [
                {
                    "facts": [
                        {
                            "text": "用户要求比较 Apple 与 Microsoft。",
                            "source_event_ids": ["event-user"],
                            "entities": ["Apple", "Microsoft"],
                        }
                    ]
                }
            ]
        )
        facts = LLMAtomicFactExtractor(client).extract(events)
        prompt = client.user_prompts[0]
        self.assertNotIn("internal provider payload", prompt)
        self.assertNotIn("internal assistant conclusion", prompt)
        self.assertIn('"resolved_entity_candidates": ["Apple", "Microsoft"]', prompt)
        self.assertEqual(facts[0].source_event_ids, ("event-user",))

    def test_atomic_fact_extractor_rejects_more_than_six_facts_for_one_run(self) -> None:
        event = ConversationEvent(
            event_id="event-user",
            sequence=1,
            kind=ConversationEventKind.USER_MESSAGE,
            content="分析 Apple。",
            occurred_at="2026-08-27T12:00:00+08:00",
            run_id="run-1",
            entities=("Apple",),
        )
        extractor = LLMAtomicFactExtractor(
            ScriptedLLM(
                [
                    {
                        "facts": [
                            {
                                "text": f"最小事实 {index}。",
                                "source_event_ids": ["event-user"],
                                "entities": ["Apple"],
                            }
                            for index in range(7)
                        ]
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "at most six"):
            extractor.extract((event,))

    def test_skill_index_discloses_full_workflow_only_after_selection(self) -> None:
        skills = [
            {
                "skill_id": "skill-selected",
                **LearnedSkill(
                    "跨来源核验",
                    "交叉核验公开信息。",
                    "需要验证多个来源时",
                    ("先查一手来源", "再查独立来源", "核对时点"),
                    ("web.search",),
                ).to_dict(),
            },
            {
                "skill_id": "skill-hidden",
                **LearnedSkill(
                    "不相关流程",
                    "不应披露。",
                    "其他任务",
                    ("步骤甲", "步骤乙"),
                    (),
                ).to_dict(),
            },
        ]
        planner = ModelPlanner(ToolHarness(), count_tokens=len, learned_skills=skills)
        state = ResearchState(
            turn=ChatTurn(message="核验一项公告"),
            runtime_policy=RuntimePolicy(),
            task_frame={"selected_skill_ids": ["skill-selected"]},
        )
        context = planner._planning_context(state, {})
        self.assertEqual([item["skill_id"] for item in context["selected_skills"]], ["skill-selected"])
        self.assertEqual(context["selected_skills"][0]["steps"][0], "先查一手来源")

    def test_learned_skill_is_persisted_versioned_and_deletable_outside_personal_memory(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(build_test_config(Path(directory)))
            skill = LearnedSkill(
                "跨来源核验",
                "交叉核验公开信息。",
                "需要验证多个来源时",
                ("先查一手来源", "再查独立来源"),
                ("web.search",),
            )
            service._save_learned_skill("local", "owner", "run-1", skill)
            service._save_learned_skill("local", "owner", "run-2", skill)
            stored = service.list_learned_skills()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["success_count"], 2)
            self.assertEqual(service.list_personal_memories(), [])
            self.assertTrue(service.delete_learned_skill(stored[0]["skill_id"]))
            self.assertEqual(service.list_learned_skills(), [])
            service.close()

    def test_failed_agent_run_keeps_queryable_persistent_logs(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            service = research_service(
                build_test_config(Path(directory)),
                llm_client=ScriptedLLM([{}]),
            )
            with self.assertRaises(ValueError):
                service.analyze(
                    "测试失败日志",
                    thread_id="failed-log-thread",
                    run_id="failed-log-run",
                    export_artifacts=False,
                )
            events = service.list_run_logs("failed-log-thread", "failed-log-run")
            event_types = [item["event_type"] for item in events]
            self.assertEqual(event_types[0], "run.started")
            self.assertIn("tool.completed", event_types)
            self.assertEqual(event_types[-1], "run.failed")
            failure = events[-1]
            self.assertEqual(failure["details"]["phase"], "agent_execution")
            service.close()


if __name__ == "__main__":
    unittest.main()
