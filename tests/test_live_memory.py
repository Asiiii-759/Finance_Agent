from __future__ import annotations

import json
import os
import unittest

from mas_finance.atomic_facts import LLMAtomicFactExtractor
from mas_finance.config import AppConfig
from mas_finance.conversation import LLMConversationSummarizer
from mas_finance.llm import DeepSeekChatClient
from mas_finance.memory_store import ConversationEvent, ConversationEventKind


@unittest.skipUnless(
    os.getenv("MAS_RUN_LIVE_MEMORY_TESTS") == "1",
    "set MAS_RUN_LIVE_MEMORY_TESTS=1 to spend DeepSeek quota on live memory tests",
)
class LiveMemoryTests(unittest.TestCase):
    def test_real_model_extracts_user_only_facts_and_preserves_correction_in_summary(self) -> None:
        client = DeepSeekChatClient(AppConfig.from_env().llm)
        events = (
            ConversationEvent(
                event_id="event-user-1",
                sequence=1,
                kind=ConversationEventKind.USER_MESSAGE,
                content="请比较苹果公司和微软公司的五年最大回撤，并且不要联网搜索新闻。",
                occurred_at="2026-09-03T09:00:00+08:00",
                run_id="run-1",
            ),
            ConversationEvent(
                event_id="event-tool-1",
                sequence=2,
                kind=ConversationEventKind.TOOL_EVENT,
                content="INTERNAL_TOOL_PAYLOAD_MARKER: provider returned hidden values.",
                occurred_at="2026-09-03T09:00:01+08:00",
                run_id="run-1",
                entities=("苹果公司", "微软公司"),
                payload={"tool_name": "market.history", "result_status": "success", "attempts": 1},
            ),
            ConversationEvent(
                event_id="event-assistant-1",
                sequence=3,
                kind=ConversationEventKind.ASSISTANT_MESSAGE,
                content="INTERNAL_ASSISTANT_MARKER: final calculations were prepared.",
                occurred_at="2026-09-03T09:00:02+08:00",
                run_id="run-1",
                entities=("苹果公司", "微软公司"),
                payload={"status": "succeeded"},
            ),
        )
        facts = LLMAtomicFactExtractor(client).extract(events)
        self.assertGreaterEqual(len(facts), 1)
        self.assertLessEqual(len(facts), 6)
        self.assertTrue(all(fact.source_event_ids == ("event-user-1",) for fact in facts))
        fact_text = " ".join(fact.text for fact in facts)
        self.assertIn("苹果", fact_text)
        self.assertIn("微软", fact_text)
        self.assertNotIn("INTERNAL_TOOL_PAYLOAD_MARKER", fact_text)
        self.assertNotIn("INTERNAL_ASSISTANT_MARKER", fact_text)

        correction = ConversationEvent(
            event_id="event-user-2",
            sequence=4,
            kind=ConversationEventKind.USER_MESSAGE,
            content="我纠正一下：改为比较五年波动率，暂时不需要最大回撤。",
            occurred_at="2026-09-03T09:01:00+08:00",
            run_id="run-2",
        )
        summary = LLMConversationSummarizer(client).summarize(
            {
                "conversation_summary": "",
                "user_goals": [],
                "requirements": [],
                "decisions": [],
                "completed_work": [],
                "successful_tools": [],
                "failed_tools": [],
                "unfinished_work": [],
                "open_questions": [],
            },
            (*events, correction),
        )
        self.assertEqual(
            set(summary),
            {
                "conversation_summary",
                "user_goals",
                "requirements",
                "decisions",
                "completed_work",
                "successful_tools",
                "failed_tools",
                "unfinished_work",
                "open_questions",
            },
        )
        summary_text = json.dumps(summary, ensure_ascii=False)
        self.assertIn("波动率", summary_text)
        self.assertIn("不需要最大回撤", summary_text)


if __name__ == "__main__":
    unittest.main()
