from __future__ import annotations

import json
import unittest

import httpx

from mas_finance.agent import AgentContext, ChatTurn, RuntimePolicy
from mas_finance.contracts import Evidence, EvidenceBundle, SourceRef, SourceType
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.llm import DeepSeekChatClient, LLMSettings
from mas_finance.synthesis import EvidenceBoundLLMSynthesizer, llm_synthesis_harness_tool


def evidence_bundle() -> EvidenceBundle:
    source = SourceRef.create(
        source_type=SourceType.DOCUMENT,
        title="Annual report",
        locator="report.pdf#page=1",
        provider="test",
    )
    bundle = EvidenceBundle()
    bundle.add_evidence(
        Evidence.create(
            source=source,
            content="ACME reported resilient demand in the quarter.",
            entity="ACME",
        )
    )
    return bundle


def synthesize(synthesizer, message: str, bundle: EvidenceBundle, *, run_id: str = "run"):
    return synthesizer.synthesize(
        ChatTurn(message=message, run_id=run_id),
        RuntimePolicy(),
        AgentContext(),
        bundle,
    )


class GoodLLM:
    backend_name = "test"

    def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1400):
        payload = json.loads(user_prompt)
        item = payload["evidence"][0]
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "ACME described demand as resilient.",
                        "evidence_ids": [item["evidence_id"]],
                        "evidence_quote": "resilient demand",
                    }
                ]
            }
        )


class BadLLM:
    backend_name = "bad"

    def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1400):
        return "not JSON"


class CapturingLLM(GoodLLM):
    def __init__(self) -> None:
        self.max_tokens = 0
        self.system_prompt = ""

    def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1400):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        return super().chat(system_prompt, user_prompt, temperature, max_tokens)


class SynthesisTests(unittest.TestCase):
    def test_synthesis_budget_supports_batched_financial_results(self) -> None:
        client = CapturingLLM()
        synthesizer = EvidenceBoundLLMSynthesizer(client, count_tokens=len)
        self.assertTrue(synthesize(synthesizer, "demand", evidence_bundle()))
        self.assertEqual(client.max_tokens, 4096)
        self.assertIn("中文金融研究撰稿人", client.system_prompt)

    def test_web_evidence_is_assembled_and_synthesized_without_crashing(self) -> None:
        source = SourceRef.create(
            source_type=SourceType.WEB,
            title="ACME update",
            locator="https://example.com/acme",
            provider="fixture-search",
            metadata={
                "domain": "example.com",
                "quality_tier": "open_web",
                "rank": 1,
                "content_basis": "search_result_snippet",
            },
        )
        bundle = EvidenceBundle()
        bundle.add_evidence(
            Evidence.create(source=source, content="ACME reported resilient demand in the quarter.")
        )
        synthesizer = EvidenceBoundLLMSynthesizer(GoodLLM(), count_tokens=len)
        claims = synthesize(synthesizer, "ACME demand", bundle)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status.value, "inferred")
        self.assertIn("search snippets", claims[0].caveat)
        manifest = synthesizer.context_manifest()
        self.assertEqual(manifest["source_type_counts"]["web"], 1)

    def test_deepseek_v4_client_disables_thinking_and_validates_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(request.url, httpx.URL("https://api.deepseek.com/v1/chat/completions"))
            self.assertEqual(payload["model"], "deepseek-v4-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            return httpx.Response(200, json={"choices": [{"message": {"content": " OK "}}]})

        settings = LLMSettings("secret", "https://api.deepseek.com/v1", "deepseek-v4-flash", 10)
        self.assertNotIn("secret", repr(settings))
        client = DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))
        self.assertEqual(client.chat("system", "user", temperature=0, max_tokens=8), "OK")

        with self.assertRaisesRegex(ValueError, "HTTPS"):
            LLMSettings("secret", "http://api.deepseek.com", "deepseek-v4-flash", 10)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            client.chat("system", "user", max_tokens=True)

        empty = DeepSeekChatClient(
            settings,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": ""}}]},
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "no final content"):
            empty.chat("system", "user", max_tokens=8)

        redirect = DeepSeekChatClient(
            settings,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    302,
                    headers={"location": "https://other.example.test"},
                )
            ),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            redirect.chat("system", "user", max_tokens=8)

    def test_deepseek_transient_failures_are_retryable_but_client_errors_are_not(self) -> None:
        settings = LLMSettings("secret", "https://api.deepseek.com/v1", "deepseek-v4-flash", 10)
        attempts = 0

        def transient_then_success(_request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(500, json={"error": "temporary"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        harness = ToolHarness()
        client = DeepSeekChatClient(settings, transport=httpx.MockTransport(transient_then_success))
        harness.register(llm_synthesis_harness_tool(client, network_access=True))
        result = harness.invoke(
            "llm.synthesize",
            {"system_prompt": "system", "user_prompt": "user", "max_tokens": 8},
            ToolContext(
                run_id="retry-model",
                thread_id="thread-model",
                policy=ExecutionPolicy(
                    allowed_capabilities=frozenset({"model.generate"}),
                    allow_network=True,
                ),
            ),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(attempts, 2)
        self.assertEqual(harness.budget_usage("retry-model").model_calls, 1)

        bad_request = DeepSeekChatClient(
            settings,
            transport=httpx.MockTransport(lambda _request: httpx.Response(400, json={"error": "bad request"})),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            bad_request.chat("system", "user", max_tokens=8)

        transport_failure = DeepSeekChatClient(
            settings,
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request))
            ),
        )
        with self.assertRaisesRegex(ConnectionError, "transport failed"):
            transport_failure.chat("system", "user", max_tokens=8)

    def test_parametric_claim_allowed_without_evidence(self) -> None:
        class ConceptLLM:
            backend_name = "concept"

            def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1400):
                del system_prompt, user_prompt, temperature, max_tokens
                return json.dumps(
                    {
                        "claims": [
                            {
                                "text": "市盈率等于股价除以每股收益，高低本身不是买卖信号。",
                                "evidence_ids": [],
                                "evidence_quote": "",
                            }
                        ]
                    }
                )

        synthesizer = EvidenceBoundLLMSynthesizer(ConceptLLM(), count_tokens=len)
        claims = synthesize(synthesizer, "什么是市盈率？", EvidenceBundle())
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status.value, "inferred")
        self.assertEqual(claims[0].evidence_ids, ())
        self.assertIn("未经检索核验", claims[0].caveat)

    def test_cited_claim_without_literal_quote_is_rejected(self) -> None:
        class FakeCiteLLM:
            backend_name = "fake-cite"

            def chat(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1400):
                payload = json.loads(user_prompt)
                item = payload["evidence"][0]
                return json.dumps(
                    {
                        "claims": [
                            {
                                "text": "这段话没有出现在证据里。",
                                "evidence_ids": [item["evidence_id"]],
                                "evidence_quote": "this quote is not in the source",
                            }
                        ]
                    }
                )

        synthesizer = EvidenceBoundLLMSynthesizer(FakeCiteLLM(), count_tokens=len)
        with self.assertRaisesRegex(RuntimeError, "LLM synthesis was unusable"):
            synthesize(synthesizer, "demand", evidence_bundle())
        synthesizer = EvidenceBoundLLMSynthesizer(GoodLLM(), count_tokens=len)
        claims = synthesize(synthesizer, "demand", evidence_bundle())
        self.assertEqual(claims[0].text, "ACME described demand as resilient.")
        self.assertEqual(synthesizer.diagnostics(), ())

    def test_invalid_model_output_fails_fast(self) -> None:
        synthesizer = EvidenceBoundLLMSynthesizer(BadLLM(), count_tokens=len)
        with self.assertRaisesRegex(RuntimeError, "LLM synthesis was unusable"):
            synthesize(synthesizer, "demand", evidence_bundle())

    def test_llm_call_uses_harness_and_omits_prompts_from_audit(self) -> None:
        harness = ToolHarness()
        harness.register(llm_synthesis_harness_tool(GoodLLM(), network_access=False))
        synthesizer = EvidenceBoundLLMSynthesizer(GoodLLM(), harness=harness, count_tokens=len)
        claims = synthesize(synthesizer, "demand", evidence_bundle(), run_id="llm-audit")
        self.assertTrue(claims)
        event = harness.audit_events("llm-audit")[0]
        self.assertEqual(event["tool_name"], "llm.synthesize")
        self.assertEqual(event["arguments"]["user_prompt"], "***CONTENT_OMITTED***")


if __name__ == "__main__":
    unittest.main()
