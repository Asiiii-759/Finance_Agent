from __future__ import annotations

import json
import unittest

import httpx
from llm_fixtures import NullPlanner, ScriptedLLM, agent_run_input, llm_backed_agent

from mas_finance.adequacy import LLMEvidenceAdequacyChecker, llm_evidence_adequacy_harness_tool
from mas_finance.agent import ChatTurn, CoverageAssessor
from mas_finance.contracts import EvidenceBundle
from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.metrics import financial_calculation_harness_tool
from mas_finance.planning import ModelPlanner, llm_planning_harness_tool
from mas_finance.research import FinancialIntent, ResearchRequirement, ResearchScope
from mas_finance.task_frame import LLMTaskInterpreter, llm_task_frame_harness_tool
from mas_finance.web_search import (
    BochaWebSearchClient,
    BraveWebSearchClient,
    WebSearchEvidenceAdapter,
    web_search_harness_tool,
)


class FixtureWebSearch:
    provider_name = "fixture-search"

    def search_json(self, payload):
        return {
            "query": payload["query"],
            "retrieved_at": "2026-08-12T10:00:00+00:00",
            "results": [
                {
                    "title": "ACME covenant update",
                    "url": "https://example.com/acme-covenant-update",
                    "description": "ACME reported that covenant headroom improved during the latest quarter.",
                    "extra_snippets": [],
                    "published_at": "2026-08-11T00:00:00+00:00",
                },
                {
                    "title": "Independent ACME analysis",
                    "url": "https://example.org/acme-covenant-analysis",
                    "description": "Independent reporting also described improved ACME covenant headroom.",
                    "extra_snippets": [],
                    "published_at": "2026-08-11T01:00:00+00:00",
                },
            ],
        }


class GraphAutonomyTests(unittest.TestCase):
    def test_graph_contains_only_business_nodes(self) -> None:
        graph = llm_backed_agent(ToolHarness(), planner=NullPlanner()).graph.get_graph()
        business_nodes = {name for name in graph.nodes if not name.startswith("__")}
        self.assertEqual(business_nodes, {"intent", "planning", "validation", "final_generation"})

    def test_model_selects_tool_from_runtime_catalog(self) -> None:
        harness = ToolHarness()
        llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "finance.calculate",
                    "arguments": {
                        "requests": [
                            {
                                "operation": "cagr",
                                "inputs": {"beginning_value": 100.0, "ending_value": 121.0, "years": 2.0},
                                "label": "cagr",
                            }
                        ]
                    },
                    "reason": "使用白名单公式计算 CAGR。",
                },
                {"action": "finish", "reason": "计算结果已有证据支持。"},
            ]
        )
        harness.register(financial_calculation_harness_tool())
        harness.register(
            llm_planning_harness_tool(
                llm,
                network_access=False,
            )
        )
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="一项投资从100增长到121，用了2年，CAGR是多少？",
                run_id="model-tool-choice",
                max_model_calls=4,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual([item.task.tool_name for item in outcome.state.observations], ["finance.calculate"])
        self.assertEqual(
            [item["tool_name"] for item in outcome.audit_events],
            ["llm.task_frame", "llm.plan", "finance.calculate", "llm.plan", "llm.synthesize"],
        )
        planning_manifest = next(
            item for item in outcome.state.context_manifests if item["phase"] == "planning"
        )
        self.assertIn("金融研究 Agent", llm.system_prompts[0])
        self.assertLessEqual(
            planning_manifest["evidence_tokens"],
            planning_manifest["max_evidence_tokens"],
        )

    def test_invalid_planner_output_gets_one_model_repair(self) -> None:
        harness = ToolHarness()
        llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "invented.calculate",
                    "arguments": {},
                    "reason": "错误地选择了不存在的工具。",
                },
                {
                    "action": "call_tool",
                    "tool_name": "finance.calculate",
                    "arguments": {
                        "requests": [
                            {
                                "operation": "cagr",
                                "inputs": {"beginning_value": 100.0, "ending_value": 121.0, "years": 2.0},
                                "label": "cagr",
                            }
                        ]
                    },
                    "reason": "根据目录改用确定性计算工具。",
                },
                {"action": "finish", "reason": "计算证据已经覆盖需求。"},
            ]
        )
        harness.register(financial_calculation_harness_tool())
        harness.register(llm_planning_harness_tool(llm, network_access=False))
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="一项投资从100增长到121，用了2年，CAGR是多少？",
                run_id="planner-output-repair",
                max_model_calls=6,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual([item.task.tool_name for item in outcome.state.observations], ["finance.calculate"])
        self.assertIn("planner_response_repair", json.loads(llm.user_prompts[1]))
        self.assertEqual(
            sum(item["tool_name"] == "llm.plan" for item in outcome.audit_events),
            3,
        )

    def test_model_changes_invalid_tool_arguments_after_structured_error(self) -> None:
        harness = ToolHarness()
        llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "finance.calculate",
                    "arguments": {
                        "requests": [
                            {
                                "operation": "cagr",
                                "inputs": {"beginning_value": 100.0, "ending_value": 121.0},
                                "label": "cagr",
                            }
                        ]
                    },
                    "reason": "先尝试计算 CAGR。",
                },
                {
                    "action": "call_tool",
                    "tool_name": "finance.calculate",
                    "arguments": {
                        "requests": [
                            {
                                "operation": "cagr",
                                "inputs": {"beginning_value": 100.0, "ending_value": 121.0, "years": 2.0},
                                "label": "cagr",
                            }
                        ]
                    },
                    "reason": "根据错误详情补充缺失的 years。",
                },
                {"action": "finish", "reason": "修正后的计算证据已经覆盖需求。"},
            ]
        )
        harness.register(financial_calculation_harness_tool())
        harness.register(llm_planning_harness_tool(llm, network_access=False))
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="一项投资从100增长到121，用了2年，CAGR是多少？",
                run_id="tool-argument-repair",
                max_model_calls=6,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(outcome.state.observations), 2)
        self.assertEqual(outcome.state.observations[0].result["error_code"], "invalid_tool_arguments")
        self.assertEqual(outcome.state.observations[0].result["error_details"]["missing_fields"], ["years"])
        self.assertTrue(outcome.state.observations[1].result["ok"])
        second_plan_context = json.loads(llm.user_prompts[1])
        self.assertEqual(second_plan_context["prior_actions"][0]["error_details"]["model_action"], "change_arguments")

    def test_task_frame_replaces_rule_scope_on_the_llm_path(self) -> None:
        harness = ToolHarness()
        llm = ScriptedLLM(
            [
                {
                    "goal": "解释市盈率的含义",
                    "entities": [],
                    "intents": ["financial_education"],
                    "requirements": [],
                    "success_criteria": ["给出市盈率的含义和注意事项"],
                    "clarification_question": None,
                },
                {"action": "finish", "reason": "概念题无需检索，直接结束收集。"},
            ]
        )
        harness.register(llm_task_frame_harness_tool(llm, network_access=False))
        harness.register(llm_planning_harness_tool(llm, network_access=False))
        outcome = llm_backed_agent(
            harness,
            planner=ModelPlanner(harness, count_tokens=len),
            task_interpreter=LLMTaskInterpreter(harness),
        ).run(
            *agent_run_input(
                query="解释市盈率",
                run_id="task-frame-scope",
                max_model_calls=4,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(outcome.state.scope.requirements, ())
        self.assertEqual(outcome.state.task_frame["goal"], "解释市盈率的含义")
        self.assertEqual(
            [item["tool_name"] for item in outcome.audit_events],
            ["llm.task_frame", "llm.plan", "llm.synthesize"],
        )

    def test_task_frame_stops_for_ambiguous_history_reference(self) -> None:
        harness = ToolHarness()
        harness.register(
            llm_task_frame_harness_tool(
                ScriptedLLM(
                    [
                        {
                            "goal": "查询历史中被指代公司的回撤",
                            "entities": [],
                            "success_criteria": [],
                            "requirements": [],
                            "clarification_question": "你指的是 Apple 还是 Microsoft？",
                        }
                    ]
                ),
                network_access=False,
            )
        )
        outcome = llm_backed_agent(
            harness,
            planner=NullPlanner(),
            task_interpreter=LLMTaskInterpreter(harness),
        ).run(
            *agent_run_input(
                query="它的回撤呢？",
                run_id="task-frame-clarification",
                thread_context={"atomic_facts": [{"entities": ["Apple", "Microsoft"]}]},
            )
        )
        self.assertEqual(outcome.status, "needs_clarification")
        self.assertEqual(outcome.state.stop_reason.value, "clarification_required")

    def test_validation_rejects_premature_model_finish(self) -> None:
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        harness.register(
            llm_planning_harness_tool(
                ScriptedLLM(
                    [
                        {"action": "finish", "reason": "I think this is enough."},
                        {
                            "action": "call_tool",
                            "tool_name": "finance.calculate",
                            "arguments": {
                                "requests": [
                                    {
                                        "operation": "cagr",
                                        "inputs": {"beginning_value": 100.0, "ending_value": 121.0, "years": 2.0},
                                        "label": "cagr",
                                    }
                                ]
                            },
                            "reason": "覆盖校验仍缺少计算结果。",
                        },
                        {"action": "finish", "reason": "计算结果已经覆盖任务需求。"},
                    ]
                ),
                network_access=False,
            )
        )
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="一项投资从100增长到121，用了2年，CAGR是多少？",
                run_id="premature-finish",
                max_iterations=3,
                max_model_calls=5,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(outcome.state.iteration, 3)
        self.assertEqual(len(outcome.state.observations), 1)

    def test_model_can_choose_open_web_search(self) -> None:
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(FixtureWebSearch())))
        llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "web.search",
                    "arguments": {
                        "query": "ACME covenant outlook latest quarter",
                        "count": 5,
                        "freshness": "pw",
                    },
                    "reason": "The question needs recent open-web evidence.",
                },
                {"action": "finish", "reason": "Two independent web domains now cover the request."},
            ]
        )
        harness.register(
            llm_planning_harness_tool(
                llm,
                network_access=False,
            )
        )
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="What changed in ACME's covenant outlook this week?",
                allow_network=True,
                run_id="model-web-search",
                max_model_calls=4,
            )
        )
        self.assertEqual(outcome.status, "degraded")
        catalog = {item["name"] for item in json.loads(llm.user_prompts[0])["available_tools"]}
        self.assertEqual(catalog, {"finance.calculate", "web.search"})
        self.assertEqual([item.task.tool_name for item in outcome.state.observations], ["web.search"])
        evidence = next(iter(outcome.state.bundle.evidence.values()))
        self.assertEqual(evidence.source.source_type.value, "web")
        self.assertEqual(evidence.source.metadata["content_basis"], "search_result_snippet")

    def test_semantic_evidence_gap_drives_retrieval_then_resolves(self) -> None:
        harness = ToolHarness()
        harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(FixtureWebSearch())))
        planner_llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "web.search",
                    "arguments": {"query": "ACME latest covenant", "count": 5},
                    "reason": "先检索最新 covenant 信息。",
                },
                {"action": "finish", "reason": "初步检索完成。"},
                {
                    "action": "call_tool",
                    "tool_name": "web.search",
                    "arguments": {"query": "ACME covenant headroom amount latest quarter", "count": 5},
                    "reason": "根据充分性缺口补查 covenant headroom 的具体金额。",
                },
                {"action": "finish", "reason": "补充检索后证据已充分。"},
            ]
        )
        adequacy_llm = ScriptedLLM(
            [
                {
                    "requirements": [
                        {
                            "requirement_key": "web:query:1",
                            "status": "insufficient",
                            "missing_information": ["缺少 covenant headroom 的具体金额"],
                            "retrieval_hint": "检索最新季度 covenant headroom amount",
                        }
                    ]
                },
                {
                    "requirements": [
                        {
                            "requirement_key": "web:query:1",
                            "status": "sufficient",
                            "missing_information": [],
                            "retrieval_hint": "",
                        }
                    ]
                },
            ]
        )
        harness.register(llm_planning_harness_tool(planner_llm, network_access=False))
        harness.register(llm_evidence_adequacy_harness_tool(adequacy_llm, network_access=False))
        outcome = llm_backed_agent(
            harness,
            planner=ModelPlanner(harness, count_tokens=len),
            adequacy_checker=LLMEvidenceAdequacyChecker(
                harness,
                max_evidence_tokens=48_000,
                count_tokens=len,
            ),
        ).run(
            *agent_run_input(
                query="最新的 ACME covenant 情况是什么？",
                run_id="semantic-retrieval-loop",
                allow_network=True,
                max_model_calls=9,
            )
        )
        self.assertEqual(outcome.status, "degraded")
        self.assertEqual([item.task.tool_name for item in outcome.state.observations], ["web.search", "web.search"])
        semantic_gaps = [item for item in outcome.state.gaps if item.code == "evidence_semantically_insufficient"]
        self.assertEqual(len(semantic_gaps), 1)
        self.assertTrue(semantic_gaps[0].resolved)
        third_planning_context = json.loads(planner_llm.user_prompts[2])
        self.assertIn("covenant headroom amount", third_planning_context["unresolved_gaps"][0]["message"])
        self.assertEqual(
            sum(item["tool_name"] == "llm.validate_evidence" for item in outcome.audit_events),
            2,
        )

    def test_semantically_insufficient_evidence_ends_with_explicit_abstention(self) -> None:
        harness = ToolHarness()
        harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(FixtureWebSearch())))
        planner_llm = ScriptedLLM(
            [
                {
                    "action": "call_tool",
                    "tool_name": "web.search",
                    "arguments": {"query": "ACME covenant amount", "count": 5},
                    "reason": "检索可用网页证据。",
                },
                {"action": "finish", "reason": "没有其它已授权来源可继续尝试。"},
            ]
        )
        adequacy_llm = ScriptedLLM(
            [
                {
                    "requirements": [
                        {
                            "requirement_key": "web:query:1",
                            "status": "insufficient",
                            "missing_information": ["搜索摘要未提供 covenant headroom 金额"],
                            "retrieval_hint": "需要包含金额的一手公告正文",
                        }
                    ]
                }
            ]
        )
        harness.register(llm_planning_harness_tool(planner_llm, network_access=False))
        harness.register(llm_evidence_adequacy_harness_tool(adequacy_llm, network_access=False))
        outcome = llm_backed_agent(
            harness,
            planner=ModelPlanner(harness, count_tokens=len),
            adequacy_checker=LLMEvidenceAdequacyChecker(
                harness,
                max_evidence_tokens=48_000,
                count_tokens=len,
            ),
        ).run(
            *agent_run_input(
                query="最新 ACME covenant headroom 的具体金额是多少？",
                run_id="semantic-abstention",
                allow_network=True,
                max_iterations=2,
                max_model_calls=6,
            )
        )
        self.assertEqual(outcome.state.stop_reason.value, "insufficient_evidence")
        self.assertEqual(outcome.status, "degraded")
        self.assertIn("当前证据不足以完整回答", outcome.state.report)
        self.assertTrue(any(not item.resolved for item in outcome.state.gaps))

    def test_web_search_deduplicates_tracking_urls_and_requires_source_diversity(self) -> None:
        class DuplicateSearch:
            provider_name = "duplicate-search"

            def search_json(self, payload):
                return {
                    "results": [
                        {
                            "title": "First",
                            "url": "https://example.com/news?id=1&utm_source=x#top",
                            "description": "ACME reported stronger liquidity.",
                        },
                        {
                            "title": "Duplicate",
                            "url": "https://EXAMPLE.com/news?id=1&utm_medium=y",
                            "description": "ACME reported stronger liquidity.",
                        },
                    ]
                }

        batch = WebSearchEvidenceAdapter(DuplicateSearch()).search({"query": "ACME liquidity", "count": 5})
        self.assertEqual(len(batch["bundle"]["evidence"]), 1)
        evidence = next(iter(batch["bundle"]["evidence"]))
        self.assertNotIn("utm_", evidence["source"]["locator"])
        self.assertNotIn("#top", evidence["source"]["locator"])

        bundle = WebSearchEvidenceAdapter(DuplicateSearch()).search(
            {"query": "ACME liquidity", "count": 5}
        )["bundle"]
        turn = ChatTurn(message="What changed in ACME liquidity this week?")
        decision = CoverageAssessor().assess(
            turn,
            EvidenceBundle.from_dict(bundle),
            ResearchScope(
                intents=(FinancialIntent.GENERAL_RESEARCH,),
                requirements=(
                    ResearchRequirement(
                        key="web:query:1",
                        category="web",
                        reason="The request requires diverse current web evidence.",
                    ),
                ),
            ),
        )
        self.assertFalse(decision.complete)

    def test_web_search_enforces_domain_allowlist_on_provider_results(self) -> None:
        class UnfilteredSearch:
            provider_name = "unfiltered-search"

            def search_json(self, payload):
                return {
                    "results": [
                        {
                            "title": "Allowed subdomain",
                            "url": "https://reports.pbc.gov.cn/financial-statistics",
                            "description": "Official statistics.",
                        },
                        {
                            "title": "Provider ignored the filter",
                            "url": "https://example.com/repost",
                            "description": "Unofficial repost.",
                        },
                    ]
                }

        result = WebSearchEvidenceAdapter(UnfilteredSearch()).search(
            {"query": "financial statistics", "domains": ["pbc.gov.cn"]}
        )
        sources = result["bundle"]["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["locator"], "https://reports.pbc.gov.cn/financial-statistics")

    def test_invalid_model_tool_selection_fails_after_one_bad_repair(self) -> None:
        harness = ToolHarness()
        harness.register(financial_calculation_harness_tool())
        harness.register(
            llm_planning_harness_tool(
                ScriptedLLM(
                    [
                        {
                            "action": "call_tool",
                            "tool_name": "browser.open_any_url",
                            "arguments": {"url": "http://127.0.0.1"},
                            "reason": "Invalid autonomous action.",
                        },
                        {
                            "action": "call_tool",
                            "tool_name": "browser.open_any_url",
                            "arguments": {"url": "https://example.com"},
                            "reason": "修复响应仍选择了不存在的工具。",
                        },
                    ]
                ),
                network_access=False,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "model planning response is unusable"):
            llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
                *agent_run_input(
                    query="解释市盈率",
                    run_id="invalid-model-tool",
                    max_model_calls=3,
                )
            )

    def test_brave_client_uses_fixed_origin_and_normalizes_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "api.search.brave.com")
            self.assertEqual(request.headers["X-Subscription-Token"], "test-key")
            self.assertEqual(request.url.params["freshness"], "pw")
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Result",
                                "url": "https://example.com/result",
                                "description": "Evidence snippet.",
                                "extra_snippets": ["More context."],
                                "page_age": "2026-08-11T00:00:00+00:00",
                            }
                        ]
                    }
                },
            )

        client = BraveWebSearchClient("test-key", transport=httpx.MockTransport(handler))
        result = client.search_json(
            {"query": "ACME results", "count": 3, "freshness": "pw", "domains": ["sec.gov"]}
        )
        self.assertEqual(result["results"][0]["url"], "https://example.com/result")

    def test_bocha_client_uses_fixed_origin_maps_freshness_and_normalizes_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, httpx.URL("https://api.bochaai.com/v1/web-search"))
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(
                json.loads(request.content),
                {
                    "query": "site:sec.gov ACME results",
                    "summary": False,
                    "count": 3,
                    "freshness": "oneWeek",
                },
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                json={
                    "code": 200,
                    "data": {
                        "webPages": {
                            "value": [
                                {
                                    "name": "Result",
                                    "url": "https://example.com/result",
                                    "snippet": "Evidence snippet.",
                                    "datePublished": "2026-08-11T00:00:00+00:00",
                                }
                            ]
                        }
                    },
                },
            )

        client = BochaWebSearchClient("test-key", transport=httpx.MockTransport(handler))
        result = client.search_json(
            {"query": "ACME results", "count": 3, "freshness": "pw", "domains": ["sec.gov"]}
        )
        self.assertEqual(result["results"][0]["url"], "https://example.com/result")
        self.assertEqual(result["results"][0]["published_at"], "2026-08-11T00:00:00+00:00")

    def test_bocha_client_rejects_api_level_error(self) -> None:
        client = BochaWebSearchClient(
            "test-key",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"Content-Type": "application/json"},
                    json={"code": 401, "msg": "invalid key", "data": None},
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "API error"):
            client.search_json({"query": "ACME"})

    def test_web_search_retries_one_http_transport_failure(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("temporary network failure", request=request)
            return httpx.Response(
                200,
                json={"web": {"results": [{"title": "Result", "url": "https://example.com", "description": "Data."}]}},
            )

        harness = ToolHarness(sleeper=lambda _seconds: None)
        client = BraveWebSearchClient("test-key", transport=httpx.MockTransport(handler))
        harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(client)))
        result = harness.invoke(
            "web.search",
            {"query": "ACME"},
            ToolContext(
                run_id="web-retry",
                thread_id="web-retry",
                policy=ExecutionPolicy(
                    allowed_capabilities=frozenset({"web.search"}),
                    allow_network=True,
                    max_network_calls=2,
                ),
            ),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)

    def test_web_search_rejects_non_domain_filters_before_provider(self) -> None:
        adapter = WebSearchEvidenceAdapter(FixtureWebSearch())
        with self.assertRaisesRegex(ValueError, "invalid public domain"):
            adapter.search({"query": "ACME", "domains": ["http://127.0.0.1"]})

        class PrivateResultSearch:
            provider_name = "private-result"

            def search_json(self, payload):
                return {
                    "results": [
                        {"title": "Private", "url": "http://127.0.0.1/admin", "description": "Do not trust."}
                    ]
                }

        with self.assertRaisesRegex(ValueError, "result URL is invalid"):
            WebSearchEvidenceAdapter(PrivateResultSearch()).search({"query": "ACME"})

    def test_model_can_select_multiple_tools_in_one_plan(self) -> None:
        harness = ToolHarness()
        llm = ScriptedLLM(
            [
                {
                    "action": "call_tools",
                    "reason": "同时计算 CAGR 和百分比变化。",
                    "tools": [
                        {
                            "tool_name": "finance.calculate",
                            "arguments": {
                                "requests": [
                                    {
                                        "operation": "cagr",
                                        "inputs": {"beginning_value": 100.0, "ending_value": 121.0, "years": 2.0},
                                        "label": "cagr",
                                    }
                                ]
                            },
                            "reason": "计算 CAGR。",
                        },
                        {
                            "tool_name": "finance.calculate",
                            "arguments": {
                                "requests": [
                                    {
                                        "operation": "percentage_change",
                                        "inputs": {"beginning_value": 100, "ending_value": 121},
                                    }
                                ]
                            },
                            "reason": "计算百分比变化。",
                        },
                    ],
                },
                {"action": "finish", "reason": "两个计算结果都已入账。"},
            ]
        )
        harness.register(financial_calculation_harness_tool())
        harness.register(llm_planning_harness_tool(llm, network_access=False))
        outcome = llm_backed_agent(harness, planner=ModelPlanner(harness, count_tokens=len)).run(
            *agent_run_input(
                query="一项投资从100增长到121，用了2年，CAGR和百分比变化是多少？",
                run_id="multi-tool-plan",
                max_model_calls=4,
                max_parallel_tool_calls=2,
            )
        )
        self.assertEqual(outcome.status, "succeeded")
        self.assertEqual(len(outcome.state.observations), 2)
        self.assertEqual({item.task.tool_name for item in outcome.state.observations}, {"finance.calculate"})
        self.assertEqual(
            [item["tool_name"] for item in outcome.audit_events],
            [
                "llm.task_frame",
                "llm.plan",
                "finance.calculate",
                "finance.calculate",
                "llm.plan",
                "llm.synthesize",
            ],
        )


if __name__ == "__main__":
    unittest.main()
