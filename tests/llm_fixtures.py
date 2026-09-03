"""脚本化 LLM 测试替身：覆盖任务理解、规划与证据约束合成，不复活规则降级。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from mas_finance.agent import AgentContext, ChatAttachment, ChatTurn, ResearchPlan, RuntimePolicy
from mas_finance.api.app import create_app
from mas_finance.graph import FinancialResearchAgent
from mas_finance.harness import ToolHarness
from mas_finance.llm import BaseLLMClient
from mas_finance.metrics import MetricOperation, MetricRequest
from mas_finance.planning import ModelPlanner, llm_planning_harness_tool
from mas_finance.service import FinanceAnalysisService
from mas_finance.synthesis import EvidenceBoundLLMSynthesizer, llm_synthesis_harness_tool
from mas_finance.task_frame import LLMTaskInterpreter, llm_task_frame_harness_tool

_PRONOUNS = ("它", "其", "前者", "后者", "这只", "这份", "刚才", "该公司", "那家")
_FIXTURE_SYMBOLS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA",
}
_DOCUMENT_PREFERRED = (
    "corpus.hybrid_search",
    "corpus.search",
    "personal.hybrid_search",
    "personal.search",
)
_PERSONAL_DOCUMENT_PREFERRED = (
    "personal.hybrid_search",
    "personal.search",
    "corpus.hybrid_search",
    "corpus.search",
)
_HISTORY_MARKERS = ("回撤", "波动", "收益率", "历史", "过去", "drawdown", "volatility", "return")
_MARKET_MARKERS = ("股价", "市值", "市盈率", "市净率", "当前价格", "snapshot", "price")
_REGULATORY_MARKERS = ("净利率", "盈利", "利润", "营收", "收入", "10-k", "sec", "监管")
_FILING_MARKERS = ("最新披露", "最新公告", "recent filing", "8-k", "10-q")
_UNSUPPORTED_MARKERS = ("精确收盘价", "预测明天")
_PERSONAL_LIBRARY_MARKERS = ("知识库", "knowledge base", "my library", "我的知识库")
_MACRO_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CPIAUCSL", ("cpi", "inflation", "consumer price", "通胀", "消费者价格")),
    ("UNRATE", ("unemployment", "jobless rate", "失业率")),
    ("GDP", ("gross domestic product", " gdp", "gdp ", "国内生产总值")),
    ("FEDFUNDS", ("federal funds", "fed funds", "联邦基金利率", "美联储利率")),
    ("DGS10", ("10-year treasury", "10 year treasury", "十年期美债", "10年期美债")),
    ("DGS2", ("2-year treasury", "2 year treasury", "两年期美债", "2年期美债")),
    ("PAYEMS", ("nonfarm payroll", "non-farm payroll", "非农就业", "非农数据")),
    ("PCEPI", ("pce price", "pce inflation", "pce物价", "pce通胀")),
    ("MORTGAGE30US", ("mortgage rate", "30-year mortgage", "房贷利率", "抵押贷款利率")),
)


def _requests_multi_document_synthesis(query: str) -> bool:
    normalized = query.casefold()
    return bool(
        re.search(
            r"(?:综合|对比|比较|分别|逐份).{0,12}(?:文档|材料|pdf|报告)|"
            r"(?:所有|全部|这些|几份).{0,12}(?:文档|材料|pdf|报告)|"
            r"\b(?:compare|synthesize|across|both|all)\b.{0,30}\b(?:documents?|files?|pdfs?)\b",
            normalized,
        )
    )


def _fixture_metric_requests(query: str) -> tuple[MetricRequest, ...]:
    normalized = query.casefold()
    numbers = [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)]
    if any(item in normalized for item in ("cagr", "复合增长率", "年复合增长率")) and len(numbers) >= 3:
        return (
            MetricRequest(
                operation=MetricOperation.CAGR,
                inputs={"beginning_value": numbers[0], "ending_value": numbers[1], "years": numbers[2]},
                label="cagr",
            ),
        )
    percentage_requested = any(
        item in normalized for item in ("percentage change", "percent change", "增长率", "变化率")
    )
    if percentage_requested and len(numbers) >= 2:
        return (
            MetricRequest(
                operation=MetricOperation.PERCENTAGE_CHANGE,
                inputs={"beginning_value": numbers[0], "ending_value": numbers[1]},
                label="percentage_change",
            ),
        )
    return ()


class ScriptedLLM(BaseLLMClient):
    backend_name = "scripted"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat(self, system_prompt, user_prompt, temperature=0.2, max_tokens=600):
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


class NullPlanner:
    """仅用于检查图结构或从 checkpoint 恢复、不再规划的测试。"""

    def plan(self, state, _available_tools) -> ResearchPlan:
        return ResearchPlan(state.iteration + 1, "空规划器", ())


class FixtureResearchLLM(BaseLLMClient):
    """可脚本化的研究模型：按 prompt 角色返回合法 JSON，不调用真实 API。"""

    backend_name = "fixture"

    def chat(self, system_prompt, user_prompt, temperature=0.2, max_tokens=600):
        del temperature, max_tokens
        payload = json.loads(user_prompt)
        if "最小语义事实" in system_prompt:
            events = list(payload.get("user_messages") or ())
            user_event = events[0] if events else None
            if user_event is None:
                return '{"facts":[]}'
            return json.dumps(
                {
                    "facts": [
                        {
                            "text": f"用户提出请求：{str(user_event.get('content') or '')[:300]}",
                            "source_event_ids": [user_event["event_id"]],
                            "entities": list(
                                dict.fromkeys(str(entity) for entity in payload.get("resolved_entity_candidates") or ())
                            ),
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "长期记忆候选" in system_prompt:
            return '{"candidates":[]}'
        if "可复用工作路径" in system_prompt:
            return '{"skill":null}'
        if "任务理解" in system_prompt:
            return json.dumps(self._task_frame(payload), ensure_ascii=False)
        if "规划组件" in system_prompt:
            return json.dumps(self._plan(payload), ensure_ascii=False)
        if "证据充分性校验组件" in system_prompt:
            return json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_key": item["key"],
                            "status": "sufficient",
                            "missing_information": [],
                            "retrieval_hint": "",
                        }
                        for item in payload.get("requirements") or ()
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(self._synthesize(payload), ensure_ascii=False)

    def _task_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        current = payload.get("current_request") or {}
        query = str(current.get("message") or "")
        current_entities: list[str] = []
        current_entities.extend(name for name in _FIXTURE_SYMBOLS if name in query and name not in current_entities)
        current_entities.extend(
            name
            for name in re.findall(r"[A-Z][A-Za-z0-9.-]{1,31}", query)
            if name not in current_entities and name not in {"PDF", "CAGR"}
        )
        symbols = {entity: _FIXTURE_SYMBOLS.get(entity, entity) for entity in current_entities}
        replay = _atomic_fact_replay(str(payload.get("atomic_fact_history") or ""))
        unique_history = _unique_replay_entities(replay)
        if _needs_clarification(query, current_entities, unique_history):
            return {
                "goal": "澄清被指代的研究对象后才能继续",
                "entities": [],
                "intents": ["general_research"],
                "requirements": [],
                "success_criteria": [],
                "clarification_question": "你指的是历史中的哪一家公司？",
            }

        entities = _frame_entities(current_entities, unique_history, replay, query)
        entity_names = [item["name"] for item in entities] or current_entities
        tools = list(payload.get("available_tools") or ())
        tool_names = {str(item.get("name")) for item in tools if item.get("name")}
        capabilities = {str(item.get("capability")) for item in tools}
        requirements = _frame_requirements(
            current,
            query,
            entity_names,
            symbols,
            capabilities,
            tool_names,
        )
        return {
            "goal": f"完成用户研究请求：{query[:180]}",
            "entities": entities,
            "intents": ["general_research"],
            "requirements": requirements,
            "success_criteria": ["在有检索需求时引用证据，概念题可直接作答"],
            "selected_skill_ids": [],
            "clarification_question": None,
        }

    def _plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        current_turn = payload.get("current_turn") or {}
        request = {
            **current_turn,
            "query": current_turn.get("message"),
            "attachments": current_turn.get("attachments") or (),
        }
        tools = {str(item["name"]): item for item in payload.get("available_tools") or () if item.get("name")}
        attempted = {
            _signature(str(item.get("tool_name") or ""), dict(item.get("arguments") or {}))
            for item in payload.get("prior_actions") or ()
        }
        coverage = payload.get("coverage") or {}
        missing = set(coverage.get("missing") or ())
        evidence = payload.get("evidence") or ()
        if coverage.get("complete"):
            return {"action": "finish", "reason": "覆盖已满足，停止收集证据。"}
        if evidence and not missing:
            return {"action": "finish", "reason": "已有证据且无未覆盖需求。"}

        scope: Mapping[str, Any] = {}
        frame = payload.get("task_frame") or {}
        request["entities"] = [str(item.get("name")) for item in frame.get("entities") or () if item.get("name")]
        if isinstance(frame.get("scope"), Mapping):
            scope = frame["scope"]
        requirements = list(scope.get("requirements") or ())
        if missing:
            requirements = [item for item in requirements if item.get("key") in missing]
        selected: list[dict[str, Any]] = []
        for requirement in requirements:
            item = _tool_for_requirement(
                requirement,
                tools,
                request,
                payload.get("mcp_tool_index") or (),
                attempted,
            )
            if item is None:
                continue
            if requirement.get("key"):
                item["requirement_key"] = requirement["key"]
            if requirement.get("entity"):
                item["entity"] = requirement["entity"]
            key = _signature(item["tool_name"], item["arguments"])
            if key in attempted:
                continue
            attempted.add(key)
            selected.append(item)
            if len(selected) >= 4:
                break
        if not selected:
            return {"action": "finish", "reason": "没有尚未尝试的合法工具。"}
        if len(selected) == 1:
            item = selected[0]
            payload_out = {
                "action": "call_tool",
                "tool_name": item["tool_name"],
                "arguments": item["arguments"],
                "reason": item["reason"],
            }
            if item.get("requirement_key"):
                payload_out["requirement_key"] = item["requirement_key"]
            if item.get("entity"):
                payload_out["entity"] = item["entity"]
            return payload_out
        return {"action": "call_tools", "tools": selected, "reason": "并行收集尚未覆盖的证据。"}

    def _synthesize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        cards = [item for item in payload.get("evidence") or () if isinstance(item, Mapping)]
        if not cards:
            query = str((payload.get("task") or {}).get("message") or "")
            text = f"根据金融概念解释：{query}" if query else "该问题可由模型根据金融概念直接回答。"
            return {
                "claims": [
                    {
                        "text": text[:500],
                        "evidence_ids": [],
                        "evidence_quote": "",
                    }
                ]
            }
        card = cards[0]
        content = str(card.get("content") or "")
        quote = _literal_quote(content)
        evidence_id = str(card.get("evidence_id") or "")
        return {
            "claims": [
                {
                    "text": f"证据表明：{quote}",
                    "evidence_ids": [evidence_id] if evidence_id else [],
                    "evidence_quote": quote,
                }
            ]
        }


def agent_run_input(**kwargs: Any) -> tuple[ChatTurn, RuntimePolicy, AgentContext]:
    message = str(kwargs.pop("query", kwargs.pop("message", "")))
    turn_fields = {
        key: kwargs.pop(key)
        for key in ("tenant_id", "user_id", "thread_id", "run_id", "allow_network")
        if key in kwargs
    }
    policy_fields = {
        key: kwargs.pop(key)
        for key in (
            "max_iterations",
            "max_tool_calls",
            "max_network_calls",
            "max_model_calls",
            "max_model_input_tokens",
            "max_model_output_tokens",
            "max_parallel_tool_calls",
        )
        if key in kwargs
    }
    policy_fields.setdefault("max_model_calls", 8)
    policy_fields.setdefault("max_iterations", 6)
    document_count = int(kwargs.pop("available_document_count", 0))
    attachments = tuple(
        ChatAttachment(document_id=f"doc-{index}", title=f"document-{index}.pdf") for index in range(document_count)
    )
    context = AgentContext(
        thread_context=dict(kwargs.pop("thread_context", {})),
        personal_context=tuple(kwargs.pop("personal_context", ())),
        skill_index=tuple(kwargs.pop("skill_index", ())),
    )
    if kwargs:
        raise TypeError(f"unsupported agent test input fields: {sorted(kwargs)}")
    return ChatTurn(message=message, attachments=attachments, **turn_fields), RuntimePolicy(**policy_fields), context


def register_research_llm(harness: ToolHarness, llm: BaseLLMClient, *, network_access: bool = False) -> None:
    names = {spec.name for spec in harness.tool_specs()}
    if "llm.task_frame" not in names:
        harness.register(llm_task_frame_harness_tool(llm, network_access=network_access))
    if "llm.plan" not in names:
        harness.register(llm_planning_harness_tool(llm, network_access=network_access))
    if "llm.synthesize" not in names:
        harness.register(llm_synthesis_harness_tool(llm, network_access=network_access))


def llm_backed_agent(harness: ToolHarness, llm: BaseLLMClient | None = None, **kwargs: Any) -> FinancialResearchAgent:
    client = llm or FixtureResearchLLM()
    register_research_llm(harness, client)
    mcp_tool_index = kwargs.pop("mcp_tool_index", ())
    if "planner" not in kwargs:
        kwargs["planner"] = ModelPlanner(harness, count_tokens=len, mcp_tool_index=mcp_tool_index)
    if "synthesizer" not in kwargs:
        kwargs["synthesizer"] = EvidenceBoundLLMSynthesizer(client, harness=harness, count_tokens=len)
    if "task_interpreter" not in kwargs:
        kwargs["task_interpreter"] = LLMTaskInterpreter(harness)
    return FinancialResearchAgent(harness, **kwargs)


def research_service(config, **kwargs: Any) -> FinanceAnalysisService:
    kwargs.setdefault("llm_client", FixtureResearchLLM())
    return FinanceAnalysisService(config, **kwargs)


def research_app(config, **kwargs: Any):
    kwargs.setdefault("llm_client", FixtureResearchLLM())
    return create_app(config, **kwargs)


def _unique_replay_entities(replay: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for event in replay:
        for name in event.get("entities") or ():
            text = str(name)
            if text and text not in names:
                names.append(text)
    return names


def _atomic_fact_replay(value: str) -> list[dict[str, Any]]:
    replay: list[dict[str, Any]] = []
    for match in re.finditer(r"event_id=([^;\]]+);[^\]]*\]\s*(.+)$", value, flags=re.MULTILINE):
        content = match.group(2)
        entities = [name for name in _FIXTURE_SYMBOLS if name in content]
        entities.extend(
            name
            for name in re.findall(r"[A-Z][A-Za-z0-9.-]{1,31}", content)
            if name not in entities and name not in {"PDF", "CAGR"}
        )
        replay.append({"event_id": match.group(1), "entities": entities})
    return replay


def _needs_clarification(query: str, current_entities: Sequence[str], history: Sequence[str]) -> bool:
    return (not current_entities) and any(marker in query for marker in _PRONOUNS) and len(history) > 1


def _frame_entities(
    current_entities: Sequence[str],
    history: Sequence[str],
    replay: Sequence[Mapping[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    entities = [{"name": name, "origin": "current_request"} for name in current_entities]
    if entities or not history:
        return entities
    if not any(marker in query for marker in _PRONOUNS):
        return entities
    chosen = history[0]
    event_id = next(
        (str(event.get("event_id")) for event in replay if chosen in (event.get("entities") or ())),
        None,
    )
    item = {"name": chosen, "origin": "conversation_memory"}
    if event_id:
        item["event_id"] = event_id
    return [item]


def _frame_requirements(
    current: Mapping[str, Any],
    query: str,
    entity_names: Sequence[str],
    symbols: Mapping[str, str],
    capabilities: set[str],
    _tool_names: set[str],
) -> list[dict[str, Any]]:
    if any(marker in query for marker in _UNSUPPORTED_MARKERS):
        return [
            {
                "category": "unsupported",
                "fields": [],
                "parameters": {"gap_code": "unsupported_research_scope"},
                "reason": "该问题超出当前证据优先研究范围。",
            }
        ]
    requirements: list[dict[str, Any]] = []
    inferred = [item.to_dict() for item in _fixture_metric_requests(query)]
    if "夏普" in query and not inferred:
        numbers = [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", query)]
        if len(numbers) >= 6:
            inferred.append(
                MetricRequest(
                    operation=MetricOperation.SHARPE_RATIO,
                    inputs={
                        "returns": numbers[:4],
                        "annualization_factor": numbers[4],
                        "annual_risk_free_rate": numbers[5],
                    },
                ).to_dict()
            )
    for item in inferred:
        request = item if "request_id" in item and item["request_id"] else MetricRequest.from_dict(item).to_dict()
        requirements.append(
            {
                "category": "calculation",
                "fields": [str(request.get("label") or request.get("operation") or "value")],
                "parameters": {"request_id": request["request_id"], "request": request},
                "reason": "需要确定性公式计算结果。",
            }
        )
    attachments = list(current.get("attachments") or ())
    want_documents = (
        bool(attachments)
        or any(
            marker in query.casefold()
            for marker in (
                "文档",
                "pdf",
                "报告",
                "材料",
                "知识库",
                "my library",
                "library",
                "政策",
                "扫描件",
                "document",
                "report",
            )
        )
        or ("covenant" in query.casefold() and "document.search" in capabilities)
    )
    if want_documents:
        minimum_documents = min(20, len(attachments)) if _requests_multi_document_synthesis(query) else 1
        requirements.append(
            {
                "category": "document",
                "fields": [],
                "parameters": ({"minimum_documents": minimum_documents} if minimum_documents > 1 else {}),
                "reason": "需要已授权文档中的证据。",
            }
        )
    series: list[str] = []
    normalized = query.casefold()
    for series_id, keywords in _MACRO_ALIASES:
        if any(keyword in normalized for keyword in keywords):
            series.append(series_id)
    if series and "macro.read" in capabilities:
        for series_id in series:
            requirements.append(
                {
                    "category": "macro",
                    "fields": ["latest_value"],
                    "parameters": {"series_id": series_id},
                    "reason": "需要官方宏观序列的最新观察值。",
                }
            )
    if "web.search" in _tool_names and any(
        marker in query.casefold() for marker in ("this week", "latest", "最新", "本周")
    ):
        requirements.append(
            {
                "category": "web",
                "fields": [],
                "parameters": {},
                "reason": "需要近期公开网页信息。",
            }
        )
    want_history = any(marker in query for marker in _HISTORY_MARKERS)
    want_market = any(marker in query for marker in _MARKET_MARKERS) and not want_history
    want_filings = any(marker in query for marker in _FILING_MARKERS)
    want_regulatory = any(marker in query for marker in _REGULATORY_MARKERS)
    for entity in entity_names:
        if want_history and "market.read" in capabilities:
            requirements.append(
                {
                    "category": "market_history",
                    "entity": entity,
                    "fields": ["total_return", "max_drawdown"],
                    "parameters": {
                        "range": "1y",
                        "symbol": symbols.get(entity, entity),
                    },
                    "reason": "需要价格历史以计算风险指标。",
                }
            )
        elif want_market and "market.read" in capabilities:
            requirements.append(
                {
                    "category": "market",
                    "entity": entity,
                    "fields": ["current_price"],
                    "parameters": {"symbol": symbols.get(entity, entity)},
                    "reason": "需要当前行情快照。",
                }
            )
        if want_filings and "regulatory.read" in capabilities:
            requirements.append(
                {
                    "category": "filings",
                    "entity": entity,
                    "fields": [],
                    "parameters": {"symbol": symbols.get(entity, entity)},
                    "reason": "需要近期监管披露元数据。",
                }
            )
        elif want_regulatory and "regulatory.read" in capabilities:
            requirements.append(
                {
                    "category": "regulatory",
                    "entity": entity,
                    "fields": ["revenue", "net_income"],
                    "parameters": {"symbol": symbols.get(entity, entity)},
                    "reason": "需要监管财务事实。",
                }
            )
    if not requirements and entity_names:
        requirements.append(
            {
                "category": "unsupported",
                "fields": [],
                "parameters": {"gap_code": "unsupported_research_scope"},
                "reason": "当前目录无法映射到可执行的证据需求。",
            }
        )
    return requirements


def _tool_for_requirement(
    requirement: Mapping[str, Any],
    tools: Mapping[str, Mapping[str, Any]],
    request: Mapping[str, Any],
    mcp_index: Sequence[Mapping[str, Any]],
    attempted: set[tuple[str, str]],
) -> dict[str, Any] | None:
    category = str(requirement.get("category") or "")
    entity = requirement.get("entity")
    query = str(request.get("query") or "")
    symbols = dict(request.get("symbols") or {})
    parameters = dict(requirement.get("parameters") or {})
    if category == "unsupported":
        return None
    if category == "calculation" and "finance.calculate" in tools:
        calc = parameters.get("request") or _calculation_from_request(request, query)
        if calc is None:
            return None
        return {
            "tool_name": "finance.calculate",
            "arguments": {"requests": [calc]},
            "reason": "调用白名单公式工具。",
        }
    if category == "document":
        return _document_tool(tools, request, requirement, mcp_index, attempted)
    if category == "macro" and "macro.fred_series" in tools:
        series_id = str(parameters.get("series_id") or (request.get("macro_series") or ["UNRATE"])[0])
        return {
            "tool_name": "macro.fred_series",
            "arguments": {"series_id": series_id},
            "reason": "读取官方宏观序列。",
        }
    if category == "market_history" and "market.history" in tools and entity:
        return {
            "tool_name": "market.history",
            "arguments": {
                "company": entity,
                "symbol": parameters.get("symbol") or symbols.get(entity),
                "range": parameters.get("range") or request.get("market_history_range") or "1y",
                "interval": request.get("market_history_interval") or "1d",
            },
            "reason": "读取价格历史。",
        }
    if category == "market" and "market.snapshot" in tools and entity:
        return {
            "tool_name": "market.snapshot",
            "arguments": {
                "company": entity,
                "symbol": parameters.get("symbol") or symbols.get(entity),
            },
            "reason": "读取当前行情快照。",
        }
    if category == "filings" and "sec.recent_filings" in tools and entity:
        symbol = parameters.get("symbol") or symbols.get(entity)
        if not symbol:
            return None
        return {
            "tool_name": "sec.recent_filings",
            "arguments": {"company": entity, "symbol": symbol},
            "reason": "读取近期监管披露。",
        }
    if category == "regulatory" and "sec.company_facts" in tools and entity:
        symbol = parameters.get("symbol") or symbols.get(entity)
        if not symbol:
            return None
        return {
            "tool_name": "sec.company_facts",
            "arguments": {"company": entity, "symbol": symbol},
            "reason": "读取监管公司事实。",
        }
    if category == "web" and "web.search" in tools:
        return {
            "tool_name": "web.search",
            "arguments": {"query": query, "count": 5},
            "reason": "检索公开网页摘要。",
        }
    return None


def _document_tool(
    tools: Mapping[str, Mapping[str, Any]],
    request: Mapping[str, Any],
    requirement: Mapping[str, Any],
    mcp_index: Sequence[Mapping[str, Any]],
    attempted: set[tuple[str, str]],
) -> dict[str, Any] | None:
    query = str(request.get("query") or "")
    search_query = " ".join((query, *(str(item) for item in request.get("entities") or ()))).strip()
    arguments: dict[str, Any] = {"query": search_query, "top_k": 5}
    minimum_documents = int((requirement.get("parameters") or {}).get("minimum_documents", 1))
    if minimum_documents > 1:
        arguments["diversify_documents"] = True
        arguments["top_k"] = min(
            20,
            max(5, minimum_documents),
        )
    if any(marker in query for marker in _PERSONAL_LIBRARY_MARKERS):
        preferred = _PERSONAL_DOCUMENT_PREFERRED
    else:
        preferred = _DOCUMENT_PREFERRED
    ordered = [name for name in preferred if name in tools]
    ordered.extend(
        name
        for name, spec in tools.items()
        if spec.get("input_contract") is not None
        and name not in ordered
        and name not in {"mcp.search_tools", "mcp.describe_tool", "mcp.call_tool"}
        and "query" in ((spec.get("input_contract") or {}).get("required") or ())
        and name not in {"web.search"}
    )
    # 其余 document.search 按目录顺序作为后备。
    for name, spec in tools.items():
        contract = spec.get("input_contract") or {}
        if name in ordered or name.startswith("llm.") or name.startswith("mcp."):
            continue
        if "query" in (contract.get("required") or ()) and name not in {"web.search", "finance.formula"}:
            ordered.append(name)
    for name in ordered:
        item = {"tool_name": name, "arguments": dict(arguments), "reason": "检索已授权文档。"}
        if _signature(name, item["arguments"]) not in attempted:
            return item
    document_mcp = [
        str(item.get("name"))
        for item in mcp_index
        if str(item.get("planner_category") or "") == "document"
        or str(item.get("capability") or "") == "document.search"
    ]
    if "mcp.call_tool" in tools:
        for name in document_mcp:
            item = {
                "tool_name": "mcp.call_tool",
                "arguments": {"name": name, "arguments": {"query": search_query}},
                "reason": "通过 MCP 调用已授权文档工具。",
            }
            if _signature("mcp.call_tool", item["arguments"]) not in attempted:
                return item
    return None


def _calculation_from_request(request: Mapping[str, Any], query: str) -> dict[str, Any] | None:
    inferred = _fixture_metric_requests(query)
    return inferred[0].to_dict() if inferred else None


def _literal_quote(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) >= 8:
        return compact[:80]
    padded = (compact + " evidence")[:12]
    return padded if len(padded) >= 8 else "evidence"


def _signature(name: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
    return name, json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, default=str)
