# LangGraph 编排、自主规划与恢复设计

状态：已实现
版本：2.0
日期：2026-08-12

## 1. 设计结论

系统使用一个 LangGraph `StateGraph` 作为唯一编排运行时。顶层只有四个有业务含义的节点：

```text
START
  ↓
intent
  ↓
planning ─────────────┐
  ↓                   │
validation ─证据不足──┘
  ↓ 证据满足或硬停止
final_generation
  ↓
validation
  ↓
END
```

没有 `ToolHarness` 节点，也没有通用 `act`、`critic`、`supervisor` 或按 provider 命名的业务节点。
Harness 是每次工具调用配套的执行 middleware；工具 adapter 是能力实现；两者都不决定图拓扑。

## 2. 为什么不是固定 workflow

固定的是生命周期和安全不变量，不固定的是研究路线：

| 固定内容 | 动态内容 |
|---|---|
| 必须先识别任务上下文 | 下一步调用哪个工具 |
| 每个动作必须经过 Harness | 检索式、实体、字段、时间窗口和 provider |
| 证据不足不能进入可信回答 | 是否需要内部 RAG、网页、SEC、行情、宏观或计算 |
| 最终 claim 必须引用 Evidence | 工具调用次数及先后顺序 |
| 预算、网络、副作用和停止上限 | 何时建议 `finish` |

这是一种受约束的自主 Agent，而不是角色流水线，也不是允许模型执行任意代码的自由 Agent。

## 3. 四个节点的准确职责

### 3.1 `intent`

输入是经过边界校验的 `ResearchRequest`。节点必须通过 `LLMTaskInterpreter` 生成可持久化 `TaskFrame` 和
`ResearchScope`：

- 中英文金融意图提示；
- entity/symbol；
- 数据时间和字段需求；
- 显式 calculation requests；
- 文档、行情、监管、宏观、知识或开放检索 requirements。

系统不存在规则式意图分析器或无模型 fallback。模型依据当前请求、线程上下文、实体事件回放和动态能力目录生成
最低 evidence requirements；历史实体必须声明来源，歧义无法可靠消解时直接提出澄清问题。Validation 只对模型已经
声明的 requirements 做确定性验收，不反向猜测用户意图。没有可用 LLM/`llm.task_frame` 时请求快速失败。

### 3.2 `planning`

每次进入可完成本轮计划中尚未观察的全部任务（默认最多 4 个并行）：

1. 组装规划上下文；
2. `ModelPlanner` 返回 `call_tool`、`call_tools` 或 `finish`；
3. 校验工具是否在本次运行目录中；
4. 用 `ToolArgumentContract` 校验参数；
5. 通过同一个 Harness 执行这些工具；
6. 把结果归一成 observation、EvidenceBundle、gap 和 audit；
7. 返回节点更新，让 LangGraph checkpoint 落盘。

模型规划输出只有两种形式：

```json
{
  "action": "call_tool",
  "tool_name": "web.search",
  "arguments": {"query": "ACME covenant outlook", "freshness": "pw"},
  "reason": "需要近期外部证据"
}
```

```json
{"action": "finish", "reason": "现有证据已经覆盖问题"}
```

规划上下文包含：

- 原始用户请求；
- intent hints 和字段级 requirements；
- 当前 coverage；
- 已调用工具及成功/错误码；
- 未解决 gaps；
- 有界 evidence 摘要；
- 本次已授权工具的描述、网络属性和输入契约。

Evidence、网页、PDF、RAG chunk、thread memory 和 provider error 都标记为不可信数据，不能向 planner 注入指令。

### 3.3 `validation`

同一个节点在两个时点执行不同的确定性合同：

- 生成前：计算 coverage、对齐期间/实体、派生可验证比率、恢复已被备用来源覆盖的 gap；
- 生成后：检查 claim/evidence 引用、报告 citation、冲突、gap 和必需 section。

模型选择 `finish` 不等于流程结束。若 requirements 仍缺失且尚有预算，validation 路由回 planning。模型执行工具后即使最低 coverage 已满足，也会获得一次基于新 Evidence 的后续规划机会；只有模型明确 finish 或到达硬预算才进入生成。
达到迭代/工具预算、没有可用 provider 或覆盖完成时，才进入 final generation。

### 3.4 `final_generation`

节点只做基于证据的语言生成：

- 使用 `FinancialContextAssembler` 按 entity/source/domain 和可选 document 分散选择 evidence cards，并记录逐轮 manifest；
- 要求每条 claim 提供已登记 evidence ID；
- 要求逐字 `evidence_quote`；
- reconciliation 显示不同来源/期间的冲突；
- 生成 report 后返回 validation。

算术仍由 `finance.calculate` 或确定性 derived-ratio 函数完成，模型不直接计算金融指标。

## 4. ModelPlanner 是唯一规划路径

研究请求必须配置 LLM。`ModelPlanner` 在模型返回非 JSON、选择未注册工具、参数非法或调用失败时快速失败，
不调用规则 planner，也不产生 `model_planner_fallback`。

模型重复相同 tool+arguments 时会得到相同 task ID，已观察动作不会再次执行，并记录
`repeated_planner_action`。这防止低质量循环浪费 provider 费用。

## 5. Harness 的位置

调用关系是：

```text
planning node
  └─ model chooses ToolTask
       └─ ToolHarness.invoke
            ├─ run identity binding
            ├─ capability / side-effect / network policy
            ├─ argument contract
            ├─ model/tool/network budgets
            ├─ timeout and read-only retry
            ├─ result contract
            └─ redacted audit
                 └─ concrete adapter/provider
```

Harness 不做这些事：

- 不推断用户意图；
- 不给 provider 排研究优先级；
- 不判断答案是否充分；
- 不修改模型选中的参数以“让它能跑”；
- 不用默认值掩盖合约错误。

因此它约束自主性，但不替代自主性。

## 6. LangGraph checkpoint 与恢复

项目删除了自建 `checkpoints.py`，不存在双 checkpoint。编译图时只注入 LangGraph checkpointer：

- 测试/短生命周期：`InMemorySaver`；
- 本地服务和后台 job：官方 `SqliteSaver`；
- 生产多实例：应替换官方 Postgres checkpointer。

LangGraph `thread_id` 不是直接使用用户输入，而是对 `tenant_id + thread_id + run_id` 计算稳定哈希，避免跨租户碰撞。
恢复时还会比较完整 `ResearchRequest`，同一个 run 不能换问题、权限、预算或身份。

每个 planning step 最多执行一个工具，因此：

```text
checkpoint N: 已有计划和 2 条 observations
planning: 执行第 3 个工具
checkpoint N+1: 第 3 条 observation + evidence + audit
```

若进程在 N+1 后崩溃，恢复从下一节点继续，不重复前三个动作。若进程在 provider 已产生响应但节点尚未提交状态时崩溃，
read-only 调用可能被再次执行；这是 at-least-once 边界。当前 Agent 不提供外部写入/交易工具，因此没有假装 exactly-once。
未来任何写工具必须带 provider 幂等键或 outbox，并在执行前使用 interrupt/人工批准。

Harness 预算由 checkpoint 中的 durable audit 重建：

- `budget_consumed=false` 的权限/输入拒绝不计入工具预算；
- network retry 按实际 attempts 恢复；
- model calls 独立计数；
- call sequence 从最后 durable call ID 继续。

后台 job 使用稳定 `job_id` 作为 run ID。首次状态为 pending 时新建图；进程崩溃留下 running 状态后，重试使用同一
job ID 和 SQLite checkpointer 恢复。终态再次 resume 只读取既有结果。

## 7. 开放联网检索

模型不能使用“任意 HTTP 请求”工具。开放网络通过 `web.search`：

```text
ModelPlanner
  └─ {query, count, freshness, domains}
       └─ Harness: web.search + network policy
            └─ WebSearchEvidenceAdapter
                 └─ configured WebSearchProvider
```

当前内置 `BochaWebSearchClient` 与 `BraveWebSearchClient`，但 Agent 只依赖
`WebSearchProvider.search_json()`，可以替换成企业搜索、SearXNG 或其他合约实现。固定 provider API
origin 是凭据和 SSRF 边界，不是固定研究路线。模型自主决定：

- 是否使用开放搜索；
- 查询词和搜索运算符；
- `pd/pw/pm/py` 时效窗口；
- 可选公开域名范围；
- 何时改用 SEC、FRED、RAG 或专用行情工具。

搜索结果被转换为 `SourceType.WEB`，保留 URL、publisher domain、发布时间、检索时间和 query；内容标记为
`search_result_snippet`。当前没有通用 `web.fetch`：搜索 snippet 可以作为弱证据，但重要监管/财务结论应优先调用
SEC、FRED、公司原始 filing 或授权 RAG。增加 fetch 前必须实现 DNS/IP 重绑定防护、内容类型/大小限制、redirect
策略和 result-URL capability，不能简单开放 `requests.get(model_url)`。

## 8. Provider 选择

Yahoo 不是全局默认路线。行情工具是否存在取决于部署注册：

- `market.snapshot` / `market.history`：当前可配置 AlphaVantage、实验 Yahoo 或 offline；
- `sec.company_facts` / `sec.recent_filings`：监管/财报；
- `macro.fred_series`：宏观；
- `corpus.search` 和注入式 retrieval tools：上传、session 或企业 RAG；
- `web.search`：开放检索；
- `finance.calculate`：白名单计算。

模型看到的是本次实际注册目录，不会看到缺少凭据或部署未启用的 SEC、FRED、RAG、搜索工具。行情的 offline
adapter 是一个有意保留的例外：它让离线请求得到明确的 `market_provider_unavailable`，而不是伪造行情。专用结构化源
通常比网页 snippet 更可靠，这个偏好写入 planner prompt，但最终工具选择由模型基于问题和现有证据完成。

## 9. 记忆与 graph state

四个平面保持分离：

| 平面 | 用途 | 是否是事实来源 |
|---|---|---|
| LangGraph checkpoint | run 恢复、历史、下一节点 | 保存既有证据，但不跨 run 自动召回 |
| Conversation memory | 有界 LLM 摘要、最近 20K token 完整 run、实体/焦点状态 | 否，只作多轮理解与消歧提示 |
| Session documents | 显式短期保留的 PDF 页文本 | 是，需本次显式召回 |
| Long-term RAG | 经过 ACL/retention 管理的持久知识 | 是，经 retrieval adapter 引用 |

LangGraph checkpointer 不是长期用户记忆。它可能包含敏感 evidence，生产必须有加密、TTL、删除和访问审计。

## 10. 已验证行为

自动化测试覆盖：

- 图中只有四个业务节点；
- 模型真实选择 `finance.calculate` 等目录内工具；
- 概念题可以不调用研究工具直接合成；
- 模型自主选择 `web.search`；
- validation 拒绝过早 `finish`；
- 未注册任意 URL 工具被拒绝并可见降级；
- 重复 task 不重复执行；
- SQLite 跨 Agent 实例恢复和 tenant 隔离；
- 合成崩溃后从 LangGraph 待执行节点恢复；
- Harness 预算和 audit 从 graph state 恢复；
- Bocha/Brave 固定认证 origin、查询参数和结果 schema；
- 总计 140 项金融、hybrid RAG、OCR、API、安全、恢复、记忆、上下文和自主规划测试。

2026-08-12 的真实 DeepSeek 小规模测试中，当时目录仍包含 `finance.knowledge`。现行契约已删除该内置词库，概念题改为模型直接合成。

## 11. 仍然明确存在的边界

- Bocha 账户已用两次小规模真实请求验证：API 原始契约与项目 EvidenceBundle 路径均成功；Brave 仍仅有
  MockTransport 契约测试。搜索结果仍只是 snippet 级发现证据。
- SQLite checkpointer 适合单进程/本地 job；多实例生产应使用 Postgres saver。
- checkpoint 尚无 KMS 加密、自动 TTL/删除 worker 和 append-only 访问审计。
- 模型 planning 使用严格 JSON，不使用 provider 原生 function-calling；当前 DeepSeek 接口测试可工作，但后续若客户端支持稳定
  tool-calling，可替换输出通道而不改变图。
- 通用 web fetch 尚未实现，这是有意的安全边界，不是遗漏一个 `requests.get`。
- 没有交易、下单或外部写工具。
