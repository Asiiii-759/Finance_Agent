# 金融研究 Agent 架构

> 本文是架构决策摘要。面向使用、开发与评审的逐层说明见
> [《MAS Finance Agent：完整架构、运行机制与能力说明》](AGENT_DETAILED_GUIDE.md)。
> 工具目录、MCP 渐进发现与报错分层见 [《工具、金融场景与自适应调用逻辑》](TOOLS_AND_REASONING.md)。

状态：已采纳并进入实现
版本：2.2
日期：2026-08-27

## 1. 产品边界

本项目是“证据优先”的金融研究系统：它从用户授权文档、内部知识库和外部行情源获取证据，经过结构化计算与校验后生成带引用的回答。它不是交易系统，不连接券商，不执行资金操作，也不把模型输出当成事实来源。

交付准则：

1. 每个事实性结论必须引用 `Evidence`，每条证据必须指向 `SourceRef`。
2. 缺失、冲突、推断与已证实内容必须显式区分；没有证据时失败关闭。
3. 所有工具调用都经过同一个 Harness 的权限、网络、副作用、预算、重试和审计边界。
4. 循环必须存在硬停止条件，并可从 JSON 检查点恢复。
5. tenant、user、thread、run 四级身份贯穿检查点、记忆、审计与接口。
6. 默认仅提供研究能力；`financial_transaction` 在代码层默认拒绝。

## 2. 单项目决策

唯一主项目为 `Multi-Agent-project`，Python 包名保持 `mas_finance`。旧的固定角色图、演示数据、双入口和
兼容节点已经删除；当前重新以 LangGraph 1.2 构建唯一业务图，不保留旧图接口。

```text
Multi-Agent-project/
├── src/mas_finance/
│   ├── graph.py           # 四节点 LangGraph、路由、恢复；每轮最多四个工具并行执行
│   ├── agent.py           # 状态、覆盖评估和报告领域对象
│   ├── planning.py        # 模型自主规划、渐进发现上下文与 llm.plan Harness tool
│   ├── research.py        # 中英金融意图、ResearchScope 与 evidence requirements
│   ├── contracts.py       # Source / Evidence / Claim 稳定契约
│   ├── harness.py         # 工具权限、预算、重试、结构化执行错误、审计
│   ├── mcp.py             # MCP Host/Client、只读过滤、渐进发现元工具
│   ├── mcp_servers/       # 本地 stdio MCP server（当前含 AllTick/必盈行情）
│   ├── rate_limit.py      # 进程内滑动窗口限流
│   ├── memory_store.py    # 持久对话事件、滚动摘要、实体/焦点状态与 namespace 隔离
│   ├── memory_consolidation.py # 长期记忆候选提取
│   ├── personal_knowledge.py # 用户隔离的持久 PDF 页文本库
│   ├── embeddings.py      # embedding provider 协议与受限 HTTP 边界
│   ├── corpus.py          # BM25、向量、RRF 与文档分散检索后端
│   ├── retrieval.py       # 检索结果到 Evidence 的适配层
│   ├── web_search.py      # provider-neutral 开放搜索与 Bocha/Brave adapters
│   ├── market_data.py     # 外部行情 provider
│   ├── market.py          # 行情字段到 Evidence 的适配层
│   ├── macro.py           # FRED 宏观序列 provider 与 Evidence 适配
│   ├── metrics.py         # 白名单金融公式与计算 Harness tool
│   ├── formula.py         # 模型自拟声明式公式的安全 AST 执行器
│   ├── ocr.py             # 有界 PaddleOCR 文档解析 adapter
│   ├── knowledge.py       # 版本化金融定义、公式与解释 caveat
│   ├── calculator.py      # 确定性金融计算
│   ├── context.py         # Prompt 上下文选择、相关排序/可选文档分散与 provenance cards
│   ├── synthesis.py       # 有证据约束的 LLM 写作
│   ├── validators.py      # 报告、claim、citation 后置校验
│   ├── service.py         # 应用服务与依赖装配
│   ├── api/               # FastAPI 边界
│   └── cli.py             # 命令行边界
├── tests/
└── docs/
```

`Agent/Finance_RAG` 不再是运行时依赖。内部检索已经通过 provider-neutral 的 `search_json()` 契约进入主项目；
本地实现支持 BM25 及可配置 embedding/RRF，后续替换为持久向量库或托管检索时只新增 adapter，不改变 Agent
状态。算法、网络权限拆分和部署接口见 [BM25 + Embedding 双路检索设计](HYBRID_RETRIEVAL.md)。

## 3. 唯一业务图

```text
START → intent → planning ─────────────┐
                   │                   │
                   │ 每轮最多四个工具   │
                   ↓                   │
               validation ──证据不足──┘
                   │
                   │ 证据满足/硬停止
                   ↓
            final_generation
                   ↓
               validation ─→ END
```

图只有 `intent / planning / validation / final_generation` 四个业务节点。不存在 `ToolHarness`、`act`、
`critic` 或按角色命名的工具节点。`planning` 每次读取动态工具目录：配置了 LLM 时由 `ModelPlanner` 选择
1–4 个已注册工具（或 `finish`），并在同一节点内经 Harness 执行；多个任务按 `max_parallel_tool_calls`
（默认 4）并行 invoke。节点返回后 LangGraph 保存 observation、证据和 audit。同一 `ResearchPlan` 若因预算
被截断，恢复后继续执行尚未 observation 的 task，而不是重跑整批。

有 LLM 时，MCP（Model Context Protocol，模型上下文协议）具体工具对规划目录隐藏；模型只看到 builtins、
短 `mcp_tool_index` 和三个发现元工具（`mcp.search_tools` / `mcp.describe_tool` / `mcp.call_tool`）。
DeepSeek 请求不带 native `tools` 字段。LLM 是研究链路的必需依赖：缺少配置、模型不可用或规划/合成 JSON
非法时快速失败，不回退到规则 planner 或确定性合成器。

`validation` 有两种确定性职责：生成前检查 evidence requirements，拒绝模型过早 `finish` 并路由回 planning；模型工具执行后即使已达到最低 coverage，也会再回 planning，由模型基于新证据明确继续或 finish（硬预算到达除外）；
生成后检查 claim/citation/report 契约并结束。它不替模型选择工具。`final_generation` 只负责基于证据建立 claims
和报告，不进行研究工具选择。

停止原因是稳定枚举：

- `clarification_required`
- `coverage_satisfied`
- `max_iterations`
- `tool_budget_exhausted`
- `no_available_action`
- `validation_failed`
- `no_evidence`

服务默认最多 6 次规划迭代、12 次研究工具调用、8 次数据 provider 尝试和 8 次模型调用（规划加最终生成）；
领域请求允许 `max_iterations=1..8`、`max_tool_calls=1..100`、`max_network_calls=0..max_tool_calls`、
`max_model_calls=0..20`、`max_parallel_tool_calls=1..8`。模型只能选择运行时已注册的 `ToolSpec`，不能构造
import、任意函数或任意 HTTP 客户端。相同 tool+arguments 会形成稳定 task ID，重复动作记
`repeated_planner_action` 且不会再次执行。模型调用不占研究工具预算，网络 retry 逐次占数据 provider 尝试预算。

## 4. 数据契约与证据账本

### SourceRef

记录 provider、locator、source type、as-of、发布时间、实际获取时间和元数据。`source_id` 由 provider、locator、类型和 as-of 稳定计算。

### Evidence

文本证据保存页码/span；结构化证据保存 entity、field、value、unit、period。检索排序分数不伪装成概率；只有经过校准的抽取置信度才能写入 `confidence`。

### Claim

- `supported`：至少引用一条已登记证据。
- `inferred`：必须带可见 caveat。
- `unsupported`：明确资料不足。
- `conflicted`：展示冲突口径，不能静默择一。

`EvidenceBundle.add_claim()` 强制引用完整性。最终校验器还会检查报告中的 citation、footnote、data gap 和风险提示。

## 5. Harness、MCP Host 与失败语义

`ToolHarness` 是每一次工具选择配套的执行 middleware，不是 LangGraph 节点。顺序固定：注册表解析 →
run identity/预算上限绑定 → capability → side effect → network → 输入契约 → 分账预算 → provider timeout/retry →
输出契约 → 审计脱敏。它不决定“应该研究什么”或“调用哪个工具”；这些属于 planner。核心输入契约拒绝缺失字段、
多余字段、非有限 JSON 和超大载荷；数据工具只能返回可验证的 `EvidenceBundle`，模型工具只能返回有界 model response。

副作用分为：

- `read_only`（默认允许）
- `local_write`
- `external_write`
- `financial_transaction`（默认拒绝）

自动重试仅适用于 read-only 工具，且必须命中该 `ToolSpec` 声明的 exception 类型。同步工具的超时目前是“观测超时”；HTTP/数据库 provider 必须同时设置底层 I/O timeout。审计参数会遮蔽 token、API key、authorization、password，并截断异常大文本。`ToolExecutionError` 会把 `error_code`、`error_message` 和 `error_details` 写入 `ToolResult`，供下一轮规划使用；它不在默认 retry 集合里。

### 5.1 MCP Host

Agent 同时是 MCP Host：`MAS_MCP_SERVERS` 部署 allowlist 连接本地 stdio 或固定 HTTPS JSON-RPC Client。
进 Harness 前过滤只读注解、既有证据 capability 和合法参数名；拒绝项记为 `McpRejection`，不进模型目录。
`tools/call` 必须能变成 canonical `EvidenceBundle`，原始 MCP JSON 不能当 Evidence。配置 AllTick 或必盈许可时
自动挂载本地 `extmarket` server。FRED、Bocha、Brave、内置行情与 MCP `tools/call` 使用进程内滑动窗口限流。
计算工具和内部研报 RAG 仍留在进程内。HTTP MCP URL 必须是启动时固定的凭据无关 HTTPS；stdio command 由部署配置。

### 5.2 报错之后怎么走

失败不是单一“自动再打一次”：

| 层 | 触发 | 行为 |
|---|---|---|
| Harness 自动重试 | `TimeoutError` / `ConnectionError` 等 ToolSpec 声明的异常 | web/RAG/FRED/SEC 多为 `max_attempts=2`；MCP 绑定工具默认 `max_attempts=1`，限流超时也只失败一次 |
| MCP 结构化错误 | `isError=true` → `ToolExecutionError` | `ok=false`，保留 `retryable`、`suggested_action` 等 `error_details`；Graph 记 resolvable gap，不合并 Evidence |
| 规划改参 | 下一轮 `prior_actions` 含错误细节 | 模型可改 arguments 再 `mcp.call_tool`；完全相同 task ID 不重跑 |
| 空结果但仍成功 | adapter 返回 bundle + `gaps`，`isError=false` | Coverage 仍缺；系统不会把 `AAPL` 改写成 `AAPL.US` |

发现元工具查不到名字会抛普通 `ValueError`，不是 MCP `isError` 契约。JSON-RPC 传输失败同样走通用工具错误。
Host 会转发 server 给出的 `field` / `candidates`；内置 `extmarket` 当前携带 `error_code`、`retryable`、
`received_arguments` 和 `suggested_action`，不保证每次都带枚举候选。

## 6. 记忆模型

| 平面 | 命名空间 | 内容 | 实现 |
|---|---|---|---|
| Run checkpoint | tenant/thread/run 哈希 thread ID | graph step、plan、observations、证据、audit | LangGraph `InMemorySaver` / `SqliteSaver` |
| Conversation memory | tenant/user/thread/kind | 完整 user/tool/assistant/atomic_fact 事件与有界 prompt 投影 | `SQLiteMemoryStore` |
| Personal memory | tenant/user/kind | profile/preference/experience；明确长期 update 可覆盖，临时要求忽略 | SQLiteMemoryStore |
| Learned Skill | tenant/user/learned_skills | 成功工作路径；索引选择后才披露完整步骤 | SQLiteMemoryStore |
| Personal knowledge | tenant/user/document | 解析页文本与来源元数据 | SQLite 文本 + BM25；配置后可在查询期 embedding/RRF；显式上传/删除 |
| Tool usage memory | tenant/user/`tool_usage_memory` | 曾成功的 MCP 参数示例与 schema fingerprint | 仅 Harness `success`；schema 变化后停用；最多五条进入规划上下文 |
| Domain corpus | tenant/KB/version | 文档 chunk、metadata、索引 | retrieval backend |
| Run log | tenant/user/thread/run | 脱敏参数、工具返回摘要、状态、耗时、失败阶段 | SQLite `run_logs` |

对话完整事件账本默认保留到用户显式删除；进入模型的投影默认上限为 300K token。达到预算 85% 时，专用 LLM 滚动生成结构化语义摘要，最近原始事件继续保留，原账本不删除。独立的原子事实由 LLM 提取最小语义短句、带来源和时间持久化，不参加摘要且全历史进入 TaskFrame；无法可靠消解时返回澄清问题。LLM 是研究链路的必需依赖，未配置则快速失败。记忆不保存 EvidenceBundle，也不能作为事实来源。详见 [记忆与日志](CONVERSATION_MEMORY.md) 与 [LLM TaskFrame](TASK_FRAME.md)。

个人长期记忆支持显式 CRUD 和受限 LLM 沉淀，最多召回八条且作为低权限 personal context；Skill 使用独立成功路径存储和渐进披露。个人 PDF 必须走独立持久上传接口，临时上传不会自动入库。完整边界见 [个人金融助手：记忆、上下文与扩展边界](PERSONAL_ASSISTANT_MEMORY_AND_CONTEXT.md)。

检查点包含恢复所需的证据文本，属于敏感 run 数据。生产部署必须设置加密、保留期与删除任务；不能把它当长期用户记忆。

## 7. 数据源接口

Agent 只理解“返回 EvidenceBundle 的工具”，不理解某个供应商 SDK。

- 内部/外部文档：部署期 `RetrievalSource` → `RAGClient.search_json(payload)` → `RetrievalEvidenceAdapter`；固定 filters 不能被 Agent 覆盖。
- 上传 PDF：PaddleOCR-VL-1.6 或部署注入的成熟 PDF 解析 MCP → request/session `InMemoryCorpus`；远程解析受双重网络授权，默认不跨请求，显式 session 仅短 TTL 保存解析页文本，citation 保留解析器、page/span。
- 开放搜索：`web.search` 是 provider-neutral 工具；模型自主生成 query、`pd/pw/pm/py` 时效窗口和可选域名范围。
  当前内置 Bocha 与 Brave adapters 使用各自固定认证 API origin；两者同时配置时优先 Bocha。结果 URL
  可以来自公开网络并登记为 `SourceType.WEB`。
  返回内容明确标记为 search-result snippet；当前不提供任意 URL fetch。
- 行情：`market.snapshot` 返回字段级快照；`market.history` 显式记录 adjusted/raw price basis 并生成收益、年化波动率和最大回撤。默认 provider 为 offline；AlphaVantage 与 Yahoo 必须显式配置，且绝不静默跨 provider fallback。Yahoo 只标记为实验适配器。配置 AllTick/必盈时额外挂载 MCP `extmarket`（snapshot/history）；模型经渐进发现调用。
- MCP / 企业只读工具：Host allowlist 接入后进入同一条 Harness；规划侧用渐进发现，不把完整 JSON Schema 一次性塞进 prompt。尚未实现 SSE Streamable HTTP 与 OAuth。
- 监管：`sec.company_facts` 使用 SEC Company Facts XBRL；`sec.recent_filings` 使用 submissions recent filings 元数据。两者要求声明组织与联系邮箱的 User-Agent。接口依据：[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。
- 宏观：`macro.fred_series` 使用 FRED 官方 series/observations API，要求独立 API key。
- 计算：`finance.calculate` 只执行白名单公式并将用户输入、公式版本和结果登记为 Evidence；账本内比率仍要求相同实体/单位/期间。
- 自拟计算：`finance.formula` 只执行声明式 AST 白名单；能保证安全与可复算，不能保证金融语义，所以 claim 标为 inferred。
- 教育解释：概念、公式含义和机制由模型直接判断；引用检索证据时才做逐字 quote 校验。没有代码内金融词库。

增加新数据源的标准步骤：固定 provider 客户端（含 timeout/认证）→ anti-corruption adapter → `ToolSpec` →
契约测试 → 服务注册。模型下一轮自动从动态目录看到新工具。模型不能直接访问 provider 或读取密钥。

## 8. LLM 使用边界

LLM 负责两类受约束决策：`ModelPlanner` 每轮从动态目录选择 1–4 个工具动作或 `finish`；
`EvidenceBoundLLMSynthesizer` 基于证据生成 claims。模型不负责权限、预算、算术或引用合法性。
规划上下文包含用户请求、模型产生的 TaskFrame/requirements、coverage、`prior_actions`（含 `ok` / `error_code` /
`error_details`）、未解决 gaps、有限 evidence 摘要、工具输入契约、MCP 短索引、发现结果和
`verified_tool_usage`；文档、网页、记忆和工具错误都明确标记为不可信数据。模型输出必须是严格 JSON，
工具名和参数先过 ToolSpec/Harness。MCP 完整 schema 只在 `mcp.describe_tool` 之后进入下一轮上下文。
最终 claims 若引用 evidence，必须提供 evidence IDs 与逐字 quote；无引用的概念判断允许作为 inferred。规划或合成输出不可用时快速失败，不由确定性基线接管。

这不是完整的语义蕴含证明，因此后续仍应加入 NLI/人工抽检。检索性事实 claim 不能仅以模型输出为 source。

## 9. API 与任务

当前 `/api/v1/analyze` 和 `/api/v1/analyze-upload` 已直接使用新 Agent。请求可显式提供 entities、symbols 和 `allow_network`；网络实际开放需要请求与服务端 `MAS_ALLOW_NETWORK=true` 双重同意。上传默认 request-local；显式 `retain_for_session/use_session_documents` 才使用进程内短 TTL 页文本层；个人永久文档使用 `/api/v1/knowledge/documents`，企业库通过受控 `RetrievalSource/evidence_tools` 接入，详见 [文档生命周期设计](DOCUMENT_LIFECYCLE.md)。

API 路由使用 async 接口，但当前容器的线程执行器不可用，所以领域服务在 event loop 内同步执行。生产部署应先使用多 Uvicorn worker 做隔离；下一步将 provider 与数据库改成原生 async 后再提供单进程高并发保证。

作业状态使用 SQLite/PostgreSQL repository；Redis list worker 目前没有 visibility timeout/lease，不能宣称 exactly-once。生产应迁移到 Redis Streams、RQ/Celery 或数据库 outbox。

## 10. 安全与运维

- 上传：数量、大小、`.pdf` 后缀、PDF magic、归一化文件名和根目录约束。
- 检索：文档内容是不可信数据，不能改变系统提示或工具权限。
- 外部访问：仅固定 provider endpoint；MCP HTTP URL 只能是启动时配置的凭据无关 HTTPS；run 默认禁网。
- 输出：检索/计算题无证据失败关闭；概念题允许无引用 inferred claim；缺口可见，引用完整性和免责声明为硬校验。
- 产物：安全文件名、随机后缀、防覆盖；生产需加密与 retention。
- 部署：API key 常量时间比较；反向代理还需 body/rate limit。
- 交易：主项目不提供 broker tool；未来必须拆成独立服务并要求人工批准和独立风控。

## 11. 测试门槛

每次结构变更至少执行：

```bash
python -m unittest discover -s tests -v
pytest --cov=mas_finance --cov-report=term
ruff check src tests run_demo.py start_api.py start_worker.py
python -m compileall -q src tests
pip check
```

覆盖范围包括：契约引用完整性和 content-addressed 防篡改、工具输入/输出契约、权限/分账预算/脱敏、MCP Host 过滤与结构化错误、provider 故障、无证据失败、SQLite checkpoint 恢复、持久对话/动态压缩/指代/显式删除、上下文裁剪、citation laundering、金融指标血缘、PDF 上传安全、API 鉴权/作业/上传和产物路径安全。可运行评测矩阵见 [企业级验证与故障注入报告](ENTERPRISE_EVALUATION.md)。

## 12. 后续优先级

1. 将 job queue 改成带 lease、重试和幂等键的可靠队列。
2. 增加持久化/向量化内部 corpus、文档 ACL 和索引 manifest。
3. 增加新闻、earnings call、商业行情、内部 SQL 等固定 endpoint adapter 与跨来源冲突检测。
4. 将 HTTP、数据库和 PDF 解析迁到真正可取消的异步/进程隔离边界。
5. 增加 OpenTelemetry、append-only audit、成本/token budget 和数据保留任务。
