# 金融研究 Agent 架构

> 本文是架构决策摘要。面向使用、开发与评审的逐层说明见
> [《MAS Finance Agent：完整架构、运行机制与能力说明》](AGENT_DETAILED_GUIDE.md)。

状态：已采纳并进入实现
版本：2.2
日期：2026-08-20

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
│   ├── graph.py           # 四节点 LangGraph、路由、恢复和单动作执行
│   ├── agent.py           # 状态、规则规划基线、覆盖评估和报告领域对象
│   ├── planning.py        # 模型自主规划与 llm.plan Harness tool
│   ├── research.py        # 中英金融意图、ResearchScope 与 evidence requirements
│   ├── contracts.py       # Source / Evidence / Claim 稳定契约
│   ├── harness.py         # 工具权限、预算、重试、审计
│   ├── memory_store.py    # 持久对话事件、滚动摘要、实体/焦点状态与 namespace 隔离
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
                   │ 每次一个工具动作  │
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
`critic` 或按角色命名的工具节点。`planning` 每次读取动态工具目录，由 `ModelPlanner` 选择一个工具及参数，
并在同一节点内经 Harness 执行；节点返回后 LangGraph 保存 observation、证据和 audit。一个节点只执行一个工具，
因此恢复粒度不会退化成“重跑整批工具”。模型选择违反工具契约、模型不可用或预算不足时，
`AdaptivePlanner` 作为可见降级基线接管。

`validation` 有两种确定性职责：生成前检查 evidence requirements，拒绝模型过早 `finish` 并路由回 planning；模型工具执行后即使已达到最低 coverage，也会再回 planning，由模型基于新证据明确继续或 finish（硬预算到达除外）；
生成后检查 claim/citation/report 契约并结束。它不替模型选择工具。无模型规则基线满足 requirement 后直接结束。`final_generation` 只负责基于证据建立 claims
和报告，不进行研究工具选择。

停止原因是稳定枚举：

- `coverage_satisfied`
- `max_iterations`
- `tool_budget_exhausted`
- `no_available_action`
- `validation_failed`
- `no_evidence`

服务默认最多 6 次规划迭代、12 次研究工具调用、8 次数据 provider 尝试和 7 次模型调用（规划加最终生成）；
请求可在受控范围内调整。模型只能选择运行时已注册的 `ToolSpec`，不能构造 import、任意函数或任意 HTTP
客户端。相同 tool+arguments 会形成稳定 task ID，重复动作不会再次执行。模型调用不占研究工具预算，网络 retry
逐次占数据 provider 尝试预算。

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

## 5. Harness 工程

`ToolHarness` 是每一次模型工具选择配套的执行 middleware，不是 LangGraph 节点。顺序固定：注册表解析 →
run identity/预算上限绑定 → capability → side effect → network → 输入契约 → 分账预算 → provider timeout/retry →
输出契约 → 审计脱敏。它不决定“应该研究什么”或“调用哪个工具”；这些属于 planner。核心输入契约拒绝缺失字段、
多余字段、非有限 JSON 和超大载荷；数据工具只能返回可验证的 `EvidenceBundle`，模型工具只能返回有界 model response。

副作用分为：

- `read_only`（默认允许）
- `local_write`
- `external_write`
- `financial_transaction`（默认拒绝）

自动重试仅适用于 read-only 工具。同步工具的超时目前是“观测超时”；HTTP/数据库 provider 必须同时设置底层 I/O timeout。审计参数会遮蔽 token、API key、authorization、password，并截断异常大文本。

## 6. 记忆模型

| 平面 | 命名空间 | 内容 | 实现 |
|---|---|---|---|
| Run checkpoint | tenant/thread/run 哈希 thread ID | graph step、plan、observations、证据、audit | LangGraph `InMemorySaver` / `SqliteSaver` |
| Conversation memory | tenant/user/thread/kind | 完整 user/tool/assistant 事件与有界 prompt 投影 | `SQLiteMemoryStore` |
| Personal memory | tenant/user/kind | 显式 profile/preference/experience/skill | SQLiteMemoryStore；同槽位显式覆盖 |
| Personal knowledge | tenant/user/document | 解析页文本与来源元数据 | SQLite 文本 + BM25；配置后可在查询期 embedding/RRF；显式上传/删除 |
| Domain corpus | tenant/KB/version | 文档 chunk、metadata、索引 | retrieval backend |
| Audit | tenant/thread/run/call | 脱敏参数、状态、耗时、错误码 | run state + artifact；生产待 append-only store |

对话完整事件账本默认保留到用户显式删除；进入模型的投影默认上限为 300K token。达到预算 85% 时，专用 LLM 滚动生成结构化语义摘要，最近原始事件继续保留，原账本不删除。实体身份不交给摘要模型：系统从用户事件确定性构建 `entity_state + focus_history`，用于前者、后者、复数及“刚刚那个公司”的指代解析；歧义单数不猜。记忆不保存 EvidenceBundle，也不能作为事实来源。详见 [持久对话记忆与动态上下文](CONVERSATION_MEMORY.md)。

个人长期记忆只由显式 CRUD 创建，最多召回八条且作为低权限 personal context；个人 PDF 必须走独立持久上传接口，临时上传不会自动入库。完整边界见 [个人金融助手：记忆、上下文与扩展边界](PERSONAL_ASSISTANT_MEMORY_AND_CONTEXT.md)。

检查点包含恢复所需的证据文本，属于敏感 run 数据。生产部署必须设置加密、保留期与删除任务；不能把它当长期用户记忆。

## 7. 数据源接口

Agent 只理解“返回 EvidenceBundle 的工具”，不理解某个供应商 SDK。

- 内部/外部文档：部署期 `RetrievalSource` → `RAGClient.search_json(payload)` → `RetrievalEvidenceAdapter`；固定 filters 不能被 Agent 覆盖。
- 上传 PDF：PaddleOCR-VL-1.6 或部署注入的成熟 PDF 解析 MCP → request/session `InMemoryCorpus`；远程解析受双重网络授权，默认不跨请求，显式 session 仅短 TTL 保存解析页文本，citation 保留解析器、page/span。
- 开放搜索：`web.search` 是 provider-neutral 工具；模型自主生成 query、`pd/pw/pm/py` 时效窗口和可选域名范围。
  当前内置 Bocha 与 Brave adapters 使用各自固定认证 API origin；两者同时配置时优先 Bocha。结果 URL
  可以来自公开网络并登记为 `SourceType.WEB`。
  返回内容明确标记为 search-result snippet；当前不提供任意 URL fetch。
- 行情：`market.snapshot` 返回字段级快照；`market.history` 显式记录 adjusted/raw price basis 并生成收益、年化波动率和最大回撤。默认 provider 为 offline；AlphaVantage 与 Yahoo 必须显式配置，且绝不静默跨 provider fallback。Yahoo 只标记为实验适配器。
- 监管：`sec.company_facts` 使用 SEC Company Facts XBRL；`sec.recent_filings` 使用 submissions recent filings 元数据。两者要求声明组织与联系邮箱的 User-Agent。接口依据：[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。
- 宏观：`macro.fred_series` 使用 FRED 官方 series/observations API，要求独立 API key。
- 计算：`finance.calculate` 只执行白名单公式并将用户输入、公式版本和结果登记为 Evidence；账本内比率仍要求相同实体/单位/期间。
- 自拟计算：`finance.formula` 只执行声明式 AST 白名单；能保证安全与可复算，不能保证金融语义，所以 claim 标为 inferred。
- 教育解释：`finance.knowledge` 返回版本化概念、公式和解释 caveat，不依赖模型常识。

增加新数据源的标准步骤：固定 provider 客户端（含 timeout/认证）→ anti-corruption adapter → `ToolSpec` →
契约测试 → 服务注册。模型下一轮自动从动态目录看到新工具；只有希望无模型模式也使用它时，才需要把能力类别加入
`AdaptivePlanner` 基线。模型不能直接访问 provider 或读取密钥。

## 8. LLM 使用边界

LLM 负责两类受约束决策：`ModelPlanner` 每轮从动态目录选择一个工具动作或 `finish`；
`EvidenceBoundLLMSynthesizer` 基于证据生成 claims。模型不负责权限、预算、算术或引用合法性。
规划上下文包含用户请求、规则产生的 intent hints、coverage、历史动作、未解决 gaps、有限 evidence 摘要和工具输入契约；
文档、网页和记忆都明确标记为不可信数据。模型输出必须是严格 JSON，工具名和参数先过 ToolSpec/Harness。
最终 claims 必须提供 evidence IDs 与逐字 quote；失败分别产生可见的 `model_planner_fallback` 或
`llm_synthesis_fallback`，由确定性基线接管。

这不是完整的语义蕴含证明，因此后续仍应加入 NLI/人工抽检。任何模型输出都不能成为事实 claim 的唯一 source。

## 9. API 与任务

当前 `/api/v1/analyze` 和 `/api/v1/analyze-upload` 已直接使用新 Agent。请求可显式提供 entities、symbols 和 `allow_network`；网络实际开放需要请求与服务端 `MAS_ALLOW_NETWORK=true` 双重同意。上传默认 request-local；显式 `retain_for_session/use_session_documents` 才使用进程内短 TTL 页文本层；个人永久文档使用 `/api/v1/knowledge/documents`，企业库通过受控 `RetrievalSource/evidence_tools` 接入，详见 [文档生命周期设计](DOCUMENT_LIFECYCLE.md)。

API 路由使用 async 接口，但当前容器的线程执行器不可用，所以领域服务在 event loop 内同步执行。生产部署应先使用多 Uvicorn worker 做隔离；下一步将 provider 与数据库改成原生 async 后再提供单进程高并发保证。

作业状态使用 SQLite/PostgreSQL repository；Redis list worker 目前没有 visibility timeout/lease，不能宣称 exactly-once。生产应迁移到 Redis Streams、RQ/Celery 或数据库 outbox。

## 10. 安全与运维

- 上传：数量、大小、`.pdf` 后缀、PDF magic、归一化文件名和根目录约束。
- 检索：文档内容是不可信数据，不能改变系统提示或工具权限。
- 外部访问：仅固定 provider endpoint；run 默认禁网。
- 输出：无证据失败关闭，缺口可见，引用完整性和免责声明为硬校验。
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

覆盖范围包括：契约引用完整性和 content-addressed 防篡改、工具输入/输出契约、权限/分账预算/脱敏、provider 故障、无证据失败、SQLite checkpoint 恢复、持久对话/动态压缩/指代/显式删除、上下文裁剪、citation laundering、金融指标血缘、PDF 上传安全、API 鉴权/作业/上传和产物路径安全。可运行评测矩阵见 [企业级验证与故障注入报告](ENTERPRISE_EVALUATION.md)。

## 12. 后续优先级

1. 将 job queue 改成带 lease、重试和幂等键的可靠队列。
2. 增加持久化/向量化内部 corpus、文档 ACL 和索引 manifest。
3. 增加新闻、earnings call、商业行情、内部 SQL 等固定 endpoint adapter 与跨来源冲突检测。
4. 将 HTTP、数据库和 PDF 解析迁到真正可取消的异步/进程隔离边界。
5. 增加 OpenTelemetry、append-only audit、成本/token budget 和数据保留任务。
