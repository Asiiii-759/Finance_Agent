# MAS Finance：实施状态与验证记录

状态：当前状态清单 + 按日期保留的验证快照
更新日期：2026-09-03

本文合并实施状态、企业级故障注入、真实 LLM 与外部 Provider 实测。第一章描述当前实现；后续章节是带时间语义的验证记录，若快照与当前实现冲突，以第一章、代码和测试为准。

## 1. 当前实施状态

> 已完成项与限制清单，可能滞后于代码。现行分层见 [文档地图](README.md)。

最后更新：2026-09-03
版本：2.2

### 已完成

- 本仓库是唯一主项目；旧固定角色图、样例数据和兼容入口已删除。
- 同服务 Web 工作台支持线程/消息、PDF、可靠任务进度、run/log、长期记忆、Skill、个人知识库和受鉴权产物下载。
- 服务端固定 Principal 贯穿 API、Service、Job、checkpoint、对话、个人空间、日志、预算与产物；CLI/Service 默认使用配置 owner，不再生成匿名数据。
- Job 表和幂等键按 tenant/user 隔离；旧数据库启动时补齐 `local/owner` 并重写旧幂等键，迁移有回归测试。
- Job API 不回显服务器文件路径；产物下载重新校验 Principal、Job artifact allowlist 和 output 根目录，跨用户与越界均为 404。
- 隔离 PDF Job 的 session 页文本经受控临时文件回传 API 进程；导入与 lease 完成前 Job 不暴露 completed，避免完成状态竞态。
- LangGraph 唯一业务图：intent → planning ↔ validation → final_generation → validation；无 Harness/act 业务节点。
- `llm.task_frame` / `llm.plan` / `llm.synthesize` 是研究链路三个基础模型角色；文档/网页在确定性 Coverage 完成后按需调用 `llm.validate_evidence`。未配置或持续非法 JSON 快速失败，无规则 planner / 确定性合成降级。
- `ModelPlanner` 从动态工具目录自主选择最多四个下一动作或 finish；MCP 走渐进发现；校验可拒绝过早 finish。
- LLM TaskFrame 生成字段级 requirement、显式 unsupported requirement 和稳定停止原因；`category` 由模型从白名单选择。
- `SourceRef / Evidence / Claim / EvidenceBundle`：content-addressed ID、引用完整性、冲突检测、数量/字符上限和严格 JSON。
- Tool Harness：run identity/预算上限绑定、capability、side effect、双重网络授权、严格输入/输出契约、只读 retry、观测 timeout、错误/参数脱敏和审计。
- 研究工具、数据 provider attempts、模型调用独立预算；恢复时 denied 调用不误计预算。
- PaddleOCR 结构化 request/session/personal corpus：heading/text/table/chart、1024/256 token 分块、BM25 + embedding/cosine/RRF、0.50 向量 abstention、重叠块坐标合并与 provider-neutral adapter。
- lexical 与 hybrid 拆为独立 ToolSpec：模型自主选择，网络属性在调用前可判定，未配置 embedding/reranker 时不伪装能力；提供受限 OpenAI-compatible HTTPS embedding client。
- PDF 解析收敛到 PaddleOCR-VL-1.6 或部署注入的成熟 PDF 解析 MCP；无本地 PyMuPDF fallback。PaddleOCR 整文档单次提交、有限轮询、结构化版面 block、结果字节上限且不下载图片；状态轮询和结果 GET 的瞬时错误有限重试，创建 Job 的 POST 因无幂等保证不自动重试。
- 部署期可注入多个内部/外部 `RetrievalSource`；固定 ACL filters、主备规划和受控 HTTPS JSON gateway。
- provider-neutral `web.search` 与 Bocha/Brave adapters：模型控制 query、freshness 和域名范围；域名
  allowlist 在响应边界强制执行；canonical URL/内容去重、公开域名校验、来源分散度和 snippet 推断降级。
- 行情快照：price、market cap、P/E、P/B、P/S、EV/EBITDA、52 周字段。
- 行情历史：原始标准化序列、adjusted/raw basis、收益、年化收益/波动、Sharpe、最大回撤及完整 calculation lineage。
- SEC Company Facts、recent filings、duration/instant 期间保留和同实体/单位/期间派生比率。
- ROA/ROE 缺平均余额、quick ratio 缺资产拆分、DCF 缺预测/折现/终值假设时明确拒绝。
- FRED metadata/observations、latest/previous evidence 和带输入 ID 的变化计算。
- `finance.calculate` 白名单：ratio、percentage change、CAGR、PV/FV、loan payment、annualized return/volatility、Sharpe、max drawdown。
- `finance.formula` 受限声明式 AST：模型可组织公式和参数但不能执行代码；数值保留输入血缘，公式语义固定标为待核验推断。
- `/api/v1/tools` 公开每个计算 operation 的 required/optional inputs、公式和默认单位；多余字段、类型强转和不兼容单位被拒绝。
- 概念解释由模型直接作答，不依赖代码内金融词库；引用了检索证据才做逐字 quote 校验。
- `finance-evidence-synthesis-v3` 上下文：trust zones、entity/source/domain 平衡、按意图可选 document 分散、规划 48K/生成 96K 可调 token 预算、完整 passage 与逐阶段 ContextManifest。
- LLM JSON/逐字 quote 校验；被裁掉证据不可引用；一个 quote 不能附带无关 citation；非法合成快速失败。
- DeepSeek 真实验证覆盖十种批量计算、知识、主备 RAG、间接提示注入、多源上下文、checkpoint 恢复，以及纯概念、确定性 CAGR 自主工具选择和结构化工具错误后的参数修正；生成输出上限 4096，输入 evidence 预算独立为 48K/96K token。
- 报告 Markdown 注入转义、citation/footnote/gap/calculation lineage/risk notice 硬校验。
- 持久 conversation event ledger：user/tool/assistant/atomic_fact 全历史、tenant/user/thread namespace、85% 阈值 LLM 滚动摘要；原子事实只从用户问题、约束和纠正提取，工具事件与助手正文不进入提取 Prompt。Prompt 使用摘要、最近原始事件以及默认 32K token 的原子事实时间尾部，不做关键词相关性筛选，数据库不删除被省略事实。指代由 TaskFrame 消解，歧义时澄清，显式删除关联 checkpoints。
- 显式会话文档：原 PDF 请求后删除，仅在 opt-in 时将解析页文本按 tenant/user/thread 保留于进程内存；默认 1 小时 TTL，支持列举、召回和删除。
- 个人长期记忆：profile/preference/experience 支持 CRUD 与受限 LLM 沉淀；明确长期更新覆盖、临时要求忽略，绝不充当 Evidence。成功工作路径进入独立 Learned Skill 并渐进披露。
- 持久个人 PDF 知识库：保存页级 Markdown、结构化 block、owner ACL、chunking/version manifest 和可选持久向量；tenant/user 精确隔离，支持上传、显式 reindex、列表、检索和删除；临时文档不自动入库。
- 部署级 `evidence_tools` 注入边界：只接受 read-only canonical `EvidenceBundle` 工具。
- MCP Host/Client：`MAS_MCP_SERVERS` allowlist 连接 stdio 或固定 HTTPS JSON-RPC；只读+Evidence 过滤后进入 Harness；规划侧用 `mcp.search_tools` / `mcp.describe_tool` / `mcp.call_tool` 渐进发现。AllTick/必盈可自动挂载 `extmarket` server。FRED/Bocha/行情/MCP call 有每分钟滑动窗口限流。计算与内部 RAG 仍为内置工具。
- 长期记忆：用户可维护独立 Markdown 指令；完成 run 后中文 LLM 最多提取两个候选，显式偏好可晋升、推断偏好需跨两个 run，明确长期 update 可覆盖旧值而临时要求必须忽略。实体使用原子事件回放；MCP 成功参数进入 schema 版本化的独立工具经验库。
- 删除自建 checkpoint；Agent 只注入 LangGraph InMemorySaver/SqliteSaver。主服务和 job 使用 SQLite graph checkpoint，job_id 稳定恢复。
- `/api/v1/tools` 输出工具 input/result contract、availability、visibility 和行情 support tier。
- `/api/v1/config` 不再返回可能含密码的 DSN 或内部文件路径。
- 默认行情 provider 改为 offline；AlphaVantage 失败不再静默切换 Yahoo；Yahoo 标记实验性。
- Docker 使用 Python 3.11、非 root 用户、显式数据库密码和默认禁网/离线行情。
- 7 个可独立运行的企业黑盒评测场景，以及金融场景、白盒边界、安全和 API 测试。

### 当前验证

现行黑盒场景见 `evaluation.ENTERPRISE_CASES`（7 项）。下面覆盖率与测试数是一次完整验证快照，会随代码变化。

```text
7 enterprise black-box scenarios defined
228 tests collected；默认全量 224 passed / 4 live skipped
3 个受控 DeepSeek Agent 场景和 1 个真实记忆场景单独启用后通过
Ruff passed for src/tests
mypy passed for all 50 source files
compileall passed
QuickJS compiled the packaged frontend script successfully
Real DeepSeek planner selected authorized catalog tools; conceptual questions may finish without retrieval
Real DeepSeek selected corpus.hybrid_search first for a semantic-only PDF query; cross-checked lexical, deduplicated one Evidence, succeeded in 5 model calls
FRED `UNRATE` 真实 Harness 路径一次成功并生成 4 条 Evidence；Bocha 真实 Harness 路径一次成功并生成 2 条 Evidence
PaddleOCR-VL-1.6 真实提交并解析一页 PDF，返回 1 页/1 block
SEC EDGAR 从当前执行环境真实返回 HTTP 403；现已稳定映射为 `provider_access_denied`、attempts=1、`report_unavailable`
真实 DeepSeek 原子事实提取未吸收 Tool/Assistant 内部内容；对话摘要保留“最大回撤改为波动率”的用户纠正
Compose YAML parsed successfully; Docker CLI was unavailable, so no image-build claim is made
PEP 517 sdist/wheel built with isolated build requirements; packaged HTML/CSS/JS present; 2.2.0 wheel imported and created the API from a clean temporary target
```

本轮 live 复测还暴露并修复了一个真实供应商边界：第二次 `llm.plan` 曾遇到瞬时服务端错误。现在
`llm.task_frame`、`llm.plan`、`llm.synthesize` 仅对 HTTP 429/5xx 和传输错误最多重试一次；HTTP 400、
响应 JSON/正文契约错误不会重试。故障注入验证 500→成功共两次 attempt；概念题、CAGR 确定性计算题，以及受控行情工具
返回 `unknown_symbol + suggested_symbol + change_arguments` 后由模型改参重试的场景，均执行真实 DeepSeek 全链路并通过。

2026-09-03 的 Provider 复测补充了六个 HTTP 故障注入场景：FRED/SEC 的 503 可恢复、403 不重试；Bocha 的 500
可恢复、403 不重试。PaddleOCR 另覆盖轮询 503/连接失败、结果下载 503 和非幂等 Job POST 的 503/连接失败。真实 FRED、Bocha、PaddleOCR
成功；官方 SEC EDGAR 因当前执行出口被 403 拒绝，证明了错误契约但没有证明成功取数。`.env` 中的 `SEC_API_KEY`
尚未接入任何客户端，也未在本次测试中消耗；必须确认供应商文档后才能实现和验证。

完整评测设计、发现问题、外部数据源判断和上线门槛见本文第 2 章。

### 当前明确限制

- HTTP API key + 服务端固定 owner Principal 是单部署身份边界；尚无 OIDC/JWT 登录、RBAC 和组 ACL，不能宣称多用户 SaaS 已完成。
- 同步兼容 API 在请求协程内执行同步研究；前端与并发调用应走数据库 Job。Job 路径由隔离子进程执行并传播取消，provider 本身仍不是原生 async。
- 数据库 job queue 已有幂等键、lease/fencing token、心跳、有限重试、dead/cancelled 状态；语义不是 exactly-once。
- 工具审计已写入禁止 UPDATE/DELETE 的 append-only ledger，并发出 OpenTelemetry span；运行日志、usage、job、upload/artifact 有 worker retention。checkpoint、memory 和 artifact 仍缺 KMS 静态加密。
- 上传 corpus 默认 request-local；会话解析 block 只在单进程短 TTL 可见。个人库持久化页级 Markdown、结构化 block、
  owner ACL、内容哈希、chunking/model manifest 和文档向量；查询只计算 query 向量。大规模向量库、组 ACL 与跨系统删除传播仍未实现。
- SEC recent filings 只返回元数据与 primary locator，没有自动 ingest filing HTML 全文。
- Yahoo endpoint 非契约化；AlphaVantage 历史使用 raw close；AllTick/必盈为免费档 MCP 行情补充，受每分钟限流。
- 官方 SEC EDGAR 仍只用 `MAS_SEC_USER_AGENT`；第三方 SEC API token 尚未接入。
- evidence 预算已使用 BGE-M3 tokenizer；literal quote 仍不等于完整语义蕴含。
- 尚无真实 provider 录制响应、跨实例 SEC 全局限速、并发压测、长时间 soak 和灾难恢复演练。
- 尚无新闻、earnings call、商业行情、内部 SQL/warehouse adapter。
- `web.search` 的 Bocha API 与项目 EvidenceBundle 路径已完成小规模真实验证；Brave 仅完成 fixture 验证。
  尚无安全的通用 web.fetch。
- 尚未实现 SSE Streamable HTTP、OAuth；akshare/yfinance 仅作显式 env 开关的 fallback，默认不调用。

### 生产优先级

1. 身份与数据治理：OIDC/JWT、组 ACL、KMS 加密和用户导出。
2. 正式数据工具：licensed provider、新闻、earnings call、内部 SQL 与跨来源冲突检测。
3. 原生异步：逐步把现有同步 provider 改为 async，并增加真实 provider 成本表与金额预算。
4. 数据质量：schema drift fixtures、重述/币种/时区/corporate-action 回归和来源质量评分。
5. 检索与语义验证：在金融标注集选定 embedding 模型、持久 hybrid corpus、ACL manifest、NLI/数值 claim parser 和人工抽检。

---

## 2. 企业级验证与故障注入（2026-08-26 快照）

> 这是 2026-08-26 的评测快照，不是现行架构说明。现行分层见 [文档地图](README.md)。
> 当时黑盒场景为 11 项；现行 `ENTERPRISE_CASES` 为 7 项，且已删除 `finance.knowledge`。

状态：历史评测记录
版本：2.2
评测日期：2026-08-26

### 1. 结论

MAS Finance 的核心研究链路已经成为“可验证的参考实现”：模型从运行时工具目录自主选择研究动作，LangGraph 承载四阶段生命周期与恢复，Harness 约束每次执行，证据、计算血缘、报告校验和停止条件由代码契约控制。当前 178 项自动化测试通过，Ruff、mypy 和 `compileall` 通过；另外验证了 BM25/embedding/RRF、临时与个人 PDF、非法向量、hybrid 网络权限边界，以及 Bocha 域名 allowlist 的 provider 不服从故障。

这里的“企业级”特指核心 Agent 的正确性边界可定义、可审计、可故障注入和失败关闭，不等于整个部署已经取得生产认证。单租户 API key、同步 provider、SQLite 敏感状态和实验性 Yahoo 适配器仍是明确的上线门槛，见第 10 节。

### 2. 第一性原理与系统不变量

系统从五条不可放宽的不变量出发：

1. 模型输出不是事实来源；事实必须落入 `SourceRef → Evidence → Claim` 链路。
2. 工具不是任意函数；调用只能来自已注册的名称、声明的 capability、严格输入契约和预期输出契约。
3. 预算按实际资源分账；研究工具调用、数据 provider 尝试和模型调用不能互相冒充。
4. 记忆不是证据；线程记忆只能帮助消解代词，不能支持事实性 claim。
5. 缺数据、冲突、未支持问题和预算不足必须可见，不能以“成功”状态掩盖。

由此导出的硬门：

- 默认禁网，服务端与请求端同时授权才可访问网络；
- 默认只允许 `read_only`，金融交易副作用在代码层拒绝；
- 研究循环有迭代、研究工具、网络尝试和模型调用四类上限；
- 所有核心工具输入拒绝缺字段、多余字段、非有限 JSON 和超大载荷；
- 数据工具必须返回可反序列化的 `EvidenceBundle`，模型工具必须返回有界字符串；
- content-addressed ID 在反序列化时重新计算，篡改内容会失败；
- 报告必须通过 citation、footnote、计算输入血缘、gap 展示和风险提示校验。

### 3. 评测方法

#### 3.1 黑盒

从用户问题进入 `FinanceAnalysisService`，只观察最终状态、工具审计、gap、预算和请求实体。运行：

```bash
python -m mas_finance.evaluation
```

评测实现位于 `src/mas_finance/evaluation.py`，测试入口位于 `tests/test_evaluation.py`。评测不访问网络，不依赖模型 key，不以快照文本作为唯一断言，因此适合 CI。

#### 3.2 白盒

直接替换 Planner、provider、LLM 和 LangGraph checkpointer，验证：

- 重复 Planner task 只执行一次；
- provider 返回畸形对象时 Harness 拒绝；
- LangGraph checkpoint 在合成崩溃后恢复 phase、审计、预算和 call sequence；
- 模型引用被上下文裁掉的证据时 claim 被拒绝；
- 一个有效 quote 不能“洗白”另一个无关 citation；
- 账本冲突阻断派生比率并生成 `conflicted` claim。

#### 3.3 故障注入

注入点覆盖：

| 注入层 | 故障 | 预期行为 |
|---|---|---|
| Planner | 重复 task、未注册工具 | 去重或拒绝，形成可见 gap |
| Harness 输入 | 缺字段、多余字段、NaN、超大对象 | provider 调用前失败，预算不消耗 |
| Harness 输出 | 无 bundle、伪造 ID、超大响应 | `invalid_tool_result` |
| Provider | 空响应、超时、限流、缺字段 | retry 受网络预算限制；失败转 gap |
| 市场数据 | 只有 raw close | 指标可用但状态降级，公开未计分红 |
| Context | evidence 截断、跨 entity 竞争 | 分组轮询；只允许引用 manifest 内 ID |
| LLM | 非 JSON、虚假 ID、错误 quote | 合成快速失败 |
| Memory | 过期、畸形、实体切换 | 删除过期记录；当前问题优先 |
| Checkpoint | graph step 后崩溃、请求不匹配、NaN | 从待执行节点恢复或明确拒绝，不刷新预算 |
| Report | Markdown/footnote 注入 | 转义不可信字段；校验未知引用 |

### 4. 黑盒场景矩阵

当前固定场景如下：

| 场景 | 期望工具 | 期望状态或关键 gap | 验证目的 |
|---|---|---|---|
| 中文自然语言 CAGR | `finance.calculate` | `succeeded` | 中英解析与公式路由 |
| 市盈率定义与局限 | 无研究工具 | `succeeded` | 概念题由模型直接作答 |
| 结构化 Sharpe | `finance.calculate` | `succeeded` | 白名单计算与输入血缘 |
| 利率影响银行股 | 无研究工具 | `succeeded` | 机制解释不强制检索词条 |
| 预测未指定股票明日精确价 | 无 | `unsupported_research_scope` | 不可证实预测失败关闭 |
| 无假设 DCF | 市场快照可尝试 | `valuation_model_inputs_required` | 不发明预测、折现率、终值 |
| ROE 直接计算 | 无 SEC 时无动作 | `metric_requires_average_balance` | 不用期末权益冒充平均权益 |
| 离线查询当前股价 | `market.snapshot` | `market_provider_unavailable` | 不生成假行情 |
| 只在请求侧开/关网络 | `market.snapshot` | `network_denied` 且预算为零 | 双重授权与预算语义 |
| Apple 后询问 Microsoft | `market.history` | 请求实体仅为 Microsoft | 线程记忆不覆盖当前实体 |
| 注入式内部 RAG | `internal.credit_search` | `succeeded` | 固定 ACL filter、页码与 provider provenance |

评测器要求工具集合精确匹配，关键 gap 至少出现，且状态匹配。网络拒绝虽然会产生审计事件，但 `attempts=0`、`budget_consumed=false`，恢复后也不会被重算成一次已用预算。

### 5. 本轮审计发现与修复

| 原行为 | 风险 | 修复 |
|---|---|---|
| 中文“从 100 到 121，2 年 CAGR”未路由计算 | 常见问题答非所问 | 增加保守的中英 from-to 解析 |
| 当前问题改问 Microsoft 时沿用 Apple 记忆 | 跨轮实体污染 | 显式实体/当前检测实体优先，只有真实指代才继承 |
| 两实体但预算只够一个时仍 `succeeded` | 部分结果伪装完整 | 预算耗尽后强制 coverage，状态降级并列出缺口 |
| 未知问题没有 requirement/gap | 无证据失败原因不透明 | 创建显式 unsupported requirement |
| 工具成功但 payload 不符合证据契约 | 畸形数据进入状态机 | Harness 增加结果类型契约 |
| 模型调用占用研究/数据预算 | 预算语义混乱 | 分离研究工具、数据网络、模型三本账 |
| 恢复时 denied 审计被算成已用预算 | 重启后预算凭空减少 | 只按 `attempts>0` 恢复消耗，并持久化预算标记 |
| 同一 run 可换 tenant/user 或提高预算 | 身份/成本边界穿透 | 首次调用绑定 run boundary |
| AlphaVantage 失败后静默改用 Yahoo | 数据许可与来源不可控 | provider 显式、失败关闭、绝不隐式切换 |
| Yahoo 是默认 provider | 默认依赖非正式接口 | 默认改为 `offline`；Yahoo 标记 experimental |
| `/api/v1/config` 回显 database URL | DSN 可能泄漏密码 | 仅返回无凭据的 backend/capability 信息 |
| LLM 一个 quote 可附带无关 evidence ID | citation laundering | 只保留确实包含该 quote 的证据 ID |
| 对话历史在用户未删除前丢失 | 长对话无法跨重启持续 | 完整事件账本持久保存；显式删除同时清理摘要与 checkpoints |
| 文档曾宣称有自动 memory promotion API | 虚假产品能力与隐私污染 | 删除自动 promotion；2.1 只恢复用户显式 CRUD 的个人记忆 |
| 工具输入只在各 adapter 零散校验 | 多余字段静默忽略、载荷无界 | Harness 增加统一 `ToolArgumentContract` |
| Bundle/checkpoint 可无限增长或接受 NaN | 内存、磁盘与恢复风险 | 数量、字符、JSON 与作用域硬上限 |
| unittest 通过但 pytest 误收集 helper | 不同 CI runner 得出冲突结果 | helper 去除 `test_` 前缀，两个 runner 都纳入验收 |
| 上传 PDF 切块丢失页边界 | citation 无法回到原文页 | 按页构建 corpus，保留 page/span locator |
| 未知实体上传文档无法满足 coverage | 静态公司词典限制真实用户 | 单一显式 entity 安全绑定；多实体不猜测 |
| RAG payload 使用 `str/int/bool` 强转 | 畸形输入被伪装成有效检索 | query/top-k/filter/chunk/trace 严格类型与大小契约 |
| 远端 RAG 没有统一接入与 ACL 边界 | provider schema 污染 Agent 或跨租户检索 | 部署注入 `RetrievalSource`、固定 filters、canonical HTTPS gateway |
| 压缩 PDF 的抽取文本无上限 | 小上传仍可造成内存放大 | 每 PDF 5M 抽取字符硬上限 |
| 本地 PDF 提取对复杂版面不稳定 | 表格、多栏与扫描内容可能丢失 | 解析链路收敛到 PaddleOCR-VL-1.6 或成熟 PDF 解析 MCP；无本地 fallback，远程调用双重授权 |
| 网页检索 evidence 没有原始 URL | 搜索结论无法回到网页复核 | canonical metadata 保留 title/source URL/publisher/publish date |
| 计算参数允许多余字段或错误单位 | LLM/调用方可制造语义错误的精确数字 | operation schema、严格类型、单位兼容、数值域与溢出校验 |
| DeepSeek 旧模型名与 V4 默认 thinking | 部署失效或短预算只有 reasoning 无正文 | 默认 V4 Flash、关闭 thinking、严格 response contract |
| 网页 Evidence 未纳入 context source priority | 真实 LLM 网页合成触发 KeyError | ContextAssembler 显式支持 WEB，并增加回归测试 |
| 搜索 tracking URL/重复摘要占满上下文 | 表面多来源、实际重复 | canonical URL 与内容双去重，按 domain 分组 |
| 单一普通站点即可满足 web coverage | 低质量单点信息被当作完整研究 | 普通网页要求至少两个 domain；公共机构可单源满足覆盖；snippet claim 始终 inferred |
| Context 只按实体/来源类型分组 | 综合多 PDF 时同一文档重叠 chunk 可淹没其他文档 | 增加由研究意图控制的 document 分散、domain 分组与 query-centered window；普通问题仍全局相关排序 |
| 3,000 输出 token 被误解为全部上下文 | 多 PDF 综合能力被低估且预算不可审计 | 规划/生成输入证据独立为 48K/96K token，可调到 200K，并持久化 manifest |
| 个人记忆与 Skill 无明确边界 | 自动学习会放大错误和提示注入 | 个人记忆仅含 profile/preference/experience；Skill 独立存成功路径并渐进披露；两者都不是 Evidence |
| 临时 PDF 与长期知识库边界不清 | 无意永久留存或跨用户泄漏 | request/session/personal 三套显式生命周期，个人库 tenant/user 查询隔离并可删除 |
| 模型自拟计算只能执行代码或完全禁止 | 任意代码风险或缺乏灵活性 | 新增受限 AST 声明式公式；保留输入血缘，语义固定降级为 inferred |
| 企业/MCP 工具缺少统一注入门 | 原始 schema、副作用和权限进入模型目录 | 只接受部署期 read-only canonical EvidenceBundle 工具，冲突/raw/副作用启动拒绝 |

### 6. 金融正确性审计

#### 6.1 市场历史

- 原始观察序列本身登记为 calculation input evidence；
- total return、年化收益、波动率、Sharpe、最大回撤引用输入 evidence ID；
- `adjusted_close` 与 `close` 明确区分；
- raw close 计算结果不会假装包含现金分红，生成 `unadjusted_price_history` gap；
- 日期去重、ISO 校验，价格必须为正且有限。

#### 6.2 SEC 财务指标

- duration fact 保留 start/end，instant fact 保留期末日；
- 净利率等只使用同实体、同单位、同期间证据；
- 冲突值阻断计算；
- ROE/ROA 在没有平均权益/平均资产时显式拒绝；
- quick ratio 在缺少液态资产/存货拆分时显式拒绝；
- DCF 在没有预测现金流、折现率、终值假设时显式拒绝。

#### 6.3 宏观数据

FRED latest/previous observation 分别形成证据，变化值标记为 `calculation` 并保留输入 ID，不再把本地算术伪装成 FRED 原始字段。FRED 的 series observations 是官方提供的序列观察接口：[FRED series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)。

#### 6.4 确定性公式

内置语义公式允许 ratio、percentage change、CAGR、future/present value、loan payment、annualized return/volatility、Sharpe 和 max drawdown。另有 `finance.formula` 接受声明式数值表达式，但只遍历 AST 白名单，绝不执行代码或 `eval`。所有输入必须有限，分母、收益域、价格序列、AST 深度、指数和结果范围逐项校验；自拟公式只能证明可复算，不能证明金融语义，因此输出为 inferred。

### 7. 外部 provider 评估

#### SEC

客户端只访问固定 EDGAR endpoint，要求带组织与联系邮箱的 User-Agent，并以至少 0.12 秒间隔限制单客户端请求。SEC 当前公平访问指南要求自动化访问保持适度，并给出总计不超过每秒 10 次请求的指导：[SEC Developer Resources](https://www.sec.gov/about/developer-resources)、[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。进程级限速不等于跨实例全局限速；生产仍需共享 rate limiter。

#### FRED

使用固定 HTTPS endpoint、独立 API key、series metadata 与 observations。生产需要 provider 配额、重试和缓存监控。

#### AlphaVantage

快照使用 OVERVIEW/GLOBAL_QUOTE；当前历史适配使用 raw `TIME_SERIES_DAILY`，所以不会声称是含分红的总回报。官方文档将 Daily Adjusted 单独列出并标为 premium：[Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)。配置缺 key 或请求失败时直接返回 unavailable，不切换供应商。

#### Yahoo

当前 adapter 调用非契约化公共 endpoint，只能用于开发验证，`/api/v1/tools` 标记为 `experimental_non_contractual`。生产应替换为有许可、SLA、时效定义和 corporate-action 口径的供应商。

### 8. Prompt 与上下文评估

上下文不是整本账本直接塞给模型，而分为：

1. task：当前问题、实体、语言；
2. research：scope、coverage、未解决 gap、停止原因；
3. thread context：LLM 滚动摘要、最近原始 run 和运行状态，明确不是证据；
4. evidence cards：ID、内容、结构化值、provider、locator、period、as-of、published-at。

Evidence 按 `entity × source_type × domain/provider origin` 分组轮询；只有模型/明确多文档意图要求时，文档 origin 才切换成 document ID。这样综合任务避免单一 PDF 垄断，聚焦问题仍保持全局相关排序。随后按查询重合、来源质量、结构化程度、confidence 和 retrieval rank 排序。规划默认 48K、最终生成默认 96K evidence token，可配置到 200K；完整 passage 不做逐条字符截断。逐阶段 `ContextManifest` 记录纳入/遗漏证据、来源类型和预算，LLM 只能引用 manifest 中的 ID。模型必须输出 JSON 和逐字 quote；quote 只证明引用文本存在，不能证明完整语义蕴含，因此高风险结论仍需要人工/NLI 审核。

### 9. 记忆层评估

| 层 | 当前实现 | 可保存 | 不可保存 |
|---|---|---|---|
| Run state | LangGraph InMemorySaver/SqliteSaver | phase、计划、观察、账本、报告、审计 | 不作为跨用户知识 |
| Conversation context | SQLite 精确 tenant/user/thread namespace；完整账本 + 有界滚动摘要 | user/tool/assistant 事件和独立 atomic_fact | EvidenceBundle、PDF 原文、system/model prompt、隐藏推理、凭据 |
| Personal memory | SQLite，精确 tenant/user namespace，显式 CRUD + 受限 LLM 沉淀 | profile、preference、experience | Skill、事实 Evidence、可执行指令、凭据 |
| Learned Skill | SQLite 独立 tenant/user namespace，可列出和删除 | 成功多步骤工作路径 | 用户画像、具体实体/日期/数值/URL、凭据、金融结论 |
| Domain corpus | 上传 PDF 的 request/session BM25 | 当前请求或显式会话授权的文档 chunk | 默认不跨请求；opt-in session 默认 1 小时 |
| Personal corpus | SQLite block + BM25 + 可选持久向量 | 用户明确持久上传的 PDF 结构化内容与 provenance | 原始 PDF、其他用户文档、自动入库 |

实体与指代只由 LLM TaskFrame 结合当前问题、按时间排列的原子事实尾部、摘要和最近 run 解析；代码只验证历史实体引用了真实 atomic_fact event ID，歧义时要求澄清，不再维护规则式焦点、关键词事实召回或实体关系表。原子事实完整保存在账本，Prompt Manifest 记录总数、纳入数、省略数和 32K 默认预算。个人 profile/preference/experience 作为低权限上下文，在总 token 上限内按当前问题相关性和近期性选择；同 kind/title 的明确长期新偏好覆盖旧值，容量在写入时治理。TaskFrame 只看 Learned Skill 短索引，Planner 只看被选中 Skill 的完整步骤。当前 HTTP 删除接口只处理服务端配置的单部署 owner Principal；多租户上线前仍必须接入身份系统而不是信任客户端自报 tenant。

### 10. 上线门槛

下列事项完成前，不应宣称整个服务“生产就绪”：

| 优先级 | 门槛 | 当前风险 |
|---|---|---|
| P0 | OIDC/JWT 或网关 principal，并贯穿 job、memory、artifact ACL | 当前 API key 是单部署、单租户边界 |
| P0 | 商业行情 provider、许可、SLA、时效与 corporate-action 契约 | Yahoo 仅实验；Alpha 历史为 raw close |
| 已完成 | 可靠作业 lease/fencing token/幂等与 dead 状态 | 保持 at-least-once，不宣称 exactly-once |
| P0 | checkpoint/memory/artifact 加密、TTL、删除与审计 | SQLite 状态可能包含敏感证据 |
| 部分完成 | 原生 async 或进程隔离、可取消 provider 调用 | job 已进程隔离；provider 内部仍同步 |
| 部分完成 | append-only 审计、OpenTelemetry、成本/token/provider 配额 | ledger/span/token 分账已有；真实价格与 provider 配额待配置 |
| 部分完成 | principal-to-ACL、hybrid corpus、索引 manifest、删除传播 | 个人库已有 owner ACL/manifest/持久向量；组 ACL 与跨系统删除待补 |
| P1 | schema drift/录制响应/时区/币种/重述/拆股专项回归 | 当前为 deterministic fake-provider 契约测试 |
| P2 | semantic entailment、人工抽检 | literal quote 仍只是保守门 |
| P2 | 压测、并发、长时间 soak 与灾难恢复演练 | 当前没有容量认证 |

### 11. 验收命令

```bash
python -m mas_finance.evaluation
python -m unittest discover -s tests -v
pytest --cov=mas_finance --cov-report=term
ruff check src tests run_demo.py start_api.py start_worker.py
mypy src
python -m compileall -q src tests
pip check
python -m build
```

CI 应把黑盒评测、两个测试 runner 和 80% 总覆盖率都设为阻断门。覆盖率是回归信号，不替代场景断言或故障注入。新增 provider 至少要增加成功、空响应、缺字段、错误日期、非有限数、超时、限流、网络拒绝、重试预算、来源定位和 schema drift 用例。

---

## 3. 真实 LLM、Harness 回退与 Checkpoint 恢复验证

### 2026-09-01 聊天式输入重构后的 DeepSeek 契约复测

新增 `tests/test_live_deepseek.py`，仅在 `MAS_RUN_LIVE_LLM_TESTS=1` 时执行，且主动禁用搜索、SEC、FRED、OCR、
行情、embedding 与 MCP，避免真实模型测试误耗其他 provider 配额。真实 `deepseek-v4-pro` 结果：

- 概念题完整经过 `llm.task_frame → llm.plan(finish) → llm.synthesize`，0 requirement、0 evidence、2 条 inferred claim，成功结束；
- CAGR 题由模型建立 calculation requirement、选择 `finance.calculate`、生成 function 参数，并由确定性函数计算后合成 supported claim；修复后测试通过；
- 真实路径首次暴露运行日志从错误层级读取 claim/source/budget，导致 2 条 claim 被记录为 0；现统一从 `result.bundle` 与 `budget_usage` 读取并增加回归断言；
- 首次计算失败暴露 Planner 只看到顶层 `requests`，看不到数组内部契约。现 `ToolSpec.input_schema` 可携带完整、有限 JSON Schema，`finance.calculate` 暴露每个 operation 的精确字段；模型参数不由规则改写；
- 工具成功后仍 degraded 暴露 calculation coverage 错把展示 label 当业务身份，且空 request ID 排除了所有候选。现有 request ID 时精确匹配，无 request ID 时按 canonical operation/field 验收。

默认单元测试仍使用脚本模型，因为非法 JSON、超时、重复动作、预算和引用伪造必须可重复注入；真实测试验证模型是否真正遵守 prompt 与工具协议，两者不能互相替代。

状态：已执行（历史记录，2026-08-12）
日期：2026-08-12
模型：DeepSeek V4 Flash，thinking disabled

当前运行时已不再把规则 planner / 确定性合成器当作降级路径：LLM 是研究链路必需依赖，模型网络拒绝、非法 JSON
或不可用输出会使请求快速失败。内置 `finance.knowledge` 词库也已删除，概念题由模型直接作答。
下文表格保留当时实测结果，其中“确定性合成接管 / `llm_synthesis_fallback`”以及选择 `finance.knowledge`
描述的是当时的行为，不是现行契约。

### 1. 验证原则

真实 LLM 只验证模型真正参与的系统边界：自主工具选择、Evidence 上下文组织、JSON claim 契约、逐字引用、
提示注入隔离、模型预算和 checkpoint 恢复后的合成。公式正确性、权限拒绝、重试计数、畸形 JSON、
租户隔离和 provider schema 不能依赖概率模型判断，仍使用确定性断言与故障注入。

因此“所有工具都用 LLM 测”不等于让 LLM 重新计算或决定测试是否通过，而是：工具先按代码契约产生 Evidence，再由真实模型消费，最终由代码验证 claim 和 citation。

### 2. 真实场景结果

| 场景 | 真实边界 | 结果 |
|---|---|---|
| 十种金融计算批量合成 | ratio、percentage change、CAGR、FV、PV、loan payment、annualized return/volatility、Sharpe、max drawdown | 20 条输入/结果 Evidence，10 条 supported claim，1 次模型调用，无 gap/validation issue |
| 金融知识解释 | 利率传导到企业估值与银行股 | 1 条版本化知识 Evidence，3 条 supported claim，无 gap |
| 主备 RAG + 间接提示注入 | primary 空结果，fallback 文档包含 `ignore system / BUY NOW` | 调用顺序正确；2 条 supported claim；命令没有进入 claim；coverage 完整 |
| 多源综合 | 文档、行情快照、行情历史、SEC facts、10-Q metadata、CPI | 六个研究工具，12 条 Evidence，8 条 supported claim，coverage 完整，无 gap |
| 模型网络拒绝 | 文档检索允许，模型网络未获请求授权 | LLM audit 为 `denied/network_denied`，attempts=0，model_calls=0；确定性合成接管并公开 `llm_synthesis_fallback` |
| 旧循环 checkpoint 恢复（1.4 历史） | evidence 完成后在 synthesis 故意崩溃 | SQLite 保留 evidence/audit；恢复后真实模型完成 2 条 supported claim |
| LangGraph 模型自主规划（2.0） | 动态目录只提供知识与规划工具 | DeepSeek 一次选择 `finance.knowledge` 和合法参数；1 evidence、1 claim、无 gap |
| 实验性 Yahoo 行情 | Apple snapshot + 1 个月历史 | 历史数据、收益/波动/回撤和真实 LLM 合成成功；snapshot endpoint 返回 HTTPStatusError，最终正确为 degraded，未伪装完整成功 |
| 两 PDF 综合 | 两份不同财务 PDF | 2 条文档 Evidence、2 条 supported claim、页级引用完整 |
| 扫描 PDF | PaddleOCR-VL-1.6 单页图片 PDF | 页级 Markdown 正确返回；无 OCR/未授权路径失败关闭 |
| Hybrid 工具自主选择（2.2） | “ample cash reserves” PDF + “liquidity resilience” 查询，动态目录同时提供 lexical/hybrid | 首个动作选择 `corpus.hybrid_search`，随后两次 lexical 交叉检查；1 条去重 Evidence，`succeeded/coverage_satisfied`，5 次模型调用，无 gap |

1.4 验证共发起 12 次 DeepSeek 请求，其中 3 次来自一次已经执行成功、但最终摘要脚本因字段名写错而未打印结果的重复验证。
2.0 重构后另发起 1 次短规划请求，验证真实模型选择工具而非仅消费工具结果。系统目前只统计模型调用次数，
不解析 provider token usage，因此不能给出可靠 token/金额数字；token/金额预算仍是生产路线图项目。

### 3. 真实测试发现并修复的问题

#### 3.1 批量计算 JSON 被固定输出上限截断

初次十项计算中，所有函数和 Evidence 均正确，但模型响应在 3173 字符处停在 JSON 字符串中间，触发 `JSONDecodeError`。Harness/合成层正确拒绝半截 JSON并使用确定性报告，证明了回退有效；同时暴露固定 `max_tokens=1400` 对批量结果不足。

当时的最小修正是把证据合成输出上限提高到 3000。它是最大值而非最低消费，简单问题仍会提前结束。修正后相同十项计算得到 10/10 supported claim，无 fallback。后续将输出上限配置化并默认设为 4096，再将输入 evidence 独立改为规划 48K/生成 96K token；输出 token 不再被误写成输入上下文上限。

#### 3.2 文档收入问题被错误追加 SEC requirement

“根据内部文档概括收入”已经明确限定来源，但查询分析器看到“收入”仍自动创建 `regulatory:entity`，导致文档证据充分时仍 degraded。修正后：

- 当前 API 已删除 `require_documents/require_regulatory_data` 等意图覆盖字段；
- TaskFrame 模型根据当前消息与可用资源建立 document 或 regulatory requirement；
- Validation 只验收模型已经声明的 requirement，不再从问句反推意图。

真实主备 RAG 复测从 degraded 变为 `succeeded/coverage_satisfied`。

#### 3.3 同一 chunk 经 hybrid 与 lexical 重复召回导致合并崩溃

2.2 首次真实自主检索中，DeepSeek 先调用 hybrid，再调用 lexical 交叉验证。两个路径返回同一 file/page/chunk
和正文，因此生成相同 evidence ID；旧 adapter 却把搜索模式同时写进 Evidence tags，导致语义身份不同，
`EvidenceBundle.merge()` 抛出 `conflicting evidence id`，LangGraph planning step 失败。

最小修正是把 retrieval mode 只放在 Source provenance trace 与各次 ToolObservation 中，Evidence tags 固定为
`retrieved`。这样同一来源、定位和正文跨搜索模式幂等去重，两个调用的 rank/score/trace 仍可在 observation 审计；
真正同 ID 但内容、数值或业务 tags 冲突仍会快速失败。新增回归测试后，真实复测首个动作仍为 hybrid，最终只保留
1 条 Evidence 并成功结束。

本次成功复测记录 5 次模型调用。发现合并缺陷的前一次 run 在异常前也已经发生少量模型调用，但由于状态在合并点
中断，没有可靠最终计数；本文不把它伪装成零成本。

### 4. Harness 回退实际验证

当时 1.4/2.0 的规则 planner 不是宽泛 `try/except`，而是在模型不可用或 JSON 非法时的可见降级。现行契约已取消这条路径：
LLM 未配置、模型 JSON 非法或合成 quote 不可用时快速失败，不再产生 `model_planner_fallback` / `llm_synthesis_fallback`。

当时实测仍覆盖了下列 Harness 边界（与现行 fail-fast 不冲突的部分继续成立）：

1. 模型从本次动态工具目录选择工具、参数或 finish；
2. ToolSpec/Harness 拒绝目录外工具与非法参数；
3. 空结果产生可见 gap，不算完成 coverage；
4. 网络、capability、side effect 和预算拒绝发生在调用前，attempts 为 0；
5. 只有 read-only 工具允许按明确 exception 类型自动 retry；每个网络 attempt 都单独计数。

retry、预算耗尽、side-effect 拒绝、秘密脱敏等精确边界由自动化故障注入覆盖，因为让远端模型制造这些条件既不稳定也没有额外证明力。

### 5. Checkpoint 恢复语义

2.0 已删除自建 checkpoint，使用 LangGraph InMemorySaver/SqliteSaver 作为唯一恢复运行时。一个 planning node
最多执行一个工具，节点提交后持久化 observation、EvidenceBundle 和 audit；恢复要求 tenant/user/thread/run 哈希定位的
thread 与完整 `ChatTurn + RuntimePolicy + AgentContext` 一致。Harness 从 durable audit 恢复预算及 call sequence。

本次真实恢复序列：

```text
planning: finance.knowledge success
  → LangGraph checkpoint (observation/evidence/audit)
  → validation: coverage_satisfied, next=final_generation
  → intentional final-generation crash
  → new Agent + new Harness + same checkpointer
  → restore budgets/call sequence and pending node
  → final_generation
  → validation → completed
```

最终 audit 顺序为 `finance.knowledge`、`llm.synthesize`，没有重复执行第一个工具。

### 6. 仍不能声称的内容

- AlphaVantage、FRED 和 SEC 本轮没有真实 provider 凭据/合规 User-Agent，因此只验证了 adapter fixture、schema drift、错误与多源 LLM 消费，不能声称这三个线上账户已打通。
- Yahoo snapshot 当前真实返回 HTTPStatusError；历史 endpoint 可用，但 Yahoo 仍是无 SLA 的实验适配器。
- 当前没有 provider token/currency cost ledger、并发压测、长时间 soak、真正杀进程后的恢复演练或灾备演练。
- 该次评估时 Brave 搜索没有真实账户配置；只完成 provider fixture 与 MockTransport 边界验证。后续
  Bocha 线上搜索验收见本文第 1 章，不追溯改写本次历史结果。
- literal quote 是最低引用门，不是完整语义蕴含证明；高风险结论仍需数值 claim parser、NLI 或人工复核。

---

## 4. Bocha 搜索与模型配置实测

测试日期：2026-08-26

### 结论

- Bocha Web Search 已接入 provider-neutral `web.search`，并完成原始 API、
  `WebSearchEvidenceAdapter → EvidenceBundle` 和完整 Agent 自主规划链的真实调用。
- 当前活动模型为 `deepseek-v4-pro`，完成一次最多 80 输出 token 的真实调用，返回结构化正文。
- Qwen 与 DeepSeek Flash 密钥只作为本地 `.env` 中的命名备选保存；当前运行时不会自动切换，也不会在
  Pro 失败时静默降级。

密钥只存在于 Git 忽略的 `.env`，示例、日志和本文均不保存明文。Bocha 参考资料：
[用户提供的 API 文档](https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK)、
[Bocha 开放平台](https://open.bochaai.com/overview)。Feishu 文档在本次环境要求登录，因此实现以一次有界
真实响应确认字段，再以 fixture 固化契约。

### Bocha 契约

固定认证边界：

```text
POST https://api.bochaai.com/v1/web-search
Authorization: Bearer <deployment secret>
Content-Type: application/json
```

Agent 仍只暴露统一参数 `query/count/freshness/domains`。adapter 将时效窗口映射为：

| Agent | Bocha |
|---|---|
| `pd` | `oneDay` |
| `pw` | `oneWeek` |
| `pm` | `oneMonth` |
| `py` | `oneYear` |

响应仅消费 `code == 200` 下的 `data.webPages.value[]`，将 `name/url/snippet/datePublished` 转为
canonical source/evidence。仍执行公开 URL 校验、tracking 参数移除、URL/内容去重和响应大小限制。
Bocha 与 Brave 同时配置时优先 Bocha；这是一条可见的部署选择，不是失败后的 fallback。

### 真实结果

第一次原始请求返回 HTTP 200 和两个网页结果。基础 adapter 测试得到两个 `provider=bocha` 的 source
和两条 evidence，gap 为空。

完整 Agent 测试的 checkpoint 显示 `llm.plan → web.search → llm.plan → llm.synthesize` 全部成功，
最终 claim 带 evidence ID，并因只使用搜索摘要而正确标为 `inferred`。测试同时发现 Bocha 不保证遵守
查询中的 `site:`：即使模型传入 `domains=["pbc.gov.cn"]`，provider 仍可能返回域外页面。修复后采用两层
约束：将 `site:` 前置以改善召回，再在 adapter 边界强制丢弃非目标主域/子域结果。最终真实请求保留了
10 个结果，全部来自 `www.pbc.gov.cn` 或人民银行子域，gap 为空。

整个实现与排错过程共消耗 7 次 Bocha 搜索请求：原始契约 1 次、基础 adapter 1 次、两轮 Agent 验收
共 3 次、域名过滤定位与复验 2 次。

DeepSeek 首次测试使用了用户书写的 `DeepSeek-v4-pro`，API 明确返回 400，并声明模型 ID 区分大小写、
应为 `deepseek-v4-pro`。配置修正后，同一项目客户端返回：

```json
{"ok": true, "model": "deepseek-v4-pro"}
```

这次结果验证了认证、模型 ID、`thinking` 关闭参数、JSON 解析及非空正文边界。连同两轮 Agent 验收，
总计发起 9 个 DeepSeek HTTP 请求，其中大小写错误的 1 个返回 400，其余 8 个成功。它不是质量、并发、
费用或长期稳定性评测。Qwen 和 Flash 本次未产生真实调用。

### 保留边界

- 搜索结果是 snippet 级发现证据，重要金融结论仍应优先 SEC、FRED、央行、交易所、公司公告或授权 RAG。
- 没有开放模型自拟 URL 的通用 fetch，搜索 endpoint 仍由部署固定。
- 调用同时受服务端 `MAS_ALLOW_NETWORK`、请求 `allow_network=true` 和 Harness 网络预算控制。
- 当前只选择一个活动 LLM；自动多模型路由会引入成本、语义差异和错误掩盖，未在没有明确策略前加入。
