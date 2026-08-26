# MAS Finance 企业级验证与故障注入报告

状态：实现与评测基线同步
版本：2.2
评测日期：2026-08-26

## 1. 结论

MAS Finance 的核心研究链路已经成为“可验证的参考实现”：模型从运行时工具目录自主选择研究动作，LangGraph 承载四阶段生命周期与恢复，Harness 约束每次执行，证据、计算血缘、报告校验和停止条件由代码契约控制。当前黑盒验收 11/11 通过；142 项自动化测试通过；源码总覆盖率 84.18%，80% 门槛生效；Ruff、mypy 和 `compileall` 通过。2.2 另外验证了 BM25/embedding/RRF、临时与个人 PDF、非法向量、hybrid 网络权限边界，以及 Bocha 域名 allowlist 的 provider 不服从故障。

这里的“企业级”特指核心 Agent 的正确性边界可定义、可审计、可故障注入和失败关闭，不等于整个部署已经取得生产认证。单租户 API key、同步 provider、Redis list 作业、SQLite 敏感状态、非 append-only 审计和实验性 Yahoo 适配器仍是明确的上线门槛，见第 10 节。

## 2. 第一性原理与系统不变量

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

## 3. 评测方法

### 3.1 黑盒

从用户问题进入 `FinanceAnalysisService`，只观察最终状态、工具审计、gap、预算和请求实体。运行：

```bash
python -m mas_finance.evaluation
```

评测实现位于 `src/mas_finance/evaluation.py`，测试入口位于 `tests/test_evaluation.py`。评测不访问网络，不依赖模型 key，不以快照文本作为唯一断言，因此适合 CI。

### 3.2 白盒

直接替换 Planner、provider、LLM 和 LangGraph checkpointer，验证：

- 重复 Planner task 只执行一次；
- provider 返回畸形对象时 Harness 拒绝；
- LangGraph checkpoint 在合成崩溃后恢复 phase、审计、预算和 call sequence；
- 模型引用被上下文裁掉的证据时 claim 被拒绝；
- 一个有效 quote 不能“洗白”另一个无关 citation；
- 账本冲突阻断派生比率并生成 `conflicted` claim。

### 3.3 故障注入

注入点覆盖：

| 注入层 | 故障 | 预期行为 |
|---|---|---|
| Planner | 重复 task、未注册工具 | 去重或拒绝，形成可见 gap |
| Harness 输入 | 缺字段、多余字段、NaN、超大对象 | provider 调用前失败，预算不消耗 |
| Harness 输出 | 无 bundle、伪造 ID、超大响应 | `invalid_tool_result` |
| Provider | 空响应、超时、限流、缺字段 | retry 受网络预算限制；失败转 gap |
| 市场数据 | 只有 raw close | 指标可用但状态降级，公开未计分红 |
| Context | evidence 截断、跨 entity 竞争 | 分组轮询；只允许引用 manifest 内 ID |
| LLM | 非 JSON、虚假 ID、错误 quote | 确定性合成并产生 fallback gap |
| Memory | 过期、畸形、实体切换 | 删除过期记录；当前问题优先 |
| Checkpoint | graph step 后崩溃、请求不匹配、NaN | 从待执行节点恢复或明确拒绝，不刷新预算 |
| Report | Markdown/footnote 注入 | 转义不可信字段；校验未知引用 |

## 4. 黑盒场景矩阵

当前固定场景如下：

| 场景 | 期望工具 | 期望状态或关键 gap | 验证目的 |
|---|---|---|---|
| 中文自然语言 CAGR | `finance.calculate` | `succeeded` | 中英解析与公式路由 |
| 市盈率定义与局限 | `finance.knowledge` | `succeeded` | 不依赖模型常识 |
| 结构化 Sharpe | `finance.calculate` | `succeeded` | 白名单计算与输入血缘 |
| 利率影响银行股 | `finance.knowledge` | `succeeded` | 用版本化机制解释而非模型常识 |
| 预测未指定股票明日精确价 | 无 | `unsupported_research_scope` | 不可证实预测失败关闭 |
| 无假设 DCF | 市场快照可尝试 | `valuation_model_inputs_required` | 不发明预测、折现率、终值 |
| ROE 直接计算 | 无 SEC 时无动作 | `metric_requires_average_balance` | 不用期末权益冒充平均权益 |
| 离线查询当前股价 | `market.snapshot` | `market_provider_unavailable` | 不生成假行情 |
| 只在请求侧开/关网络 | `market.snapshot` | `network_denied` 且预算为零 | 双重授权与预算语义 |
| Apple 后询问 Microsoft | `market.history` | 请求实体仅为 Microsoft | 线程记忆不覆盖当前实体 |
| 注入式内部 RAG | `internal.credit_search` | `succeeded` | 固定 ACL filter、页码与 provider provenance |

评测器要求工具集合精确匹配，关键 gap 至少出现，且状态匹配。网络拒绝虽然会产生审计事件，但 `attempts=0`、`budget_consumed=false`，恢复后也不会被重算成一次已用预算。

## 5. 本轮审计发现与修复

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
| 扫描 PDF 原生提取为空 | Agent 无法看见图片中的财务内容 | 扫描页诊断与可选 PaddleOCR-VL-1.6；双重授权、有限轮询且只取 Markdown |
| 网页检索 evidence 没有原始 URL | 搜索结论无法回到网页复核 | canonical metadata 保留 title/source URL/publisher/publish date |
| 计算参数允许多余字段或错误单位 | LLM/调用方可制造语义错误的精确数字 | operation schema、严格类型、单位兼容、数值域与溢出校验 |
| DeepSeek 旧模型名与 V4 默认 thinking | 部署失效或短预算只有 reasoning 无正文 | 默认 V4 Flash、关闭 thinking、严格 response contract |
| 网页 Evidence 未纳入 context source priority | 真实 LLM 网页合成触发 KeyError | ContextAssembler 显式支持 WEB，并增加回归测试 |
| 搜索 tracking URL/重复摘要占满上下文 | 表面多来源、实际重复 | canonical URL 与内容双去重，按 domain 分组 |
| 单一普通站点即可满足 web coverage | 低质量单点信息被当作完整研究 | 普通网页要求至少两个 domain；公共机构可单源满足覆盖；snippet claim 始终 inferred |
| Context 只按实体/来源类型分组 | 综合多 PDF 时同一文档重叠 chunk 可淹没其他文档 | 增加由研究意图控制的 document 分散、domain 分组与 query-centered window；普通问题仍全局相关排序 |
| 3,000 输出 token 被误解为全部上下文 | 多 PDF 综合能力被低估且预算不可审计 | 规划/生成输入证据独立为 24K/48K 字符，可调到 200K，并持久化 manifest |
| 个人偏好/skill 无明确治理 | 自动学习会放大错误和提示注入 | 只允许显式 CRUD；个人文本不作为 Evidence；同 kind/title 最新明确写入覆盖 |
| 临时 PDF 与长期知识库边界不清 | 无意永久留存或跨用户泄漏 | request/session/personal 三套显式生命周期，个人库 tenant/user 查询隔离并可删除 |
| 模型自拟计算只能执行代码或完全禁止 | 任意代码风险或缺乏灵活性 | 新增受限 AST 声明式公式；保留输入血缘，语义固定降级为 inferred |
| 企业/MCP 工具缺少统一注入门 | 原始 schema、副作用和权限进入模型目录 | 只接受部署期 read-only canonical EvidenceBundle 工具，冲突/raw/副作用启动拒绝 |

## 6. 金融正确性审计

### 6.1 市场历史

- 原始观察序列本身登记为 calculation input evidence；
- total return、年化收益、波动率、Sharpe、最大回撤引用输入 evidence ID；
- `adjusted_close` 与 `close` 明确区分；
- raw close 计算结果不会假装包含现金分红，生成 `unadjusted_price_history` gap；
- 日期去重、ISO 校验，价格必须为正且有限。

### 6.2 SEC 财务指标

- duration fact 保留 start/end，instant fact 保留期末日；
- 净利率等只使用同实体、同单位、同期间证据；
- 冲突值阻断计算；
- ROE/ROA 在没有平均权益/平均资产时显式拒绝；
- quick ratio 在缺少液态资产/存货拆分时显式拒绝；
- DCF 在没有预测现金流、折现率、终值假设时显式拒绝。

### 6.3 宏观数据

FRED latest/previous observation 分别形成证据，变化值标记为 `calculation` 并保留输入 ID，不再把本地算术伪装成 FRED 原始字段。FRED 的 series observations 是官方提供的序列观察接口：[FRED series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)。

### 6.4 确定性公式

内置语义公式允许 ratio、percentage change、CAGR、future/present value、loan payment、annualized return/volatility、Sharpe 和 max drawdown。另有 `finance.formula` 接受声明式数值表达式，但只遍历 AST 白名单，绝不执行代码或 `eval`。所有输入必须有限，分母、收益域、价格序列、AST 深度、指数和结果范围逐项校验；自拟公式只能证明可复算，不能证明金融语义，因此输出为 inferred。

## 7. 外部 provider 评估

### SEC

客户端只访问固定 EDGAR endpoint，要求带组织与联系邮箱的 User-Agent，并以至少 0.12 秒间隔限制单客户端请求。SEC 当前公平访问指南要求自动化访问保持适度，并给出总计不超过每秒 10 次请求的指导：[SEC Developer Resources](https://www.sec.gov/about/developer-resources)、[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。进程级限速不等于跨实例全局限速；生产仍需共享 rate limiter。

### FRED

使用固定 HTTPS endpoint、独立 API key、series metadata 与 observations。生产需要 provider 配额、重试和缓存监控。

### AlphaVantage

快照使用 OVERVIEW/GLOBAL_QUOTE；当前历史适配使用 raw `TIME_SERIES_DAILY`，所以不会声称是含分红的总回报。官方文档将 Daily Adjusted 单独列出并标为 premium：[Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)。配置缺 key 或请求失败时直接返回 unavailable，不切换供应商。

### Yahoo

当前 adapter 调用非契约化公共 endpoint，只能用于开发验证，`/api/v1/tools` 标记为 `experimental_non_contractual`。生产应替换为有许可、SLA、时效定义和 corporate-action 口径的供应商。

## 8. Prompt 与上下文评估

上下文不是整本账本直接塞给模型，而分为：

1. task：当前问题、实体、语言；
2. research：scope、coverage、未解决 gap、停止原因；
3. thread context：前一问题和最小实体状态，明确不是证据；
4. evidence cards：ID、内容、结构化值、provider、locator、period、as-of、published-at。

Evidence 按 `entity × source_type × domain/provider origin` 分组轮询；只有模型/明确多文档意图要求时，文档 origin 才切换成 document ID。这样综合任务避免单一 PDF 垄断，聚焦问题仍保持全局相关排序。随后按查询重合、来源质量、结构化程度、confidence 和 retrieval rank 排序。规划默认 24K、最终生成默认 48K evidence 字符，可配置到 200K；字符预算不是 tokenizer。逐阶段 `ContextManifest` 记录纳入/遗漏证据、来源类型和预算，LLM 只能引用 manifest 中的 ID。模型必须输出 JSON 和逐字 quote；quote 只证明引用文本存在，不能证明完整语义蕴含，因此高风险结论仍需要人工/NLI 审核。

## 9. 记忆层评估

| 层 | 当前实现 | 可保存 | 不可保存 |
|---|---|---|---|
| Run state | LangGraph InMemorySaver/SqliteSaver | phase、计划、观察、账本、报告、审计 | 不作为跨用户知识 |
| Conversation context | SQLite 精确 tenant/user/thread namespace；完整账本 + 有界滚动摘要 | user/tool/assistant 事件、实体时间与关系 | EvidenceBundle、PDF 原文、system/model prompt、隐藏推理、凭据 |
| Personal memory | SQLite，精确 tenant/user namespace，显式 CRUD | profile、preference、experience、skill | 自动提取、事实 Evidence、可执行指令、凭据 |
| Domain corpus | 上传 PDF 的 request/session BM25 | 当前请求或显式会话授权的文档 chunk | 默认不跨请求；opt-in session 默认 1 小时 |
| Personal corpus | SQLite 页文本 + request-time BM25 | 用户明确持久上传的 PDF 页与 provenance | 原始 PDF、其他用户文档、自动入库 |

实体解析次序为：显式实体 → 当前问题/当前文档实体 → 真实指代时的线程实体。个人 profile/preference 总是候选，experience/skill 仅在词项相关时召回，最多八条/12K 字符且明确不是证据。同 kind/title 的后一次明确写入覆盖旧值。当前 HTTP 删除接口只处理单部署默认 principal；Service 层虽有 tenant/user 隔离，多租户上线前仍必须接入身份系统而不是信任客户端自报 tenant。

## 10. 上线门槛

下列事项完成前，不应宣称整个服务“生产就绪”：

| 优先级 | 门槛 | 当前风险 |
|---|---|---|
| P0 | OIDC/JWT 或网关 principal，并贯穿 job、memory、artifact ACL | 当前 API key 是单部署、单租户边界 |
| P0 | 商业行情 provider、许可、SLA、时效与 corporate-action 契约 | Yahoo 仅实验；Alpha 历史为 raw close |
| P0 | 可靠作业 lease/visibility timeout/幂等与死信 | Redis list pop 可能在 worker 崩溃时丢任务 |
| P0 | checkpoint/memory/artifact 加密、TTL、删除与审计 | SQLite 状态可能包含敏感证据 |
| P1 | 原生 async 或进程隔离、可取消 provider 调用 | 当前同步 provider 会阻塞单 event loop |
| P1 | append-only 审计、OpenTelemetry、成本/token/provider 配额 | 当前审计随 run state 保存 |
| P1 | principal-to-ACL、hybrid corpus、索引 manifest、删除传播 | 个人库为词法 SQLite；会话 corpus 仅单进程短 TTL；注入源只有部署固定 filters |
| P1 | schema drift/录制响应/时区/币种/重述/拆股专项回归 | 当前为 deterministic fake-provider 契约测试 |
| P2 | tokenizer 预算、semantic entailment、人工抽检 | 当前字符预算与 literal quote 仅是保守门 |
| P2 | 压测、并发、长时间 soak 与灾难恢复演练 | 当前没有容量认证 |

## 11. 验收命令

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
