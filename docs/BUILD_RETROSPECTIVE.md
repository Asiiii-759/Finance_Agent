# MAS Finance 全过程构建复盘与问题总账

状态：与 1.4 实现同步
复盘日期：2026-08-12
覆盖范围：最初 Git 基线、第一次完整重构、企业级故障注入目标、RAG/上传/记忆复审目标

## 1. 这份文档回答什么

这不是发布日志，也不是只描述最终架构。它回答五个问题：

1. 在整个构建过程中实际发现了什么问题；
2. 为什么这些问题会让金融 Agent 看起来能运行、实际却不可信；
3. 每个问题最终采用了什么方案，为什么没有选择更复杂的方案；
4. 用什么测试、场景或故障注入证明方案确实工作；
5. 哪些风险仍未解决，为什么被明确留作生产门槛。

事实来源包括：

- Git `HEAD` 中的旧 `graph.py / agents.py / tools.py / memory.py / state.py`；
- 当前工作树中的最终实现；
- 黑盒评测、白盒测试、故障注入、静态检查、覆盖率和安装后冒烟；
- SEC、FRED、Alpha Vantage、NIST、OWASP 与检索系统官方资料。

## 2. 最终结论

原系统本质上是一个金融多角色演示：固定节点、样例数据、进程内“记忆”和 LLM 模板共同生成看起来完整的报告。它没有回答“这条结论来自哪里、为什么调用这个工具、缺数据时为何仍能成功、重启后预算是否仍可信、记忆是否会污染下一轮”等企业应用必须回答的问题。

当前系统被收敛为一个最小可信研究内核：

```text
user request
  → deterministic scope and requirements
  → bounded adaptive planner
  → policy-controlled tools
  → SourceRef / Evidence ledger
  → coverage and conflict assessment
  → bounded replan
  → evidence-bound synthesis
  → hard validation
  → succeeded / degraded / failed
```

核心变化不是“用了更多模型或更多 Agent”，而是把事实、权限、预算、状态和失败语义从提示词约定迁移到代码契约。

## 3. 演进阶段

### 阶段 A：旧的固定多角色演示

旧系统使用 LangGraph，把 supervisor、retrieval、quant、psychologist、critic、replanner、synthesizer 串成固定流程。无论用户问 CAGR、当前价格、财报风险还是概念解释，都需要经过相似角色。工具直接调用，没有统一权限、预算和结果契约；未知公司默认变成 Apple/Microsoft；报告允许样例数据回退。

### 阶段 B：统一项目与领域契约

首先删除双入口、LangGraph 节点、角色代理、样例金融数据和虚假知识图谱，建立唯一 `Multi-Agent-project` 和唯一 `mas_finance` 包。随后建立 `SourceRef / Evidence / Claim / EvidenceBundle`，把来源、时间、单位、期间和引用变成领域对象。

### 阶段 C：显式循环与 Tool Harness

将固定工作流替换为需求驱动循环；建立 Tool Harness，统一执行 capability、side effect、双重网络授权、输入/输出契约、重试、超时、预算和审计。Planner 只能从已注册且已授权的工具中选择。

### 阶段 D：金融数据与计算正确性

逐步加入行情、历史序列、SEC、FRED、白名单公式和版本化金融知识。所有派生指标保存输入 evidence ID；单位、实体或期间不一致时不计算；ROE、ROA、quick ratio、DCF 等输入不足时明确拒绝。

### 阶段 E：上下文、合成与记忆边界

Prompt 被拆成 task、research、thread context、evidence cards；ContextManifest 记录模型真正看到的证据；模型只能引用 manifest 内且包含逐字 quote 的 evidence。线程记忆只保存最小指代信息并带 TTL，不保存 evidence、prompt 或用户画像。

### 阶段 F：企业故障注入与代码收敛

针对恢复、预算、身份穿透、citation laundering、畸形 provider、Markdown 注入、非有限 JSON、数据冲突和异常泄密进行故障注入。删除未接入产品的长期 memory promotion、旧兼容工具名和重复 API state。

### 阶段 G：RAG、实时上传和外部检索复审

最后从真实使用路径重新审视文档能力，补上页级 provenance、未知实体文档绑定、严格 RAG payload、多个检索源的主备规划、固定 ACL filters、受控 HTTP JSON 搜索网关、响应体上限和检索 trace。外部搜索不直接抓取任意 URL，而通过部署期固定的 canonical gateway 接入。

### 阶段 H：OCR、真实模型与文档生命周期复审

扫描 PDF 接入有界 PaddleOCR-VL-1.6；DeepSeek V4 切换到当前模型名并关闭默认 thinking，避免短输出预算只消耗推理而没有 final content。随后用真实扫描 PDF 和真实两 PDF 综合问题验证远端协议。最后将上传明确拆成 request-local 与 opt-in session 两种语义；永久知识仍留在带身份和 ACL 的独立 RAG 控制面，不新增不安全的伪永久上传接口。

### 阶段 I：纠正 LangGraph 与模型自主性边界

前一阶段把“不要保留旧版本兼容写法”扩大解释成“完全删除 LangGraph”，这是一次需求理解偏差。手写循环本身可测，
但重复实现了 step persistence、历史和 pending-node recovery；同时规则 Planner 成了正常主路线，模型只负责最终写作，
自主性不足。架构图还曾把 Harness 画成业务阶段，混淆了 middleware 与 workflow node。

纠正方案不是恢复旧 supervisor/critic 角色图，而是建立全新的四节点 LangGraph：intent、planning、validation、
final_generation。ModelPlanner 每次自主选择一个工具动作或 finish，工具在同一 planning step 内经 Harness 配套执行；
validation 可拒绝过早结束。自建 checkpoint 和手写 while 被删除，LangGraph Saver 成为唯一恢复底座。随后增加
provider-neutral `web.search`，使模型能自主产生检索式和时效范围，而不是固定走 Yahoo 或某个内部数据源。

## 4. 问题总账

状态说明：`已解决` 表示当前代码和测试已闭环；`刻意不实现` 表示当前不具备治理前提；`生产门槛` 表示核心 Agent 可用但服务上线前仍必须完成。

### 4.1 项目与架构

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| ARC-01 | 曾存在项目/入口边界不清 | 同一业务可能有多个实现和配置口径 | 收敛为一个项目目录、一个包、一个 Agent 入口 | wheel 只含当前包 | 已解决 |
| ARC-02 | 固定角色图不随问题变化 | 简单计算也跑完整角色链，角色名替代业务需求 | 重建四节点 LangGraph；planning 动态选择动作 | 图节点集合与自主路由测试 | 已解决 |
| ARC-03 | supervisor/psychologist/critic 是角色拟人，不是能力边界 | 角色可重复推理、难以测试、无法描述输入输出契约 | 以 scope、planner、tool、coverage、validator 等职责对象替代 | 38 个源码文件 mypy 通过 | 已解决 |
| ARC-04 | `route_after_critic` 无论发现问题都进入 synthesizer | critic 结论不真正影响控制流 | 校验失败直接改变状态或失败，不靠路由字符串 | validation fault tests | 已解决 |
| ARC-05 | replanner 的两条分支都把 `replan_reason` 清空 | “重规划”没有可证明的状态变化 | Planner 只处理未满足 requirement，记录每轮 plan 和 task | 主/备 provider 测试 | 已解决 |
| ARC-06 | 未知问题没有显式支持范围 | 系统可能无数据仍输出泛化答案 | 建立 unsupported requirement 和 `unsupported_research_scope` | 精确预测场景失败关闭 | 已解决 |
| ARC-07 | 状态只有松散 TypedDict | 字段可缺失、恢复时难以校验 | 使用 dataclass/enum、严格 `to_dict/from_dict` 和 schema 版本 | checkpoint tamper tests | 已解决 |
| ARC-08 | 为兼容旧实现保留无入口模块 | 增加维护面、误导扩展者 | 删除旧 graph/agents/state/tools/memory；新 graph 不保留旧接口 | wheel 与 import 测试 | 已解决 |
| ARC-09 | RAG 扩展类只能从内部模块导入 | 安装包能运行，但部署方的公共扩展入口不直观 | 包根惰性导出 `RetrievalSource / HTTPJSONRAGClient`，不引入 API 可选依赖 | public export test + installed-wheel import | 已解决 |
| ARC-10 | 将“不保留兼容”误解为“不使用 LangGraph” | 重复实现恢复基础设施，偏离用户真实意图 | 明确区分旧图兼容与框架能力；LangGraph 成为唯一 runtime | SQLite saver/历史/恢复测试 | 已解决 |
| ARC-11 | Harness 被画成业务节点 | 基础设施与研究阶段职责混淆 | Harness 只作为 planning 每次工具调用的 middleware | 图中仅四个业务节点 | 已解决 |

### 4.2 规划与 Agent 循环

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| PLN-01 | 所有查询采用同一固定 workflow | 没有需求分类和字段级目标 | 中英 deterministic scope，生成 intents 与 requirements | CAGR、知识、市场、SEC、宏观场景 | 已解决 |
| PLN-02 | 模型可能选择不存在或越权工具 | 提示词不是安全边界 | Planner 输出仍需经过工具注册表和 capability allowlist | `planner_tool_denied` 注入 | 已解决 |
| PLN-03 | Planner 可重复同一 task | 浪费预算甚至循环 | task ID 由工具名和参数稳定哈希，重复 task 不执行 | duplicate task test | 已解决 |
| PLN-04 | 主数据源失败后没有真正 fallback | 固定节点无法表达同一 requirement 的多个候选工具 | 每类 requirement 配置有序候选，下一轮选择未尝试工具 | RAG primary/fallback、市场 fallback tests | 已解决 |
| PLN-05 | 部分实体完成时仍可能成功 | 只判断是否有结果，不判断完整性 | CoverageAssessor 按 requirement/entity/field 计算缺口 | 部分预算场景不得 succeeded | 已解决 |
| PLN-07 | 规则 Planner 成为正常主路线 | 模型只写最终答案，无法自主选择研究策略 | ModelPlanner 优先返回单动作/finish，规则仅作可见降级 | 真实 DeepSeek + scripted model tests | 已解决 |
| PLN-08 | 模型可能过早 finish 或重复动作 | 缺证据回答、循环浪费费用 | validation 拒绝不足的 finish；稳定 task ID 去重 | premature-finish/repeat tests | 已解决 |
| PLN-09 | 外部研究被专用 provider 路线锁定 | 新问题只能落到预设 Yahoo/SEC/RAG | 加入 provider-neutral web.search，模型控制 query/freshness/domain | web fixture + Brave MockTransport | 已解决 |
| PLN-06 | 预算耗尽与无可用工具混为一谈 | 停止原因不可运营 | 枚举稳定 stop reason，并将 gap 与预算分开 | checkpoint 与报告断言 | 已解决 |
| PLN-10 | 恢复后可能获得新预算或重复 call ID | 预算和序号未作为恢复状态 | 持久化 audit、预算消耗和 call sequence，绑定 run boundary | crash/resume tests | 已解决 |

### 4.3 Tool Harness

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| HAR-01 | 工具只是普通函数 | 没有统一权限、成本与审计 | `ToolSpec + ToolHarness + ToolContext` | Harness 单元与故障注入 | 已解决 |
| HAR-02 | 网络访问默认发生 | 旧行情默认 Yahoo，测试也可能出网 | 服务端允许与请求端同意必须同时成立；默认 offline | network denial scenarios | 已解决 |
| HAR-03 | 只读和副作用工具没有区分 | 将来加入交易工具会继承错误 retry | SideEffect 枚举；默认只允许 read_only；副作用工具禁止自动 retry | side-effect retry test | 已解决 |
| HAR-04 | 输入只由各 adapter 随意解析 | 多余字段、NaN、类型强转可穿透 | 统一参数键/大小/finite JSON 契约，adapter 再做类型校验 | malformed arguments tests | 已解决 |
| HAR-05 | provider 返回任意对象 | 畸形 payload 可进入 Agent state | 结果类型为 evidence bundle 或 model response，并反序列化验证 | invalid result tests | 已解决 |
| HAR-06 | 模型、研究工具、网络共用一本预算 | 预算含义无法解释 | 分离 research tool、network attempt、model call 三本账 | independent budget tests | 已解决 |
| HAR-07 | 被 policy 拒绝的调用也计入预算 | 重启后预算凭空减少 | 仅 `attempts > 0` 且 `budget_consumed=true` 计费 | denied-resume test | 已解决 |
| HAR-08 | 同一 run 可换 tenant/user 或提高上限 | 身份与成本边界穿透 | 首次调用绑定 run、principal、网络授权和预算 ceiling | run boundary test | 已解决 |
| HAR-09 | 错误字符串可能携带 token/URL 凭据 | 审计和 API 泄密 | 参数、Bearer、API key、密码和 credential URL 脱敏；job 只保存异常类型 | redaction/job error tests | 已解决 |
| HAR-10 | timeout 被误解成强制取消 | Python 线程不能安全终止任意同步函数 | 明确为观测边界；生产要求 async/进程隔离 | 文档与生产门槛 | 生产门槛 |

### 4.4 证据、引用与冲突

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| EVD-01 | 旧报告只有“数据来源”文字，没有可定位引用 | 结论无法复核 | 每条 evidence 持有 provider、locator、as-of、period、page/span | provenance tests | 已解决 |
| EVD-02 | 模型输出被当作事实 | LLM 可能编造事实与引用 | 模型只生成 Claim；事实只能来自 evidence | no-evidence fail-closed | 已解决 |
| EVD-03 | ID 可被 payload 伪造 | 恢复或 provider 可篡改内容 | source/evidence/claim 使用 content-addressed ID，反序列化重算 | tamper test | 已解决 |
| EVD-04 | 相同字段不同值会被覆盖或任取一个 | 金融冲突被隐藏 | 保留所有原始 evidence，不自动选赢家；生成 conflicted claim | conflict reconciliation test | 已解决 |
| EVD-05 | 冲突数据仍可能进入派生公式 | 产生看似精确的错误指标 | 只有同 entity/period/unit 且值一致的输入可派生 | ratio suppression test | 已解决 |
| EVD-06 | 一个正确 quote 可以附带无关 citation | citation laundering | 每个 evidence ID 都必须实际包含该 quote | unrelated citation test | 已解决 |
| EVD-07 | 模型可引用被上下文裁掉的 evidence | 模型声称看过实际未提供的证据 | ContextManifest 白名单 | context omission test | 已解决 |
| EVD-08 | bundle/checkpoint 可无限增长或包含 NaN | 内存/磁盘 DoS，恢复不确定 | 数量、字符、JSON 与单结果字节硬上限 | size/NaN tests | 已解决 |

冲突策略刻意不做通用“source priority 自动覆盖”。SEC filing、行情、用户输入和内部文档可能描述不同口径或时间点；在没有可证明的对齐规则时，自动选值比公开冲突更危险。系统只在确定的同实体、同字段、同期间、同单位分组内判断结构化冲突。未来若增加权威源优先级，也必须是字段级、版本化规则，并保留被覆盖的原值。

### 4.5 金融数据与计算

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| FIN-01 | 样例 Apple/Microsoft 数据可进入最终报告 | demo 数据伪装真实研究 | 删除样例数据和 fallback | offline 行情失败关闭 | 已解决 |
| FIN-02 | 未识别公司默认 Apple/Microsoft | 用户问题被静默改写 | 无实体就不调用实体工具；支持显式 entity/symbol | entity switch tests | 已解决 |
| FIN-03 | AlphaVantage 失败后静默换 Yahoo | 来源许可、口径和 SLA 不可控 | provider 显式，失败直接 unavailable | no silent fallback test | 已解决 |
| FIN-04 | Yahoo 是默认 provider | 默认依赖非契约接口 | 默认 offline，Yahoo 标记 experimental | config/tool catalog tests | 已解决 |
| FIN-05 | raw close 被当作总回报 | 忽略拆股/分红口径 | adjusted/raw basis 明示；raw 结果产生 gap | market history tests | 已解决 |
| FIN-06 | 市场指标没有输入序列血缘 | 无法重算收益/波动/回撤 | 原始序列先成为 evidence，派生项引用 input IDs | financial scenario tests | 已解决 |
| FIN-07 | SEC duration/instant 期间被弱化 | 不同期事实可能错误相除 | 保留 start/end 或 instant date；同期间才派生 | SEC ratio tests | 已解决 |
| FIN-08 | ROE/ROA 用期末余额冒充平均余额 | 公式口径错误 | 缺期初/期末时显式 `metric_requires_average_balance` | black-box scenario | 已解决 |
| FIN-09 | quick ratio 缺资产拆分仍可能计算 | current ratio 与 quick ratio 混淆 | 缺 liquid asset breakdown 时拒绝 | scope tests | 已解决 |
| FIN-10 | DCF 可能由模型补预测、WACC、终值 | 估值假设被隐藏 | 缺显式假设时拒绝生成 DCF 数值 | black-box scenario | 已解决 |
| FIN-11 | 通用 AST 公式执行面过宽 | 即使禁函数，仍不利于公式版本治理 | 改为枚举白名单公式和逐操作校验 | metric tests | 已解决 |
| FIN-12 | FRED 变化值看起来像 provider 原始字段 | 来源与本地算术混淆 | latest/previous 为宏观 evidence，change 为 calculation evidence | FRED scenario | 已解决 |
| FIN-13 | 紧凑中文“100增长到121”只返回 CAGR 定义 | 参数抽取只接受“从100…到121”，黑盒问题措辞过窄 | 在仍要求明确增长动词、终点和年数的前提下支持省略“从” | 安装后 CLI 探测 + compact CAGR 回归 | 已解决 |
| FIN-14 | 无公司实体的计算显示“未知实体” | 通用合成模板把“计算无须实体”误作“实体缺失” | calculation evidence 使用“计算结果”标签，其他证据仍保持缺实体提示 | compact CAGR report assertion | 已解决 |
| FIN-15 | 结构化计算可携带多余字段、隐式文本或伪造单位 | LLM/调用方参数若被宽松强转，会制造语义正确性假象 | request extra-forbid、严格 operation/text/numeric、operation-unit compatibility、大小与指数边界 | 全 operation 与 malformed contract tests | 已解决 |
| FIN-16 | “让 LLM 算”与“让函数算”边界不清 | 模型算术、任意表达式和隐式假设不可复算 | LLM 不执行表达式；只接受最终通过 `MetricRequest` 的白名单函数参数，函数输出登记公式与输入血缘 | tool-set、formula 与 lineage assertions | 已解决 |

FRED observations 的接口边界依据 [FRED 官方 series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)。Alpha Vantage 将 raw daily 和 adjusted daily 分开提供，当前实现据其[官方文档](https://www.alphavantage.co/documentation/)公开 raw 历史口径。SEC 客户端使用固定 EDGAR endpoint 和组织 User-Agent，并依据[SEC Developer Resources](https://www.sec.gov/about/developer-resources)保守限速；多实例生产仍需要共享 limiter。

### 4.6 PDF、上传与本地 RAG

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| RAG-01 | 旧 PDF 只取前 4000 字 excerpt | 关键证据可能永远不可检索 | 全文按页进入 request-local corpus | upload integration test | 已解决 |
| RAG-02 | 文档没有页级边界 | 引用只能指向整份文件 | 解析保留非空 page records；逐页 chunk | page=2 provenance test | 已解决 |
| RAG-03 | 旧正则抽数不保留单位/期间 | 数字可能被错误解释，且字段已无消费方 | 删除 `metric_hints/excerpt/content_sha256` 残留；原文 evidence 由 RAG 返回 | dead-code search + tests | 已解决 |
| RAG-04 | 上传文件名可能穿越目录 | 任意文件写入 | suffix、basename、NFKC、安全子路径和随机前缀 | path tests | 已解决 |
| RAG-05 | 上传只检查客户端 MIME | 可伪装可执行/任意内容 | 文件数量、字节、`.pdf`、PDF magic、页数四重限制 | upload security tests | 已解决 |
| RAG-06 | 压缩 PDF 可产生超大抽取文本 | 小文件仍可能导致内存 DoS | 增加每 PDF 5M extracted-character 上限 | text-bound test | 已解决 |
| RAG-07 | API 引用暴露服务端随机文件名前缀 | 用户无法识别自己的文件 | 仅文件系统使用随机前缀，evidence 使用原始安全显示名 | upload API test | 已解决 |
| RAG-08 | 任意未知实体无法由静态公司词典识别 | 上传 ACME 文档却无法满足 ACME requirement | 单一显式实体可作为用户声明绑定文档；多实体仍保守不猜 | ACME PDF test | 已解决 |
| RAG-09 | BM25 接口会把 list query 或 bool top_k 强转 | 畸形请求看似成功 | query/top_k/filter 严格类型与大小校验 | coercion tests | 已解决 |
| RAG-10 | lexical score 被当作 confidence | 排名分数并非概率 | 只使用显式 extraction confidence；BM25/RRF score 留在 metadata | adapter tests | 已解决 |
| RAG-11 | 文档可能包含提示注入 | RAG 并不能消除间接 prompt injection | 文档位于 evidence trust zone；不能改变工具权限；模型输出仍受引用和校验约束 | injection/Markdown tests | 已解决 |
| RAG-12 | scanned PDF 没有 OCR | 无文本时无法检索 | 原生提取先诊断；可选 PaddleOCR-VL-1.6 有界整文档解析；无 OCR 或无结果时失败关闭 | image-only、授权、MockTransport 与真实单页 smoke | 已解决 |
| RAG-13 | PDF 对象顺序不等于阅读顺序 | 多栏财报文本可能交错 | PyMuPDF `sort=True` 按视觉坐标提取，并做有限 Unicode/空白归一化 | parser tests | 已解决 |
| RAG-14 | OCR 示例无限轮询并下载所有图片 | 任务悬挂、内存/存储增长及外域资源风险 | 固定请求/轮询/文件/JSONL 上限；只读取页 Markdown；结果下载不携带 bearer token | OCR adapter tests | 已解决 |
| RAG-15 | 已限定“根据内部文档”的收入问题仍自动要求 SEC | 文档证据充分却被错误标成 degraded | 文档模式不隐式追加 regulatory requirement；显式交叉核验仍可同时要求 SEC | scope regression + real fallback RAG retest | 已解决 |

本地默认使用依赖很少的 BM25-style 检索，是为了让上传能力在无 embedding 服务时仍可确定运行。它不是“最终企业检索引擎”。生产可在相同 JSON 契约后替换为 lexical + semantic hybrid。Elastic 官方资料将 RRF 作为混合全文与语义检索的推荐方式之一：[Hybrid search](https://www.elastic.co/docs/solutions/search/hybrid-search)。是否真正提高本项目的金融召回率仍必须通过领域 query set、人工 relevance judgment 和 ACL 测试证明，不能仅凭采用向量库就宣称提升。

### 4.7 外部 RAG、内部知识库与联网搜索

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| EXT-01 | Agent 核心只能使用上传 corpus | 企业内部知识库无法接入 | `RetrievalSource` 部署期注入，不改 Agent 核心 | injected RAG black-box | 已解决 |
| EXT-02 | 增加 provider 容易把其私有 schema 泄漏进 Agent | 每接一个源都要改 Planner/Prompt | `RetrievalEvidenceAdapter` 作为 anti-corruption layer | fake provider contract tests | 已解决 |
| EXT-03 | 主 RAG 空结果时无备用源 | 单点召回失败 | Planner 接收有序 document tools，逐轮尝试未用来源 | primary/fallback test | 已解决 |
| EXT-04 | 远端 RAG 可能绕过网络同意 | “检索”被误当成本地计算 | 每个 source 声明 network_access，仍走双重授权 | denied remote RAG test | 已解决 |
| EXT-05 | Agent 可覆盖 tenant/ACL filters | 跨租户检索 | `fixed_filters` 由部署绑定并覆盖调用参数；对外 catalog 只公开是否启用 | ACL filter test | 已解决 |
| EXT-06 | 任意 URL 搜索会形成 SSRF/许可风险 | 模型生成 URL 不可作为访问授权 | `HTTPJSONRAGClient` 只接受启动时固定、无凭据 query、默认 HTTPS endpoint | MockTransport tests | 已解决 |
| EXT-07 | 远端响应、redirect 或 NaN 无界 | 内存 DoS、跳转绕过、恢复失败 | 禁止跟随 redirect、限制解压后字节、严格 JSON/object/chunk schema | gateway fault tests | 已解决 |
| EXT-08 | RAG request/index trace 丢失 | 无法关联搜索后台日志 | evidence metadata 保留白名单 trace 字段，完整 trace 留在 observation | trace test | 已解决 |
| EXT-09 | 新闻/网页内容的时效、版权和许可未定义 | “能搜到”不等于可用于金融产品 | 仅提供 canonical gateway 接缝，不内置无许可 scraper | 文档化 | 刻意不实现 |
| EXT-10 | 一旦配置 RAG 就让所有问题强制检索 | 把工具可用性误当成本次 requirement，纯计算也被污染 | 上传始终触发；普通请求仅在文档/内部资料/新闻/搜索语义或 `require_documents=true` 时触发 | unrelated CAGR test | 已解决 |
| EXT-11 | 外部搜索结果丢失网页标题和 URL | 内部文档字段集不能完整表达 web provenance | canonical chunk 接受 `title/source_url/publisher/publish_date` 并进入 SourceRef | web search provenance test | 已解决 |

NIST 的生成式 AI 风险框架指出，模型可能生成错误内容以及伪造的逻辑或引用，尤其在重要决策场景中需要监控：[NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)。OWASP 同时明确提醒 RAG 并不能完全解决 prompt injection，恶意内容可以进入知识库形成间接注入：[OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)。因此当前系统把检索结果视为不可信数据，而不是高权限指令。

### 4.8 Prompt 与上下文

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| CTX-01 | 旧 prompt 混合任务、文档、指标和状态 | 模型无法区分指令与数据 | task/research/thread/evidence 分区 | context payload tests | 已解决 |
| CTX-02 | 整个 evidence ledger 直接进入模型 | token 无界且单一实体挤占上下文 | 字符预算、item 上限、entity × source round-robin | balance test | 已解决 |
| CTX-03 | 上一轮问题可能被当作当前事实 | memory 与 evidence 混淆 | thread context 明示非证据，字段白名单 | minimal context test | 已解决 |
| CTX-04 | prompt 和原始 provider error 进入审计 | 可能泄漏敏感文档/凭据 | model audit 不保存 prompt，只保存状态和预算 | synthesis harness test | 已解决 |
| CTX-05 | 字符预算不等于模型 token | 不同语言估算误差 | 当前采用保守字符门；未来按模型 tokenizer 替换 | 文档化 | 生产门槛 |
| CTX-06 | literal quote 不能证明语义蕴含 | 模型仍可能曲解原句 | quote 只作为最低引用门；高风险结论需要数值 parser/NLI/人工抽检 | 文档化 | 生产门槛 |
| CTX-07 | DeepSeek 旧模型名已停用，V4 默认 thinking 可吃完短输出预算 | 时效变化导致空正文或部署失效 | 默认 `deepseek-v4-flash`，证据合成显式关闭 thinking，并严格校验 prompt/参数/response | MockTransport + 8-token live smoke | 已解决 |
| CTX-08 | 十项计算的真实模型 JSON 在 1400 token 上限被截断 | 函数和证据正确但合成只能降级 | 合成上限提高到 3000；简单响应仍提前结束 | 3173 字符截断诊断 + 10/10 live supported claim | 已解决 |

### 4.9 记忆

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| MEM-01 | SessionMemory 是全局进程内 list | 不隔离线程，重启丢失 | run checkpoint 与 thread memory 分开 | SQLite checkpoint tests | 已解决 |
| MEM-02 | ReasoningMemory 在 runtime 中持续追加 | 不同 run 可能相互污染，且像隐藏 CoT | 删除 reasoning memory；只保留结构化 decision/audit | state inspection | 已解决 |
| MEM-03 | NetworkX/Neo4j 图被称为 knowledge memory，但没有检索闭环 | 增加依赖却不提升回答 | 删除假知识图谱，领域知识改成版本化 evidence tool | knowledge scenario | 已解决 |
| MEM-04 | Neo4j 初始化异常静默回退内存 | 运维以为持久化，实际丢数据 | 删除该未完成能力；provider 失败必须可见 | 代码边界审计 | 已解决 |
| MEM-05 | 线程记忆没有 tenant/user/thread 精确 namespace | 跨用户泄漏 | namespace 哈希并精确查找，不提供跨 namespace 搜索 | isolation tests | 已解决 |
| MEM-06 | 线程记忆没有 TTL/删除 | 数据无限留存 | 默认 7 天 TTL，过期/畸形删除，提供 delete endpoint | expiry/delete tests | 已解决 |
| MEM-07 | 记忆内容类型可被字符串强转 | 损坏记录可能伪装合法 | `ThreadContextMemory.from_dict` 严格类型 | coercion test | 已解决 |
| MEM-08 | Apple 后改问 Microsoft 仍沿用 Apple | 旧实体优先级错误 | 显式实体 → 当前检测实体 → 真正指代时 remembered entity | entity switch black-box | 已解决 |
| MEM-09 | 设想了长期 memory promotion，但无真实产品治理 | 隐私数据无法查看、撤回、导出 | 删除 policy/API；长期用户画像明确未实现 | API/docs audit | 刻意不实现 |
| MEM-10 | 一次性 PDF 无法跨同线程追问 | 要么重复上传，要么误把临时文档永久入库 | 显式 session opt-in；只保留解析页文本、短 TTL、namespace 隔离和删除接口 | service/API follow-up tests | 已解决 |
| MEM-11 | 临时上传与永久知识库语义混在一起 | 用户同意、ACL、删除传播无法成立 | 三层生命周期；临时绝不自动 promotion，永久文档仅走受控 RAG | lifecycle contract review | 已解决 |

当前四种“记忆”语义必须分开：

1. Run checkpoint：为了崩溃恢复，不是用户知识；
2. Thread context：为了短期指代，只保存上一问题、实体、symbol、状态和 gap code；
3. Request corpus：本次授权上传，不默认跨请求保存；
4. Session documents：显式 opt-in 的解析页文本，进程内短 TTL，不是永久知识；
5. Persistent domain corpus：独立 ingestion/ACL/索引控制面，通过 `RetrievalSource` 只读接入；
6. Versioned finance knowledge：代码管理的领域定义，不是用户记忆。

任何 evidence、PDF 原文、prompt、隐藏推理、API key 都不得进入 thread memory。

### 4.10 API、上传作业与输出

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| API-01 | `/config` 回显 database URL 和路径 | DSN 可能含密码 | 只返回 backend 和 capability 信息 | API test | 已解决 |
| API-02 | AnalyzeResponse 同时返回摘要字段和完整 state | 大响应、契约重复、内部状态泄漏 | API 只返回 run/status/scope/coverage/report/bundle/gaps/audit/budget | schema tests | 已解决 |
| API-03 | 上传临时文件未保证清理 | 敏感文件留存 | sync 与 async 均在 finally/worker 完成后删除 | upload cleanup tests | 已解决 |
| API-04 | job 保存原始异常文本 | provider URL/token 可能公开 | 只持久化异常类型，详细信息交给受保护 telemetry | job error test | 已解决 |
| API-05 | artifact 文件名可路径穿越或覆盖 | 任意文件写入 | 安全 identifier、固定输出根、随机后缀 | reporting security tests | 已解决 |
| API-06 | HTTP API key 被误当多租户 identity | 所有调用共享 principal | 文档明确其仅为单部署门；多租户需要网关/OIDC principal | 生产路线图 | 生产门槛 |
| API-07 | Redis list pop 无 lease/visibility timeout | worker 崩溃后任务可能丢失 | 暂不伪装 exactly-once；建议 Streams/RQ/Celery/outbox | 文档化 | 生产门槛 |

### 4.11 测试与 Harness 工程

| ID | 发现的问题 | 根因与风险 | 采用的方案 | 验证 | 状态 |
|---|---|---|---|---|---|
| TST-01 | 旧测试默认使用 Yahoo | CI 依赖网络、结果不确定 | 所有验收默认 offline/fake provider | 断网全量测试 | 已解决 |
| TST-02 | 测试只检查报告含公司名 | 预设模板也可通过 | 检查工具集合、状态、gap、budget、lineage、citation | enterprise matrix | 已解决 |
| TST-03 | 无 provider 故障注入 | 只验证 happy path | 空响应、畸形 schema、超时、限流、NaN、ID 篡改、错误 quote | white-box suites | 已解决 |
| TST-04 | 无崩溃恢复测试 | checkpoint 只是声明 | 合成阶段注入崩溃，恢复 phase/audit/budget/call ID | checkpoint test | 已解决 |
| TST-05 | unittest 通过但 pytest 误收集 helper | 两套 CI runner 结果冲突 | helper 去除 `test_` 前缀，两个 runner 都设门 | unittest/pytest | 已解决 |
| TST-06 | 没有覆盖率门 | 大范围重构可能未触及关键路径 | pyproject 设置总覆盖率 80% | pytest-cov | 已解决 |
| TST-07 | 源码通过但安装包可能缺模块或夹带旧模块 | editable install 掩盖 packaging 问题 | PEP 517 wheel、隔离安装、包成员审计和 CLI smoke | final package check | 已解决 |

## 5. 当前面向用户的调用逻辑

### 5.1 用户上传 PDF

```text
multipart upload
  → count/size/suffix/magic validation
  → temporary randomized server path
  → PDF page/text limits
  → page-aware request corpus
  → corpus.search
  → document evidence with page/span
  → coverage/conflict/synthesis
  → temporary upload deletion
```

同一批 PDF 需要连续追问时，调用方必须提供 thread id 并显式设置 `retain_for_session=true`；后续请求再显式设置 `use_session_documents=true`。原 PDF 仍删除，只保存解析页文本，默认 1 小时 TTL，可列举和删除。默认行为没有变化。

如果 PDF 是扫描图片，系统先给出 `ocr_required` 诊断。部署配置 PaddleOCR-VL-1.6 且服务端与请求端都允许网络时，整份 PDF 只提交一次并取回页级 Markdown；否则纯扫描件失败关闭，不调用模型常识补内容。

### 5.2 企业内部 RAG

应用在启动时注入来源，而不是让用户或模型提供 endpoint：

```python
from mas_finance.retrieval import HTTPJSONRAGClient, RetrievalSource
from mas_finance.api.app import create_app

client = HTTPJSONRAGClient(
    "https://rag-gateway.example.com/search",
    api_key="deployment-secret",
)
source = RetrievalSource(
    name="internal.research_search",
    client=client,
    provider="enterprise_research_corpus",
    network_access=True,
    fixed_filters={"tenant_id": "tenant-a", "acl_group": "research"},
)
app = create_app(retrieval_sources=(source,))
```

canonical gateway 请求：

```json
{
  "query": "ACME covenant risk",
  "top_k": 5,
  "filters": {"tenant_id": "tenant-a", "acl_group": "research"},
  "search_mode": "rrf",
  "rerank": false
}
```

canonical 响应至少包含：

```json
{
  "chunks": [
    {
      "id": "chunk-1",
      "content": "verbatim source content",
      "rank": 1,
      "score": 0.82,
      "metadata": {
        "file_name": "credit-review.pdf",
        "company": "ACME",
        "source_page": 7,
        "publish_date": "2026-06-30"
      }
    }
  ],
  "trace": {
    "request_id": "provider-trace-id",
    "index_version": "2026-08-01",
    "search_mode": "rrf"
  }
}
```

Gateway 负责 provider API、版权许可、索引 ACL 和 lexical/vector/hybrid 实现；Agent 负责权限、预算、schema、provenance、coverage 与回答校验。这个分层避免每个搜索供应商的字段污染 Agent 核心。

### 5.3 外部网页或新闻搜索

推荐让自有 gateway 对接 licensed news/web provider，并输出同一 chunks contract。不推荐在 Agent 中提供 `fetch_any_url(url)`：

- URL 来自模型会形成 SSRF 与出网授权混淆；
- 网页正文可能包含间接 prompt injection；
- 新闻存在版权、地域、时间戳、修订和转载来源问题；
- 搜索 rank 不等于事实可信度；
- 一个网页可能引用另一个原始来源，需保留 source chain。

### 5.4 实时行情、监管和宏观

- 当前行情与历史：显式 provider，默认 offline；
- SEC：固定 endpoint、CIK/XBRL/filing locator；
- FRED：固定 series metadata/observations；
- 所有外部工具：服务端开网且请求端同意后才执行；
- provider unavailable 是 gap，不是切换到模型知识。

### 5.5 纯计算与知识解释

确定性公式和版本化金融知识无需外部数据。ModelPlanner 能从动态目录选择 `finance.calculate` 或
`finance.knowledge`；validation 不要求为了“多 Agent 感”追加市场、SEC 或 RAG。无模型时规则基线保持同样的最小路线。

## 6. 为什么当前设计不算过度复杂

本轮主动删除或拒绝了下列看似高级、实际无闭环的设计：

- 不保留 LangGraph 兼容节点；
- 不按“心理学家/批评家/主管”拆模型角色；
- 不默认部署向量数据库；
- 不把 NetworkX/Neo4j 图叫作长期记忆；
- 不执行任意代码；模型自拟计算只允许受限声明式 AST，并公开语义待核验；
- 不提供任意 URL 抓取；
- 不提供 broker/交易工具；
- 不自动从对话生成用户画像；单用户个人记忆只接受显式 CRUD，多用户能力继续等待可信 identity/retention；
- 不自动用 source priority 隐藏冲突；
- 不把模型 fallback 写成预设 Apple/Microsoft 结论。

扩展点只保留三个：

1. 新 Tool 必须通过部署期 `evidence_tools` 注入且转换为 read-only EvidenceBundle；
2. 新 RAG 必须实现 canonical `search_json`；
3. 新模型调用只能通过 `model.generate` Harness 工具进入 planning 或 evidence-bound synthesis。

## 7. 当前验证矩阵

验收分四层：

1. 黑盒：从 FinanceAnalysisService 输入自然问题，验证状态、工具、gap、预算与实体；
2. 白盒：直接替换 planner/provider/LLM/LangGraph checkpointer，验证内部不变量；
3. 故障注入：畸形输入、无界响应、网络拒绝、崩溃恢复、citation laundering、冲突；
4. 交付物：静态检查、覆盖率、PEP 517 wheel、隔离安装、CLI 和已安装包评测。

当前 2.2 基线：11/11 企业黑盒场景、140 项 pytest 测试、83.95% 总源码覆盖率（80% 阻断门），Ruff、mypy、
compileall、Compose YAML、PEP 517 构建和全新临时目录 wheel 导入全部通过。阶段性数字保留在各阶段记录中，
不应替代当前基线。

新增 RAG 复审专门验证：

- ACME 两页 PDF 只返回相关第 2 页；
- 未在静态词典中的单一显式实体仍正确绑定；
- 注入式内部 RAG 被 Planner 选择；
- 配置了 RAG 的纯 CAGR 仍只调用 `finance.calculate`；
- primary 空结果后 fallback 执行一次；
- external RAG 未授权时 client 调用次数为零；
- fixed tenant/ACL filters 无法被 Agent 覆盖；
- object content、list query、bool top_k 被拒绝而不强转；
- HTTP gateway 的 HTTPS、redirect、响应字节与 JSON 边界生效；
- API 引用显示用户文件名而非服务端 UUID。
- 原 PDF 删除后，同线程显式召回仍能检索解析页文本；跨 user namespace 不可见，显式删除后列表为空。
- 真实 DeepSeek V4 Flash 两 PDF 分析只调用模型一次，输出 2 条 supported claim，引用和后置验证全部通过。
- 真实 PaddleOCR-VL-1.6 单页扫描 PDF 正确返回页级 Markdown。

新增个人助手与上下文复审发现并解决：

- `SourceType.WEB` 未进入 ContextAssembler 优先级表，真实网页+LLM 路径会抛 `KeyError`；补齐类型并增加真实合成路径回归。
- 网页 tracking URL、fragment 和相同摘要可重复占位；增加 canonical URL/内容双去重，私网/IP 结果拒绝。
- 单一普通站点即可满足 web requirement；改为至少两个 domain，公共机构例外，同时 snippet claim 固定降级为 inferred。
- RAG 只按 entity/source 分组导致同一 PDF 的重叠 chunks 垄断上下文；增加 document/domain/provider origin 分组和问题附近窗口。
- 输出 token 与输入上下文混为一谈；分成规划 24K、生成 48K evidence 字符和 4096 输出 token，预算可配置且 manifest 可审计。
- `require_documents=true` 在没有 document provider 时被静默改成 false，且概念知识可能错误满足文档要求；保留显式 requirement，并排除 curated knowledge 伪装用户文档。
- 个人偏好若按内容 ID 写入会同时召回冲突版本；改为 kind+title 稳定槽位，由最新显式写入覆盖。
- 临时 PDF 与长期知识库容易发生隐式留存；拆为 request/session/personal 三套接口和生命周期，原文件始终请求后删除。
- “模型设计函数”若等同动态 Python 会突破安全边界；改为 AST 白名单声明式公式，验证执行安全与数值域，但不伪称语义正确。
- MCP/企业能力若直接暴露原始工具会带入副作用和任意 schema；部署注入只接受现有只读 capability + canonical EvidenceBundle。
- `httpx` 网络异常不在默认 retry exception 中，声明的 web/RAG 两次尝试实际只跑一次；显式加入 transport exceptions 并做行为测试。
- 初版多 PDF 修复无条件“每份文档一个名额”，把文档覆盖误当成问题相关性；改为默认全局相关排序，只有模型/明确多文档意图才启用 `diversify_documents`，并补“八份材料但只问一个数值”的反例。
- 旧路由在第一条 evidence 满足最低 coverage 后立即生成，模型实际上没有机会基于工具结果决定是否继续；ModelPlanner 现在必须在看到新 Evidence 后显式 finish，Validation 只提供最低证据门和硬预算，规则 planner 仍可直接收敛。
- 本地 adapter 默认发送 `search_mode=rrf`，但 corpus 实际始终只跑 BM25，接口语义和 trace 不一致；现在把 lexical
  与 hybrid 拆成不同 ToolSpec，只有配置 embedding 才注册 hybrid，执行真实 cosine + RRF，未配置 reranker 时快速失败。
- 真实模型用 hybrid 后再调用 lexical 命中同一 chunk，两份 Evidence 因搜索模式 tags 不同而同 ID 冲突；将模式
  留在 provenance/observation，语义 tags 固定，跨检索路径幂等去重，真实 DeepSeek 复测成功。

最终打包与安装后验证结果同时记录在 `IMPLEMENTATION_STATUS.md`。

## 8. 仍然存在的边界

### P0：上线前必须解决

1. 用 OIDC/JWT 或可信网关注入 principal，贯穿 job、memory、RAG filter 和 artifact ACL；
2. 用有许可、SLA、时效和 corporate-action 口径的行情源替换实验性 Yahoo；
3. 用带 lease、visibility timeout、幂等和死信的可靠队列替换 Redis list；
4. checkpoint、memory、upload 和 artifact 加密，建立 retention/delete/export；
5. 为新闻、网页和内部文档建立许可、ACL、索引版本和删除传播。

### P1：可靠性与质量

1. provider 改为 async 或进程隔离，支持真实取消；
2. 加入共享 rate limiter、circuit breaker、cache freshness 和 schema drift 录制回归；
3. 建立金融 RAG relevance set，测 Recall@k、nDCG、citation precision 和 no-answer accuracy；
4. 将远程 OCR 调用移入可取消的 worker/任务队列，并补供应商 retention、删除、区域和敏感文档政策；
5. 引入 tokenizer-aware context budget 与数值 claim parser。

### P2：高级能力

1. 在金融标注集上选择 embedding 模型并验证当前 hybrid BM25/vector/RRF；收益达标后再引入 reranker；
2. 增加 earnings call、新闻、内部 SQL/warehouse 的固定契约 adapter；
3. 对高风险结论增加 NLI/规则校验和人工抽检；
4. 建立 append-only audit、OpenTelemetry、成本/token/provider 配额和 SLO。

## 9. 以后如何避免重新走回旧路

新增能力前依次回答：

1. 它满足哪个用户 requirement？
2. 输入是否来自可信控制面，还是不可信数据面？
3. 输出如何转换为 SourceRef/Evidence？
4. 来源、时间、期间、单位和 entity 如何定位？
5. 网络、副作用、预算和 retry 如何声明？
6. 空数据、冲突、超时和 schema drift 时是什么状态？
7. 是否真的需要 LLM，还是确定性规则更可靠？
8. 记忆的 owner、TTL、查看、删除和撤回是什么？
9. 至少有哪些黑盒、白盒和故障注入用例？
10. 删除这个能力后，核心 Agent 是否仍然清晰？

如果这些问题没有答案，就不应通过增加一个“节点”、一个“Agent 角色”或一个“长期记忆表”掩盖设计空缺。
