# MAS Finance 工具、金融场景与自适应调用逻辑

状态：与当前实现同步
日期：2026-08-27
定位：说明 Agent 会在什么场景调用什么工具、如何形成可审计的研究策略，以及每类工具的输入、输出与安全边界。

## 1. 核心原则

MAS Finance 不为每种问题维护固定 workflow，也不允许 LLM 绕过工具边界。它采用“模型自主选择 + 确定性校验”模式：

```text
问题、会话摘要、最近事件与实体回放
  → LLM TaskFrame（LLM 未配置时快速失败）
  → ResearchScope(requirements + success criteria)
  → ModelPlanner 读取动态 ToolSpec 目录（MCP 仅短索引 + 发现元工具）
  → 每轮最多选择四个已注册工具或 finish
  → ToolHarness 配套校验并执行这些动作
  → EvidenceBundle
  → CoverageAssessor
  → 缺口驱动的重规划
  → 证据约束合成与硬校验
```

所谓“思考”在系统中表现为可持久化的结构化决策，不保存或暴露隐藏 chain-of-thought：

- `ResearchScope.rationale`：为什么识别出这些研究意图和需求；
- `ResearchRequirement.reason`：为什么需要某类证据；
- `ResearchPlan.rationale`：本轮为什么选择这些任务；
- `ToolTask.reason`：单个工具任务的目的；
- `CoverageDecision`：哪些需求已满足、哪些仍缺失；
- `ResearchGap`：失败、配置缺失或 provider 缺失的可见原因。

这些字段随 checkpoint 和 API state 返回，适合审计和调试，但不是模型的私有推理文本。

## 2. 当前工具总览

| 工具 | Capability | 网络 | 何时注册/可用 | 主要作用 |
|---|---|---:|---|---|
| `finance.calculate` | `calculation` | 否 | 始终 | 白名单确定性金融计算 |
| `corpus.search` | `document.search` | 否 | 当前上传或显式会话 PDF | request/session 文档检索 |
| `<deployment>.search` | `document.search` | 由来源声明 | 部署注入时 | 内部 RAG、licensed news 或外部搜索 gateway |
| `market.snapshot` | `market.read` | 视 provider | 有 entity 时 | 当前价格、PE、市值等快照 |
| `market.history` | `market.read` | 视 provider | 有 entity 时 | 历史价格、收益、波动率、回撤 |
| `sec.company_facts` | `regulatory.read` | 是 | 配置 SEC User-Agent 且有 entity | SEC XBRL 基本面事实 |
| `sec.recent_filings` | `regulatory.read` | 是 | 配置 SEC User-Agent 且有 entity | 最近申报表和主文档 locator |
| `macro.fred_series` | `macro.read` | 是 | 配置 FRED API key | FRED 宏观时间序列 |
| `web.search` | `web.search` | 是 | 配置 Bocha 或 Brave key | 模型自主开放检索，返回可引用网页 snippets |
| `finance.formula` | `calculation` | 否 | 始终 | 安全执行模型/用户给出的声明式公式；不执行代码，语义结果标为 inferred |
| `personal.search` | `document.search` | 否 | 当前用户已有个人文档 | 检索明确持久上传的个人 PDF 页文本 |
| deployment evidence tool | 既有只读能力 | 由工具声明 | 部署注入 | 手工注入的企业 RAG；必须返回 canonical EvidenceBundle |
| MCP Host 接入工具 | 既有只读能力 | 由 server 声明 | `MAS_MCP_SERVERS` 或 AllTick/必盈自动挂载 | 外部 MCP；模型通过渐进发现调用，不进 LLM `tools` 字段 |
| `mcp.search_tools` / `mcp.describe_tool` / `mcp.call_tool` | `mcp.discover` / `mcp.invoke` | 视被调工具 | 已连接至少一个 MCP 工具时 | Host 侧渐进发现：短索引 → 契约 → 受控执行 |
| `llm.task_frame` | `model.generate` | 是 | LLM 必需 | 根据当前请求与会话记忆生成 TaskFrame |
| `llm.plan` | `model.generate` | 是 | LLM 必需 | 从动态目录选择最多四个下一动作或 finish |
| `llm.synthesize` | `model.generate` | 是 | LLM 必需 | 从证据生成受约束 claims；输出非法则快速失败 |

运行中的实际工具列表取决于配置和请求。API 可以查询：

```http
GET /api/v1/tools
```

返回 capability、网络属性、availability、严格 input/result contract、可见性，以及行情 provider 的 support tier。
`llm.plan`/`llm.synthesize` 是内部模型调用；研究模型看到的是数据工具目录，而不是把模型工具递归提供给自己。

核心工具输入先由 Harness 的 `ToolArgumentContract` 检查：必填字段、多余字段、有限 JSON 和序列化大小均在消耗预算/访问 provider 前验证。数据工具输出必须能反序列化为 `EvidenceBundle`；模型工具输出必须是有界 model response。输入/输出契约失败都产生稳定错误码。

所有研究工具都是 `read_only`。系统没有 broker、order、transfer 或任何 `financial_transaction` 工具。

## 3. 问题场景如何映射到需求

TaskFrame 模型使用当前请求、会话摘要、最近事件和实体回放形成 `ResearchScope`；它可以声明历史实体来源或要求澄清，不能由规则静默指定指代。LLM 是研究链路必需依赖。

| 用户场景 | Intent | Requirement | 首选工具 |
|---|---|---|---|
| “什么是市盈率？” | `financial_education` | 无检索 requirement | 模型直接合成；引用了才做 quote 校验 |
| “100 到 150 的三年 CAGR” | `calculation` | `calculation:<request_id>` | `finance.calculate` |
| “Apple 当前价格与 PE” | `market_snapshot`、`valuation` | `market:Apple`，必要时 `regulatory:Apple` | market + SEC |
| “Apple 过去五年波动率和最大回撤” | `market_performance` | `market_history:Apple` | `market.history` |
| “比较 A、B 盈利和杠杆” | `comparison`、`profitability`、`solvency` | 每个 entity 的 regulatory requirement | `sec.company_facts` |
| “Apple 最近 8-K 和 10-Q” | `regulatory_filings` | `filings:Apple` | `sec.recent_filings` |
| “美国通胀和失业率如何？” | `macroeconomics` | `macro:CPIAUCSL`、`macro:UNRATE` | `macro.fred_series` |
| “这份 PDF 里管理层如何解释风险？” | `document_research`、`risk` | `document:<entity/query>` | `corpus.search` |

识别结果不是最终事实。它只决定“需要什么证据”，事实仍必须来自工具返回的 Evidence。

### 3.1 显式参数优先

调用方可通过 API 明确覆盖自动判断：

- `require_market_data`
- `require_market_history`
- `require_regulatory_data`
- `require_documents`
- `market_history_range`
- `market_history_interval`
- `macro_series`
- `calculations`

布尔值含义：

- `true`：强制建立对应 requirement；
- `false`：即使关键词命中也禁止该 requirement；
- `null`：由 QueryAnalyzer 判断。

### 3.2 一个 Scope 示例

问题：“比较 Apple 与 Microsoft 过去五年的盈利能力、波动率与最大回撤。”

```json
{
  "intents": [
    "comparison",
    "market_performance",
    "profitability"
  ],
  "requirements": [
    {
      "key": "market_history:Apple",
      "category": "market_history",
      "entity": "Apple",
      "fields": ["total_return", "annualized_volatility", "max_drawdown"],
      "parameters": {"range": "5y", "interval": "1d"}
    },
    {
      "key": "regulatory:Apple",
      "category": "regulatory",
      "entity": "Apple",
      "fields": ["net_income", "revenue"]
    }
  ]
}
```

Microsoft 会形成同构 requirement。ModelPlanner 每个 graph step 最多选择四个动作并在本轮规划节点内执行；
provider 失败、空结果或 coverage 不足时，validation 把状态送回 planning。

## 4. ModelPlanner 的自主选择

ModelPlanner 读取用户请求、intent hints、coverage、`prior_actions`、gaps、有界 evidence 摘要、当前 ToolSpec
目录、MCP 短索引、发现结果和 `verified_tool_usage`，返回严格 JSON `call_tool`、`call_tools` 或 `finish`。
模型挑选工具；LLM 未配置或其结构化响应非法时快速失败，不走规则降级。DeepSeek 请求不带 native `tools` 字段。模型重复相同 tool+arguments 时稳定 task ID 去重；模型过早 finish 时 validation 拒绝并继续规划。

### 4.1 MCP 渐进发现

具体 MCP 工具名不进入 `available_tools`，只以短 `mcp_tool_index`（name、capability、200 字描述、
planner_category）出现。模型必须：

1. 需要时用 `mcp.search_tools` 按关键词缩小候选；
2. 执行前用 `mcp.describe_tool` 拉取完整 JSON Schema（含服务端提供的字段描述、enum、default、examples）；
3. 用 `mcp.call_tool` 执行，`name` 必须是 index 中的本地名（例如 `extmarket.snapshot`）。

`mcp.describe_tool` / `mcp.search_tools` 的结果不写入 EvidenceBundle；Graph 对 `mcp.discover` 只把结果留给下一轮
`discovery_results`。计算、内部研报 RAG、request/session/personal corpus 不经 MCP。

成功的 MCP 调用参数不会写入 personal memory，而是按工具名 + input-schema fingerprint + arguments 写入用户隔离的
`tool_usage_memory`。召回时 schema 必须一致，最多五条以 `verified_tool_usage` 进入规划上下文，仍须服从当前契约。

### 4.2 报错之后的策略

“工具失败”分四层，不要当成同一种重试：

| 层 | 何时发生 | 系统做什么 | 规划侧下一步 |
|---|---|---|---|
| Harness 自动重试 | 只读工具抛出 `ToolSpec.retry` 声明的异常（默认 `TimeoutError`/`ConnectionError`） | 同一 call 内再尝试；web/RAG/FRED/SEC 多为 2 次；每个网络 attempt 占数据预算 | 仍失败才进入 observation |
| MCP 结构化错误 | server `isError=true` → `ToolExecutionError` | **不**自动重试（MCP 绑定工具默认 `max_attempts=1`）。`ok=false`，`error_details` 可含 `retryable`、`suggested_action`、`received_arguments`，以及 server 提供的 `field`/`candidates` | 写入 `prior_actions` 与 resolvable gap；模型应改参后再 `mcp.call_tool` |
| 契约/发现错误 | 缺字段、未知 MCP 名、非法 JSON-RPC | 普通 `ValueError` / 契约错误码；发现失败不是 `isError` 契约 | 下一轮换工具名或先 `describe_tool` |
| 空结果但仍 `ok=true` | adapter 返回空/不完整 bundle + `gaps`（例如行情查到但序列不够） | 合并已有 Evidence，coverage 仍缺 | 换参数或换 provider；**不会**自动把 `AAPL` 改成 `AAPL.US` |

完全相同的 tool+arguments 形成相同 task ID：Graph 记 `repeated_planner_action`，不再执行。改一个参数就是新任务。
`retryable=false`（例如缺供应商凭据）提示停止改参并报告配置问题。限流等待超时表现为一次 `TimeoutError`，不是盲打。

内置 `extmarket` 将超时/连接映射为 `provider_transient_error`，缺凭据映射为 `provider_configuration_error`，
其余参数问题映射为 `invalid_or_unresolved_arguments`。Host 会转发结构化字段，但 `extmarket` 本身不保证每次都返回枚举 `candidates`。

### 4.3 不再提供规则规划降级

LLM 未配置、`llm.plan` 失败或模型 JSON 非法时，研究请求快速失败。系统不再用 AdaptivePlanner
按 category 枚举 provider。MCP 行情补充（如 `extmarket`）只通过 §4.1 渐进发现进入规划目录。
相同 tool+arguments 仍由稳定 task ID 去重；没有可满足缺口的工具时记 `requirement_provider_unavailable`。

内置工具与 MCP 的职责划分：

```text
document       corpus.search / hybrid → 部署注入的 RetrievalSource → MCP document 工具
market         market.snapshot → planner_category=market 的 MCP 工具
market_history market.history → planner_category=market_history 的 MCP 工具
regulatory     sec.company_facts → planner_category=regulatory 的 MCP 工具
filings        sec.recent_filings → planner_category=filings 的 MCP 工具
macro          macro.fred_series → planner_category=macro 的 MCP 工具
calculation    finance.calculate
web            web.search → planner_category=web 的 MCP 工具
```

## 5. 工具详细说明

### 5.1 概念解释（无内置词库）

教育类、定义类、机制类问题**不再**调用代码内金融词条。TaskFrame 可以给出空的 requirements，规划器应 `finish`，由 `llm.synthesize` 直接作答。这类 claim 标为 `inferred`，并注明未经检索核验。

证据校验仍然约束真正的引用：一旦 `evidence_ids` 非空，`evidence_quote` 必须是对应 evidence 正文中的逐字子串，且 ID 必须在本次 manifest 内。文档、行情、监管、宏观、网页和计算事实不能靠模型常识补造。

个人知识库（`personal.search`）与代码词库不是一回事；用户上传的 PDF 仍走检索。

### 5.2 finance.calculate

不接受 Python、SQL、表达式字符串或任意代码，只接受 `MetricOperation` 白名单。

| Operation | 公式/语义 | 必需输入 |
|---|---|---|
| `ratio` | numerator / denominator | numerator、denominator |
| `percentage_change` | end / beginning - 1 | beginning_value、ending_value |
| `cagr` | (end / beginning)^(1/years) - 1 | beginning_value、ending_value、years |
| `future_value` | PV × (1+r)^n | present_value、rate、periods |
| `present_value` | FV / (1+r)^n | future_value、rate、periods |
| `loan_payment` | 等额支付公式 | principal、rate、periods |
| `annualized_return` | 复合周期收益年化 | returns、annualization_factor |
| `annualized_volatility` | 样本标准差 × sqrt(factor) | returns、annualization_factor |
| `sharpe_ratio` | (年化收益-rf)/年化波动 | returns、annualization_factor、可选 risk-free |
| `max_drawdown` | min(value/running_peak-1) | values |

工具会创建两类证据：

1. `USER_INPUT`：保留调用方提供的数值；
2. `CALCULATION`：保留公式版本、request ID、input evidence IDs 和结果。

Validator 会拒绝没有输入 evidence IDs 或引用未知输入的 calculation evidence。长数列在审计中只保存 hash 和长度。

结构化 API 示例：

```json
{
  "query": "计算三年 CAGR",
  "calculations": [
    {
      "operation": "cagr",
      "inputs": {
        "beginning_value": 100,
        "ending_value": 150,
        "years": 3
      },
      "label": "three_year_cagr"
    }
  ]
}
```

自然语言仅解析无歧义的命名输入，例如：

```text
计算 CAGR，beginning=100, ending=150, years=3
```

复杂计算应使用结构化参数，避免猜测数值含义。

这里刻意没有让 LLM 执行算术或代码。自然语言中的无歧义参数可由规则转成 `MetricRequest`，
ModelPlanner 也可以生成 `finance.calculate` 的结构化参数；无论参数来自调用方、规则还是模型，都必须先通过
operation 白名单、字段集合、严格类型、有限数值、数值域、单位兼容和大小上限，再由函数执行并登记公式版本与输入 evidence ID。
对内置 operation 无法表达的计算，模型可选择 `finance.formula` 提供声明式表达式；安全 AST 只能保证有限数值
执行和可复算，不能保证金融语义，最终 claim 因此标为 inferred。

### 5.3 corpus.search / corpus.hybrid_search

搜索本次请求或会话上传的 PDF corpus：

- chunk 默认 1600 字符、重叠 200；
- 支持中英文 token；
- `corpus.search` 固定执行 BM25；配置 embedding 后额外注册 `corpus.hybrid_search`，固定执行 BM25 + cosine + RRF；
- 模型可选择两个工具，但不能在 lexical 工具中用参数偷换为会联网的 hybrid；
- 支持 top-k、metadata filters，以及由模型/明确综合意图选择的 `diversify_documents`；内置 corpus 未配置 reranker；
- 返回 file/page/chunk locator；
- 返回 BM25/cosine/各路 rank/RRF trace，排名分数不会冒充置信概率；
- 文档内容被视为不可信数据，不能改变工具权限。

个人文档同样按 `personal.search / personal.hybrid_search` 拆分。request/session 向量只在当次 corpus 实例缓存；
个人 SQLite 当前只持久化页文本，embedding 在每次分析快照中计算和复用。完整设计见
[BM25 + Embedding 双路检索设计](HYBRID_RETRIEVAL.md)。企业持久知识库仍需 ACL、tenant filter、索引版本和删除传播。

部署可通过 `RetrievalSource` 注入多个内部或外部检索源。Planner 先使用上传 corpus，再按注册顺序尝试尚未调用的来源；空结果不会被当作 coverage。`fixed_filters` 在服务端绑定并覆盖调用参数，用于 KB/tenant/ACL 边界。远端源仍需网络双重授权。

“来源已配置”不等于“每题都要检索”。上传文档一定形成 document requirement；没有上传时，只有明确的文档/内部资料/新闻/搜索语义或 API `require_documents=true` 才启用注入式 RAG。纯计算和概念解释不会因部署了知识库而增加检索调用。

`HTTPJSONRAGClient` 提供固定 canonical gateway：endpoint 只能由部署配置，默认必须 HTTPS，不跟随 redirect，限制解压后响应字节，并严格校验 JSON、chunk、metadata 和 trace。Agent 不提供任意 URL 抓取工具。

开放联网有两条明确边界：企业/授权检索继续使用固定 `RetrievalSource` gateway；公开网页使用 `web.search`：

```text
open-web requirement or model decision
  → ModelPlanner 生成 query/count/freshness/domains
  → Harness 检查 capability、双重网络授权、预算、timeout/retry
  → WebSearchEvidenceAdapter 调用配置的 provider（当前优先 Bocha，Brave 可选）
  → 校验公开结果 URL、标题、snippet 和发布时间
  → 转为 SourceType.WEB Evidence，标记 search_result_snippet
  → coverage、冲突检查和引用约束合成
```

固定搜索 API origin 是凭据/SSRF 边界，不是固定研究路线。模型可以自主决定 query 和域名，但不能直接
`GET model_supplied_url`。当前不提供通用 web.fetch；重要结论应优先调用 SEC/FRED/filing/企业 RAG 等原始来源。

### 5.4 market.snapshot

字段级输出：

- current price
- one-month return
- market capitalization
- trailing P/E
- P/B、P/S、EV/EBITDA
- 52-week high/low

每个字段分别形成 Evidence。缺字段、缺 as-of 和 provider 不可用都会产生 gap。

### 5.5 market.history

支持范围：`1mo`、`3mo`、`6mo`、`1y`、`2y`、`5y`、`10y`。
支持间隔：`1d`、`1wk`、`1mo`。

工具读取带明确 `price_basis` 的价格序列，至少需要三个有效点，然后生成：

- history start/end price；
- total return；
- annualized return；
- annualized volatility；
- maximum drawdown。

标准化后的完整观察序列会登记为 calculation-input Evidence；派生指标引用该序列、起点和终点证据，并保存公式版本与 input evidence IDs。若 provider 只有 raw close，指标仍可用但生成 `unadjusted_price_history` gap，明确不包含现金分红。

默认 provider 是 `offline`。AlphaVantage 与 Yahoo 必须显式选择，配置失败时不会跨 provider 静默 fallback。当前 Yahoo endpoint 没有本项目可依赖的正式契约，工具目录标记为 `experimental_non_contractual`；AlphaVantage 当前历史实现使用 raw `TIME_SERIES_DAILY`。生产应使用有许可、SLA、时效和 corporate-action 口径的供应商。配置 `ALLTICK_TOKEN` 或 `BIYING_LICENCE` 时，进程额外挂载 MCP `extmarket`；模型经 §4.1 渐进发现调用。空行情通常是 `ok=true` + gaps，不是 `isError`。

### 5.6 sec.company_facts

读取 SEC Company Facts XBRL，目前规范化：

- revenue、gross profit、operating income、net income；
- total/current assets；
- total/current liabilities；
- shareholders' equity、cash；
- operating cash flow、capital expenditure；
- diluted EPS。

Agent 在同实体、同期间、同单位且无冲突时确定性派生：

- net/gross/operating margin；
- liabilities/assets、debt/equity、equity/assets；
- cash/assets、current ratio。

收入、利润等 duration facts 使用 `start/end` 期间，资产负债表 instant facts 使用单一 date。系统不会把期末资产或权益冒充平均资产/权益自动计算 ROA、ROE 或 asset turnover；只有以后取得可对齐的平均余额证据时才应生成这些指标。

SEC facts 端点和更新时间说明见 [SEC EDGAR API 官方文档](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。

### 5.7 sec.recent_filings

读取 SEC submissions recent filings，支持表单：

```text
10-K, 10-Q, 8-K, 20-F, 40-F, 6-K
```

输出 filing date、report date、accession、description 和 primary-document locator。该工具当前检索的是申报元数据，不会自动下载并全文解析 filing；如需内容分析，应由后续 filing reader 或文档 ingestion 工具完成。

### 5.8 macro.fred_series

支持显式 FRED series ID，也能从常见中英文问题映射：

| 主题 | Series |
|---|---|
| CPI/通胀 | `CPIAUCSL` |
| 失业率 | `UNRATE` |
| GDP | `GDP` |
| 联邦基金利率 | `FEDFUNDS` |
| 10 年/2 年美债收益率 | `DGS10` / `DGS2` |
| 非农就业 | `PAYEMS` |
| PCE 价格指数 | `PCEPI` |
| 30 年抵押贷款利率 | `MORTGAGE30US` |

工具读取 series metadata 与 observations，生成最新值及相邻观测变化。原始序列、前值和最新值分别形成证据；变化值是带输入 ID 的 calculation Evidence。缺失值 `.` 会跳过。FRED v1 请求需要 API key；参数和返回结构依据 [FRED 官方 observations 文档](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)。

### 5.9 llm.plan 与 llm.synthesize

`llm.plan` 负责结构化下一动作选择，`llm.synthesize` 在证据形成后负责语言合成。合成 Prompt 使用分层上下文：

```text
policy/trust boundary
task(query, entities, language)
research(scope, coverage, unresolved gaps, stop reason)
minimal thread context
balanced evidence cards with source provenance
output JSON contract
```

Evidence cards按 `entity × source_type` 分组后轮询选择，避免第一家公司或第一类来源独占上下文；同时按 query overlap、结构化程度和来源类型排序。每张卡包含 provider、locator、source type、period、as-of 和 published-at。

`ContextManifest` 明确记录真正进入 Prompt 的 evidence ID；模型不能引用被预算裁掉的证据。逐字 quote 只保留确实包含该 quote 的 citation，不能借一个有效 quote 附带无关来源。合成 JSON 非法时快速失败，不回退到确定性复述。缺少 LLM 配置同样使研究请求失败。

## 6. Coverage 如何判断工具是否真的完成任务

“工具调用成功”不等于“研究需求已满足”。Coverage 按 requirement 检查：

- entity 是否一致；
- source type/tag 是否符合类别；
- 所需 field 是否都存在；
- calculation request ID 是否一致；
- knowledge concept 是否匹配；
- market snapshot 与 market history 不相互冒充。

例如市场 provider 只返回 current price，但估值 requirement 还要求 market cap 和 trailing P/E，Coverage 仍保持 incomplete，报告显示缺口。

## 7. 持久对话记忆如何影响工具选择

同一个 `thread_id` 的下一次调用会读取有界对话投影：

```json
{
  "summary": {"conversation_summary": "", "user_goals": [], "requirements": [], "decisions": [], "completed_work": [], "successful_tools": [], "failed_tools": [], "unfinished_work": [], "open_questions": []},
  "recent_events": [{"kind": "user_message", "content": "分析 Apple 的估值", "occurred_at": "..."}],
  "atomic_facts": [{"event_id": "fact-1", "kind": "atomic_fact", "content": "用户要求分析 Apple 的估值。", "occurred_at": "...", "entities": ["Apple"], "payload": {"status": "requested", "source_event_ids": ["event-1"]}}],
  "entity_state": {"Apple": {"mention_count": 1, "symbol": "AAPL", "last_sequence": 1}},
  "focus_history": [{"sequence": 1, "entities": ["Apple"]}],
  "focus_entities": ["Apple"],
  "run_state": [{"run_id": "run-1", "status": "completed", "tools": []}],
  "manifest": {"max_context_tokens": 300000, "max_recent_context_tokens": 20000, "full_history_persisted": true, "memory_is_evidence": false}
}
```

用途：

- “那它的最大回撤呢？”可以继承 Apple/AAPL；
- TaskFrame 依据全历史原子事实、摘要和最近事件解析历史实体；歧义时返回澄清问题；
- 旧请求、结果、工具状态和 gap 可帮助多轮理解，但不能作为事实 evidence。

不会保存到对话账本：

- 原始 PDF；
- EvidenceBundle；
- 模型 prompt；
- chain-of-thought；
- API key/token；
- 自动生成的长期用户画像。

删除接口：

```http
DELETE /api/v1/conversations/{thread_id}
```

对话记忆是严格类型的持久事件账本，旧事件在 prompt 中按预算滚动压缩，原子事实不参与摘要并全历史回放，数据库记录保留到显式删除；它永远不是 Evidence。个人长期记忆保存 profile/preference/experience，明确长期更新可覆盖旧值，临时要求不得沉淀。成功工作路径进入独立 Learned Skill，并在 TaskFrame 选择后才向 Planner 披露完整步骤。个人 PDF 只有独立持久上传接口才入库，临时上传不会自动 promotion。HTTP 层仍是单部署 API-key 身份边界，多用户上线前必须增加可信 principal、导出和 retention。完整设计见 [记忆与日志](CONVERSATION_MEMORY.md) 与 [个人助手边界](PERSONAL_ASSISTANT_MEMORY_AND_CONTEXT.md)。

## 8. 网络、权限和失败行为

联网需要：

```text
MAS_ALLOW_NETWORK=true AND request.allow_network=true
```

此外：

- SEC 需要 `MAS_SEC_USER_AGENT`；
- FRED 需要 `FRED_API_KEY`；
- AlphaVantage 需要 `ALPHAVANTAGE_API_KEY`；
- DeepSeek 需要 `DEEPSEEK_API_KEY`。

配置缺失时，相关工具不注册；Scope 若仍要求该数据，最终产生 `requirement_provider_unavailable`。网络未授权时，Harness 返回 `network_denied`。二者都不会被模型静默掩盖。

研究循环里的失败分层见 §4.2。补充边界：

- 只有 read-only 工具允许 Harness 自动重试；契约错误、`ToolExecutionError`、HTTP 4xx/429（未映射为 retryable 异常时）和无结果不会盲重试。
- 网络 transport 超时/连接失败时，web/RAG/FRED/SEC 通常再试一次，两次 attempt 分别计入数据预算。
- MCP `tools/call`、FRED、Bocha、Brave 与内置行情另有每分钟滑动窗口限流；限流等待超时记为一次失败。
- `mcp.discover` 失败或未知工具名不会伪造 Evidence。

## 9. CLI 示例

历史风险分析：

```bash
mas-finance \
  --query "分析 Apple 过去五年的收益、波动率和最大回撤" \
  --entity Apple \
  --symbol Apple=AAPL \
  --require-market-history \
  --market-range 5y \
  --allow-network
```

宏观数据：

```bash
mas-finance \
  --query "比较美国通胀和失业率" \
  --macro-series CPIAUCSL \
  --macro-series UNRATE \
  --allow-network
```

结构化计算：

```bash
mas-finance \
  --query "计算三年 CAGR" \
  --calculate '{"operation":"cagr","inputs":{"beginning_value":100,"ending_value":150,"years":3}}'
```

## 10. 当前边界

目前仍缺少：

- 新闻、earnings call、分析师预期和正式商业行情 adapter；
- 内部 SQL/data warehouse 只读工具；
- filing 全文下载、HTML 清洗和段落级检索；
- 多币种换算、收益率曲线、债券现金流和期权定价工具；
- 基于真实 token 的上下文预算；
- 语义蕴含/数值 claim parser；
- 安全通用 web.fetch 与原始网页正文证据链。

增加这些能力时仍应遵循：provider client → Evidence adapter → ToolSpec → 动态目录 → 必要的 Coverage rule → tests。
外部 API 也可先做成 MCP server，再由 Host 过滤后进入同一条 Harness 目录。配置 `ALLTICK_TOKEN` 或
`BIYING_LICENCE` 时，进程会自动挂载本地 `extmarket` stdio server（snapshot/history），并经渐进发现调用。FRED、Bocha、Brave、行情与
MCP `tools/call` 都有每分钟滑动窗口限流；不要用免费档密钥做压测。官方 SEC EDGAR 仍只用
`MAS_SEC_USER_AGENT`，第三方 SEC token 尚未接入。
只注册一个函数而不补证据契约，不算完成接入。
