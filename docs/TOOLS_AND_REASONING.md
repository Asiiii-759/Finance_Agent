# MAS Finance 工具、金融场景与自适应调用逻辑

状态：与当前实现同步
日期：2026-08-12
定位：说明 Agent 会在什么场景调用什么工具、如何形成可审计的研究策略，以及每类工具的输入、输出与安全边界。

## 1. 核心原则

MAS Finance 不为每种问题维护固定 workflow，也不允许 LLM 绕过工具边界。它采用“模型自主选择 + 确定性校验”模式：

```text
问题与显式参数
  → FinancialQueryAnalyzer
  → ResearchScope(intents + requirements + calculations)
  → ModelPlanner 读取动态 ToolSpec 目录
  → 每次选择一个已注册工具或 finish
  → ToolHarness 配套校验并执行该动作
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
| `finance.knowledge` | `knowledge.read` | 否 | 始终 | 金融概念、公式与解释 caveat |
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
| deployment evidence tool | 既有只读能力 | 由工具声明 | 部署注入 | 企业 RAG/MCP gateway；必须返回 canonical EvidenceBundle |
| `llm.plan` | `model.generate` | 是 | 配置模型密钥时 | 从动态目录选择一个下一动作或 finish |
| `llm.synthesize` | `model.generate` | 是 | 配置模型密钥时 | 从证据生成受约束 claims；未配置时直接使用本地确定性合成器 |

运行中的实际工具列表取决于配置和请求。API 可以查询：

```http
GET /api/v1/tools
```

返回 capability、网络属性、availability、严格 input/result contract、可见性，以及行情 provider 的 support tier。
`llm.plan`/`llm.synthesize` 是内部模型调用；研究模型看到的是数据工具目录，而不是把模型工具递归提供给自己。

核心工具输入先由 Harness 的 `ToolArgumentContract` 检查：必填字段、多余字段、有限 JSON 和序列化大小均在消耗预算/访问 provider 前验证。数据工具输出必须能反序列化为 `EvidenceBundle`；模型工具输出必须是有界 model response。输入/输出契约失败都产生稳定错误码。

所有研究工具都是 `read_only`。系统没有 broker、order、transfer 或任何 `financial_transaction` 工具。

## 3. 问题场景如何映射到需求

`FinancialQueryAnalyzer` 使用中英文金融词汇、显式请求参数、entity 数量和上传文档状态形成 `ResearchScope`。

| 用户场景 | Intent | Requirement | 首选工具 |
|---|---|---|---|
| “什么是市盈率？” | `financial_education`、`valuation` | `knowledge:pe_ratio` | `finance.knowledge` |
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

Microsoft 会形成同构 requirement。ModelPlanner 每个 graph step 选择一个动作；provider 失败、空结果或 coverage
不足时，validation 把状态送回 planning，由模型结合历史结果重新选择，不按预设节点链执行。

## 4. ModelPlanner 的自主选择

ModelPlanner 读取用户请求、intent hints、coverage、历史动作、gaps、有界 evidence 摘要和当前 ToolSpec 目录，
返回严格 JSON `call_tool` 或 `finish`。模型决定工具、参数、检索式、时效和先后顺序；ToolSpec/Harness 决定动作
是否合法。模型重复相同 tool+arguments 时稳定 task ID 去重；模型过早 finish 时 validation 拒绝并继续规划；
工具返回后最低 coverage 即使已满足，模型仍会基于新 Evidence 明确决定继续检索还是 finish。

### 4.1 AdaptivePlanner 降级规则

没有模型或模型计划违反契约时，AdaptivePlanner 才读取：

- 已持久化的 `ResearchScope`；
- 当前 `CoverageDecision.missing`；
- 已完成 ToolTask 的稳定 ID；
- 当前 Harness 的 ToolSpec 注册表。

决策顺序：

1. 只处理尚未覆盖的 requirement；
2. 为该 category 查询有序 provider 列表；
3. 跳过未注册工具；
4. 跳过相同参数下已尝试的任务；
5. 生成带 requirement key 的 ToolTask；
6. Graph 再次检查工具名和 capability allowlist；
7. Harness 绑定 run identity 与预算上限，检查副作用、网络、输入契约和分账预算；
8. Harness 校验输出契约；Agent 再计算 coverage，而不是无条件进入下一节点。

若不存在可满足缺口的工具，系统生成 `requirement_provider_unavailable` gap 并停止，不会让模型伪造数据。

当前 provider 顺序：

```text
document       corpus.search → 按部署顺序注入的 RetrievalSource
market         market.snapshot
market_history market.history
regulatory     sec.company_facts
filings        sec.recent_filings
macro          macro.fred_series
calculation    finance.calculate
knowledge      finance.knowledge
web            web.search
```

## 5. 工具详细说明

### 5.1 finance.knowledge

用于定义、公式与解释，不使用模型常识填空。当前版本包含：

- P/E、P/B、EV/EBITDA
- 净利率、ROE、ROA
- 流动比率、债务权益比
- CAGR、波动率、Sharpe、最大回撤
- DCF、NPV、IRR、债券久期
- 利率向企业与银行股传导的双向机制

每条内容带 `knowledge://finance/<concept>/v1` locator 和 `MAS Finance curated knowledge` provider。它是版本化内部知识，不冒充监管机构或实时市场事实。

输入：

```json
{"query": "什么是 ROE？", "concepts": ["roe"], "top_k": 1}
```

输出：document EvidenceBundle；未匹配产生 `finance_knowledge_not_found`。

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

默认 provider 是 `offline`。AlphaVantage 与 Yahoo 必须显式选择，配置失败时不会跨 provider 静默 fallback。当前 Yahoo endpoint 没有本项目可依赖的正式契约，工具目录标记为 `experimental_non_contractual`；AlphaVantage 当前历史实现使用 raw `TIME_SERIES_DAILY`。生产应使用有许可、SLA、时效和 corporate-action 口径的供应商。

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

`ContextManifest` 明确记录真正进入 Prompt 的 evidence ID；模型不能引用被预算裁掉的证据。逐字 quote 只保留确实包含该 quote 的 citation，不能借一个有效 quote 附带无关来源；不合规则确定性降级。未配置模型密钥不是故障：系统不注册模型工具，直接使用 `DeterministicSynthesizer`，因此不会产生虚假的 LLM fallback gap。

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
  "summary": {"user_requests": [], "assistant_outcomes": [], "tool_activity": []},
  "recent_events": [{"kind": "user_message", "content": "分析 Apple 的估值", "occurred_at": "..."}],
  "focus_entities": ["Apple"],
  "relations": [{"subject": "Apple", "predicate": "has_symbol", "object": "AAPL"}],
  "manifest": {"full_history_persisted": true, "memory_is_evidence": false}
}
```

用途：

- “那它的最大回撤呢？”可以继承 Apple/AAPL；
- 只有明确指代才继承历史实体；当前问题显式或检测到的新实体优先；
- 前者/后者/复数按最近有序实体组解析，歧义单数不猜；
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

对话记忆是严格类型的持久事件账本，旧事件在 prompt 中按预算滚动压缩，数据库记录保留到显式删除；它永远不是 Evidence。个人长期记忆只通过显式 CRUD 保存 profile/preference/experience/skill，同类同标题采用最新明确写入且最多召回八条。个人 PDF 也只有独立持久上传接口才入库，临时上传不会自动 promotion。HTTP 层仍是单部署 API-key 身份边界，多用户上线前必须增加可信 principal、导出、retention 和审计。完整设计见 [持久对话记忆](CONVERSATION_MEMORY.md) 与 [个人助手边界](PERSONAL_ASSISTANT_MEMORY_AND_CONTEXT.md)。

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
只有无模型基线也必须使用时，才补 AdaptivePlanner category routing；只注册一个函数而不补证据契约，不算完成接入。
