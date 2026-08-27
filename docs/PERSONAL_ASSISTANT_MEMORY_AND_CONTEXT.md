# 个人金融助手：记忆、上下文与扩展边界

本文描述 MAS Finance 2.2 的实际实现，不是未来路线图。重点回答六个问题：Agent 怎样决定下一步；网页与
RAG 结果怎样进入上下文；短期/长期记忆如何隔离；个人 PDF 何时持久化；MCP/企业工具怎样接入；模型能否
自己设计计算。

## 1. 成功标准与第一性约束

“人人可用”在这里不表示 Agent 可以替用户交易，也不表示 LLM 的自然语言等于事实。当前成功标准是：

1. 普通问题可以直接解释金融概念；需要数据时由模型从运行时目录自主选工具。
2. 数值计算交给确定性函数，模型只填写参数；自拟公式也不能执行任意代码。
3. 回答中的事实必须回到 `Evidence`；缺数据时显式降级或失败，不拿常识补洞。
4. 临时文档、会话文档、个人知识库、线程记忆、个人偏好是五种不同生命周期，不相互偷偷转化。
5. 个人记忆只由明确写入动作创建；一次失败回答不会被自动“学会”。
6. 外部工具只有在部署时注入且满足只读证据契约后，才进入模型可见目录。
7. 每次规划和生成的证据上下文有硬预算、来源平衡和可审计 manifest。

## 2. 运行架构

```text
用户请求
  │
  ▼
intent ── 解析实体、意图与证据需求
  │
  ▼
planning ── ModelPlanner 选择一个动作 ── Harness ── 只读 Tool
  ▲                 │                        │
  │                 └─ 非法/不可用 ─────────┘ 可见错误或规则规划降级
  │
validation ── 覆盖、冲突、引用、预算和停止条件
  │  不足且仍可解决
  └──────────────────────────────────────────────┐
  │ 足够/预算终止                                │
  ▼                                               │
final_generation ── evidence-bound LLM / 确定性复述│
  │                                               │
  └────────── validation ───────── END            │
```

LangGraph 是唯一状态机，只有 `intent / planning / validation / final_generation` 四个业务节点。
`ToolHarness` 是每次工具执行的配套 middleware，不是一个业务节点。这样既保留 checkpoint、状态历史和恢复，
又不把工具执行误建模为固定 workflow。

模型每轮最多提出四个 `call_tool`（或一次 `call_tools`）或 `finish`。配置了 LLM 时由模型自主选择工具；
没有密钥时才用 AdaptivePlanner 规则降级。MCP 工具走渐进发现，不把完整 schema 一次性塞进规划 prompt。
反过来也一样：coverage 达到最低要求后，只要仍有迭代预算，模型会再看到新 Evidence，并明确选择继续检索或
`finish`；Validation 不会因为“已经有一条文档证据”替模型抢先结束。无模型的规则规划器则在最低 requirement
满足后直接结束，避免为了模拟自主性增加空调用。

## 3. 上下文不是 3,000 token

系统区分输入证据预算与输出预算：

| 阶段 | 默认预算 | 可配置范围 | 作用 |
|---|---:|---:|---|
| 规划 evidence | 24,000 字符 | 4,000–200,000 | 让模型判断下一工具，不把整本 PDF 重复塞入每轮 |
| 最终生成 evidence | 48,000 字符 | 4,000–200,000 | 支持多来源综合回答 |
| 单条规划 evidence | 1,200 字符 | 内部固定 | 控制单一 chunk 垄断规划上下文 |
| 单条生成 evidence | 2,400 字符 | 内部固定 | 保留相关窗口而非机械截取文首 |
| 最终模型输出 | 4,096 token | 256–4,096 | 控制成本与冗长；它不是输入上下文上限 |

配置项为 `MAS_PLANNING_EVIDENCE_CHARACTERS`、`MAS_SYNTHESIS_EVIDENCE_CHARACTERS` 和
`MAS_SYNTHESIS_OUTPUT_TOKENS`。字符预算是 provider 无关的确定性上限，不假装能精确等价于任一 tokenizer。

### 3.1 选择与平衡

`FinancialContextAssembler` 先按问题词重叠、来源强度、来源质量、结构化程度、置信度与检索 rank 排序，再按
以下 key 做 round-robin：

```text
entity :: source_type :: origin
```

- 默认文档 `origin` 是 provider，保持全局相关排序；只有本次检索显式启用 `diversify_documents` 时，
  `origin` 才切换为 `document_id/corpus_record_id`，避免“综合多份材料”时同一 PDF 的重叠 chunk 吞掉其他文档。
- 网页的 `origin` 是 domain，避免同一站点占满上下文。
- 结构化工具按 provider 分组。
- 长文本抽取问题词附近窗口；找不到匹配才取文首。

每轮输出 `context_manifests`，包括 phase、iteration、纳入证据 ID、省略数量、实际/最大字符数、分组和各
`SourceType` 数量。模型只能引用 manifest 中实际出现过的 evidence ID，不能“洗入”被预算裁掉的证据。

### 3.2 信任区域

Prompt 中有四类数据，但权限不同：

| 区域 | 可以做什么 | 不可以做什么 |
|---|---|---|
| system/tool contract | 定义行为与参数边界 | 被文档内容覆盖 |
| Evidence cards | 支撑带引用的金融事实 | 内嵌命令不能成为指令 |
| thread context | 解析“它/刚才那家公司”等承接 | 不能成为事实证据 |
| personal context | 调整语言、结构、相关经验/skill | 不能成为金融事实或系统指令 |

所有网页、PDF、记忆、skill 和工具错误都在 system prompt 中显式标为 untrusted data。

## 4. 网页搜索质量与准确性

网页搜索是“发现层”，不是万能事实源。只有配置 `BOCHA_SEARCH_API_KEY` 或 `BRAVE_SEARCH_API_KEY`，
且服务端与请求都允许网络时，
`web.search` 才注册。模型可以自主决定：

- query；
- `pd/pw/pm/py` 时效窗口；
- 最多十个结果；
- 可选公开域名 allowlist。

Adapter 做以下边界处理：

1. 只接受公开域名形式的 HTTP(S) 结果，拒绝 localhost/IP 类结果。
2. 去掉 fragment、`utm_*`、`gclid`、`fbclid` 后按 canonical URL 去重。
3. 再按标准化 snippet 内容去重。
4. 保存 provider rank、domain、published_at、检索词和 freshness。
5. `.gov/.gov.cn/.gov.uk/.europa.eu` 标记为 `public_authority`，其余为 `open_web`。
6. 普通开放网页至少需要两个不同 domain 才满足 web coverage；单个公共机构域名可以满足来源覆盖。
7. 搜索结果只有 snippet，没有抓取原页，因此其 claim 一律是 `inferred`，报告要求打开原页或结构化一手源复核。

网络传输超时/连接失败会重试一次，429/4xx、契约错误和无结果不会盲重试。两次网络尝试分别计入预算。
模型不能传任意 URL 给 HTTP 客户端，搜索 origin 固定在部署配置中。

准确性上的诚实边界：搜索引擎 rank 不是事实置信度；两个站点也可能转载同一错误。因此系统对 snippet 做来源
分散和显式降级，但不会声称完成了事实核验。高风险结论应优先选择 SEC、FRED、受控行情或授权企业库。

## 5. RAG 与 PDF 上下文

### 5.1 三种上传生命周期

| 模式 | 创建方式 | 保存内容 | 生命周期 | 适合场景 |
|---|---|---|---|---|
| request | `analyze-upload` 默认 | 当前调用内解析页 | 请求结束 | 一次综合几份 PDF |
| session | `retain_for_session=true` | 进程内解析页文本 | 默认 1 小时 TTL | 同一 thread 连续追问 |
| personal KB | `POST /knowledge/documents` | SQLite 页文本与元数据 | 显式删除前 | 个人长期资料库 |

三种模式都在上传请求结束后删除服务端原 PDF。request/session 不会自动进入个人库；个人库也必须调用独立
持久上传接口。这个选择避免用户只是临时分析财报时发生隐式长期留存。

### 5.2 PDF 解析

- PDF 只通过 PaddleOCR-VL-1.6 或部署注入的成熟 PDF 解析 MCP 解析，不保留本地 PyMuPDF 分支。
- 保留 page、document ID、span、提取方式和页级引用。
- 未配置上述解析器时上传快速失败。
- 远程解析器只有在服务端和当前请求双重允许网络时调用。
- OCR 结果只消费页级 Markdown 文本，不下载其远程图片资源。
- 页数、文件字节数、总文本字符数均在边界限制。

### 5.3 检索契约

本地 personal/session corpus 默认严格按全局 BM25 相关性取 top-k，不为了“覆盖更多 PDF”牺牲当前问题的相关性。
只有模型明确传入 `diversify_documents=true`，或无模型降级规则识别到“综合/对比/分别分析多份材料”时，
才先给各相关文档一个结果再补同文档 chunk；此时规则基线把 top-k 提高到可用文档数（最高 20）。
`diversify_documents` 是研究意图，不是检索器的固定偏好：问单个 covenant 数值时，即使上传八份 PDF，也仍可只召回最相关的一份。
Validation 对明确的多文档意图只设置“至少两份不同文档”的硬下限，不粗暴要求遍历所有上传文件；超过这个
下限是否继续由模型在看到证据后决定。若用户明确要求“全部文档”，当前仍依赖模型执行，尚未把自然语言数量
解析成逐文档强制 coverage，这一点不会被文档宣称为已解决。
企业 RAG 通过统一 JSON contract 接入：

```json
{
  "query": "...",
  "top_k": 5,
  "filters": {},
  "search_mode": "rrf",
  "rerank": false,
  "diversify_documents": false
}
```

返回 chunk 必须带稳定 ID、字符串 content、可选 rank/score 和来源 metadata；Adapter 把它转换为
`SourceRef + Evidence`。固定部署 ACL filters 后合并，覆盖模型给出的同名 filter，模型不能放宽 ACL。
rank/score 被保留用于审计与排序，但不会被伪装成概率 confidence。远端 score 的量纲依 provider 而异，
因此当前 canonical gateway 必须自己执行阈值和 rerank；核心层不会套一个错误的通用阈值。

当前个人库默认使用 `personal.search` 词法检索；配置 `EmbeddingProvider` 后会额外注册
`personal.hybrid_search`，真实执行 BM25 + cosine + RRF。两个工具的网络属性分开声明，模型可自主选择，规则基线在
provider 可用且已获网络授权时优先 hybrid。SQLite 仍只持久化页文本，向量在每次分析的个人库快照中计算和缓存，
不会因为启用语义检索而改变用户的持久化同意。算法与部署限制见 [双路检索设计](HYBRID_RETRIEVAL.md)。

## 6. 记忆模型

进入模型的上下文按权限分层，而不是拼成一段可互相覆盖的“大 system prompt”：不可变中文系统规则位于最高层；
用户维护的 Markdown 和召回长期记忆作为带来源标记的低权限数据；随后是对话摘要、最近原始 run、实体回放；
当前请求、工具 schema、工具结果和 Evidence 最后按规划/生成阶段组装。摘要器只接收旧摘要与事件账本，永远不接收系统规则。

### 6.1 持久对话记忆

SQLite 按 tenant/user/thread 持久保存 user、Harness tool 和 assistant 事件，直到用户显式调用
`DELETE /api/v1/conversations/{thread_id}`。完整账本不直接进入模型：默认 300K token 投影预算达到 85% 后，专用 LLM
将 20K token 近期完整 run 之前的事件滚动总结为对话概要、用户目标/需求、已完成工作、工具成败状态、未完成工作和未决问题；压缩不会删除账本记录。

实体身份不由 LLM 摘要决定。系统从事件确定性构建 `entity_state`、有序 `focus_history`、当前焦点和
`entity_events` 原子事件回放。原子事件记录时间、sequence、实体、动作、状态及来源 event/run；账本最多保留最近 100 条，
每轮只把与当前 query/实体最相关或最近的 20 条作为 `entity_replay` 交给模型，不把完整历史实体表塞进 prompt。
“前者/后者/它们/刚刚那个公司”仍按确定性焦点历史解析；多个候选时的单数“它”不会猜。
历史实体仅在明确指代时继承。对话内容和工具历史仍是不可信上下文，绝不能替代 Evidence。完整数据模型和删除语义见
[持久对话记忆与动态上下文](CONVERSATION_MEMORY.md)。

### 6.2 个人长期记忆

个人记忆有四类：

- `profile`：稳定背景，例如常用市场或分析期限；
- `preference`：语言、格式、风险展示偏好；
- `experience`：用户显式要求保留的使用经验；
- `skill`：用户显式要求保留的分析步骤或方法。

长期信息有两个不同来源。`MAS_USER_PROFILE_PATH` 可指向用户主动维护的 UTF-8 Markdown；它在每轮作为独立的
`user_instructions` 数据层注入，低于系统规则、高于系统推断记忆，绝不拼接或改写不可变 system prompt。
`POST /api/v1/memories` 仍支持用户显式写入。

若启用 `MAS_AUTOMATIC_MEMORY_CONSOLIDATION_ENABLED`、配置 LLM 且本次请求获得网络授权，一个完成的 run 会作为静默窗口：
专用中文 LLM prompt 只读取该 run 的用户消息和已有记忆，最多生成 0～2 个候选。它看不到 system/developer prompt、工具结果和
助手回答，因此不能总结系统指令，也不能把助手建议或金融事实沉淀成用户偏好。自动候选只允许 profile/preference/experience，
禁止自动生成可执行 skill；置信度低于 0.75 的候选直接忽略。

LLM 必须声明 `add/reinforce/update/conflict/ignore`。显式且高置信的长期陈述可在一次 run 后晋升；仅推断出的倾向必须在两个
不同 run 中以同一记忆槽位重复出现。候选按 kind/title 和词项相似度与现有记忆去重。用户显式写入的记忆不会被自动内容覆盖；
冲突候选单独保存并带来源 run，不会静默替换。晋升记忆保存来源、scope、置信度和 evidence_run_ids，便于审计。

召回规则是确定性的：`profile/preference` 始终进入候选；`experience/skill` 必须与本轮 query 在 title/content/tags 上有词项重叠。英文按长度至少 2 的字母数字词项，中文按连续文本二元组匹配。排序使用“重叠数 + profile/preference 候选加分”，同分按更新时间倒序。最多八条、12,000 字符；单条注入内容最多 2,000 字符。召回结果同时进入规划和最终生成 prompt，但只是低权限偏好/背景数据，不能作为 Evidence。

同一 kind + 规范化 title 是同一槽位，后一次明确写入覆盖内容但保留创建时间。这是当前冲突策略：显式最新值
胜出，而不是把相反偏好同时交给 LLM 猜。不同标题的语义冲突不会被模型偷偷合并；用户可以列表查看并删除。

自动提取仍不是金融事实学习机制：当前关注股票、一次性格式要求、工具输出、供应商返回值和敏感信息不得进入个人长期记忆。
工具调用经验使用独立 `tool_usage_memory`，见下一节。

### 6.3 隔离边界

Service 层所有 thread/session/personal namespace 都由 tenant + user 构成并哈希后落库。个人知识库 SQL 查询也始终
带 tenant/user 条件。测试覆盖 Alice 上传后 Bob 不可检索、重启后仍可检索、删除后消失。

但当前 FastAPI 只有部署级 API key，没有 OIDC principal。它适合单用户/单部署；多用户 SaaS 不能让客户端
自报 `X-User-ID`，必须先由可信认证网关生成 principal，再把它传给 Service。代码已有 Service 参数边界，
HTTP 多租户认证仍是明确的上线阻塞项。

## 7. MCP、企业数据和 skill 注入

Agent 现在是 MCP Host：`MAS_MCP_SERVERS` 部署 allowlist 决定连接哪些本地 stdio 或固定 HTTPS server。每个 server 对应一个 MCP Client（JSON-RPC `initialize` / `tools/list` / `tools/call`）。Host 在进 Harness 前过滤：

- 必须显式只读（`annotations.readOnlyHint=true` 或 `_meta.mas_finance.side_effect=read_only`）；
- capability 必须属于现有只读证据能力，可由 server `default_capability` 或 `_meta.mas_finance.capability` 声明；
- 副作用、未知 capability、非法参数名会被记录为 rejection，不会进入模型目录；
- `tools/call` 结果必须是 canonical `EvidenceBundle`，原始 MCP JSON 不能当 Evidence。

模型仍然只在 user prompt 的 `available_tools` 里看到 builtins 和发现元工具，DeepSeek 请求不带 `tools` 字段。
已连接的 MCP 工具以 `mcp_tool_index` 短描述出现；完整 JSON Schema（包括服务端提供的字段描述、enum、default、examples）
通过 `mcp.describe_tool` 拉取，执行走 `mcp.call_tool`。`isError=true` 的结构化错误会保留 error_code、字段、候选值、
retryable 和 suggested_action 到 ToolResult；规划模型可在后续迭代中修改参数，完全相同的任务 ID 不会重复执行。
计算工具、内部研报 `RetrievalSource`、request/session/personal corpus 继续留在进程内，不经 MCP。

`FinanceAnalysisService(..., evidence_tools=(...))` 仍是手工注入入口，约束与 MCP 工具相同。原始 MCP annotation 不能提升权限。HTTP MCP 的 URL 必须是启动时固定的凭据无关 HTTPS；stdio command 由部署配置，不接受模型提供的路径。配置 AllTick 或必盈许可时会自动挂载本地 `extmarket` server。FRED/Bocha/行情/MCP call 有每分钟限流。

个人 skill 记忆也不是可执行插件。它只是低权限文本上下文。真正可执行 MCP 必须走 Host allowlist。

成功的 MCP 调用参数不会写入 personal memory，而是按 `server + tool + input-schema fingerprint + arguments` 写入当前用户隔离的
`tool_usage_memory`。记录只来自 Harness 验证成功的调用，包含成功次数和最后验证时间；召回时要求当前工具 schema 指纹一致，
schema 变化后旧经验自动停止注入。相关的最多五条成功示例以 `verified_tool_usage` 进入规划上下文，仍必须服从当前工具契约。

## 8. 模型自拟函数与计算 Harness

模型不能生成并执行 Python、SQL、Shell、`eval` 或动态 import。系统提供两层计算：

1. `finance.calculate`：经过人工定义和单测的金融公式，语义和数值都可验证，优先使用。
2. `finance.formula`：模型/用户提供声明式表达式和有限数值变量。

声明式表达式只允许 `+ - * / **`、括号、一元正负、`abs/sqrt/log/exp/min/max`。AST 节点数、深度、变量数量、
名称、有限数、指数和结果范围都有边界；属性访问、下标、列表推导、字符串、import 和未知变量全部失败关闭。
输出同时产生输入 Evidence 和 Calculation Evidence，并用 evidence ID 记录血缘。

Harness 能证明的是“没有执行任意代码、相同输入可复算、数值域合法”，不能证明模型选择的公式在金融语义上
正确。因此自拟公式的最终 claim 固定为 `inferred` 并带 caveat。只有内置公式能提供更强语义保证。

## 9. 回退、失败和恢复

| 故障 | 行为 | 是否隐藏 |
|---|---|---|
| 模型计划非 JSON、未知工具、非法参数 | 记录 `model_planner_fallback`，规则规划器接管 | 否 |
| 模型过早 `finish` | Validation 发现 coverage 不足后回 planning | 否 |
| 首选 RAG 无结果/失败 | gap 可解决时选择下一授权 provider | 否 |
| 网络未双重授权 | Harness `network_denied`，不发请求、不消耗网络尝试 | 否 |
| 网络 transport 暂时失败 | 只读 web/RAG 重试一次，按尝试计预算 | 否 |
| LLM 合成非 JSON、无逐字 quote、引用被裁证据 | `llm_synthesis_fallback`，确定性复述 Evidence | 否 |
| 仅 web snippet | 生成 `inferred` claim，整体至少 degraded | 否 |
| 无任何 Evidence | `failed / no_evidence` | 否 |
| 最终引用或结构硬校验失败 | `failed / validation_failed` | 否 |
| 进程/Agent 中断 | LangGraph SQLite checkpoint 按 run/thread 恢复 | 否 |

副作用工具不会自动重试。当前项目没有下单工具。checkpoint 保存完整 ResearchState 和 Harness audit usage；恢复时
会重新 prime 已消费工具、网络与模型预算，避免通过重启绕过额度。

## 10. 已验证场景

自动测试覆盖：

- 四节点图结构、模型自主选工具、非法计划降级、过早结束回环；
- checkpoint 跨实例恢复、请求不匹配拒绝、恢复预算连续；
- 网页 tracking URL/内容去重、来源多样性、私网 URL 拒绝、httpx transport 重试；
- web evidence 上下文组装、snippet claim 降级；
- 多实体和多文档上下文平衡、预算裁剪后禁止引用；
- request/session/personal PDF 生命周期、OCR 网络双授权、页级引用；
- 个人记忆显式写入、同槽覆盖、相关召回、用户隔离和删除；
- 个人知识库重启持久化、跨用户不可见和删除；
- MCP-shaped 只读工具注入、raw/副作用工具拒绝；
- MCP Host/Client allowlist、stdio/HTTP JSON-RPC、只读过滤与 EvidenceBundle 契约；
- 自拟公式选择、血缘、恶意 AST、未知变量、除零、NaN 与超大幂；
- 模型合成逐字 quote、citation laundering、坏 JSON 确定性回退；
- SEC/FRED/行情/RAG 契约、网络预算、审计脱敏和报告校验。

当前全量结果：161 tests passed，ruff 与 mypy 通过。

## 11. 明确限制与下一步条件

当前没有隐藏以下限制：

1. 开放网页只消费搜索摘要，不抓原文；所以只能作为推断。若要升级，需要受控 fetcher、DNS/redirect SSRF 防护、
   正文提取、许可与缓存策略，而不只是一个 `requests.get(url)`。
2. 个人知识库默认 SQLite + BM25，规模上限 100 份；配置后可做查询期 embedding/RRF，但向量未持久化、没有
   reranker，大规模生产应迁移到带 ACL、版本和删除传播的检索服务。
3. SQLite 个人文本未做静态加密；生产需磁盘/数据库加密、备份删除和 retention policy。
4. HTTP 层还没有多用户 OIDC；API key 只能代表一个部署。
5. 企业 MCP：Host/Client 已按 allowlist 接入 stdio/HTTPS JSON-RPC，并强制只读 Evidence 契约；规划侧已启用渐进发现与每轮最多四工具。尚未实现 SSE Streamable HTTP 与 OAuth；内置计算/研报 RAG 仍留在进程内。
6. LLM 的逐字引用验证保证 quote 存在，不能数学证明自然语言蕴含关系；高风险决策仍需人审。
7. 系统是研究助手，不提供个性化投资指令、交易或保证收益。

这些限制是边界，不是静默 fallback。只有在明确提供认证系统、企业数据 contract、检索质量指标或 MCP server 后，
才应加入对应 adapter，避免为了“看起来完整”堆积不会被调用的兼容层。
