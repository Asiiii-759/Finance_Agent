# 工具、Harness 与 MCP

## 1. ToolSpec

每个工具必须声明：名称、描述、capability、是否联网、side effect、timeout、retry、结果类型、浅层参数键契约和完整
Draft 2020-12 JSON Schema。Schema 同时用于模型选择说明和执行前校验，不是仅供展示的文档。

核心结果类型包括 EvidenceBundle、model response 和 catalog。金融研究工具不得返回任意 Provider JSON。

## 2. Harness 固定顺序

```text
解析 registry
→ 绑定 run / Principal / policy ceiling
→ capability
→ side effect
→ deployment + request 双重网络授权
→ JSON 与 JSON Schema
→ tool/model/network/token 预算
→ provider timeout / 明确 retry
→ result contract
→ 脱敏 audit
```

被拒绝或参数非法的调用 attempts=0，不消耗 Provider 和研究工具预算。

## 3. 稳定错误契约

`ToolResult` 对模型暴露稳定的 `error_code`、脱敏 `error_message` 和 `error_details`。主要动作提示：

- `change_arguments`：字段、类型、缺失值、allowed values 或候选 symbol 已给出；
- `choose_alternative_tool`：供应商或工具内部失败，换来源；
- `report_unavailable`：凭据、供应商访问策略或部署环境拒绝，停止重复调用并向用户说明；
- `request_authorization`：缺少联网或未来副作用审批；
- `stop_and_report`：部署凭据缺失等不可由本轮参数解决的问题。

异常类名只作为诊断字段，不作为 Planner 的公共协议。

## 4. 重试边界

- 自动重试只允许 read-only 工具；
- web/RAG/FRED/SEC 的 429、5xx、transport timeout/connection 通常最多两次；
- FRED/SEC/Bocha 的 401/403 返回 `provider_access_denied + report_unavailable`，不重试；
- PaddleOCR 状态轮询与结果 GET 可有限重试；创建 Job 的 POST 没有幂等键，因此不自动重试；
- DeepSeek 429/5xx/transport 最多重试一次；
- 普通 HTTP 4xx、Schema 错误、空结果和 `ToolExecutionError` 不盲重试；
- MCP 绑定工具默认一次，结构化错误交给下一轮 Planner 改参；
- 每个数据网络 attempt 独立计入 network budget；模型内部瞬时重试仍算一个逻辑 model call。

同步 Harness timeout 是观测边界；HTTP/数据库 adapter 仍必须配置底层 I/O timeout。

## 5. 当前工具类别

| 类别 | 典型工具 | 说明 |
|---|---|---|
| 计算 | `finance.calculate`、`finance.formula` | 白名单公式或受限 AST；LLM 只传参数 |
| 请求/会话文档 | `corpus.search`、`corpus.hybrid_search` | 本次附件或显式保留到线程的文档 |
| 个人知识库 | `personal.search`、`personal.hybrid_search` | 当前 Principal 的持久文档索引 |
| 行情 | `market.snapshot`、`market.history` | 要求显式 symbol，不猜公司映射 |
| 监管 | `sec.company_facts`、`sec.recent_filings` | SEC 一手结构化数据和披露元数据 |
| 宏观 | `macro.fred_series` | FRED 序列元数据与 observations |
| 网页 | `web.search` | Bocha/Brave provider-neutral 搜索；snippet 降级 |
| MCP 发现 | `mcp.search_tools`、`mcp.describe_tool` | 渐进披露远程契约 |
| MCP 执行 | `mcp.call_tool` | 只允许 allowlist 中的只读证据工具 |

工具是否真正出现取决于部署配置、用户授权和本次 run 的数据范围。

## 6. MCP 渐进披露

启动时 Host 连接部署 allowlist，过滤：

- 非只读或未知 side effect；
- 未授权 capability；
- 无法映射为 EvidenceBundle 的结果；
- 非法 Schema、工具名和不安全 transport。

Planner 默认只看短 `mcp_tool_index`。需要时先 search/describe，再用本地限定名调用。JSON-RPC `-32602`、候选字段、
retryable 和 suggested action 会保留到错误详情。MCP 原始正文不直接成为金融证据。

## 7. 工具经验与 Skill

成功 MCP 调用的非敏感参数按“工具名 + Schema fingerprint”保存在独立 tool usage memory。Schema 改变后旧参数不会注入。
它只作为 verified example，当前 Schema 和用户问题优先。

成功多步骤研究路径可沉淀成 Learned Skill；Skill 不保存用户偏好、公司专属值、URL、年份或未观察 capability。

## 8. 新增工具的最小门槛

1. 定义明确用户 Requirement；
2. 写完整输入 Schema 和 Provider 边界校验；
3. 输出 SourceRef/Evidence/Gap；
4. 声明网络、side effect、timeout、retry 和成本；
5. 增加成功、空响应、Schema drift、错误参数、网络拒绝、重试预算和多租户测试；
6. 若是 MCP，只通过渐进发现暴露；
7. 若有副作用，先实现真正的 interrupt/approval/resume，不得仅增加按钮。
