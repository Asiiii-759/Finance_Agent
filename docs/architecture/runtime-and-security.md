# 运行时、Principal 与安全边界

## 1. Principal

当前部署由服务端配置提供可信 `tenant_id/user_id`，默认本地 `local/owner`。HTTP 请求不能自行提交 Principal。对话、Job、日志、
checkpoint、个人记忆、Skill、个人知识库、session 文档、预算和 artifact 查询都显式带同一 Principal filter。

这是完整多用户隔离基础，不等于已经提供生产登录。上线需要由可信网关验证 OIDC/JWT 后注入 Principal。

## 2. 网络授权

数据出网必须同时满足：

1. 服务端 `MAS_ALLOW_NETWORK=true`；
2. 用户在本轮请求明确 `allow_network=true`；
3. ToolSpec 声明 `network_access=true`；
4. network attempt 预算未耗尽。

模型调用属于必需模型边界，不被伪装成金融数据工具；外部搜索、OCR、行情、SEC、FRED、远程 RAG 和 HTTP MCP 均走数据网络授权。

## 3. 后台任务

`POST /api/v1/jobs` 和 upload 版本将任务写入数据库队列。主要语义：

- Principal-scoped idempotency key；
- pending/running/completed/failed/dead/cancel_requested/cancelled；
- lease + fencing token；
- heartbeat；
- 有限重试；
- 隔离子进程执行；
- worker 可接管租约过期任务；
- artifact 路径在下载前重新解析并限制在 run 目录。

这提供 at-least-once 工作领取，不宣称 exactly-once Provider 副作用。

## 4. Checkpoint

LangGraph SQLiteSaver 是唯一运行恢复机制。键由 tenant/user/thread/run 稳定哈希生成；恢复时 ChatTurn、RuntimePolicy 和
AgentContext 必须完全一致。线程删除会删除相关 checkpoint。

## 5. 日志与审计

两类日志分开：

- Harness audit：每个工具的 call ID、状态、attempts、错误、预算和脱敏参数/结果摘要；
- run log：run、context、tool、memory 和 terminal 生命周期事件。

密钥、Bearer、password、credential URL、Prompt、PDF、网页正文和任意巨大结果不会进入日志。失败 Job 只持久化错误类型和受控消息。

## 6. 副作用与审批

SideEffect 分为：`read_only`、`local_write`、`external_write`、`financial_transaction`。研究 Agent 当前 policy 只允许 read-only；
没有交易、转账、发信或外部删除工具，因此当前没有待审批动作。

未来接入副作用工具的正确链路是：

```text
Planner 选择工具
→ Harness 生成 approval request（工具、参数、影响范围）
→ LangGraph checkpoint interrupt
→ 前端显示并由用户批准/拒绝
→ 使用同一 run/checkpoint 恢复
→ 只执行被批准的精确调用
```

在后端 interrupt/resume 契约存在前，不增加会误导用户的“批准”按钮。

## 7. API 面

主要接口：

- `/api/v1/analyze`、`/analyze-upload`；
- `/api/v1/jobs`、upload、status、cancel、artifact；
- conversations/messages/runs/logs/delete；
- memories CRUD；
- skills list/delete；
- personal knowledge upload/list/reindex/delete；
- session documents list/delete；
- tools/config capability discovery。

`/api/v1/config` 不返回数据库 DSN、内部路径或密钥。

## 8. 上线前门槛

- OIDC/JWT 与网关 Principal；
- 数据库、checkpoint、upload、artifact 和备份加密；
- 用户 export、retention 和删除审计；
- 有许可和 SLA 的行情/新闻/研报来源；
- 共享 rate limit、circuit breaker、cache freshness 和 schema drift 监控；
- append-only 审计出口、费用账本、SLO、并发/灾备演练；
- OCR/LLM/搜索供应商的数据保留、区域和敏感文档政策。
