# 持久对话记忆与动态上下文

本文描述当前已实现的对话记忆，不把 roadmap 写成现状。其目标是同时满足：线程对话在重启后仍存在；长对话不会撑爆模型窗口；工具执行可被后续轮次理解；指代解析不靠猜；用户删除线程时相关状态一并删除。

## 1. 两层模型

```text
SQLite 完整事件账本（事实记录，不自动过期）
  user_message / tool_event / assistant_message
                    │
                    ▼ 预算达到 85% 时滚动压缩
Prompt 投影（最多 MAS_CONVERSATION_CONTEXT_CHARACTERS）
  结构化旧摘要 + 最近原始事件 + 实体状态/关系 + manifest
```

完整账本与 prompt 投影必须分开：压缩只改变模型下一轮看到的表示，不删除原事件。默认保留到用户调用 `DELETE /api/v1/conversations/{thread_id}`。这与 request/session PDF 生命周期和个人长期记忆是不同的数据平面。

## 2. 持久事件账本

`conversation_events` 以 `tenant_id + user_id + conversation_history + thread_id` 精确隔离，线程内 sequence 单调递增。事件包括：

| 类型 | 内容 | 附加状态 |
|---|---|---|
| `user_message` | 用户原始问题 | 时间、run、当前解析实体、实体关系 |
| `tool_event` | 工具名与结果状态 | capability、尝试次数、网络次数、错误码；不复制 prompt/密钥 |
| `assistant_message` | 最终报告 | 状态、claim/source 数量、未解决 gap code |

`event_id` 支持恢复重放幂等；同 ID 内容不同会快速失败。SQLite 用立即写事务分配 sequence，避免并发写入得到相同序号。数据库损坏或未知事件类型不会被静默清空，而是显式报错。

工具审计来自 Harness 已脱敏字段。对话账本不保存隐藏推理，也不把记忆提升为 `Evidence`；后续回答中的金融事实仍必须由本轮可引用数据源支撑。

## 3. 动态压缩

默认预算为 16,000 字符，保留最近 12 个原始事件。可通过以下变量调整：

```dotenv
MAS_CONVERSATION_CONTEXT_CHARACTERS=16000
MAS_CONVERSATION_RECENT_EVENTS=12
```

每次分析前从最近摘要游标继续读取事件。投影达到预算的 85% 且事件多于近期窗口时，把更老事件合并成确定性的结构化摘要：旧用户请求、旧助手结果、工具活动、实体时间状态和关系。摘要记录 `covered_through_sequence`，最近事件继续保持原始顺序和时间。若仍超预算，先移除最老的摘要条目，再减少最老的 prompt 事件；SQLite 原始账本不受影响。

本实现刻意不额外调用 LLM 做语义摘要：结构化压缩可复现、不会虚构事实、没有额外费用。代价是很老对话的细节可能不再进入 prompt，但仍可从完整账本导出；未来接入语义摘要器时可以替换投影生成，不需要迁移事件表。

`manifest` 明确给出摘要覆盖序号、近期事件数量、最新序号、`full_history_persisted=true` 和 `memory_is_evidence=false`。因此测试和审计可以区分“数据库里存在”与“模型本轮实际看见”。

## 4. 实体与指代

每次实体提及保存时间、sequence 和 mention count；关系目前包括：

- `has_symbol`：实体到行情代码；
- `co_mentioned`：同一轮相邻出现的实体，保留用户列举顺序。

解析优先级是：API 显式实体 → 当前问题检测实体 → 明确的历史指代。最近一个含实体的用户事件形成有序 `focus_entities`：

- “前者/第一个/former”选择首个；
- “后者/最后一个/latter”选择末个；
- “它们/两者/both”选择整组；
- 单数“它/该公司/it”只有在候选唯一时继承；多候选时返回空实体并保留上下文，让规划/回答暴露歧义，而不是猜一个；
- “继续/呢/what about”可承接当前焦点组。

只要线程已有历史，有界上下文就会进入规划与生成，即使新问题没有代词；但历史实体不会因此覆盖当前问题中的显式实体。

## 5. 删除与 LangGraph checkpoint

`DELETE /api/v1/conversations/{thread_id}` 删除当前 principal 下：

1. 全部对话事件；
2. 滚动摘要；
3. 账本中每个 run 对应的 LangGraph SQLite checkpoint thread。

返回删除的 event、summary 和 checkpoint-thread 数量。Session documents 与 personal memory/knowledge 有独立接口，不会因删除对话被隐式联动；这是为了避免一个按钮越权删除不同同意范围的数据。

## 6. 已验证边界

自动测试覆盖：SQLite 重启持久化、sequence 与幂等冲突、动态压缩后原始事件不丢失、实体时间/关系投影、Harness 工具事件入账、tenant/user/thread 隔离、前者/后者/复数/歧义单数指代、服务重启继续对话，以及显式删除事件/摘要/checkpoint。测试必须使用 `PYTHONPATH=src`，避免误导入虚拟环境中的旧安装包。

当前生产缺口仍包括数据库静态加密、可信 OIDC principal、对话导出、可配置合规保留策略与 append-only 访问审计。它们不应由代码中的静默 TTL 或默认用户假装解决。
