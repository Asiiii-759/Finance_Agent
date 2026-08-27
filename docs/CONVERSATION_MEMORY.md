# 对话记忆、原子事实、长期记忆与运行日志

本文描述当前代码已经实现的四个独立数据平面。它们不能互相冒充：对话历史用于延续会话，原子事实用于精确回放，个人长期记忆用于跨会话偏好，Skill 用于复用成功工作路径；运行日志用于后端排障和审计。任何一层都不是金融 Evidence。

## 1. 总体数据流

```text
当前用户请求
  ├─ 完整 conversation_events ──┬─ LLM 滚动摘要 + 最近完整 run
  │                             └─ 全历史 atomic_fact（不参加摘要）
  ├─ personal_memory：profile / preference / experience
  ├─ learned_skills：短索引 ── TaskFrame 选中 ── 完整步骤交给 Planner
  └─ run_logs：启动、上下文、工具终态、运行终态/失败
```

所有 namespace 都包含 tenant/user；会话事件、摘要和日志额外包含 thread，日志再包含 run。当前 HTTP 层仍使用部署级 API key 和默认 principal，多用户生产部署必须由可信网关/OIDC 注入 tenant/user，不能相信请求体自报身份。

## 2. 持久对话事件与动态压缩

`conversation_events` 持久保存四种事件：

| 类型 | 保存内容 | 不保存 |
|---|---|---|
| `user_message` | 用户原文、时间、run、显式实体 | system prompt、密钥 |
| `tool_event` | 工具名、终态、尝试次数、错误码 | 原始 prompt、大段返回、凭据 |
| `assistant_message` | 用户可见报告、终态与数量统计 | 隐藏推理 |
| `atomic_fact` | 最小语义短句、时间、状态、来源事件 ID、实体 | 金融数据副本、推断结论 |

事件 sequence 在线程内单调递增；`event_id` 幂等，同 ID 不同内容立即报错。完整账本不会因上下文压缩被删除，默认保留到用户显式删除线程。

Prompt 投影默认最大 300,000 token。达到 85% 时，专用 LLM 把旧摘要与近期窗口之前的完整 run 合并成结构化摘要；近期窗口默认约 20,000 token，并以 run 为边界避免截断半次工具流程。摘要保留目标、限制、纠正、已完成事项、工具成败、未完成事项和开放问题。没有可用摘要器时快速失败，不做规则摘要或静默截断。

摘要模型是应用内部模型调用，不等于访问外部金融数据，因此不受请求的 `allow_network` 数据授权开关控制。`manifest` 公开摘要游标、估算 token、近期事件数及 `memory_is_evidence=false`。

## 3. 全历史原子事实

原子事实不是结构化知识图谱，也不是规则生成的实体标签。每个已结束 run 后，专用 LLM 从该 run 的用户、工具和助手终态事件中抽取最多 12 条可独立理解的最小中文短句，例如：

```text
用户要求比较苹果公司与微软公司的五年最大回撤。
market.history 对苹果公司的调用因 provider_timeout 失败。
本轮尚未完成两家公司回撤比较。
```

模型只能记录用户明确请求/纠正、系统确实完成的动作、工具明确成败和未完成事项；不能保存助手观点、金融结论、隐藏推理或推断意图。代码只在 LLM 边界校验 JSON 结构、长度、状态枚举和 `source_event_ids` 确实属于输入事件，不用关键词或相似度重新判断语义“是否相关”。

原子事实有三个关键不变量：

1. 摘要器的输入会排除 `atomic_fact`，所以事实不会被摘要改写或吞并。
2. 构造下一轮上下文时会从完整事件账本读取所有原子事实，不按最近 N 条或检索 top-k 裁剪。
3. TaskFrame prompt 首段是“该对话已经完成的最小事实经历”，每条包含 event ID、时间、状态、实体和短句；模型用它消解指代，并为来自历史的实体返回来源 event ID。

如果所有原子事实加其他必要上下文超过硬预算，系统显式失败，而不是悄悄丢弃早期事实。这个取舍保证“对话开头提到的实体”仍可回放；代价是极端超长线程最终需要用户新建线程或未来引入可审计的事实归档策略。

`entity_state` 和 `focus_history` 仍是从用户事件确定性投影出的辅助状态，用于 symbol 与焦点展示；它们不再生成规则式事实，也不决定 TaskFrame 意图。TaskFrame 模型无法可靠消解多个候选时必须请求澄清。

## 4. 个人长期记忆

长期记忆只允许三类：

- `profile`：稳定背景；
- `preference`：长期语言、格式或分析偏好；
- `experience`：跨会话仍有用的用户经验。

用户可通过 `POST/GET/DELETE /api/v1/memories` 显式管理；同 kind/title 的明确写入覆盖旧值。`MAS_USER_PROFILE_PATH` 指向用户自己维护的 Markdown，它作为独立低权限 `user_instructions` 注入，不拼进 system prompt。

启用自动沉淀时，专用 LLM 每个完成窗口只读取用户消息和现有记忆，最多产生两条候选：

- “这次、今天、本轮、当前报告”等临时要求必须 `ignore`，不会覆盖长期记忆；
- 行为推断需在两个不同成功 run 中重复后才晋升；
- 用户明确说“从今以后”并改变长期偏好时用 `update`，可覆盖同槽位旧记忆，包括先前由用户显式写入的值；
- `reinforce` 只追加来源 run，不改变原内容。

工具结果、当前关注股票、金融事实、敏感信息和 Skill 都禁止进入个人长期记忆。记忆召回最多八条，profile/preference 全局可见，experience 需与当前问题有词项重叠；它们始终是低权限上下文而非 Evidence。

## 5. Learned Skill 与渐进披露

Skill 是成功工作路径，不是用户偏好。只有 run 为 `succeeded` 且至少有两个成功工具 observation 时，Skill 提取器才会看到目标、成功标准、计划、工具类别和 gap；普通问答、单工具任务和失败 run 不触发学习。

Skill 只允许保存名称、用途、适用条件、2～12 个步骤和所需 capability。禁止公司名、symbol、日期、数值、URL、凭据、代码和金融结论。相同名称使用稳定 skill ID；再次成功会更新内容并累计来源 run。

渐进披露分两步：

1. TaskFrame 只看到最多 100 个 Skill 的 `id/name/description/applicability` 短索引，并最多选 3 个；不存在的 ID 会被拒绝。
2. Planner 只收到被选中 Skill 的完整 steps/capabilities，未选中的步骤不会进入规划上下文。

Skill 是不可信建议，不能越过当前工具 schema、权限、证据验收或用户请求。接口为 `GET /api/v1/skills` 与 `DELETE /api/v1/skills/{skill_id}`。

## 6. 持久运行日志

`run_logs` 是独立 SQLite 表，不依赖 API 响应是否成功。当前记录：

- `run.started`：run、恢复标志和请求是否授权网络；
- `context.loaded`：原子事实和近期事件数量；
- `tool.completed`：call ID、工具/capability、成功或失败、尝试/网络次数、耗时、错误码/脱敏错误消息、返回结构摘要；
- `run.completed`：Agent 终态、stop reason、claim/source 数、未解决 gap 和预算；
- `run.failed`：失败阶段和异常类型。

成功返回摘要只保存类型、顶层 keys 和证据/来源/条目数量，不复制网页、PDF、模型 prompt 或原始返回。Harness 在写审计前对参数和异常消息脱敏。日志通过 `GET /api/v1/conversations/{thread_id}/runs/{run_id}/logs` 查询；删除对话会同时删除该线程日志。

LangGraph checkpoint 与日志含义不同：checkpoint 用于状态恢复，日志用于人和运维系统理解发生了什么；日志不能恢复执行，checkpoint 也不应被当作长期记忆。

## 7. 删除、授权与当前边界

`DELETE /api/v1/conversations/{thread_id}` 删除当前 principal 下的完整事件、滚动摘要、运行日志和该线程各 run 的 LangGraph checkpoint。Session PDF、个人知识库、个人长期记忆和 Skill 各自有独立生命周期，不随线程删除。

当前工具全部是只读取证或纯计算，没有交易、转账、发送消息、删除外部数据等危险 side effect，因此运行时没有伪造一个无消费者的审批状态机。未来加入危险工具时，正确接入点是 Harness 在执行前根据 `side_effect` 产生 approval request，LangGraph checkpoint 保存中断状态，前端展示精确工具/参数/影响范围，用户批准后以同一 run 恢复；未批准不得调用 provider。API key 鉴权不能代替逐操作授权。

## 8. 已验证边界

测试覆盖 SQLite 重启持久化、事件幂等冲突、跨 tenant/user/thread 隔离、动态摘要不删除原文、早期原子事实跨压缩保留、来源 ID 校验、模型指代消解、长期偏好显式替换与临时 ignore、Skill 选中后才披露完整步骤、失败 run 日志持久化、工具审计脱敏，以及对话删除联动日志/checkpoint。生产仍需补可信多用户 principal、静态加密、合规 retention/export 和外部日志汇聚。
