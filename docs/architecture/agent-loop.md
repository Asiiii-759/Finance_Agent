# Agent 循环、纠错与拒答

## 1. 输入契约

一次运行接收三层输入：

- `ChatTurn`：本轮消息、Principal、thread/run ID、联网同意和当前附件；
- `RuntimePolicy`：服务端工具、网络、模型、迭代和 token 上限；
- `AgentContext`：线程 Prompt 投影、个人记忆和 Skill 短索引。

旧的 `ResearchRequest` 和 `document_index` 已删除。用户只需要像聊天一样提问。

## 2. intent：模型建立最低验收

`LLMTaskInterpreter` 将当前问题、原子事实、摘要、近期事件、个人上下文、Skill 索引和短工具目录交给
`llm.task_frame`，返回：

- 中文目标；
- 带来源的实体；
- intents；
- `ResearchRequirement[]`；
- 成功标准；
- 最多三个被选中的 Skill；
- 必要时的一条澄清问题。

Requirement 是“完成回答所必需的最低证据”，不是固定 workflow。概念解释可以没有 Requirement；实时行情、文档、
网页、监管、宏观和计算问题必须声明对应类别。历史实体只有引用有效原子事实 event ID 才能进入 TaskFrame。

## 3. planning：一次只决定下一步

每轮 Planner 看到：

- TaskFrame、Coverage 和未解决 Gap；
- 已有 Evidence；
- 过去工具名、参数、结果和结构化错误；
- 当前内置 ToolSpec + JSON Schema；
- MCP 短索引和渐进发现结果；
- 已验证的工具调用经验；
- TaskFrame 选中的完整 Skill。

模型只允许返回：单工具、最多四个并行工具，或 `finish`。它不能发明 URL、工具名或目录外 capability。

### 3.1 规划输出修复

供应商 429/5xx 和 transport error 由模型 ToolSpec 最多重试一次。模型请求成功但规划 JSON、工具名或顶层契约非法时，
Planner 将经过脱敏的校验错误和同一目录重新交给模型，最多再生成一次。第二次仍非法就快速失败。

这是格式/目录修复，不是规则 Planner，也不会替模型选工具。

### 3.2 工具参数自我纠正

嵌套参数由 Harness 的 Draft 2020-12 JSON Schema 校验。失败结果进入 `prior_actions`：

```json
{
  "error_code": "invalid_tool_arguments",
  "error_details": {
    "field": "requests.0.inputs",
    "missing_fields": ["years"],
    "model_action": "change_arguments"
  }
}
```

Validation 未满足时重新进入 Planner。模型可以修改参数再调用；完全相同的 tool+arguments 具有同一 task ID，不会浪费
Provider 和预算重复执行。

## 4. validation：最低 Coverage 与语义充分性

`CoverageAssessor` 先确定性检查类别、实体、字段、时点、来源多样性和模型声明的最低文档数。

当以下条件同时满足时，Validation 在同一节点内调用 `llm.validate_evidence`：

1. 确定性 Coverage 表面完成；
2. Planner 显式选择 `finish`；
3. Requirement 属于文档或网页等非结构化证据。

充分性检查逐 Requirement 返回：

- `sufficient`：现有文本直接包含回答所需信息；
- `insufficient`：相关但缺事实；
- `conflicting`：存在会改变答案的未解决冲突。

不足项转换成 `ResearchGap`，包含 `missing_information` 和一条可直接使用的 `retrieval_hint`。Graph 回到 Planner，
模型可以重写查询、切换工具或补充来源。新证据充分后，旧 Gap 标记为 resolved，而不是删除审计历史。

## 5. 停止与拒答

主要停止原因：

- `coverage_satisfied`；
- `clarification_required`；
- `insufficient_evidence`；
- `no_evidence`；
- `no_available_action`；
- `tool_budget_exhausted`；
- `max_iterations`；
- `validation_failed`。

如果问题需要证据但 EvidenceBundle 为空，Graph 在调用普通合成器前直接进入 `no_evidence`，不会花模型额度生成无引用事实。
部分证据仍不足时，可以生成仅受现有 quote 支撑的 Claim，但报告开头必须声明“当前证据不足以完整回答”，并列出未解决
Gap、尝试过的工具和需要补充的来源。结果状态为 degraded，而不是伪装 succeeded。

## 6. final_generation 与引用门

`llm.synthesize` 只返回 Claim JSON。引用 Claim 必须：

- 只使用本次 Context Manifest 中实际披露的 Evidence ID；
- 提供不少于八个字符的逐字 quote；
- quote 必须真实存在于至少一个引用 Evidence；
- 网页搜索 snippet 和用户声明式公式只能生成带 caveat 的 inferred Claim。

概念解释允许没有引用，但不能包含未提供的公司实时数据、数值或监管事实。

## 7. LangGraph 恢复语义

每个业务节点提交后由 LangGraph checkpointer 保存完整 `ResearchState`。恢复键绑定 tenant、user、thread 和 run；输入和
RuntimePolicy 必须与原运行完全一致。Harness 从持久 audit 恢复 tool/model/network/token 使用量和 call sequence，已成功
或已拒绝的工具不会因为进程恢复被重新计费。

## 8. 验收重点

- 未知工具 → 一次规划修复 → 合法工具；
- 缺少嵌套参数 → `change_arguments` → 修正调用成功；
- 模型过早 finish → Coverage 拒绝 → 继续规划；
- 文档/网页弱证据 → 语义 Gap → 改写检索 → Gap resolved；
- 一直不足 → `insufficient_evidence` + 明确拒答；
- 无证据 → 不调用 synthesis、不产生 Claim；
- checkpoint 恢复 → 不重复工具与预算。
