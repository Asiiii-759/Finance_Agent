# LLM TaskFrame：正常研究路径的任务理解

配置 LLM 后，系统不再用关键词或“它 / 前者 / 后者”规则生成本轮研究需求。`llm.task_frame` 先读取全历史原子事实，再读取当前请求、对话摘要、最近事件和 Learned Skill 短索引，输出中文 JSON：目标、实体及来源、最低 evidence requirements、完成标准、选中的 Skill ID，或一条澄清问题。

TaskFrame 的 requirements 会写入 `ResearchScope`，因此 coverage 仍然是可审计、可复现的确定性验收：它只判断模型已经提出的证据类别是否真正落入 `EvidenceBundle`，不再替模型猜用户意图。`ModelPlanner` 继续自行决定调用什么工具、调用次数和错误后如何改参；它可以调用任何当前授权的工具，而不限于 TaskFrame 的 requirements。

原子事实是带 `event_id`、时间、实体、状态和最小语义短句的历史账本，不是实体关系推理器。模型在 TaskFrame 中声明从 `current_request` 或 `conversation_memory` 得到的实体，历史实体必须引用真实事实 event ID。若多个历史对象都合理，模型必须输出 `clarification_question`，图在不调工具的情况下以 `needs_clarification` 结束。

LLM 是研究链路的必需依赖；未配置时服务会快速失败，不存在规则式指代或规划降级。工具 allowlist、参数 schema、预算、只读权限、证据契约和最终校验始终由代码执行，不交给模型。
