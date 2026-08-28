# 文档地图

文档按“用途”而不是按代码模块拆分。现行设计只维护一份完整文档，避免架构、记忆、工具、检索和运行时各自重复描述同一套系统。

## 推荐阅读顺序

| 顺序 | 文档 | 用途 |
|---|---|---|
| 1 | [完整系统设计](AGENT_DETAILED_GUIDE.md) | 唯一现行设计契约；包含总体架构、TaskFrame、LangGraph、工具、记忆、PDF/RAG、检索、可靠队列和运维基础设施 |
| 2 | [实施状态与验证记录](VALIDATION_AND_STATUS.md) | 当前已完成项、明确限制，以及按日期保留的企业故障注入、真实 LLM 和 Provider 实测 |
| 3 | [构建复盘](BUILD_RETROSPECTIVE.md) | 从旧实现到当前实现的演进、问题总账和取舍；不是现行契约 |
| 4 | [本地 BGE-M3 运维](LOCAL_BGE_M3.md) | 独立部署、启停和健康检查备忘 |
| 5 | [简历项目稿](RESUME_PROJECT_DRAFT.md) | 对外叙述与指标口径；不是运行时说明 |

## 为什么仍保留五份正文

- 完整系统设计与代码同步，回答“现在怎么工作”。
- 验证记录具有日期语义，不能混入现行契约后让历史结果看起来仍然有效。
- 构建复盘解释为何做出当前选择，属于历史材料。
- BGE-M3 是可以单独执行的运维手册。
- 简历稿面向招聘沟通，读者和准确性口径都不同。

除此之外，不再为 TaskFrame、LangGraph、工具、记忆、PDF 生命周期、双路检索或基础设施分别维护第二份现行说明。

## 系统全貌

```text
接口  CLI / FastAPI / Job / Conversation API
  └─ 服务装配  service.py（按本次请求注册工具、记忆、MCP）
       └─ LangGraph  intent → planning ↔ validation → final_generation
            intent            llm.task_frame → TaskFrame / ResearchScope
            planning          llm.plan → Harness / MCP 执行 1–4 个工具
            validation        CoverageAssessor + 报告硬校验
            final_generation  llm.synthesize → claims / 报告
       ├─ 证据平面  SourceRef / Evidence / Claim + provider 工具
       ├─ 记忆平面  对话事件 · 原子事实 · 个人记忆 · Skill · checkpoint
       └─ 数据平面  request/session/personal corpus · ACL · index manifest
```

LLM 在研究链路上有三个角色：`llm.task_frame` 理解任务，`llm.plan` 选工具，`llm.synthesize` 写 claims。权限、预算、算术、引用合法性、队列 lease 和数据保留由代码执行。

## 读代码时的对应关系

| 你想弄清 | 打开 |
|---|---|
| 图怎么走 | `graph.py` |
| 这一轮要什么证据 | `task_frame.py` → `ResearchScope` |
| 下一步调哪个工具 | `planning.py` |
| 引用是否合法 | `synthesis.py`、`validators.py` |
| 宏观、行情、SEC 从哪来 | `macro.py`、`market.py`、`sec.py` |
| 对话如何保留与压缩 | `memory_store.py`、`compression.py` |
| 指代所需的历史对象如何保存 | `atomic_facts.py` + `task_frame.py` |
| 长期个人偏好如何沉淀 | `memory_consolidation.py` |
| 成功路径如何复用 | `skill_learning.py` |
| 后台任务如何 lease、重试与取消 | `jobs.py`、`worker.py` |
| corpus ACL 与向量索引如何持久化 | `corpus.py`、`retrieval.py` |

