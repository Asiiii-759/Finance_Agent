# 文档地图

现行设计按稳定职责拆分，避免一个 Guide 同时承担架构、接口、运维和历史记录。

## 推荐阅读顺序

| 顺序 | 文档 | 回答的问题 |
|---|---|---|
| 1 | [总体架构](ARCHITECTURE.md) | 系统是什么、边界和模块关系是什么 |
| 2 | [Agent 循环](architecture/agent-loop.md) | 模型怎样理解任务、选工具、纠错、重检索和拒答 |
| 3 | [工具与 Harness](architecture/tools-and-harness.md) | 工具契约、稳定错误、预算、MCP 和渐进披露怎样工作 |
| 4 | [上下文与记忆](architecture/context-and-memory.md) | 长对话、原子事实、摘要、个人记忆和 Skill 怎样伸缩 |
| 5 | [RAG 与文档](architecture/rag-and-documents.md) | PDF 生命周期、解析、分块、混合检索和证据充分性 |
| 6 | [运行时与安全](architecture/runtime-and-security.md) | Principal、Job、checkpoint、日志和未来审批边界 |
| 7 | [浏览器工作台](architecture/frontend.md) | 当前前端能力、数据流和明确限制 |

旧的 [详细 Guide](AGENT_DETAILED_GUIDE.md) 仅为旧链接提供迁移入口，不再维护重复正文。

## 状态、历史和运维

- [实施状态与验证记录](VALIDATION_AND_STATUS.md)：当前能力、限制和带日期的验证快照。
- [构建复盘](BUILD_RETROSPECTIVE.md)：发现过的问题、解决方式与设计取舍；不是现行契约。
- [本地 BGE-M3 运维](LOCAL_BGE_M3.md)：部署、启停和健康检查。
- [简历项目稿](RESUME_PROJECT_DRAFT.md)：对外叙述口径；不是运行时说明。

## 从问题定位代码

| 问题 | 主要实现 |
|---|---|
| 图怎样运行 | `graph.py`、`agent.py` |
| 本轮需要什么证据 | `task_frame.py` |
| 下一步选择哪个工具 | `planning.py` |
| 工具权限、参数与错误 | `harness.py`、`tools.py` |
| 证据是否充分 | `adequacy.py`、`coverage.py` |
| 引用是否合法 | `synthesis.py`、`validators.py` |
| 对话怎样持久化、摘要和投影 | `memory_store.py`、`conversation.py` |
| 原子事实怎样提取 | `atomic_facts.py` |
| 个人长期记忆怎样沉淀 | `memory_consolidation.py` |
| 成功路径怎样复用 | `skill_learning.py` |
| PDF 与持久索引 | `documents.py`、`corpus.py`、`retrieval.py` |
| 后台任务怎样 lease、重试与取消 | `queueing.py`、`worker.py`、`service.py` |
| API、Principal 与前端 | `api/app.py`、`web/` |
