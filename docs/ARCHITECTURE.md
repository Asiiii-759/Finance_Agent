# MAS Finance 总体架构

本文只描述系统全貌、稳定边界和模块关系。具体实现分别链接到对应模块文档，测试快照见
[验证与状态](VALIDATION_AND_STATUS.md)，历史问题与取舍见[构建复盘](BUILD_RETROSPECTIVE.md)。

## 1. 产品定义

MAS Finance 是面向个人金融研究的聊天式 Agent。用户提交自然语言问题和可选 PDF；模型理解任务、从当前授权工具目录
自主选择数据源和计算函数、核验证据是否足够，再生成带来源、缺口和运行状态的回答。

系统不执行交易、转账、发信或外部写入。当前所有研究工具都是只读工具。

## 2. 核心结构

```text
浏览器 / CLI / FastAPI
        │
        ▼
FinanceAnalysisService
  ├─ Principal、会话、附件与个人空间装配
  ├─ run-scoped Tool Harness / MCP Host
  └─ LangGraph
       intent
         │  llm.task_frame
         ▼
       planning ◄──────────────────┐
         │  llm.plan               │ evidence insufficient / tool error
         │  Harness 执行 1–4 工具   │
         ▼                         │
       validation ─────────────────┘
         │  deterministic coverage
         │  llm.validate_evidence（文档/网页）
         ▼
       final_generation
            llm.synthesize → Claim → 报告硬校验
```

图只有四个业务节点。Harness 是每次工具调用配套的执行边界，不是独立工作流节点。

## 3. 三个数据平面

### 3.1 控制平面

- 服务端可信 `Principal(tenant_id, user_id)`；
- `RuntimePolicy` 的工具、网络、模型和 token 上限；
- ToolSpec、JSON Schema、side effect、timeout、retry；
- LangGraph checkpoint 和可靠任务 lease。

用户消息、文档、网页、记忆、Skill 和工具结果都不能修改控制平面。

### 3.2 证据平面

- `SourceRef`：来源、provider、locator、时点和元数据；
- `Evidence`：原文或结构化数值、实体、字段、期间、单位；
- `Claim`：最终陈述、状态、引用 Evidence ID 和 caveat；
- `ResearchGap`：错误、缺失信息、冲突、是否可恢复。

金融数据只有转换成 canonical EvidenceBundle 后才能进入研究状态。

### 3.3 记忆与文档平面

- 对话事件和原子事实：完整持久化、按线程隔离；
- 对话摘要和近期窗口：Prompt 投影，不删除原始事件；
- 个人长期记忆：稳定偏好与背景；
- Learned Skill：成功工作路径，和个人偏好分开；
- request/session/personal 三类文档生命周期；
- 个人知识库持久 chunk、向量和索引 manifest。

## 4. LLM 与确定性代码分工

LLM 负责：

- 理解目标、指代和最低证据需求；
- 自主选择工具及参数；
- 判断非结构化证据能否回答问题；
- 在证据边界内组织中文 Claim；
- 压缩对话、提取最小事实、长期记忆候选和 Skill 候选。

代码负责：

- 身份、权限、网络授权、副作用和预算；
- JSON Schema、响应大小、数值域和 canonical contracts；
- 金融公式计算；
- 引用 ID 与逐字 quote 检查；
- Coverage 最低门、重复调用去重、停止条件；
- checkpoint、日志、队列、文档生命周期和删除传播。

系统不使用规则 Planner，也不在后台偷偷替模型改参数。

## 5. 关键不变量

1. 任务理解、规划和最终生成必须有 LLM；模型不可用或结构化输出持续非法时快速失败。
2. Planner 只能选择运行时已注册、当前 Principal 和 Policy 允许的工具。
3. 所有工具调用必须经过 Harness；Provider 原始对象不能直接进入 Agent state。
4. 算术由函数执行，LLM 只提供操作名和参数。
5. 文档/网页“检索到内容”不等于“足以回答”；Validation 必须检查语义充分性。
6. 需要证据但没有证据时不调用普通合成器，不生成无引用事实。
7. 完整历史保存在数据库；Prompt 只保存有界、可审计的投影。
8. 临时附件不会自动进入个人知识库。
9. 当前只读工具不需要审批；未来副作用工具必须 checkpoint 中断并由用户批准后恢复。

## 6. 当前模块导航

| 主题 | 权威文档 | 主要代码 |
|---|---|---|
| Agent 循环、纠错、充分性和拒答 | [Agent 循环](architecture/agent-loop.md) | `task_frame.py`、`planning.py`、`adequacy.py`、`graph.py`、`synthesis.py` |
| Tool、Harness、MCP 与错误 | [工具与 Harness](architecture/tools-and-harness.md) | `harness.py`、`mcp.py`、各 adapter |
| 上下文、会话和长期记忆 | [上下文与记忆](architecture/context-and-memory.md) | `context.py`、`memory_store.py`、`conversation.py`、`atomic_facts.py` |
| PDF、知识库和混合召回 | [RAG 与文档](architecture/rag-and-documents.md) | `ocr.py`、`documents.py`、`corpus.py`、`retrieval.py`、`personal_knowledge.py` |
| Principal、任务、日志和安全 | [运行时与安全](architecture/runtime-and-security.md) | `service.py`、`api/`、`queueing.py`、`worker.py`、`security.py` |
| 浏览器工作台 | [前端](architecture/frontend.md) | `web/index.html`、`web/app.js`、`web/app.css` |

## 7. 明确限制

- 本地前端目前是单用户体验，不是 OIDC 多用户产品；后端数据模型已按 tenant/user 隔离。
- 搜索摘要属于发现证据，重要结论仍应回到一手正文或结构化来源。
- Literal quote 防止伪造引用，但不是完整语义蕴含证明。
- 没有金融标注集上的 Recall@k、nDCG、citation precision 和长时间并发压测。
- Yahoo、AkShare、yfinance 等实验性来源不能替代有许可和 SLA 的生产行情源。
