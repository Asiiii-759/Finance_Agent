# 实施状态

最后更新：2026-08-27
版本：2.2

## 已完成

- `Multi-Agent-project` 是唯一主项目；旧固定角色图、样例数据和兼容入口已删除。
- LangGraph 唯一业务图：intent → planning ↔ validation → final_generation → validation；无 Harness/act 业务节点。
- `ModelPlanner` 从动态工具目录自主选择单个下一动作或 finish；校验可拒绝过早 finish；规则 planner 仅作可见降级。
- 中英金融意图、字段级 requirement、显式 unsupported requirement 和稳定停止原因。
- `SourceRef / Evidence / Claim / EvidenceBundle`：content-addressed ID、引用完整性、冲突检测、数量/字符上限和严格 JSON。
- Tool Harness：run identity/预算上限绑定、capability、side effect、双重网络授权、严格输入/输出契约、只读 retry、观测 timeout、错误/参数脱敏和审计。
- 研究工具、数据 provider attempts、模型调用独立预算；恢复时 denied 调用不误计预算。
- 页级 request/session/personal BM25 corpus、可配置 embedding/cosine/RRF 双路召回、PDF 字节/页数/抽取文本上限、provider-neutral retrieval adapter。
- lexical 与 hybrid 拆为独立 ToolSpec：模型自主选择，网络属性在调用前可判定，未配置 embedding/reranker 时不伪装能力；提供受限 OpenAI-compatible HTTPS embedding client。
- PDF 解析收敛到 PaddleOCR-VL-1.6 或部署注入的成熟 PDF 解析 MCP；无本地 PyMuPDF fallback。PaddleOCR 整文档单次提交、有限轮询、结果字节上限且不下载图片。
- 部署期可注入多个内部/外部 `RetrievalSource`；固定 ACL filters、主备规划和受控 HTTPS JSON gateway。
- provider-neutral `web.search` 与 Bocha/Brave adapters：模型控制 query、freshness 和域名范围；域名
  allowlist 在响应边界强制执行；canonical URL/内容去重、公开域名校验、来源分散度和 snippet 推断降级。
- 行情快照：price、market cap、P/E、P/B、P/S、EV/EBITDA、52 周字段。
- 行情历史：原始标准化序列、adjusted/raw basis、收益、年化收益/波动、Sharpe、最大回撤及完整 calculation lineage。
- SEC Company Facts、recent filings、duration/instant 期间保留和同实体/单位/期间派生比率。
- ROA/ROE 缺平均余额、quick ratio 缺资产拆分、DCF 缺预测/折现/终值假设时明确拒绝。
- FRED metadata/observations、latest/previous evidence 和带输入 ID 的变化计算。
- `finance.calculate` 白名单：ratio、percentage change、CAGR、PV/FV、loan payment、annualized return/volatility、Sharpe、max drawdown。
- `finance.formula` 受限声明式 AST：模型可组织公式和参数但不能执行代码；数值保留输入血缘，公式语义固定标为待核验推断。
- `/api/v1/tools` 公开每个计算 operation 的 required/optional inputs、公式和默认单位；多余字段、类型强转和不兼容单位被拒绝。
- `finance.knowledge` 版本化金融定义和 caveat，不依赖模型常识。
- `finance-evidence-synthesis-v3` 上下文：trust zones、entity/source/domain 平衡、按意图可选 document 分散、规划 24K/生成 48K 可调字符预算、逐阶段 ContextManifest。
- LLM JSON/逐字 quote 校验；被裁掉证据不可引用；一个 quote 不能附带无关 citation；失败确定性降级。
- DeepSeek 真实验证覆盖十种批量计算、知识、主备 RAG、间接提示注入、多源上下文和 checkpoint 恢复；生成输出上限提高到 4096，输入 evidence 预算独立扩到 48K 字符。
- 报告 Markdown 注入转义、citation/footnote/gap/calculation lineage/risk notice 硬校验。
- 持久 conversation event ledger：user/tool/assistant 全历史、tenant/user/thread namespace、300K token 默认 prompt 预算、85% 阈值 LLM 滚动摘要、最近原始事件、确定性实体/焦点历史、歧义代词拒绝猜测和显式删除关联 checkpoints。
- 显式会话文档：原 PDF 请求后删除，仅在 opt-in 时将解析页文本按 tenant/user/thread 保留于进程内存；默认 1 小时 TTL，支持列举、召回和删除。
- 显式个人长期记忆：profile/preference/experience/skill 仅通过 CRUD 写入；同类同标题覆盖，相关性召回且绝不充当 Evidence。
- 持久个人 PDF 知识库：只保存解析页文本和元数据，tenant/user 精确隔离，支持上传、列表、检索和删除；临时文档不自动入库。
- 部署级 `evidence_tools` 注入边界：只接受 read-only canonical `EvidenceBundle` 工具，可承接 MCP gateway/企业数据源；副作用或 raw 工具启动即拒绝。
- 删除自建 checkpoint；Agent 只注入 LangGraph InMemorySaver/SqliteSaver。主服务和 job 使用 SQLite graph checkpoint，job_id 稳定恢复。
- `/api/v1/tools` 输出工具 input/result contract、availability、visibility 和行情 support tier。
- `/api/v1/config` 不再返回可能含密码的 DSN 或内部文件路径。
- 默认行情 provider 改为 offline；AlphaVantage 失败不再静默切换 Yahoo；Yahoo 标记实验性。
- Docker 使用 Python 3.11、非 root 用户、显式数据库密码和默认禁网/离线行情。
- 11 个可独立运行的企业黑盒评测场景，以及金融场景、白盒边界、安全和 API 测试。

## 当前验证

```text
11/11 enterprise black-box scenarios passed
145 tests passed under pytest; no skips
84.18% total source coverage; 80% gate passed
Ruff passed for src/tests
mypy passed for all 42 source files
compileall passed
Real DeepSeek planner selected finance.knowledge from the dynamic catalog in one model call; no gaps
Real DeepSeek selected corpus.hybrid_search first for a semantic-only PDF query; cross-checked lexical, deduplicated one Evidence, succeeded in 5 model calls
Bocha raw API and project EvidenceBundle path both returned two results; deepseek-v4-pro short live call succeeded
Compose YAML parsed successfully; Docker CLI was unavailable, so no image-build claim is made
PEP 517 sdist/wheel built with installed build requirements; 2.2.0 wheel imported successfully from a clean temporary target
```

完整评测设计、发现问题、外部数据源判断和上线门槛见 [企业级验证与故障注入报告](ENTERPRISE_EVALUATION.md)。

## 当前明确限制

- HTTP API key 是单部署身份边界；尚无 OIDC/JWT principal、RBAC 和完整多租户 ACL。
- 当前 API 内同步 provider 会阻塞单 event loop；需要多 worker 隔离，尚无可取消的原生 async I/O。
- Redis list worker 没有 lease/visibility timeout、幂等和死信，不能宣称可靠或 exactly-once。
- LangGraph SQLite checkpoint、memory、导出 artifact 尚无 KMS 加密、自动 retention 和 append-only 访问审计。
- 上传 corpus 默认 request-local；会话页文本只在单进程短 TTL 可见。内建 hybrid 已实现，但当前个人库只持久化
  SQLite 页文本、查询期重算向量；尚无大规模持久向量索引、principal-to-ACL 映射和删除传播。
- SEC recent filings 只返回元数据与 primary locator，没有自动 ingest filing HTML 全文。
- Yahoo endpoint 非契约化；AlphaVantage 历史使用 raw close；生产必须配置正式许可行情源。
- 字符预算不是 tokenizer；literal quote 不等于完整语义蕴含。
- 尚无真实 provider 录制响应、跨实例 SEC 全局限速、并发压测、长时间 soak 和灾难恢复演练。
- 尚无新闻、earnings call、商业行情、内部 SQL/warehouse adapter。
- `web.search` 的 Bocha API 与项目 EvidenceBundle 路径已完成小规模真实验证；Brave 仅完成 fixture 验证。
  尚无安全的通用 web.fetch。
- 当前只提供 MCP-shaped 工具注入边界，没有在缺少真实 server/授权模型时猜测实现通用 MCP transport。

## 生产优先级

1. 身份与数据治理：OIDC/JWT、tenant ACL、加密、TTL、删除/导出和 append-only audit。
2. 正式行情与可靠任务：licensed provider、共享 rate limit、lease queue、幂等和死信。
3. 运行隔离：async/进程边界、取消传播、OpenTelemetry、成本/token/provider 配额。
4. 数据质量：schema drift fixtures、重述/币种/时区/corporate-action 回归和来源质量评分。
5. 检索与语义验证：在金融标注集选定 embedding 模型、持久 hybrid corpus、ACL manifest、NLI/数值 claim parser 和人工抽检。
