# MAS Finance

证据优先、可恢复、受预算约束的金融研究 Agent。系统从上传文档和受控行情源获取数据，形成带 provenance 的证据账本，再生成带引用、缺口说明和风险提示的报告。

当前由 LangGraph 提供唯一编排和运行恢复，顶层只保留四个业务节点：

```text
intent → planning ↔ validation → final_generation → validation → END
```

`planning` 由模型优先选择一个下一步工具动作并在同一节点内经 Harness 执行；Harness
是执行 middleware，不是图节点。校验节点可以拒绝模型过早结束并送回规划。未配置模型或模型计划违反契约时，
规则规划器作为可见降级基线接管。

没有证据时系统会失败关闭，不使用演示数据或模型常识填空。项目不包含交易/下单能力。

## 能力

- PDF 上传、视觉顺序文本提取、扫描页诊断、可选 PaddleOCR-VL-1.6、页级引用，以及 request/session/personal BM25 + embedding/RRF 双路检索
- 可注入内部/外部 RAG 源，以及固定 HTTPS canonical JSON 搜索网关
- 中英金融场景识别、结构化 ResearchScope、模型自主逐步规划与确定性规划降级
- 受控开放网页搜索：模型生成检索式、时效窗口和域名范围；URL/内容去重、来源分散度校验，snippet 只能形成带复核提示的推断
- 显式 provider 行情快照、adjusted/raw 历史收益、波动率和最大回撤（默认 offline）
- SEC EDGAR Company Facts、最近 filing 元数据与扩展财务比率
- FRED 宏观时间序列
- CAGR、现值/终值、贷款支付、年化收益/波动、Sharpe 等白名单计算
- 受限声明式公式：允许模型组织公式与参数，但只执行 AST 白名单，不执行模型生成的 Python/SQL/Shell；结果保留输入血缘并标注语义待核验
- 版本化金融概念、公式和解释知识工具
- 统一 `SourceRef / Evidence / Claim` 契约和 citation ledger
- 工具输入/输出契约、run identity、capability、网络、副作用、分账预算、重试、超时和审计控制
- 主/备数据源重规划、模型 `finish` 证据校验与明确停止原因
- LangGraph SQLite checkpointer、状态历史和跨 Agent 实例恢复，以及 tenant/user/thread 隔离记忆
- 最小跨轮线程上下文、显式写入的个人 profile/preference/experience/skill、用户隔离的持久个人 PDF 知识库及逐项删除
- 按 entity/source/domain 平衡、按研究意图可选 document 分散、保留 provenance 的 Prompt ContextAssembler；规划 24K、生成 48K evidence 字符预算可调，并输出逐阶段 manifest
- 只读 canonical-evidence 工具注入边界，可接企业 RAG、MCP gateway 或 licensed feed；未注入时不启用
- 带逐字 evidence quote 验证的 LLM 合成与确定性降级
- FastAPI、后台作业、CLI、报告与审计产物
- 11 个可独立运行的黑盒验收场景和 140 项自动化测试

完整的运行机制、状态契约、数据源、记忆、安全边界和扩展方法见
[Agent 详细说明](docs/AGENT_DETAILED_GUIDE.md)；架构决策摘要见
[架构文档](docs/ARCHITECTURE.md)，LangGraph、自主规划、Harness 配套执行和恢复细节见
[LangGraph 编排、自主规划与恢复设计](docs/LANGGRAPH_RUNTIME.md)，工具与调用详见
[工具及调用逻辑](docs/TOOLS_AND_REASONING.md)，当前进度和限制见
[实施状态](docs/IMPLEMENTATION_STATUS.md)，发现问题与生产上线门槛见
[企业级验证与故障注入报告](docs/ENTERPRISE_EVALUATION.md)。从最初实现到当前版本的完整问题、根因、方案和验证记录见
[全过程构建复盘与问题总账](docs/BUILD_RETROSPECTIVE.md)；真实模型、回退和恢复结果见
[真实 LLM、Harness 回退与 Checkpoint 恢复验证](docs/LIVE_LLM_EVALUATION.md)。
[Bocha 搜索与当前模型配置实测](docs/LIVE_PROVIDER_EVALUATION.md)记录了线上契约、调用结果和保留边界。
个人记忆、个人知识库、上下文预算、联网来源质量、MCP/企业扩展与模型自拟公式的当前边界见
[个人金融助手：记忆、上下文与扩展边界](docs/PERSONAL_ASSISTANT_MEMORY_AND_CONTEXT.md)。
双路召回的算法、真实工具边界、部署接口、生命周期、测试与未完成项见
[BM25 + Embedding 双路检索设计](docs/HYBRID_RETRIEVAL.md)。

## 环境

- Python 3.11
- 核心安装：`pip install -e .`
- 完整安装：`pip install -e '.[all,dev]'`

```bash
python -m mas_finance.evaluation
python -m unittest discover -s tests -v
pytest --cov=mas_finance --cov-report=term
ruff check src tests run_demo.py start_api.py start_worker.py
mypy src
```

## CLI

分析本地 PDF：

```bash
mas-finance \
  --query "分析 ACME 的需求、现金流和估值" \
  --entity ACME \
  --symbol ACME=ACME \
  --pdf ./acme-report.pdf
```

或在源码目录运行：

```bash
python run_demo.py --query "分析这份财报" --pdf ./report.pdf
```

扫描版 PDF 可选启用 PaddleOCR。密钥只配置在环境中，并继续要求双重网络授权：

```bash
set -a; source .env; set +a
MAS_ALLOW_NETWORK=true mas-finance \
  --query "分析扫描财报中的流动性风险" \
  --entity ACME \
  --pdf ./scanned-report.pdf \
  --allow-network
```

启用行情需要双重授权：服务端设置 `MAS_ALLOW_NETWORK=true`，调用方再传 `--allow-network`。如果只配置一侧，Harness 会返回可见的 `network_denied` gap。

确定性计算无需联网：

```bash
mas-finance \
  --query "计算三年 CAGR" \
  --calculate '{"operation":"cagr","inputs":{"beginning_value":100,"ending_value":150,"years":3}}'
```

## API

```bash
python start_api.py
```

JSON 请求：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "分析 Apple 当前估值",
    "entities": ["Apple"],
    "symbols": {"Apple": "AAPL"},
    "allow_network": true,
    "export_artifacts": false
  }'
```

上传 PDF：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze-upload \
  -F 'query=分析这份 Apple 财报的风险' \
  -F 'entities=Apple' \
  -F 'files=@./apple-report.pdf'
```

临时上传默认只用于当前请求；需要同一线程连续追问时，首次上传增加
`retain_for_session=true`，后续 JSON 请求增加 `use_session_documents=true`。
原 PDF 仍会在上传请求结束后删除，会话中只保留解析页文本并受短 TTL 控制。
完整边界见 [PDF、RAG 与记忆生命周期设计](docs/DOCUMENT_LIFECYCLE.md)。

明确保存个人偏好（普通问答不会自动写入）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"kind":"preference","title":"回答风格","content":"使用中文，先给结论再解释风险。","tags":["中文"]}'
```

明确持久上传个人知识文档：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge/documents \
  -F 'allow_network=false' \
  -F 'files=@./my-finance-notes.pdf'
```

分析请求默认召回个人偏好和个人知识库，可分别设置 `use_personal_memory=false`、
`use_personal_knowledge=false` 关闭。当前 HTTP API 适合单用户部署；不要在未接 OIDC 的共享实例中混用个人数据。

主要接口：

- `GET /health`
- `GET /api/v1/config`
- `GET /api/v1/tools`
- `DELETE /api/v1/memory/threads/{thread_id}`
- `POST /api/v1/memories`
- `GET /api/v1/memories`
- `DELETE /api/v1/memories/{memory_id}`
- `POST /api/v1/knowledge/documents`
- `GET /api/v1/knowledge/documents`
- `DELETE /api/v1/knowledge/documents/{document_id}`
- `GET /api/v1/session-documents/{thread_id}`
- `DELETE /api/v1/session-documents/{thread_id}`
- `POST /api/v1/analyze`
- `POST /api/v1/analyze-upload`
- `POST /api/v1/jobs`
- `POST /api/v1/jobs/upload`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`

配置 `MAS_API_KEY` 后，`/api/v1/*` 需要 `X-API-Key`；健康检查保持匿名。

## 关键配置

```dotenv
MAS_OUTPUT_DIR=outputs
MAS_UPLOAD_DIR=uploads
MAS_DB_PATH=data/mas_finance.db
MAS_DATABASE_URL=sqlite:///data/mas_finance.db
MAS_ALLOW_NETWORK=false
MAS_MARKET_DATA_PROVIDER=offline
MAS_SEC_USER_AGENT=YourCompany ops@example.com
FRED_API_KEY=
FRED_BASE_URL=https://api.stlouisfed.org
BOCHA_SEARCH_API_KEY=
BRAVE_SEARCH_API_KEY=
MAS_EMBEDDING_ENDPOINT=
MAS_EMBEDDING_MODEL=
MAS_EMBEDDING_API_KEY=
MAS_EMBEDDING_TIMEOUT_SECONDS=30
MAS_THREAD_MEMORY_ENABLED=true
MAS_THREAD_MEMORY_TTL_SECONDS=604800
MAS_PERSONAL_MEMORY_ENABLED=true
MAS_PERSONAL_KNOWLEDGE_ENABLED=true
MAS_MAX_PERSONAL_KNOWLEDGE_DOCUMENTS=100
MAS_SESSION_DOCUMENT_TTL_SECONDS=3600
MAS_MAX_SESSION_DOCUMENT_SESSIONS=100
MAS_MAX_PDF_TEXT_CHARACTERS=5000000
MAS_PLANNING_EVIDENCE_CHARACTERS=24000
MAS_SYNTHESIS_EVIDENCE_CHARACTERS=48000
MAS_SYNTHESIS_OUTPUT_TOKENS=4096
MAS_API_KEY=

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-pro

PADDLEOCR_ACCESS_TOKEN=
PADDLEOCR_JOB_URL=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
PADDLEOCR_MODEL=PaddleOCR-VL-1.6
```

本机 BGE-M3/vLLM 的部署参数、启停和健康检查见 [docs/LOCAL_BGE_M3.md](docs/LOCAL_BGE_M3.md)。

`MAS_ALLOW_NETWORK` 和行情 provider 默认关闭/离线。远程 OCR、embedding、行情、开放网页搜索和外部 RAG
都需要服务端允许且本次请求显式 `allow_network=true`。`BOCHA_SEARCH_API_KEY` 或
`BRAVE_SEARCH_API_KEY` 只启用 `web.search`；两者同时存在时显式优先 Bocha，
模型仍不能构造任意 HTTP 请求。网页搜索结果只是发现层，不会冒充 SEC/FRED/行情等结构化一手数据。
PaddleOCR 未配置时，扫描页会给出诊断，纯扫描件失败关闭；配置后只消费页级
Markdown，不下载远端图片资源。未配置 LLM 时使用确定性规划器和只复述证据的确定性合成器。
embedding 未配置时只注册 BM25 工具；配置 OpenAI-compatible HTTPS endpoint 后额外注册 hybrid/RRF 工具，
模型可自主选择，远程调用继续要求双重网络授权。系统不会把 DeepSeek 对话接口误作 embedding 接口。

项目根目录 `.env` 被 Git 忽略；本地使用前可执行 `set -a; source .env; set +a`。生产应使用 Secret Manager，而不是把密钥放进镜像或版本库。

## 状态语义

- `succeeded`：校验通过且满足所需证据覆盖。
- `degraded`：有可靠证据，但部分来源或字段缺失。
- `failed`：无证据，或最终报告未通过硬校验。

报告会显示迭代数、工具调用数、停止原因、数据缺口、来源脚注和非投资建议声明；API 结果中的
`context_manifests` 记录每次规划/生成实际纳入和省略的证据、来源类型、分组及字符预算。

## 生产注意事项

核心 Agent 已做到可验证、失败关闭，但整个部署尚未完成生产认证。当前 HTTP API key 是单部署身份边界，
所以开箱形态按单用户部署使用；Service 层已有 tenant/user 隔离，但多用户 SaaS 必须先接 OIDC/API gateway，
不能信任客户端自报身份。同步 provider 会阻塞单个 event loop；SQLite 个人数据未做静态加密；Redis list worker、
licensed market feed 和 append-only audit 尚未达到生产要求，详见企业评测报告。

Docker Compose 启动前必须显式设置 `POSTGRES_PASSWORD`，仓库不提供数据库默认密码。
