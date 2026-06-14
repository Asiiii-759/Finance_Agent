# Finance RAG 当前架构说明

版本：stage-2-wsl  
日期：2026-06-13

## 当前目标

第一阶段和第二阶段已经收敛为一个个人项目可维护的 RAG 内部模块。它不提供 FastAPI 和前端，而是作为后续金融 agent 的 Python 工具使用。

当前能力边界：

- 输入：`Data/knowledge_base/<kb>/content` 中的 PDF，以及 `raw_resolve` 中同名 JSON。
- 解析：优先复用已有 JSON；JSON 缺失时才懒加载 PDF/OCR 解析器。
- 切块：基于原始字符坐标，保留 `global_start/global_end`。
- 入库：SQLite 元数据 + FAISS 向量库。当前主开发环境为 WSL2 `FinAgent`，已验证 pip `faiss-gpu==1.14.3` 可用；通过 `FINANCE_RAG_FAISS_DEVICE=gpu` 启用 GPU。
- Embedding：默认阿里云百炼 `text-embedding-v4`，兼容 OpenAI-style embedding API。
- Rerank：已有阿里云百炼 `qwen3-rerank` provider，默认不自动调用。
- 评估：支持 span overlap 指标，适合用坐标重叠评估召回。

## 当前开发环境

```text
OS: WSL2 / Linux x86_64
conda env: FinAgent
Python: 3.11
FAISS: faiss-gpu 1.14.3，已验证 GPU API 可用且 `get_num_gpus() == 1`
PyTorch: torch 2.11.0+cu128，已验证 `torch.cuda.is_available() == True`
```

说明：
- RAG 核心链路不依赖本地 PyTorch；当前 torch 主要为后续可能接入本地 embedding/reranker/LLM 模型预留。
- `FINANCE_RAG_FAISS_DEVICE` 代码默认仍为 `cpu`，便于无 GPU 环境导入和测试；在本机 WSL2 开发/入库时应在 `.env` 或 shell 中显式设置为 `gpu`。
- 如果后续确认新的部署或开发基线，本文件必须同步更新，避免架构说明与代码事实漂移。

## 目录结构

```text
Finance_RAG/
├─ Data/
│  ├─ knowledge_base/
│  │  └─ Finance/
│  │     ├─ content/          # PDF
│  │     ├─ raw_resolve/      # PDF 解析后的 JSON
│  │     └─ vector_store/     # FAISS 索引
│  └─ logs/
├─ db/
│  └─ knowledge_repository.py # sqlite3 元数据存储
├─ parser_chunk_search/
│  ├─ chunker.py              # 字符坐标安全切块
│  ├─ embedding.py            # Mock / OpenAI-compatible / AliyunBailian
│  ├─ kb_service.py           # KB 服务与 FAISS/BM25/RRF
│  ├─ native_faiss.py         # FAISS 封装
│  └─ pdf_parser.py           # OCR/PDF 解析，懒加载使用
├─ providers/
│  └─ rerank.py               # qwen3-rerank / noop rerank
├─ tests/
├─ evaluation.py              # span overlap 评估
├─ rag.py                     # 对外函数入口
├─ settings.py
└─ requirements-rag-gpu.txt
```

## Embedding

当前推荐 provider：

```text
AliyunBailianEmbeddings
  model: text-embedding-v4
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  dimensions: 1024
  batch_size: 10
```

保留 `MockEmbeddings`，仅用于本地 smoke test，不用于召回质量评估。

环境变量：

```bash
export DASHSCOPE_API_KEY="sk-..."
export FINANCE_RAG_EMBEDDING_PROVIDER="dashscope"
export FINANCE_RAG_EMBEDDING_MODEL="text-embedding-v4"
export FINANCE_RAG_EMBEDDING_DIMENSIONS="1024"
```

## 切块与坐标

当前规则：

- `global_start/global_end` 是 1-based 闭区间。
- chunk 内容必须满足：

```python
chunk["content"] == full_text[global_start - 1:global_end]
```

- tokenizer 只用于长度估算，不参与 `token -> text -> find` 的反向定位。
- 超长表格行会按原文字符窗口安全切分。
- 如果 JSON 缺少 block 坐标，chunker 会按 `block_content` 顺序即时补齐。

这保证了后续召回评估可以稳定基于字符 span。

## 检索流程

```text
query
  -> dense vector search
  -> BM25 keyword search
  -> RRF / weighted fusion
  -> optional rerank
  -> RetrievalResult[]
```

当前默认：

- 向量库：FAISS。当前 WSL2 `FinAgent` 已安装并验证 `faiss-gpu`；实际启用 GPU 需要设置 `FINANCE_RAG_FAISS_DEVICE=gpu`。
- 关键词：jieba + rank-bm25
- 融合：RRF
- rerank：默认关闭

## 召回评估

建议用 span overlap 而不是 chunk id 作为评估基准。

评估字段：

- `coverage@k`：retrieved spans 覆盖 gold spans 的字符比例。
- `iou@k`：retrieved spans 与 gold spans 的交并比。
- `hit@k`：是否存在 gold span 被覆盖超过阈值。
- `span_recall@k`：被命中的 gold span 数 / gold span 总数。

实现位置：

```text
Finance_RAG/evaluation.py
```

## Agentic RAG 扩展规划

当前模块应继续保持为 RAG Core：负责文档解析、切块、入库、检索、rerank 和评估。后续 agentic RAG 不应直接耦合 `KBService`、`NativeFAISS` 或 chunker 内部字段，而应通过稳定工具层接入。

推荐分层：

```text
RAG Core
  parser -> chunker -> indexer -> retriever -> reranker -> evaluator

RAG Tool Layer
  FinanceRAGTool.search(...) -> RetrievalResult
  citation / evidence formatting
  stable JSON schema for agent

Agentic RAG Orchestrator
  query rewrite
  multi-query retrieval
  query decomposition
  retrieval self-check
  answer grounding check
  retry / broaden / narrow retrieval
```

原则：
- agent 只能依赖稳定工具接口，不直接依赖底层向量库、切块器或 SQLite repository。
- agentic 策略作为编排层新增，不塞进 RAG Core。
- 每次确认采用新的检索策略、metadata 策略或 parser provider 后，必须同步更新本文档。

### 渐进式 Schema

当前已有 `raw_resolve` JSON 信息不完整，因此 schema 不能设计成“所有字段都必须拿到”。下一阶段采用渐进式 schema：核心字段必须稳定，增强字段允许为空或候选化。

第一版必须落地：

```text
DocumentMetadata
  doc_id, kb_name, source_type, source_uri
  file_name, file_ext, file_size, file_mtime, imported_at, content_hash
  parser_name, parser_version, parse_status
  document_title?, document_date?, publish_date?, report_type?

ChunkMetadata
  chunk_id, doc_id, kb_name, file_name, source_type, content_type
  source_page?, block_label?, paragraph_title?
  global_start, global_end

RetrievedChunk
  id, content, metadata, rank, score, scores

RetrievalResult
  query, chunks, filters, top_k, rerank_model?, trace?
```

第二阶段再补充候选实体：

```text
EntityCandidate
  name, normalized_name?, entity_type, code?, market?
  confidence, evidence, evidence_span?

DocumentMetadata extras
  organization?, authors?, companies[], industries[], tickers[], tags[]
```

原则：
- `?` 字段允许为空，`[]` 字段允许为空列表。
- 公司、行业、股票代码只作为候选实体，不作为强事实字段。
- 每个候选实体必须带 `confidence` 和 `evidence`。
- `scores` 支持 `vector / bm25 / rrf / rerank / final` 等多路分数，避免后续融合策略变化时破坏接口。

### 金融研报 Metadata 策略

金融研报不能假设一定对应单一公司。公司、股票代码、行业、发布机构都应作为候选实体保存，而不是强行写成单值字段。

第一版不依赖额外 LLM 校验，先用规则和证据来源做保守抽取：
- 文件名、文档标题、首页/封面、段落标题、正文高频实体、股票代码模式、表格标题都可以作为候选来源。
- `evidence` 记录候选实体来自哪里，例如 `filename/title/cover/body_frequency/ticker_regex/parser_json/manual`。
- `confidence` 根据来源组合粗略打分，例如标题命中高于正文高频，股票代码命中高于普通公司名命中。
- `report_type` 允许 `company / industry / macro / strategy / thematic / unknown`，行业研报可以没有主公司。
- 高置信候选可用于过滤；低置信候选只参与 rerank、提示或 trace，不作为硬过滤条件。

时间字段分层：
- `file_mtime`：本地文件修改时间，只代表文件状态。
- `imported_at`：进入知识库的时间。
- `document_date`：文档内部出现的日期，例如封面报告日期。
- `publish_date`：发布机构声明的发布日期；无法可靠解析则为空。

agent 和评估应优先使用 `document_date/publish_date`，不要把 `file_mtime` 误当研报发布时间。

### 文档解析 Provider

当前 PDF 解析优先复用 `raw_resolve` JSON，JSON 缺失时才调用外部解析。后续应抽象 parser provider，避免 OCR API、Word、Excel 等新来源挤进当前 PDF parser。

建议接口：

```text
DocumentSource
  source_type, uri, file_name, metadata_hint

DocumentParserProvider
  supports(source) -> bool
  parse(source) -> ParsedDocument

ParsedDocument
  document_metadata, blocks, full_text, parser_trace
```

建议 provider 路线：
1. `ResolvedJsonParser`：复用当前 `raw_resolve` JSON，作为第一优先级。
2. `PaddleOcrApiParser`：调用 PaddleOCR API，替代本地 PaddleOCR 大模型。
3. `LocalPaddleOcrParser`：保留为可选本地 provider，不作为默认依赖。
4. `DocxParser` / `ExcelParser`：后续新增 Word/Excel 数据源时接入。

`PaddleOcrApiParser` 设计约束：
- API token 不写入代码和文档，统一从环境变量读取，例如 `PADDLEOCR_API_TOKEN`。
- 默认模型为 `PaddleOCR-VL-1.6`，但模型名也应通过环境变量覆盖。
- 解析结果优先保存原始 JSONL/Markdown 产物，再转换为项目统一的 `ParsedDocument`。
- 需要记录 `job_id / model / optional_payload / start_time / end_time / page_count / parser_trace`，方便失败重试和成本核算。
- 每天页数额度有限，第一轮只抽 4-5 个 PDF 做结构校准，不直接全量重跑。

PaddleOCR API 校准流程：
1. 选取 4-5 份代表性 PDF：公司研报、行业研报、宏观/策略报告、表格较多报告、图表较多报告。
2. 保存 API 原始返回，检查 markdown、图片、表格、页码、标题层级是否稳定。
3. 编写 converter，将 API 返回转为统一 `ParsedDocument.blocks`。
4. 对比已有 `raw_resolve` JSON，确认 `global_start/global_end`、页码、段落标题、表格文本是否可稳定生成。
5. 只在 converter 稳定后，再考虑批量重新解析。

当前校准脚本：

```bash
cd /home/pjx/git-repository/Agent
conda activate FinAgent
export PADDLEOCR_API_TOKEN="..."

# 不传 PDF 时，会默认选择 content 目录里体积最小的 2 个 PDF。
python -m Finance_RAG.calibration.paddle_ocr_probe --limit 2

# 也可以显式指定 1-2 个代表性 PDF。
python -m Finance_RAG.calibration.paddle_ocr_probe \
  "Finance_RAG/Data/knowledge_base/Finance/content/商业航天行业点评：SpaceX百万颗算力卫星申请，太空光伏、激光通信产业迎来新机遇.pdf"
```

输出位置：`Finance_RAG/Data/calibration/paddleocr_api/`，该目录已加入 `.gitignore`。

每个 PDF 会保存：
- `raw.jsonl`：PaddleOCR API 原始 JSONL 返回。
- `legacy.json`：转换到当前 chunker 可消费的结构。
- `summary.json`：不含正文全文的结构摘要，包括 `document_info` key、块标签分布、前几个块的形状、metadata 候选和 warning。

当前代码落地状态：
- 已新增 `Finance_RAG/schemas/document.py`，提供 `ParsedDocument / ParsedBlock / DocumentParser` 抽象，旧 JSON 和新 parser 都先转换到兼容 chunker 的结构化文档。
- 已新增 `Finance_RAG/schemas/metadata.py`，提供 `FinanceMetadataExtractor / MetadataCandidate / MetadataExtractionReport`。
- 已新增顶层 `Finance_RAG/parsers/`，其中 `ResolvedJsonParser` 读取现有 `raw_resolve` JSON 并补充 metadata 抽取报告，`PaddleOcrApiParser` 负责 PaddleOCR API。
- 已新增 `Finance_RAG/calibration/paddle_ocr_probe.py`，用于 1-2 个 PDF 的真实 API 小样本校准。
- `Finance_RAG/document.py`、`Finance_RAG/metadata_extractor.py`、`Finance_RAG/parser_chunk_search/parsers/*` 只保留兼容入口，后续新代码优先使用 `schemas/` 和 `parsers/`。
- 缺 JSON 时可通过 `FINANCE_RAG_PARSER_PROVIDER=paddleocr_api` 调用 API；token 只从 `PADDLEOCR_API_TOKEN` 读取。
- `FinanceMetadataExtractor` 只输出候选实体、证据和置信度，不把公司、行业、作者等字段强行当作确定事实。
- chunk metadata 已开始携带 `document_title / document_date / publish_date / report_type / metadata_extraction`，便于后续检索结果和评估使用。

因此最终 `DocumentMetadata / RetrievedChunk / RetrievalResult` 先不继续固化。下一步应先用 4-5 份 PDF 校准 PaddleOCR API 和规则抽取结果，再根据“实际能稳定得到的字段”收敛 schema。

已用少量现有 `raw_resolve` JSON 做结构摘要校准，结论如下：
- 旧 JSON 的 `document_info` 通常只有 `doc_source / doc_title / file_name`，日期、作者、公司、行业不是稳定字段。
- `doc_source` 可能被 OCR 识别成错误句子，例如英文的模糊图片提示，因此机构只能作为候选并需要过滤。
- `doc_title` 可能是 `未命名文档`，也可能缺失；此时应回退到文件名，不应回退到图表标题。
- 文件名里可能包含 `94.8%` 这类小数，不能用普通 `Path.stem` 盲目截断，只能剥离明确的文件扩展名。
- 行业关键词目前只适合作为低置信候选，不能用于硬过滤；股票代码可作为 `company_or_security` 候选，但不等于主公司。

### Tool Layer

后续 agent 应只调用工具层：

```text
FinanceRAGTool.search(
  query,
  kb_name="Finance",
  exp_name="default",
  top_k=5,
  filters=None,
  search_mode="rrf",
  rerank=False,
  return_trace=True,
) -> RetrievalResult
```

工具层职责：
- 统一调用 `retrieve_documents(...)` 和可选 rerank。
- 将当前 dict 结果转换为 `RetrievedChunk`。
- 补齐 document metadata、score breakdown 和 retrieval trace。
- 为后续 agent 提供稳定 JSON schema。

## 测试

运行：

```bash
cd /home/pjx/git-repository/Agent
conda activate FinAgent
python -B -m unittest discover -s Finance_RAG/tests -v
```

覆盖：

- 真实 JSON chunk 坐标一致性。
- 超长 HTML 表格行切分。
- span overlap 评估。
- mock embedding + FAISS 入库检索 smoke。

## 后续重构建议

下一步优先级：

1. 用 4-5 份代表性 PDF 跑 `PaddleOcrApiParser`，保存并检查 API 原始返回和转换后的 legacy JSON。
2. 对比 `FinanceMetadataExtractor` 的候选实体、证据和置信度，调整规则，删掉无法稳定获得的理想字段。
3. 根据校准结果再落地 `DocumentMetadata / ChunkMetadata / RetrievedChunk / RetrievalResult / RetrievalTrace`。
4. 用稳定 schema 包装当前 `retrieve_documents(...)` 返回值，短期保留旧 dict 结果兼容。
5. 封装 `FinanceRAGTool.search(...)`，供后续 agent 直接调用。
6. 将 `qwen3-rerank` 接入统一检索入口；当前已有独立 `rerank_documents(...)`，但 `retrieve_documents(...)` 尚未内置 rerank 开关。
7. 用真实百炼 embedding 跑 2-3 篇文档的小规模入库。
8. 构造最小评测集：`question + gold_spans`，并用 `evaluation.py` 跑 `coverage@k / iou@k / hit@k`。

Milvus 暂不在当前阶段实现。后续如需要规模化，再按 `VectorStore` 接口新增 Milvus 后端。
