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

1. 用真实百炼 embedding 跑 2-3 篇文档的小规模入库。
2. 构造最小评测集：`question + gold_spans`。
3. 用 `evaluation.py` 跑 `coverage@k / iou@k / hit@k`。
4. 将 `qwen3-rerank` 接入统一检索入口；当前已有独立 `rerank_documents(...)`，但 `retrieve_documents(...)` 尚未内置 rerank 开关。
5. 封装 `FinanceRAGTool`，供后续 agent 直接调用。

Milvus 暂不在当前阶段实现。后续如需要规模化，再按 `VectorStore` 接口新增 Milvus 后端。
