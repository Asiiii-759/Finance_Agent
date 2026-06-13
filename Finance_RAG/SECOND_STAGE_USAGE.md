# Finance RAG 第二阶段使用说明

## 环境

当前主开发环境：

```bash
/home/pjx/miniconda3/envs/FinAgent
```

建议在 WSL2 中运行：

```bash
cd /home/pjx/git-repository/Agent
conda activate FinAgent
python -B -c "from Finance_RAG import rag; print('ok')"
```

当前已验证：

```text
faiss-gpu 1.14.3: GPU API 可用，get_num_gpus() == 1
torch 2.11.0+cu128: torch.cuda.is_available() == True
```

## Embedding Provider

当前支持两种 provider。

### 1. Mock

只用于 smoke test，不具备语义召回能力。

```bash
export FINANCE_RAG_EMBEDDING_PROVIDER="mock"
```

也可以在调用时传：

```python
update_docs([...], embed_model="mock")
```

### 2. 阿里云百炼 text-embedding-v4

推荐用于第一版真实 RAG。

```bash
export DASHSCOPE_API_KEY="sk-..."
export FINANCE_RAG_EMBEDDING_PROVIDER="dashscope"
export FINANCE_RAG_EMBEDDING_MODEL="text-embedding-v4"
export FINANCE_RAG_EMBEDDING_DIMENSIONS="1024"
export FINANCE_RAG_EMBEDDING_BATCH_SIZE="10"
```

官方 OpenAI 兼容 base URL：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Smoke Test

Mock + FAISS：

```bash
python -B -c "from Finance_RAG.rag import update_docs, retrieve_documents; f='1Q26汽车电子收入预计同比翻倍，海外IDM布局渐完整.pdf'; print(update_docs([f], exp_name='smoke_stage2_mock', embed_model='mock')); print(len(retrieve_documents('汽车电子业务增长情况如何？', exp_name='smoke_stage2_mock', embed_model='mock', top_k=3)))"
```

如需强制使用 FAISS GPU：

```bash
export FINANCE_RAG_FAISS_DEVICE="gpu"
export FINANCE_RAG_FAISS_GPU_ID="0"
```

真实百炼 embedding：

```bash
export DASHSCOPE_API_KEY="sk-..."
export FINANCE_RAG_EMBEDDING_PROVIDER="dashscope"
python -B -c "from Finance_RAG.rag import update_docs, retrieve_documents; f='1Q26汽车电子收入预计同比翻倍，海外IDM布局渐完整.pdf'; print(update_docs([f], exp_name='dashscope_v4_1024_test')); print(len(retrieve_documents('汽车电子业务增长情况如何？', exp_name='dashscope_v4_1024_test', top_k=3)))"
```

## 注意

- `text-embedding-v4` 批次大小上限为 10，代码默认按 10 分批。
- 新建 FAISS 索引时，如果配置了 `FINANCE_RAG_EMBEDDING_DIMENSIONS`，不会为了探测维度额外调用一次 embedding API。
- 当前 WSL2 `FinAgent` 已支持 `faiss-gpu`；代码默认 device 仍为 `cpu`，需要通过 `FINANCE_RAG_FAISS_DEVICE=gpu` 显式启用。
