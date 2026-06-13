# Finance RAG 第二阶段使用说明

## 环境

虚拟环境位置：

```powershell
D:\git-repository\Agent\.venv
```

建议直接使用解释器路径运行，避免 PowerShell 激活脚本策略问题：

```powershell
D:\git-repository\Agent\.venv\Scripts\python.exe -B -c "from Finance_RAG import rag; print('ok')"
```

## Embedding Provider

当前支持两种 provider。

### 1. Mock

只用于 smoke test，不具备语义召回能力。

```powershell
$env:FINANCE_RAG_EMBEDDING_PROVIDER="mock"
```

也可以在调用时传：

```python
update_docs([...], embed_model="mock")
```

### 2. 阿里云百炼 text-embedding-v4

推荐用于第一版真实 RAG。

```powershell
$env:DASHSCOPE_API_KEY="sk-..."
$env:FINANCE_RAG_EMBEDDING_PROVIDER="dashscope"
$env:FINANCE_RAG_EMBEDDING_MODEL="text-embedding-v4"
$env:FINANCE_RAG_EMBEDDING_DIMENSIONS="1024"
$env:FINANCE_RAG_EMBEDDING_BATCH_SIZE="10"
```

官方 OpenAI 兼容 base URL：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Smoke Test

Mock + FAISS：

```powershell
D:\git-repository\Agent\.venv\Scripts\python.exe -B -c "from Finance_RAG.rag import update_docs, retrieve_documents; f='1Q26汽车电子收入预计同比翻倍，海外IDM布局渐完整.pdf'; print(update_docs([f], exp_name='smoke_stage2_mock', embed_model='mock')); print(len(retrieve_documents('汽车电子业务增长情况如何？', exp_name='smoke_stage2_mock', embed_model='mock', top_k=3)))"
```

真实百炼 embedding：

```powershell
$env:DASHSCOPE_API_KEY="sk-..."
$env:FINANCE_RAG_EMBEDDING_PROVIDER="dashscope"
D:\git-repository\Agent\.venv\Scripts\python.exe -B -c "from Finance_RAG.rag import update_docs, retrieve_documents; f='1Q26汽车电子收入预计同比翻倍，海外IDM布局渐完整.pdf'; print(update_docs([f], exp_name='dashscope_v4_1024_test')); print(len(retrieve_documents('汽车电子业务增长情况如何？', exp_name='dashscope_v4_1024_test', top_k=3)))"
```

## 注意

- `text-embedding-v4` 批次大小上限为 10，代码默认按 10 分批。
- 新建 FAISS 索引时，如果配置了 `FINANCE_RAG_EMBEDDING_DIMENSIONS`，不会为了探测维度额外调用一次 embedding API。
- Windows 下当前使用 `faiss-cpu`；embedding 主线走百炼 API。
