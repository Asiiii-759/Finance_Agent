# 本地 BGE-M3 服务

本项目使用项目内隔离的 `.runtime/vllm-venv` 和 vLLM pooling server 提供 OpenAI-compatible embeddings，避免把模型依赖装进 Agent 运行环境。

## 固定部署参数

- 模型：`BAAI/bge-m3`
- 运行栈：vLLM 0.10.2、Transformers 4.55.2、PyTorch 2.8 / CUDA 12.8
- 本地目录：`.runtime/models/bge-m3`
- 接口：`http://127.0.0.1:8001/v1/embeddings`
- 精度：FP16
- 最大序列长度：2048
- vLLM GPU memory utilization：0.65
- 日志：`.runtime/bge-m3/server.log`

`.runtime/` 已被 Git 忽略。服务只监听 loopback，不向局域网或公网暴露。

## 运维

```bash
scripts/bge_m3_service.sh start
scripts/bge_m3_service.sh status
scripts/bge_m3_service.sh stop
```

Agent 配置：

```dotenv
MAS_ALLOW_NETWORK=true
MAS_EMBEDDING_ENDPOINT=http://127.0.0.1:8001/v1/embeddings
MAS_EMBEDDING_MODEL=BAAI/bge-m3
MAS_EMBEDDING_API_KEY=
MAS_EMBEDDING_TIMEOUT_SECONDS=30
MAS_DOCUMENT_TOKENIZER_PATH=.runtime/models/bge-m3/tokenizer.json
```

文档索引使用该目录中的 BGE-M3 `tokenizer.json`，按最多 1024 tokens 切块，
相邻块保留 256 tokens 重叠。切块后的文本会独立重新计数，避免 SentencePiece
在非文首边界增加词首 token 后超过 1024 的硬上限。

`MAS_ALLOW_NETWORK` 仍由请求级 policy 二次约束。公网 embedding endpoint 必须使用 HTTPS；HTTP 仅允许 `127.0.0.1`、`::1` 和 `localhost`。

健康检查：

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
```
