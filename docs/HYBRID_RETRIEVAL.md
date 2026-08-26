# BM25 + Embedding 双路检索设计

状态：已实现，embedding 模型待部署配置
版本：2.2
日期：2026-08-20

## 1. 结论与边界

本地 request/session corpus 和个人知识库现在具备同一套双路检索实现：

```text
                         ┌─ BM25 词法排序 ───────┐
query + authorized chunks                         ├─ RRF(k=60) ─ top-k ─ Evidence
                         └─ embedding + cosine ──┘
```

代码已经实现 provider 协议、受限 HTTP embedding 客户端、向量校验、余弦排序、RRF 融合、缓存、Harness
权限隔离、trace 与失败边界；仓库不内置假语义模型，也不指定未经评测的生产模型。未配置 embedding 时只注册
lexical 工具，不把 BM25 结果标成 hybrid/RRF。

这不是固定 workflow。模型从动态工具目录自主选择 lexical、hybrid、企业 RAG、网页或其他金融工具；Harness
只限制执行边界。无模型或模型计划失败时，规则基线在 embedding 可用且网络已授权时优先 hybrid，再尝试 lexical。

## 2. 为什么必须双路

BM25 擅长精确名称、ticker、会计科目、covenant 条款、数字附近的关键词和专有名词；它无法可靠处理：

- “资金充裕”与 “liquidity resilience” 这类跨语言或同义改写；
- “降低再融资依赖”与“改善债务到期墙”这类金融语义关联；
- 用户不知道原文术语，只能描述概念的场景。

embedding 擅长语义近邻，但可能把措辞相似、事实不相关的段落排高，也不天然优先 ticker、数字和精确字段。
因此系统不以向量检索替换 BM25，而是独立产生两个排名后用 RRF 融合。RRF 只使用名次，不假定 BM25 与
cosine 分数量纲可比较。

## 3. 真实工具边界

| 工具 | 固定模式 | 网络属性 | 注册条件 | 典型用途 |
|---|---|---|---|---|
| `corpus.search` | lexical/BM25 | 否 | 当前请求或会话存在 PDF | ticker、字段、数字、精确条款 |
| `corpus.hybrid_search` | BM25 + vector + RRF | 跟随 embedding provider | PDF 存在且配置 embedding | 同义改写、跨语言、概念描述 |
| `personal.search` | lexical/BM25 | 否 | 当前用户有持久文档 | 个人库精确查询 |
| `personal.hybrid_search` | BM25 + vector + RRF | 跟随 embedding provider | 个人文档和 embedding 都存在 | 个人库语义查询 |

模式是工具的固定属性，不是同一个本地工具中的模型参数。这一点解决了条件网络调用问题：如果
`corpus.search` 接受 `search_mode=hybrid`，模型可以通过一个被声明为本地的工具间接触发远程 embedding，Harness
将无法在调用前正确执行网络授权。现在 lexical 工具拒绝多余的 `search_mode/rerank` 参数，hybrid 工具在执行前按
provider 的 `network_access` 经过双重网络授权、预算、timeout/retry 和审计。

部署注入的 `RetrievalSource` 仍可接受标准 `lexical/vector/hybrid/rrf` 参数，因为它本身是一个已声明网络属性的
受控 gateway。内置 corpus 不支持 reranker；传入 `rerank=true` 会快速失败，不会假装已执行。

## 4. 检索算法

### 4.1 词法路

文档按 1600 字符切块，默认重叠 200 字符。英文/数字使用规范化 token，中文使用双字切分；对通过 metadata
filter 的候选执行 BM25-style 排序，只保留正分候选。

### 4.2 向量路

hybrid 首次查询把缺少缓存的 candidate chunk 与 query 一次提交给 `EmbeddingProvider.embed_texts()`；chunk
向量做 L2 归一化并在当前 corpus 实例缓存，后续查询只生成 query 向量。余弦相似度产生独立全局排名。

缓存严格服从文档生命周期：request corpus 随 run 释放；session corpus 在每次请求重建；个人库每次分析创建
一个快照并仅在该分析循环内复用。当前不会把 embedding 永久写进个人 SQLite，因此没有隐式持久化、模型版本
迁移或删除传播问题，但重复查询会重新计算。这适合当前默认最多 100 份个人文档的小规模正确性路径，不是大规模
生产索引承诺。

### 4.3 RRF 融合

每个 chunk 的融合分数为：

```text
rrf(chunk) = Σ 1 / (60 + rank_channel(chunk))
```

只在该路存在排名时累加。输出保留 `bm25`、`cosine`、`lexical_rank`、`vector_rank` 和 `rrf`，并在 trace 中记录
backend、search mode、fusion、embedding backend/model、候选数与是否启用文档分散。这些排序分数只用于检索
审计，不转成 Evidence confidence。

### 4.4 多文档策略

BM25 和向量路都先对所有授权 chunks 做全局相关排序，RRF 之后才执行可选的 `diversify_documents`。默认不保证
每份 PDF 都占名额；用户只问一个数值时，结果可以全部来自一份最相关文档。只有模型判断或明确规则识别到“综合、
比较、逐份”等意图时才启用文档分散，并把 top-k 提高到可用文档数（最高 20）。这是一种研究意图约束，不是
检索器的固定偏好。

## 5. Embedding 接口与部署

可直接注入内部实现：

```python
class EmbeddingProvider(Protocol):
    backend_name: str
    model_name: str
    network_access: bool

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
```

也可以配置 OpenAI-compatible `/embeddings` HTTP endpoint：

```dotenv
MAS_EMBEDDING_ENDPOINT=https://embedding.example.com/v1/embeddings
MAS_EMBEDDING_MODEL=your-evaluated-model
MAS_EMBEDDING_API_KEY=
MAS_EMBEDDING_TIMEOUT_SECONDS=30
```

本机 vLLM 服务可使用：

```dotenv
MAS_EMBEDDING_ENDPOINT=http://127.0.0.1:8001/v1/embeddings
MAS_EMBEDDING_MODEL=BAAI/bge-m3
```

请求固定为 `{"model": ..., "input": [...]}`，响应要求 `data[].index` 与 `data[].embedding`。endpoint 来自部署
配置，必须是无内嵌凭据、无 query/fragment 的 HTTPS URL（本机仅额外允许 loopback HTTP），不跟随 redirect；一次调用最多 512 个文本、单文本
最多 32000 字符、总字符和响应字节均有硬上限。响应必须数量一致、索引完整、维度一致、全部有限且非零。任一
条件不满足即失败，错误进入 Agent gap/audit，再由 planner 决定是否尝试 lexical，而不是在检索器内部静默降级。

DeepSeek 对话配置不被当作 embedding 配置；两者是不同模型边界。只有实际提供 embedding endpoint 和经过金融
语料评测的模型后才应启用上述环境变量。

## 6. 与 Prompt、记忆和 Evidence 的关系

- embedding 只决定哪些 chunks 被召回，不把向量、整库或未召回页面放入 prompt；
- 召回结果先转换为带 file/page/chunk locator 的 `Evidence`，再由 ContextAssembler 按 entity/source/document/domain
  与字符预算选择；
- Thread memory 和个人 preference/experience/skill 不参与向量索引，也不能成为事实证据；
- personal knowledge 是用户显式上传的事实资料平面；是否持久保存由 API 生命周期决定，不由 hybrid 检索决定；
- 文档内容始终处于不可信数据区，embedding 命中不会提高其指令权限；
- 模型最终 claim 仍需引用实际 evidence，RRF 高分不能代替 citation 或事实校验。

## 7. 已验证与尚未声称

自动化测试已经覆盖：BM25 零命中而 semantic 命中、RRF trace/score、chunk 向量缓存、临时 PDF、个人持久 PDF、
未配置 provider、未配置 reranker、非法/NaN/零向量、HTTP URL/凭据/顺序契约、lexical 参数走私拒绝、远程 hybrid
网络拒绝与授权，以及同一 chunk 经 lexical/hybrid 重复召回的幂等 Evidence 合并。真实 DeepSeek 还验证了首个动作
自主选择 hybrid、随后 lexical 交叉检查并成功收敛。

尚未声称生产检索质量，因为还缺：

1. 实际 embedding 模型部署与版本固定；
2. 金融/中英标注集上的 Recall@k、nDCG、MRR、citation precision 和 no-answer 测试；
3. 大规模持久向量索引、增量更新、ACL filter 下推、模型迁移和删除传播；
4. reranker 的独立工具/网络预算与收益评测；
5. embedding 服务的共享限流、熔断、成本和延迟 SLO。

生产扩展的下一步不是继续在 Agent 内堆抽象，而是让受治理 retrieval service 持久化 BM25/向量索引并实现同一
canonical contract；Agent 的工具选择、Harness、Evidence 和上下文层无需改变。
