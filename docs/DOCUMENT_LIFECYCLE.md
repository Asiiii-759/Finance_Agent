# PDF、RAG 与记忆生命周期设计

状态：与 2.2 实现同步
日期：2026-08-20

## 1. 结论

文档不是一种“记忆”。系统把用户 PDF 分成三个必须显式选择的生命周期：

| 模式 | 适用场景 | 保存内容 | 保存位置 | 召回方式 | 删除 |
|---|---|---|---|---|---|
| 一次性请求 | 临时综合分析一个或多个 PDF | 本次进程中的页文本和 chunks | 当前 run 内存 | BM25；配置后可 hybrid/RRF | 请求结束删除原 PDF，run 结束释放语料 |
| 会话文档 | 围绕同一批 PDF 连续追问 | 解析后的非空页文本与 provenance | API 进程内存，默认 TTL 1 小时 | 同一 tenant/user/thread；BM25 或配置后 hybrid | TTL 到期或显式 DELETE；原 PDF 仍在上传请求结束后删除 |
| 个人知识库 | 单用户明确长期保存的个人 PDF | 解析页文本与 provenance，不保存原 PDF | 本地 SQLite，只持久页文本 | BM25；配置后查询期 embedding/RRF | 单文档显式 DELETE |
| 企业知识库 | 经治理的公司资料、研报、制度文档 | 文档对象、chunks、ACL、版本和索引 manifest | 外部受控 RAG/知识库 | 部署注入 `RetrievalSource/evidence_tools` | 由知识库执行权限校验、版本、删除传播和审计 |

临时上传绝不会自动“升级”为个人或企业永久知识。任何持久化都不由模型决定。

## 2. 为什么这样划分

一次性多 PDF 分析只需要在当前 run 建一个统一语料库。把它们默认写入长期向量库会引入重复文档、错误 ACL、删除困难和跨用户泄漏，而且用户通常没有表达长期保存同意。

连续追问需要跨 HTTP 请求召回，但仍不等于永久知识。因此会话层只保存已经解析的页文本，具备短 TTL、显式 opt-in 和删除接口；不保存原 PDF，不写 SQLite，也不进入企业索引。

个人知识库面向当前单用户部署：用户必须调用独立路由明确持久化，系统只保留解析页文本并提供列表/删除；Service 查询始终带 tenant/user namespace。它不是企业文档治理系统。当前 HTTP API key 只代表单部署访问，尚不能可靠派生多用户 principal 与 ACL，因此多用户 SaaS 仍必须先接认证网关。企业知识库应在独立 ingestion service 中完成病毒扫描、分类、ACL、加密、版本、retention、删除传播和审计，再通过固定 `RetrievalSource/evidence_tools` 给 Agent 只读检索。

## 3. 一次性多 PDF

`POST /api/v1/analyze-upload` 默认执行：

```text
多个 PDF
  → 数量/字节/后缀/PDF magic 校验
  → 临时随机文件名
  → 原生页文本提取，必要时受控 OCR
  → 所有非空页面合并进同一个 request corpus
  → corpus.search 或 corpus.hybrid_search 返回相关 PDF 证据
  → EvidenceBundle / claim / report
  → finally 删除所有原始上传文件
```

每条 evidence 仍保留原始显示文件名、页码、chunk 和内容哈希。多 PDF 合并只发生在检索视图，不会丢失来源边界。

## 4. 会话文档

首次上传时显式保留解析文本：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze-upload \
  -F 'query=综合分析两份材料的盈利与信用风险' \
  -F 'thread_id=credit-review-42' \
  -F 'retain_for_session=true' \
  -F 'files=@./results.pdf' \
  -F 'files=@./credit.pdf'
```

后续 JSON 请求显式召回：

```json
{
  "query": "这两份材料对 covenant headroom 的描述是什么？",
  "thread_id": "credit-review-42",
  "use_session_documents": true,
  "export_artifacts": false
}
```

管理接口：

- `GET /api/v1/session-documents/{thread_id}`：只返回 document ID、文件名、页数、OCR 页数和到期时间，不返回正文。
- `DELETE /api/v1/session-documents/{thread_id}`：删除该线程全部会话文档。
- `MAS_SESSION_DOCUMENT_TTL_SECONDS`：默认 3600，最小 60 秒。
- `MAS_MAX_SESSION_DOCUMENT_SESSIONS`：默认每进程最多 100 个会话文档 namespace；写入时先清理全部过期 namespace，达到上限后快速失败。

约束：

- `use_session_documents=true` 必须提供 `thread_id`；不存在文档时返回正常的无证据/缺口语义，不会偷偷搜索其他线程。
- namespace 是 tenant/user/thread 的稳定哈希；相同 thread 名在不同 user 下不可见。
- 同一文档按内容哈希去重；会话文档数沿用上传数量上限，页文本总量受 PDF 文本字符上限约束。
- 会话内容只在单个 API 进程中共享；进程重启或请求落到另一 worker 后不可见。这是刻意的短期语义，不是高可用存储承诺。
- 若生产需要跨 worker 的短期文档，应将同一契约迁移到带 TTL、加密和 tenant ACL 的 Redis/对象存储；这不改变 Agent、Evidence 或检索接口。

响应中的 `document_diagnostics[].lifecycle` 为 `request`、`session_retained` 或 `session`，`session_document_count` 给出当前线程保留数量。

## 5. 个人与企业永久知识库边界

个人持久上传：

```text
POST /api/v1/knowledge/documents
  → 与临时上传相同的 PDF/OCR 边界
  → SQLite personal_documents + personal_document_pages
  → 原 PDF finally 删除
  → personal.search 动态注册；配置 embedding 时同时注册 personal.hybrid_search
```

管理接口为 `GET /api/v1/knowledge/documents` 和
`DELETE /api/v1/knowledge/documents/{document_id}`。默认每用户最多 100 份，可通过
`MAS_MAX_PERSONAL_KNOWLEDGE_DOCUMENTS` 调整。列表只返回 ID、文件名、页数、字符数和创建时间，不返回正文。

企业 Agent 只消费 canonical 检索契约：

Agent 只消费 canonical 检索契约：

```text
query + top_k + server-owned filters + optional diversify_documents
  → fixed HTTPS RAG gateway
  → lexical/vector/hybrid/RRF search；rerank 由 gateway 在声明支持时执行
  → bounded canonical chunks
  → RetrievalEvidenceAdapter
  → EvidenceBundle
```

`fixed_filters` 必须由部署端绑定，模型和请求参数不能放宽。当前仓库提供 `HTTPJSONRAGClient / RetrievalSource` 与 read-only `evidence_tools` 注入接口及故障测试。开放网页另走 `web.search` 的固定 Bocha/Brave API origin，只返回搜索摘要并降级为 inferred，不会假装成企业正文检索。

永久 ingestion 控制面至少需要以下字段后才应开放：

- 服务端认证得到的 principal、tenant 与 ACL；
- `retention=persistent` 的明确用户意图；
- 文件内容哈希、版本、来源、所有者、分类和加密信息；
- ingest 状态、解析/OCR 诊断、索引版本与失败原因；
- list/get/delete/export，以及原文件、chunk、embedding、cache 的删除传播；
- append-only 访问审计和供应商数据保留政策。

个人路由不具备上述企业治理保证；在这些边界完成前，不能把个人 SQLite 库包装成共享企业知识库。

## 6. 与短期记忆、Prompt 的关系

线程记忆只保存上一问题、实体、symbol、状态和 gap code，用于理解“它”“继续”等指代；它不是事实来源。会话或个人文档正文只在用户显式启用时进入当次 corpus，检索出的有限 evidence cards 才会进入 LLM 上下文。未召回页面、原 PDF、密钥和模型隐藏推理都不会进入 prompt。

文档内文本始终是不可信数据，即使包含“忽略系统规则”也不能注册工具、改变网络权限或扩大预算。LLM 输出还必须提供 context manifest 内 evidence 的逐字 quote；失败时使用确定性合成并暴露 gap。

## 7. 已验证场景

- 两份 PDF 合并检索，证据分别保留文件和页码。
- 原 PDF 删除后，同线程可显式召回解析文本并完成追问。
- 同名 thread 在不同 user namespace 下隔离。
- 列举不返回正文，删除后立即不可召回。
- 个人 PDF 跨 Service 重启仍可检索；Alice 的文档对 Bob 不可见；显式删除后消失。
- 未提供 thread 却请求会话文档时快速失败。
- 真实 DeepSeek V4 Flash：两 PDF、2 条 evidence、2 条 supported claim、1 次模型调用、无验证错误。
- 真实 PaddleOCR-VL-1.6：单页扫描 PDF 成功提取页级 Markdown；自动化测试覆盖未授权、无 OCR、有限轮询和畸形响应。
