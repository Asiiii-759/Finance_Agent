# RAG、PDF 与文档生命周期

## 1. 三种文档语义

| 类型 | 创建方式 | 生命周期 | 是否自动长期入库 |
|---|---|---|---|
| request | 本轮随问题上传 | run 结束清理原文件 | 否 |
| session | 上传时显式保留到线程 | TTL、线程删除或主动删除 | 否 |
| personal | 个人知识库接口明确上传 | 用户删除前 | 是 |

用户临时综合多个 PDF 不会污染个人知识库。当前附件只以 `ChatAttachment(document_id,title)` 进入 ChatTurn；完整文档目录不会
预先塞进任务理解 Prompt。

## 2. PaddleOCR 解析

生产 PDF/图片解析只走配置的 PaddleOCR-VL-1.6 或显式注入的外部 Parser，不使用 PyMuPDF 文本抽取作为隐式降级。

解析结果保留：

- page、order；
- heading/text/table/chart 类型；
- Markdown 文本；
- title；
- bbox；
- 页级图片引用。

客户端限制 HTTPS endpoint、文件大小、页数、轮询状态、响应大小和下载 URL。网络解析需要部署授权和用户本轮授权同时成立。

## 3. Token-aware 分块

默认 1024 token 一块、256 token overlap，使用本地 BGE-M3 tokenizer；没有 tokenizer 文件时使用明确记录名称的估算器。

分块遵循：

- 优先标题、段落和自然边界；
- 不跨页静默合并 citation；
- Markdown/HTML 表格保持表头，HTML chunk 始终闭合；
- 保存 `global_start/global_end`、page、block 类型和原始顺序；
- chunking version 写入持久索引 manifest。

## 4. 持久索引

个人知识库在写入/显式 reindex 时分批计算 embedding，查询时不重新向量化整库。Manifest 绑定：

- embedding endpoint/model/dimension；
- chunking version；
- index status 和时间；
- document/tenant/user scope。

模型或分块版本不一致时 hybrid 工具不可宣称 ready，必须 reindex。大文档 embedding 默认分批处理，避免一次请求撑爆内存或远端限制。

## 5. 双路召回

只有 embedding 配置完整时注册 hybrid 工具：

```text
BM25 lexical candidates
             ┐
             ├─ vector minimum similarity filter → RRF → top-k
embedding cosine candidates
```

当前 BGE-M3 本机金融样本校准后默认 cosine 阈值为 0.5。RRF 只负责融合排名，不适合承担绝对相关性阈值，所以向量候选必须在
融合前过滤。BM25 和向量都没有可信匹配时返回空 bundle + Gap，而不是硬塞 top-k。

## 6. 多文档与重叠合并

默认使用全局相关排序，不强制每份上传文档占一个名额。只有用户明确要求跨多文档比较/综合时，TaskFrame 才声明
`minimum_documents`，Planner 可传 `diversify_documents=true`。

最终选中的同文档相邻/重叠 chunk 按原始坐标合并为完整 passage；不是用字符串相似度猜测，也不逐条字符截断。Context
Assembler 在总 token 预算内选择完整 passage，并记录省略 Evidence ID。

## 7. 检索不足后的闭环

确定性 Coverage 只证明来源类别、实体、字段和最低文档数。Planner 显式 finish 后，文档/网页 Evidence 还会经过语义充分性检查：

- 足够：进入生成；
- 缺失：产生 missing information + retrieval hint，重新规划；
- 冲突：产生 conflict Gap，要求补来源或在最终报告中保留冲突；
- 达到预算仍不足：明确拒答，不把弱相关 chunk 当成答案。

## 8. ACL 与删除

request/session corpus 绑定当前 tenant/user/thread；personal corpus 绑定 tenant/user。模型参数不能覆盖固定 ACL filter。删除个人文档会
同时删除页、结构化 blocks、chunks、向量和 manifest；删除线程传播到 session corpus。服务端文件名使用安全 UUID，API citation
只显示用户文件名。
