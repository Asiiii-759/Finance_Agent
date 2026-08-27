# 真实 LLM、Harness 回退与 Checkpoint 恢复验证

状态：已执行（历史记录，2026-08-12）
日期：2026-08-12
模型：DeepSeek V4 Flash，thinking disabled

当前运行时已不再把规则 planner / 确定性合成器当作降级路径：LLM 是研究链路必需依赖，模型网络拒绝、非法 JSON
或不可用输出会使请求快速失败。内置 `finance.knowledge` 词库也已删除，概念题由模型直接作答。
下文表格保留当时实测结果，其中“确定性合成接管 / `llm_synthesis_fallback`”以及选择 `finance.knowledge`
描述的是当时的行为，不是现行契约。

## 1. 验证原则

真实 LLM 只验证模型真正参与的系统边界：自主工具选择、Evidence 上下文组织、JSON claim 契约、逐字引用、
提示注入隔离、模型预算和 checkpoint 恢复后的合成。公式正确性、权限拒绝、重试计数、畸形 JSON、
租户隔离和 provider schema 不能依赖概率模型判断，仍使用确定性断言与故障注入。

因此“所有工具都用 LLM 测”不等于让 LLM 重新计算或决定测试是否通过，而是：工具先按代码契约产生 Evidence，再由真实模型消费，最终由代码验证 claim 和 citation。

## 2. 真实场景结果

| 场景 | 真实边界 | 结果 |
|---|---|---|
| 十种金融计算批量合成 | ratio、percentage change、CAGR、FV、PV、loan payment、annualized return/volatility、Sharpe、max drawdown | 20 条输入/结果 Evidence，10 条 supported claim，1 次模型调用，无 gap/validation issue |
| 金融知识解释 | 利率传导到企业估值与银行股 | 1 条版本化知识 Evidence，3 条 supported claim，无 gap |
| 主备 RAG + 间接提示注入 | primary 空结果，fallback 文档包含 `ignore system / BUY NOW` | 调用顺序正确；2 条 supported claim；命令没有进入 claim；coverage 完整 |
| 多源综合 | 文档、行情快照、行情历史、SEC facts、10-Q metadata、CPI | 六个研究工具，12 条 Evidence，8 条 supported claim，coverage 完整，无 gap |
| 模型网络拒绝 | 文档检索允许，模型网络未获请求授权 | LLM audit 为 `denied/network_denied`，attempts=0，model_calls=0；确定性合成接管并公开 `llm_synthesis_fallback` |
| 旧循环 checkpoint 恢复（1.4 历史） | evidence 完成后在 synthesis 故意崩溃 | SQLite 保留 evidence/audit；恢复后真实模型完成 2 条 supported claim |
| LangGraph 模型自主规划（2.0） | 动态目录只提供知识与规划工具 | DeepSeek 一次选择 `finance.knowledge` 和合法参数；1 evidence、1 claim、无 gap |
| 实验性 Yahoo 行情 | Apple snapshot + 1 个月历史 | 历史数据、收益/波动/回撤和真实 LLM 合成成功；snapshot endpoint 返回 HTTPStatusError，最终正确为 degraded，未伪装完整成功 |
| 两 PDF 综合 | 两份不同财务 PDF | 2 条文档 Evidence、2 条 supported claim、页级引用完整 |
| 扫描 PDF | PaddleOCR-VL-1.6 单页图片 PDF | 页级 Markdown 正确返回；无 OCR/未授权路径失败关闭 |
| Hybrid 工具自主选择（2.2） | “ample cash reserves” PDF + “liquidity resilience” 查询，动态目录同时提供 lexical/hybrid | 首个动作选择 `corpus.hybrid_search`，随后两次 lexical 交叉检查；1 条去重 Evidence，`succeeded/coverage_satisfied`，5 次模型调用，无 gap |

1.4 验证共发起 12 次 DeepSeek 请求，其中 3 次来自一次已经执行成功、但最终摘要脚本因字段名写错而未打印结果的重复验证。
2.0 重构后另发起 1 次短规划请求，验证真实模型选择工具而非仅消费工具结果。系统目前只统计模型调用次数，
不解析 provider token usage，因此不能给出可靠 token/金额数字；token/金额预算仍是生产路线图项目。

## 3. 真实测试发现并修复的问题

### 3.1 批量计算 JSON 被固定输出上限截断

初次十项计算中，所有函数和 Evidence 均正确，但模型响应在 3173 字符处停在 JSON 字符串中间，触发 `JSONDecodeError`。Harness/合成层正确拒绝半截 JSON并使用确定性报告，证明了回退有效；同时暴露固定 `max_tokens=1400` 对批量结果不足。

当时的最小修正是把证据合成输出上限提高到 3000。它是最大值而非最低消费，简单问题仍会提前结束。修正后相同十项计算得到 10/10 supported claim，无 fallback。2.1 又将输出上限配置化并默认设为 4096，同时把输入 evidence 上下文独立扩为规划 24K/生成 48K 字符；输出 token 不再被误写成输入上下文上限。

### 3.2 文档收入问题被错误追加 SEC requirement

“根据内部文档概括收入”已经明确限定来源，但查询分析器看到“收入”仍自动创建 `regulatory:entity`，导致文档证据充分时仍 degraded。修正后：

- `require_documents=true` 默认只建立 document requirement；
- 调用方显式 `require_regulatory_data=true` 时才同时建立 regulatory requirement；
- 普通非文档基本面问题仍自动走 SEC。

真实主备 RAG 复测从 degraded 变为 `succeeded/coverage_satisfied`。

### 3.3 同一 chunk 经 hybrid 与 lexical 重复召回导致合并崩溃

2.2 首次真实自主检索中，DeepSeek 先调用 hybrid，再调用 lexical 交叉验证。两个路径返回同一 file/page/chunk
和正文，因此生成相同 evidence ID；旧 adapter 却把搜索模式同时写进 Evidence tags，导致语义身份不同，
`EvidenceBundle.merge()` 抛出 `conflicting evidence id`，LangGraph planning step 失败。

最小修正是把 retrieval mode 只放在 Source provenance trace 与各次 ToolObservation 中，Evidence tags 固定为
`retrieved`。这样同一来源、定位和正文跨搜索模式幂等去重，两个调用的 rank/score/trace 仍可在 observation 审计；
真正同 ID 但内容、数值或业务 tags 冲突仍会快速失败。新增回归测试后，真实复测首个动作仍为 hybrid，最终只保留
1 条 Evidence 并成功结束。

本次成功复测记录 5 次模型调用。发现合并缺陷的前一次 run 在异常前也已经发生少量模型调用，但由于状态在合并点
中断，没有可靠最终计数；本文不把它伪装成零成本。

## 4. Harness 回退实际验证

当时 1.4/2.0 的规则 planner 不是宽泛 `try/except`，而是在模型不可用或 JSON 非法时的可见降级。现行契约已取消这条路径：
LLM 未配置、模型 JSON 非法或合成 quote 不可用时快速失败，不再产生 `model_planner_fallback` / `llm_synthesis_fallback`。

当时实测仍覆盖了下列 Harness 边界（与现行 fail-fast 不冲突的部分继续成立）：

1. 模型从本次动态工具目录选择工具、参数或 finish；
2. ToolSpec/Harness 拒绝目录外工具与非法参数；
3. 空结果产生可见 gap，不算完成 coverage；
4. 网络、capability、side effect 和预算拒绝发生在调用前，attempts 为 0；
5. 只有 read-only 工具允许按明确 exception 类型自动 retry；每个网络 attempt 都单独计数。

retry、预算耗尽、side-effect 拒绝、秘密脱敏等精确边界由自动化故障注入覆盖，因为让远端模型制造这些条件既不稳定也没有额外证明力。

## 5. Checkpoint 恢复语义

2.0 已删除自建 checkpoint，使用 LangGraph InMemorySaver/SqliteSaver 作为唯一恢复运行时。一个 planning node
最多执行一个工具，节点提交后持久化 observation、EvidenceBundle 和 audit；恢复要求 tenant/thread/run 哈希定位的
thread 与完整 `ResearchRequest` 一致。Harness 从 durable audit 恢复预算及 call sequence。

本次真实恢复序列：

```text
planning: finance.knowledge success
  → LangGraph checkpoint (observation/evidence/audit)
  → validation: coverage_satisfied, next=final_generation
  → intentional final-generation crash
  → new Agent + new Harness + same checkpointer
  → restore budgets/call sequence and pending node
  → final_generation
  → validation → completed
```

最终 audit 顺序为 `finance.knowledge`、`llm.synthesize`，没有重复执行第一个工具。

## 6. 仍不能声称的内容

- AlphaVantage、FRED 和 SEC 本轮没有真实 provider 凭据/合规 User-Agent，因此只验证了 adapter fixture、schema drift、错误与多源 LLM 消费，不能声称这三个线上账户已打通。
- Yahoo snapshot 当前真实返回 HTTPStatusError；历史 endpoint 可用，但 Yahoo 仍是无 SLA 的实验适配器。
- 当前没有 provider token/currency cost ledger、并发压测、长时间 soak、真正杀进程后的恢复演练或灾备演练。
- 该次评估时 Brave 搜索没有真实账户配置；只完成 provider fixture 与 MockTransport 边界验证。后续
  Bocha 线上搜索验收见 `IMPLEMENTATION_STATUS.md`，不追溯改写本次历史结果。
- literal quote 是最低引用门，不是完整语义蕴含证明；高风险结论仍需数值 claim parser、NLI 或人工复核。
