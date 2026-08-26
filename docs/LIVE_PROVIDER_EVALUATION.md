# Bocha 搜索与模型配置实测

测试日期：2026-08-26

## 结论

- Bocha Web Search 已接入 provider-neutral `web.search`，并完成原始 API、
  `WebSearchEvidenceAdapter → EvidenceBundle` 和完整 Agent 自主规划链的真实调用。
- 当前活动模型为 `deepseek-v4-pro`，完成一次最多 80 输出 token 的真实调用，返回结构化正文。
- Qwen 与 DeepSeek Flash 密钥只作为本地 `.env` 中的命名备选保存；当前运行时不会自动切换，也不会在
  Pro 失败时静默降级。

密钥只存在于 Git 忽略的 `.env`，示例、日志和本文均不保存明文。Bocha 参考资料：
[用户提供的 API 文档](https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK)、
[Bocha 开放平台](https://open.bochaai.com/overview)。Feishu 文档在本次环境要求登录，因此实现以一次有界
真实响应确认字段，再以 fixture 固化契约。

## Bocha 契约

固定认证边界：

```text
POST https://api.bochaai.com/v1/web-search
Authorization: Bearer <deployment secret>
Content-Type: application/json
```

Agent 仍只暴露统一参数 `query/count/freshness/domains`。adapter 将时效窗口映射为：

| Agent | Bocha |
|---|---|
| `pd` | `oneDay` |
| `pw` | `oneWeek` |
| `pm` | `oneMonth` |
| `py` | `oneYear` |

响应仅消费 `code == 200` 下的 `data.webPages.value[]`，将 `name/url/snippet/datePublished` 转为
canonical source/evidence。仍执行公开 URL 校验、tracking 参数移除、URL/内容去重和响应大小限制。
Bocha 与 Brave 同时配置时优先 Bocha；这是一条可见的部署选择，不是失败后的 fallback。

## 真实结果

第一次原始请求返回 HTTP 200 和两个网页结果。基础 adapter 测试得到两个 `provider=bocha` 的 source
和两条 evidence，gap 为空。

完整 Agent 测试的 checkpoint 显示 `llm.plan → web.search → llm.plan → llm.synthesize` 全部成功，
最终 claim 带 evidence ID，并因只使用搜索摘要而正确标为 `inferred`。测试同时发现 Bocha 不保证遵守
查询中的 `site:`：即使模型传入 `domains=["pbc.gov.cn"]`，provider 仍可能返回域外页面。修复后采用两层
约束：将 `site:` 前置以改善召回，再在 adapter 边界强制丢弃非目标主域/子域结果。最终真实请求保留了
10 个结果，全部来自 `www.pbc.gov.cn` 或人民银行子域，gap 为空。

整个实现与排错过程共消耗 7 次 Bocha 搜索请求：原始契约 1 次、基础 adapter 1 次、两轮 Agent 验收
共 3 次、域名过滤定位与复验 2 次。

DeepSeek 首次测试使用了用户书写的 `DeepSeek-v4-pro`，API 明确返回 400，并声明模型 ID 区分大小写、
应为 `deepseek-v4-pro`。配置修正后，同一项目客户端返回：

```json
{"ok": true, "model": "deepseek-v4-pro"}
```

这次结果验证了认证、模型 ID、`thinking` 关闭参数、JSON 解析及非空正文边界。连同两轮 Agent 验收，
总计发起 9 个 DeepSeek HTTP 请求，其中大小写错误的 1 个返回 400，其余 8 个成功。它不是质量、并发、
费用或长期稳定性评测。Qwen 和 Flash 本次未产生真实调用。

## 保留边界

- 搜索结果是 snippet 级发现证据，重要金融结论仍应优先 SEC、FRED、央行、交易所、公司公告或授权 RAG。
- 没有开放模型自拟 URL 的通用 fetch，搜索 endpoint 仍由部署固定。
- 调用同时受服务端 `MAS_ALLOW_NETWORK`、请求 `allow_network=true` 和 Harness 网络预算控制。
- 当前只选择一个活动 LLM；自动多模型路由会引入成本、语义差异和错误掩盖，未在没有明确策略前加入。
