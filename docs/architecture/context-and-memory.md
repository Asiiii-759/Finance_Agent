# 上下文与记忆

## 1. 四种不同状态

| 状态 | 用途 | 生命周期 |
|---|---|---|
| LangGraph checkpoint | 恢复同一个未完成 run | run/thread |
| 对话事件账本 | 保存用户、工具和助手事件 | 用户删除线程前 |
| 个人长期记忆 | 稳定偏好、背景和经验 | 用户查看、更新或删除 |
| Learned Skill | 可复用成功工作路径 | 独立版本、可删除 |

它们不能互相代替。Checkpoint 不是长期记忆；Skill 不是用户画像；运行日志也不是对话语义。

## 2. 完整持久化与 Prompt 投影

SQLite 对话账本保存完整事件和全部原子事实。构建 Prompt 时才生成有界投影：

```text
语义摘要
+ 最近完整 runs
+ 按时间顺序排列的最新原子事实尾部
+ 最近 run 状态
```

达到会话预算 85% 且存在可压缩旧 run 时，`LLMConversationSummarizer` 把旧摘要和旧事件合成新结构化摘要；原始事件不删除。
摘要保留目标、要求、纠正、完成工作、成功/失败工具和未完成事项，但不能把工具数据改写成金融事实。

## 3. 原子事实

每个结束 run 在写入助手结果后触发提取，最多六条最小中文事实，也可以不产生事实。提取模型实际只接收本轮用户消息，记录：

- 用户明确提出的问题或需求；
- 用户明确给出的约束；
- 用户对先前表达的纠正。

工具事件和助手正文不进入原子事实提取 Prompt，因此不能把模型内部取得但用户未表达的信息沉淀为对话事实。提取器只额外接收
已经解析的实体候选，用于把“它、这个、刚才那个”等代称改写成上下文支持的实体全名，不得增加新事实。后端保留原始用户
event ID、时间和序号用于审计。数据库在用户删除线程前保留全部事实；Prompt 按时间顺序注入能放入独立 32K token 预算的最新事实尾部。
超限时最早事实只从 Prompt 投影移出，不从账本删除。Manifest 记录总数、注入数、省略数和预算。

系统不对原子事实做关键词或 embedding 相关性筛选。TaskFrame 模型直接结合这条最小事实时间线、滚动摘要、近期事件和当前问题
完成指代判断；信息不足或多个实体都可能成立时必须追问。固定窗口无法永久承载无限事实，这是显式边界，不以隐式猜测掩盖。

## 4. 统一 token 预算

默认模型输入上限 300K。系统不再把 300K 同时分别许诺给每一层：

- synthesis Evidence 上限 96K；
- planning/adequacy Evidence 上限 48K；
- control reserve 默认 24K；
- 原子事实时间线默认上限 32K；
- conversation projection 使用总预算扣除最大 Evidence 和 control reserve 后的余额；
- TaskFrame、Planner、Validation、Synthesis 各自重新计算实际总输入；
- Evidence 优先于低权限 personal context；放不下的个人条目按当前问题相关性省略。

每个 `ContextManifest` 包含 thread、personal、evidence、reserved、total/max token 计数和省略数量。若强制层本身超限，系统快速
失败，而不是静默截断事实段落或突破模型窗口。

## 5. 个人长期记忆

个人记忆只保存未来仍然适用的信息。用户明确长期改变时，同 kind+title 稳定槽位可覆盖旧值；“本轮、今天、这一次”等临时
要求不能晋升为长期偏好。自动候选必须来自完成 run，推断偏好需跨 run 观察，用户可通过 API/前端查看和删除。

个人记忆是低权限呈现/理解上下文，不是金融证据，不能覆盖当前请求、系统规则、Tool Schema 或 Evidence。

用户还可维护单独 Markdown profile；它同样是低权限个人上下文。

## 6. Skill 渐进披露

TaskFrame 只看 Skill ID、名称、描述和适用范围的短索引，最多选择三个。只有被选中的 Skill 完整 steps 才进入 Planner。
Skill 提取要求 run 成功且存在多步骤工具路径；禁止保存实体专属、URL、日期、数值和未观察 capability。

## 7. 删除与隔离

所有 namespace 都绑定 tenant/user；对话还绑定 thread。删除线程会级联删除：

- 对话消息和事件；
- 原子事实和摘要；
- run 日志和运行记录；
- session 文档；
- 对应 LangGraph checkpoint。

个人知识库、个人长期记忆和 Skill 有独立删除接口，不随单个线程删除。

## 8. 可观测性

后端 run log 单独保存 context.loaded、tool.completed、memory extraction/compaction 和 terminal 事件。Audit 只记录参数摘要、
结果结构、错误和预算，不复制 PDF、网页、完整 Prompt 或密钥。上下文 Manifest 随 ResearchState 进入 run detail，前端可显示
省略和预算情况。
