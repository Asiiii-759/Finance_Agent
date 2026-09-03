# 浏览器工作台

## 1. 当前定位

前端是随 Python 包发布的无构建静态工作台，目标是提供可用的本地单用户体验，并直接消费后端真实状态，不维护第二套 Agent
状态机。文件位于 `src/mas_finance/web/`，由 FastAPI `/` 和 `/static/*` 提供。

## 2. 已有功能

- 对话列表、新建和带确认的级联删除；
- 消息历史；
- PDF 多文件上传；
- 本轮联网授权和附件 session 保留开关；
- 后台 Job 提交、轮询、取消和 artifact 下载；
- Run Inspector：TaskFrame、Requirement Coverage、Claims、引用 Evidence、Gap、工具状态/attempt/error 和运行日志；
- 长期个人记忆新增/删除；
- 个人知识库上传/列表/删除；
- Learned Skill 查看/删除；
- 当前 Principal 和 API Key session 配置。

前端对所有 Provider 文本使用 HTML escaping；只有 HTTPS locator 生成外部链接。

## 3. 数据流

```text
submit question/PDF
→ POST job
→ poll job status
→ conversation messages
→ run detail + run logs
→ render claims / evidence / gaps / audit
```

Coverage 根据 `scope.requirements` 和 `coverage.missing` 计算，不依赖不存在的前端字段。任务目标来自 `task_frame.goal`。

## 4. 审批显示

当前工具目录只读，工作台明确显示“没有待审批外部写入或金融交易”。它不提供伪审批按钮。未来只有后端实现 checkpoint
interrupt、approval request 和精确 resume 后，前端才增加批准/拒绝交互。

## 5. 当前限制

- Job 状态使用 1.2 秒轮询，不是 SSE/WebSocket；
- 页面一次只跟踪一个主动提交的运行，其他任务可在后台任务面板查看；
- Assistant 报告仍以安全纯文本显示，结构化 Claim/Evidence 在 Inspector 展开；
- 没有 OIDC 登录、团队空间、共享知识库或管理员 UI；
- 没有移动端布局和无障碍完整审计；
- 没有复杂证据图、来源全文预览或人工标注反馈。

这些是产品层后续工作，不应通过在浏览器里复制 Planner 或 Coverage 逻辑来解决。
