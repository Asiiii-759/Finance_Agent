const state = {
  apiKey: sessionStorage.getItem("mas-api-key") || "",
  threadId: null,
  runningJob: null,
};

const elements = {
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome"),
  messageList: document.querySelector("#message-list"),
  threadList: document.querySelector("#thread-list"),
  threadTitle: document.querySelector("#thread-title"),
  deleteThread: document.querySelector("#delete-thread"),
  composer: document.querySelector("#composer"),
  queryInput: document.querySelector("#query-input"),
  pdfInput: document.querySelector("#pdf-input"),
  attachmentList: document.querySelector("#attachment-list"),
  networkToggle: document.querySelector("#network-toggle"),
  retainToggle: document.querySelector("#retain-toggle"),
  sendButton: document.querySelector("#send-button"),
  systemStatus: document.querySelector("#system-status"),
  statusDot: document.querySelector(".status-dot"),
  identityName: document.querySelector("#identity-name"),
  runEmpty: document.querySelector("#run-empty"),
  runPanel: document.querySelector("#run-panel"),
  settingsDialog: document.querySelector("#settings-dialog"),
  apiKeyInput: document.querySelector("#api-key-input"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(payload.detail || `请求失败（${response.status}）`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  window.setTimeout(() => elements.toast.classList.add("hidden"), 3200);
}

function setBusy(busy, label = "") {
  elements.sendButton.disabled = busy;
  elements.queryInput.disabled = busy;
  elements.sendButton.querySelector("span").textContent = busy ? "…" : "↑";
  if (label) elements.systemStatus.textContent = label;
}

async function connect() {
  try {
    const config = await api("/api/v1/config");
    elements.identityName.textContent =
      `${config.principal.tenant_id} / ${config.principal.user_id}`;
    elements.systemStatus.textContent = config.deepseek_enabled ? "Agent 已就绪" : "模型未配置";
    elements.statusDot.classList.add("online");
    await loadThreads();
  } catch (error) {
    elements.systemStatus.textContent = "连接失败";
    elements.statusDot.classList.remove("online");
    if (error.status === 401) elements.settingsDialog.showModal();
    else toast(error.message);
  }
}

async function loadThreads() {
  const payload = await api("/api/v1/conversations");
  elements.threadList.replaceChildren();
  for (const thread of payload.conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `thread-item${thread.thread_id === state.threadId ? " active" : ""}`;
    button.textContent = thread.title;
    button.title = thread.title;
    button.addEventListener("click", () => selectThread(thread.thread_id, thread.title));
    elements.threadList.append(button);
  }
}

async function selectThread(threadId, title) {
  state.threadId = threadId;
  elements.deleteThread.classList.remove("hidden");
  elements.threadTitle.textContent = title || "金融研究";
  await Promise.all([loadMessages(), loadRuns(), loadThreads()]);
}

async function loadMessages() {
  if (!state.threadId) return;
  const payload = await api(
    `/api/v1/conversations/${encodeURIComponent(state.threadId)}/messages?limit=500`,
  );
  elements.welcome.classList.add("hidden");
  elements.messageList.replaceChildren();
  for (const message of payload.messages) renderMessage(message.role, message.content);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function renderMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (role === "assistant") {
    const meta = document.createElement("span");
    meta.className = "message-meta";
    meta.textContent = "MAS FINANCE";
    article.append(meta);
  }
  article.append(document.createTextNode(content));
  elements.messageList.append(article);
}

async function loadRuns() {
  if (!state.threadId) return;
  const payload = await api(
    `/api/v1/conversations/${encodeURIComponent(state.threadId)}/runs?limit=100`,
  );
  if (!payload.runs.length) {
    elements.runEmpty.classList.remove("hidden");
    elements.runPanel.classList.add("hidden");
    elements.runPanel.replaceChildren();
    return;
  }
  const run = payload.runs[payload.runs.length - 1];
  const [detail, logs] = await Promise.all([
    api(
      `/api/v1/conversations/${encodeURIComponent(state.threadId)}/runs/${encodeURIComponent(run.run_id)}`,
    ),
    api(
      `/api/v1/conversations/${encodeURIComponent(state.threadId)}/runs/${encodeURIComponent(run.run_id)}/logs`,
    ),
  ]);
  showRunDetails(detail, logs.events);
}

function showRun(run) {
  elements.runEmpty.classList.add("hidden");
  elements.runPanel.classList.remove("hidden");
  const status = run.status || "running";
  elements.runPanel.innerHTML = `
    <div class="run-card">
      <span class="run-status">${escapeHtml(status)}</span>
      <h3>当前研究运行</h3>
      <p>${escapeHtml(run.stop_reason || "Agent 正在规划并收集证据。")}</p>
      <p><strong>Run</strong><br>${escapeHtml(run.run_id || "pending")}</p>
    </div>
  `;
}

function showRunDetails(run, logs) {
  const result = run.result || {};
  const coverage = result.coverage || {};
  const gaps = result.gaps || [];
  const scope = result.scope || {};
  const taskFrame = result.task_frame || {};
  const requirements = scope.requirements || [];
  const bundle = result.bundle || {};
  const evidence = bundle.evidence || [];
  const claims = bundle.claims || [];
  const audit = result.audit_events || [];
  const missing = coverage.missing || [];
  elements.runEmpty.classList.add("hidden");
  elements.runPanel.classList.remove("hidden");
  elements.runPanel.innerHTML = `
    <div class="run-card">
      <span class="run-status">${escapeHtml(run.status)}</span>
      <h3>研究结论</h3>
      <p>${escapeHtml(run.stop_reason || "已生成研究终态")}</p>
      <div class="run-metric"><span>Run</span><strong>${escapeHtml(run.run_id)}</strong></div>
      <div class="run-metric"><span>证据覆盖</span><strong>${requirements.length - missing.length} / ${requirements.length}</strong></div>
      <div class="run-metric"><span>未解决缺口</span><strong>${gaps.filter((gap) => !gap.resolved).length}</strong></div>
    </div>
    <div class="run-card">
      <h3>任务理解</h3>
      <p>${escapeHtml(taskFrame.goal || "本轮未返回结构化目标。")}</p>
      ${requirements.length ? `<ul>${requirements.map((item) => `<li>${escapeHtml(item.reason)} <small>${escapeHtml(item.key)}</small></li>`).join("")}</ul>` : "<p>本轮没有强制检索需求。</p>"}
    </div>
    <div class="run-card">
      <h3>Claims 与引用</h3>
      ${renderClaims(claims)}
    </div>
    <div class="run-card">
      <h3>证据</h3>
      ${renderEvidence(evidence)}
    </div>
    <div class="run-card">
      <h3>数据缺口</h3>
      ${renderGaps(gaps)}
    </div>
    <div class="run-card">
      <h3>工具调用</h3>
      ${renderAudit(audit)}
    </div>
    <div class="run-card">
      <h3>运行事件</h3>
      ${logs.length ? `<ul>${logs.slice(-30).map((event) => `<li><small>${escapeHtml(event.event_type)} · ${escapeHtml(event.level)}</small><br>${escapeHtml(event.message)}</li>`).join("")}</ul>` : "<p>暂无运行事件。</p>"}
    </div>
    <div class="run-card">
      <h3>操作审批</h3>
      <p>当前 Agent 目录只允许只读研究工具，因此没有待审批的外部写入或金融交易。未来接入副作用工具时，必须由后端 checkpoint 中断并由用户明确批准后恢复。</p>
    </div>
  `;
}

function renderClaims(claims) {
  if (!claims.length) return "<p>当前没有可发布的 claim。</p>";
  return claims.map((claim) => {
    const citations = (claim.evidence_ids || []).map(
      (id) => `<a href="#evidence-${encodeURIComponent(id)}">[${escapeHtml(id.slice(0, 10))}]</a>`,
    ).join(" ");
    return `<div class="run-record"><small>${escapeHtml(claim.status)}</small><p>${escapeHtml(claim.text)} ${citations}</p>${claim.caveat ? `<small>${escapeHtml(claim.caveat)}</small>` : ""}</div>`;
  }).join("");
}

function renderEvidence(evidence) {
  if (!evidence.length) return "<p>没有召回可引用证据。</p>";
  return evidence.map((item) => {
    const source = item.source || {};
    const locator = String(source.locator || "");
    const locatorHtml = /^https:\/\//i.test(locator)
      ? `<a href="${escapeHtml(locator)}" target="_blank" rel="noopener noreferrer">打开来源</a>`
      : `<small>${escapeHtml(locator)}</small>`;
    return `<div class="run-record" id="evidence-${encodeURIComponent(item.evidence_id)}"><small>${escapeHtml(source.source_type || "source")} · ${escapeHtml(source.provider || "unknown")}</small><p><strong>${escapeHtml(source.title || "未命名来源")}</strong><br>${escapeHtml(String(item.content || "").slice(0, 600))}</p>${locatorHtml}</div>`;
  }).join("");
}

function renderGaps(gaps) {
  if (!gaps.length) return "<p>没有记录到数据缺口。</p>";
  return `<ul>${gaps.map((gap) => `<li><small>${gap.resolved ? "已解决" : "未解决"} · ${escapeHtml(gap.code)}</small><br>${escapeHtml(gap.message)}</li>`).join("")}</ul>`;
}

function renderAudit(events) {
  if (!events.length) return "<p>没有工具审计记录。</p>";
  return `<ul>${events.slice(-30).map((event) => `<li><small>${escapeHtml(event.result_status)} · attempts ${escapeHtml(event.attempts ?? 0)}</small><br>${escapeHtml(event.tool_name)}${event.error_code ? ` · ${escapeHtml(event.error_code)}` : ""}</li>`).join("")}</ul>`;
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value);
  return node.innerHTML;
}

async function pollJob(jobId) {
  const deadline = Date.now() + 30 * 60 * 1000;
  for (;;) {
    const job = await api(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    showRun({
      status: job.status,
      run_id: jobId,
      stop_reason:
        job.status === "running" || job.status === "pending"
          ? "Agent 正在规划、调用工具并核验证据。"
          : job.error_message,
    });
    if (["completed", "failed", "cancelled"].includes(job.status)) return job;
    if (Date.now() >= deadline) {
      throw new Error("任务仍在后台运行，请稍后从“后台任务”查看进度。");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
  }
}

async function submitResearch(event) {
  event.preventDefault();
  const query = elements.queryInput.value.trim();
  if (!query || state.runningJob) return;
  if (!state.threadId) state.threadId = `thread-${crypto.randomUUID()}`;

  elements.welcome.classList.add("hidden");
  renderMessage("user", query);
  elements.queryInput.value = "";
  setBusy(true, "研究运行中");

  try {
    let submitted;
    if (elements.pdfInput.files.length) {
      const form = new FormData();
      form.set("query", query);
      form.set("thread_id", state.threadId);
      form.set("export_artifacts", "true");
      form.set("allow_network", String(elements.networkToggle.checked));
      form.set("retain_for_session", String(elements.retainToggle.checked));
      form.set("idempotency_key", crypto.randomUUID());
      for (const file of elements.pdfInput.files) form.append("files", file);
      submitted = await api("/api/v1/jobs/upload", {method: "POST", body: form});
    } else {
      submitted = await api("/api/v1/jobs", {
        method: "POST",
        body: JSON.stringify({
          query,
          thread_id: state.threadId,
          export_artifacts: true,
          allow_network: elements.networkToggle.checked,
          use_session_documents: true,
          use_personal_memory: true,
          use_personal_knowledge: true,
          idempotency_key: crypto.randomUUID(),
        }),
      });
    }
    state.runningJob = submitted.job_id;
    showRun({status: submitted.status, run_id: submitted.job_id});
    const completed = await pollJob(submitted.job_id);
    if (completed.status === "completed") {
      await Promise.all([loadMessages(), loadRuns(), loadThreads()]);
      elements.systemStatus.textContent = "研究完成";
    } else {
      throw new Error(completed.error_message || "研究任务未完成");
    }
  } catch (error) {
    toast(error.message);
    elements.systemStatus.textContent = "运行失败";
  } finally {
    state.runningJob = null;
    elements.pdfInput.value = "";
    renderAttachments();
    setBusy(false);
  }
}

function newThread() {
  state.threadId = null;
  elements.deleteThread.classList.add("hidden");
  elements.threadTitle.textContent = "新的金融研究";
  elements.welcome.classList.remove("hidden");
  elements.messageList.replaceChildren();
  elements.runEmpty.classList.remove("hidden");
  elements.runPanel.classList.add("hidden");
  loadThreads().catch((error) => toast(error.message));
  elements.queryInput.focus();
}

async function deleteCurrentThread() {
  if (!state.threadId) return;
  if (!window.confirm("删除该对话、消息、运行日志和对应 checkpoint？此操作不可撤销。")) return;
  await api(`/api/v1/conversations/${encodeURIComponent(state.threadId)}`, {method: "DELETE"});
  newThread();
  await loadThreads();
  toast("对话已删除");
}

function renderAttachments() {
  elements.attachmentList.replaceChildren();
  for (const file of elements.pdfInput.files) {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = file.name;
    elements.attachmentList.append(chip);
  }
}

const managerDialog = document.querySelector("#manager-dialog");
const managerContent = document.querySelector("#manager-content");
const managerTitle = document.querySelector("#manager-title");

async function openManager(section) {
  if (!managerDialog.open) managerDialog.showModal();
  document.querySelectorAll(".manager-tabs [data-manager]").forEach((button) => {
    button.classList.toggle("active", button.dataset.manager === section);
  });
  managerContent.innerHTML = '<div class="empty-records">正在读取…</div>';
  try {
    if (section === "memories") await renderMemories();
    if (section === "knowledge") await renderKnowledge();
    if (section === "skills") await renderSkills();
    if (section === "jobs") await renderJobs();
  } catch (error) {
    managerContent.innerHTML = `<div class="empty-records">${escapeHtml(error.message)}</div>`;
  }
}

function recordsOrEmpty(records, render) {
  if (!records.length) return '<div class="empty-records">这里还没有内容。</div>';
  return `<div class="record-list">${records.map(render).join("")}</div>`;
}

async function renderMemories() {
  managerTitle.textContent = "长期个人记忆";
  const memories = await api("/api/v1/memories");
  managerContent.innerHTML = `
    <div class="manager-intro">
      <div><h3>稳定的个人背景与偏好</h3><p>这些内容会作为低权限 personal context 进入每一次研究，不作为金融证据。</p></div>
      <button class="manager-action" id="add-memory" type="button">新增记忆</button>
    </div>
    <form class="inline-form hidden" id="memory-form">
      <div class="form-row">
        <select name="kind"><option value="preference">偏好</option><option value="profile">个人背景</option><option value="experience">经验</option></select>
        <input name="title" required maxlength="200" placeholder="标题，例如：回答结构" />
      </div>
      <textarea name="content" required maxlength="8000" rows="3" placeholder="只写长期稳定、未来仍然适用的信息"></textarea>
      <footer><button class="manager-action" type="submit">保存</button></footer>
    </form>
    ${recordsOrEmpty(memories, (item) => `
      <article class="record"><div><h4>${escapeHtml(item.title)}</h4><p>${escapeHtml(item.content)}</p><small>${escapeHtml(item.kind)} · ${escapeHtml(item.updated_at)}</small></div>
      <button class="record-delete" data-delete-memory="${escapeHtml(item.memory_id)}" type="button">删除</button></article>`)}
  `;
  document.querySelector("#add-memory").addEventListener("click", () => {
    document.querySelector("#memory-form").classList.toggle("hidden");
  });
  document.querySelector("#memory-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    await api("/api/v1/memories", {
      method: "POST",
      body: JSON.stringify({
        kind: values.get("kind"),
        title: values.get("title"),
        content: values.get("content"),
        tags: [],
      }),
    });
    await renderMemories();
  });
  document.querySelectorAll("[data-delete-memory]").forEach((button) => {
    button.addEventListener("click", async () => {
      await api(`/api/v1/memories/${encodeURIComponent(button.dataset.deleteMemory)}`, {method: "DELETE"});
      await renderMemories();
    });
  });
}

async function renderKnowledge() {
  managerTitle.textContent = "个人知识库";
  const payload = await api("/api/v1/knowledge/documents");
  managerContent.innerHTML = `
    <div class="manager-intro">
      <div><h3>明确持久化的个人文档</h3><p>文档按当前用户隔离，保存页文本、ACL、索引 manifest 和向量；临时附件不会自动进入这里。</p></div>
    </div>
    <form class="inline-form" id="knowledge-form">
      <input name="files" type="file" accept="application/pdf,.pdf" multiple required />
      <label><input name="allow_network" type="checkbox" value="true" /> 允许将文档发送到已配置的 OCR 服务</label>
      <footer><button class="manager-action" type="submit">解析并保存</button></footer>
    </form>
    ${recordsOrEmpty(payload.documents, (item) => `
      <article class="record"><div><h4>${escapeHtml(item.filename)}</h4><p>${escapeHtml(item.page_count)} 页 · ${escapeHtml(item.index_status)}</p><small>${escapeHtml(item.created_at)}</small></div>
      <button class="record-delete" data-delete-document="${escapeHtml(item.document_id)}" type="button">删除</button></article>`)}
  `;
  document.querySelector("#knowledge-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    await api("/api/v1/knowledge/documents", {method: "POST", body: form});
    await renderKnowledge();
  });
  document.querySelectorAll("[data-delete-document]").forEach((button) => {
    button.addEventListener("click", async () => {
      await api(`/api/v1/knowledge/documents/${encodeURIComponent(button.dataset.deleteDocument)}`, {
        method: "DELETE",
      });
      await renderKnowledge();
    });
  });
}

async function renderSkills() {
  managerTitle.textContent = "Learned Skills";
  const payload = await api("/api/v1/skills");
  managerContent.innerHTML = `
    <div class="manager-intro">
      <div><h3>成功研究路径</h3><p>任务理解阶段只看到短索引；模型选择后，Planner 才会看到完整步骤。</p></div>
    </div>
    ${recordsOrEmpty(payload.skills, (item) => `
      <article class="record"><div><h4>${escapeHtml(item.name)}</h4><p>${escapeHtml(item.description)}</p><small>${escapeHtml(item.applicability)}</small></div>
      <button class="record-delete" data-delete-skill="${escapeHtml(item.skill_id)}" type="button">删除</button></article>`)}
  `;
  document.querySelectorAll("[data-delete-skill]").forEach((button) => {
    button.addEventListener("click", async () => {
      await api(`/api/v1/skills/${encodeURIComponent(button.dataset.deleteSkill)}`, {method: "DELETE"});
      await renderSkills();
    });
  });
}

async function renderJobs() {
  managerTitle.textContent = "后台任务";
  const jobs = await api("/api/v1/jobs?limit=100");
  managerContent.innerHTML = `
    <div class="manager-intro">
      <div><h3>可靠队列</h3><p>这里只显示当前 principal 的任务。任务通过 lease、幂等键和重试状态持久化。</p></div>
    </div>
    ${recordsOrEmpty(jobs, (item) => `
      <article class="record"><div><h4>${escapeHtml(item.query)}</h4><p>${escapeHtml(item.status)} · ${escapeHtml(item.thread_id)}</p><small>${escapeHtml(item.created_at)}</small></div>
      <div>${Object.keys(item.artifacts || {}).map((name) => `<button class="manager-action" data-download-job="${escapeHtml(item.job_id)}" data-download-artifact="${escapeHtml(name)}" type="button">下载 ${escapeHtml(name.replace("_path", ""))}</button>`).join("")}
      ${["pending", "running", "cancel_requested"].includes(item.status) ? `<button class="record-delete" data-cancel-job="${escapeHtml(item.job_id)}" type="button">取消</button>` : ""}</div></article>`)}
  `;
  document.querySelectorAll("[data-download-artifact]").forEach((button) => {
    button.addEventListener("click", async () => {
      await downloadArtifact(button.dataset.downloadJob, button.dataset.downloadArtifact);
    });
  });
  document.querySelectorAll("[data-cancel-job]").forEach((button) => {
    button.addEventListener("click", async () => {
      await api(`/api/v1/jobs/${encodeURIComponent(button.dataset.cancelJob)}`, {method: "DELETE"});
      await renderJobs();
    });
  });
}

async function downloadArtifact(jobId, artifactName) {
  const headers = new Headers();
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);
  const response = await fetch(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactName)}`,
    {headers},
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `下载失败（${response.status}）`);
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="([^"]+)"/);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = match ? match[1] : artifactName;
  link.click();
  URL.revokeObjectURL(url);
}

elements.composer.addEventListener("submit", submitResearch);
elements.pdfInput.addEventListener("change", renderAttachments);
elements.queryInput.addEventListener("input", () => {
  elements.queryInput.style.height = "auto";
  elements.queryInput.style.height = `${elements.queryInput.scrollHeight}px`;
});
elements.queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
document.querySelector("#new-thread").addEventListener("click", newThread);
elements.deleteThread.addEventListener("click", () => {
  deleteCurrentThread().catch((error) => toast(error.message));
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.queryInput.value = button.dataset.prompt;
    elements.queryInput.focus();
  });
});
document.querySelector("#open-settings").addEventListener("click", () => {
  elements.apiKeyInput.value = state.apiKey;
  elements.settingsDialog.showModal();
});
document.querySelectorAll("[data-manager]").forEach((button) => {
  button.addEventListener("click", () => openManager(button.dataset.manager));
});
document.querySelector("#close-manager").addEventListener("click", () => managerDialog.close());
document.querySelector("#save-settings").addEventListener("click", () => {
  state.apiKey = elements.apiKeyInput.value.trim();
  if (state.apiKey) sessionStorage.setItem("mas-api-key", state.apiKey);
  else sessionStorage.removeItem("mas-api-key");
  elements.settingsDialog.close();
  connect();
});

connect();
