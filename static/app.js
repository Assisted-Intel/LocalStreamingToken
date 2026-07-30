/* Local Streaming Token — by Assisted Intel.
   Browser front-end. Talks to the local Flask server; streams tokens via SSE.
   The browser only ever exchanges paths (never file contents) for batch/library
   filesystem operations — the server reads/writes disk directly. */

"use strict";

// ------------------------------- state -------------------------------
const S = {
  app: {}, config: {}, servers: [], presets: [], libraries: [], contextLengths: [],
  chats: [],           // sidebar summaries
  groups: [],          // sidebar tabs: [{id, name, builtin?}] (default tab first)
  activeGroup: "default", // currently selected sidebar tab
  chat: null,          // current full chat dict (client-owned)
  models: [],
  toolsSupported: null,
  runId: null,
  generating: false,
  batchRunning: false,
  activeLibrary: null, // Resources tab current library
  queue: [],           // pending queued prompts: {item_id, chat, search_query, title}
  queueStop: false,    // cancel flag for the sequential queue loop
  dataItems: [],       // data-classification blocks staged for the next send: {id, label, text}
  dataMode: false,     // data-classification composer mode on/off
  parallel: { enabled: false, mode: "balanced", servers: [] }, // multi-server config
  contextBars: {},      // live context-usage bars, keyed by chat/lane: {used, window, ...}
  personas: [],         // [{id, name, role, variants[]}]
  usePersona: false,    // "Use Persona" toggle
  personaId: "",        // active persona id
  personaRunId: null,   // current pipeline run id (for re-run-from-step)
  evals: [],            // eval-project summaries
  evalProject: null,    // current full eval-project dict (client-owned)
  evalModelCache: {},   // serverUrl -> [models] (for the eval tab dropdowns)
  evalRunId: null,
  evalRunning: false,
  evalRun: null,        // live results being assembled during a run
  evalInited: false,
  // Database tab.
  db: {
    inited: false,
    vault: { exists: false, unlocked: false },
    profiles: [],       // masked connection profiles
    sessions: [],       // import-session summaries
    grid: { sessionId: null, offset: 0, limit: 50, total: 0, columns: [] },
  },
};

const DEFAULT_GROUP_ID = "default";
const $ = (id) => document.getElementById(id);
const uid = () => Math.random().toString(36).slice(2, 14);
const escapeHtml = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

// ------------------------------- api ---------------------------------
/**
 * JSON fetch wrapper for every non-streaming endpoint.
 * Serializes `opts.body`, and turns a non-2xx response into a thrown Error carrying
 * the server's `error` field, so callers can just try/catch instead of checking status.
 * @param {string} path  API path, e.g. "/api/chats"
 * @param {object} opts  fetch options; `body` is a plain object, not a string
 * @returns {Promise<object>} the parsed JSON response
 */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

/**
 * Consume a Server-Sent Events stream from a POST endpoint.
 *
 * Uses fetch + a stream reader rather than EventSource, because EventSource can only
 * issue GETs and every streaming endpoint here needs a JSON body. Frames are split on
 * the blank-line delimiter and dispatched by event name; a partial frame stays in the
 * buffer until the rest of it arrives. Unparseable frames are skipped rather than
 * aborting the stream.
 *
 * @param {string} path
 * @param {object} body     request payload
 * @param {object} handlers map of event name to callback, e.g.
 *                          { start, chunk, status, file, progress, done, error }
 */
async function streamSSE(path, body, handlers) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch (e) {}
    (handlers.error || (() => {}))({ message: msg });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let parsed; try { parsed = JSON.parse(data); } catch (e) { continue; }
      if (handlers[event]) handlers[event](parsed);
    }
  }
}

// ------------------------------- toasts ------------------------------
function toast(msg, ms = 3500) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  $("toast-container").appendChild(t);
  setTimeout(() => t.remove(), ms);
}
function setStatus(msg) { $("status-line").textContent = msg || ""; }

// -------------------------- context-usage meter ----------------------
// Thin bottom bars showing prompt_tokens / context-window per active chat/server,
// fed live by the backend "context" SSE frames. One entry per chat (single send)
// or per lane (parallel). Green <50%, yellow 50-80%, red >80%.
function contextPct(used, window) {
  if (!window) return 0;
  return Math.max(0, Math.min(100, (used / window) * 100));
}
function contextClass(pct) {
  return pct > 80 ? "ctx-red" : (pct >= 50 ? "ctx-yellow" : "ctx-green");
}
function contextBarKey(d) {
  return d.lane != null ? ("lane:" + d.lane) : ("chat:" + (d.chat_id || "current"));
}
/**
 * Create or update one context-usage bar from a `context` SSE frame. Keyed per chat
 * or per parallel lane, so several bars can be live at once.
 */
function upsertContextBar(d) {
  if (!d) return;
  S.contextBars[contextBarKey(d)] = {
    name: d.server_name || d.server || "",
    used: d.prompt_tokens || 0,
    window: d.window || 0,
    completion: d.completion_tokens || 0,
    breakdown: d.breakdown || null,
    batchLabel: d.batch_item_label || "",
    exact: !!d.exact,
    isolation: !!d.isolation,
    phase: d.phase || "start",
  };
  renderContextBars();
}
function clearContextBars() { S.contextBars = {}; renderContextBars(); }
function clearLaneContextBars() {
  Object.keys(S.contextBars).forEach((k) => { if (k.startsWith("lane:")) delete S.contextBars[k]; });
  renderContextBars();
}
function renderContextBars() {
  const host = $("context-bars");
  if (!host) return;
  const keys = Object.keys(S.contextBars);
  host.innerHTML = keys.map((k) => {
    const b = S.contextBars[k];
    const pct = contextPct(b.used, b.window);
    const cls = contextClass(pct);
    const live = b.phase === "start" ? " ctx-live" : "";
    const nameHtml = b.name ? `<span class="ctx-name">${escapeHtml(b.name)}</span>` : "";
    const iso = b.isolation ? `<span class="ctx-iso" title="Isolation on — prior history excluded">⛶</span>` : "";
    const batch = b.batchLabel
      ? `<span class="ctx-batch" title="Current batch item">📄 ${escapeHtml(b.batchLabel)}</span>` : "";
    const warn = pct >= 90 ? ` <span class="ctx-alert" title="Context nearly full">⚠</span>` : "";
    const num = `${b.used.toLocaleString()} / ${b.window.toLocaleString()} · ${pct.toFixed(0)}%`;
    return `<div class="ctx-bar ${cls}${live}">
      <div class="ctx-track"><div class="ctx-fill" style="width:${pct}%"></div></div>
      <div class="ctx-meta">${nameHtml}${iso}<span class="ctx-num">${num}</span>${warn}${batch}</div>
    </div>`;
  }).join("");
  const area = $("context-bar-area");
  if (area) area.classList.toggle("has-bars", keys.length > 0);
}

// -------------------------- context history modal --------------------
async function openContextHistory() {
  if (!S.chat || !S.chat.id) { toast("Open a chat first."); return; }
  const list = $("context-history-list");
  list.innerHTML = "<div class='ch-empty'>Loading…</div>";
  openModal("modal-context-history");
  try {
    const r = await api(`/api/chats/${S.chat.id}/context-history`);
    renderContextHistory(r.history || []);
  } catch (e) {
    list.innerHTML = `<div class='ch-empty'>Could not load history: ${escapeHtml(e.message)}</div>`;
  }
}
function renderContextHistory(history) {
  const list = $("context-history-list");
  if (!history.length) {
    list.innerHTML = "<div class='ch-empty'>No LLM calls recorded for this chat yet.</div>";
    return;
  }
  // Newest first.
  const rows = history.slice().reverse().map((h) => {
    const when = (h.timestamp || "").replace("T", " ").slice(0, 19);
    const pct = h.num_ctx_at_time ? Math.round((h.total_prompt_tokens / h.num_ctx_at_time) * 100) : 0;
    const cls = contextClass(pct);
    const iso = h.isolation ? " <span class='ctx-iso' title='Isolation on'>⛶</span>" : "";
    const batch = h.batch_item_label ? ` · 📄 ${escapeHtml(h.batch_item_label)}` : "";
    const note = h.notes ? ` · ${escapeHtml(h.notes)}` : "";
    return `<div class="ch-row">
      <div class="ch-row-head">
        <span class="ch-when">${escapeHtml(when)}</span>${iso}
        <span class="ch-server">${escapeHtml(h.server_name || h.server_id || "")}</span>${batch}${note}
        <span class="ch-total ${cls}">${(h.total_prompt_tokens || 0).toLocaleString()} / ${(h.num_ctx_at_time || 0).toLocaleString()} · ${pct}%</span>
      </div>
      <div class="ch-breakdown">
        <span class="ch-seg ch-sys">System ${(h.system_tokens || 0).toLocaleString()}</span>
        <span class="ch-seg ch-rag">RAG ${(h.rag_tokens || 0).toLocaleString()}</span>
        <span class="ch-seg ch-user">User ${(h.user_tokens || 0).toLocaleString()}</span>
        <span class="ch-seg ch-asst">Assistant ${(h.assistant_tokens || 0).toLocaleString()}</span>
      </div>
    </div>`;
  }).join("");
  list.innerHTML = rows;
}
async function clearContextHistory() {
  if (!S.chat || !S.chat.id) return;
  if (!confirm("Clear the recorded context-usage history for this chat?")) return;
  try {
    await api(`/api/chats/${S.chat.id}/context-history`, { method: "DELETE" });
    renderContextHistory([]);
    toast("Context history cleared");
  } catch (e) { toast("Could not clear history: " + e.message); }
}

// ------------------------------- modals ------------------------------
function openModal(id) {
  $("modal-backdrop").classList.remove("hidden");
  document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
  $(id).classList.remove("hidden");
}
function closeModal() {
  $("modal-backdrop").classList.add("hidden");
  document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
}
function promptModal(title, def = "") {
  return new Promise((resolve) => {
    $("prompt-title").textContent = title;
    $("prompt-input").value = def;
    openModal("modal-prompt");
    $("prompt-input").focus();
    const cancelBtn = $("modal-prompt").querySelector(".modal-close");
    const ok = () => { cleanup(); resolve($("prompt-input").value); };
    const cancel = () => { cleanup(); resolve(null); };
    function key(e) { if (e.key === "Enter") ok(); if (e.key === "Escape") cancel(); }
    function cleanup() {
      closeModal();
      $("btn-prompt-ok").removeEventListener("click", ok);
      if (cancelBtn) cancelBtn.removeEventListener("click", cancel);
      $("prompt-input").removeEventListener("keydown", key);
    }
    $("btn-prompt-ok").addEventListener("click", ok);
    if (cancelBtn) cancelBtn.addEventListener("click", cancel);
    $("prompt-input").addEventListener("keydown", key);
  });
}

// ------------------------------- profiles ----------------------------
// Two independent axes: data profiles (chats/prompts/resources/evals/database)
// and settings profiles (providers/keys/defaults). Switching either re-hydrates the
// whole app via a page reload, which is the simplest reliable way to swap every view.
function renderProfiles() {
  const p = S.profiles;
  if (!p) return;
  const inco = p.data.incognito;

  const dsel = $("data-profile-select");
  dsel.innerHTML = "";
  p.data.profiles.forEach((pr) => {
    const o = document.createElement("option");
    o.value = pr.id; o.textContent = pr.name;
    dsel.appendChild(o);
  });
  const io = document.createElement("option");
  io.value = "__incognito__"; io.textContent = "🕶 Incognito (private)…";
  dsel.appendChild(io);
  dsel.value = inco ? "__incognito__" : (p.data.active || "");

  $("incognito-badge").classList.toggle("hidden", !inco);
  $("btn-incognito-save").classList.toggle("hidden", !inco);
  $("btn-data-profile-rename").disabled = inco;   // no live rename/delete of the private session
  $("btn-data-profile-delete").disabled = inco;

  const ssel = $("settings-profile-select");
  ssel.innerHTML = "";
  p.settings.profiles.forEach((pr) => {
    const o = document.createElement("option");
    o.value = pr.id; o.textContent = pr.name;
    ssel.appendChild(o);
  });
  ssel.value = p.settings.active || "";
}

function currentDataProfile() {
  const p = S.profiles && S.profiles.data;
  return (p && p.profiles.find((x) => x.id === p.active)) || null;
}
function checkedRadio(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
}

async function onDataProfileChange() {
  const val = $("data-profile-select").value;
  if (val === "__incognito__") {
    renderProfiles();   // revert the dropdown until the user confirms
    document.querySelector('input[name="incognito-seed"][value="blank"]').checked = true;
    openModal("modal-incognito");
    return;
  }
  if (S.profiles && val === S.profiles.data.active && !S.profiles.data.incognito) return;
  try {
    await api(`/api/profiles/data/${encodeURIComponent(val)}/activate`, { method: "POST" });
    location.reload();
  } catch (e) { toast("Switch failed: " + e.message); renderProfiles(); }
}

async function onSettingsProfileChange() {
  const val = $("settings-profile-select").value;
  if (S.profiles && val === S.profiles.settings.active) return;
  try {
    await api(`/api/profiles/settings/${encodeURIComponent(val)}/activate`, { method: "POST" });
    location.reload();
  } catch (e) { toast("Switch failed: " + e.message); renderProfiles(); }
}

function openNewDataProfile() {
  $("newdata-name").value = "";
  document.querySelector('input[name="newdata-seed"][value="blank"]').checked = true;
  openModal("modal-newdata");
  $("newdata-name").focus();
}
async function createDataProfile() {
  const name = $("newdata-name").value.trim();
  if (!name) { toast("Enter a profile name."); return; }
  try {
    await api("/api/profiles/data", { method: "POST",
      body: { name, seed: checkedRadio("newdata-seed") || "blank", activate: true } });
    closeModal();
    location.reload();
  } catch (e) { toast("Create failed: " + e.message); }
}

async function createSettingsProfile() {
  const name = await promptModal("New settings profile name", "");
  if (name == null || !name.trim()) return;
  try {
    await api("/api/profiles/settings", { method: "POST", body: { name: name.trim(), activate: true } });
    location.reload();
  } catch (e) { toast("Create failed: " + e.message); }
}

async function renameDataProfile() {
  const cur = currentDataProfile();
  if (!cur) return;
  const name = await promptModal("Rename profile", cur.name);
  if (name == null || !name.trim()) return;
  try {
    const r = await api(`/api/profiles/data/${cur.id}`, { method: "PATCH", body: { name: name.trim() } });
    S.profiles = r.profiles; renderProfiles();
  } catch (e) { toast("Rename failed: " + e.message); }
}
async function renameSettingsProfile() {
  const p = S.profiles.settings;
  const cur = p.profiles.find((x) => x.id === p.active);
  if (!cur) return;
  const name = await promptModal("Rename settings profile", cur.name);
  if (name == null || !name.trim()) return;
  try {
    const r = await api(`/api/profiles/settings/${cur.id}`, { method: "PATCH", body: { name: name.trim() } });
    S.profiles = r.profiles; renderProfiles();
  } catch (e) { toast("Rename failed: " + e.message); }
}

let _delAxis = "data";
function openDeleteProfile(axis) {
  _delAxis = axis;
  const p = axis === "data" ? S.profiles.data : S.profiles.settings;
  const sel = $("delprofile-select");
  sel.innerHTML = "";
  p.profiles.filter((x) => x.id !== p.active).forEach((pr) => {
    const o = document.createElement("option");
    o.value = pr.id; o.textContent = pr.name;
    sel.appendChild(o);
  });
  if (!sel.options.length) { toast("There's no other profile to delete."); return; }
  $("delprofile-title").textContent = axis === "data" ? "Delete Profile" : "Delete Settings Profile";
  openModal("modal-delprofile");
}
async function deleteProfile() {
  const id = $("delprofile-select").value;
  if (!id) return;
  const base = _delAxis === "data" ? "/api/profiles/data/" : "/api/profiles/settings/";
  try {
    const r = await api(base + id, { method: "DELETE" });
    S.profiles = r.profiles; closeModal(); renderProfiles();
    toast("Profile deleted.");
  } catch (e) { toast("Delete failed: " + e.message); }
}

async function startIncognito() {
  try {
    await api("/api/profiles/data/incognito", { method: "POST",
      body: { seed: checkedRadio("incognito-seed") || "blank" } });
    closeModal();
    location.reload();
  } catch (e) { toast("Could not start incognito: " + e.message); }
}
function openIncognitoSave() {
  $("incosave-name").value = "";
  document.querySelector('input[name="incosave-mode"][value="new"]').checked = true;
  $("incosave-new-row").classList.remove("hidden");
  $("incosave-merge-row").classList.add("hidden");
  const tsel = $("incosave-target");
  tsel.innerHTML = "";
  S.profiles.data.profiles.forEach((pr) => {
    const o = document.createElement("option");
    o.value = pr.id; o.textContent = pr.name;
    tsel.appendChild(o);
  });
  openModal("modal-incognito-save");
}
function onIncoSaveModeChange() {
  const mode = checkedRadio("incosave-mode");
  $("incosave-new-row").classList.toggle("hidden", mode !== "new");
  $("incosave-merge-row").classList.toggle("hidden", mode !== "merge");
}
async function saveIncognito() {
  const mode = checkedRadio("incosave-mode");
  const body = { mode };
  if (mode === "new") {
    const name = $("incosave-name").value.trim();
    if (!name) { toast("Enter a profile name."); return; }
    body.name = name;
  } else {
    body.target_id = $("incosave-target").value;
    if (!body.target_id) { toast("Pick a profile to merge into."); return; }
  }
  try {
    await api("/api/profiles/data/incognito/save", { method: "POST", body });
    closeModal();
    location.reload();
  } catch (e) { toast("Save failed: " + e.message); }
}

function bindProfileEvents() {
  $("data-profile-select").addEventListener("change", onDataProfileChange);
  $("settings-profile-select").addEventListener("change", onSettingsProfileChange);
  $("btn-data-profile-new").addEventListener("click", openNewDataProfile);
  $("btn-data-profile-rename").addEventListener("click", renameDataProfile);
  $("btn-data-profile-delete").addEventListener("click", () => openDeleteProfile("data"));
  $("btn-settings-profile-new").addEventListener("click", createSettingsProfile);
  $("btn-settings-profile-rename").addEventListener("click", renameSettingsProfile);
  $("btn-settings-profile-delete").addEventListener("click", () => openDeleteProfile("settings"));
  $("btn-newdata-ok").addEventListener("click", createDataProfile);
  $("btn-incognito-ok").addEventListener("click", startIncognito);
  $("btn-incognito-save").addEventListener("click", openIncognitoSave);
  $("btn-incosave-ok").addEventListener("click", saveIncognito);
  $("btn-delprofile-ok").addEventListener("click", deleteProfile);
  document.querySelectorAll('input[name="incosave-mode"]').forEach(
    (r) => r.addEventListener("change", onIncoSaveModeChange));
}

// ------------------------------- init --------------------------------
/** Boot the app: load state from the server, populate every control, wire events,
 *  and restore the last active chat. Runs once at the bottom of this file. */
async function init() {
  const st = await api("/api/state");
  S.app = st.app; S.config = st.config; S.servers = st.servers;
  S.presets = st.presets; S.libraries = st.libraries;
  S.prompts = st.prompts || { system: [], pre: [] };
  S.chats = st.chats; S.contextLengths = st.context_lengths;
  S.profiles = st.profiles || null;
  S.groups = st.chat_groups || [{ id: DEFAULT_GROUP_ID, name: "My Chats", builtin: true }];
  S.activeGroup = DEFAULT_GROUP_ID;
  S.minCrawledPages = st.min_crawled_pages || 7;
  S.providerPresets = st.provider_presets || [];
  S.rawServers = (st.config && st.config.servers) || [];
  S.defaultEvalPrompt = st.default_eval_prompt || "";
  S.evals = st.evals || [];
  S.defaultCriteria = st.default_criteria || [];
  S.parallel = {
    enabled: !!S.config.parallel_enabled,
    mode: S.config.parallel_mode || "balanced",
    servers: S.config.parallel_servers || [],
  };

  document.title = S.app.name;
  $("brand-name").textContent = S.app.name;
  $("brand-author").textContent = "by " + S.app.author;

  populateServers();
  populateContextLengths();
  populateDomains();
  $("chk-restrict").checked = !!S.config.restrict_to_approved;
  $("chk-parallel").checked = S.parallel.enabled;
  updateQueueUI();

  bindEvents();
  renderProfiles();
  bindProfileEvents();
  renderChatTabs();
  renderChatList();
  renderLibraryList();
  bindPersonaEditorEvents();
  await loadPersonas();

  // Load models for the selected server, then open a chat.
  await refreshModels(false);
  if (S.chats.length) {
    await loadChat(S.chats[0].id);
  } else {
    await newPrivateChat(true);
  }
}

// ------------------------------- servers/models ----------------------
function populateServers() {
  const sel = $("server-select");
  sel.innerHTML = "";
  S.servers.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.url; o.textContent = s.label;
    sel.appendChild(o);
  });
  sel.value = S.config.last_server_url || S.servers[0].url;
}
function currentServerUrl() { return $("server-select").value; }

async function refreshModels(showErrors) {
  const server = currentServerUrl();
  setStatus("Loading models…");
  try {
    const r = await api(`/api/models?server=${encodeURIComponent(server)}`);
    S.models = r.models || [];
    if (r.error && showErrors) toast("Could not load models: " + r.error);
    setStatus(r.error ? "Model load error: " + r.error : `${S.models.length} model(s) on ${server}`);
  } catch (e) {
    S.models = [];
    if (showErrors) toast("Model error: " + e.message);
    setStatus("Model error: " + e.message);
  }
  populateModels();
}
function getSelectedModel() {
  const manual = $("model-manual");
  if (!manual.classList.contains("hidden")) return manual.value.trim();
  return $("model-select").value;
}
function setSelectedModel(m) {
  const manual = $("model-manual");
  if (!manual.classList.contains("hidden")) { manual.value = m || ""; }
  else if ([...$("model-select").options].some((o) => o.value === m)) { $("model-select").value = m; }
}
function populateModels() {
  const sel = $("model-select");
  const manual = $("model-manual");
  sel.innerHTML = "";
  if (!S.models.length) {
    // No model list (e.g. a provider whose /models endpoint is unavailable) —
    // let the user type a model name.
    sel.classList.add("hidden");
    manual.classList.remove("hidden");
    manual.value = (S.chat && S.chat.model) || "";
    onModelChanged();
    return;
  }
  sel.classList.remove("hidden");
  manual.classList.add("hidden");
  S.models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  if (S.chat && S.chat.model && S.models.includes(S.chat.model)) {
    sel.value = S.chat.model;
  } else {
    sel.value = S.models[0];
    if (S.chat) S.chat.model = S.models[0];
  }
  onModelChanged();
}
async function onModelChanged() {
  const model = getSelectedModel();
  if (S.chat) S.chat.model = model;
  if (!model) { S.toolsSupported = null; updateToolsStatus(); return; }
  try {
    const r = await api(`/api/models/capabilities?server=${encodeURIComponent(currentServerUrl())}&model=${encodeURIComponent(model)}`);
    S.toolsSupported = r.tools;
  } catch (e) { S.toolsSupported = null; }
  updateToolsStatus();
}
function updateToolsStatus() {
  const el = $("tools-status");
  if (S.toolsSupported === true) { el.textContent = "✓ model supports tools"; el.style.color = "var(--status-ok)"; }
  else if (S.toolsSupported === false) { el.textContent = "✗ no tool support (app-side crawl only)"; el.style.color = "var(--status-warn)"; }
  else { el.textContent = ""; }
}

function populateContextLengths() {
  const sel = $("ctx-select");
  sel.innerHTML = "";
  S.contextLengths.forEach((n) => {
    const o = document.createElement("option");
    o.value = n; o.textContent = n.toLocaleString();
    sel.appendChild(o);
  });
}

// ------------------------------- chats -------------------------------
async function refreshChatSummaries() {
  const r = await api("/api/chats");
  S.chats = r.chats || [];
  renderChatList();
}
// Sidebar tabs (chat groups). The built-in "My Chats" tab holds chats with no
// group; each imported file becomes its own renameable/deletable tab.
function renderChatTabs() {
  const bar = $("chat-tabs");
  bar.innerHTML = "";
  S.groups.forEach((g) => {
    const tab = document.createElement("div");
    tab.className = "chat-tab" + (S.activeGroup === g.id ? " active" : "");
    const label = document.createElement("span");
    label.className = "label"; label.textContent = g.name;
    label.onclick = () => selectGroup(g.id);
    tab.appendChild(label);
    if (!g.builtin) {
      label.title = "Double-click to rename";
      label.ondblclick = (e) => { e.stopPropagation(); renameGroup(g.id, g.name); };
      const x = document.createElement("button");
      x.className = "tab-close"; x.textContent = "✕";
      x.title = "Delete this tab and its chats";
      x.onclick = (e) => { e.stopPropagation(); deleteGroup(g.id); };
      tab.appendChild(x);
    }
    bar.appendChild(tab);
  });
}
function selectGroup(id) {
  S.activeGroup = id;
  renderChatTabs();
  renderChatList();
}

function renderChatList() {
  const list = $("chat-list");
  list.innerHTML = "";
  const chats = S.chats.filter((c) => (c.group_id || DEFAULT_GROUP_ID) === S.activeGroup);
  chats.forEach((c) => {
    const row = document.createElement("div");
    row.className = "chat-row" + (S.chat && S.chat.id === c.id ? " selected" : "");
    const title = document.createElement("span");
    title.className = "title"; title.textContent = c.title || "New Chat";
    title.onclick = () => loadChat(c.id);
    const exp = document.createElement("button");
    exp.className = "export"; exp.textContent = "⬆";
    exp.title = "Export this chat";
    exp.onclick = (e) => { e.stopPropagation(); exportChat(c.id); };
    const del = document.createElement("button");
    del.className = "del"; del.textContent = "🗑";
    del.title = "Delete chat";
    del.onclick = (e) => { e.stopPropagation(); deleteChat(c.id); };
    row.appendChild(title); row.appendChild(exp); row.appendChild(del);
    list.appendChild(row);
  });
}

async function renameGroup(id, current) {
  const name = await promptModal("Rename tab", current || "");
  if (name === null) return;
  const t = name.trim();
  if (!t) return;
  try {
    const r = await api(`/api/chat-groups/${id}`, { method: "PATCH", body: { name: t } });
    S.groups = r.chat_groups;
    renderChatTabs();
  } catch (e) { toast("Rename failed: " + e.message); }
}
async function deleteGroup(id) {
  if (!confirm("Delete this tab and all of its chats?")) return;
  try {
    const r = await api(`/api/chat-groups/${id}`, { method: "DELETE" });
    S.groups = r.chat_groups; S.chats = r.chats;
    if (S.activeGroup === id) S.activeGroup = DEFAULT_GROUP_ID;
    renderChatTabs();
    renderChatList();
    if (S.chat && !S.chats.some((c) => c.id === S.chat.id)) {
      if (S.chats.length) loadChat(S.chats[0].id); else newPrivateChat(true);
    }
  } catch (e) { toast("Delete failed: " + e.message); }
}

// ------------------------------- export / import ---------------------
async function exportAllChats() {
  try {
    const r = await api("/api/chats/export", { method: "POST", body: { scope: "all" } });
    if (r.cancelled) return;
    toast(`Exported ${r.count} chat(s)`);
  } catch (e) { toast("Export failed: " + e.message); }
}
async function exportChat(id) {
  try {
    const r = await api(`/api/chats/${id}/export`, { method: "POST", body: {} });
    if (r.cancelled) return;
    toast("Chat exported");
  } catch (e) { toast("Export failed: " + e.message); }
}
async function importChats() {
  try {
    const r = await api("/api/chats/import", { method: "POST", body: {} });
    if (r.cancelled) return;
    S.chats = r.chats; S.groups = r.chat_groups;
    S.activeGroup = r.group.id;
    renderChatTabs();
    renderChatList();
    toast(`Imported ${r.count} chat(s) into “${r.group.name}”`);
  } catch (e) { toast("Import failed: " + e.message); }
}

function showChatView() {
  // Return from the parallel lane view to the normal single-chat message view.
  $("parallel-lanes").classList.add("hidden");
  $("messages").classList.remove("hidden");
}

/**
 * Make `chat` the active chat and sync every control to its saved settings
 * (server, model, context length, pre-prompt, toggles), then render its messages.
 * Central place where chat state becomes UI state.
 */
function loadChatObject(chat) {
  S.chat = chat;
  // The bottom usage bar tracks the active chat: drop any other chat's stale bar.
  Object.keys(S.contextBars).forEach((k) => {
    if (k.startsWith("chat:") && k !== "chat:" + (chat && chat.id)) delete S.contextBars[k];
  });
  renderContextBars();
  // Keep the sidebar tab in sync with the chat being opened.
  S.activeGroup = chat.group_id || DEFAULT_GROUP_ID;
  showChatView();
  // Sync UI to chat settings.
  $("chat-title").textContent = chat.title || "New Chat";
  $("private-badge").classList.toggle("hidden", !chat.private);
  $("btn-save-private").classList.toggle("hidden", !chat.private);
  $("btn-delete-chat").classList.toggle("hidden", !!chat.private);
  $("system-prompt").value = chat.system_prompt || "";
  $("system-on").checked = !!chat.system_on;
  $("pre-prompt").value = chat.pre_prompt || "";
  $("pre-on").checked = !!chat.pre_on;
  $("ctx-select").value = chat.num_ctx || S.config.default_num_ctx || 4096;
  $("chk-isolate").checked = !!chat.isolated;
  $("chk-hide-thinking").checked = !!chat.hide_thinking;
  $("chk-websearch").checked = !!chat.web_search;
  $("chk-strict").checked = !!chat.library_strict;
  $("crawl-pages").value = chat.crawl_pages || S.minCrawledPages || 7;
  $("chk-rag").checked = !!chat.rag_enabled;
  $("chk-rag-auto").checked = !!chat.rag_auto;
  $("rag-threshold").value = chat.rag_threshold || 400;
  $("chk-multipass").checked = !!chat.multi_pass;
  $("mp-passes").value = chat.passes || 2;
  $("chk-mp-system").checked = chat.pass_use_system !== false;
  $("mp-eval-prompt").value = chat.eval_prompt || S.defaultEvalPrompt;
  updateLibraryButton();
  updateWebsearchVisibility();
  updateMultipassVisibility();
  renderChatTabs();

  // Switch server/model if needed.
  if (chat.server_url && chat.server_url !== currentServerUrl()) {
    $("server-select").value = chat.server_url;
    refreshModels(false);
  } else {
    setSelectedModel(chat.model);
    onModelChanged();
  }
  renderMessages();
  renderChatList();
}

async function loadChat(id) {
  try {
    const r = await api(`/api/chats/${id}`);
    loadChatObject(r.chat);
  } catch (e) { toast("Could not load chat: " + e.message); }
}

async function newChat() {
  const r = await api("/api/chats", { method: "POST", body: {
    server_url: currentServerUrl(), model: getSelectedModel(), group_id: S.activeGroup,
  }});
  await refreshChatSummaries();
  loadChatObject(r.chat);
}
async function newPrivateChat(silent) {
  const r = await api("/api/chats", { method: "POST", body: {
    private: true, server_url: currentServerUrl(), model: getSelectedModel(),
    group_id: S.activeGroup,
  }});
  loadChatObject(r.chat);
  if (!silent) toast("Private chat (not saved)");
}
async function deleteChat(id) {
  if (!confirm("Delete this chat?")) return;
  await api(`/api/chats/${id}`, { method: "DELETE" });
  if (S.chat && S.chat.id === id) { S.chat = null; }
  await refreshChatSummaries();
  if (!S.chat) { if (S.chats.length) loadChat(S.chats[0].id); else newPrivateChat(true); }
}
async function renameChat() {
  if (!S.chat) return;
  const name = await promptModal("Rename chat", S.chat.title || "");
  if (name === null) return;
  S.chat.title = name.trim() || "New Chat";
  $("chat-title").textContent = S.chat.title;
  await persistChat(true);
  refreshChatSummaries();
}
async function clearMessages() {
  if (!S.chat || !S.chat.messages.length) { toast("Nothing to clear"); return; }
  if (S.generating) stopGeneration();
  if (!confirm("Clear all messages in this chat? Settings are kept.")) return;
  S.chat.messages = [];
  renderMessages();
  await persistChat(true);
  toast("Messages cleared");
}
async function savePrivate() {
  if (!S.chat || !S.chat.private) return;
  const r = await api(`/api/chats/${S.chat.id}/save`, { method: "POST", body: { chat: S.chat }});
  S.chat = r.chat;
  loadChatObject(S.chat);
  await refreshChatSummaries();
  toast("Chat saved");
}

let persistTimer = null;
/**
 * Save the active chat to the server. Debounced by default so typing settings changes
 * doesn't cause a write per keystroke; pass `immediate` to flush now (before switching
 * chats or leaving the page). Private chats are not persisted.
 */
async function persistChat(immediate) {
  if (!S.chat || S.chat.private) return;
  clearTimeout(persistTimer);
  const doIt = async () => {
    try { await api(`/api/chats/${S.chat.id}/persist`, { method: "POST", body: { chat: S.chat }}); }
    catch (e) {}
  };
  if (immediate) return doIt();
  persistTimer = setTimeout(doIt, 400);
}

// ------------------------------- messages ----------------------------
/** Rebuild the whole message list from `S.chat.messages`. Cheap enough at these
 *  sizes, and it keeps the DOM a pure function of state rather than something the
 *  streaming code has to patch incrementally. */
function renderMessages() {
  const box = $("messages");
  box.innerHTML = "";
  const hide = S.chat && S.chat.hide_thinking;
  (S.chat ? S.chat.messages : []).forEach((m, i) => {
    if (m.role !== "user" && m.role !== "assistant") return;
    // Saved reasoning renders as a collapsed bubble before the answer.
    if (m.role === "assistant" && m.reasoning && !hide) {
      box.appendChild(makeReasoningBubble(m.reasoning, true).el);
    }
    const isLastAssistant = m.role === "assistant" && i === lastAssistantIndex();
    box.appendChild(makeBubble(m.role, m.content, {
      regen: isLastAssistant, label: m.pass_label, intermediate: m.intermediate,
    }));
  });
  scrollBottom();
}
/** Build the collapsible chain-of-thought bubble shown above an answer when the
 *  model streams reasoning separately. */
function makeReasoningBubble(text, collapsed) {
  const el = document.createElement("div");
  el.className = "reasoning-bubble" + (collapsed ? " collapsed" : "");
  const head = document.createElement("div");
  head.className = "reasoning-head";
  const caret = document.createElement("span");
  caret.className = "reasoning-caret"; caret.textContent = "▸";
  head.appendChild(caret);
  head.appendChild(document.createTextNode(" 🧠 Reasoning"));
  head.onclick = () => el.classList.toggle("collapsed");
  const body = document.createElement("div");
  body.className = "reasoning-body"; body.textContent = text || "";
  el.appendChild(head); el.appendChild(body);
  return { el, body };
}
function lastAssistantIndex() {
  const msgs = S.chat ? S.chat.messages : [];
  for (let i = msgs.length - 1; i >= 0; i--) if (msgs[i].role === "assistant") return i;
  return -1;
}
/**
 * Build one message bubble element.
 * @param {string} role      "user" | "assistant"
 * @param {string} content   message text
 * @param {object} opts      `streaming: true` leaves the body open for token appends;
 *                           the returned element exposes `_body` for that purpose
 * @returns {HTMLElement}
 */
function makeBubble(role, content, opts = {}) {
  const b = document.createElement("div");
  b.className = "bubble " + role + (opts.streaming ? " streaming" : "") + (opts.intermediate ? " intermediate" : "");
  const r = document.createElement("div");
  r.className = "role";
  r.textContent = role === "user" ? "You" : ("Assistant" + (opts.label ? " · " + opts.label : ""));
  const body = document.createElement("div");
  body.className = "body"; body.textContent = content || "";
  b.appendChild(r); b.appendChild(body);
  const actions = document.createElement("div");
  actions.className = "actions";
  const copy = document.createElement("button");
  copy.className = "small"; copy.textContent = "Copy";
  copy.onclick = () => { navigator.clipboard.writeText(body.textContent); toast("Copied"); };
  actions.appendChild(copy);
  if (opts.regen && !opts.streaming) actions.appendChild(makeRegenControls());
  b.appendChild(actions);
  b._body = body;
  return b;
}
function makeRegenControls() {
  const wrap = document.createElement("span");
  const sel = document.createElement("select");
  S.models.forEach((m) => {
    const o = document.createElement("option"); o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  sel.value = (S.chat && S.chat.model) || (S.models[0] || "");
  const btn = document.createElement("button");
  btn.className = "small"; btn.textContent = "↻ Regenerate";
  btn.onclick = () => regenerate(sel.value);
  wrap.appendChild(sel); wrap.appendChild(btn);
  return wrap;
}
function scrollBottom() { const box = $("messages"); box.scrollTop = box.scrollHeight; }
// True when the view is already pinned near the bottom. Used to decide whether a
// streaming update should auto-follow — if the user has scrolled up, we leave the
// scroll position alone so they can read earlier content without being yanked down.
function isNearBottom(box) { return box.scrollHeight - box.scrollTop - box.clientHeight < 80; }

// Drag handle under the chat thread: pins #messages to an explicit height while
// dragging (double-click clears it to restore the default flex-fill). Pointer
// capture keeps the drag tracking even when the cursor moves over the controls below.
function setupMessagesResizer() {
  const handle = $("messages-resizer"), box = $("messages");
  if (!handle || !box) return;
  let dragging = false, startY = 0, startH = 0;
  handle.addEventListener("pointerdown", (e) => {
    dragging = true; startY = e.clientY; startH = box.getBoundingClientRect().height;
    handle.setPointerCapture(e.pointerId); document.body.style.userSelect = "none";
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    box.style.flex = "0 0 auto";
    box.style.height = Math.max(160, startH + (e.clientY - startY)) + "px";
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false; document.body.style.userSelect = "";
    try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
  };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  handle.addEventListener("dblclick", () => { box.style.flex = ""; box.style.height = ""; });
}

// ------------------------------- generation --------------------------
/** Flip the UI between idle and generating: swap Send for Stop and disable the
 *  controls that must not change mid-run. */
function setGeneratingUI(on) {
  S.generating = on;
  $("btn-send").classList.toggle("hidden", on);
  $("btn-batch").classList.toggle("hidden", on);
  $("btn-stop").classList.toggle("hidden", !on);
}

// --------------------------- data classification ---------------------------
function toggleDataMode() {
  S.dataMode = !S.dataMode;
  $("btn-data-mode").classList.toggle("active", S.dataMode);
  $("data-entry-row").classList.toggle("hidden", !S.dataMode);
  if (S.dataMode) $("data-label").focus();
}

// Turn a free-text label into a valid XML tag base name.
function sanitizeTag(label) {
  let t = String(label || "").trim().replace(/[^A-Za-z0-9_.-]+/g, "_");
  t = t.replace(/^[^A-Za-z_]+/, "");            // a tag name must start with a letter or underscore
  return t || "Data";
}

function addDataItem() {
  const text = $("input-box").value.trim();
  if (!text) { toast("Type the data into the message box first."); return; }
  const label = $("data-label").value.trim() || "Data";
  S.dataItems.push({ id: uid(), label, text });
  $("input-box").value = "";                    // clear the data box; keep the label so repeats increment
  $("input-box").focus();
  renderDataChips();
}

// Per-item tag: first block of a label is bare, later blocks of the same label get _0001, _0002…
function dataItemTags() {
  const counts = {};
  return S.dataItems.map((it) => {
    const base = sanitizeTag(it.label);
    const n = counts[base] || 0;
    counts[base] = n + 1;
    return n === 0 ? base : `${base}_${String(n).padStart(4, "0")}`;
  });
}

function renderDataChips() {
  const box = $("data-chips");
  box.innerHTML = "";
  const tags = dataItemTags();
  S.dataItems.forEach((it, i) => {
    const chip = document.createElement("span");
    chip.className = "data-chip";
    const name = document.createElement("button");
    name.type = "button"; name.className = "data-chip-name";
    name.textContent = tags[i];
    name.title = `Edit "${it.label}" block`;
    name.onclick = () => editDataItem(it.id);
    const del = document.createElement("button");
    del.type = "button"; del.className = "data-chip-x";
    del.textContent = "✕"; del.title = "Remove this data block";
    del.onclick = () => { S.dataItems = S.dataItems.filter((d) => d.id !== it.id); renderDataChips(); };
    chip.appendChild(name); chip.appendChild(del);
    box.appendChild(chip);
  });
  box.classList.toggle("hidden", S.dataItems.length === 0);
}

// Load a staged block back into the composer for editing (removing it from the list).
function editDataItem(id) {
  const it = S.dataItems.find((d) => d.id === id);
  if (!it) return;
  $("data-label").value = it.label;
  $("input-box").value = it.text;
  S.dataItems = S.dataItems.filter((d) => d.id !== id);
  if (!S.dataMode) toggleDataMode();
  $("input-box").focus();
  renderDataChips();
}

// Serialize staged blocks into a single <Data>…</Data> XML string ("" when none staged).
/** Serialize the staged data items into the <Data> block that gets prepended to the
 *  user turn. Returns "" when nothing is staged. */
function buildDataXml() {
  if (!S.dataItems.length) return "";
  const tags = dataItemTags();
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const blocks = S.dataItems.map((it, i) => `  <${tags[i]}>\n${esc(it.text)}\n  </${tags[i]}>`);
  return `<Data>\n${blocks.join("\n")}\n</Data>`;
}

function clearDataItems() {
  S.dataItems = [];
  renderDataChips();
}

// On-demand instruction improvement for the composer. Rewrites the typed message
// into a clearer prompt, keeping the original for one-click undo.
let _rewriteOriginal = null;
/** Send the composer text to the server to be rewritten into a clearer instruction,
 *  keeping the original so undoRewrite() can restore it. */
async function rewritePrompt() {
  if (S.generating || S.batchRunning) return;
  const box = $("input-box");
  const text = box.value.trim();
  if (!text) { toast("Type a message to rewrite first."); return; }
  const model = getSelectedModel();
  if (!model) { toast("Please select a model first."); return; }
  const btn = $("btn-rewrite");
  const prev = box.value;
  btn.disabled = true; btn.textContent = "✨ Rewriting…";
  try {
    const res = await api("/api/rewrite-prompt", {
      method: "POST",
      body: {
        text,
        server_url: currentServerUrl(),
        model,
        messages: S.chat ? S.chat.messages : [],
      },
    });
    const rewritten = (res.rewritten || "").trim();
    if (rewritten && rewritten !== prev) {
      _rewriteOriginal = prev;
      box.value = rewritten;
      $("btn-rewrite-undo").classList.remove("hidden");
      if (res.error) toast("Rewrite fell back to original: " + res.error);
    } else {
      toast(res.error ? ("Rewrite failed: " + res.error) : "No change suggested.");
    }
  } catch (e) {
    toast("Rewrite failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "✨ Rewrite";
    box.focus();
  }
}

function undoRewrite() {
  if (_rewriteOriginal == null) return;
  $("input-box").value = _rewriteOriginal;
  _rewriteOriginal = null;
  $("btn-rewrite-undo").classList.add("hidden");
  $("input-box").focus();
}

/**
 * Handle Send: assemble the user turn, push it into the chat, and dispatch to either
 * the persona pipeline or ordinary generation.
 *
 * Any staged data blocks are prepended to the typed text as XML, so the model sees
 * the data before the instruction. Creates a private chat on the fly if none is open,
 * and auto-titles a new chat from the first message. No-ops while a run is active.
 */
async function sendMessage() {
  if (S.generating || S.batchRunning) return;
  const text = $("input-box").value.trim();
  const dataXml = buildDataXml();
  if (!text && !dataXml) return;
  const model = getSelectedModel();
  if (!model) { toast("Please select a model first."); return; }
  if (!S.chat) await newPrivateChat(true);
  showChatView();
  S.chat.model = model;
  S.chat.server_url = currentServerUrl();

  // Add user turn (data blocks first, then any prompt).
  const content = dataXml ? (text ? dataXml + "\n\n" + text : dataXml) : text;
  S.chat.messages.push({ role: "user", content });
  // Auto-title.
  const titleText = text || ("Data: " + S.dataItems.map((d) => d.label).join(", "));
  if (["New Chat", "Private Chat", ""].includes(S.chat.title || "")) {
    let short = titleText.slice(0, 55).replace(/\n/g, " ").trim();
    if (titleText.length > 55) short += "…";
    S.chat.title = short;
    $("chat-title").textContent = short;
  }
  $("input-box").value = "";
  clearDataItems();

  // Web search query (per-message).
  let searchQuery = "";
  if ($("chk-websearch").checked) {
    searchQuery = $("search-query").value.trim();
    $("search-query").value = "";
  }
  renderMessages();
  if (S.usePersona && S.personaId) {
    await runPersona();
  } else {
    await runGeneration(searchQuery);
  }
  if (!S.chat.private) refreshChatSummaries();
}

/**
 * Re-answer the last user turn, optionally with a different model. Trailing assistant
 * messages are dropped first so the chat ends on the user turn the model must answer.
 */
async function regenerate(model) {
  if (S.generating || S.batchRunning) return;
  if (!S.chat) return;
  // Drop trailing assistant messages so the last turn is the user prompt.
  while (S.chat.messages.length && S.chat.messages[S.chat.messages.length - 1].role === "assistant") {
    S.chat.messages.pop();
  }
  if (model) { S.chat.model = model; setSelectedModel(model); }
  renderMessages();
  await runGeneration("");
}

// ------------------------------- Personas -------------------------------
async function loadPersonas() {
  try {
    const r = await api("/api/personas");
    S.personas = r.personas || [];
  } catch (e) { S.personas = []; }
  renderPersonaControls();
}

function renderPersonaControls() {
  const sel = $("persona-select");
  if (!sel) return;
  sel.innerHTML = "";
  S.personas.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = p.name;
    sel.appendChild(o);
  });
  if (S.personaId && S.personas.some((p) => p.id === S.personaId)) sel.value = S.personaId;
  else if (S.personas.length) { S.personaId = S.personas[0].id; sel.value = S.personaId; }
  sel.classList.toggle("hidden", !S.usePersona || !S.personas.length);
  renderVariantControl();
}

function renderVariantControl() {
  const vsel = $("persona-variant");
  const p = S.personas.find((x) => x.id === S.personaId);
  const variants = (p && p.variants) || [];
  vsel.innerHTML = "";
  const def = document.createElement("option"); def.value = ""; def.textContent = "(default voice)";
  vsel.appendChild(def);
  variants.forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; vsel.appendChild(o); });
  vsel.classList.toggle("hidden", !S.usePersona || !variants.length);
}

function onPersonaToggle() {
  S.usePersona = $("chk-persona").checked;
  if (S.usePersona && !S.personas.length) {
    toast("No personas yet — create one in the Personas tab.");
  }
  renderPersonaControls();
}

// Render one collapsible step card into a pipeline panel.
/**
 * Build one collapsible pipeline step card. The card owns its expand/collapse and its
 * "Re-run from here" button; the JSON textarea is editable so the user can correct a
 * step's output and re-run everything downstream of it.
 * @param {number} index step position, used as the re-run anchor
 */
function makeStepCard(index, id, type) {
  const card = document.createElement("div");
  card.className = "step-card running";
  card.dataset.index = index;
  card.innerHTML =
    `<div class="step-head"><span class="step-status">⏳</span>` +
    `<span class="step-name">${index + 1}. ${id} <em>(${type})</em></span>` +
    `<button class="step-toggle ghost small">▸</button></div>` +
    `<div class="step-body hidden"><textarea class="step-json" rows="6"></textarea>` +
    `<div class="step-raw"></div>` +
    `<button class="step-rerun small">Re-run from here ▶</button></div>`;
  const head = card.querySelector(".step-head");
  const body = card.querySelector(".step-body");
  head.querySelector(".step-toggle").onclick = () => body.classList.toggle("hidden");
  head.onclick = (e) => { if (e.target.tagName !== "BUTTON") body.classList.toggle("hidden"); };
  card.querySelector(".step-rerun").onclick = () => rerunFromStep(index, card);
  return card;
}

/**
 * Re-run the pipeline from `index` forward, substituting whatever is currently in that
 * card's JSON textarea as the step's output. Invalid JSON is passed through as a raw
 * string rather than rejected — some steps legitimately produce free text.
 */
async function rerunFromStep(index, card) {
  if (S.generating || !S.personaRunId) return;
  let output = null;
  const ta = card.querySelector(".step-json");
  const raw = ta.value.trim();
  if (raw) { try { output = JSON.parse(raw); } catch (e) { output = raw; } }
  await streamPersonaRun(`/api/runs/${S.personaRunId}/rerun`,
    { index, output }, card.closest(".persona-run"), true);
}

/** Start a persona-backed answer: build the step panel plus the final answer bubble,
 *  then stream the run into them. */
async function runPersona() {
  setGeneratingUI(true);
  S.runId = uid();
  S.personaRunId = S.runId;
  // Assistant turn: a pipeline panel (step cards) + the final answer bubble.
  const wrap = document.createElement("div");
  wrap.className = "persona-run";
  wrap.innerHTML = `<div class="persona-steps"></div>`;
  const bubble = makeBubble("assistant", "", { streaming: true });
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  bubble._body.textContent = "…";
  scrollBottom();
  await streamPersonaRun(`/api/personas/${S.personaId}/chat`,
    { chat: S.chat, run_id: S.runId }, wrap, false, bubble);
}

/**
 * Shared streamer for both an initial persona run and a re-run-from-step.
 *
 * Renders `step_started` / `step_output` / `step_failed` frames as cards and `token`
 * frames into the answer bubble. On a re-run, the cards from `index` onward are
 * replaced while earlier ones are left intact.
 *
 * @param {HTMLElement} wrap      the .persona-run container owning the step panel
 * @param {boolean}     isRerun   true when resuming mid-pipeline
 * @param {HTMLElement} bubbleArg existing answer bubble; a re-run creates its own
 */
async function streamPersonaRun(path, body, wrap, isRerun, bubbleArg) {
  const stepsEl = wrap.querySelector(".persona-steps");
  let bubble = bubbleArg;
  if (isRerun) {
    // clear cards at/after the edited index and reset the answer bubble
    stepsEl.innerHTML = "";
    bubble = wrap.querySelector(".bubble.assistant") || makeBubble("assistant", "", { streaming: true });
    if (!wrap.contains(bubble)) wrap.appendChild(bubble);
    bubble._body.textContent = "…";
  }
  let finalText = "";
  const cards = {};
  await streamSSE(path, body, {
    start: () => setStatus("Running persona pipeline…"),
    step_started: (d) => {
      const c = makeStepCard(d.index, d.id, d.type);
      cards[d.index] = c; stepsEl.appendChild(c);
      if (bubble._body.textContent === "…") bubble._body.textContent = "";
      scrollBottom();
    },
    step_output: (d) => {
      const c = cards[d.index]; if (!c) return;
      c.classList.remove("running"); c.classList.add("done");
      c.querySelector(".step-status").textContent = "✓";
      c.querySelector(".step-json").value =
        typeof d.output === "string" ? d.output : JSON.stringify(d.output, null, 2);
    },
    step_failed: (d) => {
      const c = cards[d.index]; if (!c) return;
      c.classList.remove("running"); c.classList.add("failed");
      c.querySelector(".step-status").textContent = "✗";
      c.querySelector(".step-body").classList.remove("hidden");
      c.querySelector(".step-raw").textContent = "Error: " + (d.error || "") + (d.raw ? ("\nRaw: " + d.raw) : "");
    },
    token: (d) => {
      const follow = isNearBottom($("messages"));
      finalText += d.text; bubble._body.textContent = finalText;
      if (follow) scrollBottom();
    },
    run_paused: () => { setStatus("Pipeline paused — edit a step and re-run."); },
    run_complete: (d) => { if (d.final) { finalText = d.final; bubble._body.textContent = finalText; } },
    error: (d) => { bubble._body.textContent = "[Error] " + d.message; toast("Persona error: " + d.message); },
    done: () => {},
  });
  bubble.classList.remove("streaming");
  setGeneratingUI(false); setStatus("");
  // Save the final answer as the assistant turn (once), then persist.
  if (!isRerun && finalText) {
    S.chat.messages.push({ role: "assistant", content: finalText });
    if (!S.chat.private) await persistChat(true);
    // Offer to save this exchange as a persona memory.
    const btn = document.createElement("button");
    btn.className = "small ghost save-memory-btn";
    btn.textContent = "💾 Save as memory";
    const userMsg = [...S.chat.messages].reverse().find((m) => m.role === "user");
    btn.onclick = () => openSaveMemory([userMsg, { role: "assistant", content: finalText }]);
    wrap.appendChild(btn);
  }
}

let _memPersonaId = "";
async function openSaveMemory(messages) {
  _memPersonaId = S.personaId;
  if (!_memPersonaId) { toast("No active persona."); return; }
  $("mem-title").value = ""; $("mem-desc").value = "…drafting…";
  $("mem-weight").value = 5; $("mem-tags").value = "";
  openModal("modal-memory");
  try {
    const r = await api(`/api/personas/${_memPersonaId}/draft-memory`, {
      method: "POST",
      body: { messages, server_url: currentServerUrl(), model: getSelectedModel() },
    });
    const d = r.draft || {};
    $("mem-title").value = d.title || "";
    $("mem-desc").value = d.description || "";
    $("mem-weight").value = d.emotional_weight || 5;
  } catch (e) {
    $("mem-desc").value = (messages.map((m) => m.content).join("\n\n")).slice(0, 400);
  }
}

async function saveMemory() {
  if (!_memPersonaId) return;
  const memory = {
    title: $("mem-title").value.trim(),
    description: $("mem-desc").value.trim(),
    emotional_weight: parseInt($("mem-weight").value) || 5,
    tags: $("mem-tags").value.split(",").map((t) => t.trim()).filter(Boolean),
  };
  if (!memory.title && !memory.description) { toast("Add a title or description."); return; }
  try {
    await api(`/api/personas/${_memPersonaId}/memories`, { method: "POST", body: { memory } });
    toast("Memory saved.");
    closeModal();
    if (S.editingPersona && S.editingPersona.id === _memPersonaId && $("tab-personas").classList.contains("active"))
      renderPersonaMemories();
  } catch (e) { toast("Save failed: " + e.message); }
}

// ===================== Personas tab (editors) =====================
async function renderPersonaHealth() {
  const b = $("persona-health"); if (!b) return;
  try {
    const h = await api("/api/persona-health");
    if (!h.ollama) {
      b.textContent = `⚠ Embedding server ${h.embed_url} is unreachable. Start Ollama, or set a reachable embedding server in Settings. (Keyword-mode personas still work without it.)`;
      b.classList.remove("hidden");
    } else if (!h.embed_present) {
      b.textContent = `⚠ Embedding model "${h.embed_model}" not found on ${h.embed_url}. Pull it:  ollama pull ${h.embed_model}`;
      b.classList.remove("hidden");
    } else { b.classList.add("hidden"); }
  } catch (e) { b.classList.add("hidden"); }
}

async function renderPersonaTab() {
  await loadPersonas();
  renderPersonaHealth();
  const list = $("persona-list");
  list.innerHTML = "";
  S.personas.forEach((p) => {
    const row = document.createElement("div");
    row.className = "side-item" + (S.editingPersona && S.editingPersona.id === p.id ? " active" : "");
    row.textContent = p.name + (p.role ? ` — ${p.role}` : "");
    row.onclick = () => openPersonaEditor(p.id);
    list.appendChild(row);
  });
  if (!S.personas.length) { $("persona-editor").classList.add("hidden"); $("persona-empty").classList.remove("hidden"); }
}

async function openPersonaEditor(id) {
  try {
    const r = await api(`/api/personas/${id}`);
    S.editingPersona = r.persona;
  } catch (e) { toast("Load failed: " + e.message); return; }
  const p = S.editingPersona;
  $("persona-empty").classList.add("hidden");
  $("persona-editor").classList.remove("hidden");
  $("pe-name").value = p.profile.name || "";
  $("pe-role").value = p.profile.role || "";
  $("pe-bio").value = p.profile.bio || "";
  $("pe-chat-model").value = p.models.chat_model || "";
  $("pe-embed-model").value = p.models.embedding_model || "";
  $("pe-temp").value = p.models.temperature ?? 0.7;
  $("pe-retrieval").value = p.stores.retrieval || "hybrid";
  $("pe-reword").checked = p.stores.prompt_reword !== false;
  $("pe-tone").value = p.speaking.tone || "";
  $("pe-formality").value = p.speaking.formality || "";
  $("pe-vocab").value = p.speaking.vocabulary || "";
  $("pe-quirks").value = p.speaking.quirks || "";
  renderPeVariants(); renderPeExamples(); renderPeSteps();
  renderPersonaKb(); renderPersonaMemories(); renderEmbedBanner();
  $("pe-compile-progress").classList.add("hidden");
  refreshCompileStatus("persona", p.id, $("pe-compile-badge"));
  peShowSub("profile");
  renderPersonaTab();  // refresh active highlight
}

function peShowSub(name) {
  document.querySelectorAll(".pe-tab").forEach((t) => t.classList.toggle("active", t.dataset.petab === name));
  document.querySelectorAll(".pe-page").forEach((pg) => pg.classList.toggle("active", pg.id === "pe-" + name));
}

function collectPersona() {
  const p = S.editingPersona; if (!p) return null;
  p.profile.name = $("pe-name").value.trim();
  p.profile.role = $("pe-role").value.trim();
  p.profile.bio = $("pe-bio").value;
  p.models.chat_model = $("pe-chat-model").value.trim();
  p.models.embedding_model = $("pe-embed-model").value.trim() || "nomic-embed-text";
  p.models.temperature = parseFloat($("pe-temp").value) || 0.7;
  p.stores.retrieval = $("pe-retrieval").value;
  p.stores.prompt_reword = $("pe-reword").checked;
  p.speaking.tone = $("pe-tone").value;
  p.speaking.formality = $("pe-formality").value;
  p.speaking.vocabulary = $("pe-vocab").value;
  p.speaking.quirks = $("pe-quirks").value;
  // variants + examples + steps are collected live into p by their editors
  return p;
}

async function savePersona() {
  const p = collectPersona(); if (!p) return;
  if (!collectPeSteps()) return;   // validates step JSON schemas
  try {
    const r = await api(`/api/personas/${p.id}`, { method: "PUT", body: { persona: p } });
    S.editingPersona = r.persona;
    toast("Saved.");
    renderPersonaTab();
  } catch (e) { toast("Save failed: " + e.message); }
}

async function createPersonaUI() {
  const name = await promptModal("New persona name", "");
  if (!name) return;
  try {
    const r = await api("/api/personas", { method: "POST", body: { name, chat_model: getSelectedModel() } });
    await renderPersonaTab();
    openPersonaEditor(r.persona.id);
  } catch (e) { toast("Create failed: " + e.message); }
}

async function duplicatePersonaUI() {
  if (!S.editingPersona) return;
  const name = await promptModal("Duplicate as", S.editingPersona.profile.name + " copy");
  if (!name) return;
  const r = await api(`/api/personas/${S.editingPersona.id}/duplicate`, { method: "POST", body: { name } });
  await renderPersonaTab(); openPersonaEditor(r.persona.id);
}

async function deletePersonaUI() {
  if (!S.editingPersona) return;
  if (!confirm(`Delete persona "${S.editingPersona.profile.name}" and all its data?`)) return;
  await api(`/api/personas/${S.editingPersona.id}`, { method: "DELETE" });
  S.editingPersona = null;
  $("persona-editor").classList.add("hidden"); $("persona-empty").classList.remove("hidden");
  renderPersonaTab();
}

// ---- Speaking: variants + examples ----
function renderPeVariants() {
  const host = $("pe-variants"); host.innerHTML = "";
  (S.editingPersona.speaking.variants || []).forEach((v, i) => {
    const row = document.createElement("div"); row.className = "row-controls";
    row.innerHTML = `<input type="text" placeholder="name" value="${escapeHtml(v.name || "")}" />
      <input type="text" placeholder="description" value="${escapeHtml(v.description || "")}" style="flex:1" />
      <button class="small danger">✕</button>`;
    const [n, d] = row.querySelectorAll("input");
    n.oninput = () => v.name = n.value; d.oninput = () => v.description = d.value;
    row.querySelector("button").onclick = () => { S.editingPersona.speaking.variants.splice(i, 1); renderPeVariants(); };
    host.appendChild(row);
  });
}
function renderPeExamples() {
  const host = $("pe-examples"); host.innerHTML = "";
  (S.editingPersona.speaking.examples || []).forEach((ex, i) => {
    const row = document.createElement("div"); row.className = "row-controls";
    row.innerHTML = `<input type="text" placeholder="user" value="${escapeHtml(ex.user || "")}" style="flex:1" />
      <input type="text" placeholder="reply" value="${escapeHtml(ex.reply || "")}" style="flex:1" />
      <button class="small danger">✕</button>`;
    const [u, r] = row.querySelectorAll("input");
    u.oninput = () => ex.user = u.value; r.oninput = () => ex.reply = r.value;
    row.querySelector("button").onclick = () => { S.editingPersona.speaking.examples.splice(i, 1); renderPeExamples(); };
    host.appendChild(row);
  });
}

// ---- Pipeline steps ----
function renderPeSteps() {
  const host = $("pe-steps"); host.innerHTML = "";
  (S.editingPersona.pipeline || []).forEach((st, i) => {
    const card = document.createElement("div"); card.className = "step-card done";
    card.innerHTML =
      `<div class="step-head"><span class="step-name"><input class="s-id" value="${escapeHtml(st.id || "")}" style="width:8em" />
        <select class="s-type">
          <option value="llm">llm</option><option value="knowledge_retrieval">knowledge_retrieval</option>
          <option value="memory_retrieval">memory_retrieval</option></select>
        <label class="chk"><input type="checkbox" class="s-hist" ${st.use_history ? "checked" : ""}/> history</label>
        <input class="s-model" placeholder="model override" value="${escapeHtml(st.model || "")}" style="width:9em" />
        </span>
        <button class="small s-up">↑</button><button class="small s-down">↓</button><button class="small danger s-del">✕</button></div>
      <div class="step-body"><textarea class="s-prompt" rows="3" placeholder="prompt template">${escapeHtml(st.prompt || "")}</textarea>
        <textarea class="s-schema" rows="3" placeholder="JSON schema (optional)">${st.schema ? escapeHtml(JSON.stringify(st.schema, null, 2)) : ""}</textarea></div>`;
    card.querySelector(".s-type").value = st.type;
    card.querySelector(".s-up").onclick = () => { if (i > 0) { const a = S.editingPersona.pipeline; [a[i-1],a[i]]=[a[i],a[i-1]]; collectPeSteps(); renderPeSteps(); } };
    card.querySelector(".s-down").onclick = () => { const a = S.editingPersona.pipeline; if (i < a.length-1) { [a[i+1],a[i]]=[a[i],a[i+1]]; collectPeSteps(); renderPeSteps(); } };
    card.querySelector(".s-del").onclick = () => { S.editingPersona.pipeline.splice(i,1); renderPeSteps(); };
    host.appendChild(card);
  });
}
function collectPeSteps() {
  const cards = $("pe-steps").querySelectorAll(".step-card");
  const steps = [];
  for (const c of cards) {
    const schemaText = c.querySelector(".s-schema").value.trim();
    let schema = null;
    if (schemaText) { try { schema = JSON.parse(schemaText); } catch (e) { toast("Invalid JSON schema in step " + c.querySelector(".s-id").value); return false; } }
    steps.push({
      id: c.querySelector(".s-id").value.trim(),
      type: c.querySelector(".s-type").value,
      use_history: c.querySelector(".s-hist").checked,
      model: c.querySelector(".s-model").value.trim(),
      prompt: c.querySelector(".s-prompt").value,
      schema,
    });
  }
  S.editingPersona.pipeline = steps;
  return true;
}

// ---- Knowledge ----
async function renderPersonaKb() {
  const host = $("pe-kb-list"); host.innerHTML = "Loading…";
  try {
    const r = await api(`/api/personas/${S.editingPersona.id}/knowledge`);
    host.innerHTML = "";
    (r.documents || []).forEach((d) => {
      const row = document.createElement("div"); row.className = "side-item";
      row.innerHTML = `<span>${escapeHtml(d.item_id)} <em>(${d.chunks} chunks)</em></span> <button class="small danger">✕</button>`;
      row.querySelector("button").onclick = async () => {
        await api(`/api/personas/${S.editingPersona.id}/knowledge/${encodeURIComponent(d.item_id)}`, { method: "DELETE" });
        renderPersonaKb();
      };
      host.appendChild(row);
    });
    if (!(r.documents || []).length) host.innerHTML = "<div class='muted'>No documents yet.</div>";
  } catch (e) { host.innerHTML = "<div class='muted'>Load failed.</div>"; }
}
async function addPersonaKbFiles() {
  toast("Choose documents in the dialog…");
  try {
    const r = await api(`/api/personas/${S.editingPersona.id}/knowledge/add-files`, { method: "POST", body: {} });
    if (r.errors && r.errors.length) toast(r.errors.join("; "));
    else toast(`Added ${(r.added || []).length} document(s).`);
    renderPersonaKb(); renderEmbedBanner();
    refreshCompileStatus("persona", S.editingPersona.id, $("pe-compile-badge"));
  } catch (e) { toast("Add failed: " + e.message); }
}
function renderEmbedBanner() {
  const b = $("pe-embed-banner");
  const used = S.editingPersona.stores.embedding_model_used;
  const cur = S.editingPersona.models.embedding_model;
  if (used && cur && used !== cur && S.editingPersona.stores.retrieval !== "keyword") {
    b.textContent = `⚠ Knowledge was embedded with "${used}" but this persona now uses "${cur}". Re-add documents to re-index.`;
    b.classList.remove("hidden");
  } else b.classList.add("hidden");
}

// ---- Memories ----
async function renderPersonaMemories() {
  const host = $("pe-mem-list"); host.innerHTML = "Loading…";
  try {
    const r = await api(`/api/personas/${S.editingPersona.id}/memories`);
    host.innerHTML = "";
    (r.memories || []).forEach((m) => {
      const row = document.createElement("div"); row.className = "side-item";
      row.innerHTML = `<span><b>${escapeHtml(m.title || "(untitled)")}</b> — ${escapeHtml((m.description||"").slice(0,60))} <em>(w${m.emotional_weight})</em></span> <button class="small danger">✕</button>`;
      row.querySelector("button").onclick = async () => {
        await api(`/api/personas/${S.editingPersona.id}/memories/${m.id}`, { method: "DELETE" });
        renderPersonaMemories();
      };
      host.appendChild(row);
    });
    if (!(r.memories || []).length) host.innerHTML = "<div class='muted'>No memories yet.</div>";
  } catch (e) { host.innerHTML = "<div class='muted'>Load failed.</div>"; }
}
function addPersonaMemoryUI() {
  _memPersonaId = S.editingPersona.id;
  $("mem-title").value = ""; $("mem-desc").value = ""; $("mem-weight").value = 5; $("mem-tags").value = "";
  openModal("modal-memory");
}

// ---- Test run ----
async function runPersonaTest() {
  const msg = $("pe-test-msg").value.trim(); if (!msg) return;
  if (!collectPeSteps()) return;
  // Save first so the test uses the current definition.
  await savePersona();
  const wrap = $("pe-test-output"); wrap.innerHTML = `<div class="persona-steps"></div>`;
  const bubble = makeBubble("assistant", "", { streaming: true }); wrap.appendChild(bubble); bubble._body.textContent = "…";
  S.runId = uid(); S.personaRunId = S.runId; S.generating = true;
  const syntheticChat = { id: "test", private: true, messages: [{ role: "user", content: msg }],
                          server_url: currentServerUrl(), model: getSelectedModel() };
  const savedChat = S.chat; S.chat = syntheticChat;   // streamPersonaRun reads S.chat
  await streamPersonaRun(`/api/personas/${S.editingPersona.id}/chat`,
    { chat: syntheticChat, run_id: S.runId }, wrap, false, bubble);
  S.chat = savedChat; S.generating = false;
}

function bindPersonaEditorEvents() {
  $("btn-persona-new").onclick = createPersonaUI;
  $("btn-persona-save").onclick = savePersona;
  $("btn-persona-duplicate").onclick = duplicatePersonaUI;
  $("btn-persona-delete").onclick = deletePersonaUI;
  document.querySelectorAll(".pe-tab").forEach((t) => t.onclick = () => peShowSub(t.dataset.petab));
  $("btn-pe-variant-add").onclick = () => { S.editingPersona.speaking.variants.push({ name: "", description: "" }); renderPeVariants(); };
  $("btn-pe-example-add").onclick = () => { S.editingPersona.speaking.examples.push({ user: "", reply: "" }); renderPeExamples(); };
  $("btn-pe-step-add").onclick = () => { collectPeSteps(); S.editingPersona.pipeline.push({ id: "step", type: "llm", use_history: false, model: "", prompt: "", schema: null }); renderPeSteps(); };
  $("btn-pe-pipeline-default").onclick = async () => { const r = await api("/api/persona-default-pipeline"); S.editingPersona.pipeline = r.pipeline; renderPeSteps(); toast("Default pipeline restored (save to keep)."); };
  $("btn-pe-kb-add").onclick = addPersonaKbFiles;
  $("btn-pe-compile").onclick = async () => {
    if (!S.editingPersona) { toast("Open a persona first"); return; }
    await runCompile("persona", S.editingPersona.id, { force: $("pe-compile-force").checked },
                     $("pe-compile-progress"), $("pe-compile-badge"));
    renderPersonaKb(); renderEmbedBanner();
  };
  $("btn-pe-mem-add").onclick = addPersonaMemoryUI;
  $("btn-pe-test").onclick = runPersonaTest;
  $("btn-persona-export-xml").onclick = () => window.open(`/api/personas/${S.editingPersona.id}/export.xml`, "_blank");
  $("btn-persona-export-bundle").onclick = () => window.open(`/api/personas/${S.editingPersona.id}/export.zip`, "_blank");
  $("btn-persona-import").onclick = importPersonaUI;
}

// Import is implemented in Phase 10; stub keeps the button harmless until then.
async function importPersonaUI() {
  try {
    const r = await api("/api/personas/import", { method: "POST", body: {} });
    if (r.error) { toast(r.error); return; }
    toast("Imported: " + (r.persona ? r.persona.profile.name : "ok"));
    await renderPersonaTab();
    if (r.persona) openPersonaEditor(r.persona.id);
  } catch (e) { toast("Import: " + e.message); }
}

async function runGeneration(searchQuery, opts = {}) {
  setGeneratingUI(true);
  S.runId = uid();
  // The chat sent to the backend (for its context + settings). Defaults to the
  // active chat. A queue run passes an isolated single-turn snapshot here so each
  // prompt is answered independently, while all rendering/persistence below stays
  // targeted at the active S.chat so results stack into one thread.
  const sendChat = opts.sendChat || S.chat;
  // Pass-driven: one answer (+optional reasoning) bubble per pass. A single-pass
  // generation is just one unlabeled pass.
  let bubble = null, reasonBubble = null;
  let curContent = "", curReason = "", curLabel = "", curIntermediate = false;
  let errored = false;

  function finalizePass() {
    if (!bubble) return;
    bubble.classList.remove("streaming");
    if (!errored) {
      const msg = { role: "assistant", content: curContent };
      if (curReason) msg.reasoning = curReason;
      if (curLabel) { msg.pass_label = curLabel; msg.intermediate = curIntermediate; }
      S.chat.messages.push(msg);
    }
    bubble = null; reasonBubble = null;
    curContent = ""; curReason = ""; curLabel = ""; curIntermediate = false;
  }

  await streamSSE(`/api/chats/${sendChat.id}/send`, {
    chat: sendChat, search_query: searchQuery, run_id: S.runId,
  }, {
    start: () => setStatus("Generating…"),
    status: (d) => { if (bubble) bubble._body.textContent = d.message; },
    context: (d) => upsertContextBar(d),
    pass_start: (d) => {
      const follow = isNearBottom($("messages"));
      finalizePass();  // save the previous pass, if any
      curLabel = d.label || ""; curIntermediate = !!d.intermediate;
      bubble = makeBubble("assistant", "", { streaming: true, label: curLabel, intermediate: curIntermediate });
      $("messages").appendChild(bubble);
      if (follow) scrollBottom();
    },
    reasoning: (d) => {
      if (!bubble) return;
      const follow = isNearBottom($("messages"));
      if (!reasonBubble) {
        reasonBubble = makeReasoningBubble("", false);
        $("messages").insertBefore(reasonBubble.el, bubble);
      }
      curReason += d.content; reasonBubble.body.textContent = curReason;
      if (follow) scrollBottom();
    },
    chunk: (d) => {
      if (!bubble) return;
      const follow = isNearBottom($("messages"));
      if (curContent === "") bubble._body.textContent = "";
      curContent += d.content; bubble._body.textContent = curContent;
      if (follow) scrollBottom();
    },
    pass_end: (d) => {
      if (d.content !== undefined) curContent = d.content;
      if (d.reasoning !== undefined) curReason = d.reasoning;
      if (bubble) bubble._body.textContent = curContent;
      finalizePass();
    },
    error: (d) => {
      errored = true;
      if (!bubble) { bubble = makeBubble("assistant", "", {}); $("messages").appendChild(bubble); }
      bubble._body.textContent = "[Error] " + d.message;
      bubble.classList.remove("streaming");
      toast("Generation error: " + d.message);
    },
    done: () => {},
  });

  finalizePass();  // safety: save any pass that didn't get a pass_end (e.g. stop)
  setGeneratingUI(false);
  setStatus("");
  if (!errored) await persistChat(true);
  renderMessages(); // re-render so the final assistant gets regen controls
}

async function stopGeneration() {
  S.queueStop = true;   // also halt a sequential queue loop after the current item
  if (S.runId) await api("/api/stop", { method: "POST", body: { run_id: S.runId }});
}

// ------------------------------- batch -------------------------------
async function batchProcess() {
  if (S.generating || S.batchRunning) return;
  const model = getSelectedModel();
  if (!model) { toast("Please select a model first."); return; }
  if (!S.chat) await newPrivateChat(true);
  S.chat.model = model; S.chat.server_url = currentServerUrl();

  setStatus("Waiting for folder selection…");
  let folder = "";
  try {
    const r = await api("/api/pick-folder", { method: "POST", body: { title: "Choose a folder of prompt files (.txt / .md)" }});
    folder = r.path;
  } catch (e) { setStatus(""); toast("Folder picker failed: " + e.message); return; }
  if (!folder) { setStatus(""); return; }

  S.runId = uid();
  toast("Batch started: " + folder, 5000);

  // Parallel batch: distribute files across the selected servers (lane view).
  if (S.parallel.enabled && (S.parallel.servers || []).length) {
    await runParallel("/api/batch/start", { chat: S.chat, folder, run_id: S.runId }, { batch: true });
    if (!S.chat.private) refreshChatSummaries();
    return;
  }

  S.batchRunning = true;
  setGeneratingUI(true);

  await streamSSE("/api/batch/start", { chat: S.chat, folder, run_id: S.runId }, {
    start: (d) => setStatus(`Batch: 0/${d.total} → ${d.responses_dir}`),
    progress: (d) => setStatus(`Batch ${d.index}/${d.total}: ${d.name}`),
    context: (d) => upsertContextBar(d),
    file: (d) => {
      S.chat.messages.push({ role: "user", content: `📄 ${d.name}\n\n${d.prompt}` });
      S.chat.messages.push({ role: "assistant", content: d.response });
      renderMessages();
      persistChat(true);
    },
    done: (d) => {
      const verb = d.stopped ? "stopped" : "complete";
      toast(`Batch ${verb} — ${d.count} file(s) → ${d.responses_dir}`, 6000);
      setStatus(`Batch ${verb} — ${d.count} file(s)`);
    },
    error: (d) => { toast("Batch error: " + d.message); setStatus(""); },
  });

  S.batchRunning = false;
  setGeneratingUI(false);
  if (!S.chat.private) refreshChatSummaries();
}

// ------------------------------- queue -------------------------------
function shortTitle(text) {
  let s = text.slice(0, 55).replace(/\n/g, " ").trim();
  if (text.length > 55) s += "…";
  return s || "Queued prompt";
}
/**
 * Snapshot the current chat settings into a standalone chat carrying one user turn.
 *
 * Queued and parallel items must each be independently answerable, because they may
 * run out of order or simultaneously on different servers. Copying the settings at
 * enqueue time also means later changes to the open chat don't retroactively alter
 * work that is already queued.
 */
function buildChatSnapshot(text, model) {
  const src = S.chat || {};
  return {
    id: uid(),
    title: shortTitle(text),
    private: false,
    server_url: currentServerUrl(),
    model: model || src.model || "",
    num_ctx: src.num_ctx || S.config.default_num_ctx || 4096,
    system_prompt: src.system_prompt || "",
    system_on: !!src.system_on,
    pre_prompt: src.pre_prompt || "",
    pre_on: !!src.pre_on,
    isolated: !!src.isolated,
    hide_thinking: !!src.hide_thinking,
    web_search: !!src.web_search,
    crawl_pages: src.crawl_pages || S.minCrawledPages || 7,
    multi_pass: !!src.multi_pass,
    passes: src.passes || 2,
    pass_use_system: src.pass_use_system !== false,
    eval_prompt: src.eval_prompt || S.defaultEvalPrompt,
    library_ids: (src.library_ids || []).slice(),
    library_strict: !!src.library_strict,
    messages: [{ role: "user", content: text }],
  };
}
function addToQueue() {
  const text = $("input-box").value.trim();
  const dataXml = buildDataXml();
  if (!text && !dataXml) { toast("Type a prompt or add data to queue"); return; }
  const model = getSelectedModel();
  let sq = "";
  if ($("chk-websearch").checked) { sq = $("search-query").value.trim(); $("search-query").value = ""; }
  const content = dataXml ? (text ? dataXml + "\n\n" + text : dataXml) : text;
  const chat = buildChatSnapshot(content, model);
  chat.title = shortTitle(text || ("Data: " + S.dataItems.map((d) => d.label).join(", ")));
  S.queue.push({ item_id: uid(), chat, search_query: sq, title: chat.title });
  $("input-box").value = "";
  clearDataItems();
  updateQueueUI();
  toast(`Queued (${S.queue.length})`);
}
function updateQueueUI() {
  const n = S.queue.length;
  $("btn-queue-run").textContent = `▶ Run Queue (${n})`;
  $("btn-queue-run").classList.toggle("hidden", n === 0);
  $("btn-queue-clear").classList.toggle("hidden", n === 0);
}
function clearQueue() {
  if (!S.queue.length) return;
  S.queue = [];
  updateQueueUI();
  toast("Queue cleared");
}
/** Run the queued prompts — fanned out across servers when parallel processing is
 *  enabled and configured, otherwise one at a time via runQueueSequential(). */
async function runQueue() {
  if (S.generating || S.batchRunning) { toast("Already processing — wait for it to finish."); return; }
  if (!S.queue.length) return;
  if (S.parallel.enabled && (S.parallel.servers || []).length) {
    if (!S.chat) await newPrivateChat(true);
    const items = S.queue.map((q) => ({ item_id: q.item_id, chat: q.chat, search_query: q.search_query, title: q.title }));
    const byId = {}; S.queue.forEach((q) => { byId[q.item_id] = q; });
    S.runId = uid();
    await runParallel("/api/parallel/start", { items, run_id: S.runId }, { queue: byId });
    S.queue = []; updateQueueUI();
    if (!S.chat.private) refreshChatSummaries();
  } else {
    await runQueueSequential();
  }
}
// Fallback when parallel is off: process queued prompts one at a time on the current
// server, stacking each result (user turn + reasoning + answer) into the active chat.
async function runQueueSequential() {
  S.queueStop = false;
  const items = S.queue.slice();
  S.queue = []; updateQueueUI();
  if (!S.chat) await newPrivateChat(true);
  showChatView();
  setStatus(`Queue: 0/${items.length}`);
  for (let i = 0; i < items.length; i++) {
    if (S.queueStop) break;
    setStatus(`Queue ${i + 1}/${items.length}: ${items[i].title}`);
    // Show this prompt's user turn, then generate independently from its snapshot.
    S.chat.messages.push({ role: "user", content: items[i].chat.messages[0].content });
    renderMessages();
    await runGeneration(items[i].search_query || "", { sendChat: items[i].chat });
  }
  if (!S.chat.private) refreshChatSummaries();
  setStatus(S.queueStop ? "Queue stopped" : "Queue complete");
}

// ------------------------------- parallel lanes ----------------------
/** Build one lane column (header, per-item title, body) and return it together with
 *  the mutable streaming state runParallel() tracks for that lane. */
function makeLaneColumn(l) {
  const el = document.createElement("div");
  el.className = "lane-column";
  const head = document.createElement("div");
  head.className = "lane-head busy";
  const dot = document.createElement("span"); dot.className = "lane-dot";
  const title = document.createElement("span"); title.className = "lane-title";
  title.textContent = l.name || l.server || `Lane ${l.index}`;
  const model = document.createElement("span"); model.className = "lane-model";
  model.textContent = l.model ? "· " + l.model : "";
  const count = document.createElement("span"); count.className = "lane-count"; count.textContent = "0 done";
  head.appendChild(dot); head.appendChild(title); head.appendChild(model); head.appendChild(count);
  const itemTitle = document.createElement("div"); itemTitle.className = "lane-item-title";
  const body = document.createElement("div"); body.className = "lane-body";
  el.appendChild(head); el.appendChild(itemTitle); el.appendChild(body);
  return { el, head, body, titleEl: itemTitle, countEl: count, info: l,
           count: 0, curBubble: null, curContent: "", curReason: "", reasonBubble: null };
}
function laneScroll(L) { if (isNearBottom(L.body)) L.body.scrollTop = L.body.scrollHeight; }

/**
 * Stream a multiplexed parallel run into side-by-side lane columns.
 *
 * Every lane's tokens arrive interleaved on ONE SSE stream, tagged with a lane index;
 * this routes each frame to the right column and keeps per-lane bubble state so the
 * partial answers don't bleed into each other.
 *
 * @param {object} opts  `opts.queue` = {item_id: queueItem} stacks each finished item
 *                       into the active chat; `opts.batch` = true means the server
 *                       already wrote the responses to disk, so nothing is stacked.
 */
async function runParallel(path, body, opts = {}) {
  setGeneratingUI(true);
  S.batchRunning = true;      // reuse the guard so Send/Batch stay disabled
  S.runId = body.run_id;
  const lanesBox = $("parallel-lanes");
  lanesBox.innerHTML = "";
  lanesBox.classList.remove("hidden");
  $("messages").classList.add("hidden");
  const lanes = {};
  let doneCount = 0;

  await streamSSE(path, body, {
    start: (d) => {
      (d.lanes || []).forEach((l) => { const col = makeLaneColumn(l); lanesBox.appendChild(col.el); lanes[l.index] = col; });
      setStatus(`Parallel (${d.mode}) — 0/${d.total} across ${(d.lanes || []).length} server(s)`);
    },
    item_start: (d) => {
      const L = lanes[d.lane]; if (!L) return;
      L.head.classList.add("busy"); L.head.classList.remove("idle");
      L.body.innerHTML = ""; L.titleEl.textContent = d.title || "";
      L.curBubble = null; L.curContent = ""; L.curReason = ""; L.reasonBubble = null;
    },
    status: (d) => { const L = lanes[d.lane]; if (L && L.curBubble) L.curBubble._body.textContent = d.message; },
    context: (d) => upsertContextBar(d),
    pass_start: (d) => {
      const L = lanes[d.lane]; if (!L) return;
      if (L.curBubble) L.curBubble.classList.remove("streaming");
      L.curContent = ""; L.curReason = ""; L.reasonBubble = null;
      L.curBubble = makeBubble("assistant", "", { streaming: true, label: d.label, intermediate: d.intermediate });
      L.body.appendChild(L.curBubble);
      laneScroll(L);
    },
    reasoning: (d) => {
      const L = lanes[d.lane]; if (!L || !L.curBubble) return;
      if (!L.reasonBubble) { L.reasonBubble = makeReasoningBubble("", false); L.body.insertBefore(L.reasonBubble.el, L.curBubble); }
      L.curReason += d.content; L.reasonBubble.body.textContent = L.curReason; laneScroll(L);
    },
    chunk: (d) => {
      const L = lanes[d.lane]; if (!L || !L.curBubble) return;
      if (L.curContent === "") L.curBubble._body.textContent = "";
      L.curContent += d.content; L.curBubble._body.textContent = L.curContent; laneScroll(L);
    },
    pass_end: (d) => {
      const L = lanes[d.lane]; if (!L) return;
      if (d.content !== undefined && L.curBubble) { L.curContent = d.content; L.curBubble._body.textContent = d.content; }
      if (L.curBubble) L.curBubble.classList.remove("streaming");
    },
    error: (d) => {
      const L = lanes[d.lane];
      if (!L) { toast("Parallel error: " + d.message); return; }
      if (!L.curBubble) { L.curBubble = makeBubble("assistant", "", {}); L.body.appendChild(L.curBubble); }
      L.curBubble._body.textContent = "[Error] " + d.message; L.curBubble.classList.remove("streaming");
    },
    item_done: (d) => {
      const L = lanes[d.lane];
      if (L) {
        L.head.classList.remove("busy"); L.head.classList.add("idle");
        L.count++; L.countEl.textContent = L.count + " done";
        if (L.curBubble) L.curBubble.classList.remove("streaming");
      }
      doneCount++;
      setStatus(`Parallel — ${doneCount} item(s) done`);
      if (opts.queue) { const item = opts.queue[d.item_id]; if (item) appendQueuedResultToChat(item, d); }
    },
    done: (d) => {
      Object.values(lanes).forEach((L) => { L.head.classList.remove("busy"); L.head.classList.add("idle"); });
      const verb = d.stopped ? "stopped" : "complete";
      setStatus(`Parallel ${verb} — ${doneCount} item(s)`);
      toast(`Parallel ${verb} — ${doneCount} item(s)`, 5000);
      // Ephemeral per-lane bars: clear them once the parallel run ends.
      clearLaneContextBars();
      // Reveal the stacked results in the main chat (hidden during the lane run).
      if (opts.queue) { showChatView(); renderMessages(); }
    },
  });

  S.batchRunning = false;
  setGeneratingUI(false);
}
// Stack a finished queued item (user prompt + answer + reasoning) into the active
// chat. User and assistant are pushed together so each answer stays paired with its
// prompt regardless of the order lanes complete in.
function appendQueuedResultToChat(item, frame) {
  if (!S.chat) return;
  S.chat.messages.push({ role: "user", content: item.chat.messages[0].content });
  const asst = { role: "assistant", content: frame.content || "" };
  if (frame.reasoning) asst.reasoning = frame.reasoning;
  S.chat.messages.push(asst);
  persistChat(true);
}

// ------------------------------- multi-server config -----------------
function renderParallelServers() {
  const box = $("parallel-server-rows");
  if (!box) return;
  box.innerHTML = "";
  const chosen = {};
  (S.parallel.servers || []).forEach((s) => { chosen[(s.base_url || "").replace(/\/+$/, "")] = s.model || ""; });
  S.servers.forEach((sv) => {
    const key = (sv.url || "").replace(/\/+$/, "");
    const row = document.createElement("div"); row.className = "parallel-server-row";
    const lab = document.createElement("label"); lab.className = "chk";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.className = "ps-check"; cb.dataset.url = sv.url;
    cb.checked = key in chosen;
    const nm = document.createElement("span"); nm.className = "ps-name"; nm.textContent = sv.label;
    lab.appendChild(cb); lab.appendChild(nm);
    const sel = document.createElement("select"); sel.className = "ps-model"; sel.dataset.url = sv.url;
    const opt = document.createElement("option"); opt.value = ""; opt.textContent = "(loading models…)"; sel.appendChild(opt);
    row.appendChild(lab); row.appendChild(sel);
    box.appendChild(row);
    loadServerModelsInto(sel, sv.url, chosen[key] || "");
  });
  const mode = S.parallel.mode || "balanced";
  $("pmode-balanced").checked = mode === "balanced";
  $("pmode-isolation").checked = mode === "isolation";
}
async function loadServerModelsInto(sel, url, selected) {
  try {
    const r = await api(`/api/models?server=${encodeURIComponent(url)}`);
    const models = r.models || [];
    sel.innerHTML = "";
    if (!models.length) { const o = document.createElement("option"); o.value = ""; o.textContent = "(no models)"; sel.appendChild(o); return; }
    models.forEach((m) => { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); });
    if (selected && models.includes(selected)) sel.value = selected;
  } catch (e) {
    sel.innerHTML = ""; const o = document.createElement("option"); o.value = ""; o.textContent = "(error)"; sel.appendChild(o);
  }
}
async function useCommonModel() {
  const urls = [];
  document.querySelectorAll(".parallel-server-row .ps-check:checked").forEach((cb) => urls.push(cb.dataset.url));
  if (!urls.length) { toast("Select at least one server first"); return; }
  setStatus("Finding models installed on all selected servers…");
  let models = [];
  try {
    const r = await api(`/api/parallel/common-model?servers=${encodeURIComponent(urls.join(","))}`);
    models = r.models || [];
  } catch (e) { toast("Common-model lookup failed: " + e.message); setStatus(""); return; }
  setStatus("");
  if (!models.length) { toast("No single model is installed on all selected servers."); return; }
  let choice = models[0];
  if (models.length > 1) {
    const picked = await promptModal(`Common model (installed on all selected servers): ${models.join(", ")}`, models[0]);
    if (picked === null) return;
    const p = (picked || "").trim();
    choice = models.includes(p) ? p : models[0];
  }
  document.querySelectorAll(".parallel-server-row").forEach((row) => {
    const cb = row.querySelector(".ps-check"); if (!cb.checked) return;
    const sel = row.querySelector(".ps-model");
    if (![...sel.options].some((o) => o.value === choice)) { const o = document.createElement("option"); o.value = choice; o.textContent = choice; sel.appendChild(o); }
    sel.value = choice;
  });
  toast("Set common model: " + choice);
}
async function saveParallelConfig() {
  const servers = [];
  document.querySelectorAll(".parallel-server-row").forEach((row) => {
    const cb = row.querySelector(".ps-check"); if (!cb.checked) return;
    servers.push({ base_url: cb.dataset.url, model: (row.querySelector(".ps-model").value || "") });
  });
  const mode = $("pmode-isolation").checked ? "isolation" : "balanced";
  try {
    const r = await api("/api/parallel/config", { method: "PUT", body: { parallel_servers: servers, parallel_mode: mode } });
    S.parallel.servers = r.parallel_servers || []; S.parallel.mode = r.parallel_mode || "balanced";
    toast(`Multi-server config saved (${servers.length} server(s), ${S.parallel.mode})`);
  } catch (e) { toast("Save failed: " + e.message); }
}
async function toggleParallel() {
  S.parallel.enabled = $("chk-parallel").checked;
  try { await api("/api/parallel/config", { method: "PUT", body: { parallel_enabled: S.parallel.enabled } }); } catch (e) {}
  if (S.parallel.enabled && !(S.parallel.servers || []).length) {
    toast("Select servers in Settings → Multi-Server Processing to use parallel mode.", 5000);
  }
}

// --------------------- prompt library (System/Pre) -------------------
// A 3-column browser (Group -> Category -> Prompt) over two independent trees,
// S.prompts.system and S.prompts.pre. The whole tree is saved back on every edit.
const PB = { kind: "system", groupId: null, catId: null, promptId: null, pendingText: null };

function promptTree(kind) {
  if (!S.prompts) S.prompts = { system: [], pre: [] };
  if (!Array.isArray(S.prompts[kind])) S.prompts[kind] = [];
  return S.prompts[kind];
}
function pbGroup() { return promptTree(PB.kind).find((g) => g.id === PB.groupId); }
function pbCat() { const g = pbGroup(); return g && (g.categories || []).find((c) => c.id === PB.catId); }
function pbPromptItem() { const c = pbCat(); return c && (c.prompts || []).find((p) => p.id === PB.promptId); }

async function savePrompts() {
  const r = await api("/api/prompts", { method: "POST", body: { prompts: S.prompts } });
  if (r && r.prompts) S.prompts = r.prompts;
}

// promptModal() hijacks the shared modal, so reopen the browser afterward.
async function pbNameModal(title, def) {
  const v = await promptModal(title, def);
  openModal("modal-prompt-browser");
  return v;
}

function pbItemRow(label, active, handlers) {
  const row = document.createElement("div");
  row.className = "pb-item" + (active ? " active" : "");
  const name = document.createElement("span");
  name.className = "pb-item-name";
  name.textContent = label;
  name.title = label;
  name.onclick = handlers.onSelect;
  row.appendChild(name);
  const acts = document.createElement("span");
  acts.className = "pb-item-acts";
  const mk = (txt, title, fn) => {
    const b = document.createElement("button");
    b.className = "pb-act"; b.textContent = txt; b.title = title;
    b.onclick = (e) => { e.stopPropagation(); fn(); };
    return b;
  };
  if (handlers.onExport) acts.appendChild(mk("⇩", "Export XML", handlers.onExport));
  if (handlers.onRename) acts.appendChild(mk("✎", "Rename", handlers.onRename));
  if (handlers.onDelete) acts.appendChild(mk("🗑", "Delete", handlers.onDelete));
  row.appendChild(acts);
  return row;
}

function renderPb() {
  $("pb-title").textContent = PB.kind === "pre" ? "Pre-prompt Library" : "System Prompt Library";
  renderPbGroups(); renderPbCats(); renderPbPrompts(); renderPbEdit();
  $("btn-pb-new-cat").disabled = !pbGroup();
  $("btn-pb-new-prompt").disabled = !pbCat();
}
function renderPbGroups() {
  const box = $("pb-groups"); box.innerHTML = "";
  promptTree(PB.kind).forEach((g) => {
    box.appendChild(pbItemRow(g.name || "(unnamed)", g.id === PB.groupId, {
      onSelect: () => { PB.groupId = g.id; PB.catId = null; PB.promptId = null; renderPb(); },
      onExport: () => pbExport({ group_id: g.id, name: g.name }),
      onRename: () => pbRename("group", g),
      onDelete: () => pbDelete("group", g),
    }));
  });
}
function renderPbCats() {
  const box = $("pb-cats"); box.innerHTML = "";
  const g = pbGroup(); if (!g) return;
  (g.categories || []).forEach((c) => {
    box.appendChild(pbItemRow(c.name || "(unnamed)", c.id === PB.catId, {
      onSelect: () => { PB.catId = c.id; PB.promptId = null; renderPb(); },
      onExport: () => pbExport({ group_id: g.id, category_id: c.id, name: c.name }),
      onRename: () => pbRename("category", c),
      onDelete: () => pbDelete("category", c),
    }));
  });
}
function renderPbPrompts() {
  const box = $("pb-prompts"); box.innerHTML = "";
  const g = pbGroup(), c = pbCat(); if (!c) return;
  (c.prompts || []).forEach((p) => {
    box.appendChild(pbItemRow(p.name || "(unnamed)", p.id === PB.promptId, {
      onSelect: () => { PB.promptId = p.id; renderPbEdit(); highlightPbPrompts(); },
      onExport: () => pbExport({ group_id: g.id, category_id: c.id, prompt_id: p.id, name: p.name }),
      onRename: () => pbRename("prompt", p),
      onDelete: () => pbDelete("prompt", p),
    }));
  });
}
function highlightPbPrompts() {
  const c = pbCat(); if (!c) return;
  const box = $("pb-prompts");
  Array.from(box.children).forEach((row, i) => {
    row.classList.toggle("active", (c.prompts[i] || {}).id === PB.promptId);
  });
}
function renderPbEdit() {
  const p = pbPromptItem();
  const nm = $("pb-edit-name"), tx = $("pb-edit-text");
  nm.disabled = tx.disabled = !p;
  $("btn-pb-use").disabled = !(p && S.chat);
  $("btn-pb-save").disabled = !p;
  nm.value = p ? (p.name || "") : "";
  tx.value = p ? (p.prompt || "") : "";
}

async function pbNewGroup() {
  const name = await pbNameModal("New group name", "New Group");
  if (!name || !name.trim()) return;
  const g = { id: uid(), name: name.trim(), categories: [] };
  promptTree(PB.kind).push(g);
  PB.groupId = g.id; PB.catId = null; PB.promptId = null;
  await savePrompts(); renderPb();
}
async function pbNewCat() {
  const g = pbGroup(); if (!g) { toast("Select a group first"); return; }
  const name = await pbNameModal("New category name", "New Category");
  if (!name || !name.trim()) return;
  const c = { id: uid(), name: name.trim(), prompts: [] };
  g.categories = g.categories || []; g.categories.push(c);
  PB.catId = c.id; PB.promptId = null;
  await savePrompts(); renderPb();
}
async function pbNewPrompt() {
  const c = pbCat(); if (!c) { toast("Select a category first"); return; }
  const name = await pbNameModal("New prompt name", "New Prompt");
  if (!name || !name.trim()) return;
  const text = PB.pendingText != null ? PB.pendingText : "";
  const p = { id: uid(), name: name.trim(), prompt: text };
  c.prompts = c.prompts || []; c.prompts.push(p);
  PB.promptId = p.id; PB.pendingText = null;
  await savePrompts(); renderPb();
}
async function pbRename(level, obj) {
  const name = await pbNameModal("Rename", obj.name || "");
  if (name == null || !name.trim()) return;
  obj.name = name.trim();
  await savePrompts(); renderPb();
}
async function pbDelete(level, obj) {
  if (!confirm(`Delete this ${level} "${obj.name || ""}"?`)) return;
  if (level === "group") {
    S.prompts[PB.kind] = promptTree(PB.kind).filter((g) => g.id !== obj.id);
    if (PB.groupId === obj.id) { PB.groupId = null; PB.catId = null; PB.promptId = null; }
  } else if (level === "category") {
    const g = pbGroup(); if (g) g.categories = (g.categories || []).filter((c) => c.id !== obj.id);
    if (PB.catId === obj.id) { PB.catId = null; PB.promptId = null; }
  } else {
    const c = pbCat(); if (c) c.prompts = (c.prompts || []).filter((p) => p.id !== obj.id);
    if (PB.promptId === obj.id) PB.promptId = null;
  }
  await savePrompts(); renderPb();
}
async function pbSaveEdit() {
  const p = pbPromptItem(); if (!p) return;
  p.name = ($("pb-edit-name").value || "").trim() || p.name;
  p.prompt = $("pb-edit-text").value;
  await savePrompts(); renderPb();
  toast("Saved");
}
function pbUse() {
  const p = pbPromptItem(); if (!p || !S.chat) return;
  if (PB.kind === "pre") {
    $("pre-prompt").value = p.prompt; S.chat.pre_prompt = p.prompt;
    $("pre-on").checked = true; S.chat.pre_on = true;
  } else {
    $("system-prompt").value = p.prompt; S.chat.system_prompt = p.prompt;
    $("system-on").checked = true; S.chat.system_on = true;
  }
  persistChat(); closeModal();
  toast("Applied to chat");
}
async function pbExport(ids) {
  const body = { kind: PB.kind, ...ids, default_name: (ids.name || (PB.kind + "_prompts")) };
  const r = await api("/api/prompts/export-xml", { method: "POST", body });
  if (r && r.ok) toast("Exported to " + r.path);
  else if (r && r.cancelled) { /* user cancelled */ }
  else toast("Export failed" + (r && r.error ? ": " + r.error : ""));
}
async function pbImport() {
  const r = await api("/api/prompts/import-xml", { method: "POST", body: {} });
  if (r && r.prompts) { S.prompts = r.prompts; renderPb(); }
  if (r && r.imported) toast(`Imported ${r.imported} file(s)`);
  if (r && r.errors && r.errors.length) toast("Some files failed: " + r.errors.join("; "));
}

function openPromptBrowser(kind, saveText) {
  if (PB.kind !== kind) { PB.groupId = null; PB.catId = null; PB.promptId = null; }
  PB.kind = kind;
  PB.pendingText = (saveText != null) ? saveText : null;
  renderPb();
  openModal("modal-prompt-browser");
  if (saveText != null) toast("Pick a category, then ＋ on Prompts to save the current text.");
}

// ------------------------------- web search --------------------------
function updateWebsearchVisibility() {
  $("websearch-panel").classList.toggle("hidden", !$("chk-websearch").checked);
}
function updateMultipassVisibility() {
  $("multipass-panel").classList.toggle("hidden", !$("chk-multipass").checked);
}
function populateDomains() {
  const sel = $("domain-list");
  sel.innerHTML = "";
  (S.config.approved_domains || []).forEach((d) => {
    const o = document.createElement("option"); o.value = d; o.textContent = d;
    sel.appendChild(o);
  });
}
async function saveWebsearchConfig() {
  const r = await api("/api/websearch/config", { method: "POST", body: {
    approved_domains: S.config.approved_domains || [],
    restrict_to_approved: $("chk-restrict").checked,
  }});
  S.config = r.config;
}
function addDomain() {
  const v = $("domain-input").value.trim();
  if (!v) return;
  S.config.approved_domains = S.config.approved_domains || [];
  S.config.approved_domains.push(v);
  $("domain-input").value = "";
  populateDomains(); saveWebsearchConfig();
}
function removeDomains() {
  const sel = $("domain-list");
  const chosen = new Set(Array.from(sel.selectedOptions).map((o) => o.value));
  S.config.approved_domains = (S.config.approved_domains || []).filter((d) => !chosen.has(d));
  populateDomains(); saveWebsearchConfig();
}
function clearDomains() {
  S.config.approved_domains = [];
  populateDomains(); saveWebsearchConfig();
}

// ------------------------------- Settings tab ------------------------
function renderSettings() {
  $("brave-state").textContent = S.config.has_brave_token ? "✓ saved" : "not set";
  $("brightdata-state").textContent = S.config.has_brightdata_token ? "✓ saved" : "not set";
  $("set-bd-zone").value = S.config.brightdata_zone || "web_unlocker1";
  const ctxSel = $("set-default-ctx");
  if (!ctxSel.options.length) {
    S.contextLengths.forEach((n) => {
      const o = document.createElement("option"); o.value = n; o.textContent = n.toLocaleString();
      ctxSel.appendChild(o);
    });
  }
  ctxSel.value = S.config.default_num_ctx || 4096;
  $("set-max-output").value = S.config.max_output_tokens || 16000;
  const pcw = S.config.provider_context_windows || {};
  $("set-ctx-openai").value = pcw.openai || 128000;
  $("set-ctx-anthropic").value = pcw.anthropic || 200000;
  $("set-auto-reason").checked = !!S.config.auto_detect_reasoning;
  $("set-rag-embed-url").value = S.config.rag_embed_server_url || "http://127.0.0.1:11434";
  $("set-rag-embed-model").value = S.config.rag_embed_model || "nomic-embed-text";
  $("set-rag-topk").value = S.config.rag_top_k || 6;
  $("set-rag-mode").value = S.config.rag_retrieval_mode || "hybrid";
  $("set-rag-context").checked = !!S.config.rag_contextual_chunking;
  $("set-rag-context-model").value = S.config.rag_context_model || "";
  $("set-rag-qrewrite").checked = S.config.rag_query_rewrite !== false;
  $("set-rewrite-model").value = S.config.rewrite_model || "";
  $("set-mem-influence").value = S.config.memory_weight_influence ?? 0.35;
  $("set-pipeline-retries").value = S.config.pipeline_max_retries ?? 3;
  renderServerRows();
  renderParallelServers();
}
function presetForServer(s) {
  const base = (s.base_url || "").replace(/\/+$/, "");
  let p = S.providerPresets.find((x) => (x.base_url || "").replace(/\/+$/, "") === base && base);
  if (p) return p.key;
  if (s.type === "anthropic") return "anthropic";
  if (s.type === "ollama") return "ollama";
  return "custom";
}
function renderServerRows() {
  const box = $("settings-server-rows");
  box.innerHTML = "";
  (S.rawServers || []).forEach((s) => box.appendChild(makeServerRow(s)));
  if (!(S.rawServers || []).length) box.appendChild(makeServerRow({}));
}
function _ollamaParts(baseUrl) {
  // Parse an Ollama base_url into {ip, port}. Defaults to port 11434.
  try {
    const u = new URL(baseUrl);
    return { ip: u.hostname || "", port: u.port || "11434" };
  } catch (e) { return { ip: "", port: "11434" }; }
}
function makeServerRow(s) {
  const row = document.createElement("div");
  row.className = "provider-row";

  const typeSel = document.createElement("select");
  typeSel.className = "sv-type";
  S.providerPresets.forEach((p) => {
    const o = document.createElement("option"); o.value = p.key; o.textContent = p.label;
    typeSel.appendChild(o);
  });
  // A brand-new (empty) row defaults to Ollama; existing rows map to their preset.
  const isNew = !s || (!s.base_url && !s.type);
  typeSel.value = isNew ? "ollama" : presetForServer(s);

  const name = document.createElement("input");
  name.className = "sv-name"; name.value = s.name || "";

  const fields = document.createElement("span");
  fields.className = "sv-fields";

  const rm = document.createElement("button");
  rm.className = "small"; rm.textContent = "✕"; rm.title = "Remove";
  rm.onclick = () => row.remove();

  const parts = _ollamaParts(s.base_url || "http://127.0.0.1:11434");

  function renderFields() {
    const p = S.providerPresets.find((x) => x.key === typeSel.value) || {};
    fields.innerHTML = "";
    if (p.type === "ollama") {
      name.placeholder = "Name (optional, defaults to IP)";
      const ip = document.createElement("input");
      ip.className = "sv-ip"; ip.placeholder = "IP address, e.g. 192.168.1.50";
      ip.value = isNew ? "" : parts.ip;
      const port = document.createElement("input");
      port.className = "sv-port"; port.type = "number"; port.min = "1"; port.max = "65535";
      port.value = parts.port || "11434"; port.title = "Port";
      fields.appendChild(ip); fields.appendChild(port);
    } else if (p.key === "custom") {
      name.placeholder = "Name (optional)";
      const url = document.createElement("input");
      url.className = "sv-url"; url.placeholder = "base URL (https://…/v1)";
      url.value = (presetForServer(s) === "custom" ? (s.base_url || "") : "");
      const key = document.createElement("input");
      key.className = "sv-key"; key.type = "password"; key.autocomplete = "off";
      key.placeholder = s.has_key ? "•••• saved (blank keeps it)" : "API key (optional)";
      fields.appendChild(url); fields.appendChild(key);
    } else {
      // Known cloud provider — base URL is automatic; only ask for a key if needed.
      name.placeholder = `Name (optional, defaults to "${p.label || ""}")`;
      if (p.needs_key) {
        const key = document.createElement("input");
        key.className = "sv-key"; key.type = "password"; key.autocomplete = "off";
        key.placeholder = s.has_key ? "•••• saved (blank keeps it)" : "API key";
        fields.appendChild(key);
      } else {
        const note = document.createElement("span");
        note.className = "muted"; note.textContent = p.base_url || "(no key needed)";
        fields.appendChild(note);
      }
    }
  }
  typeSel.onchange = renderFields;
  renderFields();

  row.appendChild(typeSel); row.appendChild(name); row.appendChild(fields); row.appendChild(rm);
  return row;
}
async function saveServerList(servers) {
  const r = await api("/api/servers", { method: "PUT", body: { servers } });
  S.servers = r.servers; S.rawServers = r.raw;
  populateServers();
  renderServerRows();
}
async function saveSettingsServers() {
  const servers = [];
  document.querySelectorAll(".provider-row").forEach((row) => {
    const presetKey = row.querySelector(".sv-type").value;
    const preset = S.providerPresets.find((x) => x.key === presetKey) || { type: "ollama" };
    const nameInput = (row.querySelector(".sv-name").value || "").trim();
    const keyEl = row.querySelector(".sv-key");
    const api_key = keyEl ? keyEl.value.trim() : "";

    let base_url = "", name = nameInput;
    if (preset.type === "ollama") {
      const ip = (row.querySelector(".sv-ip").value || "").trim();
      if (!ip) return;                     // skip incomplete Ollama rows
      const port = (row.querySelector(".sv-port").value || "11434").trim() || "11434";
      base_url = `http://${ip}:${port}`;
      name = name || ip;
    } else if (preset.key === "custom") {
      base_url = (row.querySelector(".sv-url").value || "").trim();
      if (!base_url) return;               // skip incomplete Custom rows
      name = name || base_url;
    } else {
      base_url = preset.base_url;
      name = name || preset.label;
    }
    servers.push({ name, type: preset.type, base_url, api_key });
  });
  await saveServerList(servers);
  toast("Servers saved");
}
async function saveApiKeys() {
  const body = { brightdata_zone: $("set-bd-zone").value.trim() || "web_unlocker1" };
  const b = $("set-brave").value.trim(); if (b) body.brave_token = b;
  const d = $("set-brightdata").value.trim(); if (d) body.brightdata_token = d;
  const r = await api("/api/settings", { method: "POST", body });
  S.config = r.config; $("set-brave").value = ""; $("set-brightdata").value = "";
  renderSettings(); toast("Keys saved");
}
async function saveGeneral() {
  const body = {
    default_num_ctx: parseInt($("set-default-ctx").value) || 4096,
    max_output_tokens: parseInt($("set-max-output").value) || 16000,
    provider_context_windows: {
      openai: Math.max(1, parseInt($("set-ctx-openai").value) || 128000),
      anthropic: Math.max(1, parseInt($("set-ctx-anthropic").value) || 200000),
    },
    auto_detect_reasoning: $("set-auto-reason").checked,
    rag_embed_server_url: $("set-rag-embed-url").value.trim() || "http://127.0.0.1:11434",
    rag_embed_model: $("set-rag-embed-model").value.trim() || "nomic-embed-text",
    rag_top_k: Math.max(1, parseInt($("set-rag-topk").value) || 6),
    rag_retrieval_mode: $("set-rag-mode").value || "hybrid",
    rag_contextual_chunking: $("set-rag-context").checked,
    rag_context_model: $("set-rag-context-model").value.trim(),
    rag_query_rewrite: $("set-rag-qrewrite").checked,
    rewrite_model: $("set-rewrite-model").value.trim(),
    memory_weight_influence: parseFloat($("set-mem-influence").value) || 0,
    pipeline_max_retries: parseInt($("set-pipeline-retries").value) || 3,
  };
  const r = await api("/api/settings", { method: "POST", body });
  S.config = { ...S.config, ...r.config };
  toast("Settings saved");
}
async function scanRange() {
  const range = $("scan-range").value.trim();
  if (!range) { toast("Enter an IP range (e.g. 192.168.1.0/24)"); return; }
  $("scan-status").textContent = "Scanning…";
  $("scan-results").innerHTML = "";
  $("btn-add-scanned").classList.add("hidden");
  try {
    const r = await api("/api/servers/scan", { method: "POST", body: {
      range, port: parseInt($("scan-port").value) || 11434,
    }});
    renderScanResults(r.found || []);
    $("scan-status").textContent = `Scanned ${r.scanned} address(es); found ${(r.found || []).length} Ollama server(s).`;
  } catch (e) { $("scan-status").textContent = "Scan error: " + e.message; }
}
function renderScanResults(found) {
  const box = $("scan-results");
  box.innerHTML = "";
  found.forEach((f) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = f.base_url; cb.dataset.name = f.name; cb.checked = true;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(` ${f.base_url} — ${f.model_count} model(s)`));
    box.appendChild(lab);
  });
  $("btn-add-scanned").classList.toggle("hidden", !found.length);
}
async function addScanned() {
  const chosen = [...$("scan-results").querySelectorAll("input:checked")].map((c) => ({
    name: c.dataset.name, type: "ollama", base_url: c.value, api_key: "",
  }));
  if (!chosen.length) { toast("Select at least one server"); return; }
  const merged = (S.rawServers || []).map((s) => ({
    name: s.name, type: s.type, base_url: s.base_url, api_key: "",
  }));
  const urls = new Set(merged.map((s) => s.base_url.replace(/\/+$/, "")));
  chosen.forEach((c) => { if (!urls.has(c.base_url.replace(/\/+$/, ""))) merged.push(c); });
  await saveServerList(merged);
  $("scan-results").innerHTML = ""; $("btn-add-scanned").classList.add("hidden");
  $("scan-status").textContent = "";
  toast("Added to servers");
}

// ------------------------------- Compile Data (RAG) ------------------
// A library/persona must be "compiled" (semantic-chunked + embedded into the vector
// store) before chat can retrieve over it. These helpers drive the Compile button,
// the Compiled/Stale/Not-compiled badge, and the live SSE progress log.
function compileBadge(st) {
  if (!st || !st.state) return "";
  if (st.state === "compiled")
    return `<span class="cbadge ok" title="${st.chunks || 0} chunks · ${escapeHtml(st.embedding_model || "")} · ${escapeHtml(st.chunker || "")}">Compiled ✓ ${st.chunks || 0} chunks</span>`;
  if (st.state === "stale")
    return `<span class="cbadge stale" title="${escapeHtml((st.reasons || []).join("; "))}">Stale — recompile</span>`;
  return `<span class="cbadge none">Not compiled</span>`;
}
async function refreshCompileStatus(kind, id, badgeEl) {
  if (!badgeEl || !id) return null;
  try {
    const base = kind === "library" ? `/api/libraries/${id}` : `/api/personas/${id}`;
    const st = await api(`${base}/compile-status`);
    badgeEl.innerHTML = compileBadge(st);
    return st;
  } catch (e) { badgeEl.innerHTML = ""; return null; }
}
async function runCompile(kind, id, opts, progressEl, badgeEl) {
  if (!id) { toast("Nothing to compile yet."); return; }
  opts = opts || {};
  const base = kind === "library" ? `/api/libraries/${id}` : `/api/personas/${id}`;
  progressEl.classList.remove("hidden");
  progressEl.innerHTML = `<div class="compile-bar"><div class="compile-bar-fill"></div></div><div class="compile-log"></div>`;
  const fill = progressEl.querySelector(".compile-bar-fill");
  const log = progressEl.querySelector(".compile-log");
  if (badgeEl) badgeEl.innerHTML = `<span class="cbadge working">Compiling…</span>`;
  let total = 0, done = 0;
  const line = (msg) => { const d = document.createElement("div"); d.textContent = msg; log.appendChild(d); log.scrollTop = log.scrollHeight; };
  await streamSSE(`${base}/compile`, { force: !!opts.force }, {
    begin: (d) => { total = d.total || 0; line(`Compiling ${d.name || ""} — ${total} item(s)…`); },
    item_start: (d) => { line(`• ${d.name}…`); },
    item_done: (d) => {
      done++; if (total) fill.style.width = Math.round((done / total) * 100) + "%";
      line(`   ${d.skipped ? "skipped (unchanged)" : "embedded " + (d.chunks || 0) + " chunk(s)"}: ${d.name}`);
    },
    warn: (d) => line(`   ⚠ ${d.name}: ${d.message}`),
    compiled: (d) => {
      fill.style.width = "100%";
      line(`Done — ${d.embedded} embedded, ${d.skipped} skipped, ${d.chunks} chunks total.`);
      toast(`Compiled: ${d.chunks} chunks (${d.embedded} embedded, ${d.skipped} skipped).`);
    },
    error: (d) => { line(`Error: ${d.message}`); toast("Compile error: " + d.message); },
    done: () => {},
  });
  if (badgeEl) await refreshCompileStatus(kind, id, badgeEl);
}

// ------------------------------- library selector --------------------
function updateLibraryButton() {
  const ids = (S.chat && S.chat.library_ids) || [];
  const names = S.libraries.filter((l) => ids.includes(l.id)).map((l) => l.name);
  $("btn-library").textContent = "Library: " + (names.length ? names.join(", ") : "none");
}
function openLibrarySelector() {
  const box = $("libselect-list");
  box.innerHTML = "";
  $("libselect-compile-progress").classList.add("hidden");
  const ids = (S.chat && S.chat.library_ids) || [];
  if (!S.libraries.length) {
    box.innerHTML = "<p class='muted'>No libraries yet. Create some in the Resources tab.</p>";
  }
  S.libraries.forEach((l) => {
    const lab = document.createElement("label"); lab.className = "libselect-row";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = l.id; cb.checked = ids.includes(l.id);
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(` ${l.name} (${(l.items || []).length} item(s)) `));
    const badge = document.createElement("span"); badge.className = "compile-badge-slot";
    lab.appendChild(badge);
    const btn = document.createElement("button");
    btn.className = "small"; btn.textContent = "⚙ Compile"; btn.style.marginLeft = "6px";
    btn.onclick = async (e) => {
      e.preventDefault();
      await runCompile("library", l.id, { force: false }, $("libselect-compile-progress"), badge);
    };
    lab.appendChild(btn);
    box.appendChild(lab);
    refreshCompileStatus("library", l.id, badge);
  });
  openModal("modal-libselect");
}
async function saveLibrarySelection() {
  const ids = Array.from($("libselect-list").querySelectorAll("input:checked")).map((c) => c.value);
  if (S.chat) { S.chat.library_ids = ids; updateLibraryButton(); await persistChat(true); }
  closeModal();
}

// ------------------------------- Resources tab -----------------------
function renderLibraryList() {
  const sel = $("lib-list");
  sel.innerHTML = "";
  S.libraries.forEach((l) => {
    const o = document.createElement("option");
    o.value = l.id; o.textContent = l.name;
    sel.appendChild(o);
  });
  if (S.activeLibrary && S.libraries.find((l) => l.id === S.activeLibrary.id)) {
    sel.value = S.activeLibrary.id;
  }
}
function selectLibrary(id) {
  S.activeLibrary = S.libraries.find((l) => l.id === id) || null;
  renderLibraryEditor();
}
function renderLibraryEditor() {
  const lib = S.activeLibrary;
  $("lib-name").value = lib ? lib.name : "";
  const box = $("lib-items");
  box.innerHTML = "";
  $("lib-compile-progress").classList.add("hidden");
  const badge = $("lib-compile-badge");
  if (badge) badge.innerHTML = "";
  if (!lib) return;
  (lib.items || []).forEach((it, idx) => box.appendChild(makeLibItem(it, idx)));
  refreshCompileStatus("library", lib.id, badge);
}
function makeLibItem(it, idx) {
  const wrap = document.createElement("div");
  wrap.className = "lib-item";
  const head = document.createElement("div");
  head.className = "item-head";
  const tag = document.createElement("span");
  const kind = it.type === "file" ? "file" : (it.type === "url" ? "url" : "write");
  tag.className = "tag " + kind;
  tag.textContent = kind.toUpperCase();
  const label = document.createElement("input");
  label.className = "item-label"; label.type = "text";
  label.placeholder = "Label"; label.value = it.label || it.filename || "";
  label.oninput = () => { it.label = label.value; saveLibrary(); };
  const rm = document.createElement("button");
  rm.className = "small"; rm.textContent = "Remove";
  rm.onclick = () => { S.activeLibrary.items.splice(idx, 1); renderLibraryEditor(); saveLibrary(true); };
  head.appendChild(tag); head.appendChild(label); head.appendChild(rm);
  wrap.appendChild(head);
  // Show the scraped source link (URL items store the source in `filename`).
  if (it.type === "url" && it.filename) {
    const src = document.createElement("a");
    src.className = "item-source"; src.href = it.filename; src.textContent = it.filename;
    src.target = "_blank"; src.rel = "noopener noreferrer";
    wrap.appendChild(src);
  }
  const ta = document.createElement("textarea");
  ta.value = it.content || "";
  ta.oninput = () => { it.content = ta.value; saveLibrary(); };
  wrap.appendChild(ta);
  return wrap;
}
let libSaveTimer = null;
async function saveLibrary(immediate) {
  if (!S.activeLibrary) return;
  S.activeLibrary.name = $("lib-name").value;
  clearTimeout(libSaveTimer);
  const doIt = async () => {
    try {
      const r = await api(`/api/libraries/${S.activeLibrary.id}`, { method: "PUT", body: { library: S.activeLibrary }});
      // Backfill the server-assigned item ids into the LIVE objects (without re-rendering,
      // so typing isn't disrupted). Otherwise every save would re-send id-less items and
      // the store would mint fresh ids each time, needlessly staling the compiled index.
      (r.library.items || []).forEach((it, i) => {
        if (S.activeLibrary.items[i] && !S.activeLibrary.items[i].id) S.activeLibrary.items[i].id = it.id;
      });
      S.libraries = S.libraries.map((l) => (l.id === r.library.id ? r.library : l));
      renderLibraryList();
      updateLibraryButton();
      // Content changed → the compiled index may be stale now; reflect it in the badge.
      refreshCompileStatus("library", S.activeLibrary.id, $("lib-compile-badge"));
    } catch (e) {}
  };
  if (immediate) return doIt();
  libSaveTimer = setTimeout(doIt, 400);
}
async function newLibrary() {
  const r = await api("/api/libraries", { method: "POST", body: { name: "New Library" }});
  S.libraries = r.libraries; renderLibraryList();
  selectLibrary(r.library.id); $("lib-list").value = r.library.id;
}
async function removeLibrary() {
  if (!S.activeLibrary) return;
  if (!confirm(`Remove library "${S.activeLibrary.name}"?`)) return;
  const id = S.activeLibrary.id;
  const r = await api(`/api/libraries/${id}`, { method: "DELETE" });
  S.libraries = r.libraries; S.activeLibrary = null;
  renderLibraryList(); renderLibraryEditor(); updateLibraryButton();
}
async function addWriteIn() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  S.activeLibrary.items = S.activeLibrary.items || [];
  S.activeLibrary.items.push({ type: "write", label: "", content: "", filename: "" });
  renderLibraryEditor(); saveLibrary(true);
}
async function addTextFiles() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  setStatus("Waiting for file selection…");
  try {
    const r = await api(`/api/libraries/${S.activeLibrary.id}/add-text-files`, { method: "POST" });
    S.activeLibrary = r.library;
    S.libraries = S.libraries.map((l) => (l.id === r.library.id ? r.library : l));
    renderLibraryEditor();
    if (r.added.length) toast(`Added ${r.added.length} file(s)`);
    if (r.errors.length) toast("Some files failed: " + r.errors.join("; "));
  } catch (e) { toast("Add files failed: " + e.message); }
  setStatus("");
}
function showLibPanel(which) {
  $("lib-url-panel").classList.toggle("hidden", which !== "url");
  $("lib-search-panel").classList.toggle("hidden", which !== "search");
  if (which === "url") { $("lib-url-input").value = ""; $("lib-url-input").focus(); }
  if (which === "search") {
    $("lib-search-query").value = ""; $("lib-search-sites").value = "";
    const prog = $("lib-search-progress"); prog.classList.add("hidden"); prog.textContent = "";
    $("lib-search-query").focus();
  }
}
function hideLibPanels() {
  $("lib-url-panel").classList.add("hidden");
  $("lib-search-panel").classList.add("hidden");
}
async function addByUrl() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const url = $("lib-url-input").value.trim();
  if (!url) { toast("Enter a URL"); return; }
  const btn = $("btn-lib-url-fetch");
  btn.disabled = true; setStatus("Fetching " + url + " …");
  try {
    const r = await api(`/api/libraries/${S.activeLibrary.id}/add-url`, { method: "POST", body: { url }});
    S.activeLibrary = r.library;
    S.libraries = S.libraries.map((l) => (l.id === r.library.id ? r.library : l));
    renderLibraryEditor(); updateLibraryButton();
    refreshCompileStatus("library", S.activeLibrary.id, $("lib-compile-badge"));
    if (r.added && r.added.length) {
      toast(`Added: ${r.added.join(", ")}${r.via ? " (via " + r.via + ")" : ""}`);
      hideLibPanels();
    }
    if (r.errors && r.errors.length) toast("Fetch failed: " + r.errors.join("; "));
  } catch (e) { toast("Add URL failed: " + e.message); }
  btn.disabled = false; setStatus("");
}
let libSearchES = null;
function braveSearch() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const q = $("lib-search-query").value.trim();
  if (!q) { toast("Enter a search term"); return; }
  const sites = $("lib-search-sites").value.trim();
  const max = Math.max(1, parseInt($("lib-search-max").value, 10) || 5);
  const prog = $("lib-search-progress");
  prog.classList.remove("hidden"); prog.textContent = "Searching Brave…";
  const goBtn = $("btn-lib-search-go"); goBtn.disabled = true;
  if (libSearchES) { libSearchES.close(); libSearchES = null; }
  const qs = new URLSearchParams({ q, sites, max: String(max) }).toString();
  const es = new EventSource(`/api/libraries/${S.activeLibrary.id}/brave-search?${qs}`);
  libSearchES = es;
  const finish = () => { es.close(); if (libSearchES === es) libSearchES = null; goBtn.disabled = false; };
  es.addEventListener("progress", (ev) => {
    const d = JSON.parse(ev.data);
    prog.textContent = `Crawled ${d.done}/${d.target} — ${d.ok ? "✓" : "✗"} ${d.title || d.url}`;
  });
  es.addEventListener("complete", (ev) => {
    const d = JSON.parse(ev.data);
    S.activeLibrary = d.library;
    S.libraries = S.libraries.map((l) => (l.id === d.library.id ? d.library : l));
    renderLibraryEditor(); updateLibraryButton();
    refreshCompileStatus("library", S.activeLibrary.id, $("lib-compile-badge"));
    toast(`Added ${d.added.length} page(s) from ${d.attempted} result(s)` +
          (d.errors.length ? `, ${d.errors.length} skipped` : ""));
    finish(); hideLibPanels();
  });
  es.addEventListener("error", (ev) => {
    // SSE 'error' fires both on our emitted error frame and on normal stream close.
    if (ev.data) { try { toast("Search error: " + (JSON.parse(ev.data).message || "unknown")); prog.textContent = "Search failed."; } catch (e) {} }
    finish();
  });
}
async function loadLibraryXML() {
  setStatus("Waiting for XML selection…");
  try {
    const r = await api("/api/libraries/import-xml", { method: "POST" });
    S.libraries = r.libraries; renderLibraryList();
    if (r.imported.length) { selectLibrary(r.imported[0].id); $("lib-list").value = r.imported[0].id; toast(`Imported ${r.imported.length} library(ies)`); }
    if (r.errors.length) toast("Some XML failed: " + r.errors.join("; "));
  } catch (e) { toast("Load XML failed: " + e.message); }
  setStatus("");
}
async function saveLibraryXML() {
  if (!S.activeLibrary) { toast("Select a library first"); return; }
  await saveLibrary(true);
  setStatus("Waiting for save location…");
  try {
    const r = await api(`/api/libraries/${S.activeLibrary.id}/export-xml`, { method: "POST", body: { library: S.activeLibrary }});
    if (r.ok) toast("Saved: " + r.path);
    else if (!r.cancelled) toast("Save failed: " + (r.error || "unknown"));
  } catch (e) { toast("Save XML failed: " + e.message); }
  setStatus("");
}

// ===================================================================
//  Database Processing tab
//  Phase 1: encrypted connection vault + connection profiles.
//  Import / staging / processing / write-back arrive in later phases.
// ===================================================================
function initDatabaseTab() {
  if (!S.db.inited) {
    dbBindEvents();
    S.db.inited = true;
  }
  dbLoadState();
}

async function dbLoadState() {
  try {
    const st = await api("/api/db/state");
    S.db.vault = st.vault || { exists: false, unlocked: false };
    S.db.sessions = st.projects || [];
    dbRenderVault();
    dbRenderSessions();
    if (S.db.vault.unlocked) dbLoadProfiles();
  } catch (e) {
    toast("Database state: " + e.message);
  }
}

function dbRenderVault() {
  const v = S.db.vault;
  const locked = $("db-vault-locked"), unlocked = $("db-vault-unlocked");
  locked.classList.toggle("hidden", !!v.unlocked);
  unlocked.classList.toggle("hidden", !v.unlocked);
  $("db-conn-card").classList.toggle("hidden", !v.unlocked);
  $("db-sessions-card").classList.toggle("hidden", !v.unlocked);
  // First run vs returning user copy.
  $("db-vault-hint").textContent = v.exists
    ? "" : "First time — choose a master password to create the vault.";
  $("db-vault-pw-label").textContent = v.exists ? "Master password" : "Create master password";
}

async function dbUnlock() {
  const pw = $("db-vault-pw").value;
  if (!pw) { toast("Enter a master password."); return; }
  try {
    const r = await api("/api/db/vault/unlock", { method: "POST", body: { password: pw } });
    S.db.vault = r.vault;
    $("db-vault-pw").value = "";
    dbRenderVault();
    dbLoadProfiles();
    toast("Vault unlocked.");
  } catch (e) {
    toast(e.message);
  }
}

async function dbLock() {
  try {
    const r = await api("/api/db/vault/lock", { method: "POST" });
    S.db.vault = r.vault;
    S.db.profiles = [];
    dbRenderVault();
  } catch (e) { toast(e.message); }
}

async function dbLoadProfiles() {
  try {
    const r = await api("/api/db/profiles");
    S.db.profiles = r.profiles || [];
    dbRenderProfiles();
  } catch (e) {
    if (e.message) toast(e.message);
  }
}

function dbRenderProfiles() {
  const box = $("db-profile-list");
  box.innerHTML = "";
  if (!S.db.profiles.length) {
    box.innerHTML = '<div class="muted">No connections yet.</div>';
    return;
  }
  for (const p of S.db.profiles) {
    const row = document.createElement("div");
    row.className = "db-profile-row";
    const loc = p.url || [p.host, p.database].filter(Boolean).join(" / ") || p.database || "";
    row.innerHTML =
      `<span class="db-profile-name">${escapeHtml(p.name || "(unnamed)")}</span>` +
      `<span class="db-profile-engine">${escapeHtml(p.engine)}</span>` +
      `<span class="muted">${escapeHtml(loc)}</span>`;
    const edit = document.createElement("button");
    edit.className = "small"; edit.textContent = "Edit";
    edit.onclick = () => dbShowProfileForm(p);
    const del = document.createElement("button");
    del.className = "ghost"; del.textContent = "🗑";
    del.onclick = () => dbDeleteProfile(p.id);
    row.append(edit, del);
    box.appendChild(row);
  }
}

function dbShowProfileForm(p) {
  p = p || {};
  $("dbf-id").value = p.id || "";
  $("dbf-name").value = p.name || "";
  $("dbf-engine").value = p.engine || "sqlite";
  $("dbf-url").value = p.url || "";
  $("dbf-host").value = p.host || "";
  $("dbf-port").value = p.port || "";
  $("dbf-database").value = p.database || "";
  $("dbf-username").value = p.username || "";
  $("dbf-password").value = "";
  $("dbf-ssl").checked = !!p.ssl_enabled;
  $("dbf-ssl-ca").value = p.ssl_ca || "";
  $("dbf-ssl-cert").value = p.ssl_cert || "";
  $("dbf-ssl-key").value = "";
  $("dbf-status").textContent = "";
  dbSyncEngineFields();
  $("dbf-ssl-fields").classList.toggle("hidden", !$("dbf-ssl").checked);
  $("db-profile-form").classList.remove("hidden");
}

function dbSyncEngineFields() {
  const engine = $("dbf-engine").value;
  const isSqlite = engine === "sqlite";
  // SQLite only needs the file path (in the Database field); hide host/port/user/pass.
  ["dbf-host-row", "dbf-port-row", "dbf-user-row", "dbf-pass-row"].forEach((id) =>
    $(id).classList.toggle("hidden", isSqlite));
  $("dbf-db-label").textContent = isSqlite ? "File path" : "Database";
}

function dbProfileFromForm() {
  const prof = {
    id: $("dbf-id").value || "",
    name: $("dbf-name").value.trim(),
    engine: $("dbf-engine").value,
    url: $("dbf-url").value.trim(),
    host: $("dbf-host").value.trim(),
    port: $("dbf-port").value ? parseInt($("dbf-port").value, 10) : null,
    database: $("dbf-database").value.trim(),
    username: $("dbf-username").value.trim(),
    ssl_enabled: $("dbf-ssl").checked,
    ssl_ca: $("dbf-ssl-ca").value.trim(),
    ssl_cert: $("dbf-ssl-cert").value.trim(),
  };
  // Only send secrets when the user typed something (blank keeps the stored value).
  if ($("dbf-password").value) prof.password = $("dbf-password").value;
  if ($("dbf-ssl-key").value) prof.ssl_key = $("dbf-ssl-key").value.trim();
  return prof;
}

async function dbSaveProfile() {
  try {
    const r = await api("/api/db/profiles", { method: "POST", body: { profile: dbProfileFromForm() } });
    $("db-profile-form").classList.add("hidden");
    await dbLoadProfiles();
    toast("Connection saved.");
    return r.profile;
  } catch (e) { toast(e.message); }
}

async function dbDeleteProfile(id) {
  if (!confirm("Delete this connection?")) return;
  try {
    await api(`/api/db/profiles/${id}`, { method: "DELETE" });
    await dbLoadProfiles();
  } catch (e) { toast(e.message); }
}

async function dbTestConnection() {
  $("dbf-status").textContent = "Testing…";
  try {
    const r = await api("/api/db/test-connection", { method: "POST", body: { profile: dbProfileFromForm() } });
    $("dbf-status").textContent = r.ok ? `✓ ${r.message || "Connected"}` : `✗ ${r.message || "Failed"}`;
  } catch (e) {
    $("dbf-status").textContent = "✗ " + e.message;
  }
}

function dbRenderSessions() {
  const box = $("db-session-list");
  if (!box) return;
  box.innerHTML = "";
  if (!S.db.sessions.length) {
    box.innerHTML = '<div class="muted">No import sessions yet.</div>';
    return;
  }
  for (const s of S.db.sessions) {
    const row = document.createElement("div");
    row.className = "db-session-row";
    row.innerHTML =
      `<span class="db-session-name">${escapeHtml(s.name || "(untitled)")}</span>` +
      `<span class="muted">${escapeHtml(s.table || "")}</span>` +
      `<span class="db-session-status">${escapeHtml(s.status || "")}</span>`;
    const open = document.createElement("button");
    open.className = "small"; open.textContent = "Open";
    open.onclick = () => dbOpenSession(s.id);
    const del = document.createElement("button");
    del.className = "ghost"; del.textContent = "🗑";
    del.onclick = () => dbDeleteSession(s.id);
    row.append(open, del);
    box.appendChild(row);
  }
}

// ---------------------------- import flow ----------------------------
function dbNewImport() {
  const sel = $("dbi-profile");
  sel.innerHTML = "";
  for (const p of S.db.profiles) {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = `${p.name} (${p.engine})`;
    sel.appendChild(o);
  }
  if (!S.db.profiles.length) { toast("Add a connection first."); return; }
  $("dbi-table").innerHTML = "";
  $("dbi-tables-status").textContent = "";
  $("dbi-progress").textContent = "";
  dbSyncImportMode();
  $("db-import-form").classList.remove("hidden");
}

function dbSyncImportMode() {
  const mode = $("dbi-mode").value;
  $("dbi-n-row").classList.toggle("hidden", mode === "full" || mode === "custom");
  $("dbi-where-row").classList.toggle("hidden", mode !== "custom");
}

async function dbLoadTables() {
  const pid = $("dbi-profile").value;
  if (!pid) return;
  $("dbi-tables-status").textContent = "Loading…";
  try {
    const r = await api("/api/db/tables", { method: "POST", body: { profile_id: pid } });
    const sel = $("dbi-table");
    sel.innerHTML = "";
    for (const t of r.tables) {
      const o = document.createElement("option"); o.value = t; o.textContent = t; sel.appendChild(o);
    }
    $("dbi-tables-status").textContent = `${r.tables.length} table(s)`;
  } catch (e) {
    $("dbi-tables-status").textContent = "✗ " + e.message;
  }
}

function dbRunImport() {
  const pid = $("dbi-profile").value;
  const table = $("dbi-table").value;
  if (!pid || !table) { toast("Pick a connection and table."); return; }
  const mode = $("dbi-mode").value;
  const body = {
    profile_id: pid, table,
    name: $("dbi-name").value.trim() || table,
    selection: {
      mode,
      n: parseInt($("dbi-n").value || "100", 10),
      where: $("dbi-where").value.trim(),
      order_by: $("dbi-order").value.trim(),
    },
    run_id: uid(),
  };
  $("dbi-progress").textContent = "Starting…";
  streamSSE("/api/db/import", body, {
    status: (d) => { $("dbi-progress").textContent = d.message || ""; },
    progress: (d) => { $("dbi-progress").textContent = `Imported ${d.done} / ${d.total}…`; },
    done: (d) => {
      $("dbi-progress").textContent = `Done — ${d.row_count} rows staged.`;
      $("db-import-form").classList.add("hidden");
      dbLoadState();
      if (d.session_id) dbOpenSession(d.session_id);
    },
    error: (d) => { $("dbi-progress").textContent = "✗ " + (d.message || "error"); },
  });
}

// ---------------------------- staging grid ----------------------------
async function dbOpenSession(sessionId, offset) {
  const g = S.db.grid;
  g.sessionId = sessionId;
  if (offset != null) g.offset = offset;
  else g.offset = g.offset || 0;
  g.limit = g.limit || 50;
  try {
    const page = await api(`/api/db/session/${sessionId}/rows?offset=${g.offset}&limit=${g.limit}`);
    g.total = page.total; g.columns = page.columns;
    const sess = S.db.sessions.find((s) => s.id === sessionId);
    $("db-grid-title").textContent = sess ? sess.name : "Staging";
    dbRenderGrid(page);
    $("db-grid-wrap").classList.remove("hidden");
  } catch (e) { toast(e.message); }
}

function dbRenderGrid(page) {
  const box = $("db-grid");
  const t = document.createElement("table");
  t.className = "db-grid-table";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  htr.appendChild(Object.assign(document.createElement("th"), { textContent: "#" }));
  for (const col of page.columns) {
    htr.appendChild(Object.assign(document.createElement("th"), { textContent: col }));
  }
  thead.appendChild(htr); t.appendChild(thead);
  const tb = document.createElement("tbody");
  for (const row of page.rows) {
    const tr = document.createElement("tr");
    if (row.__dirty) tr.classList.add("db-row-dirty");
    const idc = document.createElement("td");
    idc.className = "db-rowid"; idc.textContent = row.__rowid;
    tr.appendChild(idc);
    for (const col of page.columns) {
      const td = document.createElement("td");
      const inp = document.createElement("input");
      inp.className = "db-cell"; inp.value = row[col] == null ? "" : row[col];
      inp.onchange = () => dbEditCell(row.__rowid, col, inp.value);
      td.appendChild(inp); tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  box.innerHTML = ""; box.appendChild(t);
  const g = S.db.grid;
  const from = g.total ? g.offset + 1 : 0;
  const to = Math.min(g.offset + g.limit, g.total);
  $("db-grid-pageinfo").textContent = `${from}–${to} of ${g.total}`;
  $("db-grid-prev").disabled = g.offset <= 0;
  $("db-grid-next").disabled = to >= g.total;
}

async function dbEditCell(rowid, column, value) {
  try {
    await api(`/api/db/session/${S.db.grid.sessionId}/cell`,
      { method: "PUT", body: { rowid, column, value } });
  } catch (e) { toast(e.message); }
}

async function dbAddColumn() {
  const name = prompt("New column name:");
  if (!name) return;
  try {
    await api(`/api/db/session/${S.db.grid.sessionId}/columns`,
      { method: "POST", body: { name: name.trim(), ctype: "output" } });
    dbOpenSession(S.db.grid.sessionId, S.db.grid.offset);
  } catch (e) { toast(e.message); }
}

async function dbDeleteSession(id) {
  if (!confirm("Delete this import session and its staged copy?")) return;
  try {
    await api(`/api/db/session/${id}`, { method: "DELETE" });
    if (S.db.grid.sessionId === id) $("db-grid-wrap").classList.add("hidden");
    dbLoadState();
  } catch (e) { toast(e.message); }
}

// ---------------------------- AI processing ----------------------------
function dbShowProcess() {
  const sel = $("dbp-server");
  sel.innerHTML = "";
  for (const s of S.servers) {
    const o = document.createElement("option");
    o.value = s.url; o.textContent = s.label; sel.appendChild(o);
  }
  if ($("dbp-cols").children.length === 0) dbAddProcCol();
  $("db-process-panel").classList.remove("hidden");
}

function dbAddProcCol() {
  const wrap = document.createElement("div");
  wrap.className = "dbp-col";
  wrap.innerHTML =
    `<div class="control-row">` +
    `<input type="text" class="dbp-name" placeholder="new column name" />` +
    `<select class="dbp-type"><option value="prompt">Prompt (LLM)</option>` +
    `<option value="web_source">Web source</option></select>` +
    `<button class="ghost dbp-del">✕</button></div>` +
    `<textarea class="dbp-tmpl" rows="2" placeholder="Prompt template with {Column} placeholders"></textarea>` +
    `<input type="text" class="dbp-query hidden" placeholder="web search query, e.g. reviews of {name}" />`;
  wrap.querySelector(".dbp-del").onclick = () => wrap.remove();
  wrap.querySelector(".dbp-type").onchange = (e) => {
    const isWeb = e.target.value === "web_source";
    wrap.querySelector(".dbp-tmpl").classList.toggle("hidden", isWeb);
    wrap.querySelector(".dbp-query").classList.toggle("hidden", !isWeb);
  };
  $("dbp-cols").appendChild(wrap);
}

function dbProcColumns() {
  const cols = [];
  for (const w of $("dbp-cols").querySelectorAll(".dbp-col")) {
    const name = w.querySelector(".dbp-name").value.trim();
    if (!name) continue;
    const ctype = w.querySelector(".dbp-type").value;
    cols.push({
      name, ctype, duckdb_type: "VARCHAR",
      prompt_template: w.querySelector(".dbp-tmpl").value,
      search_query: w.querySelector(".dbp-query").value,
    });
  }
  return cols;
}

function dbRunProcess() {
  const model = $("dbp-model").value.trim();
  if (!model) { toast("Enter a model name."); return; }
  const columns = dbProcColumns();
  if (!columns.length) { toast("Add at least one AI column."); return; }
  const body = { server_url: $("dbp-server").value, model, columns, run_id: uid() };
  S.db.processRunId = body.run_id;
  $("dbp-progress").textContent = "Starting…";
  streamSSE(`/api/db/session/${S.db.grid.sessionId}/process`, body, {
    status: (d) => { $("dbp-progress").textContent = d.message || ""; },
    progress: (d) => { $("dbp-progress").textContent = `Row ${d.done} / ${d.total}…`; },
    cell_done: () => {},
    error: (d) => { $("dbp-progress").textContent = "✗ " + (d.message || "error"); },
    done: (d) => {
      $("dbp-progress").textContent = d.stopped ? "Stopped." : "Done.";
      dbOpenSession(S.db.grid.sessionId, S.db.grid.offset);
    },
  });
}

// ---------------------------- write-back ----------------------------
async function dbDryRun() {
  const pre = $("dbw-preview");
  pre.textContent = "Building preview…";
  try {
    const r = await api(`/api/db/session/${S.db.grid.sessionId}/dry-run`, {
      method: "POST", body: { max_rows: parseInt($("dbw-maxrows").value || "50", 10),
                              max_cols: parseInt($("dbw-maxcols").value || "20", 10) } });
    let out = `${r.total_statements} statement(s)` + (r.truncated ? " (truncated)" : "") + "\n";
    if (r.skipped_columns.length) out += `Skipped (not in source): ${r.skipped_columns.join(", ")}\n`;
    if (r.warnings.length) out += "⚠ " + r.warnings.join("\n⚠ ") + "\n";
    out += "\n" + r.statements.map((s) => s.literal).join("\n");
    pre.textContent = out;
  } catch (e) { pre.textContent = "✗ " + e.message; }
}

async function dbCheckConflicts() {
  const pre = $("dbw-preview");
  pre.textContent = "Checking source for changes…";
  try {
    const r = await api(`/api/db/session/${S.db.grid.sessionId}/check-conflicts`, { method: "POST", body: {} });
    let out = r.has_conflict ? "⚠ Source has changed since import.\n" : "✓ No conflicting changes detected.\n";
    out += `changed: ${r.changed.length}, new: ${r.new.length}, deleted: ${r.deleted.length}, unchanged: ${r.unchanged}\n`;
    if (r.note) out += "\n" + r.note + "\n";
    if (r.changed.length) out += "\nChanged keys:\n" + r.changed.slice(0, 50).join("\n");
    pre.textContent = out;
  } catch (e) { pre.textContent = "✗ " + e.message; }
}

function dbWriteBack() {
  if (!$("dbw-approve").checked) { toast("Tick the approval box to write to the source."); return; }
  const body = {
    mode: $("dbw-mode").value, on_conflict: $("dbw-conflict").value,
    approved: true, run_id: uid(),
  };
  $("dbw-progress").textContent = "Writing…";
  streamSSE(`/api/db/session/${S.db.grid.sessionId}/writeback`, body, {
    guard: (d) => { if (d.has_conflict) $("dbw-preview").textContent =
      `⚠ Conflict: changed ${d.changed.length}, deleted ${d.deleted.length}. on_conflict=${d.on_conflict}`; },
    progress: (d) => { $("dbw-progress").textContent = `${d.done} / ${d.total}…`; },
    error: (d) => { $("dbw-progress").textContent = "✗ " + (d.message || "error"); },
    done: (d) => {
      $("dbw-progress").textContent =
        `Done — ${d.applied_rows} row(s), ${d.applied_cells} cell(s) written` +
        (d.skipped_columns && d.skipped_columns.length ? `; skipped ${d.skipped_columns.join(", ")}` : "") + ".";
      dbLoadState();
    },
  });
}

async function dbViewAudit() {
  const pre = $("dbw-preview");
  try {
    const r = await api(`/api/db/session/${S.db.grid.sessionId}/audit`);
    if (!r.total) { pre.textContent = "No audit entries yet."; return; }
    pre.textContent = `${r.total} audit entr(ies):\n\n` + r.entries.slice(0, 100).map((e) =>
      `[${e.ts}] ${e.status} ${e.table} ${JSON.stringify(e.key)} ${e.column}: ${e.old} → ${e.new}`).join("\n");
  } catch (e) { pre.textContent = "✗ " + e.message; }
}

function dbBindEvents() {
  $("db-vault-unlock").onclick = dbUnlock;
  $("db-vault-lock").onclick = dbLock;
  $("db-vault-pw").addEventListener("keydown", (e) => { if (e.key === "Enter") dbUnlock(); });
  $("db-add-profile").onclick = () => dbShowProfileForm({});
  $("dbf-cancel").onclick = () => $("db-profile-form").classList.add("hidden");
  $("dbf-save").onclick = dbSaveProfile;
  $("dbf-test").onclick = dbTestConnection;
  $("dbf-engine").onchange = dbSyncEngineFields;
  $("dbf-ssl").onchange = () => $("dbf-ssl-fields").classList.toggle("hidden", !$("dbf-ssl").checked);
  // Import + grid.
  $("db-new-import").onclick = dbNewImport;
  $("dbi-cancel").onclick = () => $("db-import-form").classList.add("hidden");
  $("dbi-load-tables").onclick = dbLoadTables;
  $("dbi-mode").onchange = dbSyncImportMode;
  $("dbi-import").onclick = dbRunImport;
  $("db-grid-add-col").onclick = dbAddColumn;
  $("db-grid-process").onclick = dbShowProcess;
  $("dbp-add").onclick = dbAddProcCol;
  $("dbp-run").onclick = dbRunProcess;
  $("dbp-close").onclick = () => $("db-process-panel").classList.add("hidden");
  $("db-grid-writeback").onclick = () => $("db-writeback-panel").classList.toggle("hidden");
  $("dbw-dryrun").onclick = dbDryRun;
  $("dbw-conflicts").onclick = dbCheckConflicts;
  $("dbw-run").onclick = dbWriteBack;
  $("dbw-audit").onclick = dbViewAudit;
  $("dbw-close").onclick = () => $("db-writeback-panel").classList.add("hidden");
  $("db-grid-close").onclick = () => $("db-grid-wrap").classList.add("hidden");
  $("db-grid-prev").onclick = () => dbOpenSession(S.db.grid.sessionId, Math.max(0, S.db.grid.offset - S.db.grid.limit));
  $("db-grid-next").onclick = () => dbOpenSession(S.db.grid.sessionId, S.db.grid.offset + S.db.grid.limit);
}

// ------------------------------- tabs --------------------------------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("tab-chat").classList.toggle("active", name === "chat");
  $("tab-evals").classList.toggle("active", name === "evals");
  $("tab-database").classList.toggle("active", name === "database");
  $("tab-resources").classList.toggle("active", name === "resources");
  $("tab-personas").classList.toggle("active", name === "personas");
  $("tab-settings").classList.toggle("active", name === "settings");
  if (name === "resources" && !S.activeLibrary && S.libraries.length) {
    selectLibrary(S.libraries[0].id); $("lib-list").value = S.libraries[0].id;
  }
  if (name === "settings") renderSettings();
  if (name === "evals") initEvalTab();
  if (name === "database") initDatabaseTab();
  if (name === "personas") renderPersonaTab();
}

// ------------------------------- events ------------------------------
function bindEvents() {
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => switchTab(t.dataset.tab));
  document.querySelectorAll(".modal-close").forEach((b) => b.onclick = closeModal);
  $("modal-backdrop").onclick = (e) => { if (e.target === $("modal-backdrop")) closeModal(); };

  $("btn-new-chat").onclick = () => newChat();
  $("btn-new-private").onclick = () => newPrivateChat();
  $("btn-import-chats").onclick = () => importChats();
  $("btn-export-chats").onclick = () => exportAllChats();
  $("btn-manage-servers").onclick = () => switchTab("settings");
  $("btn-mem-save").onclick = saveMemory;

  $("server-select").onchange = () => { if (S.chat) S.chat.server_url = currentServerUrl(); refreshModels(true); };
  $("model-select").onchange = () => { onModelChanged(); persistChat(); };
  $("model-manual").onchange = () => { onModelChanged(); persistChat(); };
  $("btn-refresh-models").onclick = () => refreshModels(true);

  // Settings tab.
  $("btn-save-keys").onclick = saveApiKeys;
  $("btn-add-server-row").onclick = () => $("settings-server-rows").appendChild(makeServerRow({}));
  $("btn-save-servers").onclick = saveSettingsServers;
  $("btn-scan").onclick = scanRange;
  $("btn-add-scanned").onclick = addScanned;
  $("btn-save-general").onclick = saveGeneral;

  $("btn-library").onclick = openLibrarySelector;
  $("btn-libselect-save").onclick = saveLibrarySelection;
  $("chk-strict").onchange = () => { if (S.chat) { S.chat.library_strict = $("chk-strict").checked; persistChat(); } };

  $("btn-rename").onclick = renameChat;
  $("btn-clear").onclick = clearMessages;
  $("btn-delete-chat").onclick = () => { if (S.chat) deleteChat(S.chat.id); };
  $("btn-save-private").onclick = savePrivate;

  // System prompt + Pre-prompt fields and their library browsers.
  $("system-prompt").oninput = () => { if (S.chat) { S.chat.system_prompt = $("system-prompt").value; persistChat(); } };
  $("system-on").onchange = () => { if (S.chat) { S.chat.system_on = $("system-on").checked; persistChat(); } };
  $("btn-system-browse").onclick = () => openPromptBrowser("system");
  $("btn-system-save").onclick = () => openPromptBrowser("system", $("system-prompt").value);

  $("pre-prompt").oninput = () => { if (S.chat) { S.chat.pre_prompt = $("pre-prompt").value; persistChat(); } };
  $("pre-on").onchange = () => { if (S.chat) { S.chat.pre_on = $("pre-on").checked; persistChat(); } };
  $("btn-pre-browse").onclick = () => openPromptBrowser("pre");
  $("btn-pre-save").onclick = () => openPromptBrowser("pre", $("pre-prompt").value);

  // Prompt-browser modal controls.
  $("btn-pb-new-group").onclick = pbNewGroup;
  $("btn-pb-new-cat").onclick = pbNewCat;
  $("btn-pb-new-prompt").onclick = pbNewPrompt;
  $("btn-pb-use").onclick = pbUse;
  $("btn-pb-save").onclick = pbSaveEdit;
  $("btn-pb-import").onclick = pbImport;
  $("ctx-select").onchange = () => { if (S.chat) { S.chat.num_ctx = parseInt($("ctx-select").value); persistChat(); } };
  $("chk-isolate").onchange = () => { if (S.chat) { S.chat.isolated = $("chk-isolate").checked; persistChat(); } };
  $("chk-hide-thinking").onchange = () => { if (S.chat) { S.chat.hide_thinking = $("chk-hide-thinking").checked; persistChat(); } };
  $("chk-websearch").onchange = () => { if (S.chat) { S.chat.web_search = $("chk-websearch").checked; persistChat(); } updateWebsearchVisibility(); };
  $("crawl-pages").onchange = () => { if (S.chat) { S.chat.crawl_pages = parseInt($("crawl-pages").value) || 7; persistChat(); } };
  $("chk-rag").onchange = () => { if (S.chat) { S.chat.rag_enabled = $("chk-rag").checked; persistChat(); } };
  $("chk-rag-auto").onchange = () => { if (S.chat) { S.chat.rag_auto = $("chk-rag-auto").checked; persistChat(); } };
  $("rag-threshold").onchange = () => { if (S.chat) { S.chat.rag_threshold = Math.max(1, parseInt($("rag-threshold").value) || 400); persistChat(); } };
  $("chk-multipass").onchange = () => { if (S.chat) { S.chat.multi_pass = $("chk-multipass").checked; persistChat(); } updateMultipassVisibility(); };
  $("mp-passes").onchange = () => { if (S.chat) { S.chat.passes = Math.max(1, parseInt($("mp-passes").value) || 2); persistChat(); } };
  $("chk-mp-system").onchange = () => { if (S.chat) { S.chat.pass_use_system = $("chk-mp-system").checked; persistChat(); } };
  $("mp-eval-prompt").oninput = () => { if (S.chat) { S.chat.eval_prompt = $("mp-eval-prompt").value; persistChat(); } };

  $("btn-add-domain").onclick = addDomain;
  $("btn-remove-domain").onclick = removeDomains;
  $("btn-clear-domains").onclick = clearDomains;
  $("chk-restrict").onchange = saveWebsearchConfig;

  $("btn-send").onclick = sendMessage;
  $("btn-rewrite").onclick = rewritePrompt;
  $("btn-rewrite-undo").onclick = undoRewrite;
  $("chk-persona").onchange = onPersonaToggle;
  $("persona-select").onchange = (e) => { S.personaId = e.target.value; renderVariantControl(); };
  $("btn-data-mode").onclick = toggleDataMode;
  $("btn-add-data").onclick = addDataItem;
  $("btn-batch").onclick = batchProcess;
  $("btn-context-history").onclick = openContextHistory;
  $("btn-context-history-clear").onclick = clearContextHistory;
  $("btn-stop").onclick = stopGeneration;
  $("btn-queue-add").onclick = addToQueue;
  $("btn-queue-run").onclick = runQueue;
  $("btn-queue-clear").onclick = clearQueue;
  $("chk-parallel").onchange = toggleParallel;
  $("btn-save-parallel").onclick = saveParallelConfig;
  $("btn-parallel-common").onclick = useCommonModel;
  $("input-box").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  setupMessagesResizer();

  // Evaluate tab.
  $("eval-project-select").onchange = () => onEvalProjectSelected($("eval-project-select").value);
  $("btn-eval-new").onclick = () => newEvalProject();
  $("btn-eval-save").onclick = () => saveEvalProject();
  $("btn-eval-delete").onclick = () => deleteEvalProject();
  $("btn-eval-addrow").onclick = () => evalAddRow();
  $("btn-eval-addcol").onclick = () => evalAddColumn();
  $("btn-eval-import-csv").onclick = () => importEvalFile("csv");
  $("btn-eval-import-txt").onclick = () => importEvalFile("txt");
  $("eval-gen-server").onchange = () => onEvalServerChange("gen");
  $("eval-gen-model").onchange = () => { if (S.evalProject) S.evalProject.gen_model = $("eval-gen-model").value; };
  $("eval-grader-server").onchange = () => onEvalServerChange("grader");
  $("eval-grader-model").onchange = () => { if (S.evalProject) S.evalProject.grader_model = $("eval-grader-model").value; };
  $("eval-output-col").onchange = () => { if (S.evalProject) { S.evalProject.output_column = $("eval-output-col").value; renderEvalChips(); renderEvalGenInstr(); } };
  $("eval-prompt").oninput = () => { if (S.evalProject) S.evalProject.prompt_template = $("eval-prompt").value; };
  $("btn-eval-addcrit").onclick = () => evalAddCriterion();
  $("btn-eval-addmodel").onclick = () => evalAddBatchModel();
  $("btn-eval-run").onclick = () => runEval(false);
  $("btn-eval-run-batch").onclick = () => runEval(true);
  $("btn-eval-stop").onclick = () => stopEval();
  $("btn-eval-gen-run").onclick = () => runEvalGen();
  $("btn-eval-gen-stop").onclick = () => stopEvalGen();
  $("eval-gen-rows").oninput = () => { if (S.evalProject) S.evalProject.gen_num_rows = parseInt($("eval-gen-rows").value, 10) || 1; };

  // Resources tab.
  $("lib-list").onchange = () => selectLibrary($("lib-list").value);
  $("btn-lib-new").onclick = newLibrary;
  $("btn-lib-remove").onclick = removeLibrary;
  $("btn-lib-load").onclick = loadLibraryXML;
  $("btn-lib-save-xml").onclick = saveLibraryXML;
  $("btn-lib-add-write").onclick = addWriteIn;
  $("btn-lib-add-files").onclick = addTextFiles;
  $("btn-lib-add-url").onclick = () => showLibPanel("url");
  $("btn-lib-add-search").onclick = () => showLibPanel("search");
  $("btn-lib-url-fetch").onclick = addByUrl;
  $("btn-lib-url-cancel").onclick = hideLibPanels;
  $("lib-url-input").onkeydown = (e) => { if (e.key === "Enter") addByUrl(); };
  $("btn-lib-search-go").onclick = braveSearch;
  $("btn-lib-search-cancel").onclick = hideLibPanels;
  $("btn-lib-compile").onclick = () => {
    if (!S.activeLibrary) { toast("Select or create a library first"); return; }
    runCompile("library", S.activeLibrary.id, { force: $("lib-compile-force").checked },
               $("lib-compile-progress"), $("lib-compile-badge"));
  };
  $("lib-name").oninput = () => saveLibrary();

  // Global keyboard shortcuts.
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && !e.shiftKey && (e.key === "n" || e.key === "N")) {
      if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
      e.preventDefault(); newChat();
    } else if (e.ctrlKey && e.shiftKey && (e.key === "n" || e.key === "N")) {
      e.preventDefault(); newPrivateChat();
    }
  });
}

// ============================ Evaluate tab ============================
function evEsc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function newEvalProjectLocal() {
  const url = (S.servers[0] && S.servers[0].url) || S.config.default_local_url || "";
  return {
    id: "", name: "New Evaluation",
    columns: ["Input", "Response"],
    rows: [{ Input: "", Response: "" }],
    gen_server_url: url, gen_model: "",
    prompt_template: "", output_column: "Response",
    grader_server_url: url, grader_model: "",
    criteria: (S.defaultCriteria || []).map((c) => ({ ...c })),
    batch_models: [],
    gen_instructions: {},
    gen_num_rows: 10,
    num_ctx: S.config.default_num_ctx || 4096,
  };
}

function initEvalTab() {
  if (!S.evalInited) {
    S.evalInited = true;
    if (!S.evalProject) S.evalProject = newEvalProjectLocal();
  }
  populateEvalProjectSelect();
  renderEvalProject();
}

function populateEvalProjectSelect() {
  const sel = $("eval-project-select");
  sel.innerHTML = "";
  const optNew = document.createElement("option");
  optNew.value = ""; optNew.textContent = "— New (unsaved) —";
  sel.appendChild(optNew);
  S.evals.forEach((e) => {
    const o = document.createElement("option");
    o.value = e.id; o.textContent = e.name || "Untitled eval";
    sel.appendChild(o);
  });
  sel.value = (S.evalProject && S.evalProject.id) || "";
}

// -- fill a <server>/<model> pair from S.servers + the models endpoint --
function evalFillServerSelect(sel, chosen) {
  sel.innerHTML = "";
  S.servers.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.url; o.textContent = s.label;
    sel.appendChild(o);
  });
  if (chosen && [...sel.options].some((o) => o.value === chosen)) sel.value = chosen;
}

async function evalFillModelSelect(serverUrl, modelSel, chosen) {
  modelSel.innerHTML = '<option value="">loading…</option>';
  let models = S.evalModelCache[serverUrl];
  if (!models) {
    try {
      const r = await api(`/api/models?server=${encodeURIComponent(serverUrl)}`);
      models = r.models || [];
      S.evalModelCache[serverUrl] = models;
    } catch (e) { models = []; }
  }
  modelSel.innerHTML = "";
  // Always keep the saved model selectable even if the list can't be fetched.
  const list = [...models];
  if (chosen && !list.includes(chosen)) list.unshift(chosen);
  if (!list.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "(no models — check server)";
    modelSel.appendChild(o);
    return;
  }
  list.forEach((m) => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    modelSel.appendChild(o);
  });
  modelSel.value = chosen && list.includes(chosen) ? chosen : list[0];
}

async function onEvalServerChange(which) {
  const p = S.evalProject; if (!p) return;
  if (which === "gen") {
    p.gen_server_url = $("eval-gen-server").value;
    await evalFillModelSelect(p.gen_server_url, $("eval-gen-model"), p.gen_model);
    p.gen_model = $("eval-gen-model").value;
  } else {
    p.grader_server_url = $("eval-grader-server").value;
    await evalFillModelSelect(p.grader_server_url, $("eval-grader-model"), p.grader_model);
    p.grader_model = $("eval-grader-model").value;
  }
}

// ---- render the whole project into the tab ----
function renderEvalProject() {
  const p = S.evalProject; if (!p) return;
  $("eval-name").value = p.name || "";
  $("eval-prompt").value = p.prompt_template || "";
  evalFillServerSelect($("eval-gen-server"), p.gen_server_url);
  evalFillServerSelect($("eval-grader-server"), p.grader_server_url || p.gen_server_url);
  evalFillModelSelect($("eval-gen-server").value, $("eval-gen-model"), p.gen_model).then(() => {
    p.gen_model = $("eval-gen-model").value || p.gen_model;
  });
  evalFillModelSelect($("eval-grader-server").value, $("eval-grader-model"), p.grader_model || p.gen_model).then(() => {
    p.grader_model = $("eval-grader-model").value || p.grader_model;
  });
  renderEvalGrid();
  renderEvalOutputCol();
  renderEvalChips();
  renderEvalGenInstr();
  $("eval-gen-rows").value = p.gen_num_rows || 10;
  renderEvalCriteria();
  renderEvalBatchModels();
  $("eval-results").innerHTML = "";
  $("eval-progress").textContent = "";
  $("eval-gen-progress").textContent = "";
}

async function onEvalProjectSelected(id) {
  if (!id) { S.evalProject = newEvalProjectLocal(); renderEvalProject(); return; }
  try {
    const r = await api(`/api/evals/${id}`);
    S.evalProject = r.eval;
    if (!S.evalProject.grader_server_url) S.evalProject.grader_server_url = S.evalProject.gen_server_url;
    renderEvalProject();
  } catch (e) { toast("Could not load eval: " + e.message); }
}

function newEvalProject() {
  S.evalProject = newEvalProjectLocal();
  populateEvalProjectSelect();
  renderEvalProject();
}

async function saveEvalProject() {
  const p = S.evalProject; if (!p) return;
  p.name = $("eval-name").value.trim() || "Untitled evaluation";
  p.prompt_template = $("eval-prompt").value;
  p.input_columns = p.columns.filter((c) => c !== p.output_column);
  try {
    const r = await api("/api/evals", { method: "POST", body: { eval: p } });
    S.evalProject = r.eval;
    S.evals = r.evals || [];
    populateEvalProjectSelect();
    toast("Evaluation saved");
  } catch (e) { toast("Save failed: " + e.message); }
}

async function deleteEvalProject() {
  const p = S.evalProject; if (!p) return;
  if (!p.id) { newEvalProject(); return; }
  if (!confirm("Delete this evaluation project?")) return;
  try {
    const r = await api(`/api/evals/${p.id}`, { method: "DELETE" });
    S.evals = r.evals || [];
    S.evalProject = newEvalProjectLocal();
    populateEvalProjectSelect();
    renderEvalProject();
    toast("Deleted");
  } catch (e) { toast("Delete failed: " + e.message); }
}

// ---- data grid ----
function renderEvalGrid() {
  const p = S.evalProject;
  const table = $("eval-grid");
  table.innerHTML = "";
  // header
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  p.columns.forEach((col, ci) => {
    const th = document.createElement("th");
    const inp = document.createElement("input");
    inp.className = "eval-col-name"; inp.value = col;
    inp.title = "Rename column";
    inp.onchange = () => evalRenameColumn(ci, inp.value.trim());
    const del = document.createElement("button");
    del.className = "eval-x"; del.textContent = "✕"; del.title = "Delete column";
    del.onclick = () => evalDeleteColumn(ci);
    th.appendChild(inp); th.appendChild(del);
    htr.appendChild(th);
  });
  const thEnd = document.createElement("th"); thEnd.className = "eval-rownum-head"; thEnd.textContent = "";
  htr.appendChild(thEnd);
  thead.appendChild(htr);
  table.appendChild(thead);
  // body
  const tbody = document.createElement("tbody");
  p.rows.forEach((row, ri) => {
    const tr = document.createElement("tr");
    p.columns.forEach((col) => {
      const td = document.createElement("td");
      const inp = document.createElement("input");
      inp.className = "eval-cell"; inp.value = row[col] == null ? "" : row[col];
      inp.oninput = () => { row[col] = inp.value; };
      td.appendChild(inp);
      tr.appendChild(td);
    });
    const tdDel = document.createElement("td"); tdDel.className = "eval-rownum";
    const del = document.createElement("button");
    del.className = "eval-x"; del.textContent = "✕"; del.title = "Delete row";
    del.onclick = () => { p.rows.splice(ri, 1); if (!p.rows.length) evalAddRow(); else renderEvalGrid(); };
    tdDel.appendChild(del);
    tr.appendChild(tdDel);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function evalAddRow() {
  const p = S.evalProject;
  const row = {}; p.columns.forEach((c) => (row[c] = ""));
  p.rows.push(row);
  renderEvalGrid();
}

async function evalAddColumn() {
  const p = S.evalProject;
  const name = await promptModal("New column name", `Column ${p.columns.length + 1}`);
  if (name === null) return;
  let col = name.trim() || `Column ${p.columns.length + 1}`;
  while (p.columns.includes(col)) col += "_2";
  p.columns.push(col);
  p.rows.forEach((r) => (r[col] = ""));
  renderEvalGrid(); renderEvalOutputCol(); renderEvalChips(); renderEvalGenInstr();
}

function evalRenameColumn(ci, newName) {
  const p = S.evalProject;
  const old = p.columns[ci];
  newName = newName || old;
  if (newName === old) return;
  if (p.columns.includes(newName)) { toast("A column with that name already exists"); renderEvalGrid(); return; }
  p.columns[ci] = newName;
  p.rows.forEach((r) => { r[newName] = r[old]; delete r[old]; });
  if (p.output_column === old) p.output_column = newName;
  if (p.gen_instructions && p.gen_instructions[old] != null) {
    p.gen_instructions[newName] = p.gen_instructions[old];
    delete p.gen_instructions[old];
  }
  renderEvalGrid(); renderEvalOutputCol(); renderEvalChips(); renderEvalGenInstr();
}

function evalDeleteColumn(ci) {
  const p = S.evalProject;
  if (p.columns.length <= 1) { toast("Keep at least one column"); return; }
  const old = p.columns[ci];
  p.columns.splice(ci, 1);
  p.rows.forEach((r) => delete r[old]);
  if (p.gen_instructions) delete p.gen_instructions[old];
  if (p.output_column === old) p.output_column = p.columns[p.columns.length - 1];
  renderEvalGrid(); renderEvalOutputCol(); renderEvalChips(); renderEvalGenInstr();
}

function renderEvalOutputCol() {
  const p = S.evalProject;
  const sel = $("eval-output-col");
  sel.innerHTML = "";
  p.columns.forEach((c) => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    sel.appendChild(o);
  });
  if (!p.columns.includes(p.output_column)) p.output_column = p.columns[p.columns.length - 1];
  sel.value = p.output_column;
}

// ---- per-column "generate this cell" instructions ----
function renderEvalGenInstr() {
  const p = S.evalProject;
  const table = $("eval-gen-instr");
  if (!table) return;
  if (!p.gen_instructions) p.gen_instructions = {};
  table.innerHTML = "";
  const tbody = document.createElement("tbody");
  p.columns.forEach((col) => {
    const isOut = col === p.output_column;
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    tdName.className = "eval-gen-colname";
    tdName.textContent = col;
    const tdInp = document.createElement("td");
    const inp = document.createElement("input");
    inp.className = "eval-cell";
    if (isOut) {
      inp.value = "";
      inp.disabled = true;
      inp.placeholder = "response column — filled by an evaluation run";
    } else {
      inp.value = p.gen_instructions[col] || "";
      inp.placeholder = "how to fill this column, e.g. a realistic full name";
      inp.oninput = () => { p.gen_instructions[col] = inp.value; };
    }
    tdInp.appendChild(inp);
    tr.appendChild(tdName); tr.appendChild(tdInp);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function renderEvalChips() {
  const p = S.evalProject;
  const box = $("eval-col-chips");
  box.innerHTML = "";
  p.columns.filter((c) => c !== p.output_column).forEach((c) => {
    const chip = document.createElement("button");
    chip.className = "eval-chip"; chip.type = "button";
    chip.textContent = `{${c}}`;
    chip.title = "Insert into the prompt";
    chip.onclick = () => insertAtCursor($("eval-prompt"), `{${c}}`);
    box.appendChild(chip);
  });
}

function insertAtCursor(ta, text) {
  const start = ta.selectionStart || 0, end = ta.selectionEnd || 0;
  ta.value = ta.value.slice(0, start) + text + ta.value.slice(end);
  ta.selectionStart = ta.selectionEnd = start + text.length;
  ta.focus();
  if (S.evalProject) S.evalProject.prompt_template = ta.value;
}

// ---- file import ----
async function importEvalFile(kind) {
  const p = S.evalProject;
  let delimiter = "", isRegex = false, targetCol = null;
  if (kind === "txt") {
    const d = await promptModal("Split the text file by (blank = blank lines):", "");
    if (d === null) return;
    delimiter = d;
    const cols = p.columns.join(", ");
    const tc = await promptModal(`Put each chunk into which column? (${cols})`, p.columns[0]);
    if (tc === null) return;
    targetCol = tc.trim();
    if (!p.columns.includes(targetCol)) {
      p.columns.push(targetCol);
      p.rows.forEach((r) => (r[targetCol] = ""));
    }
  }
  setStatus("Choose a file…");
  try {
    const r = await api("/api/evals/import", { method: "POST",
      body: { kind, delimiter, is_regex: isRegex } });
    if (!r.ok) { if (!r.cancelled) toast("Import failed: " + (r.error || "unknown")); setStatus(""); return; }
    if (kind === "csv") {
      p.columns = r.columns.length ? r.columns : p.columns;
      p.rows = r.rows || [];
      if (!p.rows.length) p.rows = [{}];
      if (!p.columns.includes(p.output_column)) {
        // add a response column to receive generated output
        const respName = p.columns.includes("Response") ? "Response" : "Response";
        if (!p.columns.includes(respName)) { p.columns.push(respName); p.rows.forEach((row) => (row[respName] = "")); }
        p.output_column = respName;
      }
      toast(`Imported ${p.rows.length} row(s) from ${r.name}`);
    } else {
      const cells = r.cells || [];
      // Fill the target column, extending rows as needed.
      cells.forEach((val, i) => {
        if (i >= p.rows.length) { const row = {}; p.columns.forEach((c) => (row[c] = "")); p.rows.push(row); }
        p.rows[i][targetCol] = val;
      });
      toast(`Imported ${cells.length} cell(s) into "${targetCol}" from ${r.name}`);
    }
    renderEvalGrid(); renderEvalOutputCol(); renderEvalChips();
  } catch (e) { toast("Import error: " + e.message); }
  setStatus("");
}

// ---- criteria ----
function renderEvalCriteria() {
  const p = S.evalProject;
  const box = $("eval-criteria");
  box.innerHTML = "";
  (p.criteria || []).forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "eval-crit-row";

    const label = document.createElement("input");
    label.className = "eval-crit-label"; label.placeholder = "Label (e.g. Accuracy)";
    label.value = c.label || "";
    label.oninput = () => (c.label = label.value);

    const mode = document.createElement("select");
    mode.className = "eval-crit-mode";
    [["score", "Score"], ["reasoning", "Reasoning"]].forEach(([v, t]) => {
      const o = document.createElement("option"); o.value = v; o.textContent = t; mode.appendChild(o);
    });
    mode.value = c.mode || "score";

    const min = document.createElement("input");
    min.type = "number"; min.className = "eval-crit-num"; min.value = c.min == null ? 1 : c.min;
    min.title = "Min score";
    min.onchange = () => (c.min = parseInt(min.value) || 0);
    const max = document.createElement("input");
    max.type = "number"; max.className = "eval-crit-num"; max.value = c.max == null ? 10 : c.max;
    max.title = "Max score";
    max.onchange = () => (c.max = parseInt(max.value) || 10);
    const rangeWrap = document.createElement("span");
    rangeWrap.className = "eval-crit-range";
    rangeWrap.appendChild(document.createTextNode(" "));
    rangeWrap.appendChild(min);
    rangeWrap.appendChild(document.createTextNode("–"));
    rangeWrap.appendChild(max);
    const syncRange = () => { rangeWrap.style.display = mode.value === "score" ? "" : "none"; };
    mode.onchange = () => { c.mode = mode.value; syncRange(); };
    syncRange();

    const guidance = document.createElement("textarea");
    guidance.className = "eval-crit-guidance"; guidance.rows = 2;
    guidance.placeholder = "How should the grader judge this? (the rubric)";
    guidance.value = c.guidance || "";
    guidance.oninput = () => (c.guidance = guidance.value);

    const del = document.createElement("button");
    del.className = "eval-x"; del.textContent = "✕"; del.title = "Remove criterion";
    del.onclick = () => { p.criteria.splice(i, 1); renderEvalCriteria(); };

    const top = document.createElement("div");
    top.className = "eval-crit-top";
    top.appendChild(label); top.appendChild(mode); top.appendChild(rangeWrap); top.appendChild(del);
    row.appendChild(top);
    row.appendChild(guidance);
    box.appendChild(row);
  });
}

function evalAddCriterion() {
  S.evalProject.criteria.push({ label: "", guidance: "", mode: "score", min: 1, max: 10 });
  renderEvalCriteria();
}

// ---- batch models ----
function renderEvalBatchModels() {
  const p = S.evalProject;
  const box = $("eval-batch-models");
  box.innerHTML = "";
  (p.batch_models || []).forEach((m, i) => {
    const row = document.createElement("div");
    row.className = "eval-batch-row";
    const ssel = document.createElement("select");
    S.servers.forEach((s) => { const o = document.createElement("option"); o.value = s.url; o.textContent = s.label; ssel.appendChild(o); });
    if (m.server_url) ssel.value = m.server_url; else m.server_url = ssel.value;
    const msel = document.createElement("select");
    evalFillModelSelect(ssel.value, msel, m.model).then(() => { m.model = msel.value || m.model; });
    ssel.onchange = () => { m.server_url = ssel.value; evalFillModelSelect(ssel.value, msel, m.model).then(() => (m.model = msel.value)); };
    msel.onchange = () => (m.model = msel.value);
    const del = document.createElement("button");
    del.className = "eval-x"; del.textContent = "✕";
    del.onclick = () => { p.batch_models.splice(i, 1); renderEvalBatchModels(); };
    row.appendChild(ssel); row.appendChild(msel); row.appendChild(del);
    box.appendChild(row);
  });
}

function evalAddBatchModel() {
  const p = S.evalProject;
  const url = p.gen_server_url || (S.servers[0] && S.servers[0].url) || "";
  p.batch_models.push({ server_url: url, model: p.gen_model || "" });
  renderEvalBatchModels();
}

// ---- run ----
function evalSyncFromUI() {
  const p = S.evalProject;
  p.name = $("eval-name").value.trim() || "Untitled evaluation";
  p.prompt_template = $("eval-prompt").value;
  p.gen_server_url = $("eval-gen-server").value;
  p.gen_model = $("eval-gen-model").value;
  p.grader_server_url = $("eval-grader-server").value;
  p.grader_model = $("eval-grader-model").value;
  p.output_column = $("eval-output-col").value;
  p.input_columns = p.columns.filter((c) => c !== p.output_column);
  p.gen_num_rows = parseInt($("eval-gen-rows").value, 10) || p.gen_num_rows || 10;
}

async function runEval(batch) {
  if (S.evalRunning) return;
  const p = S.evalProject; if (!p) return;
  evalSyncFromUI();
  if (!p.rows.length) { toast("Add at least one data row"); return; }
  if (!p.prompt_template.trim()) { toast("Write a prompt to evaluate"); return; }
  if (!(p.criteria || []).some((c) => (c.label || "").trim())) { toast("Add at least one grading criterion"); return; }
  if (batch && !(p.batch_models || []).some((m) => m.model)) { toast("Add at least one batch model"); return; }
  if (!batch && !p.gen_model) { toast("Select a model to run the prompt"); return; }

  S.evalRunId = uid();
  S.evalRunning = true;
  S.evalRun = { batch, models: [], criteria: (p.criteria || []).filter((c) => (c.label || "").trim()) };
  $("btn-eval-stop").classList.remove("hidden");
  $("btn-eval-run").disabled = $("btn-eval-run-batch").disabled = true;
  $("eval-results").innerHTML = "";
  $("eval-progress").textContent = "Starting…";

  await streamSSE("/api/evals/run",
    { eval: p, run_id: S.evalRunId, batch },
    {
      start: (d) => { $("eval-progress").textContent = `Running ${d.total_rows} row(s)…`; },
      model_start: (d) => {
        S.evalRun.models[d.index] = { server: d.server, model: d.model, rows: [], aggregate: null };
        $("eval-progress").textContent = `Model ${d.index + 1}/${d.total}: ${d.model} — generating…`;
      },
      gen_progress: (d) => {
        const mi = S.evalRun.models.length - 1;
        $("eval-progress").textContent = `${S.evalRun.models[mi] ? S.evalRun.models[mi].model : ""}: generated ${d.done}/${d.total}`;
      },
      row_result: (d) => {
        const m = S.evalRun.models[d.model_index];
        if (m) m.rows[d.index] = { response: d.response, grades: d.grades, ungraded: d.ungraded };
        $("eval-progress").textContent = `${m ? m.model : ""}: graded row ${d.index + 1}`;
      },
      model_done: (d) => {
        const m = S.evalRun.models[d.index];
        if (m) m.aggregate = d.aggregate;
        renderEvalResults();
      },
      status: (d) => { if (d.message) setStatus(d.message); },
      summary: (d) => { S.evalRun.summary = d; renderEvalResults(); },
      error: (d) => { toast("Eval error: " + (d.message || "unknown")); },
      done: (d) => {
        $("eval-progress").textContent = d.stopped ? "Stopped." : "Done.";
        endEvalRun();
        renderEvalResults();
      },
    }
  ).catch((e) => { toast("Run failed: " + e.message); endEvalRun(); });
}

function endEvalRun() {
  S.evalRunning = false;
  $("btn-eval-stop").classList.add("hidden");
  $("btn-eval-run").disabled = $("btn-eval-run-batch").disabled = false;
}

async function stopEval() {
  if (!S.evalRunId) return;
  try { await api("/api/stop", { method: "POST", body: { run_id: S.evalRunId } }); } catch (e) {}
  $("eval-progress").textContent = "Stopping…";
}

// ---- generate dummy test rows with the LLM ----
async function runEvalGen() {
  if (S.evalGenRunning) return;
  const p = S.evalProject; if (!p) return;
  evalSyncFromUI();
  if (!p.gen_instructions) p.gen_instructions = {};
  const targets = p.columns.filter((c) => c !== p.output_column && (p.gen_instructions[c] || "").trim());
  if (!targets.length) { toast("Add a generation instruction to at least one non-response column"); return; }
  if (!p.gen_model) { toast("Select a model (step 2) to generate data"); return; }
  const numRows = parseInt($("eval-gen-rows").value, 10) || 0;
  if (numRows < 1) { toast("Enter how many rows to generate"); return; }

  S.evalGenRunId = uid();
  S.evalGenRunning = true;
  $("btn-eval-gen-stop").classList.remove("hidden");
  $("btn-eval-gen-run").disabled = true;
  $("eval-gen-progress").textContent = "Starting…";

  await streamSSE("/api/evals/gen-data",
    { eval: p, run_id: S.evalGenRunId, num_rows: numRows },
    {
      start: (d) => { $("eval-gen-progress").textContent = `Generating ${d.total} row(s)…`; },
      row_result: (d) => {
        const row = {};
        p.columns.forEach((c) => (row[c] = (d.row && d.row[c] != null) ? d.row[c] : ""));
        p.rows.push(row);
        renderEvalGrid();
      },
      gen_progress: (d) => { $("eval-gen-progress").textContent = `Generated ${d.done}/${d.total}`; },
      status: (d) => { if (d.message) setStatus(d.message); },
      error: (d) => { toast("Generation error: " + (d.message || "unknown")); },
      done: (d) => {
        $("eval-gen-progress").textContent = d.stopped ? "Stopped." : "Done.";
        endEvalGen();
      },
    }
  ).catch((e) => { toast("Generation failed: " + e.message); endEvalGen(); });
}

function endEvalGen() {
  S.evalGenRunning = false;
  $("btn-eval-gen-stop").classList.add("hidden");
  $("btn-eval-gen-run").disabled = false;
}

async function stopEvalGen() {
  if (!S.evalGenRunId) return;
  try { await api("/api/stop", { method: "POST", body: { run_id: S.evalGenRunId } }); } catch (e) {}
  $("eval-gen-progress").textContent = "Stopping…";
}

// ---- results rendering ----
function evalScoreClass(pct) {
  if (pct == null) return "";
  if (pct >= 75) return "good";
  if (pct >= 50) return "ok";
  return "bad";
}

function svgBarChart(items) {
  // items: [{label, value(0-100)|null}]
  const W = 560, rowH = 30, padL = 150, padR = 54, top = 8;
  const H = top * 2 + Math.max(1, items.length) * rowH;
  const barW = W - padL - padR;
  let body = "";
  items.forEach((it, i) => {
    const y = top + i * rowH;
    const v = it.value == null ? 0 : Math.max(0, Math.min(100, it.value));
    const w = (v / 100) * barW;
    const cls = evalScoreClass(it.value);
    body += `<text x="${padL - 8}" y="${y + 19}" text-anchor="end" class="evc-lbl">${evEsc(it.label)}</text>`;
    body += `<rect x="${padL}" y="${y + 7}" width="${barW}" height="15" rx="3" class="evc-track"/>`;
    body += `<rect x="${padL}" y="${y + 7}" width="${w.toFixed(1)}" height="15" rx="3" class="evc-bar ${cls}"/>`;
    body += `<text x="${padL + barW + 6}" y="${y + 19}" class="evc-val">${it.value == null ? "—" : it.value + "%"}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" class="eval-chart" preserveAspectRatio="xMinYMin meet">${body}</svg>`;
}

function renderEvalResults() {
  const run = S.evalRun; if (!run) return;
  const box = $("eval-results");
  box.innerHTML = "";
  const criteria = run.criteria || [];
  const scoreCrit = criteria.filter((c) => (c.mode || "score") === "score");

  if (!run.batch) {
    const m = run.models[0]; if (!m) return;
    // Headline
    const head = document.createElement("div");
    head.className = "eval-headline";
    const agg = m.aggregate;
    const overall = agg && agg.overall != null ? agg.overall : null;
    head.innerHTML = `<div class="eval-score-big ${evalScoreClass(overall)}">${overall == null ? "—" : overall + "%"}</div>` +
      `<div class="eval-score-cap">Overall prompt score<br><span class="muted">${evEsc(m.model)}</span></div>`;
    box.appendChild(head);
    // Per-criterion chart
    if (agg && scoreCrit.length) {
      const items = scoreCrit.map((c) => ({ label: c.label, value: (agg.per_criterion[c.label] || {}).avg_pct }));
      const wrap = document.createElement("div");
      wrap.innerHTML = `<h4>Average score per criterion</h4>` + svgBarChart(items);
      box.appendChild(wrap);
    }
    box.appendChild(buildRowTable(m, criteria));
    return;
  }

  // Batch: overall per model + per-criterion breakdown
  const overallItems = run.models.filter(Boolean).map((m) => ({
    label: m.model, value: m.aggregate && m.aggregate.overall != null ? m.aggregate.overall : null,
  }));
  const wrapO = document.createElement("div");
  wrapO.innerHTML = `<h4>Overall score by model</h4>` + svgBarChart(overallItems);
  box.appendChild(wrapO);

  if (scoreCrit.length) {
    const h = document.createElement("h4"); h.textContent = "Criterion breakdown by model";
    box.appendChild(h);
    box.appendChild(buildBreakdownTable(run.models.filter(Boolean), scoreCrit));
  }
  // Per-model detail tables
  run.models.filter(Boolean).forEach((m) => {
    const det = document.createElement("details");
    det.className = "eval-model-detail";
    const sum = document.createElement("summary");
    const ov = m.aggregate && m.aggregate.overall != null ? m.aggregate.overall + "%" : "—";
    sum.textContent = `${m.model} — ${ov}`;
    det.appendChild(sum);
    det.appendChild(buildRowTable(m, criteria));
    box.appendChild(det);
  });
}

function buildBreakdownTable(models, scoreCrit) {
  const table = document.createElement("table");
  table.className = "eval-table";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  ["Model", "Overall", ...scoreCrit.map((c) => c.label)].forEach((t) => {
    const th = document.createElement("th"); th.textContent = t; htr.appendChild(th);
  });
  thead.appendChild(htr); table.appendChild(thead);
  const tb = document.createElement("tbody");
  models.forEach((m) => {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td"); nameTd.textContent = m.model; tr.appendChild(nameTd);
    const agg = m.aggregate || { per_criterion: {} };
    const ovTd = document.createElement("td");
    ovTd.className = "eval-cell-score " + evalScoreClass(agg.overall);
    ovTd.textContent = agg.overall == null ? "—" : agg.overall + "%";
    tr.appendChild(ovTd);
    scoreCrit.forEach((c) => {
      const pc = (agg.per_criterion || {})[c.label] || {};
      const td = document.createElement("td");
      td.className = "eval-cell-score " + evalScoreClass(pc.avg_pct);
      td.textContent = pc.avg_pct == null ? "—" : pc.avg_pct + "%";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  return table;
}

function buildRowTable(m, criteria) {
  const p = S.evalProject;
  const inputCols = p.columns.filter((c) => c !== p.output_column);
  const table = document.createElement("table");
  table.className = "eval-table eval-rows-table";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  ["#", ...inputCols, "Response", ...criteria.map((c) => c.label)].forEach((t) => {
    const th = document.createElement("th"); th.textContent = t; htr.appendChild(th);
  });
  thead.appendChild(htr); table.appendChild(thead);
  const tb = document.createElement("tbody");
  (m.rows || []).forEach((res, i) => {
    if (!res) return;
    const tr = document.createElement("tr");
    const numTd = document.createElement("td"); numTd.textContent = i + 1; tr.appendChild(numTd);
    inputCols.forEach((c) => {
      const td = document.createElement("td"); td.className = "eval-td-input";
      td.textContent = (p.rows[i] && p.rows[i][c]) || ""; tr.appendChild(td);
    });
    const rTd = document.createElement("td"); rTd.className = "eval-td-response";
    rTd.textContent = res.response || ""; tr.appendChild(rTd);
    criteria.forEach((c) => {
      const td = document.createElement("td");
      const g = (res.grades || {})[c.label] || {};
      if ((c.mode || "score") === "score") {
        td.className = "eval-cell-score";
        td.textContent = g.score == null ? "—" : String(g.score);
        if (g.reasoning) td.title = g.reasoning;
      } else {
        td.className = "eval-td-reasoning";
        td.textContent = g.reasoning || "—";
      }
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  return table;
}

// ------------------------------- go ----------------------------------
init().catch((e) => { console.error(e); toast("Startup error: " + e.message, 8000); });
