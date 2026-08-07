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
  visionSupported: null, // true | false | null(unknown) — can this model read images?
  imageOutput: false,    // does this model hand images back?
  _imgWarned: false,     // "this model can't read images" said once per model
  runId: null,
  generating: false,
  batchRunning: false,
  activeLibrary: null, // Resources tab current library
  queue: [],           // pending queued prompts: {item_id, chat, search_query, title}
  queueStop: false,    // cancel flag for the sequential queue loop
  dataItems: [],       // data-classification blocks staged for the next send: {id, label, text}
  dataMode: false,     // data-classification composer mode on/off
  parallel: { enabled: false, mode: "balanced", servers: [] }, // multi-server config
  ragServers: [],       // extra embedding endpoints for RAG builds: [{base_url, enabled}]
  contextBars: {},      // live context-usage bars, keyed by chat/lane: {used, window, ...}
  personas: [],         // [{id, name, role, variants[]}] (+ {broken, error} if unreadable)
  usePersona: false,    // "Use Persona" toggle
  personaId: "",        // active persona id
  personaVariant: "",   // selected speaking variant name ("" = the persona's default voice)
  personaRunId: null,   // most recent pipeline run id; a re-run uses its own wrapper's id
  editingPersona: null, // persona open in the Personas tab editor
  memoryCores: [],      // user memory cores: [{id, name, entries[], ...}] (Memory tab)
  memoryCategories: {}, // category id -> label, from the server
  activeMemoryCore: "",   // core being edited in the Memory tab
  lastMemoryCoreId: "",   // default core for a newly opened chat that has none
  editingMemoryEntry: null, // entry id open in the memory-entry modal (null = adding)
  evals: [],            // eval-project summaries
  evalProject: null,    // current full eval-project dict (client-owned)
  evalModelCache: {},   // serverUrl -> [models] (for the eval tab dropdowns)
  evalRunId: null,
  evalRunning: false,
  evalRun: null,        // live results being assembled during a run (inputs snapshotted)
  evalGenRunId: null,   // the ✨ Generate Data run, independent of an evaluation run
  evalGenRunning: false,
  evalInited: false,
  // Batch tab. `batchRunning` above belongs to the chat composer's older folder-batch
  // button and stays separate, so a Batch-tab run doesn't lock the chat.
  batch: {
    inited: false,
    project: null,      // current full batch-project dict (client-owned)
    projects: [],       // saved batch-project summaries
    items: [],          // resolved preview items: {item_id, title, kind, chars, ...}
    results: [],        // finished items this run: {item_id, title, prompt, response}
    running: false,
    runId: null,
    previewRunId: null,
    modelCache: {},     // serverUrl -> [models]
  },
  // Database tab.
  db: {
    inited: false,
    vault: { exists: false, unlocked: false },
    profiles: [],       // masked connection profiles
    sessions: [],       // import-session summaries
    grid: { sessionId: null, offset: 0, limit: 50, total: 0, columns: [], types: {} },
    // Live run ids per streaming flow, so POST /api/stop can reach them.
    runs: { import: null, process: null, writeback: null },
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
 * POST files as multipart/form-data.
 *
 * Separate from api() because that one unconditionally JSON-stringifies its body and
 * sets a JSON content type, both of which are wrong here. Setting no Content-Type at
 * all is deliberate: the browser has to write it itself so the multipart boundary
 * matches. Same non-2xx-throws contract as api().
 *
 * @param {string} path
 * @param {File[]|Blob[]} files
 * @param {object} [fields]  extra plain-text form fields
 */
async function postFiles(path, files, fields = {}) {
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f, f.name || "pasted-image"));
  Object.entries(fields).forEach(([k, v]) => fd.append(k, String(v)));
  const res = await fetch(path, { method: "POST", body: fd });
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
  let sawDone = false;
  try {
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
        if (event === "done") sawDone = true;
        if (handlers[event]) handlers[event](parsed);
      }
    }
  } finally {
    // A connection dropped mid-stream never delivers the server's `done` frame, so
    // callers that tear down teardown-worthy state there (progress bars keep a 1s
    // interval alive) would leak it for the life of the page. Synthesize one.
    if (!sawDone && handlers.done) { try { handlers.done({}); } catch (e) {} }
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
// Dialogs are stacked, not swapped: opening one on top of another (the prompt
// library asking for a name, say) leaves the one underneath visible and inert.
// Two ways out, and the distinction matters:
//   closeModal()   - the action succeeded and the caller has already settled up.
//   dismissModal() - the user backed out (Esc / Cancel / clicking the backdrop),
//                    so the opener's onDismiss runs. That callback is what keeps
//                    promptModal()'s promise from being abandoned mid-flight.
const MODAL_STACK = [];   // [{ id, onDismiss }], last entry is on top

function syncModalStack() {
  const top = MODAL_STACK.length - 1;
  MODAL_STACK.forEach((m, i) => {
    const el = $(m.id);
    if (!el) return;
    el.classList.remove("hidden");
    el.classList.toggle("modal-below", i !== top);
    el.style.zIndex = String(101 + i);
  });
  $("modal-backdrop").classList.toggle("hidden", !MODAL_STACK.length);
}

function openModal(id, onDismiss) {
  const el = $(id);
  if (!el) return;
  // Re-opening something already on the stack raises it rather than duplicating it.
  const at = MODAL_STACK.findIndex((m) => m.id === id);
  if (at >= 0) MODAL_STACK.splice(at, 1);
  MODAL_STACK.push({ id, onDismiss: onDismiss || null });
  syncModalStack();
}

/** Pop a modal without running its onDismiss. Defaults to whatever is on top. */
function closeModal(id) {
  const at = id ? MODAL_STACK.findIndex((m) => m.id === id) : MODAL_STACK.length - 1;
  if (at < 0) return;
  const [m] = MODAL_STACK.splice(at, 1);
  const el = $(m.id);
  if (el) { el.classList.add("hidden"); el.classList.remove("modal-below"); el.style.zIndex = ""; }
  syncModalStack();
}

/** User-initiated close: pop the top modal and let its opener know it was abandoned. */
function dismissModal() {
  const m = MODAL_STACK[MODAL_STACK.length - 1];
  if (!m) return;
  closeModal(m.id);             // pop first, so a re-entrant close from onDismiss is a no-op
  if (m.onDismiss) m.onDismiss();
}

function modalIsOpen() { return MODAL_STACK.length > 0; }

/**
 * Ask for a string. Resolves the entered value, or null if cancelled.
 *
 * Pass `checkbox` (a label) to show one tickbox under the field — the promise then
 * resolves {value, checked} instead of a bare string, so plain callers are unaffected.
 */
function promptModal(title, def = "", { checkbox = "", checked = false } = {}) {
  return new Promise((resolve) => {
    const input = $("prompt-input");
    const okBtn = $("btn-prompt-ok");
    const cancelBtn = $("modal-prompt").querySelector(".modal-close");
    const checkRow = $("prompt-check-row");
    const check = $("prompt-check");
    let done = false;
    // Idempotent by construction: even if a listener somehow outlives its dialog,
    // it can only ever settle this promise once.
    const finish = (val) => {
      if (done) return;
      done = true;
      okBtn.removeEventListener("click", okClick);
      if (cancelBtn) cancelBtn.removeEventListener("click", cancelClick);
      input.removeEventListener("keydown", key);
      if (checkRow) checkRow.classList.add("hidden");
      closeModal("modal-prompt");
      resolve(val);
    };
    const okClick = () => finish(checkbox ? { value: input.value, checked: check.checked } : input.value);
    const cancelClick = () => finish(null);
    const key = (e) => {
      if (e.key === "Enter") { e.preventDefault(); okClick(); }
    };
    okBtn.addEventListener("click", okClick);
    if (cancelBtn) cancelBtn.addEventListener("click", cancelClick);
    input.addEventListener("keydown", key);

    $("prompt-title").textContent = title;
    input.value = def;
    if (checkRow && check) {
      checkRow.classList.toggle("hidden", !checkbox);
      check.checked = !!checked;
      if (checkbox) $("prompt-check-label").textContent = checkbox;
    }
    openModal("modal-prompt", () => finish(null));
    input.focus();
    input.select();   // so typing replaces the old name without a drag-select
  });
}

/** Yes/no confirmation. Resolves true only if the user clicks OK. */
function confirmModal(title) {
  return new Promise((resolve) => {
    const okBtn = $("btn-confirm-ok");
    const cancelBtn = $("modal-confirm").querySelector(".modal-close");
    let done = false;
    const finish = (val) => {
      if (done) return;
      done = true;
      okBtn.removeEventListener("click", okClick);
      if (cancelBtn) cancelBtn.removeEventListener("click", cancelClick);
      closeModal("modal-confirm");
      resolve(val);
    };
    const okClick = () => finish(true);
    const cancelClick = () => finish(false);
    okBtn.addEventListener("click", okClick);
    if (cancelBtn) cancelBtn.addEventListener("click", cancelClick);

    $("confirm-title").textContent = title;
    openModal("modal-confirm", () => finish(false));
    okBtn.focus();
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
  S.batch.projects = st.batch_projects || [];
  S.defaultBatchProject = st.default_batch_project || null;
  S.batchSourceKinds = st.batch_source_kinds || [];
  S.batchExts = st.batch_exts || [];
  S.memoryCores = st.memory_cores || [];
  S.memoryCategories = st.memory_categories || {};
  S.lastMemoryCoreId = (S.memoryCores[0] || {}).id || "";
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
  bindMemoryEvents();
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
  const want = (S.chat && S.chat.model) || "";
  if (want && !S.models.includes(want)) {
    // The chat's model isn't on this server (server switch, or the model was
    // removed). An explicit choice must never be silently replaced with
    // whatever happens to be first in the list, so keep it and flag it.
    sel.insertBefore(missingModelOption(want), sel.firstChild);
    sel.value = want;
    warnMissingModel(want);
  } else if (want) {
    sel.value = want;
  } else {
    sel.value = S.models[0];
    if (S.chat) S.chat.model = S.models[0];  // only when the chat had no model at all
  }
  onModelChanged();
}
/** An <option> for a model the current server doesn't list, so selecting it stays
 *  possible and the mismatch is visible rather than silently corrected. */
function missingModelOption(model) {
  const o = document.createElement("option");
  o.value = model;
  o.textContent = model + "  (not on this server)";
  o.className = "model-missing";
  return o;
}
// One toast per model+server pair: refreshing models or a flapping server would
// otherwise re-warn about the same mismatch every few seconds.
const _missingModelWarned = new Set();
function warnMissingModel(model) {
  const key = model + "@" + currentServerUrl();
  if (_missingModelWarned.has(key)) return;
  _missingModelWarned.add(key);
  toast(`"${model}" isn't on this server — it's still selected.`, 5000);
}
async function onModelChanged() {
  const model = getSelectedModel();
  if (S.chat) S.chat.model = model;
  if (!model) {
    S.toolsSupported = null; S.visionSupported = null; S.imageOutput = false;
    updateToolsStatus(); updateVisionStatus(); return;
  }
  try {
    const r = await api(`/api/models/capabilities?server=${encodeURIComponent(currentServerUrl())}&model=${encodeURIComponent(model)}`);
    S.toolsSupported = r.tools;
    S.visionSupported = r.vision;
    S.imageOutput = !!r.image_output;
  } catch (e) { S.toolsSupported = null; S.visionSupported = null; S.imageOutput = false; }
  // A different model deserves a fresh warning if it also can't read images.
  S._imgWarned = false;
  updateToolsStatus();
  updateVisionStatus();
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
/** An export is plaintext JSON on purpose, so it opens on any machine. When it
 *  carries pictures, that is worth saying out loud at the moment it's written. */
function reportExport(r, what) {
  toast(what);
  (r.warnings || []).forEach((w) => toast(w, 7000));
  if (r.images) {
    toast(`${r.images} image(s) are embedded in that file — it is not encrypted.`, 8000);
  }
}
async function exportAllChats() {
  try {
    const r = await api("/api/chats/export", { method: "POST", body: { scope: "all" } });
    if (r.cancelled) return;
    reportExport(r, `Exported ${r.count} chat(s)`);
  } catch (e) { toast("Export failed: " + e.message); }
}
async function exportChat(id) {
  try {
    const r = await api(`/api/chats/${id}/export`, { method: "POST", body: {} });
    if (r.cancelled) return;
    reportExport(r, "Chat exported");
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
  applyPersonaSelection(chat);
  $("chk-rag").checked = !!chat.rag_enabled;
  $("chk-rag-auto").checked = !!chat.rag_auto;
  $("rag-threshold").value = chat.rag_threshold || 400;
  $("chk-multipass").checked = !!chat.multi_pass;
  $("mp-passes").value = chat.passes || 2;
  $("chk-mp-system").checked = chat.pass_use_system !== false;
  $("mp-eval-prompt").value = chat.eval_prompt || S.defaultEvalPrompt;
  $("chk-memory").checked = !!chat.memory_enabled;
  // A chat that predates the setting inherits the configured default rather than
  // silently reading as "scaled" while the user's preference says otherwise.
  if (chat.image_full_res === undefined) chat.image_full_res = !!S.config.image_full_res_default;
  updateImageResButton();
  // A chat that has never picked a core inherits the last one used, so turning the
  // toggle on just works instead of silently doing nothing.
  if (!chat.memory_core_id) chat.memory_core_id = S.lastMemoryCoreId || "";
  updateLibraryButton();
  updateWebsearchVisibility();
  updateMultipassVisibility();
  updateMemoryVisibility();
  // Staged blocks belong to the composer, pinned attachments to the chat — switching
  // chats swaps the pinned set and leaves whatever the user was staging alone.
  renderDataChips();
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
      images: m.images,
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
  // Images sit under the text: attached ones on a user turn, generated ones on an
  // assistant turn. Streaming appends to this strip as image frames arrive.
  const strip = document.createElement("div");
  strip.className = "bubble-images hidden";
  b.appendChild(strip);
  b._images = strip;
  (opts.images || []).forEach((rec) => addBubbleImage(b, rec));
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
/** Add one image thumbnail to a bubble's strip. `rec` is a stored-image record (or
 *  the bare {id} that a saved user turn keeps); clicking opens the full-size modal. */
function addBubbleImage(bubble, rec) {
  if (!rec || !rec.id) return;
  const img = document.createElement("img");
  img.src = imageUrl(rec.id, true);
  img.alt = rec.name || "";
  img.title = rec.name || "Click to view full size";
  img.onclick = () => showImage(rec, false);
  bubble._images.appendChild(img);
  bubble._images.classList.remove("hidden");
}

function makeRegenControls() {
  const wrap = document.createElement("span");
  const sel = document.createElement("select");
  const want = (S.chat && S.chat.model) || "";
  // Same rule as populateModels(): a model this server doesn't list still has to be
  // selectable, otherwise sel.value falls to "" and Regenerate sends no model at all.
  if (want && !S.models.includes(want)) sel.appendChild(missingModelOption(want));
  S.models.forEach((m) => {
    const o = document.createElement("option"); o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  sel.value = want || (S.models[0] || "");
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

/** The staged blocks that become <Data> XML — everything except images, which carry
 *  no text and go to the model as real image parts on the user turn. */
function textDataItems() {
  return S.dataItems.filter((it) => it.type !== "image");
}
/** The staged images, in the order they were attached. */
function stagedImages() {
  return S.dataItems.filter((it) => it.type === "image");
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

/** Render both kinds of chip into one strip: staged one-shot blocks from
 *  `S.dataItems`, then pinned attachments from `S.chat.attachments`. They look alike
 *  because they mean the same thing to the user — material riding along with the
 *  conversation — and differ only in whether Send consumes them. */
function renderDataChips() {
  const box = $("data-chips");
  box.innerHTML = "";
  const tags = dataItemTags();
  S.dataItems.forEach((it, i) => box.appendChild(makeDataChip(it, tags[i], false)));
  (S.chat?.attachments || []).forEach((a) => box.appendChild(makeDataChip(a, a.label, true)));
  box.classList.toggle("hidden",
    S.dataItems.length === 0 && !(S.chat?.attachments || []).length);
}

function makeDataChip(it, tagText, pinned) {
  const kind = SOURCE_KINDS.includes(it.type) ? it.type : "write";
  const isImage = kind === "image";
  const chip = document.createElement("span");
  chip.className = "data-chip" + (pinned ? " pinned" : "");

  const name = document.createElement("button");
  name.type = "button"; name.className = "data-chip-name";
  // An image chip leads with the picture: a filename is a much worse answer to
  // "which one did I attach?" than 18 pixels of the thing itself.
  if (isImage) {
    const thumb = document.createElement("img");
    thumb.className = "data-chip-thumb";
    thumb.src = imageUrl(it.id, true);
    thumb.alt = "";
    name.appendChild(thumb);
  }
  name.appendChild(document.createTextNode(tagText));
  name.title = kind === "write" && !pinned
    ? `Edit "${it.label}" block` : `View "${it.label}"`;
  name.onclick = () => {
    if (isImage) return showImage(it, pinned);
    return (kind === "write" && !pinned) ? editDataItem(it.id) : showAttachment(it, pinned);
  };

  if (kind !== "write") {
    const badge = document.createElement("span");
    badge.className = "data-chip-kind " + kind;
    badge.textContent = kind.toUpperCase();
    name.appendChild(badge);
  }

  const pin = document.createElement("button");
  pin.type = "button"; pin.className = "data-chip-pin";
  pin.textContent = pinned ? "📌" : "📍";
  pin.title = pinned
    ? "Pinned to this chat — click to make it one-shot again"
    : "Pin to this chat so it stays attached after sending";
  pin.onclick = () => (pinned ? unpinAttachment(it.id) : pinDataItem(it.id));

  const del = document.createElement("button");
  del.type = "button"; del.className = "data-chip-x";
  del.textContent = "✕"; del.title = "Remove";
  del.onclick = () => removeChip(it.id, pinned);

  chip.appendChild(name); chip.appendChild(pin); chip.appendChild(del);
  return chip;
}

function removeChip(id, pinned) {
  if (pinned) {
    S.chat.attachments = (S.chat.attachments || []).filter((a) => a.id !== id);
    persistChat();
  } else {
    S.dataItems = S.dataItems.filter((d) => d.id !== id);
  }
  renderDataChips();
}

/** Promote a staged block to a chat attachment: it survives Send and is re-sent as a
 *  system message on every turn (see logic.inject_attachments). */
function pinDataItem(id) {
  const it = S.dataItems.find((d) => d.id === id);
  if (!it) return;
  if (!S.chat) { toast("Open a chat first"); return; }
  S.chat.attachments = S.chat.attachments || [];
  // `image` rides along so the chip and modal can show dimensions without a fetch;
  // `content` stays empty, which is what makes the server's text attachment block
  // skip it (the bytes go out as real image parts instead).
  const att = { id: it.id, type: it.type || "write", label: it.label,
                content: it.text, source: it.source || "" };
  if (it.image) att.image = it.image;
  S.chat.attachments.push(att);
  S.dataItems = S.dataItems.filter((d) => d.id !== id);
  renderDataChips();
  persistChat();
}

function unpinAttachment(id) {
  const a = (S.chat?.attachments || []).find((x) => x.id === id);
  if (!a) return;
  S.chat.attachments = S.chat.attachments.filter((x) => x.id !== id);
  const item = { id: a.id, label: a.label, text: a.content,
                 type: a.type, source: a.source };
  if (a.image) item.image = a.image;
  S.dataItems.push(item);
  renderDataChips();
  persistChat();
}

/** Read-only look at fetched material, which is far too big to load into the composer. */
function showAttachment(it, pinned) {
  $("attachment-title").textContent = it.label || "Attachment";
  const src = $("attachment-source");
  const url = it.source || "";
  src.textContent = url; src.href = url || "#";
  src.classList.toggle("hidden", !url);
  $("attachment-text").value = it.content || it.text || "";
  $("btn-attachment-remove").onclick = () => { removeChip(it.id, pinned); closeModal(); };
  openModal("modal-attachment");
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

// --------------------------- chat-scoped sources ---------------------------
// The composer's ＋ Add menu. Each source fetches through the same services the
// Resources tab uses, but the result is staged as a chip rather than written to a
// library — for material that belongs to this one conversation.

function toggleComposerAdd(force) {
  const row = $("composer-add-row");
  const show = force !== undefined ? force : row.classList.contains("hidden");
  row.classList.toggle("hidden", !show);
  $("btn-composer-add").classList.toggle("active", show);
  if (!show) hideComposerPanels();
}
function showComposerPanel(which) {
  $("add-url-panel").classList.toggle("hidden", which !== "url");
  $("add-yt-panel").classList.toggle("hidden", which !== "youtube");
  $("add-search-panel").classList.toggle("hidden", which !== "search");
  if (which === "url") { $("add-url-input").value = ""; $("add-url-input").focus(); }
  if (which === "youtube") {
    $("add-yt-input").value = "";
    const p = $("add-yt-progress"); p.classList.add("hidden"); p.textContent = "";
    $("add-yt-input").focus();
  }
  if (which === "search") {
    $("add-search-query").value = ""; $("add-search-sites").value = "";
    const p = $("add-search-progress"); p.classList.add("hidden"); p.textContent = "";
    $("add-search-query").focus();
  }
}
function hideComposerPanels() {
  $("add-url-panel").classList.add("hidden");
  $("add-yt-panel").classList.add("hidden");
  $("add-search-panel").classList.add("hidden");
}

/** Stage fetched content as a one-shot chip. Labels come from the source's own title
 *  so a chip reads as "Interview with X" rather than "Data". */
function stageSource(type, label, text, source) {
  S.dataItems.push({ id: uid(), type, label: label || type, text: text || "",
                     source: source || "" });
  renderDataChips();
}

// --------------------------- images ---------------------------
// Unlike every other attachment, an image is bytes rather than text. It is stored
// server-side and referenced by id from here on: chips, bubbles and the modal all
// just point <img> at /api/images/<id>.

/** URL for a stored image; `small` asks for the cached thumbnail. */
function imageUrl(id, small) {
  return `/api/images/${encodeURIComponent(id)}` + (small ? "?thumb=1" : "");
}

/** Stage an uploaded/picked image as a one-shot chip. The chip's id IS the stored
 *  image's id, so pinning, sending and rendering all key off the same value. */
function stageImage(rec) {
  S.dataItems.push({ id: rec.id, type: "image", label: rec.name || "image",
                     text: "", source: rec.source || "", image: rec });
  if (rec.note) toast(`${rec.name}: ${rec.note}`, 5000);
  warnIfNoVision();
  renderDataChips();
}

/** Say once per session when the selected model can't read images. Deliberately not
 *  a block: the capability report is a heuristic for cloud models, and a server may
 *  proxy to a vision-capable backend behind a name we don't recognise. */
function warnIfNoVision() {
  if (S.visionSupported !== false || S._imgWarned) return;
  S._imgWarned = true;
  toast("This model doesn't appear to read images — sending anyway.", 6000);
}

/** Upload dropped/pasted image blobs. The only place in the app that uploads bytes. */
async function uploadImageBlobs(blobs) {
  const files = [...blobs].filter(Boolean);
  if (!files.length) return;
  setStatus(`Uploading ${files.length} image(s)…`);
  try {
    const r = await postFiles("/api/images/upload", files);
    (r.images || []).forEach(stageImage);
    if ((r.errors || []).length) toast("Some images failed: " + r.errors.join("; "), 6000);
    if ((r.images || []).length) toast(`Attached ${r.images.length} image(s)`);
  } catch (e) {
    toast("Image upload failed: " + e.message);
  }
  setStatus("");
}

/** Attach images through the native picker — the path for files already on disk. */
async function composerAddImages() {
  const progressEl = $("add-image-progress");
  let ui = null;
  setStatus("Waiting for image selection…");
  try {
    await streamSSE("/api/images/pick", {}, {
      begin: (d) => {
        setStatus("");
        if (!d.total) return;
        ui = makeProgressUI(progressEl, {});
        ui.plan([{ id: "images", label: "Reading images", weight: 1 }]);
        ui.line(`Reading ${d.total} image(s)…`);
      },
      progress: (d) => { if (ui) ui.update({ ...d, label: "Reading images" }); },
      complete: (d) => {
        const imgs = d.images || [];
        if (ui) { ui.finish(); ui.line(`Attached ${imgs.length} image(s).`); }
        imgs.forEach(stageImage);
        if (imgs.length) toast(`Attached ${imgs.length} image(s)`);
        if ((d.errors || []).length) toast("Some images failed: " + d.errors.join("; "), 6000);
        hideComposerPanels();
      },
      error: (d) => { if (ui) ui.stop(); toast("Attach images failed: " + d.message); },
      done: () => { if (ui) ui.stop(); },
    });
  } catch (e) { toast("Attach images failed: " + e.message); }
  setStatus("");
}

/** Full-size viewer for one image, with Save-to-disk and Remove. */
function showImage(it, pinned) {
  const rec = it.image || it;
  $("image-modal-title").textContent = rec.name || it.label || "Image";
  const bits = [];
  if (rec.width && rec.height) bits.push(`${rec.width} × ${rec.height}`);
  if (rec.bytes) bits.push(`${Math.round(rec.bytes / 1024).toLocaleString()} KB`);
  if (rec.media_type) bits.push(rec.media_type);
  if (rec.orig_media_type) bits.push(`converted from ${rec.orig_media_type}`);
  if (rec.source) bits.push(rec.source);
  $("image-modal-meta").textContent = bits.join(" · ");
  $("image-modal-img").src = imageUrl(it.id);
  $("btn-image-save").onclick = () => saveImageToDisk(it.id, rec.name);
  // A model's own picture has no chip to remove; only user attachments do.
  const removable = it.type === "image";
  $("btn-image-remove").classList.toggle("hidden", !removable);
  $("btn-image-remove").onclick = () => { removeChip(it.id, pinned); closeModal(); };
  openModal("modal-image");
}

async function saveImageToDisk(id, defaultName) {
  const btn = $("btn-image-save");
  btn.disabled = true;
  try {
    const r = await api(`/api/images/${encodeURIComponent(id)}/save`, {
      method: "POST", body: { default_name: defaultName || "image" },
    });
    if (r.ok) toast("Saved to " + r.path);
    else if (!r.cancelled) toast("Save failed" + (r.error ? ": " + r.error : ""));
  } catch (e) { toast("Save failed: " + e.message); }
  btn.disabled = false;
}

/** Drag-and-drop onto the composer, and Ctrl+V of a clipboard image. */
function setupImageDropPaste() {
  const row = $("input-row");
  const hasFiles = (dt) => !!dt && [...(dt.types || [])].includes("Files");
  ["dragenter", "dragover"].forEach((ev) => row.addEventListener(ev, (e) => {
    if (!hasFiles(e.dataTransfer)) return;
    e.preventDefault();
    row.classList.add("drop-target");
  }));
  ["dragleave", "drop"].forEach((ev) =>
    row.addEventListener(ev, () => row.classList.remove("drop-target")));
  row.addEventListener("drop", (e) => {
    const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    e.preventDefault();
    uploadImageBlobs(files);
  });
  // Only intercept a paste that actually carries an image — ordinary text paste
  // must keep working exactly as it does now.
  $("input-box").addEventListener("paste", (e) => {
    const imgs = [...(e.clipboardData?.items || [])]
      .filter((i) => i.kind === "file" && i.type.startsWith("image/"))
      .map((i) => i.getAsFile());
    if (!imgs.length) return;
    e.preventDefault();
    uploadImageBlobs(imgs);
  });
  // A drop that misses the composer would otherwise navigate away from the app,
  // silently losing whatever was typed.
  ["dragover", "drop"].forEach((ev) =>
    document.addEventListener(ev, (e) => { if (hasFiles(e.dataTransfer)) e.preventDefault(); }));
}

/** The 🖼 Scaled / 🖼 Full res toggle. Per-chat, defaulting from Settings. */
function updateImageResButton() {
  const full = !!(S.chat && S.chat.image_full_res);
  const btn = $("btn-image-res");
  const cap = S.config.image_max_dim || 1568;
  btn.textContent = full ? "🖼 Full res" : "🖼 Scaled";
  btn.classList.toggle("active", full);
  btn.title = full
    ? "Images are sent at their original resolution — more detail, much more context used."
    : `Images are scaled to ${cap}px on the longest edge before sending. Click for full resolution.`;
}
function toggleImageRes() {
  if (!S.chat) return;
  S.chat.image_full_res = !S.chat.image_full_res;
  updateImageResButton();
  persistChat();
}

function updateVisionStatus() {
  const el = $("vision-status");
  if (S.visionSupported === true) {
    el.textContent = "✓ reads images"; el.style.color = "var(--status-ok)";
  } else if (S.visionSupported === false) {
    el.textContent = "✗ no image input"; el.style.color = "var(--status-warn)";
  } else {
    el.textContent = "";   // unknown: say nothing rather than guess
  }
}

async function composerAddUrl() {
  const url = $("add-url-input").value.trim();
  if (!url) { toast("Enter a URL"); return; }
  const btn = $("btn-add-url-fetch"); btn.disabled = true;
  setStatus("Fetching " + url + " …");
  try {
    const r = await api("/api/fetch-url", { method: "POST", body: { url }});
    stageSource("url", r.title || r.url, r.text, r.url);
    toast(`Attached: ${r.title || r.url}${r.via ? " (via " + r.via + ")" : ""}`);
    hideComposerPanels();
  } catch (e) { toast("Fetch failed: " + e.message); }
  btn.disabled = false; setStatus("");
}

async function composerAddFiles() {
  const progressEl = $("add-files-progress");
  let ui = null;
  setStatus("Waiting for file selection…");
  try {
    await streamSSE("/api/extract-files", {}, {
      begin: (d) => {
        setStatus("");
        if (!d.total) return;
        ui = makeProgressUI(progressEl, {});
        ui.plan([{ id: "parse", label: "Reading documents", weight: 1 }]);
        ui.line(`Reading ${d.total} document(s)…`);
      },
      progress: (d) => { if (ui) ui.update({ ...d, label: "Reading documents" }); },
      complete: (d) => {
        const docs = d.docs || [];
        if (ui) { ui.finish(); ui.line(`Attached ${docs.length} file(s).`); }
        docs.forEach((doc) =>
          stageSource("file", doc.title || doc.filename, doc.text, doc.filename));
        if (docs.length) toast(`Attached ${docs.length} file(s)`);
        if ((d.errors || []).length) toast("Some files failed: " + d.errors.join("; "));
        hideComposerPanels();
      },
      error: (d) => { if (ui) ui.stop(); toast("Attach files failed: " + d.message); },
      done: () => { if (ui) ui.stop(); },
    });
  } catch (e) { toast("Attach files failed: " + e.message); }
  setStatus("");
}

let composerYtES = null;
function composerAddYouTube() {
  const url = $("add-yt-input").value.trim();
  if (!url) { toast("Enter a YouTube URL"); return; }
  const comments = $("add-yt-comments").checked;
  const max = Math.max(5, Math.min(2000, parseInt($("add-yt-max").value, 10) || 100));
  const prog = $("add-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Starting…";
  const btn = $("btn-add-yt-fetch"); btn.disabled = true;
  if (composerYtES) { composerYtES.close(); composerYtES = null; }
  composerYtES = youtubeStream("/api/youtube/fetch", { url, comments, max }, {
    progress: (d) => { prog.textContent = ytProgressText(d); },
    complete: (d) => {
      stageSource("youtube", d.title, d.text, d.url);
      toast(`Attached: ${d.title}${d.via ? " (via " + d.via + ")" : ""}` +
            (d.comment_count ? ` — ${d.comment_count} comment(s)` : ""));
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 6000);
      hideComposerPanels();
    },
    failed: (msg) => { toast("YouTube fetch failed: " + msg); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; composerYtES = null; },
  });
}

let composerSearchES = null;
function composerAddSearch() {
  const q = $("add-search-query").value.trim();
  if (!q) { toast("Enter a search term"); return; }
  const sites = $("add-search-sites").value.trim();
  // Clamped at both ends, like the Resources-tab crawl: the input's max="20" is not
  // enforced against a typed value. The server clamps to the same bound.
  const max = Math.max(1, Math.min(20, parseInt($("add-search-max").value, 10) || 5));
  const prog = $("add-search-progress");
  prog.classList.remove("hidden"); prog.textContent = "Searching Brave…";
  const goBtn = $("btn-add-search-go"); goBtn.disabled = true;
  if (composerSearchES) { composerSearchES.close(); composerSearchES = null; }
  const qs = new URLSearchParams({ q, sites, max: String(max) }).toString();
  const es = new EventSource(`/api/brave-search-text?${qs}`);
  composerSearchES = es;
  let gotFrame = false;
  const finish = () => { es.close(); if (composerSearchES === es) composerSearchES = null; goBtn.disabled = false; };
  es.addEventListener("start", () => { gotFrame = true; });
  es.addEventListener("progress", (ev) => {
    gotFrame = true;
    const d = JSON.parse(ev.data);
    prog.textContent = `Crawled ${d.done}/${d.target} — ${d.ok ? "✓" : "✗"} ${d.title || d.url}`;
  });
  es.addEventListener("complete", (ev) => {
    gotFrame = true;
    const d = JSON.parse(ev.data);
    (d.pages || []).forEach((p) => stageSource("search", p.title || p.url, p.text, p.url));
    toast(`Attached ${(d.pages || []).length} page(s) from ${d.attempted} result(s)` +
          ((d.errors || []).length ? `, ${d.errors.length} skipped` : ""));
    finish(); hideComposerPanels();
  });
  es.addEventListener("error", (ev) => {
    if (ev.data) { try { toast("Search error: " + (JSON.parse(ev.data).message || "unknown")); prog.textContent = "Search failed."; } catch (e) {} }
    else if (!gotFrame) {
      // A JSON error response (or an expired session) never reaches EventSource as
      // data, so say something rather than leaving a dead button.
      toast("Search failed — the server rejected the request (it may need a reload or login).");
      prog.textContent = "Search failed.";
    }
    finish();
  });
}

// Serialize staged blocks into a single <Data>…</Data> XML string ("" when none staged).
/** Serialize the staged data items into the <Data> block that gets prepended to the
 *  user turn. Returns "" when nothing is staged. */
function buildDataXml() {
  const tags = dataItemTags();
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Images are skipped, and not only because base64 in an XML tag would be useless:
  // <Data> is the RAG corpus path, so a leaked image would be chunked and embedded.
  const blocks = S.dataItems
    .map((it, i) => (it.type === "image"
      ? null : `  <${tags[i]}>\n${esc(it.text)}\n  </${tags[i]}>`))
    .filter(Boolean);
  if (!blocks.length) return "";
  return `<Data>\n${blocks.join("\n")}\n</Data>`;
}

/** Consume the staged one-shot blocks after a send. Pinned attachments live on the
 *  chat, not here, so they deliberately survive. */
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
 * Copy every chat-scoped control's current value out of the DOM and into S.chat.
 *
 * The individual oninput/onchange handlers in bindEvents() already do this as the
 * user types, but that makes "did my edit take effect?" depend on which event
 * happened to fire — a value set programmatically, an unblurred field, or a handler
 * that no-opped because S.chat was momentarily null all leave the chat stale. Calling
 * this at the top of every dispatch point makes the answer unconditional: what is on
 * screen when you press Send is what gets sent.
 *
 * Reverse of loadChatObject() — keep the two assignment lists in step.
 */
function syncSettingsFromUI() {
  if (!S.chat) return;
  const c = S.chat;
  c.server_url = currentServerUrl();
  c.model = getSelectedModel();
  c.system_prompt = $("system-prompt").value;
  c.system_on = $("system-on").checked;
  c.pre_prompt = $("pre-prompt").value;
  c.pre_on = $("pre-on").checked;
  c.num_ctx = parseInt($("ctx-select").value) || c.num_ctx;
  c.isolated = $("chk-isolate").checked;
  c.hide_thinking = $("chk-hide-thinking").checked;
  c.web_search = $("chk-websearch").checked;
  c.library_strict = $("chk-strict").checked;
  c.crawl_pages = parseInt($("crawl-pages").value) || S.minCrawledPages || 7;
  c.rag_enabled = $("chk-rag").checked;
  c.rag_auto = $("chk-rag-auto").checked;
  c.rag_threshold = Math.max(1, parseInt($("rag-threshold").value) || 400);
  c.multi_pass = $("chk-multipass").checked;
  c.passes = Math.max(1, parseInt($("mp-passes").value) || 2);
  c.pass_use_system = $("chk-mp-system").checked;
  c.eval_prompt = $("mp-eval-prompt").value;
  c.memory_enabled = $("chk-memory").checked;
  c.memory_core_id = $("memory-core-select").value || c.memory_core_id || "";
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
  // An image on its own is a complete message ("what is this?" is implied), so the
  // send guard has to look past the text.
  if (!text && !dataXml && !stagedImages().length) return;
  const model = getSelectedModel();
  if (!model) { toast("Please select a model first."); return; }
  if (!S.chat) await newPrivateChat(true);
  showChatView();
  syncSettingsFromUI();

  // Add user turn (data blocks first, then any prompt).
  const content = dataXml ? (text ? dataXml + "\n\n" + text : dataXml) : text;
  const msg = { role: "user", content };
  // Staged images become image parts on this turn — stored by id, so the chat stays
  // small and the picture survives a reload.
  const imgs = stagedImages();
  if (imgs.length) msg.images = imgs.map((it) => ({ id: it.id }));
  S.chat.messages.push(msg);
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
  const personaReady = S.usePersona && S.personaId &&
                       usablePersonas().some((p) => p.id === S.personaId);
  if (S.usePersona && !personaReady) toast("Persona unavailable — answering normally.");
  if (personaReady) {
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
  // Sync first, so the regen dropdown's model still wins over the topbar's.
  syncSettingsFromUI();
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

/** Personas that can actually be used — a broken persona.xml can't be loaded, so it must
 *  never be selectable in the composer (only listed in the tab, to be fixed or deleted). */
function usablePersonas() {
  return S.personas.filter((p) => !p.broken);
}

function renderPersonaControls() {
  const sel = $("persona-select");
  if (!sel) return;
  const usable = usablePersonas();
  sel.innerHTML = "";
  usable.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = p.name;
    sel.appendChild(o);
  });
  // When there are no usable personas the list may simply not have loaded yet, so keep
  // whatever a chat restored rather than clobbering it — loadPersonas() re-renders.
  if (S.personaId && usable.some((p) => p.id === S.personaId)) sel.value = S.personaId;
  else if (usable.length) { S.personaId = usable[0].id; sel.value = S.personaId; }
  sel.classList.toggle("hidden", !S.usePersona || !usable.length);
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
  // Keep the selection when it still exists on this persona; otherwise fall back to the
  // default voice rather than silently sending a variant the persona no longer has.
  // `!p` means the list hasn't loaded yet — don't discard a chat's restored choice.
  if (S.personaVariant && variants.includes(S.personaVariant)) vsel.value = S.personaVariant;
  else if (p) { S.personaVariant = ""; vsel.value = ""; }
  vsel.classList.toggle("hidden", !S.usePersona || !variants.length);
}

/** Mirror the composer's persona controls onto the open chat so the selection survives a
 *  reload and travels with the chat, like system_prompt and library_ids do. */
function savePersonaSelection() {
  if (!S.chat) return;
  S.chat.persona_on = !!S.usePersona;
  S.chat.persona_id = S.personaId || "";
  S.chat.persona_variant = S.personaVariant || "";
  persistChat();
}

/** Restore the persona controls from a chat being opened. */
function applyPersonaSelection(chat) {
  S.usePersona = !!(chat && chat.persona_on);
  S.personaId = (chat && chat.persona_id) || "";
  S.personaVariant = (chat && chat.persona_variant) || "";
  const chk = $("chk-persona");
  if (chk) chk.checked = S.usePersona;
  renderPersonaControls();
}

function onPersonaToggle() {
  S.usePersona = $("chk-persona").checked;
  if (S.usePersona && !usablePersonas().length) {
    toast("No personas yet — create one in the Personas tab.");
  }
  renderPersonaControls();
  savePersonaSelection();
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
 *
 * The run id comes from the card's own `.persona-run` wrapper, not from a global. With
 * a global, every step card in the chat pointed at the NEWEST run, so re-running a step
 * on an earlier answer re-ran a different pipeline and rendered it into the wrong message.
 */
async function rerunFromStep(index, card) {
  if (S.generating) return;
  const wrap = card.closest(".persona-run");
  const runId = wrap && wrap.dataset.runId;
  if (!runId) return;
  let output = null;
  const ta = card.querySelector(".step-json");
  const raw = ta.value.trim();
  if (raw) { try { output = JSON.parse(raw); } catch (e) { output = raw; } }
  // A re-run is a generation like any other: show Stop and block a second one.
  setGeneratingUI(true);
  S.runId = runId;
  await streamPersonaRun(`/api/runs/${runId}/rerun`, { index, output }, wrap, true,
                         null, { fromIndex: index, persist: wrap.dataset.persist !== "false" });
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
  wrap.dataset.runId = S.runId;   // re-runs target THIS answer's run, not the latest
  wrap.innerHTML = `<div class="persona-steps"></div>`;
  const bubble = makeBubble("assistant", "", { streaming: true });
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  bubble._body.textContent = "…";
  scrollBottom();
  await streamPersonaRun(`/api/personas/${S.personaId}/chat`,
    { chat: S.chat, run_id: S.runId, variant: S.personaVariant || "" }, wrap, false, bubble);
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
 * @param {HTMLElement} bubbleArg existing answer bubble; a re-run reuses the wrap's own
 * @param {object}      opts      {fromIndex} first re-executed step; {persist} false for
 *                                the editor's test-run panel, which owns no chat turn
 */
async function streamPersonaRun(path, body, wrap, isRerun, bubbleArg, opts = {}) {
  const stepsEl = wrap.querySelector(".persona-steps");
  const persist = opts.persist !== false;
  let bubble = bubbleArg;
  if (isRerun) {
    // Drop only the cards that are about to be re-executed; the upstream steps still
    // hold valid output and the server won't re-emit them (it resumes at fromIndex).
    const from = opts.fromIndex || 0;
    stepsEl.querySelectorAll(".step-card").forEach((c) => {
      if (Number(c.dataset.index) >= from) c.remove();
    });
    bubble = wrap.querySelector(".bubble.assistant") || makeBubble("assistant", "", { streaming: true });
    if (!wrap.contains(bubble)) wrap.appendChild(bubble);
    bubble._body.textContent = "…";
  }
  let finalText = "";
  // Seed from the surviving cards so a late frame for an upstream step still lands.
  const cards = {};
  stepsEl.querySelectorAll(".step-card").forEach((c) => { cards[Number(c.dataset.index)] = c; });
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
    run_paused: (d) => {
      setStatus(d && d.stopped ? "Pipeline stopped." : "Pipeline paused — edit a step and re-run.");
    },
    run_complete: (d) => { if (d.final) { finalText = d.final; bubble._body.textContent = finalText; } },
    error: (d) => {
      bubble._body.textContent = "[Error] " + d.message; toast("Persona error: " + d.message);
      // A run the server has forgotten (LRU eviction or a restart) can never be
      // resumed, so stop offering buttons that will only fail again.
      if (/run not found/i.test(d.message || "")) {
        wrap.querySelectorAll(".step-rerun").forEach((b) => {
          b.disabled = true; b.title = "This run has expired — send a new message.";
        });
      }
    },
    done: () => {},
  });
  bubble.classList.remove("streaming");
  setGeneratingUI(false); setStatus("");
  if (!finalText || !persist) return;
  if (isRerun) {
    // A re-run replaces the answer it corrected rather than appending a second one.
    // Target the turn THIS wrapper produced; fall back to the trailing assistant turn
    // if the transcript has shifted underneath us (edited or trimmed messages).
    const at = Number(wrap.dataset.msgIndex);
    let idx = (Number.isInteger(at) && S.chat.messages[at] &&
               S.chat.messages[at].role === "assistant") ? at : -1;
    for (let i = S.chat.messages.length - 1; idx < 0 && i >= 0; i--) {
      if (S.chat.messages[i].role === "assistant") idx = i;
    }
    if (idx >= 0) S.chat.messages[idx].content = finalText;
    else S.chat.messages.push({ role: "assistant", content: finalText });
    if (!S.chat.private) await persistChat(true);
    return;
  }
  // Save the final answer as the assistant turn (once), then persist.
  S.chat.messages.push({ role: "assistant", content: finalText });
  wrap.dataset.msgIndex = String(S.chat.messages.length - 1);
  if (!S.chat.private) await persistChat(true);
  // Offer to save this exchange as a persona memory.
  const btn = document.createElement("button");
  btn.className = "small ghost save-memory-btn";
  btn.textContent = "💾 Save as memory";
  const userMsg = [...S.chat.messages].reverse().find((m) => m.role === "user");
  btn.onclick = () => openSaveMemory([userMsg, { role: "assistant", content: finalText }]);
  wrap.appendChild(btn);
}

let _memPersonaId = "";
let _memDraftSeq = 0;
async function openSaveMemory(messages) {
  _memPersonaId = S.personaId;
  if (!_memPersonaId) { toast("No active persona."); return; }
  const placeholder = "…drafting…";
  $("mem-title").value = ""; $("mem-desc").value = placeholder;
  $("mem-weight").value = 5; $("mem-tags").value = "";
  const seq = ++_memDraftSeq;
  openModal("modal-memory");
  // The draft lands after a model round-trip. Don't overwrite anything the user
  // has typed in the meantime, and drop it entirely if they've moved on.
  const stillWanted = () => seq === _memDraftSeq && MODAL_STACK.some((m) => m.id === "modal-memory");
  const fill = (id, val) => {
    const el = $(id);
    const untouched = id === "mem-desc" ? el.value === placeholder : el.value === "";
    if (untouched) el.value = val;
  };
  try {
    const r = await api(`/api/personas/${_memPersonaId}/draft-memory`, {
      method: "POST",
      body: { messages, server_url: currentServerUrl(), model: getSelectedModel() },
    });
    if (!stillWanted()) return;
    const d = r.draft || {};
    fill("mem-title", d.title || "");
    fill("mem-desc", d.description || "");
    if ($("mem-weight").value === "5") $("mem-weight").value = d.emotional_weight || 5;
  } catch (e) {
    if (!stillWanted()) return;
    fill("mem-desc", (messages.map((m) => m.content).join("\n\n")).slice(0, 400));
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
    if (p.broken) {
      // Unparseable persona.xml. Listed rather than hidden so it can be inspected and
      // deleted — it still owns its id, sources and memories.
      row.classList.add("broken");
      row.innerHTML = `<span>⚠ ${escapeHtml(p.id)} <em>(unreadable)</em></span> <button class="small danger">✕</button>`;
      row.title = p.error || "persona.xml could not be parsed";
      row.onclick = () => toast(`"${p.id}" can't be opened: ${p.error || "invalid persona.xml"}`);
      row.querySelector("button").onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete the unreadable persona "${p.id}" and all its data?`)) return;
        await api(`/api/personas/${p.id}`, { method: "DELETE" });
        renderPersonaTab();
      };
    } else {
      row.textContent = p.name + (p.role ? ` — ${p.role}` : "");
      row.onclick = () => openPersonaEditor(p.id);
    }
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
  // `|| 0.7` turned a deliberate temperature of 0 (fully deterministic) into 0.7.
  const temp = parseFloat($("pe-temp").value);
  p.models.temperature = Number.isFinite(temp) ? temp : 0.7;
  p.stores.retrieval = $("pe-retrieval").value;
  p.stores.prompt_reword = $("pe-reword").checked;
  p.speaking.tone = $("pe-tone").value;
  p.speaking.formality = $("pe-formality").value;
  p.speaking.vocabulary = $("pe-vocab").value;
  p.speaking.quirks = $("pe-quirks").value;
  // variants + examples + steps are collected live into p by their editors
  return p;
}

/** Persist the open persona. Returns true on success so callers (the test run) can
 *  bail instead of testing a definition that was never saved. */
async function savePersona() {
  const p = collectPersona(); if (!p) return false;
  // The server rejects this too, but catching it here keeps the name field's contents
  // in front of the user instead of round-tripping an error.
  if (!p.profile.name) { toast("Persona name is required."); $("pe-name").focus(); return false; }
  if (!collectPeSteps()) return false;   // validates step JSON schemas
  try {
    const r = await api(`/api/personas/${p.id}`, { method: "PUT", body: { persona: p } });
    S.editingPersona = r.persona;
    toast("Saved.");
    renderPersonaTab();
    return true;
  } catch (e) { toast("Save failed: " + e.message); return false; }
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
    // Collect the DOM into the array FIRST, then mutate the array. Doing it the other
    // way round (swap, then collect) rebuilt the array from the un-swapped cards, so
    // ↑/↓ were silent no-ops and ✕ discarded every unsaved edit in the other steps.
    const editStructure = (fn) => {
      if (!collectPeSteps()) return;
      fn(S.editingPersona.pipeline);
      renderPeSteps();
    };
    card.querySelector(".s-up").onclick = () =>
      editStructure((a) => { if (i > 0) [a[i-1], a[i]] = [a[i], a[i-1]]; });
    card.querySelector(".s-down").onclick = () =>
      editStructure((a) => { if (i < a.length - 1) [a[i+1], a[i]] = [a[i], a[i+1]]; });
    card.querySelector(".s-del").onclick = () => editStructure((a) => a.splice(i, 1));
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
  const pid = S.editingPersona.id;
  // The server mints the run id (unique per invocation) and hands it back in `begin`;
  // Cancel has to use that one or it stops nothing.
  let runId = "";
  const progressEl = $("pe-compile-progress");
  let ui = null;
  try {
    // Each document here is parsed AND embedded, so this is the slow path that most
    // needs a progress bar.
    await streamSSE(`/api/personas/${pid}/knowledge/add-files`, {}, {
      begin: (d) => {
        runId = d.run_id || runId;
        if (!d.total) return;
        ui = makeProgressUI(progressEl, {
          onCancel: () => runId && api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {}),
        });
        ui.plan([{ id: "parse", label: "Ingesting documents", weight: 1 }]);
        ui.line(`Ingesting ${d.total} document(s)…`);
      },
      progress: (d) => { if (ui) ui.update({ ...d, label: "Ingesting documents" }); },
      complete: (r) => {
        if (ui) ui.finish();
        if (r.errors && r.errors.length) toast(r.errors.join("; "));
        else toast(`Added ${(r.added || []).length} document(s).`);
        renderPersonaKb(); renderEmbedBanner();
        refreshCompileStatus("persona", pid, $("pe-compile-badge"));
      },
      error: (d) => { if (ui) ui.stop(); toast("Add failed: " + d.message); },
      done: () => { if (ui) ui.stop(); },
    });
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
  if (S.generating || S.batchRunning) { toast("Wait for the current run to finish."); return; }
  const msg = $("pe-test-msg").value.trim(); if (!msg) return;
  if (!collectPeSteps()) return;
  // Save first so the test uses the current definition. A rejected save (e.g. a blank
  // name) would otherwise test the LAST saved definition while showing the new one.
  if (!(await savePersona())) return;
  const wrap = $("pe-test-output"); wrap.innerHTML = `<div class="persona-steps"></div>`;
  const bubble = makeBubble("assistant", "", { streaming: true }); wrap.appendChild(bubble); bubble._body.textContent = "…";
  S.runId = uid(); S.personaRunId = S.runId; S.generating = true;
  wrap.dataset.runId = S.runId;
  wrap.dataset.persist = "false";   // a re-run here must not touch the open chat either
  const syntheticChat = { id: "test", private: true, messages: [{ role: "user", content: msg }],
                          server_url: currentServerUrl(), model: getSelectedModel() };
  const savedChat = S.chat; S.chat = syntheticChat;   // streamPersonaRun reads S.chat
  // persist:false — the test panel is not a chat turn, so nothing is appended or saved.
  await streamPersonaRun(`/api/personas/${S.editingPersona.id}/chat`,
    { chat: syntheticChat, run_id: S.runId }, wrap, false, bubble, { persist: false });
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
  $("btn-pe-pipeline-default").onclick = async () => {
    if (!confirm("Replace this persona's pipeline with the default 5 steps? Your current steps will be lost.")) return;
    const r = await api("/api/persona-default-pipeline");
    S.editingPersona.pipeline = r.pipeline; renderPeSteps();
    toast("Default pipeline restored (save to keep).");
  };
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
  let curImages = [];
  let errored = false;

  function finalizePass() {
    if (!bubble) return;
    bubble.classList.remove("streaming");
    if (!errored) {
      const msg = { role: "assistant", content: curContent };
      if (curReason) msg.reasoning = curReason;
      if (curImages.length) msg.images = curImages;
      if (curLabel) { msg.pass_label = curLabel; msg.intermediate = curIntermediate; }
      S.chat.messages.push(msg);
    }
    bubble = null; reasonBubble = null;
    curContent = ""; curReason = ""; curLabel = ""; curIntermediate = false;
    curImages = [];
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
      curImages = [];
      bubble = makeBubble("assistant", "", { streaming: true, label: curLabel, intermediate: curIntermediate });
      $("messages").appendChild(bubble);
      if (follow) scrollBottom();
    },
    // The server has already stored the picture; the frame carries its record, so
    // this only has to point an <img> at it.
    image: (d) => {
      if (!bubble || !d.id) return;
      const follow = isNearBottom($("messages"));
      curImages.push(d);
      addBubbleImage(bubble, d);
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
  if (!errored) await maybeExtractMemories();
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
  warnPersonaIgnoredInBulk();
  if (!S.chat) await newPrivateChat(true);
  syncSettingsFromUI();

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
function buildChatSnapshot(text, model, images) {
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
    // Pinned attachments come along: material pinned to the conversation is meant to
    // be in view for every prompt run from it, queued and batched ones included.
    attachments: (src.attachments || []).slice(),
    image_full_res: !!src.image_full_res,
    messages: [{ role: "user", content: text,
                 ...((images || []).length ? { images: images } : {}) }],
  };
}
/** Queue, Batch, and the parallel lanes all run through the ordinary generation path —
 *  none of them carry a persona. Say so once per session rather than silently handing
 *  back plain answers while the "Use Persona" toggle is lit. */
let _personaBulkWarned = false;
function warnPersonaIgnoredInBulk() {
  if (!S.usePersona || _personaBulkWarned) return;
  _personaBulkWarned = true;
  toast("Queue and Batch don't use personas — these will be answered normally.", 5000);
}

function addToQueue() {
  const text = $("input-box").value.trim();
  const dataXml = buildDataXml();
  const imgs = stagedImages();
  if (!text && !dataXml && !imgs.length) { toast("Type a prompt or add data to queue"); return; }
  warnPersonaIgnoredInBulk();
  syncSettingsFromUI();   // the snapshot below must capture what's on screen now
  const model = getSelectedModel();
  let sq = "";
  if ($("chk-websearch").checked) { sq = $("search-query").value.trim(); $("search-query").value = ""; }
  const content = dataXml ? (text ? dataXml + "\n\n" + text : dataXml) : text;
  const chat = buildChatSnapshot(content, model, imgs.map((it) => ({ id: it.id })));
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
    const turn = items[i].chat.messages[0];
    S.chat.messages.push({ role: "user", content: turn.content,
                           ...(turn.images ? { images: turn.images } : {}) });
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

// ---------------- RAG vector-store backend (Settings → RAG) ----------------
// Two stores can hold data at once and they are independent files, so switching is
// reversible. The note under the selector states what each one costs, because the
// LanceDB store is NOT encrypted and that must not be a surprise.
async function refreshRagBackend() {
  const note = $("rag-backend-note");
  if (!note) return;
  let info;
  try { info = await api("/api/rag/backend"); }
  catch (e) { note.textContent = ""; return; }
  $("set-rag-backend").value = info.backend || "lance";
  const st = info.status || {};
  const bits = [];
  if (info.backend === "lance") {
    bits.push(`<strong>Not encrypted.</strong> Chunk text and embeddings are stored in plain files at <code>${escapeHtml(info.lance_dir || "")}</code>, readable without your login password.`);
    if (st.rows) bits.push(`${(st.rows).toLocaleString()} chunk(s) indexed.`);
  } else {
    bits.push(`Encrypted at rest with your login password. No durable vector index, and keyword search scans the whole scope — noticeably slower on large corpora.`);
  }
  if (info.duckdb_rows && info.backend === "lance") {
    bits.push(`Your DuckDB store still holds ${info.duckdb_rows.toLocaleString()} chunk(s) — ` +
              `<button id="btn-rag-migrate" class="small">Copy them into LanceDB</button> ` +
              `to avoid re-embedding.`);
  }
  note.innerHTML = bits.join(" ");
  const mig = $("btn-rag-migrate");
  if (mig) mig.onclick = migrateRagStore;
}
async function migrateRagStore() {
  const progressEl = $("rag-migrate-progress");
  const ui = makeProgressUI(progressEl, {
    onCancel: () => api("/api/stop", { method: "POST", body: { run_id: "rag-migrate" } }).catch(() => {}),
  });
  ui.plan([{ id: "migrate", label: "Copying vectors", weight: 1 }]);
  await streamSSE("/api/rag/migrate", {}, {
    begin: (d) => ui.line(`Copying ${(d.total || 0).toLocaleString()} chunk(s) from DuckDB…`),
    progress: (d) => ui.update({ ...d, label: "Copying vectors" }),
    warn: (d) => ui.line(`   ⚠ ${d.name}: ${d.message}`),
    complete: (d) => {
      ui.finish();
      ui.line(`Copied ${(d.moved || 0).toLocaleString()} of ${(d.total || 0).toLocaleString()}; verified: ${d.verified ? "yes" : "NO"}.`);
      if (d.verified) toast(`Migrated ${(d.moved || 0).toLocaleString()} chunks to LanceDB.`);
      else toast("Migration finished but verification failed — your DuckDB store is untouched.", 6000);
      refreshRagBackend();
    },
    error: (d) => { ui.stop(); toast("Migration failed: " + d.message); },
    done: () => ui.stop(),
  });
}

// ---------------- RAG embedding servers (Settings → RAG) ----------------
// A list separate from the chat parallel_servers: a box with chat models often has no
// embedding model pulled. There is intentionally no per-server model — the whole pool
// embeds with rag_embed_model, because vectors from different embedding models live in
// different spaces and mixing them would corrupt the index.
function renderRagServers(health) {
  const box = $("rag-servers-list");
  const list = S.ragServers || [];
  if (!list.length) {
    box.className = "muted";
    box.textContent = "No extra embedding servers — the embedding server URL above is used on its own.";
    return;
  }
  box.className = "";
  const byUrl = {};
  ((health || {}).servers || []).forEach((h) => { byUrl[h.base_url] = h; });
  box.innerHTML = "";
  list.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "rag-server-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = s.enabled !== false;
    cb.onchange = () => { S.ragServers[i].enabled = cb.checked; };
    row.appendChild(cb);
    const url = document.createElement("span");
    url.className = "url";
    url.textContent = s.base_url;
    row.appendChild(url);
    const h = byUrl[s.base_url];
    if (h) {
      const st = document.createElement("span");
      if (!h.reachable) { st.className = "health bad"; st.textContent = "✗ unreachable"; }
      else if (!h.has_model) { st.className = "health warn"; st.textContent = "⚠ embedding model not pulled"; }
      else { st.className = "health ok"; st.textContent = "✓ ready"; }
      row.appendChild(st);
    }
    const del = document.createElement("button");
    del.className = "small";
    del.textContent = "✕";
    del.onclick = () => { S.ragServers.splice(i, 1); renderRagServers(health); };
    row.appendChild(del);
    box.appendChild(row);
  });
}
function addRagServersFromList() {
  const existing = new Set((S.ragServers || []).map((s) => s.base_url));
  const primary = ($("set-rag-embed-url").value || "").trim().replace(/\/+$/, "");
  let added = 0;
  (S.config.servers || []).forEach((s) => {
    const url = (s.base_url || "").replace(/\/+$/, "");
    // Only Ollama speaks /api/embed, and the primary is already in the pool.
    if (!url || url === primary || existing.has(url) || s.type !== "ollama") return;
    S.ragServers.push({ base_url: url, enabled: true });
    existing.add(url);
    added++;
  });
  renderRagServers();
  toast(added ? `Added ${added} server(s) — Save general to keep them.`
              : "No new Ollama servers to add (configure them under Servers first).");
}
async function checkRagServers() {
  setStatus("Checking embedding servers…");
  try {
    const h = await api("/api/rag/embed-health");
    renderRagServers(h);
    const bad = (h.servers || []).filter((s) => !s.reachable || !s.has_model);
    if (!bad.length) toast(`All ${h.servers.length} server(s) ready for "${h.embed_model}".`);
    else toast(`${bad.length} server(s) not ready — see the list.`, 5000);
  } catch (e) { toast("Check failed: " + e.message); }
  setStatus("");
}

// --------------------- prompt library (System/Pre) -------------------
// A 3-column browser (Group -> Category -> Prompt) over two independent trees,
// S.prompts.system and S.prompts.pre. The whole tree is saved back on every edit.
// `onUse` lets a caller outside the chat (the Batch tab) receive the chosen prompt
// instead of having it written into S.chat and the chat's textareas.
const PB = { kind: "system", groupId: null, catId: null, promptId: null, pendingText: null,
             busy: false, onUse: null };

function promptTree(kind) {
  if (!S.prompts) S.prompts = { system: [], pre: [] };
  if (!Array.isArray(S.prompts[kind])) S.prompts[kind] = [];
  return S.prompts[kind];
}
function pbGroup() { return promptTree(PB.kind).find((g) => g.id === PB.groupId); }
function pbCat() { const g = pbGroup(); return g && (g.categories || []).find((c) => c.id === PB.catId); }
function pbPromptItem() { const c = pbCat(); return c && (c.prompts || []).find((p) => p.id === PB.promptId); }

/** Whole-tree save. Returns false (and says so) if it didn't reach disk. */
async function savePrompts() {
  try {
    const r = await api("/api/prompts", { method: "POST", body: { prompts: S.prompts } });
    if (r && r.prompts) S.prompts = r.prompts;
    return true;
  } catch (e) {
    toast("Could not save the prompt library: " + e.message, 5000);
    return false;
  }
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

// The name box now stacks on top of the browser, so none of these has to tear the
// library down and rebuild it. PB.busy stops a second ＋ from racing the first.
async function pbNewGroup() {
  if (PB.busy) return;
  PB.busy = true;
  try {
    const name = await promptModal("New group name", "New Group");
    if (!name || !name.trim()) return;
    const g = { id: uid(), name: name.trim(), categories: [] };
    promptTree(PB.kind).push(g);
    PB.groupId = g.id; PB.catId = null; PB.promptId = null;
    await savePrompts(); renderPb();
  } finally { PB.busy = false; }
}
async function pbNewCat() {
  if (PB.busy) return;
  const g = pbGroup(); if (!g) { toast("Select a group first"); return; }
  PB.busy = true;
  try {
    const name = await promptModal("New category name", "New Category");
    if (!name || !name.trim()) return;
    const c = { id: uid(), name: name.trim(), prompts: [] };
    g.categories = g.categories || []; g.categories.push(c);
    PB.catId = c.id; PB.promptId = null;
    await savePrompts(); renderPb();
  } finally { PB.busy = false; }
}
async function pbNewPrompt() {
  if (PB.busy) return;
  const c = pbCat(); if (!c) { toast("Select a category first"); return; }
  PB.busy = true;
  try {
    const name = await promptModal("New prompt name", "New Prompt");
    if (!name || !name.trim()) return;
    const text = PB.pendingText != null ? PB.pendingText : "";
    const p = { id: uid(), name: name.trim(), prompt: text };
    c.prompts = c.prompts || []; c.prompts.push(p);
    PB.promptId = p.id; PB.pendingText = null;
    await savePrompts(); renderPb();
  } finally { PB.busy = false; }
}
async function pbRename(level, obj) {
  if (PB.busy) return;
  PB.busy = true;
  try {
    const name = await promptModal("Rename", obj.name || "");
    if (name == null || !name.trim()) return;
    obj.name = name.trim();
    await savePrompts(); renderPb();
  } finally { PB.busy = false; }
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
  const p = pbPromptItem(); if (!p) return;
  if (PB.onUse) { PB.onUse(p.prompt, PB.kind); closeModal(); return; }
  if (!S.chat) return;
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

function openPromptBrowser(kind, saveText, onUse) {
  if (PB.kind !== kind) { PB.groupId = null; PB.catId = null; PB.promptId = null; }
  PB.kind = kind;
  PB.pendingText = (saveText != null) ? saveText : null;
  PB.onUse = onUse || null;
  renderPb();
  openModal("modal-prompt-browser");
  if (saveText != null) toast("Pick a category, then ＋ on Prompts to save the current text.");
}

// ------------------------------- memory ------------------------------
// A memory core is what the assistant has learned about the USER, grown from chats and
// edited in the Memory tab. Selection is per-chat (toggle + core dropdown); the whole
// core is injected server-side. Unrelated to the persona Memories sub-tab above.

/** The six built-in categories, for the window between boot and /api/state landing.
 *  Server-side these live in memory.CATEGORIES. */
const MEMORY_CATEGORIES_FALLBACK = {
  preferences: "Preferences", facts: "Facts", interests: "Interests",
  style: "Communication style", goals: "Goals & projects", other: "Other",
};
function memoryCategories() {
  return Object.keys(S.memoryCategories).length ? S.memoryCategories : MEMORY_CATEGORIES_FALLBACK;
}

/** Replace a core in S.memoryCores with the server's copy and refresh anything showing it.
 *  A core we don't know about is ignored rather than added: a pass that was already in
 *  flight when the core was deleted would otherwise resurrect it in the list, leaving the
 *  tab showing something the server no longer has. */
function applyMemoryCore(core) {
  if (!core || !core.id) return;
  const i = S.memoryCores.findIndex((c) => c.id === core.id);
  if (i < 0) return;
  S.memoryCores[i] = core;
  if ($("tab-memory").classList.contains("active")) renderMemoryTab();
}

// ---- chat controls ----
/**
 * Fill the core dropdown and make sure it agrees with the chat. When the chat's core is
 * missing (deleted elsewhere, or a stale id from another profile) the browser would
 * otherwise leave the first option selected, so the control claimed a core the chat was
 * not using and nothing was ever injected. Settle on one id and write it back.
 */
function populateMemoryCoreSelect() {
  const sel = $("memory-core-select");
  const want = (S.chat && S.chat.memory_core_id) || S.lastMemoryCoreId || "";
  sel.innerHTML = "";
  S.memoryCores.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = c.name + ` (${(c.entries || []).length})`;
    sel.appendChild(o);
  });
  if (!S.memoryCores.length) { sel.value = ""; return; }
  const effective = S.memoryCores.some((c) => c.id === want) ? want : S.memoryCores[0].id;
  sel.value = effective;
  S.lastMemoryCoreId = effective;
  if (S.chat && S.chat.memory_core_id !== effective) {
    S.chat.memory_core_id = effective;
    // Only worth a write when the chat is actually using memory; otherwise this is
    // just the default the dropdown will offer if the toggle is ever switched on.
    if (S.chat.memory_enabled) persistChat();
  }
}

function updateMemoryVisibility() {
  const on = $("chk-memory").checked;
  populateMemoryCoreSelect();
  $("memory-core-select").classList.toggle("hidden", !on);
  $("btn-memory-extract").classList.toggle("hidden", !on);
}

function onMemoryToggle() {
  const on = $("chk-memory").checked;
  if (on && !S.memoryCores.length) {
    // Nothing to enable yet — send them where cores are made rather than silently
    // turning on a toggle that would do nothing.
    $("chk-memory").checked = false;
    updateMemoryVisibility();
    toast("Create a memory core first.");
    switchTab("memory");
    return;
  }
  if (S.chat) {
    S.chat.memory_enabled = on;
    if (on && !S.chat.memory_core_id) {
      S.chat.memory_core_id = S.lastMemoryCoreId || S.memoryCores[0].id;
    }
    persistChat();
  }
  updateMemoryVisibility();
}

/**
 * Run one extraction pass over the active chat. `mode` is "auto" (the every-N-replies
 * trigger) or "manual" (the Extract button). An auto pass never surfaces its failures:
 * it rides along behind an ordinary send and must not interrupt the conversation.
 */
async function runMemoryExtract(mode) {
  const chat = S.chat;
  if (!chat) return false;
  // A saved chat is already on the server, so posting the whole thing back — every
  // message, every turn — just to have it truncated to a 12k-char tail is wasted
  // bandwidth. Only a private chat, which has no stored copy, has to send itself.
  const body = { chat_id: chat.id, core_id: chat.memory_core_id, mode };
  if (chat.private) body.chat = chat;
  try {
    const r = await api("/api/memory/extract", { method: "POST", body });
    if (r.core) applyMemoryCore(r.core);
    if (r.error) { if (mode === "manual") toast(r.error); return false; }
    if (r.skipped) {
      if (mode === "manual") {
        toast(r.skipped === "incognito"
          ? "Skipped — an incognito session never writes to a memory core."
          : "Skipped — private chat.");
      }
      return false;
    }
    if (r.text) toast("🧠 Memory updated: " + r.text);
    else if (mode === "manual") toast("🧠 Nothing new worth remembering here.");
    return true;
  } catch (e) {
    if (mode === "manual") toast("Memory: " + e.message);
    return false;
  }
}

/** Called after every successful generation: count the turn, extract on the Nth.
 *  The count is the server's — it zeroes `memory_turns_since` when a pass actually
 *  runs — so a reload mid-cycle and a second tab on the same chat can't skew it. */
async function maybeExtractMemories() {
  const chat = S.chat;
  if (!chat || !chat.memory_enabled || !chat.memory_core_id || chat.private) return;
  const core = S.memoryCores.find((c) => c.id === chat.memory_core_id);
  if (!core || !core.auto_extract) return;
  chat.memory_turns_since = (chat.memory_turns_since || 0) + 1;
  if (chat.memory_turns_since < Math.max(1, core.extract_every || 6)) {
    persistChat();
    return;
  }
  // Don't zero the counter on the way in: a pass that fails (model down, server error)
  // would otherwise cost a full cycle of silence before the next attempt.
  if (await runMemoryExtract("auto")) chat.memory_turns_since = 0;
  // persistChat writes S.chat — only ours if the user hasn't switched chats meanwhile.
  if (S.chat === chat) persistChat();
}

async function extractMemoriesNow() {
  const chat = S.chat;
  if (!chat) { toast("Open a chat first."); return; }
  if (!chat.memory_core_id) { toast("Pick a memory core first."); return; }
  if (!(chat.messages || []).length) { toast("Nothing to learn from yet."); return; }
  if (chat.private && !confirm(
      "This is a private chat.\n\nSave what the assistant learns from it into the memory core?")) return;
  const btn = $("btn-memory-extract");
  btn.disabled = true; btn.textContent = "🧠 Learning…";
  // The server reads the stored copy, so flush any debounced edits before it does.
  await persistChat(true);
  if (await runMemoryExtract("manual")) chat.memory_turns_since = 0;
  btn.disabled = false; btn.textContent = "🧠 Extract";
  if (S.chat === chat) persistChat();
}

// ---- Memory tab ----
/** Re-read the cores when entering the tab, so edits from another browser tab or from a
 *  background extraction pass don't leave this one showing a stale core. */
async function refreshMemoryCores() {
  try {
    const r = await api("/api/memory/cores");
    S.memoryCores = r.cores || [];
    if (r.categories) S.memoryCategories = r.categories;
    renderMemoryTab();
    populateMemoryCoreSelect();
  } catch (e) { /* the tab still renders from what we already have */ }
}

function renderMemoryTab() {
  const has = S.memoryCores.length > 0;
  if (has && !S.memoryCores.some((c) => c.id === S.activeMemoryCore)) {
    S.activeMemoryCore = S.memoryCores[0].id;
  }
  if (!has) S.activeMemoryCore = "";
  renderMemoryCoreList();
  // Nothing written during an incognito session reaches disk unless the session is
  // saved, and the Memory tab is otherwise fully editable — so say so.
  $("mc-incognito").classList.toggle(
    "hidden", !(S.profiles && S.profiles.data && S.profiles.data.incognito));
  $("mc-empty").classList.toggle("hidden", has);
  $("mc-editor").classList.toggle("hidden", !has);
  if (has) renderMemoryEditor();
}

function renderMemoryCoreList() {
  const list = $("mc-list");
  list.innerHTML = "";
  S.memoryCores.forEach((c) => {
    const row = document.createElement("div");
    row.className = "side-item" + (c.id === S.activeMemoryCore ? " active" : "");
    row.textContent = `${c.name} — ${(c.entries || []).length}`;
    row.title = c.name;
    row.onclick = () => { S.activeMemoryCore = c.id; renderMemoryTab(); };
    list.appendChild(row);
  });
}

function activeCore() {
  return S.memoryCores.find((c) => c.id === S.activeMemoryCore) || null;
}

function renderMemoryEditor() {
  const core = activeCore();
  if (!core) return;
  $("mc-name").value = core.name || "";
  $("mc-auto").checked = core.auto_extract !== false;
  $("mc-every").value = core.extract_every || 6;
  $("mc-limit").value = core.inject_limit || 40;

  const entries = core.entries || [];
  // `injected` and `est_tokens` come from the server, which owns the selection rule and
  // counts the preamble the block is wrapped in — the tab used to recompute both and
  // quietly disagree with what was actually sent.
  const sent = entries.filter((e) => e.injected).length;
  $("mc-stats").textContent =
    `${entries.length} memories · ${sent} sent · ~${core.est_tokens || 0} tokens`;
  renderMemoryEntries();
}

function renderMemoryEntries() {
  const core = activeCore();
  const wrap = $("mc-entries");
  wrap.innerHTML = "";
  const entries = (core && core.entries) || [];
  if (!entries.length) {
    wrap.innerHTML = `<p class="muted mc-none">Nothing learned yet. Turn on 🧠 Memory in a
      chat and it will fill in as you talk — or add something yourself with ＋ Add memory.</p>`;
    return;
  }
  const cats = memoryCategories();

  Object.keys(cats).forEach((cat) => {
    const rows = entries.filter((e) => (e.category || "other") === cat)
      .sort((a, b) => (b.importance || 5) - (a.importance || 5));
    if (!rows.length) return;
    const sec = document.createElement("div");
    sec.className = "mc-cat";
    sec.innerHTML = `<div class="mc-cat-head">${escapeHtml(cats[cat])}</div>` +
      rows.map((e) => `
        <div class="mc-entry${e.injected ? "" : " muted-entry"}" data-id="${e.id}">
          <span class="mc-importance" title="Importance">${e.importance || 5}</span>
          <span class="mc-text">${escapeHtml(e.text || "")}</span>
          <span class="mc-tags">
            ${e.pinned ? '<span class="mc-flag" title="Always sent to the model">📌</span>' : ""}
            ${e.origin === "user" ? '<span class="mc-flag" title="You wrote this">✍</span>' : ""}
            ${e.injected ? "" : '<span class="mc-flag" title="Below the injection limit — stored but not sent">💤</span>'}
          </span>
          <button class="icon mc-edit" title="Edit">✎</button>
          <button class="icon mc-del" title="Delete">🗑</button>
        </div>`).join("");
    wrap.appendChild(sec);
  });

  wrap.querySelectorAll(".mc-edit").forEach((b) => {
    b.onclick = () => openMemoryEntry(b.closest(".mc-entry").dataset.id);
  });
  wrap.querySelectorAll(".mc-del").forEach((b) => {
    b.onclick = () => deleteMemoryEntry(b.closest(".mc-entry").dataset.id);
  });
}

async function newMemoryCore() {
  const name = await promptModal("Name this memory core", "My memories");
  if (name === null) return;
  try {
    const r = await api("/api/memory/cores", { method: "POST", body: { name } });
    S.memoryCores = r.cores;
    S.activeMemoryCore = r.core.id;
    S.lastMemoryCoreId = S.lastMemoryCoreId || r.core.id;
    renderMemoryTab();
    updateMemoryVisibility();
    toast("Memory core created");
  } catch (e) { toast("Could not create: " + e.message); }
}

/** Save the core's name and tuning. Debounced by the caller's change events only —
 *  cores are tiny, so a write per edit is cheap. */
async function saveMemoryCoreSettings() {
  const core = activeCore();
  if (!core) return;
  try {
    const r = await api(`/api/memory/cores/${core.id}`, { method: "PATCH", body: {
      name: $("mc-name").value,
      auto_extract: $("mc-auto").checked,
      // Clamp to the same bounds normalize_core enforces, and show the result — typing
      // 0 used to come back as 6 with no indication the value had been replaced.
      extract_every: clampField("mc-every", 1, 100, 6),
      inject_limit: clampField("mc-limit", 1, 500, 40),
    }});
    S.memoryCores = r.cores;
    renderMemoryTab();
    populateMemoryCoreSelect();
  } catch (e) { toast("Could not save: " + e.message); }
}

/** Read a numeric input, clamp it into range, and write the clamped value back so the
 *  field shows what was actually saved. */
function clampField(id, lo, hi, fallback) {
  const raw = parseInt($(id).value);
  const val = Math.max(lo, Math.min(hi, Number.isNaN(raw) ? fallback : raw));
  $(id).value = val;
  return val;
}

async function deleteMemoryCore() {
  const core = activeCore();
  if (!core) return;
  if (!confirm(`Delete the memory core “${core.name}” and everything in it?\n\nThis cannot be undone.`)) return;
  try {
    const r = await api(`/api/memory/cores/${core.id}`, { method: "DELETE" });
    S.memoryCores = r.cores;
    if (r.chats) { S.chats = r.chats; renderChatList(); }
    if (S.lastMemoryCoreId === core.id) S.lastMemoryCoreId = (S.memoryCores[0] || {}).id || "";
    // The server has already cleared the id and the toggle off every stored chat that
    // referenced this core — leaving it behind used to strand those chats reading
    // "memory on" while nothing was ever injected. Mirror that onto the open chat,
    // whose live copy the server didn't touch.
    if (S.chat && S.chat.memory_core_id === core.id) {
      S.chat.memory_core_id = "";
      S.chat.memory_enabled = false;
      S.chat.memory_turns_since = 0;
      $("chk-memory").checked = false;
      persistChat();
    }
    renderMemoryTab();
    updateMemoryVisibility();
    toast("Memory core deleted");
  } catch (e) { toast("Could not delete: " + e.message); }
}

function openMemoryEntry(entryId) {
  const core = activeCore();
  if (!core) return;
  const entry = (core.entries || []).find((e) => e.id === entryId) || null;
  S.editingMemoryEntry = entry ? entry.id : null;
  $("mce-title").textContent = entry ? "Edit memory" : "Add memory";
  $("mce-text").value = entry ? entry.text || "" : "";
  $("mce-importance").value = entry ? entry.importance || 5 : 5;
  $("mce-pinned").checked = entry ? !!entry.pinned : false;

  const sel = $("mce-category");
  sel.innerHTML = "";
  Object.entries(memoryCategories()).forEach(([id, label]) => {
    const o = document.createElement("option");
    o.value = id; o.textContent = label;
    sel.appendChild(o);
  });
  sel.value = entry ? entry.category || "other" : "preferences";
  openModal("modal-mc-entry");
  $("mce-text").focus();
}

async function saveMemoryEntry() {
  const core = activeCore();
  if (!core) return;
  const text = $("mce-text").value.trim();
  if (!text) { toast("Write something for the assistant to remember."); return; }
  try {
    const r = await api(`/api/memory/cores/${core.id}/entries`, { method: "POST", body: {
      entry: {
        id: S.editingMemoryEntry || "",
        text,
        category: $("mce-category").value,
        importance: parseInt($("mce-importance").value) || 5,
        pinned: $("mce-pinned").checked,
      },
    }});
    closeModal();
    S.editingMemoryEntry = null;
    applyMemoryCore(r.core);
  } catch (e) {
    // The entry or the core may have gone while the modal was open (another tab, a
    // background pass). Re-read rather than leaving the tab showing what isn't there.
    closeModal();
    S.editingMemoryEntry = null;
    toast("Could not save: " + e.message);
    refreshMemoryCores();
  }
}

async function deleteMemoryEntry(entryId) {
  const core = activeCore();
  if (!core) return;
  try {
    const r = await api(`/api/memory/cores/${core.id}/entries/${entryId}`, { method: "DELETE" });
    applyMemoryCore(r.core);
  } catch (e) {
    toast("Could not delete: " + e.message);
    refreshMemoryCores();
  }
}

async function refineMemoryCore() {
  const core = activeCore();
  if (!core) return;
  const btn = $("btn-mc-refine");
  btn.disabled = true; btn.textContent = "✨ Refining…";
  try {
    const r = await api(`/api/memory/cores/${core.id}/consolidate`, { method: "POST", body: {
      model: getSelectedModel(), server_url: currentServerUrl(),
    }});
    if (r.core) applyMemoryCore(r.core);
    if (r.error) toast(r.error);
    else if (r.skipped) toast("Skipped — an incognito session never writes to a memory core.");
    else toast(r.text ? "🧠 Refined: " + r.text : "🧠 Nothing to compact — the core is already tidy.");
    renderMemoryTab();
  } catch (e) { toast("Refine failed: " + e.message); }
  btn.disabled = false; btn.textContent = "✨ Refine core";
}

// ---- retroactive build over existing chats ----
function openMemoryBuild() {
  const core = activeCore();
  if (!core) return;
  const eligible = S.chats.filter((c) => !c.private);
  const list = $("mcb-list");
  if (!eligible.length) {
    list.innerHTML = `<p class="muted">You have no saved non-private chats to learn from yet.</p>`;
  } else {
    list.innerHTML = eligible.map((c) => `
      <label class="chk mcb-row">
        <input type="checkbox" class="mcb-chat" value="${c.id}" checked />
        ${escapeHtml(c.title || "Untitled")}
      </label>`).join("");
  }
  openModal("modal-mc-build");
}

async function runMemoryBuild() {
  const core = activeCore();
  if (!core) return;
  const ids = Array.from(document.querySelectorAll(".mcb-chat:checked")).map((i) => i.value);
  if (!ids.length) { toast("Pick at least one chat."); return; }
  const reread = $("mcb-reread").checked;
  closeModal();
  // Unique per run: a run id fixed to the core meant two builds shared one, so
  // cancelling either stopped both.
  const runId = `membuild-${core.id}-${uid()}`;
  const ui = makeProgressUI($("mc-build-progress"), {
    onCancel: () => {
      ui.line("Stopping…");
      api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
    },
  });
  ui.line(`Learning from ${ids.length} chat(s)…`);
  await streamSSE(`/api/memory/cores/${core.id}/build`,
    { chat_ids: ids, run_id: runId, reread }, {
    plan: (d) => ui.plan(d.phases || []),
    status: (d) => ui.line(d.message),
    progress: (d) => ui.update(d),
    built: (d) => {
      ui.finish();
      ui.line(d.text ? `Done — ${d.text}.` : "Done — nothing new to remember.");
      if (d.core) applyMemoryCore(d.core);
      toast(d.text ? "🧠 Memory built: " + d.text : "🧠 Nothing new to remember.");
      renderMemoryTab();
    },
    error: (d) => { ui.stop(); ui.line(`Error: ${d.message}`); toast("Build error: " + d.message); },
    done: () => ui.stop(),
  });
}

// ---- scoring the extractor ----
// Everything else in this tab is about whether memory *works*. This is about whether it
// remembers the right things: real extraction over sample transcripts, graded by a second
// model. Nothing here touches a memory core.
let MCS = { rows: [], criteria: [] };

async function openMemoryScore() {
  try {
    const r = await api("/api/memory/eval/seed");
    MCS = { rows: r.rows || [], criteria: r.criteria || [] };
  } catch (e) { toast("Could not load the sample transcripts: " + e.message); return; }

  const model = getSelectedModel();
  [$("mcs-gen-model"), $("mcs-grader-model")].forEach((sel) => {
    sel.innerHTML = "";
    (S.models.length ? S.models : [model].filter(Boolean)).forEach((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    });
    if (model && [...sel.options].some((o) => o.value === model)) sel.value = model;
  });

  $("mcs-rows").innerHTML = MCS.rows.map((row, i) => `
    <div class="mcs-row">
      <div class="mcs-row-head">${i + 1}. ${escapeHtml(row.Note || "")}</div>
      <pre class="mcs-transcript">${escapeHtml(row.Transcript || "")}</pre>
      ${row.ExistingMemories
        ? `<div class="mcs-existing">Already known: ${escapeHtml(row.ExistingMemories.split("\n").join(" · "))}</div>`
        : ""}
    </div>`).join("");
  $("mcs-results").innerHTML = "";
  $("mcs-progress").classList.add("hidden");
  openModal("modal-mc-score");
}

async function runMemoryScore() {
  if (!MCS.rows.length) { toast("Nothing to score."); return; }
  const genModel = $("mcs-gen-model").value;
  const graderModel = $("mcs-grader-model").value;
  if (!genModel || !graderModel) { toast("Pick both models first."); return; }

  const btn = $("btn-mcs-go");
  btn.disabled = true; btn.textContent = "Running…";
  $("mcs-results").innerHTML = "";
  const runId = `memscore-${uid()}`;
  const ui = makeProgressUI($("mcs-progress"), {
    onCancel: () => {
      ui.line("Stopping…");
      api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
    },
  });

  const project = {
    rows: MCS.rows, criteria: MCS.criteria,
    gen_model: genModel, gen_server_url: currentServerUrl(),
    grader_model: graderModel, grader_server_url: currentServerUrl(),
  };
  await streamSSE("/api/memory/eval", { eval: project, run_id: runId }, {
    plan: (d) => ui.plan(d.phases || []),
    status: (d) => ui.line(d.message),
    progress: (d) => ui.update(d),
    row_result: (d) => renderMemoryScoreRow(d),
    summary: (d) => { ui.finish(); renderMemoryScoreSummary(d); },
    error: (d) => { ui.stop(); ui.line(`Error: ${d.message}`); toast("Score error: " + d.message); },
    done: () => ui.stop(),
  });
  btn.disabled = false; btn.textContent = "Run";
}

function renderMemoryScoreRow(d) {
  const row = MCS.rows[d.row] || {};
  const el = document.createElement("div");
  el.className = "mcs-result";
  if (d.ungraded) {
    el.innerHTML = `<div class="mcs-row-head">${d.row + 1}. ungraded${
      d.note ? " — " + escapeHtml(d.note) : ""}</div>`;
    $("mcs-results").appendChild(el);
    return;
  }
  const scores = Object.entries(d.grades || {})
    .filter(([, g]) => g.score !== null && g.score !== undefined)
    .map(([label, g]) =>
      `<span class="mcs-score" title="${escapeHtml(g.reasoning || "")}">${escapeHtml(label)} ${g.score}/${g.max}</span>`)
    .join("");
  const ops = (d.operations || []).map((o) =>
    `<li>${escapeHtml(o.op || "?")}: ${escapeHtml(o.text || o.reason || "")}</li>`).join("");
  el.innerHTML =
    `<div class="mcs-row-head">${d.row + 1}. ${escapeHtml(row.Note || "")}</div>
     <div class="mcs-scores">${scores}</div>
     ${ops ? `<ul class="mcs-ops">${ops}</ul>`
           : `<p class="muted mcs-ops">No operations — nothing recorded.</p>`}`;
  $("mcs-results").appendChild(el);
}

function renderMemoryScoreSummary(d) {
  const agg = d.aggregate || {};
  const per = Object.entries(agg.criteria || {})
    .map(([label, c]) => `${escapeHtml(label)} ${Math.round(c.avg_pct || 0)}%`)
    .join(" · ");
  const el = document.createElement("div");
  el.className = "mcs-summary";
  el.innerHTML = `<b>Overall ${Math.round(agg.overall || 0)}%</b> over ${
    agg.graded || 0}/${agg.total || 0} graded rows${per ? ` — ${per}` : ""}`;
  $("mcs-results").appendChild(el);
}

// ---- export / import (native dialogs, like chats) ----
async function exportMemoryCore() {
  const core = activeCore();
  if (!core) { toast("Nothing to export yet."); return; }
  try {
    const r = await api("/api/memory/cores/export", { method: "POST", body: { ids: [core.id] } });
    if (r.cancelled) return;
    toast(r.ok ? `Exported to ${r.path}` : "Export failed: " + (r.error || "unknown"));
  } catch (e) { toast("Export failed: " + e.message); }
}

async function importMemoryCore() {
  try {
    const r = await api("/api/memory/cores/import", { method: "POST", body: {} });
    if (r.cancelled) return;
    if (!r.ok) { toast("Import failed: " + (r.error || "unknown")); return; }
    S.memoryCores = r.cores;
    if (r.imported && r.imported.length) S.activeMemoryCore = r.imported[0];
    S.lastMemoryCoreId = S.lastMemoryCoreId || (S.memoryCores[0] || {}).id || "";
    renderMemoryTab();
    updateMemoryVisibility();
    toast(`Imported ${r.count} memory core(s)`);
  } catch (e) { toast("Import failed: " + e.message); }
}

function bindMemoryEvents() {
  $("btn-mc-new").onclick = newMemoryCore;
  $("btn-mc-new-empty").onclick = newMemoryCore;
  $("btn-mc-delete").onclick = deleteMemoryCore;
  $("btn-mc-import").onclick = importMemoryCore;
  $("btn-mc-export").onclick = exportMemoryCore;
  $("mc-name").onchange = saveMemoryCoreSettings;
  $("mc-auto").onchange = saveMemoryCoreSettings;
  $("mc-every").onchange = saveMemoryCoreSettings;
  $("mc-limit").onchange = saveMemoryCoreSettings;
  $("btn-mc-add").onclick = () => openMemoryEntry(null);
  $("btn-mc-refine").onclick = refineMemoryCore;
  $("btn-mc-build").onclick = openMemoryBuild;
  $("btn-mc-score").onclick = openMemoryScore;
  $("btn-mcs-go").onclick = runMemoryScore;
  $("btn-mce-save").onclick = saveMemoryEntry;
  $("btn-mcb-go").onclick = runMemoryBuild;
  $("btn-mcb-all").onclick = () =>
    document.querySelectorAll(".mcb-chat").forEach((i) => (i.checked = true));
  $("btn-mcb-none").onclick = () =>
    document.querySelectorAll(".mcb-chat").forEach((i) => (i.checked = false));
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
  $("set-image-max-dim").value = S.config.image_max_dim || 1568;
  $("set-image-full-res").checked = !!S.config.image_full_res_default;
  $("set-rag-embed-url").value = S.config.rag_embed_server_url || "http://127.0.0.1:11434";
  $("set-rag-embed-model").value = S.config.rag_embed_model || "nomic-embed-text";
  $("set-rag-backend").value = S.config.rag_backend || "lance";
  refreshRagBackend();
  $("set-rag-parallel").checked = !!S.config.rag_embed_parallel;
  $("set-rag-batch").value = S.config.rag_embed_batch_size || 64;
  $("set-rag-concurrency").value = S.config.rag_embed_concurrency || 3;
  $("set-rag-ann").checked = S.config.rag_ann_enabled !== false;
  S.ragServers = (S.config.rag_embed_servers || []).map((s) => ({ ...s }));
  renderRagServers();
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
    rag_backend: $("set-rag-backend").value || "lance",
    rag_embed_parallel: $("set-rag-parallel").checked,
    rag_embed_servers: S.ragServers || [],
    rag_embed_batch_size: Math.max(1, parseInt($("set-rag-batch").value) || 64),
    rag_embed_concurrency: Math.max(1, parseInt($("set-rag-concurrency").value) || 3),
    rag_ann_enabled: $("set-rag-ann").checked,
    rag_top_k: Math.max(1, parseInt($("set-rag-topk").value) || 6),
    rag_retrieval_mode: $("set-rag-mode").value || "hybrid",
    rag_contextual_chunking: $("set-rag-context").checked,
    rag_context_model: $("set-rag-context-model").value.trim(),
    rag_query_rewrite: $("set-rag-qrewrite").checked,
    rewrite_model: $("set-rewrite-model").value.trim(),
    memory_weight_influence: parseFloat($("set-mem-influence").value) || 0,
    pipeline_max_retries: parseInt($("set-pipeline-retries").value) || 3,
    image_max_dim: parseInt($("set-image-max-dim").value) || 1568,
    image_full_res_default: $("set-image-full-res").checked,
  };
  const r = await api("/api/settings", { method: "POST", body });
  S.config = { ...S.config, ...r.config };
  updateImageResButton();   // its tooltip quotes the cap that just changed
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
// `stillCurrent` is re-checked AFTER the request: the status call is async and the badge
// slot is shared, so a slow reply for a library the user has left would otherwise land in
// the badge and describe the wrong one.
async function refreshCompileStatus(kind, id, badgeEl, stillCurrent) {
  if (!badgeEl || !id) return null;
  try {
    const base = kind === "library" ? `/api/libraries/${id}` : `/api/personas/${id}`;
    const st = await api(`${base}/compile-status`);
    if (stillCurrent && !stillCurrent()) return st;
    badgeEl.innerHTML = compileBadge(st);
    return st;
  } catch (e) {
    if (!stillCurrent || stillCurrent()) badgeEl.innerHTML = "";
    return null;
  }
}
// Duration as "d h m s", dropping leading units that are zero:
//   45 -> "45s"   200 -> "3m 20s"   3902 -> "1h 5m 2s"   184446 -> "2d 3h 14m 6s"
function fmtDur(s) {
  if (s == null || !isFinite(s) || s < 0) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400),
        h = Math.floor((s % 86400) / 3600),
        m = Math.floor((s % 3600) / 60),
        sec = s % 60;
  const out = [];
  if (d) out.push(d + "d");
  if (d || h) out.push(h + "h");
  if (d || h || m) out.push(m + "m");
  out.push(sec + "s");
  return out.join(" ");
}

// A progress widget with a live ETA. Phases (parse/chunk/context/embed) are measured
// in different units, so each carries a weight and the overall fraction is the
// weighted sum — that way the bar stays honest when embedding dominates.
//
// The ETA is computed HERE rather than server-side, from an exponentially-weighted
// moving average of fraction-per-second, and re-rendered on a 1s timer. EWMA rather
// than a plain elapsed/fraction average so the estimate adapts when a second embedding
// server joins or a large document stalls; the timer so the countdown ticks down
// between SSE frames instead of lurching whenever one happens to arrive.
function makeProgressUI(progressEl, opts) {
  opts = opts || {};
  progressEl.classList.remove("hidden");
  progressEl.innerHTML =
    `<div class="compile-bar"><div class="compile-bar-fill"></div></div>
     <div class="compile-meta">
       <span class="compile-phase"></span>
       <span class="compile-counts"></span>
       <span class="compile-eta"></span>
       ${opts.onCancel ? '<button class="small compile-cancel">Stop</button>' : ""}
     </div>
     <div class="compile-lanes"></div>
     <div class="compile-log"></div>`;
  const fill = progressEl.querySelector(".compile-bar-fill");
  const log = progressEl.querySelector(".compile-log");
  const phaseEl = progressEl.querySelector(".compile-phase");
  const countsEl = progressEl.querySelector(".compile-counts");
  const etaEl = progressEl.querySelector(".compile-eta");
  const lanesEl = progressEl.querySelector(".compile-lanes");
  const cancelBtn = progressEl.querySelector(".compile-cancel");
  if (cancelBtn) cancelBtn.onclick = () => { cancelBtn.disabled = true; opts.onCancel(); };

  const started = Date.now();
  let weights = {}, fracs = {}, order = [];
  let rate = null, lastFrac = 0, lastAt = started, finished = false;

  const overall = () =>
    order.reduce((a, id) => a + (weights[id] || 0) * (fracs[id] || 0), 0);

  function render() {
    const f = finished ? 1 : Math.min(0.999, overall());
    fill.style.width = (f * 100).toFixed(1) + "%";
    const elapsed = (Date.now() - started) / 1000;
    if (finished) {
      etaEl.textContent = `done in ${fmtDur(elapsed)}`;
      return;
    }
    // Hold off on a number until there's enough signal for it to mean anything.
    if (!rate || f < 0.03 || elapsed < 5) {
      etaEl.textContent = `estimating… · ${fmtDur(elapsed)} elapsed`;
      return;
    }
    const remaining = Math.max(0, 1 - f) / rate;
    etaEl.textContent = `ETA ${fmtDur(remaining)} · ${fmtDur(elapsed)} elapsed`;
  }

  const timer = setInterval(render, 1000);

  return {
    line(msg) {
      const d = document.createElement("div");
      d.textContent = msg;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
    },
    plan(phases) {
      order = phases.map((p) => p.id);
      weights = {};
      phases.forEach((p) => { weights[p.id] = p.weight; fracs[p.id] = 0; });
      render();
    },
    update(d) {
      const id = d.phase;
      if (!(id in weights)) {   // a phase the plan didn't mention (e.g. bare parse)
        weights[id] = 1; order = [id];
      }
      const total = d.total || 0;
      fracs[id] = total ? Math.min(1, (d.done || 0) / total) : 0;
      // Everything before this phase in the plan must be complete.
      const at = order.indexOf(id);
      order.forEach((pid, i) => { if (i < at) fracs[pid] = 1; });

      const now = Date.now();
      const f = overall();
      const dt = (now - lastAt) / 1000;
      if (dt >= 0.5 && f > lastFrac) {
        const inst = (f - lastFrac) / dt;
        rate = rate == null ? inst : rate * 0.8 + inst * 0.2;   // EWMA, alpha 0.2
        lastFrac = f; lastAt = now;
      }
      const label = d.label || id;
      phaseEl.textContent = label.charAt(0).toUpperCase() + label.slice(1);
      const parts = [];
      if (total) parts.push(`${(d.done || 0).toLocaleString()} / ${total.toLocaleString()} ${d.unit || ""}`.trim());
      if (d.cached) parts.push(`${d.cached.toLocaleString()} reused`);
      if (d.failed) parts.push(`${d.failed.toLocaleString()} failed`);
      countsEl.textContent = parts.join(" · ");
      render();
    },
    lanes(list) {
      if (!list || list.length < 2) { lanesEl.textContent = ""; return; }
      lanesEl.innerHTML = list.map((l) =>
        `<span class="compile-lane${l.down ? " down" : ""}">${escapeHtml(l.name)}: ` +
        `${(l.chunks || 0).toLocaleString()} chunks${l.rate ? ` (${l.rate}/s)` : ""}` +
        `${l.down ? " — offline" : ""}</span>`).join("");
    },
    finish() { finished = true; render(); clearInterval(timer); },
    stop() { clearInterval(timer); },
  };
}

// `opts.stillCurrent()` guards every write into `badgeEl`: a compile outlives the view
// that started it, and a shared badge slot (the Resources tab has exactly one) would
// otherwise end up describing a library the user has since navigated away from.
async function runCompile(kind, id, opts, progressEl, badgeEl) {
  if (!id) { toast("Nothing to compile yet."); return; }
  opts = opts || {};
  const current = opts.stillCurrent || (() => true);
  const base = kind === "library" ? `/api/libraries/${id}` : `/api/personas/${id}`;
  // Unique per invocation so two compiles don't share (and cross-cancel) a stop event.
  const runId = `compile-${kind}-${id}-${uid()}`;
  const ui = makeProgressUI(progressEl, {
    onCancel: () => {
      ui.line("Stopping…");
      api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
    },
  });
  if (badgeEl && current()) badgeEl.innerHTML = `<span class="cbadge working">Compiling…</span>`;
  let total = 0;
  await streamSSE(`${base}/compile`, { force: !!opts.force, run_id: runId }, {
    begin: (d) => { total = d.total || 0; ui.line(`Compiling ${d.name || ""} — ${total} item(s)…`); },
    plan: (d) => ui.plan(d.phases || []),
    progress: (d) => ui.update(d),
    item_start: (d) => { if (total <= 20) ui.line(`• ${d.name}…`); },
    item_done: (d) => ui.line(
      `   ${d.skipped ? "skipped (unchanged)" : "embedded " + (d.chunks || 0) + " chunk(s)"}: ${d.name}`),
    warn: (d) => ui.line(`   ⚠ ${d.name}: ${d.message}`),
    compiled: (d) => {
      ui.finish();
      ui.lanes(d.lanes || []);
      const bits = [`${d.embedded} embedded`, `${d.skipped} skipped`];
      if (d.cached) bits.push(`${d.cached.toLocaleString()} vectors reused`);
      if (d.failed) bits.push(`${d.failed} failed`);
      ui.line(`${d.stopped ? "Stopped" : "Done"} — ${bits.join(", ")}; ${d.chunks} chunks total.`);
      if (d.stopped) toast("Compile stopped — partial progress kept; recompile to finish.");
      else if (d.failed) toast(`Compiled with ${d.failed} failed chunk(s) — check your embedding server, then recompile.`);
      else toast(`Compiled: ${d.chunks} chunks (${d.embedded} embedded, ${d.skipped} skipped).`);
    },
    error: (d) => { ui.stop(); ui.line(`Error: ${d.message}`); toast("Compile error: " + d.message); },
    done: () => ui.stop(),
  });
  if (badgeEl && current()) await refreshCompileStatus(kind, id, badgeEl, current);
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
// The source vocabulary shared by library items and chat attachments. Each has a
// matching `.tag.<kind>` colour in styles.css.
// "image" is chat-only: the Resources tab can't create one, because a library item
// is text to be chunked and retrieved, which an image is not.
const SOURCE_KINDS = ["write", "file", "url", "youtube", "search", "image"];
// Kinds whose `filename` (library) / `source` (attachment) holds a clickable URL.
const LINKED_KINDS = ["url", "youtube", "search"];

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
async function selectLibrary(id) {
  await flushLibrarySave();   // don't let a pending edit follow us to the next library
  S.activeLibrary = S.libraries.find((l) => l.id === id) || null;
  renderLibraryEditor();
}
// Library ids with a compile/fetch streaming right now. renderLibraryEditor must not
// tear their progress panel down, and runCompile must not write a finished badge into
// an editor that has since moved to a different library.
const libRunInFlight = new Set();
function renderLibraryEditor() {
  const lib = S.activeLibrary;
  $("lib-name").value = lib ? lib.name : "";
  const box = $("lib-items");
  box.innerHTML = "";
  const running = lib && libRunInFlight.has(lib.id);
  if (!running) $("lib-compile-progress").classList.add("hidden");
  const badge = $("lib-compile-badge");
  if (badge && !running) badge.innerHTML = "";
  if (!lib) return;
  (lib.items || []).forEach((it, idx) => box.appendChild(makeLibItem(it, idx)));
  refreshLibBadge(lib.id);
}
function makeLibItem(it, idx) {
  const wrap = document.createElement("div");
  wrap.className = "lib-item";
  const head = document.createElement("div");
  head.className = "item-head";
  const tag = document.createElement("span");
  const kind = SOURCE_KINDS.includes(it.type) ? it.type : "write";
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
  // Show the scraped source link (URL/YouTube items store the source in `filename`).
  if (LINKED_KINDS.includes(it.type) && it.filename) {
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
let libSavePending = null;   // the in-flight/queued save, so callers can await it
// The library being saved is captured HERE, at schedule time — not read from
// S.activeLibrary when the timer fires 400ms later. Otherwise switching libraries just
// after a keystroke made the pending save write the *newly selected* library, and the
// edit to the previous one was never persisted (it survived in memory, so nothing looked
// wrong until the next reload).
async function saveLibrary(immediate) {
  const lib = S.activeLibrary;
  if (!lib) return;
  lib.name = $("lib-name").value;
  clearTimeout(libSaveTimer);
  const doIt = async () => {
    try {
      const r = await api(`/api/libraries/${lib.id}`, { method: "PUT", body: { library: lib }});
      // Backfill the server-assigned item ids into the LIVE objects (without re-rendering,
      // so typing isn't disrupted). Otherwise every save would re-send id-less items and
      // the store would mint fresh ids each time, needlessly staling the compiled index.
      (r.library.items || []).forEach((it, i) => {
        if (lib.items[i] && !lib.items[i].id) lib.items[i].id = it.id;
      });
      S.libraries = S.libraries.map((l) => (l.id === r.library.id ? lib : l));
      if (S.activeLibrary === lib) {
        renderLibraryList();
        // Content changed → the compiled index may be stale now; reflect it in the badge.
        refreshLibBadge(lib.id);
      }
      updateLibraryButton();
    } catch (e) {}
  };
  const run = () => { libSavePending = doIt().finally(() => { libSavePending = null; }); return libSavePending; };
  if (immediate) return run();
  libSaveTimer = setTimeout(run, 400);
}
/** Write out any debounced edit right now and wait for it to land. Call before anything
 *  that swaps the editor's library or hands the library to a long-running server job. */
async function flushLibrarySave() {
  if (libSaveTimer) { clearTimeout(libSaveTimer); libSaveTimer = null; await saveLibrary(true); }
  if (libSavePending) await libSavePending;
}
/** Refresh the Resources-tab compile badge, but only while `libId` is still the library
 *  on screen — a slow status call for a library the user has navigated away from used to
 *  land in the badge and describe the wrong one. */
function refreshLibBadge(libId) {
  const isCurrent = () => !!S.activeLibrary && S.activeLibrary.id === libId;
  if (isCurrent()) refreshCompileStatus("library", libId, $("lib-compile-badge"), isCurrent);
}
/** Append server-created items to the open editor WITHOUT re-rendering it, so textareas
 *  the user is mid-edit in (and their unsaved keystrokes) survive a fetch completing. */
function adoptAddedItems(libId, added) {
  if (!S.activeLibrary || S.activeLibrary.id !== libId) return;  // user moved on
  const box = $("lib-items");
  S.activeLibrary.items = S.activeLibrary.items || [];
  (added || []).forEach((it) => {
    box.appendChild(makeLibItem(it, S.activeLibrary.items.length));
    S.activeLibrary.items.push(it);
  });
  S.libraries = S.libraries.map((l) => (l.id === libId ? S.activeLibrary : l));
}
async function newLibrary() {
  await flushLibrarySave();   // S.libraries is about to be replaced wholesale
  const r = await api("/api/libraries", { method: "POST", body: { name: "New Library" }});
  S.libraries = r.libraries; renderLibraryList();
  selectLibrary(r.library.id); $("lib-list").value = r.library.id;
}
async function removeLibrary() {
  if (!S.activeLibrary) return;
  if (!confirm(`Remove library "${S.activeLibrary.name}"?`)) return;
  const id = S.activeLibrary.id;
  clearTimeout(libSaveTimer); libSaveTimer = null;   // don't resurrect what we're deleting
  const r = await api(`/api/libraries/${id}`, { method: "DELETE" });
  S.libraries = r.libraries; S.activeLibrary = null;
  // The server strips the id from every chat's library_ids; mirror that locally so the
  // open chat doesn't keep RAG pinned on with a reference to a library that is gone.
  if (S.chat && (S.chat.library_ids || []).includes(id)) {
    S.chat.library_ids = S.chat.library_ids.filter((x) => x !== id);
  }
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
  await flushLibrarySave();
  setStatus("Waiting for file selection…");
  const progressEl = $("lib-compile-progress");
  const libId = S.activeLibrary.id;
  // The server mints the run id (unique per invocation, so two runs on one library
  // can't orphan each other's stop event) and announces it in the `start` frame.
  // Cancel has to use that one or it stops nothing.
  let runId = "";
  libRunInFlight.add(libId);
  // Parsing a stack of ebooks is minutes of CPU, so this streams per-file progress
  // instead of blocking on one opaque request.
  let ui = null;
  try {
    await streamSSE(`/api/libraries/${libId}/add-text-files`, {}, {
      start: (d) => { runId = d.run_id || runId; },
      begin: (d) => {
        setStatus("");
        if (!d.total) return;
        ui = makeProgressUI(progressEl, {
          onCancel: () => runId && api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {}),
        });
        ui.plan([{ id: "parse", label: "Reading documents", weight: 1 }]);
        ui.line(`Reading ${d.total} document(s)…`);
      },
      progress: (d) => { if (ui) ui.update({ ...d, label: "Reading documents" }); },
      complete: (r) => {
        adoptAddedItems(libId, r.added_items);
        refreshLibBadge(libId);
        if (ui) { ui.finish(); ui.line(`Added ${(r.added || []).length} file(s).`); }
        if ((r.added || []).length) toast(`Added ${r.added.length} file(s)`);
        if ((r.errors || []).length) toast("Some files failed: " + r.errors.join("; "));
      },
      error: (d) => { if (ui) ui.stop(); toast("Add files failed: " + d.message); },
      done: () => { if (ui) ui.stop(); },
    });
  } catch (e) { toast("Add files failed: " + e.message); }
  libRunInFlight.delete(libId);
  setStatus("");
}
function showLibPanel(which) {
  $("lib-url-panel").classList.toggle("hidden", which !== "url");
  $("lib-yt-panel").classList.toggle("hidden", which !== "youtube");
  $("lib-search-panel").classList.toggle("hidden", which !== "search");
  if (which === "url") { $("lib-url-input").value = ""; $("lib-url-input").focus(); }
  if (which === "youtube") {
    $("lib-yt-input").value = "";
    const prog = $("lib-yt-progress"); prog.classList.add("hidden"); prog.textContent = "";
    $("lib-yt-input").focus();
  }
  if (which === "search") {
    $("lib-search-query").value = ""; $("lib-search-sites").value = "";
    const prog = $("lib-search-progress"); prog.classList.add("hidden"); prog.textContent = "";
    $("lib-search-query").focus();
  }
}
function hideLibPanels() {
  $("lib-url-panel").classList.add("hidden");
  $("lib-yt-panel").classList.add("hidden");
  $("lib-search-panel").classList.add("hidden");
}
async function addByUrl() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const url = $("lib-url-input").value.trim();
  if (!url) { toast("Enter a URL"); return; }
  await flushLibrarySave();
  const libId = S.activeLibrary.id;
  const btn = $("btn-lib-url-fetch");
  btn.disabled = true; setStatus("Fetching " + url + " …");
  try {
    const r = await api(`/api/libraries/${libId}/add-url`, { method: "POST", body: { url }});
    adoptAddedItems(libId, r.added_items);
    updateLibraryButton();
    refreshLibBadge(libId);
    if (r.added && r.added.length) {
      toast(`Added: ${r.added.join(", ")}${r.via ? " (via " + r.via + ")" : ""}`);
      hideLibPanels();
    }
    if (r.errors && r.errors.length) toast("Fetch failed: " + r.errors.join("; "));
  } catch (e) { toast("Add URL failed: " + e.message); }
  btn.disabled = false; setStatus("");
}
// ---- YouTube ----
// Both the Resources tab and the composer stream the same fetch, so the wire
// handling lives here once and the callers only decide what to do with the result.
function ytProgressText(d) {
  if (d.phase === "page") return `Fetching the video page${d.via ? " via " + d.via : ""}…`;
  if (d.phase === "transcript") {
    return d.chars ? `Transcript: ${d.chars.toLocaleString()} characters` : "No transcript found";
  }
  if (d.phase === "comments") return `Comments: ${d.done}/${d.target}…`;
  return "Working…";
}
/** Open the YouTube SSE stream. `path` is the route, `opts` the query params.
 *  Returns the EventSource so the caller can close it. */
function youtubeStream(path, opts, handlers) {
  const qs = new URLSearchParams({
    url: opts.url,
    comments: opts.comments ? "1" : "0",
    max: String(opts.max || 100),
  }).toString();
  const es = new EventSource(`${path}?${qs}`);
  let finished = false;
  let gotFrame = false;   // did the server ever speak SSE to us?
  const finish = () => { if (!finished) { finished = true; es.close(); handlers.finally?.(); } };
  // The route mints a unique run id and announces it here. Cancel needs THIS id: it is
  // not derivable from the library, and after a library switch a composed one would
  // name a different run (or none).
  es.addEventListener("start", (ev) => {
    gotFrame = true;
    let d = null; try { d = JSON.parse(ev.data); } catch (e) {}
    if (d && d.run_id) handlers.started?.(d.run_id);
  });
  es.addEventListener("progress", (ev) => { gotFrame = true; handlers.progress?.(JSON.parse(ev.data)); });
  es.addEventListener("complete", (ev) => { gotFrame = true; handlers.complete?.(JSON.parse(ev.data)); finish(); });
  es.addEventListener("error", (ev) => {
    // SSE 'error' fires both on our emitted error frame and on a normal stream close.
    if (ev.data) {
      let msg = "unknown error";
      try { msg = JSON.parse(ev.data).message || msg; } catch (e) {}
      handlers.failed?.(msg);
    } else if (!gotFrame) {
      // The route answered with a JSON error (404 / 400) or the session expired (401).
      // EventSource cannot read a response body, so without this the button was simply
      // dead — no toast, no progress, nothing.
      handlers.failed?.("the server rejected the request (it may need a reload or login)");
    }
    finish();
  });
  return es;
}
let libYtES = null;
let libYtRunId = null;
async function addYouTube() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const url = $("lib-yt-input").value.trim();
  if (!url) { toast("Enter a YouTube URL"); return; }
  await flushLibrarySave();
  const libId = S.activeLibrary.id;
  const comments = $("lib-yt-comments").checked;
  const max = Math.max(5, Math.min(2000, parseInt($("lib-yt-max").value, 10) || 100));
  const prog = $("lib-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Starting…";
  const btn = $("btn-lib-yt-fetch"); btn.disabled = true;
  cancelLibYouTube();
  libYtES = youtubeStream(`/api/libraries/${libId}/add-youtube`,
                          { url, comments, max }, {
    started: (id) => { libYtRunId = id; },
    progress: (d) => { prog.textContent = ytProgressText(d); },
    complete: (d) => {
      adoptAddedItems(libId, d.added_items);
      updateLibraryButton();
      refreshLibBadge(libId);
      toast(`Added: ${d.added.join(", ")}${d.via ? " (via " + d.via + ")" : ""}` +
            (d.comment_count ? ` — ${d.comment_count} comment(s)` : ""));
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 6000);
      hideLibPanels();
    },
    failed: (msg) => { toast("YouTube fetch failed: " + msg); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; libYtES = null; libYtRunId = null; },
  });
}
/** Abort an in-flight YouTube fetch. Cancel used to only hide the panel, so the stream
 *  kept running and still appended the video to a library the user had walked away from. */
function cancelLibYouTube() {
  if (!libYtES) return;
  // The id the RUN announced, not one composed from whichever library is selected now:
  // after a library switch the composed id named a different run, so Cancel stopped
  // nothing and the video still landed. Same rule as cancelLibSearch below.
  const runId = libYtRunId;
  libYtES.close(); libYtES = null; libYtRunId = null;
  // Closing the socket alone leaves the worker fetching comment pages until the server
  // notices the disconnect; ask it to stop outright.
  if (runId) api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
  $("btn-lib-yt-fetch").disabled = false;
  $("lib-yt-progress").textContent = "Cancelled.";
}
let libSearchES = null;
let libSearchRunId = null;
async function braveSearch() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const q = $("lib-search-query").value.trim();
  if (!q) { toast("Enter a search term"); return; }
  await flushLibrarySave();
  const libId = S.activeLibrary.id;
  const sites = $("lib-search-sites").value.trim();
  // Clamped at both ends: the input's max="20" is not enforced against a typed value,
  // and a mistyped 500 would fire 500 page fetches. The server clamps to the same bound.
  const max = Math.max(1, Math.min(20, parseInt($("lib-search-max").value, 10) || 5));
  const prog = $("lib-search-progress");
  prog.classList.remove("hidden"); prog.textContent = "Searching Brave…";
  const goBtn = $("btn-lib-search-go"); goBtn.disabled = true;
  cancelLibSearch();
  const qs = new URLSearchParams({ q, sites, max: String(max) }).toString();
  const es = new EventSource(`/api/libraries/${libId}/brave-search?${qs}`);
  libSearchES = es; libSearchRunId = null;
  const finish = () => {
    es.close();
    if (libSearchES === es) { libSearchES = null; libSearchRunId = null; }
    goBtn.disabled = false;
  };
  // The route mints a unique id per invocation, so it can no longer be composed from
  // the library id — take the one the run announces.
  es.addEventListener("start", (ev) => {
    let d = null; try { d = JSON.parse(ev.data); } catch (e) {}
    if (d && d.run_id && libSearchES === es) libSearchRunId = d.run_id;
  });
  es.addEventListener("progress", (ev) => {
    const d = JSON.parse(ev.data);
    prog.textContent = `Crawled ${d.done}/${d.target} — ${d.ok ? "✓" : "✗"} ${d.title || d.url}`;
  });
  es.addEventListener("complete", (ev) => {
    if (libSearchES !== es) return;          // cancelled — ignore a late frame
    const d = JSON.parse(ev.data);
    adoptAddedItems(libId, d.added_items);
    updateLibraryButton();
    refreshLibBadge(libId);
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
/** Abort an in-flight Brave crawl, server side too. Cancel used to only hide the panel:
 *  the crawl ran to completion and still appended its pages. */
function cancelLibSearch() {
  if (!libSearchES) return;
  const runId = libSearchRunId;
  libSearchES.close(); libSearchES = null; libSearchRunId = null;
  if (runId) api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
  $("btn-lib-search-go").disabled = false;
  $("lib-search-progress").textContent = "Cancelled.";
}
async function loadLibraryXML() {
  await flushLibrarySave();   // S.libraries is about to be replaced wholesale
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

// --------------------- streaming-run bookkeeping ---------------------
// Every DB-tab stream registers with the server's run registry and honours its stop
// event, but nothing here ever called POST /api/stop — so an import, an AI pass over
// thousands of rows, or a row-by-row write-back could not be called off. These three
// helpers keep the run id and its Stop button in step, mirroring the eval tab.
function dbRunStart(kind, runId, stopBtnId) {
  S.db.runs[kind] = runId;
  const b = $(stopBtnId);
  if (b) b.disabled = false;
}

function dbRunEnd(kind, stopBtnId) {
  S.db.runs[kind] = null;
  const b = $(stopBtnId);
  if (b) b.disabled = true;
}

async function dbStopRun(kind) {
  const runId = S.db.runs[kind];
  if (!runId) return;
  try { await api("/api/stop", { method: "POST", body: { run_id: runId } }); }
  catch (e) { toast("Stop failed: " + e.message); }
}

/** Append a line to a status element without clobbering what is already there —
 *  `warn` frames must survive the next `progress` tick that overwrites the line. */
function dbNote(id, text) {
  const el = $(id);
  if (el) el.textContent = (el.textContent ? el.textContent + "\n" : "") + text;
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
  dbRunStart("import", body.run_id, "dbi-stop");
  streamSSE("/api/db/import", body, {
    status: (d) => { $("dbi-progress").textContent = d.message || ""; },
    progress: (d) => { $("dbi-progress").textContent = `Imported ${d.done} / ${d.total}…`; },
    warn: (d) => { dbNote("dbi-progress", "⚠ " + (d.message || "")); },
    done: async (d) => {
      $("dbi-progress").textContent = d.stopped
        ? `Stopped — ${d.row_count} rows staged so far.`
        : `Done — ${d.row_count} rows staged.`;
      $("db-import-form").classList.add("hidden");
      // Await the refresh: dbOpenSession reads the session list for the grid title,
      // and firing it against a stale list showed the fallback "Staging" instead.
      await dbLoadState();
      if (d.session_id) dbOpenSession(d.session_id);
    },
    error: (d) => { $("dbi-progress").textContent = "✗ " + (d.message || "error"); },
  }).catch((e) => { $("dbi-progress").textContent = "✗ " + e.message; })
    .finally(() => dbRunEnd("import", "dbi-stop"));
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
    g.types = page.column_types || {};
    const sess = S.db.sessions.find((s) => s.id === sessionId);
    $("db-grid-title").textContent = sess ? sess.name : "Staging";
    $("db-grid-dirty").textContent = page.dirty ? `${page.dirty} row(s) with pending edits` : "";
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
      inp.className = "db-cell";
      const isNull = row[col] == null;
      inp.value = isNull ? "" : row[col];
      // An empty text box means NULL for a NULL cell and "" for an empty string, and
      // those write back differently — mark the NULL ones so they read apart.
      inp.placeholder = isNull ? "∅" : "";
      inp.classList.toggle("db-cell-null", isNull);
      inp.onchange = () => dbEditCell(row.__rowid, col, inp);
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

/** Text-ish DuckDB types, where an empty box legitimately means the empty string.
 *  For everything else (BIGINT, DATE, DECIMAL…) an empty box can only mean NULL —
 *  sending "" made DuckDB fail the cast, so a numeric cell could never be cleared. */
function dbIsTextType(t) {
  return /VARCHAR|TEXT|STRING|CHAR|JSON|BLOB/i.test(t || "VARCHAR");
}

async function dbEditCell(rowid, column, inp) {
  const raw = inp.value;
  const value = (raw === "" && !dbIsTextType(S.db.grid.types[column])) ? null : raw;
  try {
    await api(`/api/db/session/${S.db.grid.sessionId}/cell`,
      { method: "PUT", body: { rowid, column, value } });
    inp.classList.remove("db-cell-bad");
    inp.classList.toggle("db-cell-null", value === null);
    inp.placeholder = value === null ? "∅" : "";
    inp.title = "";
    // The server just marked the row dirty; show it now rather than only after the
    // page is re-opened.
    const tr = inp.closest("tr");
    if (tr) tr.classList.add("db-row-dirty");
  } catch (e) {
    // Keep the failure ON the cell. A toast disappears, leaving the grid showing a
    // value the staging table never accepted.
    inp.classList.add("db-cell-bad");
    inp.title = e.message;
    toast(`${column}: ${e.message}`);
  }
}

async function dbAddColumn() {
  const name = await promptModal("New column name", "");
  if (!name || !name.trim()) return;
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
  $("dbp-progress").textContent = "Starting…";
  // Per-cell failures arrive as `error` frames and the run continues; collect them so
  // the next `progress` tick does not wipe the only notice the user gets.
  let cellErrors = 0;
  dbRunStart("process", body.run_id, "dbp-stop");
  streamSSE(`/api/db/session/${S.db.grid.sessionId}/process`, body, {
    status: (d) => { $("dbp-progress").textContent = d.message || ""; },
    progress: (d) => {
      $("dbp-progress").textContent = `Row ${d.done} / ${d.total}…`
        + (cellErrors ? `  (${cellErrors} cell error(s))` : "");
    },
    cell_done: () => {},
    warn: (d) => { dbNote("dbp-progress", "⚠ " + (d.message || "")); },
    error: (d) => {
      cellErrors++;
      // A frame carrying a rowid is one cell failing, not the run dying.
      if (d.rowid == null) $("dbp-progress").textContent = "✗ " + (d.message || "error");
      else toast(`Row ${d.rowid} / ${d.column}: ${d.message}`);
    },
    done: (d) => {
      $("dbp-progress").textContent = (d.stopped ? "Stopped." : "Done.")
        + (cellErrors ? ` ${cellErrors} cell(s) failed — see the grid.` : "");
      dbOpenSession(S.db.grid.sessionId, S.db.grid.offset);
    },
  }).catch((e) => { $("dbp-progress").textContent = "✗ " + e.message; })
    .finally(() => dbRunEnd("process", "dbp-stop"));
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
    if (r.note) out += "\nℹ " + r.note + "\n";
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
  const notes = [];
  dbRunStart("writeback", body.run_id, "dbw-stop");
  streamSSE(`/api/db/session/${S.db.grid.sessionId}/writeback`, body, {
    guard: (d) => { if (d.has_conflict) $("dbw-preview").textContent =
      `⚠ Conflict: changed ${d.changed.length}, deleted ${d.deleted.length}. on_conflict=${d.on_conflict}`; },
    progress: (d) => { $("dbw-progress").textContent = `${d.done} / ${d.total}…`; },
    // The server warns when a row matched nothing, or when the staging baseline could
    // not be refreshed after a successful write. These were being dropped on the floor:
    // streamSSE discards any frame with no handler.
    warn: (d) => { notes.push("⚠ " + (d.message || "")); },
    error: (d) => { $("dbw-progress").textContent = "✗ " + (d.message || "error"); },
    done: async (d) => {
      let msg = d.partial
        ? `Stopped after an error — ${d.applied_rows} row(s), ${d.applied_cells} cell(s) were written before it.`
        : (d.stopped ? "Cancelled — " : "Done — ")
          + `${d.applied_rows} row(s), ${d.applied_cells} cell(s) written`;
      if (d.unmatched_rows) msg += `; ${d.unmatched_rows} matched no source row`;
      if (d.skipped_columns && d.skipped_columns.length) msg += `; skipped ${d.skipped_columns.join(", ")}`;
      $("dbw-progress").textContent = msg.replace(/[.\s]*$/, ".");
      if (d.error) notes.push("✗ " + d.error);
      if (notes.length) $("dbw-preview").textContent = notes.join("\n");
      // Retired cells are no longer dirty — refresh the grid so it stops showing them
      // as pending.
      await dbLoadState();
      if (S.db.grid.sessionId) dbOpenSession(S.db.grid.sessionId, S.db.grid.offset);
    },
  }).catch((e) => { $("dbw-progress").textContent = "✗ " + e.message; })
    .finally(() => dbRunEnd("writeback", "dbw-stop"));
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
  $("dbi-stop").onclick = () => dbStopRun("import");
  $("db-grid-add-col").onclick = dbAddColumn;
  $("db-grid-process").onclick = dbShowProcess;
  $("dbp-add").onclick = dbAddProcCol;
  $("dbp-run").onclick = dbRunProcess;
  $("dbp-stop").onclick = () => dbStopRun("process");
  $("dbp-close").onclick = () => $("db-process-panel").classList.add("hidden");
  $("db-grid-writeback").onclick = () => $("db-writeback-panel").classList.toggle("hidden");
  $("dbw-dryrun").onclick = dbDryRun;
  $("dbw-conflicts").onclick = dbCheckConflicts;
  $("dbw-run").onclick = dbWriteBack;
  $("dbw-stop").onclick = () => dbStopRun("writeback");
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
  $("tab-batch").classList.toggle("active", name === "batch");
  $("tab-evals").classList.toggle("active", name === "evals");
  $("tab-database").classList.toggle("active", name === "database");
  $("tab-resources").classList.toggle("active", name === "resources");
  $("tab-personas").classList.toggle("active", name === "personas");
  $("tab-memory").classList.toggle("active", name === "memory");
  $("tab-settings").classList.toggle("active", name === "settings");
  if (name === "resources" && !S.activeLibrary && S.libraries.length) {
    selectLibrary(S.libraries[0].id); $("lib-list").value = S.libraries[0].id;
  }
  if (name === "settings") renderSettings();
  if (name === "batch") initBatchTab();
  if (name === "evals") initEvalTab();
  if (name === "database") initDatabaseTab();
  if (name === "personas") renderPersonaTab();
  // Render from what we have so the tab is never blank, then re-read: a background
  // extraction pass or a second browser tab may have moved the cores on.
  if (name === "memory") { renderMemoryTab(); refreshMemoryCores(); }
}

// ------------------------------- events ------------------------------
function bindEvents() {
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => switchTab(t.dataset.tab));
  document.querySelectorAll(".modal-close").forEach((b) => b.onclick = () => dismissModal());
  // Click-outside only counts when the press AND the release both land on the backdrop.
  // Otherwise a drag-select that overshoots the edge of a text field would close the
  // dialog: `click` retargets to the common ancestor of mousedown and mouseup.
  let backdropPress = false;
  $("modal-backdrop").addEventListener("pointerdown", (e) => {
    backdropPress = e.target === e.currentTarget;
  });
  $("modal-backdrop").addEventListener("click", (e) => {
    const pressed = backdropPress;
    backdropPress = false;
    if (pressed && e.target === e.currentTarget) dismissModal();
  });

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
  $("btn-rag-add-servers").onclick = addRagServersFromList;
  $("btn-rag-check-servers").onclick = checkRagServers;
  // Refresh the note as soon as the store is changed, before Save, so the encryption
  // trade-off is visible at the moment of choosing.
  $("set-rag-backend").onchange = () => {
    const v = $("set-rag-backend").value;
    $("rag-backend-note").innerHTML = v === "lance"
      ? "<strong>Not encrypted.</strong> Save to switch — chunk text and embeddings will be stored in plain files."
      : "Encrypted at rest with your login password. Save to switch.";
  };

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
  $("chk-memory").onchange = onMemoryToggle;
  $("memory-core-select").onchange = () => {
    if (!S.chat) return;
    S.chat.memory_core_id = $("memory-core-select").value;
    S.lastMemoryCoreId = S.chat.memory_core_id;
    persistChat();
  };
  $("btn-memory-extract").onclick = () => extractMemoriesNow();
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
  $("persona-select").onchange = (e) => {
    S.personaId = e.target.value;
    S.personaVariant = "";   // variants are per-persona
    renderVariantControl(); savePersonaSelection();
  };
  $("persona-variant").onchange = (e) => { S.personaVariant = e.target.value; savePersonaSelection(); };
  $("btn-data-mode").onclick = toggleDataMode;
  $("btn-add-data").onclick = addDataItem;

  // Composer ＋ Add menu (chat-scoped sources).
  $("btn-composer-add").onclick = () => toggleComposerAdd();
  $("btn-add-write").onclick = () => { hideComposerPanels(); if (!S.dataMode) toggleDataMode(); $("data-label").focus(); };
  $("btn-add-files").onclick = composerAddFiles;
  $("btn-add-image").onclick = () => { hideComposerPanels(); composerAddImages(); };
  $("btn-image-res").onclick = toggleImageRes;
  setupImageDropPaste();
  $("btn-add-url").onclick = () => showComposerPanel("url");
  $("btn-add-youtube").onclick = () => showComposerPanel("youtube");
  $("btn-add-search").onclick = () => showComposerPanel("search");
  $("btn-add-url-fetch").onclick = composerAddUrl;
  $("btn-add-url-cancel").onclick = hideComposerPanels;
  $("add-url-input").onkeydown = (e) => { if (e.key === "Enter") composerAddUrl(); };
  $("btn-add-yt-fetch").onclick = composerAddYouTube;
  $("btn-add-yt-cancel").onclick = hideComposerPanels;
  $("add-yt-input").onkeydown = (e) => { if (e.key === "Enter") composerAddYouTube(); };
  $("btn-add-search-go").onclick = composerAddSearch;
  $("btn-add-search-cancel").onclick = hideComposerPanels;
  $("add-search-query").onkeydown = (e) => { if (e.key === "Enter") composerAddSearch(); };
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
  $("btn-lib-add-youtube").onclick = () => showLibPanel("youtube");
  $("btn-lib-add-search").onclick = () => showLibPanel("search");
  $("btn-lib-url-fetch").onclick = addByUrl;
  $("btn-lib-url-cancel").onclick = hideLibPanels;
  $("lib-url-input").onkeydown = (e) => { if (e.key === "Enter") addByUrl(); };
  $("btn-lib-yt-fetch").onclick = addYouTube;
  $("btn-lib-yt-cancel").onclick = () => { cancelLibYouTube(); hideLibPanels(); };
  $("lib-yt-input").onkeydown = (e) => { if (e.key === "Enter") addYouTube(); };
  $("btn-lib-search-go").onclick = braveSearch;
  $("btn-lib-search-cancel").onclick = () => { cancelLibSearch(); hideLibPanels(); };
  $("btn-lib-compile").onclick = async () => {
    if (!S.activeLibrary) { toast("Select or create a library first"); return; }
    const libId = S.activeLibrary.id;
    await flushLibrarySave();   // compile the text the user can see, not the last save
    libRunInFlight.add(libId);
    try {
      await runCompile("library", libId,
                       { force: $("lib-compile-force").checked,
                         stillCurrent: () => !!S.activeLibrary && S.activeLibrary.id === libId },
                       $("lib-compile-progress"), $("lib-compile-badge"));
    } finally { libRunInFlight.delete(libId); }
  };
  $("lib-name").oninput = () => saveLibrary();

  // Global keyboard shortcuts.
  document.addEventListener("keydown", (e) => {
    // Esc backs out of whatever dialog is on top. Only #modal-prompt used to handle
    // this, and only while its input had focus; every other dialog was Esc-proof.
    if (e.key === "Escape" && modalIsOpen()) { e.preventDefault(); dismissModal(); return; }
    // Enter activates a dialog's primary button. Deliberately only `.primary` — the
    // one-button-away destructive actions (delete profile, clear history) are `.danger`
    // and should stay a deliberate click. Textareas keep Enter for newlines.
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && modalIsOpen()) {
      const el = document.activeElement;
      if (el && (el.tagName === "TEXTAREA" || el.tagName === "BUTTON")) return;
      const top = $(MODAL_STACK[MODAL_STACK.length - 1].id);
      if (!top) return;
      // #modal-prompt has its own Enter handler; the library browser is a workspace
      // rather than a form, so Enter there means "save the row I'm editing".
      if (top.id === "modal-prompt") return;
      if (top.id === "modal-prompt-browser") {
        if (el && el.id === "pb-edit-name" && !$("btn-pb-save").disabled) {
          e.preventDefault(); $("btn-pb-save").click();
        }
        return;
      }
      const primary = top.querySelector(".modal-btns button.primary");
      if (primary && !primary.disabled) { e.preventDefault(); primary.click(); }
      return;
    }
    if (e.ctrlKey && (e.key === "n" || e.key === "N")) {
      // Both variants need the guard - Ctrl+Shift+N used to fire mid-rename.
      const el = document.activeElement;
      if (modalIsOpen()) return;
      if (el && (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable)) return;
      e.preventDefault();
      if (e.shiftKey) newPrivateChat(); else newChat();
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
  // Re-entering the tab re-reads the model lists. Without this, a server that was down
  // at page load — or a model pulled since — needs a full reload to show up.
  if (!S.evalRunning && !S.evalGenRunning) S.evalModelCache = {};
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
  // An empty array is truthy, so caching one would pin the dropdown to
  // "(no models — check server)" for the rest of the session after a single blip.
  // Only a non-empty list is worth remembering.
  if (!models || !models.length) {
    try {
      const r = await api(`/api/models?server=${encodeURIComponent(serverUrl)}`);
      models = r.models || [];
      if (models.length) S.evalModelCache[serverUrl] = models;
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
// Bumped on every render. The model fetches are async and share one <select>, so a
// slow reply from a previous project would otherwise repopulate the dropdown with the
// wrong server's models and write the result back into an abandoned project object.
let evalRenderToken = 0;

function renderEvalProject() {
  const p = S.evalProject; if (!p) return;
  const token = ++evalRenderToken;
  $("eval-name").value = p.name || "";
  $("eval-prompt").value = p.prompt_template || "";
  evalFillServerSelect($("eval-gen-server"), p.gen_server_url);
  evalFillServerSelect($("eval-grader-server"), p.grader_server_url || p.gen_server_url);
  evalFillModelSelect($("eval-gen-server").value, $("eval-gen-model"), p.gen_model).then(() => {
    if (token !== evalRenderToken) return;
    p.gen_model = $("eval-gen-model").value || p.gen_model;
  });
  evalFillModelSelect($("eval-grader-server").value, $("eval-grader-model"), p.grader_model || p.gen_model).then(() => {
    if (token !== evalRenderToken) return;
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

// `silent` is for the auto-save after a run: same write, no "saved" toast.
async function saveEvalProject({ silent = false } = {}) {
  const p = S.evalProject; if (!p) return;
  evalSyncFromUI();   // one sync path, so save and run can never disagree
  try {
    const r = await api("/api/evals", { method: "POST", body: { eval: p } });
    // Keep the live object rather than swapping in the server's copy: every grid cell
    // and criterion row has an event handler closed over these exact objects, and
    // replacing them would silently orphan the whole editor until the next re-render.
    // The id (assigned on first save) and updated stamp are all we need back.
    p.id = (r.eval && r.eval.id) || p.id;
    p.updated = (r.eval && r.eval.updated) || p.updated;
    S.evals = r.evals || [];
    populateEvalProjectSelect();
    if (!silent) toast("Evaluation saved");
  } catch (e) { toast("Save failed: " + e.message); }
}

async function deleteEvalProject() {
  const p = S.evalProject; if (!p) return;
  if (!p.id) { newEvalProject(); return; }
  if (!(await confirmModal("Delete this evaluation project?"))) return;
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
  // Emptying the box reverts to the old name — re-render so the header stops showing
  // a blank field that disagrees with the column it names.
  if (newName === old) { renderEvalGrid(); return; }
  if (p.columns.includes(newName)) { toast("A column with that name already exists"); renderEvalGrid(); return; }
  p.columns[ci] = newName;
  p.rows.forEach((r) => { r[newName] = r[old]; delete r[old]; });
  if (p.output_column === old) p.output_column = newName;
  if (p.gen_instructions && p.gen_instructions[old] != null) {
    p.gen_instructions[newName] = p.gen_instructions[old];
    delete p.gen_instructions[old];
  }
  // Follow the rename into the prompt. fill_prompt only substitutes placeholders whose
  // name is a current input column, so a stale {Old} would otherwise be sent to the
  // model as literal text — silently, with nothing in the UI to hint at it.
  const before = p.prompt_template || "";
  const after = before.replace(/\{([^{}]+)\}/g, (m, key) => (key.trim() === old ? `{${newName}}` : m));
  if (after !== before) {
    p.prompt_template = after;
    $("eval-prompt").value = after;
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
    const d = await promptModal("Split the text file by (blank = blank lines):", "",
      { checkbox: "Treat the delimiter as a regular expression" });
    if (d === null) return;
    delimiter = d.value;
    isRegex = d.checked;
    const cols = p.columns.join(", ");
    const tc = await promptModal(`Put each chunk into which column? (${cols})`, p.columns[0]);
    if (tc === null) return;
    targetCol = tc.trim() || p.columns[0];
  }
  setStatus("Choose a file…");
  try {
    const r = await api("/api/evals/import", { method: "POST",
      body: { kind, delimiter, is_regex: isRegex } });
    // Nothing is mutated before this point: creating the target column earlier would
    // strand it in the project when the user cancels the file picker.
    if (!r.ok) { if (!r.cancelled) toast("Import failed: " + (r.error || "unknown")); setStatus(""); return; }
    if (kind === "csv") {
      p.columns = r.columns.length ? r.columns : p.columns;
      p.rows = r.rows || [];
      if (!p.rows.length) p.rows = [{}];
      if (!p.columns.includes(p.output_column)) {
        // The import replaced the columns, so add somewhere for responses to land.
        if (!p.columns.includes("Response")) {
          p.columns.push("Response");
          p.rows.forEach((row) => (row.Response = ""));
        }
        p.output_column = "Response";
      }
      // Instructions keyed to the columns the import just replaced are dead weight —
      // they'd never match a target column again but would still be saved and shown.
      Object.keys(p.gen_instructions || {}).forEach((k) => {
        if (!p.columns.includes(k)) delete p.gen_instructions[k];
      });
      toast(`Imported ${p.rows.length} row(s) from ${r.name}`);
    } else {
      if (!p.columns.includes(targetCol)) {
        p.columns.push(targetCol);
        p.rows.forEach((row) => (row[targetCol] = ""));
      }
      const cells = r.cells || [];
      // Fill the target column, extending rows as needed.
      cells.forEach((val, i) => {
        if (i >= p.rows.length) { const row = {}; p.columns.forEach((c) => (row[c] = "")); p.rows.push(row); }
        p.rows[i][targetCol] = val;
      });
      toast(`Imported ${cells.length} cell(s) into "${targetCol}" from ${r.name}`);
    }
    renderEvalGrid(); renderEvalOutputCol(); renderEvalChips(); renderEvalGenInstr();
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
    const max = document.createElement("input");
    max.type = "number"; max.className = "eval-crit-num"; max.value = c.max == null ? 10 : c.max;
    max.title = "Max score";
    const rangeWrap = document.createElement("span");
    rangeWrap.className = "eval-crit-range";
    rangeWrap.appendChild(document.createTextNode(" "));
    rangeWrap.appendChild(min);
    rangeWrap.appendChild(document.createTextNode("–"));
    rangeWrap.appendChild(max);
    // `parseInt(v) || fallback` turned a legitimate 0 into the fallback, so 0 couldn't
    // be entered as a bound. Take any finite number and only fall back on a blank box.
    const readNum = (el, fallback) => {
      const n = parseFloat(el.value);
      return Number.isFinite(n) ? n : fallback;
    };
    const syncRange = () => {
      rangeWrap.style.display = mode.value === "score" ? "" : "none";
      // A reversed range can't produce a percentage; aggregate() falls back to 1–10,
      // so flag it here rather than letting the scores quietly come out wrong. An
      // absent bound is not an error — it just means the 1–10 default, same as the
      // number boxes show.
      const lo = c.min == null ? 1 : c.min;
      const hi = c.max == null ? 10 : c.max;
      const bad = mode.value === "score" && !(lo < hi);
      rangeWrap.classList.toggle("invalid", bad);
      rangeWrap.title = bad ? "Min must be less than max — scoring falls back to 1–10" : "";
    };
    min.onchange = () => { c.min = readNum(min, 1); syncRange(); };
    max.onchange = () => { c.max = readNum(max, 10); syncRange(); };
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
  p.gen_server_url = $("eval-gen-server").value || p.gen_server_url;
  p.grader_server_url = $("eval-grader-server").value || p.grader_server_url;
  // The model selects read "" while their list is still loading (or when the server
  // is unreachable). Never let that placeholder erase a model the project already has.
  p.gen_model = $("eval-gen-model").value || p.gen_model;
  p.grader_model = $("eval-grader-model").value || p.grader_model;
  p.output_column = $("eval-output-col").value || p.output_column;
  p.input_columns = p.columns.filter((c) => c !== p.output_column);
  p.gen_num_rows = parseInt($("eval-gen-rows").value, 10) || p.gen_num_rows || 10;
}

// Warn about {placeholders} that aren't input columns: fill_prompt only substitutes
// those, so a stale or mistyped one reaches the model as literal text with no other
// signal. Advisory only — braces in a prompt are sometimes deliberate.
function evalUnknownPlaceholders(p) {
  const known = new Set(p.input_columns || []);
  const found = new Set();
  for (const m of (p.prompt_template || "").matchAll(/\{([^{}]+)\}/g)) {
    const key = m[1].trim();
    if (!known.has(key)) found.add(key);
  }
  return [...found];
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
  // The server grades with this and 400s without it; catching it here keeps the run
  // from failing after the user has already waited for generation.
  if (!(p.grader_model || p.gen_model)) { toast("Select a grader model"); return; }
  const unknown = evalUnknownPlaceholders(p);
  if (unknown.length) toast(`Prompt has unknown placeholder(s): ${unknown.map((u) => `{${u}}`).join(", ")}`, 6000);

  S.evalRunId = uid();
  S.evalRunning = true;
  // Snapshot the inputs the run is about to use. The results tables read only from
  // here, so editing the grid (or the write-back below) can't rewrite history.
  S.evalRun = {
    batch, models: [],
    criteria: JSON.parse(JSON.stringify((p.criteria || []).filter((c) => (c.label || "").trim()))),
    inputCols: p.columns.filter((c) => c !== p.output_column),
    inputRows: p.rows.map((r) => ({ ...r })),
  };
  $("btn-eval-stop").classList.remove("hidden");
  $("btn-eval-run").disabled = $("btn-eval-run-batch").disabled = true;
  $("eval-results").innerHTML = "";
  $("eval-progress").textContent = "Starting…";

  // A single-model run fills the output column in the grid; a batch run has no one
  // authoritative response, so it leaves the grid alone.
  const writeBack = !batch && !!p.output_column;
  let wroteAny = false;
  let gridDirty = false;
  const flushGrid = () => { if (gridDirty) { gridDirty = false; renderEvalGrid(); } };
  const gridTimer = writeBack ? setInterval(flushGrid, 400) : null;

  try {
    await streamSSE("/api/evals/run",
      { eval: p, run_id: S.evalRunId, batch },
      {
        start: (d) => { $("eval-progress").textContent = `Running ${d.total_rows} row(s)…`; },
        model_start: (d) => {
          S.evalRun.models[d.index] = { server: d.server, servers: d.servers, model: d.model, rows: [], aggregate: null };
          $("eval-progress").textContent = `Model ${d.index + 1}/${d.total}: ${d.model} — generating…`;
        },
        gen_progress: (d) => {
          const mi = S.evalRun.models.length - 1;
          $("eval-progress").textContent = `${S.evalRun.models[mi] ? S.evalRun.models[mi].model : ""}: generated ${d.done}/${d.total}`;
        },
        row_result: (d) => {
          const m = S.evalRun.models[d.model_index];
          if (m) m.rows[d.index] = { response: d.response, grades: d.grades, ungraded: d.ungraded, error: d.error };
          if (writeBack && p.rows[d.index]) {
            p.rows[d.index][p.output_column] = d.response || "";
            wroteAny = true;
            gridDirty = true;   // batched by the timer — a 200-row run shouldn't rebuild the table 200 times
          }
          $("eval-progress").textContent = `${m ? m.model : ""}: graded row ${d.index + 1}`;
        },
        model_done: (d) => {
          const m = S.evalRun.models[d.index];
          if (m) { m.aggregate = d.aggregate; m.server = d.server; m.servers = d.servers; }
          renderEvalResults();
        },
        status: (d) => { if (d.message) setStatus(d.message); },
        summary: (d) => { S.evalRun.summary = d; renderEvalResults(); },
        error: (d) => { toast("Eval error: " + (d.message || "unknown")); },
        done: (d) => {
          $("eval-progress").textContent = d.stopped ? "Stopped." : "Done.";
          renderEvalResults();
        },
      }
    );
  } catch (e) {
    toast("Run failed: " + e.message);
  } finally {
    // The one place the run is torn down: streamSSE resolves normally on an HTTP
    // error (no throw, no `done` frame), so `finally` is what guarantees the buttons
    // come back in every path — success, stop, 400, or thrown error.
    if (gridTimer) clearInterval(gridTimer);
    flushGrid();
    endEvalRun();
  }

  // Persist the responses that just landed in the grid.
  if (wroteAny) await saveEvalProject({ silent: true });
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

  // Slots are reserved once the run has actually started (so the placeholder rows
  // aren't uploaded with the request) and then filled by index: under Parallel
  // Processing rows finish out of order, and pushing on arrival would scramble them.
  let base = 0;
  let started = false;
  let gridDirty = false;
  const filled = new Array(numRows).fill(false);
  const flushGrid = () => { if (gridDirty) { gridDirty = false; renderEvalGrid(); } };
  const reserveRows = () => {
    // Drop a single leading all-empty row (the placeholder a fresh project starts
    // with) so generated data doesn't sit under a blank line.
    if (p.rows.length === 1 && !p.columns.some((c) => (p.rows[0][c] || "").trim())) p.rows.length = 0;
    base = p.rows.length;
    for (let i = 0; i < numRows; i++) {
      const row = {}; p.columns.forEach((c) => (row[c] = ""));
      p.rows.push(row);
    }
    gridDirty = true;
  };
  const gridTimer = setInterval(flushGrid, 400);

  try {
    await streamSSE("/api/evals/gen-data",
      { eval: p, run_id: S.evalGenRunId, num_rows: numRows },
      {
        start: (d) => {
          $("eval-gen-progress").textContent = `Generating ${d.total} row(s)…`;
          started = true; reserveRows(); flushGrid();
        },
        row_result: (d) => {
          if (!started) return;
          const i = d.index;
          if (typeof i !== "number" || i < 0 || i >= numRows) return;
          if (d.ok === false) {
            setStatus(`Row ${i + 1}: the model's reply didn't parse as JSON — left blank`);
            return;
          }
          const row = p.rows[base + i];
          if (!row) return;
          p.columns.forEach((c) => { if (d.row && d.row[c] != null) row[c] = d.row[c]; });
          filled[i] = true;
          gridDirty = true;
        },
        gen_progress: (d) => { $("eval-gen-progress").textContent = `Generated ${d.done}/${d.total}`; },
        status: (d) => { if (d.message) setStatus(d.message); },
        error: (d) => { toast("Generation error: " + (d.message || "unknown")); },
        done: (d) => { $("eval-gen-progress").textContent = d.stopped ? "Stopped." : "Done."; },
      }
    );
  } catch (e) {
    toast("Generation failed: " + e.message);
  } finally {
    clearInterval(gridTimer);
    // Reclaim the slots nothing landed in (stopped early, parse failure, HTTP error).
    // Descending, so each splice leaves the lower indices untouched.
    if (started) for (let i = numRows - 1; i >= 0; i--) if (!filled[i]) p.rows.splice(base + i, 1);
    if (!p.rows.length) evalAddRow(); else renderEvalGrid();
    endEvalGen();
  }
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

// "graded 8/10" — makes rows the grader never scored visible instead of letting them
// silently shrink the sample the average is drawn from.
function evalGradedNote(agg) {
  if (!agg || agg.total == null || agg.graded == null || agg.graded === agg.total) return "";
  return `<br>graded ${agg.graded}/${agg.total} rows`;
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
      `<div class="eval-score-cap">Overall prompt score<br><span class="muted">${evEsc(m.model)}` +
      `${evalGradedNote(agg)}</span></div>`;
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
    const nameTd = document.createElement("td");
    nameTd.textContent = m.model;
    // Where the work really ran — a parallel run fans across every lane hosting the
    // model, so this is not always the server picked in the batch list.
    if (m.server) nameTd.title = `ran on: ${m.server}`;
    tr.appendChild(nameTd);
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
  // Read the run's snapshot, never the live project: the grid is editable during and
  // after a run, and a single-model run writes responses back into it.
  const run = S.evalRun || {};
  const inputCols = run.inputCols || [];
  const inputRows = run.inputRows || [];
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
      td.textContent = (inputRows[i] && inputRows[i][c]) || ""; tr.appendChild(td);
    });
    const rTd = document.createElement("td"); rTd.className = "eval-td-response";
    if (res.error) {
      // A failed generation is shown as a failure and left ungraded — it is never sent
      // to the grader, so it can't contribute a fabricated score to the average.
      rTd.classList.add("eval-td-error");
      rTd.textContent = `⚠ generation failed: ${res.error}`;
      rTd.title = res.error;
    } else {
      rTd.textContent = res.response || "";
    }
    tr.appendChild(rTd);
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

// ============================ Batch tab ==============================
// Distinct from the chat composer's 📂 Batch button, which runs a folder of .txt/.md
// files where each file IS the prompt. Here an input item is CONTENT and one prompt
// template runs against every item, with the results going to a batch chat and/or to
// exported files. Self-contained like the Database tab: its own bind function, its own
// run id, and no dependency on S.chat.

/** Field ids that map 1:1 onto a project key, so render/sync stay one list. */
const BATCH_FIELDS = [
  ["batch-name", "name", "str"],
  ["batch-max-chars", "max_item_chars", "int"],
  ["batch-yt-comments", "yt_comments", "bool"],
  ["batch-yt-max", "yt_max_comments", "int"],
  ["batch-server", "server_url", "str"],
  ["batch-model", "model", "str"],
  ["batch-ctx", "num_ctx", "int"],
  ["batch-system-on", "system_on", "bool"],
  ["batch-system-prompt", "system_prompt", "str"],
  ["batch-pre-on", "pre_on", "bool"],
  ["batch-pre-prompt", "pre_prompt", "str"],
  ["batch-strict", "library_strict", "bool"],
  ["batch-multipass", "multi_pass", "bool"],
  ["batch-passes", "passes", "int"],
  ["batch-mp-system", "pass_use_system", "bool"],
  ["batch-eval-prompt", "eval_prompt", "str"],
  ["batch-output-mode", "output_mode", "str"],
  ["batch-combined-name", "combined_name", "str"],
  ["batch-name-mode", "name_mode", "str"],
  ["batch-name-prefix", "name_prefix", "str"],
  ["batch-name-suffix", "name_suffix", "str"],
  ["batch-ext", "ext", "str"],
  ["batch-name-prompt", "name_prompt", "str"],
  ["batch-include-source", "include_source", "bool"],
  ["batch-include-prompt", "include_prompt", "bool"],
  ["batch-strip-md", "strip_markdown", "bool"],
  ["batch-image-fullres", "image_full_res", "bool"],
];

function newBatchProjectLocal() {
  // The server ships a fully-defaulted project in /api/state, so the defaults live in
  // exactly one place (app/batch.py) rather than being duplicated here.
  const base = S.defaultBatchProject ? JSON.parse(JSON.stringify(S.defaultBatchProject)) : {};
  return {
    ...base,
    id: "", name: "Untitled batch",
    server_url: currentServerUrl() || base.server_url || "",
    model: getSelectedModel() || base.model || "",
    sources: [{ kind: "folder", path: "", recursive: true, exts: [] }],
  };
}

function initBatchTab() {
  if (!S.batch.inited) {
    bindBatchEvents();
    S.batch.inited = true;
  }
  if (!S.batch.project) S.batch.project = newBatchProjectLocal();
  $("batch-exts").textContent = (S.batchExts || []).join(", ");
  populateBatchProjectSelect();
  renderBatchProject();
}

function populateBatchProjectSelect() {
  const sel = $("batch-project-select");
  sel.innerHTML = "";
  const optNew = document.createElement("option");
  optNew.value = ""; optNew.textContent = "— New (unsaved) —";
  sel.appendChild(optNew);
  (S.batch.projects || []).forEach((p) => {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = p.name || "Untitled batch";
    sel.appendChild(o);
  });
  sel.value = (S.batch.project && S.batch.project.id) || "";
}

// ---- sources ----
/** Per-kind field markup. Values are harvested back in batchSyncFromUI by data-key. */
function batchSourceFields(src, idx) {
  const k = src.kind || "folder";
  if (k === "youtube") {
    return `<textarea class="bsrc-field" data-key="urls" rows="3"
              placeholder="One YouTube URL per line">${escapeHtml(src.urls || "")}</textarea>`;
  }
  if (k === "playlist") {
    return `<div class="control-row">
        <input type="text" class="bsrc-field" data-key="url" style="flex:1 1 22em"
               placeholder="https://www.youtube.com/playlist?list=…" value="${escapeHtml(src.url || "")}" />
        <label>Max videos: <input type="number" class="bsrc-field" data-key="limit" min="0"
               style="width:6em" value="${src.limit || 0}" title="0 = every video" /></label>
      </div>`;
  }
  if (k === "search") {
    return `<div class="control-row">
        <input type="text" class="bsrc-field" data-key="query" style="flex:1 1 20em"
               placeholder="Search query" value="${escapeHtml(src.query || "")}" />
        <label>Pages: <input type="number" class="bsrc-field" data-key="max_results" min="1" max="50"
               style="width:5em" value="${src.max_results || 5}"
               title="Up to 50 — the server clamps to core.MAX_BATCH_CRAWL_PAGES" /></label>
        <input type="text" class="bsrc-field" data-key="sites" style="flex:1 1 14em"
               placeholder="Limit to sites (comma separated)" value="${escapeHtml((src.sites || []).join(", "))}" />
      </div>`;
  }
  // folder + images share the picker markup, so batchSyncFromUI needs no new case.
  const label = k === "images" ? "🖼 Choose image folder…" : "📂 Choose folder…";
  return `<div class="control-row">
      <button class="small bsrc-pick" data-idx="${idx}">${label}</button>
      <span class="muted bsrc-path">${escapeHtml(src.path || "(no folder chosen)")}</span>
      <label class="chk"><input type="checkbox" class="bsrc-field" data-key="recursive"
             ${src.recursive ? "checked" : ""} /> Include subfolders</label>
    </div>`;
}

function renderBatchSources() {
  const p = S.batch.project; if (!p) return;
  const box = $("batch-sources");
  box.innerHTML = "";
  (p.sources || []).forEach((src, idx) => {
    const row = document.createElement("div");
    row.className = "batch-source";
    row.dataset.idx = String(idx);
    row.innerHTML =
      `<div class="control-row">
         <select class="bsrc-kind">
           <option value="folder">📁 Folder of documents</option>
           <option value="images">🖼 Folder of images</option>
           <option value="youtube">▶ YouTube video(s)</option>
           <option value="playlist">▶ YouTube playlist</option>
           <option value="search">🔍 Web search results</option>
         </select>
         <span class="spacer"></span>
         <button class="small danger bsrc-del">✕ Remove</button>
       </div>
       ${batchSourceFields(src, idx)}`;
    row.querySelector(".bsrc-kind").value = src.kind || "folder";
    box.appendChild(row);
  });

  box.querySelectorAll(".bsrc-kind").forEach((sel) => {
    sel.onchange = () => {
      const idx = Number(sel.closest(".batch-source").dataset.idx);
      batchSyncFromUI();
      // Kind change swaps the whole field set; keep only the shared keys.
      S.batch.project.sources[idx] = { kind: sel.value, recursive: true, exts: [] };
      renderBatchSources();
    };
  });
  box.querySelectorAll(".bsrc-del").forEach((btn) => {
    btn.onclick = () => {
      const idx = Number(btn.closest(".batch-source").dataset.idx);
      batchSyncFromUI();
      S.batch.project.sources.splice(idx, 1);
      renderBatchSources();
    };
  });
  box.querySelectorAll(".bsrc-pick").forEach((btn) => {
    btn.onclick = async () => {
      const idx = Number(btn.dataset.idx);
      batchSyncFromUI();
      try {
        const r = await api("/api/pick-folder", {
          method: "POST", body: { title: "Choose a folder of documents to process" } });
        if (r.path) { S.batch.project.sources[idx].path = r.path; renderBatchSources(); }
      } catch (e) { toast("Folder picker failed: " + e.message); }
    };
  });
}

// ---- render / sync ----
function renderBatchProject() {
  const p = S.batch.project; if (!p) return;
  BATCH_FIELDS.forEach(([id, key, type]) => {
    const el = $(id); if (!el) return;
    if (type === "bool") el.checked = !!p[key];
    else el.value = p[key] == null ? "" : p[key];
  });
  // A project that has never set one opens showing the shipped template to edit, rather
  // than an empty box — the same fallback loadChatObject applies to #mp-eval-prompt.
  // (An empty value still works server-side: fill_eval_prompt falls back to the same
  // default, so an older saved project is unaffected either way.)
  if (!p.eval_prompt) $("batch-eval-prompt").value = S.defaultEvalPrompt || "";
  batchFillServerSelect($("batch-server"), p.server_url);
  batchFillModelSelect(p.server_url, $("batch-model"), p.model);
  batchFillCtxSelect(p.num_ctx);
  const mode = p.export_mode || "per_item";
  document.querySelectorAll('input[name="batch-export-mode"]').forEach((r) => {
    r.checked = r.value === mode;
  });
  $("batch-outdir").textContent = p.output_dir || "(no folder chosen)";
  updateBatchLibraryLabel();
  renderBatchSources();
  renderBatchRefImages();
  updateBatchVisibility();
}

/** The project's reference images — sent alongside EVERY item. Kept outside
 *  BATCH_FIELDS (like `sources` and `output_dir`) because it isn't one control's
 *  value; renderBatchProject and the picker are its only writers. */
function renderBatchRefImages() {
  const p = S.batch.project; if (!p) return;
  const box = $("batch-refimgs");
  box.innerHTML = "";
  (p.reference_images || []).forEach((rec) => {
    const wrap = document.createElement("span");
    wrap.className = "batch-refimg";
    const img = document.createElement("img");
    img.src = imageUrl(rec.id, true);
    img.alt = rec.name || "";
    img.title = `${rec.name || "image"} — click to view`;
    img.onclick = () => showImage({ id: rec.id, image: rec }, false);
    const x = document.createElement("button");
    x.type = "button"; x.className = "batch-refimg-x";
    x.textContent = "✕"; x.title = "Remove this reference image";
    x.onclick = () => {
      p.reference_images = (p.reference_images || []).filter((r) => r.id !== rec.id);
      renderBatchRefImages();
    };
    wrap.appendChild(img); wrap.appendChild(x);
    box.appendChild(wrap);
  });
  const n = (p.reference_images || []).length;
  $("batch-refimgs-note").textContent = n
    ? `${n} reference image(s) sent with every item.`
    : "";
}

async function batchAddRefImages() {
  const btn = $("btn-batch-addrefimg");
  btn.disabled = true;
  setStatus("Waiting for image selection…");
  try {
    const r = await api("/api/batch/reference-images", { method: "POST", body: {} });
    const p = S.batch.project;
    p.reference_images = [...(p.reference_images || []), ...(r.images || [])];
    renderBatchRefImages();
    if ((r.images || []).length) toast(`Added ${r.images.length} reference image(s)`);
    if ((r.errors || []).length) toast("Some images failed: " + r.errors.join("; "), 6000);
  } catch (e) { toast("Could not add reference images: " + e.message); }
  btn.disabled = false;
  setStatus("");
}

/** DOM -> S.batch.project. The SINGLE sync path, so Save and Run can never disagree
 *  about what the user configured (the discipline evalSyncFromUI established). */
function batchSyncFromUI() {
  const p = S.batch.project; if (!p) return p;
  BATCH_FIELDS.forEach(([id, key, type]) => {
    const el = $(id); if (!el) return;
    if (type === "bool") p[key] = !!el.checked;
    else if (type === "int") p[key] = parseInt(el.value, 10) || 0;
    else p[key] = el.value;
  });
  const picked = document.querySelector('input[name="batch-export-mode"]:checked');
  p.export_mode = picked ? picked.value : "per_item";
  // `parseInt("") || 0` above leaves a blank box as 0, and generate_one reads
  // `passes` as the number of refinement rounds — so Multi-Pass would be ticked and do
  // nothing at all. Same floor syncSettingsFromUI applies to the chat's #mp-passes.
  p.passes = Math.max(1, p.passes || 2);

  document.querySelectorAll("#batch-sources .batch-source").forEach((row) => {
    const idx = Number(row.dataset.idx);
    const src = p.sources[idx]; if (!src) return;
    src.kind = row.querySelector(".bsrc-kind").value;
    row.querySelectorAll(".bsrc-field").forEach((el) => {
      const key = el.dataset.key;
      if (el.type === "checkbox") src[key] = !!el.checked;
      else if (el.type === "number") src[key] = parseInt(el.value, 10) || 0;
      else if (key === "sites") {
        src.sites = el.value.split(",").map((s) => s.trim()).filter(Boolean);
      } else src[key] = el.value;
    });
  });
  return p;
}

/** Show only the controls that apply to the chosen output + export mode. Also the
 *  place the beside-source guard is surfaced, matching the server-side check. */
function updateBatchVisibility() {
  const p = S.batch.project; if (!p) return;
  const files = (p.output_mode || "both") !== "chat";
  const mode = p.export_mode || "per_item";
  $("batch-file-options").classList.toggle("hidden", !files);
  $("batch-combined-row").classList.toggle("hidden", mode !== "combined");
  $("batch-outdir-row").classList.toggle("hidden", mode === "beside_source");
  $("batch-nameprompt-row").classList.toggle("hidden", p.name_mode !== "llm");
  $("batch-multipass-panel").classList.toggle("hidden", !p.multi_pass);
  const needsAppend = files && mode === "beside_source"
    && !(p.name_prefix || "").trim() && !(p.name_suffix || "").trim();
  $("batch-beside-warn").classList.toggle("hidden", !needsAppend);
}

function batchFillServerSelect(sel, chosen) {
  sel.innerHTML = "";
  S.servers.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.url; o.textContent = s.label;
    sel.appendChild(o);
  });
  if (chosen && [...sel.options].some((o) => o.value === chosen)) sel.value = chosen;
}

async function batchFillModelSelect(serverUrl, sel, chosen) {
  sel.innerHTML = '<option value="">loading…</option>';
  let models = S.batch.modelCache[serverUrl];
  if (!models || !models.length) {
    try {
      const r = await api(`/api/models?server=${encodeURIComponent(serverUrl)}`);
      models = r.models || [];
      if (models.length) S.batch.modelCache[serverUrl] = models;
    } catch (e) { models = []; }
  }
  sel.innerHTML = "";
  const list = [...models];
  if (chosen && !list.includes(chosen)) list.unshift(chosen);
  if (!list.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "(no models — check server)";
    sel.appendChild(o);
    return;
  }
  list.forEach((m) => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  sel.value = chosen && list.includes(chosen) ? chosen : list[0];
}

function batchFillCtxSelect(chosen) {
  const sel = $("batch-ctx");
  sel.innerHTML = "";
  const optDef = document.createElement("option");
  optDef.value = "0"; optDef.textContent = "Default";
  sel.appendChild(optDef);
  (S.contextLengths || []).forEach((n) => {
    const o = document.createElement("option");
    o.value = String(n); o.textContent = n.toLocaleString();
    sel.appendChild(o);
  });
  sel.value = String(chosen || 0);
}

function updateBatchLibraryLabel() {
  const ids = (S.batch.project && S.batch.project.library_ids) || [];
  const names = S.libraries.filter((l) => ids.includes(l.id)).map((l) => l.name);
  $("batch-library-state").textContent = names.length ? names.join(", ") : "none selected";
}

/** Reuses the chat's library-selector modal. That modal writes into S.chat on save, so
 *  the Batch tab swaps in its own save handler for the duration of the dialog. */
function openBatchLibrarySelector() {
  const p = S.batch.project; if (!p) return;
  const saved = S.chat;
  // openLibrarySelector reads S.chat.library_ids to pre-tick the boxes.
  S.chat = { library_ids: p.library_ids || [] };
  openLibrarySelector();
  S.chat = saved;
  const btn = $("btn-libselect-save");
  const prev = btn.onclick;
  btn.onclick = async () => {
    p.library_ids = Array.from(
      $("libselect-list").querySelectorAll("input:checked")).map((c) => c.value);
    updateBatchLibraryLabel();
    btn.onclick = prev;
    closeModal();
  };
}

// ---- project CRUD ----
async function saveBatchProject({ silent = false } = {}) {
  const p = batchSyncFromUI();
  try {
    const r = await api("/api/batch/projects", { method: "POST", body: p });
    S.batch.project = r.project;
    S.batch.projects = r.projects || [];
    populateBatchProjectSelect();
    if (!silent) toast("Batch project saved");
  } catch (e) { toast("Save failed: " + e.message); }
}

async function loadBatchProject(id) {
  if (!id) { S.batch.project = newBatchProjectLocal(); renderBatchProject(); return; }
  try {
    const r = await api(`/api/batch/projects/${id}`);
    S.batch.project = r.project;
    renderBatchProject();
  } catch (e) { toast("Could not load that batch: " + e.message); }
}

async function deleteBatchProject() {
  const p = S.batch.project;
  if (!p || !p.id) { S.batch.project = newBatchProjectLocal(); renderBatchProject(); return; }
  if (!await confirmModal(`Delete the batch project "${p.name || "Untitled"}"?`)) return;
  try {
    const r = await api(`/api/batch/projects/${p.id}`, { method: "DELETE" });
    S.batch.projects = r.projects || [];
    S.batch.project = newBatchProjectLocal();
    populateBatchProjectSelect();
    renderBatchProject();
    toast("Batch project deleted");
  } catch (e) { toast("Delete failed: " + e.message); }
}

// ---- preview ----
function renderBatchItems() {
  const box = $("batch-items");
  const items = S.batch.items || [];
  if (!items.length) { box.innerHTML = ""; return; }
  box.innerHTML =
    `<div class="batch-items-head">${items.length} item(s) ready</div>` +
    items.map((it) =>
      `<div class="batch-item-row">
         <span class="tag ${escapeHtml(it.kind)}">${escapeHtml(it.kind)}</span>
         <span class="batch-item-title">${escapeHtml(it.title)}</span>
         <span class="muted">${(it.chars || 0).toLocaleString()} chars</span>
       </div>`).join("");
}

async function previewBatch() {
  if (S.batch.running) return;
  const p = batchSyncFromUI();
  const runId = uid();
  S.batch.previewRunId = runId;
  S.batch.items = [];
  $("btn-batch-preview-stop").classList.remove("hidden");
  $("batch-preview-status").textContent = "Reading inputs…";

  const ui = makeProgressUI($("batch-preview-progress"), {
    onCancel: () => api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {}),
  });
  ui.plan([{ id: "resolve", label: "Reading inputs", weight: 1 }]);

  await streamSSE("/api/batch/preview", { project: p, run_id: runId }, {
    progress: (d) => {
      ui.update({ phase: "resolve", label: d.phase || "reading",
                  done: d.done || 0, total: d.total || 0, unit: "item" });
      if (d.name) ui.line(d.name);
    },
    items: (d) => {
      S.batch.items = d.items || [];
      (d.errors || []).forEach((e) => ui.line("⚠ " + e));
      $("batch-preview-status").textContent =
        `${(d.items || []).length} item(s)` + ((d.errors || []).length ? ` · ${d.errors.length} problem(s)` : "");
      renderBatchItems();
    },
    error: (d) => { ui.line("✗ " + d.message); toast("Preview failed: " + d.message); },
    done: () => { ui.finish(); $("btn-batch-preview-stop").classList.add("hidden"); },
  });
  S.batch.previewRunId = null;
}

// ---- run ----
function setBatchRunningUI(on) {
  S.batch.running = on;
  $("btn-batch-run").classList.toggle("hidden", on);
  $("btn-batch-stop").classList.toggle("hidden", !on);
}

function renderBatchResults() {
  const box = $("batch-results");
  const rows = S.batch.results || [];
  $("btn-batch-tochat").disabled = !rows.length;
  box.innerHTML = rows.map((r) => {
    const imgs = (r.images || []).map((rec) =>
      `<img src="${imageUrl(rec.id, true)}" alt="" data-image-id="${escapeHtml(rec.id)}"
            title="Click to view full size" />`).join("");
    return `<div class="batch-result">
       <div class="batch-result-head">${escapeHtml(r.title || "Untitled")}
         ${r.out_path ? `<span class="muted">→ ${escapeHtml(r.out_path)}</span>` : ""}
         ${(r.out_image_paths || []).map((p) =>
           `<span class="muted">🖼 ${escapeHtml(p)}</span>`).join("")}
         ${r.export_error ? `<span class="warn-inline">⚠ ${escapeHtml(r.export_error)}</span>` : ""}
       </div>
       <div class="batch-result-body">${escapeHtml(r.response || "")}</div>
       ${imgs ? `<div class="bubble-images">${imgs}</div>` : ""}
     </div>`;
  }).join("");
  // Delegated rather than per-image: the whole list is rebuilt on every item_done.
  box.querySelectorAll("[data-image-id]").forEach((img) => {
    const rec = (rows.flatMap((r) => r.images || []))
      .find((x) => x.id === img.dataset.imageId);
    img.onclick = () => showImage(rec || { id: img.dataset.imageId }, false);
  });
}

async function runBatch() {
  if (S.batch.running) return;
  const p = batchSyncFromUI();
  const runId = uid();
  S.batch.runId = runId;
  S.batch.results = [];
  renderBatchResults();
  setBatchRunningUI(true);
  $("batch-run-status").textContent = "Reading inputs…";

  const lanesBox = $("batch-lanes");
  lanesBox.innerHTML = "";
  lanesBox.classList.add("hidden");
  const lanes = {};          // lane index -> column state
  let total = 0, done = 0;

  const ui = makeProgressUI($("batch-progress"), {
    onCancel: () => api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {}),
  });
  ui.plan([
    { id: "resolve", label: "Reading inputs", weight: 0.2 },
    { id: "generate", label: "Generating", weight: 0.8 },
  ]);

  await streamSSE("/api/batch/run", { project: p, run_id: runId }, {
    progress: (d) => ui.update({ phase: "resolve", label: d.phase || "reading",
                                 done: d.done || 0, total: d.total || 0, unit: "item" }),
    plan: (d) => {
      total = d.total || 0;
      (d.resolve_errors || []).forEach((e) => ui.line("⚠ " + e));
      ui.update({ phase: "generate", label: "generating", done: 0, total, unit: "item" });
      $("batch-run-status").textContent = `0 / ${total}`;
      // Only show lane columns when there is genuinely more than one server at work;
      // a single synthetic lane is just the sequential case.
      if ((d.lanes || []).length > 1) {
        lanesBox.classList.remove("hidden");
        d.lanes.forEach((l) => {
          const col = makeLaneColumn(l);
          lanes[l.index] = col;
          lanesBox.appendChild(col.el);
        });
      }
    },
    item_start: (d) => {
      const L = lanes[d.lane];
      if (L) { L.titleEl.textContent = d.title || ""; L.body.textContent = ""; L.curContent = ""; }
      ui.line("▶ " + (d.title || d.item_id));
    },
    chunk: (d) => {
      const L = lanes[d.lane];
      if (!L) return;
      L.curContent += d.content || "";
      L.body.textContent = L.curContent;
      laneScroll(L);
    },
    item_done: (d) => {
      done += 1;
      const L = lanes[d.lane];
      if (L) { L.count += 1; L.countEl.textContent = `${L.count} done`; }
      S.batch.results.push({
        item_id: d.item_id, title: d.title || "", response: d.content || "",
        out_path: d.out_path || "", export_error: d.export_error || "",
        images: d.images || [], out_image_paths: d.out_image_paths || [],
      });
      renderBatchResults();
      ui.update({ phase: "generate", label: "generating", done, total, unit: "item" });
      $("batch-run-status").textContent = `${done} / ${total}`;
      if (d.export_error) ui.line("⚠ " + d.export_error);
      else if (d.out_path) ui.line("✓ " + d.out_path);
      (d.out_image_paths || []).forEach((p_) => ui.line("🖼 " + p_));
    },
    export: (d) => {
      if (d.combined) ui.line("✓ " + d.combined);
      const verb = d.stopped ? "stopped" : "complete";
      const wrote = (d.files || []).length;
      $("batch-run-status").textContent =
        `Batch ${verb} — ${d.count} item(s)` + (wrote ? `, ${wrote} file(s) written` : "");
      toast(`Batch ${verb} — ${d.count} item(s)` + (wrote ? `, ${wrote} file(s)` : ""), 6000);
    },
    error: (d) => { ui.line("✗ " + d.message); toast("Batch error: " + d.message, 6000); },
    done: () => { ui.finish(); setBatchRunningUI(false); },
  });
  S.batch.runId = null;
}

function stopBatch() {
  if (S.batch.runId) {
    api("/api/stop", { method: "POST", body: { run_id: S.batch.runId } }).catch(() => {});
  }
  $("batch-run-status").textContent = "Stopping…";
}

/** Promote this run's results into a real chat so they can be worked with normally. */
async function saveBatchResultsToChat() {
  const rows = S.batch.results || [];
  if (!rows.length) return;
  const p = S.batch.project || {};
  const name = await promptModal("Name for the new chat",
                                 `${p.name || "Batch"} results`);
  if (name == null) return;

  const chat = {
    id: uid(),
    name: name || "Batch results",
    created: new Date().toISOString(),
    model: p.model || "",
    server_url: p.server_url || "",
    system_prompt: p.system_prompt || "",
    system_on: !!p.system_on,
    pre_prompt: "", pre_on: false,
    isolated: true, private: false,
    group_id: S.activeGroup || DEFAULT_GROUP_ID,
    messages: [],
  };
  rows.forEach((r) => {
    chat.messages.push({ role: "user", content: `📄 ${r.title}` });
    // Pictures the model produced travel with the answer, in the same `images` shape
    // finalizePass persists. Dropping them here didn't just lose them from the chat:
    // once no saved document referenced the ids, the image sweep unlinked the bytes.
    const msg = { role: "assistant", content: r.response || "" };
    if ((r.images || []).length) msg.images = r.images;
    chat.messages.push(msg);
  });

  try {
    await api(`/api/chats/${chat.id}/persist`, { method: "POST", body: { chat } });
    await refreshChatSummaries();
    renderChatList();
    toast("Saved to chat: " + chat.name, 5000);
    switchTab("chat");
    await loadChat(chat.id);
  } catch (e) { toast("Could not save to chat: " + e.message); }
}

function bindBatchEvents() {
  $("batch-project-select").onchange = (e) => loadBatchProject(e.target.value);
  $("btn-batch-new").onclick = () => {
    S.batch.project = newBatchProjectLocal();
    populateBatchProjectSelect();
    renderBatchProject();
  };
  $("btn-batch-save").onclick = () => saveBatchProject();
  $("btn-batch-delete").onclick = deleteBatchProject;

  $("btn-batch-addsource").onclick = () => {
    batchSyncFromUI();
    S.batch.project.sources.push({ kind: "folder", path: "", recursive: true, exts: [] });
    renderBatchSources();
  };
  $("btn-batch-addrefimg").onclick = batchAddRefImages;
  $("btn-batch-preview").onclick = previewBatch;
  $("btn-batch-preview-stop").onclick = () => {
    if (S.batch.previewRunId) {
      api("/api/stop", { method: "POST", body: { run_id: S.batch.previewRunId } }).catch(() => {});
    }
  };

  $("batch-server").onchange = async () => {
    const p = batchSyncFromUI();
    await batchFillModelSelect(p.server_url, $("batch-model"), "");
    p.model = $("batch-model").value;
  };

  $("btn-batch-sys-browse").onclick = () =>
    openPromptBrowser("system", null, (text) => {
      $("batch-system-prompt").value = text;
      $("batch-system-on").checked = true;
      batchSyncFromUI();
      toast("Applied to batch");
    });
  $("btn-batch-sys-save").onclick = () =>
    openPromptBrowser("system", $("batch-system-prompt").value);
  $("btn-batch-pre-browse").onclick = () =>
    openPromptBrowser("pre", null, (text) => {
      $("batch-pre-prompt").value = text;
      $("batch-pre-on").checked = true;
      batchSyncFromUI();
      toast("Applied to batch");
    });
  $("btn-batch-pre-save").onclick = () =>
    openPromptBrowser("pre", $("batch-pre-prompt").value);

  $("btn-batch-library").onclick = openBatchLibrarySelector;

  $("btn-batch-outdir").onclick = async () => {
    batchSyncFromUI();
    try {
      const r = await api("/api/pick-folder", {
        method: "POST", body: { title: "Choose where to save the batch output" } });
      if (r.path) {
        S.batch.project.output_dir = r.path;
        $("batch-outdir").textContent = r.path;
      }
    } catch (e) { toast("Folder picker failed: " + e.message); }
  };

  // Anything that changes which controls are relevant re-syncs and re-renders.
  ["batch-output-mode", "batch-name-mode", "batch-multipass",
   "batch-name-prefix", "batch-name-suffix"].forEach((id) => {
    $(id).onchange = () => { batchSyncFromUI(); updateBatchVisibility(); };
  });
  document.querySelectorAll('input[name="batch-export-mode"]').forEach((r) => {
    r.onchange = () => { batchSyncFromUI(); updateBatchVisibility(); };
  });

  $("btn-batch-run").onclick = runBatch;
  $("btn-batch-stop").onclick = stopBatch;
  $("btn-batch-tochat").onclick = saveBatchResultsToChat;
}

// ------------------------------- go ----------------------------------
init().catch((e) => { console.error(e); toast("Startup error: " + e.message, 8000); });
