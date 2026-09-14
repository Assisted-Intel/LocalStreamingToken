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
  editingMessage: null, // index in S.chat.messages of the bubble open for in-place editing
  messagesDirty: false, // a renderMessages() was deferred while that edit was open
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
  voice: {
    on: false,
    speaking: false,
    skipSpeak: false,
    handlingTurn: false,
    pollTimer: null,
    speaker: null,
    listenState: "idle",
    speakChain: Promise.resolve(),
    helperLoading: false,
    modelsReady: false,
  },
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


// ------------------------------- File transfer -----------------------------
// Every file button in this app was written around a native dialog that opens on the
// machine running the server. From any other device that is a button which hangs. These
// two helpers give those same buttons a browser path: pick locally, stage on the server,
// hand the route the paths it already knows how to consume (see app/transfer.py).

/** Show the browser's file picker. Resolves to a (possibly empty) array of File. */
function chooseFiles({ accept = "", multiple = true, capture = "" } = {}) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    if (accept) input.accept = accept;
    if (multiple) input.multiple = true;
    // On a phone this is what puts "Take Photo" at the top of the sheet.
    if (capture) input.capture = capture;
    input.style.position = "fixed";
    input.style.opacity = "0";
    input.style.pointerEvents = "none";
    document.body.appendChild(input);
    let done = false;
    const finish = (files) => {
      if (done) return;
      done = true;
      input.remove();
      resolve(files);
    };
    input.addEventListener("change", () => finish([...input.files]), { once: true });
    // Not every browser fires `cancel`; without the focus fallback a dismissed picker
    // would leave the caller awaiting a promise that never settles.
    input.addEventListener("cancel", () => finish([]), { once: true });
    window.addEventListener("focus", () => setTimeout(() => finish([]), 400), { once: true });
    input.click();
  });
}

/** Choose files and stage them on the server.
 *
 *  Returns `{ paths: [...] }` to merge into the route's body, `{}` when this browser is
 *  on the server's own machine and the native dialog should be used exactly as before,
 *  or `null` when the user cancelled.
 */
async function chooseAndStage(opts = {}) {
  if (isLocalBrowser()) return {};
  const files = await chooseFiles(opts);
  if (!files.length) return null;
  setStatus(`Uploading ${files.length} file(s)…`);
  try {
    const r = await postFiles("/api/uploads", files);
    return { paths: r.paths };
  } finally {
    setStatus("");
  }
}

/** A route that could not open a Save dialog stages its output instead; collect it. */
function takeDownload(r) {
  if (r && r.download) {
    window.open(r.download, "_blank");
    return true;
  }
  return false;
}

/** Folder selection has no browser equivalent — there is no way for a page to hand a
 *  directory to the server. Say so rather than opening a picker on someone else's
 *  desktop (which the server also refuses). */
function folderPickingUnavailable() {
  if (isLocalBrowser()) return false;
  toast("Choosing a folder only works on the computer running the app. Browse to it "
        + "there, or attach the files individually.", 8000);
  return true;
}

// Accept lists, kept here so a route and its picker cannot drift apart.
const ACCEPT_DOCS = ".txt,.md,.markdown,.pdf,.docx,.epub,.csv,.json,.log,.rst";
const ACCEPT_IMAGES = "image/*";
const ACCEPT_MEDIA = "audio/*,video/*";
const ACCEPT_JSON = ".json,application/json";
const ACCEPT_XML = ".xml,text/xml,application/xml";

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
  // After the chat is loaded, so the summary reflects real values. Absent key ⇒ collapsed.
  setThreadSettingsCollapsed(S.config.chat_settings_collapsed !== false, false);
  maybeAutostartAvatar();
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
    const r = await api("/api/chats/export",
                        { method: "POST", body: { scope: "all", download: !isLocalBrowser() } });
    if (r.cancelled) return;
    if (takeDownload(r)) { toast(`Exported ${r.count} chat(s)`); return; }
    reportExport(r, `Exported ${r.count} chat(s)`);
  } catch (e) { toast("Export failed: " + e.message); }
}
async function exportChat(id) {
  try {
    const r = await api(`/api/chats/${id}/export`,
                        { method: "POST", body: { download: !isLocalBrowser() } });
    if (r.cancelled) return;
    if (takeDownload(r)) { toast("Chat exported"); return; }
    reportExport(r, "Chat exported");
  } catch (e) { toast("Export failed: " + e.message); }
}
async function importChats() {
  try {
    const staged = await chooseAndStage({ accept: ACCEPT_JSON, multiple: false });
    if (staged === null) return;
    const r = await api("/api/chats/import", { method: "POST", body: staged });
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
  // An open in-place edit indexes into the OUTGOING chat's message array. Switching
  // chats invalidates that index, so drop the edit rather than let a save land on
  // whichever message happens to sit at that position in the new chat.
  S.editingMessage = null;
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
  $("btn-voice-reason").classList.toggle("active", !!chat.voice_read_reasoning);
  $("chk-websearch").checked = !!chat.web_search;
  $("chk-strict").checked = !!chat.library_strict;
  $("crawl-pages").value = chat.crawl_pages || S.minCrawledPages || 7;
  applyPersonaSelection(chat);
  $("chk-rag").checked = !!chat.rag_enabled;
  $("chk-rag-auto").checked = !!chat.rag_auto;
  $("rag-threshold").value = chat.rag_threshold || 400;
  // Chats saved before the scope control existed read as "attachments", which is what
  // RAG has always done.
  $("rag-scope").value = chat.rag_scope || "attachments";
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
  renderThreadSettingsSummary();   // the chips follow whichever chat is now loaded
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
  S.editingMessage = null;   // the message it pointed at is about to stop existing
  S.chat.messages = [];
  renderMessages();
  await persistChat(true);
  // The next send would prune the indexed thread anyway, but a chat that is cleared and
  // never used again would keep its chunks forever. Fire-and-forget: nothing here should
  // hold up the clear.
  api(`/api/chats/${S.chat.id}/rag/forget`, { method: "DELETE" }).catch(() => {});
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
  // An open in-place edit lives ONLY in the DOM, so a rebuild would silently discard
  // whatever the user has typed. Everything that re-renders while an edit is open is a
  // background event (a queued item finishing, a parallel run ending), and those only
  // ever append — so defer the rebuild instead of destroying the edit. Save and Cancel
  // both end in another renderMessages(), which replays it.
  if (S.editingMessage != null) { S.messagesDirty = true; return; }
  S.messagesDirty = false;
  const box = $("messages");
  box.innerHTML = "";
  const hide = S.chat && S.chat.hide_thinking;
  (S.chat ? S.chat.messages : []).forEach((m, i) => {
    if (m.role !== "user" && m.role !== "assistant") return;
    // Saved reasoning renders as a collapsed bubble before the answer.
    if (m.role === "assistant" && m.reasoning && !hide) {
      box.appendChild(makeReasoningBubble(m.reasoning, true).el);
    }
    // The chunks RAG retrieved for this answer, under the reasoning. Not gated on
    // `hide_thinking`: this is evidence for the answer, not the model's private
    // deliberation, and it's the only way to check where the answer came from.
    if (m.role === "assistant" && (m.sources || []).length) {
      box.appendChild(makeSourcesBubble(m.sources, true).el);
    }
    const isLastAssistant = m.role === "assistant" && i === lastAssistantIndex();
    box.appendChild(makeBubble(m.role, m.content, {
      regen: isLastAssistant, label: m.pass_label, intermediate: m.intermediate,
      images: m.images, index: i,
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
// How each retrieved-chunk kind reads in the panel. Keys match logic._SOURCE_KINDS.
const SOURCE_KIND_LABEL = {
  library: "LIBRARY", attachment: "ATTACHED", thread: "TURN",
  persona: "PERSONA", inline: "IN-MEMORY",
};
/** Build the collapsible "what RAG picked" bubble shown under the reasoning. Each row
 *  names the document the excerpt came from and, where that document still exists in
 *  this browser's state, links to the exact passage inside it. */
function makeSourcesBubble(sources, collapsed) {
  const list = sources || [];
  const el = document.createElement("div");
  el.className = "sources-bubble" + (collapsed ? " collapsed" : "");
  const head = document.createElement("div");
  head.className = "sources-head";
  const caret = document.createElement("span");
  caret.className = "sources-caret"; caret.textContent = "▸";
  head.appendChild(caret);
  const title = document.createElement("span");
  title.className = "sources-title";
  head.appendChild(title);
  head.onclick = () => el.classList.toggle("collapsed");
  const body = document.createElement("div");
  body.className = "sources-body";
  el.appendChild(head); el.appendChild(body);
  const bubble = { el, body, sources: [] };
  bubble.set = (next) => {
    bubble.sources = next || [];
    title.textContent = ` 📚 Sources (${bubble.sources.length})`;
    body.innerHTML = "";
    bubble.sources.forEach((src, i) => body.appendChild(makeSourceRow(src, i + 1)));
  };
  bubble.set(list);
  return bubble;
}
/** One row of the Sources panel: [n] label · kind · score, then the excerpt itself. */
function makeSourceRow(src, n) {
  const row = document.createElement("div");
  row.className = "source-row";
  const line = document.createElement("div");
  line.className = "source-line";

  const target = sourceTarget(src);
  const name = document.createElement(target ? "button" : "span");
  name.className = "source-link" + (target ? "" : " dead");
  name.textContent = `[${n}] ${src.label || "excerpt"}`;
  if (target) {
    name.type = "button";
    name.title = target.kind === "library"
      ? `Open in Resources → ${src.library_name || "library"}`
      : (target.kind === "message" ? "Jump to that turn" : "View the source text");
    name.onclick = () => openSource(src);
  } else {
    name.title = src.kind === "inline"
      ? "Private chat — retrieved in memory, nothing was indexed to link to"
      : "The source document is no longer available in this chat";
  }
  line.appendChild(name);

  const kind = document.createElement("span");
  kind.className = "source-kind " + (src.kind || "inline");
  kind.textContent = SOURCE_KIND_LABEL[src.kind] || String(src.kind || "").toUpperCase();
  line.appendChild(kind);
  if (src.kind === "library" && src.library_name) {
    const lib = document.createElement("span");
    lib.className = "source-lib"; lib.textContent = src.library_name;
    line.appendChild(lib);
  }
  if (typeof src.score === "number") {
    const score = document.createElement("span");
    score.className = "source-score"; score.textContent = src.score.toFixed(3);
    score.title = "Retrieval score";
    line.appendChild(score);
  }
  row.appendChild(line);

  const ex = document.createElement("div");
  ex.className = "source-excerpt clamped";
  ex.textContent = src.content || "";
  ex.title = "Click to expand";
  ex.onclick = () => ex.classList.toggle("clamped");
  row.appendChild(ex);
  return row;
}

// --------------------------- source navigation ---------------------------
// Nothing in the vector store records WHERE a chunk sits in its document — the chunker's
// offsets are discarded at index time (see app/rag.py chunk_text_semantic). So a source
// link resolves its passage the other way round: find the chunk text inside the document
// the browser already holds. That works retroactively for everything already compiled.

/** Resolve a source row to something on screen, or null if it can't be reached. */
function sourceTarget(src) {
  if (!src) return null;
  if (src.kind === "library") {
    const lib = S.libraries.find((l) => l.id === src.library_id);
    const item = lib && (lib.items || []).find((it) => it.id === src.item_id);
    return item ? { kind: "library", lib, item } : null;
  }
  if (src.kind === "attachment") {
    // A `data:N` id is an inline <Data> block living in turn N of the (non-intermediate)
    // message list, not a chip — it resolves to the message that carries it.
    const m = /^data:(\d+)$/.exec(src.item_id || "");
    if (m) {
      const index = rawMessageIndex(parseInt(m[1], 10));
      return index >= 0 ? { kind: "message", index } : null;
    }
    const att = (S.chat?.attachments || []).find((a) => a.id === src.item_id);
    if (att) return { kind: "attachment", item: att, pinned: true };
    const staged = S.dataItems.find((d) => d.id === src.item_id);
    return staged ? { kind: "attachment", item: staged, pinned: false } : null;
  }
  if (src.kind === "thread") {
    const i = src.message_index;
    const msgs = (S.chat && S.chat.messages) || [];
    return (Number.isInteger(i) && i >= 0 && i < msgs.length)
      ? { kind: "message", index: i } : null;
  }
  return null;
}
/** Map an index over the non-intermediate messages (what logic.attachment_items counts)
 *  back to an index over the raw S.chat.messages array. -1 when it doesn't exist. */
function rawMessageIndex(filteredIndex) {
  const msgs = (S.chat && S.chat.messages) || [];
  let n = 0;
  for (let i = 0; i < msgs.length; i++) {
    if (msgs[i].intermediate) continue;
    if (n === filteredIndex) return i;
    n++;
  }
  return -1;
}

/** Go to where a retrieved chunk came from and highlight the passage itself. */
async function openSource(src) {
  const target = sourceTarget(src);
  if (!target) { toast("That source is no longer available"); return; }
  if (target.kind === "library") {
    // A debounced edit must land first: switching libraries below would otherwise write
    // the pending text into whichever library the editor moves to.
    await flushLibrarySave();
    switchTab("resources");
    if (!S.activeLibrary || S.activeLibrary.id !== target.lib.id) {
      await selectLibrary(target.lib.id);
      $("lib-list").value = target.lib.id;
    }
    const wrap = $("lib-items").querySelector(`.lib-item[data-item-id="${cssEscape(src.item_id)}"]`);
    if (!wrap) { toast("That item is no longer in the library"); return; }
    wrap.scrollIntoView({ block: "center", behavior: "smooth" });
    flashElement(wrap);
    selectInTextarea(wrap.querySelector("textarea"), src.content);
    return;
  }
  if (target.kind === "attachment") {
    showAttachment(target.item, target.pinned);
    selectInTextarea($("attachment-text"), src.content);
    return;
  }
  // A conversation turn: scroll its bubble into view and mark the passage inside it.
  switchTab("chat");
  const bubble = $("messages").querySelector(`.bubble[data-msg-index="${target.index}"]`);
  if (!bubble) { toast("That turn is no longer on screen"); return; }
  bubble.scrollIntoView({ block: "center", behavior: "smooth" });
  flashElement(bubble);
  markPassage(bubble._body || bubble.querySelector(".body"), src.content);
}

/** Select `chunkText` inside a textarea and scroll it into view. */
function selectInTextarea(ta, chunkText) {
  if (!ta) return;
  const hit = locatePassage(ta.value, chunkText);
  if (!hit) { toast("Couldn't pinpoint that passage — it may have been edited since"); return; }
  ta.focus();
  ta.setSelectionRange(hit.start, hit.end);
  scrollTextareaTo(ta, hit.start);
}

/** Wrap `chunkText` in a <mark> inside a plain-text element. Purely visual and undone by
 *  the next renderMessages(), which rebuilds bubbles from state. */
function markPassage(el, chunkText) {
  if (!el) return;
  const text = el.textContent || "";
  const hit = locatePassage(text, chunkText);
  if (!hit) return;
  const mark = document.createElement("mark");
  mark.className = "source-mark";
  mark.textContent = text.slice(hit.start, hit.end);
  el.textContent = "";
  el.appendChild(document.createTextNode(text.slice(0, hit.start)));
  el.appendChild(mark);
  el.appendChild(document.createTextNode(text.slice(hit.end)));
  mark.scrollIntoView({ block: "center", behavior: "smooth" });
}

function flashElement(el) {
  if (!el) return;
  el.classList.remove("source-flash");
  void el.offsetWidth;            // restart the animation if it's already running
  el.classList.add("source-flash");
  setTimeout(() => el.classList.remove("source-flash"), 1600);
}

function cssEscape(s) {
  return (window.CSS && CSS.escape) ? CSS.escape(String(s)) : String(s).replace(/["\\]/g, "\\$&");
}

// Whitespace-collapsed views of recently searched documents. Small and bounded: a pinned
// YouTube transcript can be 200k characters and a user clicks several of its chunks.
const _normDocCache = new Map();
const _NORM_CACHE_MAX = 4;
const WHITESPACE = /\s/;

/** Collapse runs of whitespace to a single space, keeping an index back to the original
 *  string so a match in the normalized text maps to a real character range. */
function normalizedDoc(text) {
  // Length plus both ends: cheap, and enough to notice the library textarea being
  // edited between two clicks — a stale map would point into the text as it was.
  const key = text.length + "\u0000" + text.slice(0, 64) + text.slice(-32);
  const hit = _normDocCache.get(key);
  if (hit) return hit;
  const map = new Int32Array(text.length);
  const chars = [];
  let n = 0, pendingSpace = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (WHITESPACE.test(c)) {
      pendingSpace = n > 0;      // never lead with a space
      continue;
    }
    if (pendingSpace) { chars.push(" "); map[n++] = i; pendingSpace = false; }
    chars.push(c); map[n++] = i;
  }
  const out = { norm: chars.join(""), map: map.subarray(0, n) };
  _normDocCache.set(key, out);
  while (_normDocCache.size > _NORM_CACHE_MAX) {
    _normDocCache.delete(_normDocCache.keys().next().value);
  }
  return out;
}

/**
 * Find a retrieved chunk inside its source document.
 * @returns {{start:number,end:number}|null} a range in `docText`, or null if not found.
 *
 * Chunks from the semantic chunker are verbatim substrings, so the exact search almost
 * always wins. The fallback word-window chunker (and the private-chat in-memory path)
 * rejoin words with single spaces, which is why the whitespace-collapsed search exists.
 * A head-only match is the last resort: it still lands the user on the right paragraph.
 */
function locatePassage(docText, chunkText) {
  const doc = docText || "";
  const chunk = (chunkText || "").trim();
  if (!doc || !chunk) return null;
  const exact = doc.indexOf(chunk);
  if (exact >= 0) return { start: exact, end: exact + chunk.length };

  const { norm, map } = normalizedDoc(doc);
  const needle = chunk.replace(/\s+/g, " ").trim();
  let at = norm.indexOf(needle), len = needle.length;
  if (at < 0) {
    const head = needle.slice(0, 40);
    if (head.length < 12) return null;
    at = norm.indexOf(head); len = head.length;
    if (at < 0) return null;
  }
  return { start: map[at], end: map[at + len - 1] + 1 };
}

/** Scroll a textarea so that character `index` sits mid-view. setSelectionRange alone
 *  doesn't reliably scroll, so measure a hidden mirror laid out the same way. */
function scrollTextareaTo(ta, index) {
  try {
    const cs = getComputedStyle(ta);
    const mirror = document.createElement("div");
    const s = mirror.style;
    ["fontFamily", "fontSize", "fontWeight", "fontStyle", "letterSpacing", "lineHeight",
     "textTransform", "wordSpacing", "textIndent", "paddingTop", "paddingBottom",
     "paddingLeft", "paddingRight"].forEach((p) => { s[p] = cs[p]; });
    s.position = "absolute"; s.top = "0"; s.left = "-9999px";
    s.visibility = "hidden"; s.whiteSpace = "pre-wrap"; s.overflowWrap = "break-word";
    // clientWidth is content + padding for either box-sizing, so pin border-box.
    s.boxSizing = "border-box"; s.width = ta.clientWidth + "px";
    mirror.textContent = ta.value.slice(0, index) + "\u200b";
    document.body.appendChild(mirror);
    const y = mirror.scrollHeight;
    document.body.removeChild(mirror);
    ta.scrollTop = Math.max(0, y - ta.clientHeight / 2);
  } catch (e) { /* best effort — the selection is set either way */ }
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
 *                           the returned element exposes `_body` for that purpose.
 *                           `index` is the message's position in S.chat.messages and
 *                           is what makes the bubble editable — see below.
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
  // Only a bubble that KNOWS its index in S.chat.messages can be edited — renderMessages
  // skips non-user/assistant roles, so DOM position is not the array index. Streaming and
  // parallel-lane bubbles are deliberately passed none: they are views of an in-flight run,
  // not of stored state, and the re-render when the run ends gives them the button.
  if (opts.index != null && !opts.streaming) {
    // Also what a Sources link scrolls to when a chunk came from an earlier turn.
    b.dataset.msgIndex = String(opts.index);
    const edit = document.createElement("button");
    edit.className = "small btn-edit"; edit.textContent = "Edit";
    edit.title = "Edit this message — nothing regenerates";
    edit.onclick = () => beginEditMessage(b, opts.index);
    actions.appendChild(edit);
  }
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

// --------------------------- editing a message ---------------------------
// Editing an assistant answer is the cheapest steering tool the app has: the client
// owns the message array and re-posts it whole on every send, so a corrected answer
// simply becomes what the model sees next turn. No server route is involved.

/** Swap a bubble's body for a textarea, in place. Deliberately leaves the reasoning
 *  bubble (a separate element rendered before this one) and the image strip alone —
 *  an edit is about the text. */
function beginEditMessage(bubble, index) {
  if (S.generating || S.batchRunning) { toast("Wait for the answer to finish"); return; }
  if (S.editingMessage != null) { toast("Finish the open edit first"); return; }
  const msg = S.chat && S.chat.messages[index];
  if (!msg) return;
  S.editingMessage = index;
  bubble.classList.add("editing");

  const ta = document.createElement("textarea");
  ta.className = "bubble-edit";
  ta.value = msg.content || "";
  const row = document.createElement("div");
  row.className = "bubble-edit-actions";
  const save = document.createElement("button");
  save.className = "small primary"; save.textContent = "Save";
  save.onclick = () => saveEditMessage(index, ta.value);
  const cancel = document.createElement("button");
  cancel.className = "small ghost"; cancel.textContent = "Cancel";
  cancel.onclick = cancelEditMessage;
  const hint = document.createElement("span");
  hint.className = "muted"; hint.textContent = "Ctrl+Enter saves · Esc cancels";
  row.appendChild(save); row.appendChild(cancel); row.appendChild(hint);

  // Open the textarea at the height the text already occupied, so saving a one-line
  // fix to a long answer doesn't collapse the thread under the cursor.
  const wanted = bubble._body.scrollHeight;
  bubble._body.classList.add("hidden");
  bubble.insertBefore(ta, bubble._images);
  bubble.insertBefore(row, bubble._images);
  ta.style.height = Math.max(90, wanted + 12) + "px";
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  ta.onkeydown = (e) => {
    if (e.key === "Escape") { e.preventDefault(); cancelEditMessage(); }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveEditMessage(index, ta.value); }
  };
}

/** Replace the text and nothing else. Later messages are untouched and nothing
 *  regenerates — the edit simply becomes what the model sees on the next send. */
function saveEditMessage(index, text) {
  const msg = S.chat && S.chat.messages[index];
  S.editingMessage = null;
  if (!msg) { renderMessages(); return; }
  msg.content = text;          // reasoning / images / pass_label left exactly as they were
  persistChat(true);
  renderMessages();
  warnIfEditIgnoredByIsolation(index);
  toast("Message updated");
}

function cancelEditMessage() { S.editingMessage = null; renderMessages(); }

/** Isolation sends ONLY the last user turn (app/logic.py build_messages), so an edit to
 *  an assistant message — or to any but the final user message — will never reach the
 *  model while this chat is isolated. Say so rather than let the edit look broken. */
function warnIfEditIgnoredByIsolation(index) {
  if (!S.chat || !S.chat.isolated) return;
  let lastUser = -1;
  S.chat.messages.forEach((m, i) => { if (m.role === "user") lastUser = i; });
  if (index !== lastUser) {
    toast("This chat is isolated — only the last message is sent, so this edit won't reach the model.", 7000);
  }
}

function scrollBottom() { const box = $("messages"); box.scrollTop = box.scrollHeight; }
// True when the view is already pinned near the bottom. Used to decide whether a
// streaming update should auto-follow — if the user has scrolled up, we leave the
// scroll position alone so they can read earlier content without being yanked down.
function isNearBottom(box) {
  // Looser on a touchscreen: momentum scrolling overshoots the bottom routinely, and
  // a soft keyboard opening changes clientHeight under us. At 80px a phone drops out
  // of follow-the-stream constantly.
  const slack = (typeof isTouch === "function" && isTouch()) ? 160 : 80;
  return box.scrollHeight - box.scrollTop - box.clientHeight < slack;
}

// Drag handle under the chat thread: pins #messages to an explicit height while
// dragging (double-click clears it to restore the default flex-fill). Pointer
// capture keeps the drag tracking even when the cursor moves over the controls below.
function setupMessagesResizer() {
  const handle = $("messages-resizer"), box = $("messages");
  if (!handle || !box) return;
  let dragging = false, startY = 0, startH = 0;
  // Without this the browser claims a touch drag for scrolling and pointermove never
  // fires. The handle is hidden below the breakpoint anyway, but a touch laptop has it.
  handle.style.touchAction = "none";
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

// ------------------------------- Small screens -----------------------------
// Everything that a stylesheet cannot do on its own: moving the nav into a drawer,
// keeping the composer usable with a soft keyboard, and not sending a message every
// time someone reaches for a newline.
//
// The breakpoint is duplicated from mobile.css by necessity — CSS custom properties
// cannot be read by a media query and matchMedia cannot read a stylesheet — so the two
// have to be changed together.
const MOBILE_MQ = matchMedia("(max-width: 820px)");
const COARSE_MQ = matchMedia("(pointer: coarse)");
const isTouch = () => COARSE_MQ.matches;

/** Move the tab strip and the profile bar into the drawer below the breakpoint, and put
 *  them back above it.
 *
 *  Moved rather than duplicated: two copies would mean two elements answering to
 *  `data-tab` and two `#data-profile-select`s, and every handler in this file binds to a
 *  single element by id. Re-parenting keeps those bindings intact — listeners belong to
 *  the element, not to its position in the tree. */
function syncNav() {
  const drawer = $("nav-drawer");
  const tabs = document.querySelector("nav.tabs");
  const profiles = $("profile-bar");
  if (!drawer || !tabs || !profiles) return;
  if (MOBILE_MQ.matches) {
    if (tabs.parentElement !== drawer) drawer.appendChild(tabs);
    if (profiles.parentElement !== drawer) drawer.appendChild(profiles);
  } else {
    closeDrawers();
    if (tabs.parentElement !== $("app-header")) $("app-header").appendChild(tabs);
    if (profiles.parentElement !== document.body) {
      document.body.insertBefore(profiles, $("app-header").nextSibling);
    }
    // The resizer writes an explicit pixel height onto #messages, and it is hidden below
    // the breakpoint — coming back to a wide window with a stale height would leave the
    // thread the wrong size with no visible handle to have caused it.
    const box = $("messages");
    if (box) { box.style.flex = ""; box.style.height = ""; }
  }
}

/** Open or close one of the two slide-over panels (the nav drawer, the chat list).
 *  Both share the scrim, and only one can be open at a time. */
function setDrawer(panel, toggle, open) {
  const scrim = $("nav-scrim");
  if (open) {
    panel.hidden = false;
    // Two frames: [hidden] must be gone and the browser must have laid the panel out
    // before .open flips the transform, or there is nothing to transition from.
    requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.add("open")));
    scrim.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
    const first = panel.querySelector("button, select, a, input");
    if (first) first.focus();
  } else {
    panel.classList.remove("open");
    toggle.setAttribute("aria-expanded", "false");
    scrim.hidden = true;
    // #sidebar is a real element on desktop, so it must never be left hidden; only the
    // drawer, which is empty above the breakpoint, gets its attribute back.
    if (panel.id === "nav-drawer") setTimeout(() => { if (!panel.classList.contains("open")) panel.hidden = true; }, 200);
  }
}

function closeDrawers() {
  const drawer = $("nav-drawer"), side = $("sidebar");
  if (drawer && (drawer.classList.contains("open") || !drawer.hidden)) {
    setDrawer(drawer, $("btn-nav"), false);
  }
  if (side && side.classList.contains("open")) setDrawer(side, $("btn-chats"), false);
  $("nav-scrim").hidden = true;
}

function toggleDrawer(panel, toggle) {
  setDrawer(panel, toggle, !panel.classList.contains("open"));
}

/** Grow the composer with its content instead of leaving it stuck at three rows.
 *  `resize: vertical` is a mouse-only grabber and does nothing on a touchscreen. */
function autoGrowComposer() {
  const ta = $("input-box");
  if (!ta) return;
  if (!MOBILE_MQ.matches) { ta.style.height = ""; return; }
  ta.style.height = "auto";
  // Capped in CSS via max-height; this keeps the element itself in step with it.
  ta.style.height = Math.min(ta.scrollHeight, Math.round(innerHeight * 0.4)) + "px";
}

function setupMobile() {
  const drawer = $("nav-drawer"), side = $("sidebar");
  $("btn-nav").onclick = () => toggleDrawer(drawer, $("btn-nav"));
  $("btn-chats").onclick = () => toggleDrawer(side, $("btn-chats"));
  $("nav-scrim").onclick = closeDrawers;
  // Picking a destination should not leave the drawer sitting over it.
  drawer.addEventListener("click", (e) => { if (e.target.closest(".tab")) closeDrawers(); });
  side.addEventListener("click", (e) => {
    if (e.target.closest(".chat-row, .chat-tab, #btn-new-chat, #btn-new-private")) closeDrawers();
  });

  syncNav();
  MOBILE_MQ.addEventListener("change", () => { syncNav(); autoGrowComposer(); });

  // On a phone the Return key is how you start a new line. Enter-to-send is a keyboard
  // convenience and there is a Send button two centimetres away; without this guard the
  // composer cannot produce a newline at all.
  const ta = $("input-box");
  ta.addEventListener("input", autoGrowComposer);
  if (isTouch()) {
    ta.placeholder = "Type a message…";
  }

  // The soft keyboard shrinks the visual viewport rather than resizing the window, so
  // nothing in the layout reacts to it. If the thread was pinned to the bottom, keep it
  // there once the keyboard has settled.
  if (window.visualViewport) {
    visualViewport.addEventListener("resize", () => {
      const box = $("messages");
      if (box && isNearBottom(box)) setTimeout(scrollBottom, 50);
      autoGrowComposer();
    });
  }
}

// --------------------------- thread settings -------------------------
// The system prompt, pre-prompt and options row collapse into a one-line summary so the
// thread gets the vertical space. The state is a global preference, not per-chat.

/** Show or hide the settings region. `persist` saves the choice for every chat. */
function setThreadSettingsCollapsed(collapsed, persist) {
  const panel = $("thread-settings"), bar = $("thread-settings-bar");
  if (!panel || !bar) return;
  panel.classList.toggle("collapsed", collapsed);
  bar.classList.toggle("open", !collapsed);
  $("btn-thread-settings").setAttribute("aria-expanded", String(!collapsed));
  renderThreadSettingsSummary();
  // A dragged #messages is pinned to an explicit height (setupMessagesResizer), so without
  // this the space the collapse frees would never reach the thread. Same reset as dblclick.
  const box = $("messages");
  if (box) { box.style.flex = ""; box.style.height = ""; }
  if (!persist) return;
  S.config.chat_settings_collapsed = collapsed;
  // Fire-and-forget: /api/settings patches only the keys it is sent.
  api("/api/settings", { method: "POST", body: { chat_settings_collapsed: collapsed } })
    .then((r) => { S.config = { ...S.config, ...r.config }; })
    .catch(() => {});
}

/** Rebuild the collapsed summary: the context size, plus a chip per option that is on. */
function renderThreadSettingsSummary() {
  const el = $("thread-settings-summary");
  if (!el) return;
  const chips = [];
  if ($("system-on").checked && $("system-prompt").value.trim()) chips.push("System ✓");
  if ($("pre-on").checked && $("pre-prompt").value.trim()) chips.push("Pre ✓");
  const ctx = $("ctx-select").selectedOptions[0];
  if (ctx) chips.push(ctx.textContent.trim());
  const flags = [
    ["chk-isolate", "Isolated"], ["chk-hide-thinking", "Hide thinking"],
    ["chk-multipass", "🔁 Multi-Pass"], ["chk-websearch", "🌐 Web"],
    ["chk-rag", "📚 RAG"], ["chk-parallel", "⚡ Parallel"],
    ["chk-persona", "🎭 Persona"], ["chk-memory", "🧠 Memory"],
  ];
  for (const [id, label] of flags) if ($(id).checked) chips.push(label);
  el.textContent = "";
  for (const c of chips) {
    const s = document.createElement("span");
    s.className = "ts-chip"; s.textContent = c;
    el.appendChild(s);
  }
}

/** One delegated listener keeps the chips fresh without touching the ~20 existing
 *  per-control handlers inside the region. */
function wireThreadSettings() {
  const panel = $("thread-settings");
  if (!panel) return;
  panel.addEventListener("change", renderThreadSettingsSummary);
  panel.addEventListener("input", renderThreadSettingsSummary);
  $("btn-thread-settings").onclick = () =>
    setThreadSettingsCollapsed(!panel.classList.contains("collapsed"), true);
  $("thread-settings-summary").onclick = () => setThreadSettingsCollapsed(false, true);
}

// ------------------------------- generation --------------------------
/** Flip the UI between idle and generating: swap Send for Stop and disable the
 *  controls that must not change mid-run. */
function setGeneratingUI(on) {
  S.generating = on;
  // Greys out the per-message Edit buttons (Copy stays live). beginEditMessage guards
  // this too, because S.batchRunning is set outside this function.
  $("messages").classList.toggle("generating", on);
  $("btn-batch").classList.toggle("hidden", on);
  refreshStopVisibility();
}

function refreshStopVisibility() {
  const stopOn = !!S.generating || (S.voice.on && S.voice.speaking);
  $("btn-send").classList.toggle("hidden", stopOn);
  $("btn-stop").classList.toggle("hidden", !stopOn);
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
  $("add-rss-panel").classList.toggle("hidden", which !== "rss");
  $("add-search-panel").classList.toggle("hidden", which !== "search");
  if (which === "url") { $("add-url-input").value = ""; $("add-url-input").focus(); }
  if (which === "rss") {
    $("add-rss-input").value = "";
    $("add-rss-limit").value = String(S.config.rss_max_episodes ?? 25);
    resetRssFilters("add-rss");
    const p = $("add-rss-progress"); p.classList.add("hidden"); p.textContent = "";
    // The Whisper box is only offerable if faster-whisper is actually importable.
    refreshWhisperStatus();
    $("add-rss-input").focus();
  }
  if (which === "youtube") {
    $("add-yt-input").value = "";
    $("add-yt-limit").value = "0";
    $("add-yt-kind-video").checked = true;
    const p = $("add-yt-progress"); p.classList.add("hidden"); p.textContent = "";
    updateYouTubePanelKind();
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
  $("add-rss-panel").classList.add("hidden");
  $("add-search-panel").classList.add("hidden");
}

/** Reveal the playlist affordances the URL actually calls for: the ambiguity radio only
 *  when both a video id and a list id parse, Max-videos only when a playlist is what
 *  will be fetched. Also relabels the button, so it says what it is about to do. */
function updateYouTubePanelKind() {
  const kind = ytKindOf($("add-yt-input").value);
  $("add-yt-choice").classList.toggle("hidden", kind !== "both");
  const asPlaylist = composerYouTubeIsPlaylist(kind);
  $("add-yt-limit-wrap").classList.toggle("hidden", !asPlaylist);
  $("btn-add-yt-fetch").textContent = asPlaylist ? "Fetch playlist" : "Fetch";
}
function composerYouTubeIsPlaylist(kind) {
  return kind === "playlist" || (kind === "both" && $("add-yt-kind-playlist").checked);
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
  const stagedImgs = await chooseAndStage({ accept: ACCEPT_IMAGES, capture: "environment" });
  if (stagedImgs === null) return;
  setStatus(isLocalBrowser() ? "Waiting for image selection…" : "Reading images…");
  try {
    await streamSSE("/api/images/pick", stagedImgs, {
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
      method: "POST",
      body: { default_name: defaultName || "image", download: !isLocalBrowser() },
    });
    if (takeDownload(r)) toast("Image downloaded");
    else if (r.ok) toast("Saved to " + r.path);
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
  const stagedDocs = await chooseAndStage({ accept: ACCEPT_DOCS });
  if (stagedDocs === null) return;
  setStatus(isLocalBrowser() ? "Waiting for file selection…" : "Reading files…");
  try {
    await streamSSE("/api/extract-files", stagedDocs, {
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
let composerYtRunId = null;

/** Dispatch a composer YouTube fetch. A `list=` id means the playlist route, whose
 *  result is one staged chip per video rather than one blob. */
function composerAddYouTube() {
  const url = $("add-yt-input").value.trim();
  if (!url) { toast("Enter a YouTube URL"); return; }
  const opts = {
    url,
    comments: $("add-yt-comments").checked,
    max: Math.max(5, Math.min(2000, parseInt($("add-yt-max").value, 10) || 100)),
    refresh: $("add-yt-refresh").checked,
  };
  return composerYouTubeIsPlaylist(ytKindOf(url))
    ? composerAddYouTubePlaylist(opts)
    : composerAddYouTubeVideo(opts);
}

function composerAddYouTubeVideo(opts) {
  const prog = $("add-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Starting…";
  const btn = $("btn-add-yt-fetch"); btn.disabled = true;
  cancelComposerYouTube(true);
  composerYtES = sourceStream("/api/youtube/fetch", opts, {
    started: (id) => { composerYtRunId = id; },
    progress: (d) => { prog.textContent = ytProgressText(d); },
    complete: (d) => {
      stageSource("youtube", d.title, d.text, d.url);
      toast(`Attached: ${d.title}${d.via ? " (via " + d.via + ")" : ""}` +
            (d.comment_count ? ` — ${d.comment_count} comment(s)` : ""));
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 6000);
      hideComposerPanels();
    },
    failed: (msg) => { toast("YouTube fetch failed: " + msg); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; composerYtES = null; composerYtRunId = null; },
  });
}

/** Stream a whole playlist into the composer, staging ONE chip per video as each
 *  arrives — so a 40-video playlist is watchable and interruptible rather than a long
 *  spinner ending in a single undivisible blob. */
function composerAddYouTubePlaylist(opts) {
  opts.limit = Math.max(0, parseInt($("add-yt-limit").value, 10) || 0);
  const prog = $("add-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the playlist…";
  const btn = $("btn-add-yt-fetch"); btn.disabled = true;
  cancelComposerYouTube(true);
  let staged = 0, total = 0, chars = 0;
  composerYtES = sourceStream("/api/youtube/fetch-playlist", opts, {
    started: (id) => { composerYtRunId = id; },
    playlist: (d) => {
      total = d.total || 0;
      prog.textContent = `Playlist: ${total} video(s) — fetching…`;
      // Enumerating is one cheap request; fetching them is not. Say so before the user
      // walks away, and point at the exit.
      if (total > 25) toast(`${total} videos queued — press Cancel to stop early.`, 6000);
    },
    progress: (d) => { prog.textContent = ytProgressText(d); },
    video: (d) => {
      stageSource("youtube", d.title, d.text, d.url);
      staged++; chars += (d.text || "").length;
      prog.textContent = `Fetched ${staged}/${total} — ${d.title}`;
    },
    video_error: (d) => { toast(`Skipped ${d.title}: ${d.message}`, 5000); },
    complete: (d) => {
      prog.textContent = `Attached ${staged} of ${d.total} video(s).`;
      // The character count matters: every chip lands in the next send's data block.
      toast(`Attached ${staged} video(s)` + (d.failed ? `, ${d.failed} skipped` : "") +
            ` — ~${Math.round(chars / 1000)}k characters`, 6000);
      if (staged) hideComposerPanels();
    },
    failed: (msg) => { toast("Playlist fetch failed: " + msg); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; composerYtES = null; composerYtRunId = null; },
  });
}

/** Abort an in-flight composer fetch. Cancel used to only hide the panel, which was
 *  survivable for one video and is not for a playlist: the worker kept fetching and kept
 *  pushing chips into a chat the user had walked away from. `quiet` reuses this to tear
 *  down a previous stream before starting a new one. */
function cancelComposerYouTube(quiet) {
  const runId = composerYtRunId;
  if (composerYtES) { composerYtES.close(); composerYtES = null; }
  composerYtRunId = null;
  if (runId) api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
  $("btn-add-yt-fetch").disabled = false;
  if (!quiet) hideComposerPanels();
}

// ---- RSS / podcast (composer) ----

let composerRssES = null;
let composerRssRunId = null;

/** Fetch a feed and stage one chip per episode, as each lands. */
function composerAddRss() {
  const url = $("add-rss-input").value.trim();
  if (!url) { toast("Enter a feed URL"); return; }
  const opts = {
    url,
    limit: Math.max(0, Math.min(500, parseInt($("add-rss-limit").value, 10) || 0)),
    notes: $("add-rss-notes").checked,
    whisper: $("add-rss-whisper").checked,
    refresh: $("add-rss-refresh").checked,
    ...rssFilterOpts("add-rss"),
  };
  const prog = $("add-rss-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the feed…";
  const btn = $("btn-add-rss-fetch"); btn.disabled = true;
  // Tear down any previous stream first — otherwise a re-click leaves the old worker
  // running and it keeps pushing chips into the chat. Same rule as the YouTube panel.
  cancelComposerRss(true);
  let staged = 0;
  composerRssES = sourceStream("/api/rss/fetch-feed", opts, {
    started: (id) => { composerRssRunId = id; },
    feed: (d) => {
      // Three numbers, three meanings: the whole feed, what the filter accepted, what the
      // limit then took. Only the ones that differ are shown.
      const narrowed = rssMatchText(d);
      prog.textContent = `${d.title || "Feed"} — ${d.total} episode(s)` +
                         (narrowed ? ` · ${narrowed}`
                                   : d.total_available > d.total ? ` of ${d.total_available}` : "");
      if (d.total > 25) toast(`${d.total} episodes — that's a lot of context`, 6000);
    },
    warning: (d) => { toast(d.message, 8000); },
    progress: (d) => { prog.textContent = rssProgressText(d); },
    episode: (d) => {
      stageSource("rss", d.title, d.text, d.url);
      staged++;
      if (d.truncated) {
        toast(`${d.title}: truncated at ${d.text.length.toLocaleString()} of ` +
              `${d.full_chars.toLocaleString()} characters`, 7000);
      }
    },
    episode_error: (d) => { toast(`${d.title}: ${d.message}`, 6000); },
    complete: (d) => {
      toast(`Attached ${staged} episode(s)` + (d.failed ? ` — ${d.failed} failed` : ""));
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 8000);
      hideComposerPanels();
    },
    failed: (msg) => { toast("Feed fetch failed: " + msg, 8000); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; composerRssES = null; composerRssRunId = null; },
  });
}

/** Abort an in-flight feed fetch. `quiet` reuses this to tear down a previous stream
 *  before starting a new one, without also closing the panel. */
function cancelComposerRss(quiet) {
  const runId = composerRssRunId;
  if (composerRssES) { composerRssES.close(); composerRssES = null; }
  composerRssRunId = null;
  // Closing the socket alone leaves the worker downloading — and, with Whisper on,
  // holding the GPU. Ask it to stop outright.
  if (runId) api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
  $("btn-add-rss-fetch").disabled = false;
  if (!quiet) hideComposerPanels();
}

/** Render an RSS progress frame. Separate from ytProgressText rather than one function
 *  with twelve branches: the two vocabularies barely overlap. */
function rssProgressText(d) {
  const head = d.count > 1 && d.index ? `${d.index}/${d.count} — ${d.name || ""}` : "";
  let phase = "Working…";
  if (d.phase === "feed") {
    phase = d.total !== undefined
      ? `${d.total} episode(s)${d.from_cache ? " (feed unchanged)" : ""}`
      : "Reading the feed…";
  } else if (d.phase === "cache") {
    phase = d.need_transcript ? "From cache — fetching the transcript…"
                              : `From cache${d.source ? ` (${d.source})` : ""}`;
  } else if (d.phase === "page") {
    phase = "Fetching the article page…";
  } else if (d.phase === "transcript") {
    phase = d.chars
      ? `Transcript: ${d.chars.toLocaleString()} characters` +
        (d.speakers ? ", with speakers" : "")
      : "No published transcript";
  } else if (d.phase === "download" || d.phase === "whisper") {
    // The slow phases share their formatter with the media-file path.
    phase = mediaProgressText({ ...d, count: 0, name: "" });
  }
  return head ? `${head} · ${phase}` : phase;
}

// ---- local transcription (composer) ----

/** Render a transcription progress frame. Whisper is the slow one, so it gets a real
 *  ETA rather than a spinner: a three-hour episode is minutes of GPU even when it goes
 *  well, and a silent bar reads as a hang. */
function mediaProgressText(d) {
  // `count` is the number of FILES; `total` on a whisper/download frame is seconds or
  // bytes. Conflating them is how the bar ends up reading "3/60 files".
  const head = d.count > 1 && d.index ? `${d.index}/${d.count} — ${d.name || ""}` : (d.name || "");
  let phase = d.message || "Working…";
  if (d.phase === "download") {
    const mb = (n) => (n / 1048576).toFixed(1) + " MB";
    phase = d.total ? `Downloading ${mb(d.done)} of ${mb(d.total)}…` : `Downloading ${mb(d.done)}…`;
  } else if (d.phase === "whisper") {
    if (d.waiting) phase = d.message || "Waiting for the transcriber…";
    else if (d.fallback) phase = d.message || "Falling back to the CPU…";
    else if (d.total) {
      const pct = Math.min(100, Math.round((d.done / d.total) * 100));
      phase = `Transcribing ${fmtClock(d.done)} of ${fmtClock(d.total)} (${pct}%)` +
              (d.device ? ` on ${d.device}` : "");
    } else if (d.message) phase = d.message;
    else phase = "Transcribing…";
  }
  return head ? `${head} · ${phase}` : phase;
}

function fmtClock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(s % 60).padStart(2, "0")}`;
}

let composerMediaES = null;
let composerMediaRunId = null;

/** Pick audio/video files and stage each transcript as a chat attachment.
 *  POST rather than GET, because the native picker runs on the request thread before the
 *  stream opens — so this uses streamSSE, not sourceStream (which is EventSource/GET). */
async function composerAddMediaFiles() {
  const prog = $("add-media-progress");
  const stagedMedia = await chooseAndStage({ accept: ACCEPT_MEDIA });
  if (stagedMedia === null) return;
  prog.classList.remove("hidden");
  prog.textContent = isLocalBrowser() ? "Waiting for file selection…" : "Transcribing…";
  const btn = $("btn-add-media"); btn.disabled = true;
  try {
    await streamSSE("/api/transcribe/files", stagedMedia, {
      start: (d) => { composerMediaRunId = d.run_id || null; },
      begin: (d) => {
        prog.textContent = d.total ? `Transcribing ${d.total} file(s)…` : "Nothing selected.";
      },
      progress: (d) => { prog.textContent = mediaProgressText(d); },
      file: (d) => {
        stageSource("audio", d.name, d.text, d.source);
        prog.textContent = `${d.name}: ${d.chars.toLocaleString()} characters` +
                           (d.fallback ? " (CPU fallback)" : "");
      },
      file_error: (d) => { toast(`${d.name}: ${d.message}`, 6000); },
      complete: (d) => {
        const n = (d.results || []).length;
        if (n) toast(`Attached ${n} transcript(s)`);
        if ((d.errors || []).length) toast("Some files failed: " + d.errors.join("; "), 6000);
        prog.classList.add("hidden");
        hideComposerPanels();
      },
      error: (d) => { toast("Transcription failed: " + d.message, 8000); prog.textContent = "Failed."; },
      done: () => { btn.disabled = false; composerMediaRunId = null; },
    });
  } catch (e) {
    toast("Transcription failed: " + e.message);
    prog.textContent = "Failed.";
    btn.disabled = false;
  }
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
  c.voice_read_reasoning = $("btn-voice-reason").classList.contains("active");
  c.web_search = $("chk-websearch").checked;
  c.library_strict = $("chk-strict").checked;
  c.crawl_pages = parseInt($("crawl-pages").value) || S.minCrawledPages || 7;
  c.rag_enabled = $("chk-rag").checked;
  c.rag_auto = $("chk-rag-auto").checked;
  c.rag_threshold = Math.max(1, parseInt($("rag-threshold").value) || 400);
  c.rag_scope = $("rag-scope").value || "attachments";
  c.multi_pass = $("chk-multipass").checked;
  c.passes = Math.max(1, parseInt($("mp-passes").value) || 2);
  c.pass_use_system = $("chk-mp-system").checked;
  c.eval_prompt = $("mp-eval-prompt").value;
  c.memory_enabled = $("chk-memory").checked;
  c.memory_core_id = $("memory-core-select").value || c.memory_core_id || "";
}

// ------------------------------- voice chat --------------------------
function avatarUrl() {
  return String(S.config.avatar_url || "http://127.0.0.1:8765").replace(/\/+$/, "");
}

async function avatarFetch(path, opts) {
  const method = (opts && opts.method) || "GET";
  let body = opts && opts.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch (e) { body = {}; }
  }
  return api("/api/avatar/rpc", { method: "POST", body: { path, method, body: body || null } });
}

function speakableText(text) {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]+`/g, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\[\d+\]/g, " ")
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b/g, " ")
    .replace(/^\s*\d+[.)]\s+/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/[_#]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function pullSentences(text) {
  const complete = [];
  const re = /[.!?]["')\]]*(?=\s+|$)|\n{2,}/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    const piece = text.slice(last, m.index + m[0].length);
    if (/(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sr|Jr|St|vs)\.\s*$/i.test(piece.trim()) && piece.trim().length < 8) {
      continue;
    }
    const trimmed = piece.trim();
    if (trimmed) complete.push(trimmed);
    last = m.index + m[0].length;
  }
  return { complete, rest: text.slice(last) };
}

function avatarSpeak(text) {
  const clean = speakableText(text);
  if (!clean || S.voice.skipSpeak || !S.voice.on) return;
  S.voice.speakChain = S.voice.speakChain.then(async () => {
    if (S.voice.skipSpeak || !S.voice.on) return;
    await avatarFetch("/speak", { method: "POST", body: { text: clean } });
    S.voice.speaking = true;
    refreshStopVisibility();
  }).catch((e) => { toast("Voice: " + e.message); });
}

function createSentenceSpeaker() {
  // F5 pipelines *inside* one /speak job. Sending one short sentence at a time
  // leaves a gap while the next job generates, and short pieces make F5 emit
  // junk (often a blurt of digits) before the real words.
  const START_SENTENCES = 2;
  const BATCH_SENTENCES = 5;
  let buf = "";
  let pending = [];
  let started = false;
  let dead = false;
  function sendBatch(parts) {
    const joined = parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
    if (joined) avatarSpeak(joined);
  }
  function pump(force) {
    if (dead || S.voice.skipSpeak) { pending = []; buf = ""; return; }
    if (!started) {
      if (!force && pending.length < START_SENTENCES) return;
      started = true;
      sendBatch(pending.splice(0, force ? pending.length : START_SENTENCES));
    }
    if (force) {
      if (pending.length) sendBatch(pending.splice(0, pending.length));
      return;
    }
    while (pending.length >= BATCH_SENTENCES) {
      sendBatch(pending.splice(0, BATCH_SENTENCES));
    }
  }
  return {
    feed(text) {
      if (dead || S.voice.skipSpeak) return;
      buf += text || "";
      const { complete, rest } = pullSentences(buf);
      buf = rest;
      pending.push.apply(pending, complete);
      pump(false);
    },
    endSection() {
      if (buf.trim()) pending.push(buf.trim());
      buf = "";
      pump(true);
      started = false;
    },
    end() {
      if (buf.trim()) pending.push(buf.trim());
      buf = "";
      pump(true);
      dead = true;
    },
  };
}

function setVoiceControlsLocked(locked) {
  [
    "btn-voice-reason", "set-avatar-gate", "set-avatar-silence",
    "avatar-start-app", "avatar-start-voice", "set-avatar-dir", "set-avatar-url",
    "btn-avatar-browse-dir", "btn-avatar-save",
  ].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = !!locked;
  });
}

function setVoiceHelperLoading(on, why) {
  S.voice.helperLoading = !!on;
  setVoiceControlsLocked(on);
  const btn = $("btn-voice");
  if (!btn) return;
  btn.disabled = !!on;
  btn.classList.toggle("loading", !!on);
  if (on) {
    btn.textContent = "⏳ Loading voice helper";
    btn.title = why || "Waiting for TTS and transcription models to load…";
    btn.classList.remove("active", "listening", "speaking");
  } else if (!S.voice.on) {
    btn.textContent = "🎤 Voice";
    btn.title = S.voice.modelsReady
      ? "Start voice chat — helper is ready. Listening begins when you press this."
      : "Voice chat — talk, or hold the Avatar PTT hotkey. Silence sends.";
  } else {
    btn.textContent = "🎤 Voice";
  }
}

function updateVoiceButton(listen, info) {
  const btn = $("btn-voice");
  if (!btn) return;
  if (S.voice.helperLoading) return;
  btn.textContent = "🎤 Voice";
  const state = typeof listen === "string" ? listen : (listen && listen.state) || "idle";
  btn.classList.toggle("active", S.voice.on);
  btn.classList.toggle("listening", S.voice.on && (state === "listening" || state === "speech" || state === "silence"));
  btn.classList.toggle("speaking", S.voice.on && S.voice.speaking);
  const meter = $("voice-meter");
  if (meter) meter.classList.toggle("hidden", !S.voice.on);
  if (!S.voice.on) btn.title = "Voice chat — talk, or hold the Avatar PTT hotkey. Silence sends.";
  else if (state === "transcribing") btn.title = "Transcribing…";
  else if (info && info.ptt) btn.title = "PTT — release the hotkey to send";
  else if (state === "speech" || state === "silence") btn.title = "Listening… silence will send";
  else if (S.voice.speaking) btn.title = "Speaking — talk or PTT to interrupt, or Stop to drop this reply";
  else btn.title = "Voice chat on — listening";
}

function updateVoiceMeter(listen) {
  const fills = [$("voice-meter-fill"), $("set-avatar-meter-fill"), $("voice-hud-meter-fill")];
  const meters = [$("voice-meter"), $("set-avatar-meter"), $("voice-hud-meter")];
  const level = listen && typeof listen.level_db === "number" ? listen.level_db : -90;
  const gate = listen && typeof listen.gate_db === "number" ? listen.gate_db : -45;
  const pct = Math.max(0, Math.min(100, ((level + 60) / 42) * 100));
  fills.forEach((el) => { if (el) el.style.width = pct + "%"; });
  meters.forEach((el) => { if (el) el.classList.toggle("open", level >= gate); });
  const text = $("set-avatar-meter-text");
  if (text) {
    const live = listen && listen.state && listen.state !== "idle";
    text.textContent = live
      ? ("Mic: " + level.toFixed(0) + " dB  ·  gate " + gate.toFixed(0) + " dB"
         + (listen.ptt ? "  ·  PTT" : ""))
      : "Mic: — (turn Voice on to see a live level)";
  }
}

function updateVoiceHud(st) {
  const hud = $("voice-hud");
  if (!hud) return;
  hud.classList.toggle("hidden", !S.voice.on);
  if (!S.voice.on) return;
  const listen = (st && st.listen) || {};
  const badge = $("voice-hud-badge");
  const detail = $("voice-hud-detail");
  const levelEl = $("voice-hud-level");
  const speaking = !!(st && st.speaking);
  const snippet = (st && (st.current_snippet || st.current_text)) || "";
  let kind = "listening";
  let label = "LISTENING";
  let text = "Speak, or hold the Avatar PTT hotkey (Mouse Back by default).";
  if (listen.error) {
    kind = "error"; label = "ERROR"; text = listen.error;
  } else if (!listen.state || listen.state === "idle") {
    kind = "error"; label = "NOT LISTENING";
    text = "The helper is up but listen is idle. Press Voice off and on again. "
         + (listen.model_loading ? "Speech model is still loading." : "");
  } else if (listen.ptt) {
    kind = "ptt"; label = "PTT — HOLDING";
    text = "Keep holding. Release the hotkey to transcribe and send.";
  } else if (listen.state === "transcribing") {
    kind = "transcribing"; label = "TRANSCRIBING";
    text = "Turning speech into text…";
  } else if (listen.state === "ready" && (listen.transcript || listen.last_transcript)) {
    kind = "ready"; label = "HEARD YOU";
    text = "“" + (listen.transcript || listen.last_transcript) + "”";
  } else if (listen.state === "speech") {
    kind = "speech"; label = "HEARING YOU";
    const need = (typeof listen.silence_seconds === "number" ? listen.silence_seconds : silenceSeconds());
    text = "Keep talking. " + need + " seconds of silence will send.";
  } else if (listen.state === "silence") {
    kind = "silence"; label = "WAITING FOR SILENCE";
    const ms = listen.silence_ms || 0;
    const need = (typeof listen.silence_seconds === "number" ? listen.silence_seconds : silenceSeconds());
    text = "Quiet for " + (ms / 1000).toFixed(1) + "s — send at " + need
         + "s, or hold PTT and release to send now.";
  } else if (speaking) {
    kind = "speaking"; label = "SPEAKING";
    text = snippet ? ("Reading: “" + snippet + "”") : "Reading the reply aloud.";
  } else if (S.generating) {
    kind = "listening"; label = "WAITING FOR REPLY";
    text = "Model is writing. Speech starts after two sentences.";
  } else {
    const last = listen.last_transcript || "";
    text = last
      ? ("Listening. Last heard: “" + last + "”")
      : "Listening for speech. Hold PTT to talk past the noise gate.";
  }
  if (speaking && kind !== "ptt" && kind !== "transcribing" && kind !== "speech") {
    kind = "speaking";
    label = "SPEAKING";
    text = snippet ? ("Reading: “" + snippet + "”") : "Reading the reply aloud.";
  }
  badge.className = "voice-hud-badge " + kind;
  badge.textContent = label;
  detail.textContent = text;
  if (levelEl) {
    if (typeof listen.level_db === "number") {
      levelEl.textContent = "Mic " + listen.level_db.toFixed(0) + " dB  ·  gate "
        + (typeof listen.gate_db === "number" ? listen.gate_db.toFixed(0) : "—") + " dB";
    } else {
      levelEl.textContent = "";
    }
  }
}

async function pushNoiseGate(value) {
  const gate = Math.max(0, Math.min(100, Number(value) || 0));
  try {
    await avatarFetch("/settings", { method: "POST", body: { noise_gate: gate } });
  } catch (e) { /* helper may not be up yet */ }
}

function silenceSeconds() {
  const n = parseInt(S.config.avatar_silence_seconds, 10);
  return Number.isFinite(n) ? Math.max(1, Math.min(30, n)) : 6;
}

async function pushSilenceSeconds(value) {
  const seconds = Math.max(1, Math.min(30, parseInt(value, 10) || 6));
  try {
    await avatarFetch("/settings", { method: "POST", body: { silence_seconds: seconds } });
  } catch (e) { /* helper may not be up yet */ }
}

async function ensureAvatarHelper() {
  const r = await api("/api/avatar/ensure", { method: "POST" });
  if (r && r.url) S.config.avatar_url = r.url;
  if (!r || !r.ok) throw new Error((r && r.error) || "Could not start the Avatar helper.");
  return r;
}

function statusSaysModelsReady(st) {
  if (!st) return false;
  if (st.models_ready) return true;
  const f5 = st.f5 || {};
  const stt = !!(st.stt_ready
    || (st.listen && st.listen.model_ready)
    || (st.dictation && st.dictation.model_ready));
  const engine = st.engine || st.tts_engine || (f5.model || f5.ready ? "f5" : "");
  const tts = engine !== "f5" || !!(st.tts_ready || f5.ready);
  return stt && tts;
}

function describeHelperLoad(st) {
  if (!st) return "Waiting for the helper…";
  const f5 = st.f5 || {};
  const ttsOk = !!(st.tts_ready || f5.ready || (st.tts_engine && st.tts_engine !== "f5"));
  const sttOk = !!(st.stt_ready || (st.dictation && st.dictation.model_ready) || (st.listen && st.listen.model_ready));
  const bits = [];
  bits.push(ttsOk ? "TTS ready" : (st.tts_loading || f5.loading ? "loading TTS" : "waiting for TTS"));
  bits.push(sttOk ? "transcription ready" : (st.stt_loading || (st.dictation && st.dictation.model_loading) ? "loading transcription" : "waiting for transcription"));
  return bits.join(" · ");
}

async function fetchHelperLoadState() {
  let lastErr = "";
  for (const path of ["/ready", "/status", "/health"]) {
    try {
      return await avatarFetch(path);
    } catch (e) {
      lastErr = e.message || String(e);
    }
  }
  throw new Error(lastErr || "Cannot reach the Avatar helper.");
}

async function waitForHelperModels(timeoutMs) {
  const deadline = Date.now() + (timeoutMs || 300000);
  let lastErr = "";
  while (Date.now() < deadline) {
    try {
      const st = await api("/api/avatar/voice-status");
      if (st && st.ready) {
        S.voice.modelsReady = true;
        if ($("avatar-helper-state")) {
          $("avatar-helper-state").textContent = st.detail || "Voice helper ready.";
        }
        return st;
      }
      if (st && (st.stt_error || st.tts_error) && st.stt !== "loading" && st.tts !== "loading") {
        const failed = [st.stt_error, st.tts_error].filter(Boolean).join(" · ");
        if (failed) throw new Error(failed);
      }
      lastErr = (st && st.detail) || "Waiting for the helper to publish ready…";
      const btn = $("btn-voice");
      if (btn && S.voice.helperLoading) {
        btn.title = lastErr;
        btn.textContent = "⏳ Loading voice helper";
      }
      if ($("avatar-helper-state")) $("avatar-helper-state").textContent = lastErr;
    } catch (e) {
      lastErr = e.message || String(e);
      if ($("avatar-helper-state")) $("avatar-helper-state").textContent = lastErr;
      if (/numpy|CUDA|Select an F5/i.test(lastErr)) throw e;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
  throw new Error("Voice helper models did not finish loading" + (lastErr ? " (" + lastErr + ")" : "") + ".");
}

async function onVoiceButtonClick() {
  if (S.voice.helperLoading) return;
  if (S.voice.on) {
    await setVoiceChat(false);
    return;
  }
  // Auto-start only loads models. The session (mic + replies) starts here.
  if (!S.voice.modelsReady) {
    setVoiceHelperLoading(true);
    try {
      await ensureAvatarHelper();
      await waitForHelperModels();
    } catch (e) {
      setVoiceHelperLoading(false);
      toast(e.message, 8000);
      if ($("avatar-helper-state")) $("avatar-helper-state").textContent = e.message;
      return;
    }
    setVoiceHelperLoading(false);
  }
  await setVoiceChat(true);
}

async function setVoiceChat(on) {
  if (on === S.voice.on) return;
  if (on) {
    S.voice.on = true;
    S.voice.skipSpeak = false;
    S.voice.handlingTurn = false;
    updateVoiceButton("listening");
    updateVoiceHud({ listen: { state: "listening" }, speaking: false });
    try {
      await pushNoiseGate($("set-avatar-gate") ? $("set-avatar-gate").value : S.config.avatar_noise_gate);
      await pushSilenceSeconds($("set-avatar-silence") ? $("set-avatar-silence").value : S.config.avatar_silence_seconds);
      await avatarFetch("/listen/start", { method: "POST", body: {
        silence_seconds: silenceSeconds(),
        noise_gate: parseInt($("set-avatar-gate") ? $("set-avatar-gate").value : S.config.avatar_noise_gate, 10) || 0,
      } });
    } catch (e) {
      S.voice.on = false;
      updateVoiceButton("idle");
      updateVoiceHud(null);
      toast(e.message, 8000);
      return;
    }
    startVoicePoll();
    setStatus("Voice chat on — talk, or hold the PTT hotkey");
  } else {
    S.voice.on = false;
    stopVoicePoll();
    S.voice.skipSpeak = true;
    S.voice.speaking = false;
    avatarFetch("/listen/cancel", { method: "POST", body: {} }).catch(() => {});
    avatarFetch("/stop", { method: "POST", body: {} }).catch(() => {});
    updateVoiceButton("idle");
    updateVoiceHud(null);
    refreshStopVisibility();
    setStatus("");
  }
}

function startVoicePoll() {
  stopVoicePoll();
  S.voice.pollTimer = setInterval(() => { voicePoll().catch(() => {}); }, 250);
}

function stopVoicePoll() {
  if (S.voice.pollTimer) {
    clearInterval(S.voice.pollTimer);
    S.voice.pollTimer = null;
  }
}

async function voicePoll() {
  if (!S.voice.on) return;
  let st;
  try {
    st = await avatarFetch("/status");
  } catch (e) {
    updateVoiceHud({ listen: { state: "idle", error: e.message } });
    setStatus("Voice: " + e.message);
    return;
  }
  const listen = st.listen || {};
  S.voice.listenState = listen.state || "idle";
  const ttsOn = !!st.speaking;
  if (S.voice.speaking !== ttsOn) {
    S.voice.speaking = ttsOn;
    refreshStopVisibility();
  }
  updateVoiceButton(listen.state, listen);
  updateVoiceMeter(listen);
  updateVoiceHud(st);
  if (listen.ptt) setStatus("Voice PTT — release to send");
  else if (listen.state === "speech") setStatus("Voice — hearing you");
  else if (listen.state === "silence") setStatus("Voice — waiting for silence…");
  else if (listen.state === "transcribing") setStatus("Voice — transcribing…");
  else if (S.voice.on && !S.generating && !S.voice.speaking) setStatus("Voice chat on — listening");

  if (listen.barge_in && (S.generating || S.voice.speaking) && !S.voice.skipSpeak) {
    await stopGeneration();
  }

  if (listen.state === "ready" && listen.transcript && !S.voice.handlingTurn && !S.generating) {
    const text = String(listen.transcript || "").trim();
    S.voice.handlingTurn = true;
    try {
      await avatarFetch("/listen/ack", { method: "POST", body: {} });
      if (/[A-Za-z]/.test(text)) {
        S.voice.skipSpeak = false;
        await sendMessage(text);
      }
    } catch (e) {
      toast("Voice: " + e.message);
    } finally {
      S.voice.handlingTurn = false;
    }
  }
}

function renderAvatarSettings() {
  if (!$("set-avatar-dir")) return;
  $("set-avatar-dir").value = S.config.avatar_dir || "";
  $("set-avatar-url").value = S.config.avatar_url || "http://127.0.0.1:8765";
  const mode = S.config.avatar_start_mode || "voice_button";
  $("avatar-start-app").checked = mode === "app_start";
  $("avatar-start-voice").checked = mode !== "app_start";
  const gate = S.config.avatar_noise_gate != null ? S.config.avatar_noise_gate : 30;
  $("set-avatar-gate").value = gate;
  $("set-avatar-gate-label").textContent = String(gate);
  $("set-avatar-silence").value = silenceSeconds();
  const open = $("link-avatar-settings");
  if (open) open.href = ($("set-avatar-url").value.trim() || "http://127.0.0.1:8765").replace(/\/+$/, "");
}

async function saveAvatarSettings() {
  const body = {
    avatar_dir: $("set-avatar-dir").value.trim(),
    avatar_url: $("set-avatar-url").value.trim() || "http://127.0.0.1:8765",
    avatar_start_mode: $("avatar-start-app").checked ? "app_start" : "voice_button",
    avatar_noise_gate: parseInt($("set-avatar-gate").value, 10) || 0,
    avatar_silence_seconds: parseInt($("set-avatar-silence").value, 10) || 6,
  };
  const r = await api("/api/settings", { method: "POST", body });
  S.config = { ...S.config, ...r.config };
  renderAvatarSettings();
  await pushNoiseGate(body.avatar_noise_gate);
  await pushSilenceSeconds(body.avatar_silence_seconds);
  toast("Voice settings saved");
}

async function maybeAutostartAvatar() {
  const mode = S.config.avatar_start_mode || "voice_button";
  if (mode !== "app_start") return;
  setVoiceHelperLoading(true, "Starting voice helper…");
  try {
    const r = await ensureAvatarHelper();
    if ($("avatar-helper-state")) {
      $("avatar-helper-state").textContent = r.started
        ? "Helper started. Loading models…"
        : "Helper already running. Loading models…";
    }
    await waitForHelperModels();
    S.voice.modelsReady = true;
    if ($("avatar-helper-state")) {
      $("avatar-helper-state").textContent = "Voice helper ready. Press Voice in the chat to start listening.";
    }
    setVoiceHelperLoading(false);
    setStatus("Voice helper ready — press Voice to start listening");
  } catch (e) {
    setVoiceHelperLoading(false);
    if ($("avatar-helper-state")) $("avatar-helper-state").textContent = e.message;
  }
}

/**
 * Handle Send: assemble the user turn, push it into the chat, and dispatch to either
 * the persona pipeline or ordinary generation.
 *
 * Any staged data blocks are prepended to the typed text as XML, so the model sees
 * the data before the instruction. Creates a private chat on the fly if none is open,
 * and auto-titles a new chat from the first message. No-ops while a run is active.
 * ``presetText`` is used by voice chat so a transcript can send without touching
 * the composer.
 */
async function sendMessage(presetText) {
  if (S.generating || S.batchRunning) return;
  const fromVoice = typeof presetText === "string";
  const text = fromVoice ? presetText.trim() : $("input-box").value.trim();
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
  if (!fromVoice) $("input-box").value = "";
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
  const speaker = S.voice.on ? createSentenceSpeaker() : null;
  if (speaker) {
    S.voice.skipSpeak = false;
    S.voice.speaker = speaker;
  }
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
      if (speaker) speaker.feed(d.text);
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
  if (speaker) speaker.end();
  S.voice.speaker = null;
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
    // Walk an RSS drafting queue one save at a time. No-op when there isn't one.
    openNextMemoryDraft();
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
  const stagedKb = await chooseAndStage({ accept: ACCEPT_DOCS });
  if (stagedKb === null) return;
  if (isLocalBrowser()) toast("Choose documents in the dialog…");
  const pid = S.editingPersona.id;
  // The server mints the run id (unique per invocation) and hands it back in `begin`;
  // Cancel has to use that one or it stops nothing.
  let runId = "";
  const progressEl = $("pe-compile-progress");
  let ui = null;
  try {
    // Each document here is parsed AND embedded, so this is the slow path that most
    // needs a progress bar.
    await streamSSE(`/api/personas/${pid}/knowledge/add-files`, stagedKb, {
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
/** Import a feed's episodes as persona knowledge documents.
 *  POST + streamSSE (not sourceStream) because the body carries the options. */
async function addPersonaKbRss() {
  const url = $("pe-rss-input").value.trim();
  if (!url) { toast("Enter a feed URL"); return; }
  const pid = S.editingPersona.id;
  const body = {
    url,
    limit: Math.max(0, Math.min(500, parseInt($("pe-rss-limit").value, 10) || 0)),
    whisper: $("pe-rss-whisper").checked,
    ...rssFilterOpts("pe-rss"),
  };
  const prog = $("pe-rss-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the feed…";
  const btn = $("btn-pe-rss-fetch"); btn.disabled = true;
  let n = 0;
  try {
    await streamSSE(`/api/personas/${pid}/knowledge/add-rss`, body, {
      start: (d) => { _peRssRunId = d.run_id || ""; },
      feed: (d) => { prog.textContent = `${d.title || "Feed"} — ${d.total} episode(s)`; },
      warning: (d) => { toast(d.message, 8000); },
      progress: (d) => { prog.textContent = rssProgressText(d); },
      document: (d) => { n++; prog.textContent = `${d.title || d.name}: ${d.chunks} chunk(s)`; },
      episode_error: (d) => { toast(`${d.title}: ${d.message}`, 6000); },
      complete: (r) => {
        toast(`Imported ${n} episode(s) as knowledge documents.`);
        if ((r.errors || []).length) toast("Notes: " + r.errors.join("; "), 8000);
        prog.classList.add("hidden");
        $("pe-rss-panel").classList.add("hidden");
        renderPersonaKb(); renderEmbedBanner();
        refreshCompileStatus("persona", pid, $("pe-compile-badge"));
      },
      error: (d) => { toast("Import failed: " + d.message, 8000); prog.textContent = "Failed."; },
      done: () => { btn.disabled = false; _peRssRunId = ""; },
    });
  } catch (e) {
    toast("Import failed: " + e.message);
    prog.textContent = "Failed.";
    btn.disabled = false;
  }
}

let _peRssRunId = "";

function cancelPersonaKbRss() {
  if (_peRssRunId) {
    api("/api/stop", { method: "POST", body: { run_id: _peRssRunId } }).catch(() => {});
  }
  $("pe-rss-panel").classList.add("hidden");
  $("btn-pe-rss-fetch").disabled = false;
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

let _peMemRssRunId = "";
let _peMemDrafts = [];

/** Draft memories from a feed's episodes and queue them for review.
 *  Nothing is saved here: the drafts are opened one at a time in the existing memory
 *  editor, so every one is a deliberate human save. Auto-saving a few hundred
 *  machine-written recollections would skew the weight-blended retrieval permanently. */
async function draftMemoriesFromRss() {
  const url = $("pe-mem-rss-input").value.trim();
  if (!url) { toast("Enter a feed URL"); return; }
  const pid = S.editingPersona.id;
  const body = {
    url,
    limit: Math.max(1, Math.min(50, parseInt($("pe-mem-rss-limit").value, 10) || 3)),
    server_url: currentServerUrl(),
    model: getSelectedModel(),
    ...rssFilterOpts("pe-mem-rss"),
  };
  const prog = $("pe-mem-rss-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the feed…";
  const btn = $("btn-pe-mem-rss-go"); btn.disabled = true;
  _peMemDrafts = [];
  try {
    await streamSSE(`/api/personas/${pid}/draft-memories-from-rss`, body, {
      start: (d) => { _peMemRssRunId = d.run_id || ""; },
      feed: (d) => { prog.textContent = `${d.title || "Feed"} — drafting ${d.total}…`; },
      // The only panel that was missing this. A filter matching nothing completes with
      // "No drafts were produced" and the warning saying WHY had nowhere to land.
      warning: (d) => { toast(d.message, 8000); },
      progress: (d) => { prog.textContent = rssProgressText(d); },
      draft: (d) => {
        _peMemDrafts.push(d.draft);
        prog.textContent = `Drafted ${_peMemDrafts.length}: ${d.episode}`;
      },
      episode_error: (d) => { toast(`${d.title}: ${d.message}`, 6000); },
      complete: () => {
        prog.classList.add("hidden");
        $("pe-mem-rss-panel").classList.add("hidden");
        if (!_peMemDrafts.length) { toast("No drafts were produced."); return; }
        toast(`${_peMemDrafts.length} draft(s) — review and save each one`, 6000);
        openNextMemoryDraft();
      },
      error: (d) => { toast("Drafting failed: " + d.message, 8000); prog.textContent = "Failed."; },
      done: () => { btn.disabled = false; _peMemRssRunId = ""; },
    });
  } catch (e) {
    toast("Drafting failed: " + e.message);
    prog.textContent = "Failed.";
    btn.disabled = false;
  }
}

/** Load the next queued draft into the existing memory editor. Closing it without
 *  saving simply drops that draft — which is the intended escape hatch. */
function openNextMemoryDraft() {
  const d = _peMemDrafts.shift();
  if (!d) return;
  _memPersonaId = S.editingPersona.id;
  $("mem-title").value = d.title || "";
  $("mem-desc").value = d.description || "";
  $("mem-weight").value = d.emotional_weight ?? 5;
  $("mem-tags").value = (d.tags || []).join(", ");
  openModal("modal-memory");
  if (_peMemDrafts.length) toast(`${_peMemDrafts.length} more draft(s) after this`, 4000);
}

function cancelDraftMemoriesFromRss() {
  if (_peMemRssRunId) {
    api("/api/stop", { method: "POST", body: { run_id: _peMemRssRunId } }).catch(() => {});
  }
  $("pe-mem-rss-panel").classList.add("hidden");
  $("btn-pe-mem-rss-go").disabled = false;
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
  $("btn-pe-kb-rss").onclick = () => {
    const panel = $("pe-rss-panel");
    panel.classList.remove("hidden");
    $("pe-rss-limit").value = String(S.config.rss_max_episodes ?? 25);
    refreshWhisperStatus();
    $("pe-rss-input").focus();
  };
  $("btn-pe-rss-fetch").onclick = addPersonaKbRss;
  $("btn-pe-rss-cancel").onclick = cancelPersonaKbRss;
  $("btn-pe-compile").onclick = async () => {
    if (!S.editingPersona) { toast("Open a persona first"); return; }
    await runCompile("persona", S.editingPersona.id, { force: $("pe-compile-force").checked },
                     $("pe-compile-progress"), $("pe-compile-badge"));
    renderPersonaKb(); renderEmbedBanner();
  };
  $("btn-pe-mem-add").onclick = addPersonaMemoryUI;
  $("btn-pe-mem-rss").onclick = () => {
    $("pe-mem-rss-panel").classList.remove("hidden");
    $("pe-mem-rss-input").focus();
  };
  $("btn-pe-mem-rss-go").onclick = draftMemoriesFromRss;
  $("btn-pe-mem-rss-cancel").onclick = cancelDraftMemoriesFromRss;
  $("btn-pe-test").onclick = runPersonaTest;
  $("btn-persona-export-xml").onclick = () => window.open(`/api/personas/${S.editingPersona.id}/export.xml`, "_blank");
  $("btn-persona-export-bundle").onclick = () => window.open(`/api/personas/${S.editingPersona.id}/export.zip`, "_blank");
  $("btn-persona-import").onclick = importPersonaUI;
}

// Import is implemented in Phase 10; stub keeps the button harmless until then.
async function importPersonaUI() {
  try {
    const staged = await chooseAndStage({ accept: ".xml,.zip", multiple: false });
    if (staged === null) return;
    const r = await api("/api/personas/import", { method: "POST", body: staged });
    if (r.error) { toast(r.error); return; }
    toast("Imported: " + (r.persona ? r.persona.profile.name : "ok"));
    await renderPersonaTab();
    if (r.persona) openPersonaEditor(r.persona.id);
  } catch (e) { toast("Import: " + e.message); }
}

async function runGeneration(searchQuery, opts = {}) {
  setGeneratingUI(true);
  S.runId = uid();
  const speaker = S.voice.on ? createSentenceSpeaker() : null;
  if (speaker) {
    S.voice.skipSpeak = false;
    S.voice.speaker = speaker;
  }
  // The chat sent to the backend (for its context + settings). Defaults to the
  // active chat. A queue run passes an isolated single-turn snapshot here so each
  // prompt is answered independently, while all rendering/persistence below stays
  // targeted at the active S.chat so results stack into one thread.
  const sendChat = opts.sendChat || S.chat;
  // Pass-driven: one answer (+optional reasoning) bubble per pass. A single-pass
  // generation is just one unlabeled pass.
  let bubble = null, reasonBubble = null, sourceBubble = null;
  let curContent = "", curReason = "", curLabel = "", curIntermediate = false;
  let reasonSpoken = false;
  let curImages = [];
  // The retrieved chunks arrive once, during pass 0, and describe every pass — so unlike
  // the per-pass state above they are NOT cleared by finalizePass.
  let curSources = [];
  let errored = false;

  function finalizePass() {
    if (!bubble) return;
    bubble.classList.remove("streaming");
    if (!errored) {
      const msg = { role: "assistant", content: curContent };
      if (curReason) msg.reasoning = curReason;
      // Multi-Pass reuses one retrieval for every round, so the panel belongs to the
      // answer the user keeps rather than being repeated above each intermediate pass.
      if (curSources.length && !curIntermediate) msg.sources = curSources;
      if (curImages.length) msg.images = curImages;
      if (curLabel) { msg.pass_label = curLabel; msg.intermediate = curIntermediate; }
      S.chat.messages.push(msg);
    }
    bubble = null; reasonBubble = null; sourceBubble = null;
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
      if (speaker) { speaker.endSection(); reasonSpoken = false; }
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
      if (speaker && $("btn-voice-reason").classList.contains("active")) {
        speaker.feed(d.content);
        reasonSpoken = true;
      }
      if (follow) scrollBottom();
    },
    // What RAG retrieved for this turn — shown under the reasoning, collapsed, so the
    // panel doesn't push the answer off screen while it streams.
    sources: (d) => {
      curSources = d.items || [];
      if (!bubble || !curSources.length) return;
      const follow = isNearBottom($("messages"));
      if (!sourceBubble) {
        sourceBubble = makeSourcesBubble(curSources, true);
        $("messages").insertBefore(sourceBubble.el, bubble);
      } else {
        sourceBubble.set(curSources);
      }
      if (follow) scrollBottom();
    },
    chunk: (d) => {
      if (!bubble) return;
      const follow = isNearBottom($("messages"));
      if (curContent === "") bubble._body.textContent = "";
      curContent += d.content; bubble._body.textContent = curContent;
      if (speaker) {
        if (reasonSpoken) { speaker.endSection(); reasonSpoken = false; }
        speaker.feed(d.content);
      }
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
  if (speaker) speaker.end();
  S.voice.speaker = null;
  setGeneratingUI(false);
  setStatus("");
  if (!errored) await persistChat(true);
  renderMessages(); // re-render so the final assistant gets regen controls
  if (!errored) await maybeExtractMemories();
}

async function stopGeneration() {
  S.queueStop = true;   // also halt a sequential queue loop after the current item
  if (S.runId) await api("/api/stop", { method: "POST", body: { run_id: S.runId }});
  S.voice.skipSpeak = true;
  S.voice.speaker = null;
  if (S.voice.on) {
    avatarFetch("/stop", { method: "POST", body: {} }).catch(() => {});
    S.voice.speaking = false;
    refreshStopVisibility();
  }
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
    if (folderPickingUnavailable()) return;
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
    // This snapshot gets a throwaway id and a single synthetic turn, so indexing its
    // corpus would leave one orphaned scope per queued prompt that nothing ever deletes.
    // Retrieval still works — it just happens in memory for these.
    rag_ephemeral: true,
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
      // Cleared per ITEM, not per pass: one retrieval serves every pass of an item.
      L.sourceBubble = null;
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
    sources: (d) => {
      const L = lanes[d.lane]; if (!L || !L.curBubble || !(d.items || []).length) return;
      if (!L.sourceBubble) { L.sourceBubble = makeSourcesBubble(d.items, true); L.body.insertBefore(L.sourceBubble.el, L.curBubble); }
      else L.sourceBubble.set(d.items);
      laneScroll(L);
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
  if ((frame.sources || []).length) asst.sources = frame.sources;
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
  body.download = !isLocalBrowser();
  const r = await api("/api/prompts/export-xml", { method: "POST", body });
  if (takeDownload(r)) { toast("Exported"); return; }
  if (r && r.ok) toast("Exported to " + r.path);
  else if (r && r.cancelled) { /* user cancelled */ }
  else toast("Export failed" + (r && r.error ? ": " + r.error : ""));
}
async function pbImport() {
  const staged = await chooseAndStage({ accept: ACCEPT_XML, multiple: false });
  if (staged === null) return;
  const r = await api("/api/prompts/import-xml", { method: "POST", body: staged });
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
    const r = await api("/api/memory/cores/export",
                        { method: "POST", body: { ids: [core.id], download: !isLocalBrowser() } });
    if (takeDownload(r)) { toast("Exported"); return; }
    if (r.cancelled) return;
    toast(r.ok ? `Exported to ${r.path}` : "Export failed: " + (r.error || "unknown"));
  } catch (e) { toast("Export failed: " + e.message); }
}

async function importMemoryCore() {
  try {
    const staged = await chooseAndStage({ accept: ACCEPT_JSON, multiple: false });
    if (staged === null) return;
    const r = await api("/api/memory/cores/import", { method: "POST", body: staged });
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
  $("set-whisper-model").value = S.config.whisper_model || "large-v3";
  $("set-whisper-device").value = S.config.whisper_device || "auto";
  $("set-whisper-compute").value = S.config.whisper_compute_type || "float16";
  $("set-whisper-batch").value = S.config.whisper_batch_size ?? 8;
  $("set-whisper-lang").value = S.config.whisper_language || "";
  $("set-whisper-vad").checked = S.config.whisper_vad !== false;
  renderServerRows();
  renderParallelServers();
  refreshNetworkCard();
  refreshYouTubeCacheStats();
  refreshRssCacheStats();
  refreshWhisperStatus();
  renderAvatarSettings();
}

// ------------------------------- Network Access ----------------------
// Bind address and port come from /api/network rather than /api/state. They live in a
// plaintext file outside the encrypted per-profile settings, because main.py has to
// read them before anyone has logged in to decrypt anything — and detecting this
// machine's addresses resolves a hostname, which is too slow to sit on the page-load
// path that /api/state serves.
let S_net = null;

const isLocalBrowser = () =>
  ["localhost", "127.0.0.1", "::1", "[::1]"].includes(location.hostname);

async function refreshNetworkCard(force = false) {
  try {
    S_net = await api("/api/network" + (force ? "?refresh=1" : ""));
  } catch (e) {
    $("net-status").textContent = "Could not read the network settings: " + e.message;
    return;
  }
  $("set-lan-enabled").checked = !!S_net.saved.lan_enabled;
  const portInput = $("set-net-port");
  portInput.value = S_net.saved.port;
  // From the server, so the field cannot disagree with what the route enforces.
  portInput.min = S_net.limits.min;
  portInput.max = S_net.limits.max;

  const rt = S_net.runtime;
  let status;
  if (!rt) {
    status = "The server was not started by main.py, so there is nothing live to " +
             "compare against. Saved settings apply the next time you run it.";
  } else if (rt.lan_enabled) {
    status = `Right now: shared on your network, port ${rt.port}.`;
  } else {
    status = `Right now: this computer only, port ${rt.port}.`;
  }
  if (rt && rt.fell_back && rt.port !== rt.requested_port) {
    status += ` (Port ${rt.requested_port} was unavailable when the app started.)`;
  }
  $("net-status").textContent = status;

  const creds = $("net-creds-warning");
  const defaults = !!S_net.using_default_creds;
  creds.classList.toggle("hidden", !defaults);
  creds.innerHTML =
    "<b>Your login is still admin / admin.</b> Anyone who can reach this app over the " +
    "network can sign in with it, read your chats, and read and write files on this " +
    "computer. Change it before you share this.";
  $("net-creds-actions").classList.toggle("hidden", !defaults);
  if (!defaults) $("net-pw-form").classList.add("hidden");
  // Only while the form is closed — this refresh also runs on an ordinary settings
  // render, and must not overwrite a username someone is halfway through typing.
  if ($("net-pw-form").classList.contains("hidden")) {
    $("net-pw-user").value = S_net.username || "admin";
  }

  $("net-firewall-cmd").textContent = S_net.firewall_command;
  renderNetAddresses();
  markNetworkDirty();
}

/** The addresses another machine would type. Every candidate is listed rather than
 *  guessed at: a VPN or a WSL/Docker adapter looks exactly like a real LAN address
 *  from the outside, and picking wrong is worse than explaining. */
function renderNetAddresses() {
  const box = $("net-addresses");
  box.innerHTML = "";
  const on = $("set-lan-enabled").checked;
  if (!S_net || !on) return;
  const port = parseInt($("set-net-port").value) || S_net.saved.port;
  if (!S_net.addresses.length) {
    box.innerHTML = '<div class="muted">No network address found — this computer ' +
                    "may not be connected to a network right now.</div>";
    return;
  }
  const head = document.createElement("div");
  head.className = "muted";
  head.textContent = S_net.addresses.length > 1
    ? "Type one of these on the other computer. The one marked “best” is usually right; " +
      "172.x entries are often WSL or Docker and will not work."
    : "Type this on the other computer:";
  box.appendChild(head);
  S_net.addresses.forEach((a) => {
    const url = `http://${a.ip}:${port}`;
    const row = document.createElement("div");
    row.className = "net-addr";
    const code = document.createElement("code");
    code.textContent = url;
    row.appendChild(code);
    if (a.default_route) {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = "best";
      row.appendChild(b);
    }
    if (a.kind === "cgnat") {
      const m = document.createElement("span");
      m.className = "muted";
      m.textContent = "VPN — only reachable over that VPN";
      row.appendChild(m);
    }
    const copy = document.createElement("button");
    copy.className = "small";
    copy.textContent = "Copy";
    copy.onclick = () => { navigator.clipboard.writeText(url); toast("Address copied"); };
    row.appendChild(copy);
    box.appendChild(row);
  });
}

/** Reflect unsaved edits: what still needs a restart, and what the user is agreeing to
 *  by ticking the box. Runs on every keystroke, so it must not call the server. */
function markNetworkDirty() {
  if (!S_net) return;
  const on = $("set-lan-enabled").checked;
  const port = parseInt($("set-net-port").value) || 0;
  const rt = S_net.runtime;

  $("net-share-warning").classList.toggle("hidden", !on);

  const pw = $("net-port-warning");
  // A port in the ephemeral range can be claimed by an outgoing connection before the
  // app starts, which surfaces as an occasional, baffling failure to bind.
  const lim = S_net.limits;
  const ephemeral = port >= lim.ephemeral_from && port <= lim.max;
  pw.classList.toggle("hidden", !ephemeral);
  if (ephemeral) {
    pw.textContent = `Port ${port} is in the range Windows hands out to outgoing ` +
      "connections, so it may occasionally be taken when you start the app. " +
      `Something between ${lim.min} and ${lim.ephemeral_from - 1} is a safer choice.`;
  }

  const note = $("net-restart-note");
  const forced = rt && (rt.port_forced || rt.lan_forced);
  const changed = rt && ((port !== rt.port && !rt.port_forced) ||
                         (on !== rt.lan_enabled && !rt.lan_forced));
  note.classList.toggle("hidden", !changed && !forced);
  if (forced) {
    // A --port / --lan flag wins for this run, so promising a restart would be a lie.
    note.textContent = "A command-line flag is overriding this for the current run. " +
      "What you save here applies the next time you start the app without that flag.";
  } else if (changed) {
    note.textContent = S_net.restart_supported
      ? "These changes only take effect once the server restarts — use Save & restart server."
      : "These changes take effect the next time you start the app.";
  }
  renderNetAddresses();
}

/** Change the login from here, because the warning above is otherwise a dead end: the
 *  only other ways are the login-page nag (which this user has already dismissed to get
 *  this far) and `python main.py --reset-password` at a terminal. */
async function changeLoginPassword() {
  const np = $("net-pw-new").value;
  if (!np) { toast("Enter a new password"); return; }
  if (np !== $("net-pw-confirm").value) { toast("The two passwords do not match"); return; }
  try {
    await api("/api/auth/change", {
      method: "POST",
      body: {
        current_password: $("net-pw-current").value,
        new_username: $("net-pw-user").value.trim() || "admin",
        new_password: np,
      },
    });
  } catch (e) {
    toast(e.message, 6000);
    return;
  }
  ["net-pw-current", "net-pw-new", "net-pw-confirm"].forEach((id) => ($(id).value = ""));
  $("net-pw-form").classList.add("hidden");
  // The same data key is re-wrapped under the new password rather than the data being
  // re-encrypted, so the session survives.
  toast("Password changed — you are still signed in.", 6000);
  refreshNetworkCard();
}

async function saveNetwork(restart) {
  const port = parseInt($("set-net-port").value);
  const buttons = ["btn-save-network", "btn-restart-network"];
  const lan = $("set-lan-enabled").checked;
  // Turning sharing off from a remote browser cuts the branch you are sitting on.
  if (!lan && !isLocalBrowser() &&
      !confirm("You are connected from another computer. Turning network sharing off " +
               "will disconnect you as soon as the server restarts. Continue?")) {
    return;
  }
  const enable = () => buttons.forEach((id) => ($(id).disabled = false));
  buttons.forEach((id) => ($(id).disabled = true));
  let r;
  try {
    r = await api("/api/network", {
      method: "POST",
      body: { lan_enabled: lan, port, restart: !!restart },
    });
  } catch (e) {
    enable();
    toast(e.message, 6000);
    return;
  }
  const wasPort = S_net && S_net.runtime ? S_net.runtime.port : null;
  S_net = r;
  // Stay disabled only while a restart really is under way — the page is about to follow
  // the server to its new address and a second click would be meaningless.
  if (!r.restarting) enable();
  if (!r.restarting) {
    toast(r.restart_required ? "Saved — restart the app to apply it"
                             : "Network settings saved");
    refreshNetworkCard();
    return;
  }
  // next_port, not saved.port: a --port flag on the command line pins the port for the
  // whole run, and following the saved one would send the page to a dead address.
  if (wasPort !== null && r.next_port !== wasPort) {
    followRestart(r.next_port);
  } else {
    toast("Restarting the server…", 6000);
    setTimeout(() => refreshNetworkCard(true), 2500);
  }
}

/** Follow the server to its new port. A different port is a different origin, so the
 *  probe has to be a no-cors fetch — an opaque response that merely resolves is proof
 *  the new listener is answering. location.hostname, not the server address, so a
 *  browser on another machine follows itself rather than being sent to the host. */
async function followRestart(port) {
  const url = `${location.protocol}//${location.hostname}:${port}/`;
  toast(`Restarting on port ${port} — this page will follow…`, 9000);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  await sleep(1200);
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try {
      await fetch(url + "login", { mode: "no-cors", cache: "no-store" });
      location.href = url;
      return;
    } catch (e) {
      await sleep(600);
    }
  }
  toast(`The server has not come back yet. Open ${url} once it has.`, 15000);
}

/** Size the on-disk YouTube cache for the Settings card. Cheap — the route stats file
 *  sizes without decrypting anything. */
async function refreshYouTubeCacheStats() {
  const el = $("yt-cache-stats");
  if (!el) return;
  try {
    const r = await api("/api/youtube/cache");
    const mb = (r.bytes || 0) / (1024 * 1024);
    el.textContent = r.entries
      ? `${r.entries} video(s) cached · ${mb < 0.1 ? "<0.1" : mb.toFixed(1)} MB`
      : "Nothing cached yet.";
  } catch (e) { el.textContent = "—"; }
}
async function clearYouTubeCache() {
  if (!(await confirmModal("Clear every cached YouTube transcript and comment set for this data profile?"))) return;
  try {
    const r = await api("/api/youtube/cache", { method: "DELETE" });
    toast(`Cleared ${r.removed} cached video(s)`);
  } catch (e) { toast("Clear failed: " + e.message); }
  refreshYouTubeCacheStats();
}

/** Size the on-disk RSS cache for the Settings card. Unlike the YouTube one this does
 *  decrypt each episode, to count the locally transcribed ones — see clearRssCache. */
async function refreshRssCacheStats() {
  const el = $("rss-cache-stats");
  if (!el) return;
  try {
    const r = await api("/api/rss/cache");
    const mb = (r.bytes || 0) / (1024 * 1024);
    if (!r.episodes && !r.feeds) { el.textContent = "Nothing cached yet."; return; }
    el.textContent = `${r.episodes} episode(s), ${r.feeds} feed listing(s) · ` +
      `${mb < 0.1 ? "<0.1" : mb.toFixed(1)} MB` +
      (r.transcribed ? ` · ${r.transcribed} transcribed locally` : "");
  } catch (e) { el.textContent = "—"; }
}

/** Clear part of the RSS cache. The destructive variant names the cost: a locally
 *  transcribed episode is GPU-minutes that clearing throws away for good. */
async function clearRssCache(what) {
  let msg = "Re-read every feed listing on the next import? Cached episodes are kept.";
  if (what !== "feeds") {
    let extra = "";
    try {
      const s = await api("/api/rss/cache");
      extra = s.transcribed
        ? `\n\n${s.transcribed} of them were transcribed locally — those took minutes of ` +
          "GPU time each and will have to be redone."
        : "";
    } catch (e) {}
    msg = "Clear every cached podcast episode for this data profile?" + extra;
  }
  if (!(await confirmModal(msg))) return;
  try {
    const qs = what ? `?what=${encodeURIComponent(what)}` : "";
    const r = await api(`/api/rss/cache${qs}`, { method: "DELETE" });
    toast(`Cleared ${r.removed} cached record(s)`);
  } catch (e) { toast("Clear failed: " + e.message); }
  refreshRssCacheStats();
}

/** Report whether local transcription is usable, and on what. Loads no model — this is
 *  the same probe the RSS panels use to decide whether to offer the Whisper checkbox. */
async function refreshWhisperStatus() {
  const el = $("whisper-status");
  if (!el) return;
  try {
    const s = await api("/api/transcribe/status");
    S.whisper = s;
    if (!s.available) {
      el.textContent = "faster-whisper is not installed — pip install faster-whisper";
    } else if (s.loaded) {
      el.textContent = `Loaded: ${s.model} on ${s.device} (${s.compute_type})`;
    } else if (s.cuda_error) {
      // The sm_120 case: CUDA looked available, then the first decode had no kernels.
      el.textContent = "Ready (CPU only — the GPU couldn't run the model: " +
                       s.cuda_error.slice(0, 90) + ")";
    } else {
      el.textContent = s.cuda_usable ? "Ready — GPU available, no model loaded"
                                     : "Ready — CPU only, no model loaded";
    }
    updateWhisperOffers();
  } catch (e) { el.textContent = "—"; }
}

/** Enable or disable every "transcribe missing episodes" affordance from the last known
 *  probe. Ticking a box that cannot possibly work is worse than not offering it. */
function updateWhisperOffers() {
  const ok = !!(S.whisper && S.whisper.available);
  ["add-rss-whisper", "lib-rss-whisper", "batch-rss-whisper"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.disabled = !ok;
    if (!ok) el.checked = false;
    const label = el.closest("label");
    if (label) {
      label.title = ok
        ? "Download the audio and transcribe it locally. Slow — minutes per episode."
        : "Needs faster-whisper:  pip install faster-whisper";
      label.classList.toggle("disabled", !ok);
    }
  });
}

async function resetWhisperModel() {
  try {
    await api("/api/transcribe/reset", { method: "POST" });
    toast("Speech model unloaded");
  } catch (e) { toast("Unload failed: " + e.message); }
  refreshWhisperStatus();
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
    whisper_model: $("set-whisper-model").value.trim() || "large-v3",
    whisper_device: $("set-whisper-device").value,
    whisper_compute_type: $("set-whisper-compute").value,
    whisper_batch_size: parseInt($("set-whisper-batch").value) || 8,
    whisper_language: $("set-whisper-lang").value.trim(),
    whisper_vad: $("set-whisper-vad").checked,
  };
  const r = await api("/api/settings", { method: "POST", body });
  S.config = { ...S.config, ...r.config };
  updateImageResButton();   // its tooltip quotes the cap that just changed
  // Saving a whisper_* key drops the loaded model server-side, so the card's "Loaded:"
  // line is now stale — and a device change may have re-enabled the GPU.
  refreshWhisperStatus();
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
const SOURCE_KINDS = ["write", "file", "url", "youtube", "rss", "audio", "search", "image"];
// Kinds whose `filename` (library) / `source` (attachment) holds a clickable URL.
// "audio" is out: its filename is a local path, and linking one would be a dead <a>.
const LINKED_KINDS = ["url", "youtube", "rss", "search"];

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
  // Keyed by the item's STABLE id, not its position: a Sources link has to find this
  // item after other items have been removed above it.
  if (it.id) wrap.dataset.itemId = it.id;
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
  const stagedLib = await chooseAndStage({ accept: ACCEPT_DOCS });
  if (stagedLib === null) return;
  setStatus(isLocalBrowser() ? "Waiting for file selection…" : "Reading files…");
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
    await streamSSE(`/api/libraries/${libId}/add-text-files`, stagedLib, {
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
  // The RSS panel was populated below but never un-hidden, so "Add RSS / Podcast" hid
  // every panel and showed nothing.
  $("lib-rss-panel").classList.toggle("hidden", which !== "rss");
  $("lib-search-panel").classList.toggle("hidden", which !== "search");
  if (which === "url") { $("lib-url-input").value = ""; $("lib-url-input").focus(); }
  if (which === "youtube") {
    $("lib-yt-input").value = "";
    $("lib-yt-limit").value = "0";
    $("lib-yt-kind-video").checked = true;
    const prog = $("lib-yt-progress"); prog.classList.add("hidden"); prog.textContent = "";
    updateLibraryYouTubePanelKind();
    $("lib-yt-input").focus();
  }
  if (which === "rss") {
    $("lib-rss-input").value = "";
    $("lib-rss-limit").value = String(S.config.rss_max_episodes ?? 25);
    resetRssFilters("lib-rss");
    const prog = $("lib-rss-progress"); prog.classList.add("hidden"); prog.textContent = "";
    refreshWhisperStatus();
    $("lib-rss-input").focus();
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
  $("lib-rss-panel").classList.add("hidden");
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

// Deliberate ports of youtube.parse_video_id / parse_playlist_id, kept in sync by hand.
// Duplicated rather than served by a route because the composer panel reacts per
// keystroke to reveal a field, and a round trip per keystroke for a regex is not worth
// it. It is safe because the JS answer only chooses WHICH ROUTE TO CALL — both routes
// re-parse server-side and reject a mismatch, and the Python version is the tested one.
// Don't let this grow features the Python side lacks.
const YT_VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
const YT_PLAYLIST_ID = /^(?:PL|UU|UL|OL|RD|FL|LL)[A-Za-z0-9_-]{10,}$/;   // WL excluded: needs owner cookies
const YT_PATH_FORMS = ["/shorts/", "/live/", "/embed/", "/v/"];
const YT_HOSTS = ["youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
                  "youtube-nocookie.com", "www.youtube.com", "www.youtube-nocookie.com"];

function ytUrlOf(raw) {
  const s = (raw || "").trim();
  if (!s) return null;
  try { return new URL(/^https?:\/\//i.test(s) ? s : "https://" + s); }
  catch (e) { return null; }
}
function ytParseVideoId(raw) {
  const s = (raw || "").trim();
  if (YT_VIDEO_ID.test(s)) return s;          // a bare id
  const u = ytUrlOf(s);
  if (!u) return "";
  const host = u.hostname.toLowerCase().replace(/^www\./, "");
  if (!YT_HOSTS.includes(host) && !YT_HOSTS.includes(u.hostname.toLowerCase())) return "";
  if (host === "youtu.be") {
    const id = u.pathname.split("/").filter(Boolean)[0] || "";
    return YT_VIDEO_ID.test(id) ? id : "";
  }
  const v = u.searchParams.get("v") || "";
  if (YT_VIDEO_ID.test(v)) return v;
  for (const form of YT_PATH_FORMS) {
    const at = u.pathname.indexOf(form);
    if (at >= 0) {
      const id = u.pathname.slice(at + form.length).split("/")[0];
      if (YT_VIDEO_ID.test(id)) return id;
    }
  }
  return "";
}
function ytParsePlaylistId(raw) {
  const s = (raw || "").trim();
  if (YT_PLAYLIST_ID.test(s)) return s;
  const u = ytUrlOf(s);
  if (!u) return "";
  const list = u.searchParams.get("list") || "";
  return YT_PLAYLIST_ID.test(list) ? list : "";
}
/** "" | "video" | "playlist" | "both" — "both" is a watch?v=…&list=… URL, which is
 *  genuinely ambiguous and is the only case the panel asks the user about. */
function ytKindOf(raw) {
  const v = ytParseVideoId(raw), p = ytParsePlaylistId(raw);
  if (v && p) return "both";
  if (p) return "playlist";
  if (v) return "video";
  return "";
}

function ytProgressText(d) {
  // A playlist run tags every frame with its position, so the same formatter serves
  // both routes and a single-video fetch reads exactly as it always did.
  const head = d.index ? `Fetching ${d.index}/${d.total} — ${d.title || ""}` : "";
  let phase = "Working…";
  if (d.phase === "playlist") {
    phase = d.total ? `${d.total} video(s) found` : "Reading the playlist…";
  } else if (d.phase === "cache") {
    phase = (d.need_transcript || d.need_comments)
      ? "From cache — fetching the rest…"
      : `From cache${d.comments ? ` — ${d.comments} comment(s)` : ""}`;
  } else if (d.phase === "page") {
    phase = `Fetching the video page${d.via ? " via " + d.via : ""}…`;
  } else if (d.phase === "transcript") {
    phase = d.chars ? `Transcript: ${d.chars.toLocaleString()} characters` : "No transcript found";
  } else if (d.phase === "comments") {
    phase = `Comments: ${d.done}/${d.target}…`;
  }
  return head ? `${head} · ${phase}` : phase;
}
// ---- SSE source streams ----
/** Open a GET SSE stream for any long-running source fetch. `path` is the route, `opts`
 *  the query params (booleans become 1/0). Every key in `handlers` beyond the reserved
 *  ones below is registered as a listener for the event of that name, which is what lets
 *  routes with completely different frame vocabularies share one function: a single
 *  video (progress/complete), a playlist (playlist/video/video_error/complete), an RSS
 *  feed (feed/episode/episode_error/complete) and a transcription (file/file_error).
 *
 *  Named for what it does rather than what first used it — it was `youtubeStream` while
 *  YouTube was the only caller. Returns the EventSource so the caller can close it. */
function sourceStream(path, opts, handlers) {
  const params = {};
  Object.keys(opts).forEach((k) => {
    const v = opts[k];
    if (v === undefined || v === null) return;
    params[k] = typeof v === "boolean" ? (v ? "1" : "0") : String(v);
  });
  const es = new EventSource(`${path}?${new URLSearchParams(params).toString()}`);
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
  // `start` and `error` stay hand-written below: they carry the run id and the
  // no-frame fallback, neither of which is a plain data handler.
  const RESERVED = ["started", "failed", "finally", "start", "error"];
  Object.keys(handlers).forEach((name) => {
    if (RESERVED.includes(name)) return;
    es.addEventListener(name, (ev) => {
      gotFrame = true;
      let d = {}; try { d = JSON.parse(ev.data); } catch (e) {}
      handlers[name](d);
      if (name === "complete") finish();
    });
  });
  // The stream still has to close itself if a caller doesn't care about the result.
  if (!handlers.complete) es.addEventListener("complete", () => { gotFrame = true; finish(); });
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

/* ---- RSS category/keyword filters, shared by every panel that imports a feed ----
 *
 * Four panels carry the same controls under a `<prefix>-` id convention, the way the
 * whisper checkboxes are gathered by id in updateWhisperOffers. The batch tab is not in
 * this list: its filter is PER SOURCE, so it lives in batchSourceFields and rides along
 * in the project's source dict. */
const RSS_FILTER_PANELS = ["add-rss", "lib-rss", "pe-rss", "pe-mem-rss"];

/** The filter fields of one panel, as the opts/body keys the server reads.
 *
 *  Strings, not arrays, so both transports carry the same thing: sourceStream stringifies
 *  with String(v) — which would turn an array into "a,b" only by accident — while the
 *  persona panels send a JSON body. parse_filters accepts either, and treats an empty
 *  string as "no filter", which is what an untouched box sends. */
function rssFilterOpts(prefix) {
  const val = (suffix) => ($(`${prefix}-${suffix}`)?.value || "").trim();
  return {
    categories: val("categories"),
    keywords: val("keywords"),
    exclude: val("exclude"),
    match: $(`${prefix}-match`)?.value || "any",
  };
}

/** Clear one panel's filter boxes.
 *
 *  Called from showComposerPanel and showLibPanel, which blank their URL box every time
 *  they open: there a filter left over from a previous feed would silently narrow the
 *  next, unrelated import with nothing on screen to explain it. The two persona panels
 *  deliberately don't call this — they keep their URL too, so whatever is still in the
 *  boxes is what the user can see and is about to send. */
function resetRssFilters(prefix) {
  ["categories", "keywords", "exclude"].forEach((k) => {
    const el = $(`${prefix}-${k}`); if (el) el.value = "";
  });
  const match = $(`${prefix}-match`); if (match) match.value = "any";
}

/** "12 matched of 226" — only when a filter actually narrowed something, so an unfiltered
 *  import reads exactly as it did before. */
function rssMatchText(d) {
  if (!d || !d.matched || d.matched >= (d.total_available || 0)) return "";
  return `${d.matched} matched of ${d.total_available}`;
}

let libYtES = null;
let libYtRunId = null;

/** Reveal the playlist affordances the URL actually calls for — the composer's
 *  updateYouTubePanelKind, against the library panel's ids. Shares ytKindOf, so the
 *  two tabs can never disagree about what a URL is. */
function updateLibraryYouTubePanelKind() {
  const kind = ytKindOf($("lib-yt-input").value);
  $("lib-yt-choice").classList.toggle("hidden", kind !== "both");
  const asPlaylist = libraryYouTubeIsPlaylist(kind);
  $("lib-yt-limit-wrap").classList.toggle("hidden", !asPlaylist);
  $("btn-lib-yt-fetch").textContent = asPlaylist ? "Fetch playlist" : "Fetch";
}
function libraryYouTubeIsPlaylist(kind) {
  return kind === "playlist" || (kind === "both" && $("lib-yt-kind-playlist").checked);
}

/** Dispatch a library YouTube fetch. A `list=` id means the playlist route, which
 *  appends one library item per video rather than one blob. */
async function addYouTube() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const url = $("lib-yt-input").value.trim();
  if (!url) { toast("Enter a YouTube URL"); return; }
  await flushLibrarySave();
  const opts = {
    url,
    comments: $("lib-yt-comments").checked,
    max: Math.max(5, Math.min(2000, parseInt($("lib-yt-max").value, 10) || 100)),
    refresh: $("lib-yt-refresh").checked,
  };
  const libId = S.activeLibrary.id;
  return libraryYouTubeIsPlaylist(ytKindOf(url))
    ? addYouTubePlaylist(libId, opts)
    : addYouTubeVideo(libId, opts);
}

function addYouTubeVideo(libId, opts) {
  const prog = $("lib-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Starting…";
  const btn = $("btn-lib-yt-fetch"); btn.disabled = true;
  cancelLibYouTube(true);
  libYtES = sourceStream(`/api/libraries/${libId}/add-youtube`, opts, {
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

/** Stream a whole playlist into the library, one item per video. Items are appended
 *  server-side as each lands, so the badge is refreshed per video rather than only at
 *  the end of what may be a very long run — and Cancel keeps whatever already saved. */
function addYouTubePlaylist(libId, opts) {
  opts.limit = Math.max(0, parseInt($("lib-yt-limit").value, 10) || 0);
  const prog = $("lib-yt-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the playlist…";
  const btn = $("btn-lib-yt-fetch"); btn.disabled = true;
  cancelLibYouTube(true);
  let added = 0, total = 0;
  libYtES = sourceStream(`/api/libraries/${libId}/add-youtube-playlist`, opts, {
    started: (id) => { libYtRunId = id; },
    playlist: (d) => {
      total = d.total || 0;
      prog.textContent = `Playlist: ${total} video(s) — fetching…`;
      // Enumerating is one cheap request; fetching them is not. Say so before the user
      // walks away, and point at the exit.
      if (total > 25) toast(`${total} videos queued — press Cancel to stop early.`, 6000);
    },
    progress: (d) => { prog.textContent = ytProgressText(d); },
    video: (d) => {
      added++;
      refreshLibBadge(libId);
      prog.textContent = `Added ${added}/${total} — ${d.title}`;
    },
    video_error: (d) => { toast(`Skipped ${d.title}: ${d.message}`, 5000); },
    complete: (d) => {
      adoptAddedItems(libId, d.added_items);
      updateLibraryButton();
      refreshLibBadge(libId);
      prog.textContent = `Added ${added} of ${d.total} video(s).`;
      toast(`Added ${added} video(s)` + (d.failed ? `, ${d.failed} skipped` : ""), 6000);
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 8000);
      if (added) hideLibPanels();
    },
    failed: (msg) => { toast("Playlist fetch failed: " + msg); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; libYtES = null; libYtRunId = null; },
  });
}
let libRssES = null;
let libRssRunId = null;

/** Fetch a feed into the active library, one 'rss' item per episode. */
async function addRssFeed() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  const url = $("lib-rss-input").value.trim();
  if (!url) { toast("Enter a feed URL"); return; }
  await flushLibrarySave();
  const libId = S.activeLibrary.id;
  const opts = {
    url,
    limit: Math.max(0, Math.min(500, parseInt($("lib-rss-limit").value, 10) || 0)),
    notes: $("lib-rss-notes").checked,
    whisper: $("lib-rss-whisper").checked,
    refresh: $("lib-rss-refresh").checked,
    ...rssFilterOpts("lib-rss"),
  };
  const prog = $("lib-rss-progress");
  prog.classList.remove("hidden"); prog.textContent = "Reading the feed…";
  const btn = $("btn-lib-rss-fetch"); btn.disabled = true;
  cancelLibRss(true);
  libRssES = sourceStream(`/api/libraries/${libId}/add-rss`, opts, {
    started: (id) => { libRssRunId = id; },
    feed: (d) => {
      const narrowed = rssMatchText(d);
      prog.textContent = `${d.title || "Feed"} — ${d.total} episode(s)` +
                         (narrowed ? ` · ${narrowed}` : "") +
                         (d.from_cache ? " (feed unchanged)" : "");
    },
    warning: (d) => { toast(d.message, 8000); },
    progress: (d) => { prog.textContent = rssProgressText(d); },
    // Items are appended server-side as they land, so the editor is refreshed per
    // episode rather than only at the end of what may be an hours-long run.
    episode: () => { refreshLibBadge(libId); },
    episode_error: (d) => { toast(`${d.title}: ${d.message}`, 6000); },
    complete: (d) => {
      adoptAddedItems(libId, d.added_items);
      updateLibraryButton();
      refreshLibBadge(libId);
      toast(`Added ${(d.added || []).length} episode(s) from ${d.feed_title || "the feed"}` +
            (d.failed ? ` — ${d.failed} failed` : ""));
      if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 8000);
      hideLibPanels();
    },
    failed: (msg) => { toast("Feed fetch failed: " + msg, 8000); prog.textContent = "Failed."; },
    finally: () => { btn.disabled = false; libRssES = null; libRssRunId = null; },
  });
}

function cancelLibRss(quiet) {
  const runId = libRssRunId;
  if (libRssES) { libRssES.close(); libRssES = null; }
  libRssRunId = null;
  if (runId) api("/api/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
  $("btn-lib-rss-fetch").disabled = false;
  if (!quiet) {
    $("lib-rss-progress").textContent = "Cancelled.";
    hideLibPanels();
  }
}

/** Pick audio/video files and append one transcript item each to the active library.
 *  The library twin of composerAddMediaFiles; POST for the same native-picker reason. */
async function addLibMediaFiles() {
  if (!S.activeLibrary) { toast("Select or create a library first"); return; }
  await flushLibrarySave();
  const libId = S.activeLibrary.id;
  const prog = $("lib-media-progress");
  const stagedMedia = await chooseAndStage({ accept: ACCEPT_MEDIA });
  if (stagedMedia === null) return;
  prog.classList.remove("hidden");
  prog.textContent = isLocalBrowser() ? "Waiting for file selection…" : "Transcribing…";
  const btn = $("btn-lib-add-media"); btn.disabled = true;
  try {
    await streamSSE(`/api/libraries/${libId}/add-media-files`, stagedMedia, {
      begin: (d) => {
        prog.textContent = d.total ? `Transcribing ${d.total} file(s)…` : "Nothing selected.";
      },
      progress: (d) => { prog.textContent = mediaProgressText(d); },
      file: (d) => { prog.textContent = `${d.name}: ${d.chars.toLocaleString()} characters`; },
      file_error: (d) => { toast(`${d.name}: ${d.message}`, 6000); },
      complete: (d) => {
        adoptAddedItems(libId, d.added_items);
        updateLibraryButton();
        refreshLibBadge(libId);
        if ((d.added || []).length) toast(`Added: ${d.added.join(", ")}`);
        if ((d.errors || []).length) toast("Notes: " + d.errors.join("; "), 6000);
        prog.classList.add("hidden");
        hideLibPanels();
      },
      error: (d) => { toast("Transcription failed: " + d.message, 8000); prog.textContent = "Failed."; },
      done: () => { btn.disabled = false; },
    });
  } catch (e) {
    toast("Transcription failed: " + e.message);
    prog.textContent = "Failed.";
    btn.disabled = false;
  }
}

/** Abort an in-flight YouTube fetch. Cancel used to only hide the panel, so the stream
 *  kept running and still appended the video to a library the user had walked away from.
 *  `quiet` reuses this to tear down a previous stream before starting a new one, without
 *  stamping "Cancelled." over the progress line the new run is about to write. */
function cancelLibYouTube(quiet) {
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
  if (!quiet) $("lib-yt-progress").textContent = "Cancelled.";
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
    const staged = await chooseAndStage({ accept: ACCEPT_XML, multiple: false });
    if (staged === null) return;
    const r = await api("/api/libraries/import-xml", { method: "POST", body: staged });
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
    const r = await api(`/api/libraries/${S.activeLibrary.id}/export-xml`,
      { method: "POST", body: { library: S.activeLibrary, download: !isLocalBrowser() }});
    if (takeDownload(r)) { toast("Exported"); return; }
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
  $("btn-save-network").onclick = () => saveNetwork(false);
  $("btn-restart-network").onclick = () => saveNetwork(true);
  $("btn-net-redetect").onclick = () => refreshNetworkCard(true);
  $("btn-net-show-pw").onclick = () => $("net-pw-form").classList.toggle("hidden");
  $("btn-net-change-pw").onclick = changeLoginPassword;
  $("btn-net-copy-firewall").onclick = () => {
    navigator.clipboard.writeText($("net-firewall-cmd").textContent || "");
    toast("Command copied — paste it into an administrator PowerShell");
  };
  $("set-lan-enabled").onchange = markNetworkDirty;
  $("set-net-port").oninput = markNetworkDirty;
  $("btn-yt-cache-clear").onclick = clearYouTubeCache;
  $("btn-whisper-reset").onclick = resetWhisperModel;
  $("btn-rss-cache-clear").onclick = () => clearRssCache("episodes");
  $("btn-rss-cache-clear-feeds").onclick = () => clearRssCache("feeds");
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
  $("rag-scope").onchange = () => {
    if (!S.chat) return;
    S.chat.rag_scope = $("rag-scope").value || "attachments";
    // The vector store is plaintext on disk under the default backend, so a private
    // chat is never indexed. It still works — retrieval just happens in memory each
    // send — but the user should know why it's slower and leaves no trace.
    if (S.chat.private && S.chat.rag_scope !== "attachments") {
      toast("Private chats are retrieved in memory and never indexed.", 6000);
    }
    persistChat();
  };
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
  $("btn-voice").onclick = () => onVoiceButtonClick();
  $("btn-voice-reason").onclick = () => {
    $("btn-voice-reason").classList.toggle("active");
    if (S.chat) {
      S.chat.voice_read_reasoning = $("btn-voice-reason").classList.contains("active");
      persistChat();
    }
  };
  $("btn-avatar-save").onclick = saveAvatarSettings;
  $("btn-avatar-restart").onclick = async () => {
    const btn = $("btn-avatar-restart");
    const state = $("avatar-helper-state");
    btn.disabled = true;
    if (state) state.textContent = "Restarting helper…";
    try {
      const r = await api("/api/avatar/restart", { method: "POST" });
      if (state) {
        state.textContent = r.started ? "Helper restarted." : (r.running ? "Helper is running." : "Helper did not start.");
      }
      toast("Avatar helper restarted");
    } catch (e) {
      if (state) state.textContent = e.message;
      toast(e.message, 8000);
    } finally {
      btn.disabled = false;
    }
  };
  $("set-avatar-url").oninput = () => {
    const open = $("link-avatar-settings");
    if (open) open.href = ($("set-avatar-url").value.trim() || "http://127.0.0.1:8765").replace(/\/+$/, "");
  };
  $("link-avatar-settings").onclick = () => {
    const url = ($("set-avatar-url").value.trim() || "http://127.0.0.1:8765").replace(/\/+$/, "");
    $("link-avatar-settings").href = url;
    ensureAvatarHelper().catch((err) => toast(err.message, 8000));
  };
  $("set-avatar-gate").oninput = () => {
    $("set-avatar-gate-label").textContent = $("set-avatar-gate").value;
  };
  $("set-avatar-gate").onchange = async () => {
    const gate = parseInt($("set-avatar-gate").value, 10) || 0;
    S.config.avatar_noise_gate = gate;
    try {
      const r = await api("/api/settings", { method: "POST", body: { avatar_noise_gate: gate } });
      S.config = { ...S.config, ...r.config };
    } catch (e) { /* keep the local value */ }
    await pushNoiseGate(gate);
  };
  $("set-avatar-silence").onchange = async () => {
    const seconds = Math.max(1, Math.min(30, parseInt($("set-avatar-silence").value, 10) || 6));
    $("set-avatar-silence").value = seconds;
    S.config.avatar_silence_seconds = seconds;
    try {
      const r = await api("/api/settings", { method: "POST", body: { avatar_silence_seconds: seconds } });
      S.config = { ...S.config, ...r.config };
    } catch (e) { /* keep the local value */ }
    await pushSilenceSeconds(seconds);
  };
  $("btn-avatar-browse-dir").onclick = async () => {
    try {
      const r = await api("/api/pick-folder", { method: "POST", body: { title: "Avatar Read Server folder" } });
      if (r.path) $("set-avatar-dir").value = r.path;
    } catch (e) { toast(e.message); }
  };
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
  $("btn-add-rss").onclick = () => showComposerPanel("rss");
  $("btn-add-rss-fetch").onclick = composerAddRss;
  $("btn-add-rss-cancel").onclick = () => cancelComposerRss(false);
  $("btn-add-media").onclick = composerAddMediaFiles;
  $("btn-add-search").onclick = () => showComposerPanel("search");
  $("btn-add-url-fetch").onclick = composerAddUrl;
  $("btn-add-url-cancel").onclick = hideComposerPanels;
  $("add-url-input").onkeydown = (e) => { if (e.key === "Enter") composerAddUrl(); };
  $("btn-add-yt-fetch").onclick = composerAddYouTube;
  $("btn-add-yt-cancel").onclick = () => cancelComposerYouTube();
  $("add-yt-input").onkeydown = (e) => { if (e.key === "Enter") composerAddYouTube(); };
  // Per-keystroke, because the panel reshapes itself around what the URL turns out to be.
  $("add-yt-input").addEventListener("input", updateYouTubePanelKind);
  $("add-yt-kind-video").onchange = updateYouTubePanelKind;
  $("add-yt-kind-playlist").onchange = updateYouTubePanelKind;
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
    // Not on a touch keyboard: Return there is how you type a newline, and Shift+Enter
    // does not exist. Send is a button away.
    if (e.key === "Enter" && !e.shiftKey && !isTouch()) { e.preventDefault(); sendMessage(); }
  });
  setupMessagesResizer();
  setupMobile();
  wireThreadSettings();

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
  $("btn-lib-add-rss").onclick = () => showLibPanel("rss");
  $("btn-lib-rss-fetch").onclick = addRssFeed;
  $("btn-lib-rss-cancel").onclick = () => cancelLibRss(false);
  $("btn-lib-add-media").onclick = addLibMediaFiles;
  $("btn-lib-add-search").onclick = () => showLibPanel("search");
  $("btn-lib-url-fetch").onclick = addByUrl;
  $("btn-lib-url-cancel").onclick = hideLibPanels;
  $("lib-url-input").onkeydown = (e) => { if (e.key === "Enter") addByUrl(); };
  $("btn-lib-yt-fetch").onclick = addYouTube;
  $("btn-lib-yt-cancel").onclick = () => { cancelLibYouTube(); hideLibPanels(); };
  $("lib-yt-input").onkeydown = (e) => { if (e.key === "Enter") addYouTube(); };
  // Per-keystroke so the panel reveals the playlist controls as soon as a list id
  // appears, and the radio re-decides whether Max-videos is relevant.
  $("lib-yt-input").addEventListener("input", updateLibraryYouTubePanelKind);
  $("lib-yt-kind-video").addEventListener("change", updateLibraryYouTubePanelKind);
  $("lib-yt-kind-playlist").addEventListener("change", updateLibraryYouTubePanelKind);
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
    // Then the slide-over panels, which sit below any dialog in the same visual stack.
    if (e.key === "Escape") { closeDrawers(); }
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
    const stagedEval = await chooseAndStage({ accept: kind === "csv" ? ".csv,text/csv" : ".txt,text/plain",
                                              multiple: false });
    if (stagedEval === null) return;
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
  // Wrapped so a project with several criteria scrolls the table instead of stretching
  // the card it sits in. buildRowTable does the same via .eval-rows-table.
  const wrap = document.createElement("div");
  wrap.className = "eval-table-wrap";
  wrap.appendChild(table);
  return wrap;
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
  ["batch-yt-refresh", "yt_refresh", "bool"],
  ["batch-rss-whisper", "rss_whisper", "bool"],
  ["batch-rss-notes", "rss_notes", "bool"],
  ["batch-rss-refresh", "rss_refresh", "bool"],
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
  if (k === "rss") {
    // The filter is per source, not per project: two feeds in one batch routinely want
    // different categories. Stored as plain strings, which is what parse_filters reads and
    // what batchSyncFromUI's default branch already harvests — no list handling needed.
    return `<div class="control-row">
        <input type="text" class="bsrc-field" data-key="url" style="flex:1 1 22em"
               placeholder="https://feeds.example.com/show.xml" value="${escapeHtml(src.url || "")}" />
        <label>Max episodes: <input type="number" class="bsrc-field" data-key="limit" min="0" max="500"
               style="width:6em" value="${src.limit || 0}"
               title="Newest first, in feed order — counted AFTER the filter below, so this is the newest N matching episodes. 0 = every item the feed lists. Cached episodes cost nothing, so raising this is how you pick up what's new." /></label>
      </div>
      <div class="control-row rss-filter-row">
        <span class="muted rss-filter-label">Only include:</span>
        <input type="text" class="bsrc-field" data-key="categories" style="flex:1 1 12em"
               placeholder="Categories (comma separated)" value="${escapeHtml(src.categories || "")}"
               title="Matches a WHOLE category or tag on the episode — “Arts”, not “art”. Case and punctuation are ignored. Filtering here means the episodes you skip are never downloaded or transcribed." />
        <input type="text" class="bsrc-field" data-key="keywords" style="flex:1 1 12em"
               placeholder="Keywords (anywhere in the text)" value="${escapeHtml(src.keywords || "")}"
               title="Matches anywhere in the title, categories, author or show notes. Never the transcript — reading that would cost the download and GPU time this filter exists to save." />
        <input type="text" class="bsrc-field" data-key="exclude" style="flex:1 1 8em"
               placeholder="Exclude" value="${escapeHtml(src.exclude || "")}"
               title="Skips any episode whose text matches. Beats the two boxes to the left, and works on its own." />
        <label class="rss-filter-label" title="Any: one listed term is enough. All: every listed category and every listed keyword must be present.">Match:
          <select class="bsrc-field" data-key="match">
            <option value="any"${(src.match || "any") === "any" ? " selected" : ""}>any</option>
            <option value="all"${src.match === "all" ? " selected" : ""}>all</option>
          </select>
        </label>
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
  // folder + images + media share the picker markup, so batchSyncFromUI needs no new case.
  const label = k === "images" ? "🖼 Choose image folder…"
              : k === "media" ? "🎙 Choose audio/video folder…"
              : "📂 Choose folder…";
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
           <option value="media">🎙 Folder of audio/video</option>
           <option value="youtube">▶ YouTube video(s)</option>
           <option value="playlist">▶ YouTube playlist</option>
           <option value="rss">📡 RSS / Podcast feed</option>
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
        const kind = S.batch.project.sources[idx].kind || "folder";
        const title = kind === "images" ? "Choose a folder of images to process"
                    : kind === "media" ? "Choose a folder of audio/video to transcribe"
                    : "Choose a folder of documents to process";
        if (folderPickingUnavailable()) return;
        const r = await api("/api/pick-folder", { method: "POST", body: { title } });
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
    const staged = await chooseAndStage({ accept: ACCEPT_IMAGES });
    if (staged === null) return;
    const r = await api("/api/batch/reference-images", { method: "POST", body: staged });
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
    if (folderPickingUnavailable()) return;
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
