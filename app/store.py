#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Persistence layer. User data (chats/presets/libraries) lives in data/*.json; all
configuration and secrets (API keys, servers/providers, web-search tuning) live in
the gitignored settings/settings.json. Held in memory, written through on every
mutation, guarded by a re-entrant lock (the Flask server is multi-threaded).
"""

import threading
import uuid
from datetime import datetime

from . import core, crypto, memory
# The per-profile file paths (CHATS_FILE, SETTINGS_FILE, …) are read via the core
# module at call time (core.CHATS_FILE), NOT bound here, because they are reassigned
# whenever the active profile changes. Only the profile-invariant helpers are imported.
from .core import LEGACY_CONFIG_FILE, DEFAULT_LOCAL_URL, load_json, save_json

# The built-in sidebar tab that holds the user's own chats. Always present, never
# persisted (chats with no group_id belong to it), never renamed or deleted.
DEFAULT_GROUP_ID = "default"
DEFAULT_GROUP = {"id": DEFAULT_GROUP_ID, "name": "My Chats", "builtin": True}

# Keys that must never be returned to the browser (only a has_<key> boolean is).
SECRET_KEYS = ("brave_token", "brightdata_token")

DEFAULT_SETTINGS = {
    "servers": [],                       # [{name, type, base_url, api_key}]
    "default_num_ctx": 4096,
    "last_server_url": DEFAULT_LOCAL_URL,
    "auto_detect_reasoning": False,
    "approved_domains": [],
    "restrict_to_approved": False,
    "brave_token": "",
    "brightdata_token": "",
    "brightdata_zone": "web_unlocker1",
    "max_output_tokens": 16000,          # for cloud providers (Anthropic/OpenAI)
    # Context-usage monitor: per-provider context-window size (the % bar denominator)
    # for cloud backends, which have no user-set num_ctx like Ollama. {type: tokens}.
    "provider_context_windows": {"openai": 128000, "anthropic": 200000},
    # Multi-server parallel processing (queue + batch fan-out across servers).
    "parallel_enabled": False,           # the global "Enable Parallel Processing" toggle
    "parallel_mode": "balanced",         # "balanced" (any free server pulls next) | "isolation" (1 prompt : 1 server)
    "parallel_servers": [],              # [{base_url, model}] — which servers participate + per-server model
    # RAG (Retrieval Augmented Generation): embeddings always go through this
    # dedicated Ollama endpoint, independent of the chat's provider.
    "rag_embed_server_url": DEFAULT_LOCAL_URL,
    "rag_embed_model": "nomic-embed-text",
    "rag_top_k": 6,                      # number of chunks retrieved and injected per send
    "rag_retrieval_mode": "hybrid",      # "vector" | "keyword" (BM25) | "hybrid" (both, RRF-merged)
    # Chunking for the explicit "Compile Data" step (token-accurate, structure-aware via
    # chonkie's RecursiveChunker; falls back to word windows if chonkie is unavailable).
    "rag_chunker": "recursive",          # chunker algorithm id
    "rag_chunk_size": 512,               # target chunk size (tokens for chonkie; words for fallback)
    "rag_chunk_overlap": 64,             # overlap (used by the word-window fallback chunker)
    # Compile throughput. During "Compile Data" the changed items' chunks are pooled and
    # embedded in batches (fewer, larger /api/embed calls) and optionally in parallel.
    "rag_embed_batch_size": 64,          # chunks per /api/embed request (backend-agnostic win)
    "rag_embed_concurrency": 3,          # in-flight embed requests PER SERVER. GPU: helps; CPU-only:
                                         # set to 1 (it just contends for cores). Keep <= the Ollama
                                         # server's OLLAMA_NUM_PARALLEL, else requests just queue.
    # Multi-server embedding. Deliberately its own list rather than reusing
    # parallel_servers: a box with chat models often has no embedding model pulled, and
    # RAG fan-out shouldn't be gated on the chat parallel toggle. There is no per-server
    # model — vectors from different embedding models are not comparable, so the whole
    # pool always uses rag_embed_model.
    "rag_embed_parallel": False,         # fan compile embedding out across rag_embed_servers
    "rag_embed_servers": [],             # [{base_url, enabled}] — extra embedding endpoints
    "rag_ann_enabled": True,             # DuckDB backend only: in-memory HNSW sidecar
    # Which vector store backs RAG.
    #   "lance"  — LanceDB. Durable vector index + native full-text (tantivy) index.
    #              Measured at 50k chunks x 768 dims: writes 155x faster, keyword search
    #              174x faster. NOT ENCRYPTED — chunk text and embeddings are plaintext
    #              on disk, readable independently of the login password.
    #   "duckdb" — the original store. AES-256 encrypted at rest via the login password,
    #              but no durable vector index and keyword search scans the whole scope.
    # The two stores are independent files, so switching never destroys the other's data.
    "rag_backend": "lance",
    # Contextual chunking: prepend an LLM-written situating sentence to each chunk before
    # embedding/indexing. Off by default because library indexing here is synchronous on
    # send (one LLM call per new/changed chunk); enable for higher-quality retrieval.
    "rag_contextual_chunking": False,
    "rag_context_model": "",             # local Ollama chat model for it; "" = use the chat's model
    "rag_query_rewrite": True,           # rewrite the message into 2-3 retrieval queries before RAG
    "rewrite_model": "",                 # model for query rewrite + "Rewrite" button; "" = the chat model
    # Persona pipeline / memory tuning.
    "memory_weight_influence": 0.35,     # how strongly emotional weight reorders memory retrieval (0=off)
    "pipeline_max_retries": 3,           # structured-JSON retries per llm step before pausing
    # Image attachments. Originals are always stored at full resolution; this only
    # caps the long edge of the copy that is SENT, because an unscaled phone photo
    # costs the same context as several pages of text for no extra detail the model
    # can use. Per-chat and per-batch-project toggles override it with "full res".
    "image_max_dim": 1568,               # long-edge clamp for outgoing images (px)
    "image_full_res_default": False,     # new chats start with the full-res toggle on
}

DEFAULT_PRESETS = [
    {"name": "Helpful Assistant", "prompt": "You are a helpful, concise assistant."},
    {"name": "Code Reviewer", "prompt": "You are an expert code reviewer. Point out issues, suggest improvements, and explain trade-offs."},
    {"name": "Concise", "prompt": "Answer as briefly as possible while remaining accurate."},
]

LOCAL_SERVER = {"name": "Local", "type": "ollama", "base_url": DEFAULT_LOCAL_URL, "api_key": ""}


def _normalize_server(s):
    """Coerce a server entry into {name, type, base_url, api_key}. Handles the
    legacy Ollama shape {name, url}."""
    base_url = (s.get("base_url") or s.get("url") or "").strip().rstrip("/")
    return {
        "name": (s.get("name") or "").strip() or base_url,
        "type": (s.get("type") or "ollama").strip().lower(),
        "base_url": base_url,
        "api_key": s.get("api_key") or "",
    }


class Store:
    def __init__(self):
        self._lock = threading.RLock()
        # When True (Incognito data profile), every data save_* is a no-op so nothing
        # is written to disk. Settings saves (save_config) are unaffected — the active
        # settings profile keeps persisting.
        self.incognito = False
        # Encryption gate: data files are AES-encrypted at rest and can only be read
        # once the user has logged in (which activates the Data Encryption Key). At
        # construction the app is still LOCKED, so start empty and DON'T touch disk —
        # the server calls reload_settings()/reload_data() right after a successful
        # login, which populates everything (and runs the one-time migrations then,
        # under the active key). Reading/writing here while locked would either fail
        # to decrypt or clobber encrypted data with plaintext defaults.
        if crypto.is_unlocked():
            self.config = self._load_settings()
            self._load_data_collections()
            self._apply_tokens()
        else:
            self.config = dict(DEFAULT_SETTINGS)
            self._empty_collections()

    def _empty_collections(self):
        """Start with empty in-memory collections (used while locked, before login)."""
        self.chats = []
        self.chat_groups = []
        self.presets = []
        self.prompts = None
        self.libraries = []
        self.evals = []
        self.batch_projects = []
        self.db_projects = []
        self.context_history = {}
        self.memory_cores = []

    def _load_data_collections(self):
        """(Re)load every data collection from the CURRENT core.* profile paths and run
        the one-time backfills/migrations. Used at boot and on a data-profile switch."""
        self.chats = load_json(core.CHATS_FILE, [])
        self.chat_groups = load_json(core.CHAT_GROUPS_FILE, [])  # imported-chat tabs: [{id, name}]
        self.presets = load_json(core.PRESETS_FILE, [])
        self.prompts = load_json(core.PROMPTS_FILE, None)
        self.libraries = load_json(core.LIBRARIES_FILE, [])
        self.evals = load_json(core.EVALS_FILE, [])
        self.batch_projects = load_json(core.BATCH_PROJECTS_FILE, [])
        self.db_projects = load_json(core.DB_PROJECTS_FILE, [])
        # Per-chat context-usage history: {chat_id: [ContextHistoryEntry, ...]}.
        self.context_history = load_json(core.CONTEXT_HISTORY_FILE, {})
        # User memory cores. Normalized on load so a hand-edited file can't reach the
        # injector or the Memory tab in a bad shape.
        self.memory_cores = [memory.normalize_core(c)
                             for c in load_json(core.MEMORY_CORES_FILE, [])
                             if isinstance(c, dict)]

        # One-time migration: backfill stable per-item ids on existing libraries so
        # the RAG store keys embeddings by identity (survives reorder/mid-list edits).
        # Normalize every library (no short-circuit) before deciding to persist.
        if len([1 for l in self.libraries if core.ensure_item_ids(l)]) > 0:
            self.save_libraries()

        if not self.presets:
            self.presets = [dict(p) for p in DEFAULT_PRESETS]
            self.save_presets()

        # One-time migration: seed the System/Pre-prompt library from the flat presets.
        if self.prompts is None:
            self.prompts = self._migrate_prompts_from_presets()
            self.save_prompts()

        # Backfill the new system_prompt/pre_prompt toggle fields on legacy chats, and
        # drop library_ids pointing at libraries that no longer exist (deletions before
        # prune_library_refs left them behind, pinning RAG on with nothing to retrieve).
        dirty = len([1 for c in self.chats if self._ensure_prompt_fields(c)]) > 0
        if self._prune_orphan_library_refs():
            dirty = True
        if dirty:
            self.save_chats()

    def _apply_tokens(self):
        core.configure_tokens(
            self.config.get("brave_token"),
            self.config.get("brightdata_token"),
            self.config.get("brightdata_zone"),
        )

    def reload_data(self):
        """Re-read all data collections after the active data profile changed."""
        with self._lock:
            self._load_data_collections()

    def reload_settings(self):
        """Re-read config after the active settings profile changed."""
        with self._lock:
            self.config = self._load_settings()
            self._apply_tokens()

    def flush_all(self):
        """Force-write every data collection to the current core.* paths, ignoring the
        incognito flag. Used when an incognito session is saved to a real profile."""
        with self._lock:
            save_json(core.CHATS_FILE, self.chats)
            save_json(core.CHAT_GROUPS_FILE, self.chat_groups)
            save_json(core.PRESETS_FILE, self.presets)
            save_json(core.PROMPTS_FILE, self.prompts)
            save_json(core.LIBRARIES_FILE, self.libraries)
            save_json(core.EVALS_FILE, self.evals)
            save_json(core.BATCH_PROJECTS_FILE, self.batch_projects)
            save_json(core.DB_PROJECTS_FILE, self.db_projects)
            save_json(core.CONTEXT_HISTORY_FILE, self.context_history)
            save_json(core.MEMORY_CORES_FILE, self.memory_cores)

    @staticmethod
    def _merge_images(chats, batch_projects, dest_dir):
        """Copy every image file the given chats/projects reference into ``dest_dir``.

        Read + write go through core.read_bytes/write_bytes rather than a file copy so
        the bytes land re-encrypted under the same key rather than being moved as
        opaque ciphertext — the two profiles share the app key, but going through the
        helpers is what keeps this correct if that ever stops being true."""
        from pathlib import Path
        from . import images as images_mod
        live = images_mod.collect_ids(chats, batch_projects)
        if not live:
            return
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for image_id in live:
            try:
                data, _mt = images_mod.load(image_id)
            except Exception:
                continue          # already gone: the chat keeps a dead reference, not a crash
            core.write_bytes(dest / f"{image_id}.bin", data)

    def merge_into(self, target_dir):
        """Merge the current in-memory (incognito) data INTO the profile at target_dir
        without overwriting anything there, leaving the merged result in memory (the
        caller then repoints core at target_dir and calls flush_all). Chats land in the
        target's default tab with fresh ids; presets/libraries/evals append (fresh ids
        on clash); prompt trees merge by name. Incognito database sessions are not
        merged — their staging files + vault live only in the private scratch."""
        from pathlib import Path
        target_dir = Path(target_dir)
        with self._lock:
            inco_chats = list(self.chats)
            inco_presets = list(self.presets)
            inco_prompts = self.prompts or {"system": [], "pre": []}
            inco_libs = list(self.libraries)
            inco_evals = list(self.evals)
            inco_batch = list(self.batch_projects)
            inco_cores = list(self.memory_cores)

            # ---- images: copy the bytes across before the records travel ----
            # Image records reference a file in the ACTIVE profile's image dir, which
            # is about to stop being this one. Without this, every merged chat would
            # come out of incognito pointing at pictures that no longer exist. Ids are
            # uuids, so they can be reused verbatim in the target.
            self._merge_images(inco_chats, inco_batch, target_dir / "images")

            # ---- chats: append into the target's default tab with fresh ids ----
            self.chats = load_json(target_dir / "chats.json", [])
            existing_ids = {c.get("id") for c in self.chats}
            for src in reversed(inco_chats):
                c = dict(src)
                c["id"] = uuid.uuid4().hex[:12]
                c.pop("group_id", None)   # target may not have the same tabs
                c["private"] = False
                existing_ids.add(c["id"])
                self.chats.insert(0, c)
            self.chat_groups = load_json(target_dir / "chat_groups.json", [])

            # ---- presets: union by name ----
            self.presets = load_json(target_dir / "presets.json", [])
            have = {p.get("name") for p in self.presets}
            for p in inco_presets:
                if p.get("name") not in have:
                    self.presets.append(dict(p))
                    have.add(p.get("name"))

            # ---- prompts: reuse the name-aware tree merge ----
            self.prompts = load_json(target_dir / "prompts.json", None) or {"system": [], "pre": []}
            self.merge_prompts_tree("system", inco_prompts.get("system") or [])
            self.merge_prompts_tree("pre", inco_prompts.get("pre") or [])

            # ---- libraries / evals: append with fresh ids ----
            self.libraries = load_json(target_dir / "libraries.json", [])
            for lib in inco_libs:
                lib = dict(lib)
                lib["id"] = uuid.uuid4().hex[:12]
                core.ensure_item_ids(lib)
                self.libraries.append(lib)
            self.evals = load_json(target_dir / "evals.json", [])
            ev_ids = {e.get("id") for e in self.evals}
            for ev in inco_evals:
                ev = dict(ev)
                if ev.get("id") in ev_ids:
                    ev["id"] = uuid.uuid4().hex[:12]
                ev_ids.add(ev.get("id"))
                self.evals.insert(0, ev)

            # ---- batch projects: append with fresh ids (same rule as evals) ----
            self.batch_projects = load_json(target_dir / "batch_projects.json", [])
            bp_ids = {b.get("id") for b in self.batch_projects}
            for bp in inco_batch:
                bp = dict(bp)
                if bp.get("id") in bp_ids:
                    bp["id"] = uuid.uuid4().hex[:12]
                bp_ids.add(bp.get("id"))
                self.batch_projects.insert(0, bp)

            # ---- memory cores: append with fresh ids, disambiguating names ----
            self.memory_cores = [memory.normalize_core(c) for c
                                 in load_json(target_dir / "memory_cores.json", [])
                                 if isinstance(c, dict)]
            have_names = {c.get("name") for c in self.memory_cores}
            for src in inco_cores:
                mc = memory.normalize_core(src)
                mc["id"] = uuid.uuid4().hex[:12]
                if mc["name"] in have_names:
                    mc["name"] = f"{mc['name']} (incognito)"
                have_names.add(mc["name"])
                self.memory_cores.append(mc)

            # db_projects: keep the target's as-is (incognito DB sessions aren't merged).
            self.db_projects = load_json(target_dir / "db_projects.json", [])

    def _load_settings(self):
        """Load the active settings profile's settings.json, migrating from the legacy
        data/config.json the first time (servers coerced to the new shape). A fresh
        profile starts with empty API keys — the user adds them in the Settings tab."""
        if core.SETTINGS_FILE.exists():
            cfg = load_json(core.SETTINGS_FILE, dict(DEFAULT_SETTINGS))
        else:
            legacy = load_json(LEGACY_CONFIG_FILE, {})
            cfg = {**DEFAULT_SETTINGS, **legacy}
        # Which keys the user's file actually carried, BEFORE defaults are filled in.
        # Lets a newly-introduced setting distinguish "never chosen" from "chose the
        # default" — e.g. rag_backend, where an upgrade must not switch an existing
        # corpus out from under the user (see server._apply_rag_backend).
        self._explicit_keys = set(cfg.keys())
        for k, v in DEFAULT_SETTINGS.items():
            cfg.setdefault(k, v)
        cfg["servers"] = [_normalize_server(s) for s in cfg.get("servers", [])]
        if not core.SETTINGS_FILE.exists():
            save_json(core.SETTINGS_FILE, cfg)
        return cfg

    def config_has_explicit(self, key: str) -> bool:
        """True when ``key`` was present in the user's settings file, as opposed to
        being filled in from DEFAULT_SETTINGS."""
        return key in getattr(self, "_explicit_keys", set())

    # ---------------- low-level saves ----------------
    # Data saves become no-ops in Incognito (kept in memory only); settings persist.
    def save_config(self):
        with self._lock:
            save_json(core.SETTINGS_FILE, self.config)

    def save_chats(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.CHATS_FILE, self.chats)

    def save_chat_groups(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.CHAT_GROUPS_FILE, self.chat_groups)

    def save_context_history(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.CONTEXT_HISTORY_FILE, self.context_history)

    def save_presets(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.PRESETS_FILE, self.presets)

    def save_prompts(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.PROMPTS_FILE, self.prompts)

    def save_libraries(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.LIBRARIES_FILE, self.libraries)

    def save_evals(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.EVALS_FILE, self.evals)

    def save_batch_projects(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.BATCH_PROJECTS_FILE, self.batch_projects)

    def save_db_projects(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.DB_PROJECTS_FILE, self.db_projects)

    def save_memory_cores(self):
        with self._lock:
            if self.incognito:
                return
            save_json(core.MEMORY_CORES_FILE, self.memory_cores)

    # ---------------- settings (masked view) ----------------
    def masked_config(self):
        """Config safe to send to the browser: secrets replaced by has_<key> flags."""
        with self._lock:
            out = {k: v for k, v in self.config.items() if k not in SECRET_KEYS}
            for k in SECRET_KEYS:
                out[f"has_{k}"] = bool(self.config.get(k))
            # Never leak server api_keys either.
            out["servers"] = [
                {"name": s["name"], "type": s["type"], "base_url": s["base_url"],
                 "has_key": bool(s.get("api_key"))}
                for s in self.config.get("servers", [])
            ]
            return out

    # ---------------- servers ----------------
    def all_servers(self):
        """Full server list (Local first) with credentials — server-side only."""
        with self._lock:
            return [dict(LOCAL_SERVER)] + [dict(s) for s in self.config.get("servers", [])]

    def server_choices(self):
        """[{label, url, type}] for the dropdown (Local first). url == base_url."""
        with self._lock:
            out = []
            for s in self.all_servers():
                label = s["name"] if s["type"] == "ollama" and s["base_url"] == DEFAULT_LOCAL_URL \
                    else f"{s['name']} ({s['type']})"
                out.append({"label": label, "url": s["base_url"], "type": s["type"]})
            return out

    def resolve_server(self, base_url):
        """Return the full server entry (type + api_key) for a base_url, defaulting
        to Local Ollama. Existing chats store an Ollama url here, so anything
        unmatched is treated as an Ollama server at that url."""
        base_url = (base_url or DEFAULT_LOCAL_URL).rstrip("/")
        with self._lock:
            for s in self.all_servers():
                if s["base_url"].rstrip("/") == base_url:
                    return dict(s)
        return {"name": base_url, "type": "ollama", "base_url": base_url, "api_key": ""}

    def set_servers(self, servers):
        with self._lock:
            self.config["servers"] = [
                _normalize_server(s) for s in (servers or []) if (s.get("base_url") or s.get("url"))
            ]
            self.save_config()
            return self.config["servers"]

    # ---------------- parallel (multi-server) config ----------------
    def set_parallel_config(self, patch: dict):
        """Update the multi-server parallel-processing settings. Accepts any of
        parallel_enabled / parallel_mode / parallel_servers; ignores unknown keys."""
        with self._lock:
            if "parallel_enabled" in patch:
                self.config["parallel_enabled"] = bool(patch["parallel_enabled"])
            if "parallel_mode" in patch:
                mode = str(patch["parallel_mode"]).lower()
                self.config["parallel_mode"] = mode if mode in ("balanced", "isolation") else "balanced"
            if "parallel_servers" in patch:
                clean = []
                for s in (patch.get("parallel_servers") or []):
                    base = (s.get("base_url") or "").strip().rstrip("/")
                    if not base:
                        continue
                    clean.append({"base_url": base, "model": (s.get("model") or "").strip()})
                self.config["parallel_servers"] = clean
            self.save_config()
            return {
                "parallel_enabled": self.config.get("parallel_enabled", False),
                "parallel_mode": self.config.get("parallel_mode", "balanced"),
                "parallel_servers": self.config.get("parallel_servers", []),
            }

    # ---------------- config ----------------
    def update_config(self, patch: dict):
        with self._lock:
            # Ignore empty-string secrets so a blank field doesn't wipe a saved key.
            clean = {}
            for k, v in (patch or {}).items():
                if k in SECRET_KEYS and v == "":
                    continue
                clean[k] = v
            self.config.update(clean)
            self.save_config()
            if any(k in clean for k in ("brave_token", "brightdata_token", "brightdata_zone")):
                core.configure_tokens(
                    self.config.get("brave_token"),
                    self.config.get("brightdata_token"),
                    self.config.get("brightdata_zone"),
                )
            return self.config

    # ---------------- chats ----------------
    def chat_summaries(self):
        with self._lock:
            return [
                {"id": c["id"], "title": c.get("title", "New Chat"),
                 "updated": c.get("updated", ""), "private": bool(c.get("private", False)),
                 "group_id": c.get("group_id") or DEFAULT_GROUP_ID}
                for c in self.chats
            ]

    def get_chat(self, chat_id):
        with self._lock:
            for c in self.chats:
                if c.get("id") == chat_id:
                    return c
            return None

    def add_chat(self, chat):
        with self._lock:
            self.chats.insert(0, chat)
            self.save_chats()
            return chat

    def upsert_chat(self, chat):
        with self._lock:
            existing = self.get_chat(chat.get("id"))
            if existing is None:
                self.chats.insert(0, chat)
            else:
                idx = self.chats.index(existing)
                self.chats[idx] = chat
            self.save_chats()
            return chat

    def delete_chat(self, chat_id):
        with self._lock:
            before = len(self.chats)
            self.chats = [c for c in self.chats if c.get("id") != chat_id]
            self.save_chats()
            return len(self.chats) < before

    # ---------------- chat groups (sidebar tabs) ----------------
    def groups(self):
        """All sidebar tabs: the built-in default first, then imported-chat tabs."""
        with self._lock:
            return [dict(DEFAULT_GROUP)] + [dict(g) for g in self.chat_groups]

    def add_group(self, name):
        """Create a new (renameable) tab and return it."""
        with self._lock:
            group = {"id": uuid.uuid4().hex[:12], "name": (name or "Imported").strip() or "Imported"}
            self.chat_groups.append(group)
            self.save_chat_groups()
            return dict(group)

    def rename_group(self, group_id, name):
        with self._lock:
            if group_id == DEFAULT_GROUP_ID:
                return None  # the built-in tab can't be renamed
            for g in self.chat_groups:
                if g.get("id") == group_id:
                    g["name"] = (name or "").strip() or g.get("name", "Imported")
                    self.save_chat_groups()
                    return dict(g)
            return None

    def delete_group(self, group_id):
        """Remove a tab and every chat that belongs to it. The built-in tab is
        protected. Returns True if a tab was removed."""
        with self._lock:
            if group_id == DEFAULT_GROUP_ID:
                return False
            before = len(self.chat_groups)
            self.chat_groups = [g for g in self.chat_groups if g.get("id") != group_id]
            if len(self.chat_groups) == before:
                return False
            self.chats = [c for c in self.chats if (c.get("group_id") or DEFAULT_GROUP_ID) != group_id]
            self.save_chat_groups()
            self.save_chats()
            return True

    # ---------------- chat export / import ----------------
    # Total base64 an export will carry before it stops embedding images. An export
    # is a plaintext JSON file meant to be moved between machines; past a few hundred
    # megabytes it stops being one.
    EXPORT_IMAGE_BUDGET = 200 * 1024 * 1024

    def _export_images(self, chats):
        """Embed the images the exported chats reference, as {id: {…record, data}}.

        Ids alone would be useless on another machine — the bytes live in this
        profile's image dir. Returns (map, warnings)."""
        import base64
        from . import images as images_mod
        out, warnings, spent = {}, [], 0
        for image_id in sorted(images_mod.collect_ids(chats)):
            try:
                data, media_type = images_mod.load(image_id)
            except Exception:
                continue          # a dead reference exports as a dead reference
            encoded = base64.b64encode(data).decode("ascii")
            if spent + len(encoded) > self.EXPORT_IMAGE_BUDGET:
                warnings.append(
                    "Some images were left out: the export reached its "
                    f"{self.EXPORT_IMAGE_BUDGET // (1024 * 1024)} MB image limit.")
                break
            spent += len(encoded)
            out[image_id] = {"media_type": media_type, "data": encoded}
        return out, warnings

    def export_chats(self, ids=None):
        """Return an export envelope. ids=None exports every saved chat; otherwise
        only the listed ids. Chats never hold secrets, and group_id (a local tab
        reference) is dropped — grouping is decided at import time.

        Any attached or generated images travel inside the envelope as base64: the
        file has to open on another machine, where this profile's image store does
        not exist."""
        with self._lock:
            if ids is None:
                selected = list(self.chats)
            else:
                idset = set(ids)
                selected = [c for c in self.chats if c.get("id") in idset]
            out = []
            for c in selected:
                copy = dict(c)
                copy.pop("group_id", None)
                copy["private"] = False
                out.append(copy)
            image_map, warnings = self._export_images(out)
            envelope = {
                "app": "LocalStreamingToken",
                "kind": "chats",
                "version": 1,
                "exported": datetime.utcnow().isoformat(),
                "chats": out,
            }
            if image_map:
                envelope["images"] = image_map
            if warnings:
                envelope["warnings"] = warnings
            return envelope

    def import_chats(self, envelope, group_name):
        """Import chats from an export envelope into a new sidebar tab. Each chat
        gets a fresh id (so existing chats are never overwritten) and is assigned
        to the new group. Returns (group, imported_count)."""
        with self._lock:
            chats = (envelope or {}).get("chats")
            if not isinstance(chats, list):
                raise ValueError("Not a valid chat export file.")
            group = self.add_group(group_name)
            # Write any embedded images into THIS profile's store first, keeping a
            # map from the file's ids to the fresh ones, so the references rewritten
            # below point at bytes that actually exist here.
            id_map = self._import_images((envelope or {}).get("images"))
            count = 0
            # Insert preserving the file's order (first chat ends up on top).
            for src in reversed(chats):
                if not isinstance(src, dict):
                    continue
                chat = dict(src)
                chat["id"] = uuid.uuid4().hex[:12]
                chat["group_id"] = group["id"]
                chat["private"] = False
                chat.setdefault("messages", [])
                chat.setdefault("title", "Imported Chat")
                if id_map:
                    self._remap_image_ids(chat, id_map)
                self.chats.insert(0, chat)
                count += 1
            self.save_chats()
            return group, count

    @staticmethod
    def _import_images(image_map):
        """Store an envelope's embedded images here. Returns {old_id: new_id}.
        Fresh ids on purpose: an import must never overwrite an image this profile
        already has under the same id."""
        import base64
        from . import images as images_mod
        out = {}
        for old_id, entry in (image_map or {}).items():
            if not isinstance(entry, dict):
                continue
            try:
                data = base64.b64decode(entry.get("data") or "", validate=False)
                if not data:
                    continue
                rec = images_mod.store(data, entry.get("media_type") or "image/png",
                                       name=entry.get("name") or "",
                                       origin=entry.get("origin") or "user")
            except Exception:
                continue
            out[old_id] = rec["id"]
        return out

    @staticmethod
    def _remap_image_ids(chat, id_map):
        """Point a freshly imported chat's image references at the new local ids.

        Rebuilds rather than mutating in place: the chat dict is a shallow copy of the
        envelope's, so editing a message would reach back into the file's own data.
        An id with no mapping is left alone rather than dropped — the reference is
        already dead, and quietly editing the message would hide that."""
        def remap(ref):
            if isinstance(ref, dict) and ref.get("id") in id_map:
                return {**ref, "id": id_map[ref["id"]]}
            return ref

        msgs = []
        for msg in chat.get("messages") or []:
            if isinstance(msg, dict) and msg.get("images"):
                msg = {**msg, "images": [remap(r) for r in msg["images"]]}
            msgs.append(msg)
        chat["messages"] = msgs
        if chat.get("attachments"):
            chat["attachments"] = [remap(a) for a in chat["attachments"]]

    # ---------------- memory cores ----------------
    # Cores are small (a few dozen short entries), so every mutation rewrites the whole
    # file, exactly like libraries. The core dicts themselves are shaped by app.memory.
    def add_memory_core(self, name):
        with self._lock:
            mc = memory.new_core(name)
            self.memory_cores.append(mc)
            self.save_memory_cores()
            return mc

    def get_memory_core(self, core_id):
        with self._lock:
            for mc in self.memory_cores:
                if mc.get("id") == core_id:
                    return mc
            return None

    def memory_cores_snapshot(self):
        """A shallow copy of the core list for read paths (generation, /api/state) that
        must not observe the list mid-rewrite while a pass or a delete is running."""
        with self._lock:
            return list(self.memory_cores)

    def mutate_memory_core(self, core_id, fn):
        """Apply ``fn(core)`` to a core under the lock, then save. Returns
        ``(core, result)``, or ``(None, None)`` if the core no longer exists.

        Extraction passes used to hold a core they resolved at request start and mutate
        it minutes later with no lock, racing every entry edit and every other pass. The
        core must be re-resolved by id *inside* the lock each time, because a delete
        between resolve and write would otherwise have the pass writing into an orphan.
        The slow part — the LLM call — stays outside; only applying its result is
        serialised."""
        with self._lock:
            mc = self.get_memory_core(core_id)
            if not mc:
                return None, None
            result = fn(mc)
            self.save_memory_cores()
            return mc, result

    def update_memory_core(self, core_id, patch):
        """Patch a core's name and per-core tuning. Entries are never touched here —
        they go through the entry helpers or an extraction pass."""
        with self._lock:
            mc = self.get_memory_core(core_id)
            if not mc:
                return None
            for key in ("name", "auto_extract", "extract_every", "inject_limit"):
                if key in (patch or {}):
                    mc[key] = patch[key]
            # Re-validate the scalars only; the live entries list is left alone (it was
            # normalized on load and is maintained by the entry helpers).
            clean = memory.normalize_core({**mc, "entries": []})
            for key in ("name", "auto_extract", "extract_every", "inject_limit"):
                mc[key] = clean[key]
            memory.touch(mc)
            self.save_memory_cores()
            return mc

    def delete_memory_core(self, core_id):
        """Delete a core and scrub every chat that pointed at it. Leaving the id behind
        keeps those chats reporting memory_enabled with a core that no longer resolves —
        the 🧠 toggle reads on while nothing is ever injected, and only the chat open at
        the time would have been repaired by the browser."""
        with self._lock:
            before = len(self.memory_cores)
            self.memory_cores = [c for c in self.memory_cores if c.get("id") != core_id]
            if len(self.memory_cores) == before:
                return False
            self.save_memory_cores()
            touched = False
            for chat in self.chats:
                if chat.get("memory_core_id") == core_id:
                    chat["memory_core_id"] = ""
                    chat["memory_enabled"] = False
                    chat["memory_turns_since"] = 0
                    touched = True
            if touched:
                self.save_chats()
            return True

    def upsert_memory_entry(self, core_id, entry):
        """Add or edit one entry. An entry with no id is created; entries created this
        way are the user's own, so they carry origin='user' and are protected from
        automated deletion.

        Returns ``(entry, '')`` on success or ``(None, reason)`` — the reasons are
        distinct so the caller can say which went wrong. An edit naming an id that is
        gone reports 'gone' rather than silently reappearing as a new user entry, which
        is what a second tab deleting it out from under this one looks like."""
        with self._lock:
            mc = self.get_memory_core(core_id)
            if not mc:
                return None, "core"
            entry = entry or {}
            wanted_id = entry.get("id") or ""
            existing = None
            for e in mc.get("entries", []):
                if e.get("id") and e.get("id") == wanted_id:
                    existing = e
                    break
            if existing:
                saved = memory.apply_entry_patch(existing, entry)
            elif wanted_id:
                return None, "gone"
            else:
                saved = memory.new_entry(entry.get("text", ""), entry.get("category"),
                                         entry.get("importance", 5), origin="user",
                                         pinned=bool(entry.get("pinned")))
                if not saved["text"]:
                    return None, "text"
                mc.setdefault("entries", []).append(saved)
            memory.touch(mc)
            self.save_memory_cores()
            return saved, ""

    def delete_memory_entry(self, core_id, entry_id):
        with self._lock:
            mc = self.get_memory_core(core_id)
            if not mc:
                return False
            entries = mc.setdefault("entries", [])
            before = len(entries)
            # Mutate in place rather than rebinding: an extraction pass that resolved
            # this list moments ago would otherwise be appending to a detached object,
            # and its new memories would vanish on the next save.
            entries[:] = [e for e in entries if e.get("id") != entry_id]
            if len(entries) == before:
                return False
            memory.touch(mc)
            self.save_memory_cores()
            return True

    # ---------------- memory core export / import ----------------
    def export_memory_cores(self, ids=None):
        """Export envelope for one or more cores. Cores hold no secrets; ids are kept
        as-is here and reassigned at import time."""
        with self._lock:
            if ids is None:
                selected = list(self.memory_cores)
            else:
                idset = set(ids)
                selected = [c for c in self.memory_cores if c.get("id") in idset]
            return {
                "app": "LocalStreamingToken",
                "kind": "memory_cores",
                "version": 1,
                "exported": datetime.utcnow().isoformat(),
                "cores": [dict(c) for c in selected],
            }

    def import_memory_cores(self, envelope):
        """Import cores from an export envelope. Each core (and every entry in it) gets
        a fresh id so an existing core is never overwritten, and a clashing name is
        suffixed. Returns (imported_cores, count)."""
        with self._lock:
            cores = (envelope or {}).get("cores")
            if not isinstance(cores, list):
                raise ValueError("Not a valid memory core export file.")
            have_names = {c.get("name") for c in self.memory_cores}
            imported = []
            for src in cores:
                if not isinstance(src, dict):
                    continue
                mc = memory.normalize_core(src)
                mc["id"] = uuid.uuid4().hex[:12]
                for e in mc["entries"]:
                    e["id"] = uuid.uuid4().hex[:12]
                if mc["name"] in have_names:
                    mc["name"] = f"{mc['name']} (imported)"
                have_names.add(mc["name"])
                self.memory_cores.append(mc)
                imported.append(mc)
            if imported:
                self.save_memory_cores()
            return imported, len(imported)

    # ---------------- presets ----------------
    def set_presets(self, presets):
        with self._lock:
            self.presets = presets
            self.save_presets()
            return self.presets

    def upsert_preset(self, name, prompt):
        with self._lock:
            for p in self.presets:
                if p.get("name") == name:
                    p["prompt"] = prompt
                    break
            else:
                self.presets.append({"name": name, "prompt": prompt})
            self.save_presets()
            return self.presets

    def delete_preset(self, name):
        with self._lock:
            self.presets = [p for p in self.presets if p.get("name") != name]
            self.save_presets()
            return self.presets

    # ---------------- prompt library (System/Pre) ----------------
    def _migrate_prompts_from_presets(self):
        """Build the initial {system, pre} tree, dropping every existing flat preset
        into System > Imported > Uncategorized (they were used as system-style prompts)."""
        cat = core.new_prompt_category("Uncategorized")
        cat["prompts"] = [core.new_prompt_item(p.get("name") or "Untitled", p.get("prompt") or "")
                          for p in (self.presets or [])]
        group = core.new_prompt_group("Imported")
        group["categories"] = [cat]
        return {"system": [group], "pre": []}

    @staticmethod
    def _ensure_prompt_fields(chat):
        """Backfill/migrate the independent system_prompt + pre_prompt toggle fields on a
        chat. Legacy chats carried a single ``pre_prompt`` + ``pre_as_system`` flag; map
        that onto the new model. Returns True if the chat was mutated."""
        if "system_on" in chat and "pre_on" in chat and "system_prompt" in chat:
            return False
        pre = (chat.get("pre_prompt") or "")
        pre_as_system = bool(chat.get("pre_as_system", True))
        chat.setdefault("system_prompt", "")
        if pre.strip() and pre_as_system:
            # Was being sent as a system message -> becomes the System prompt.
            chat["system_prompt"] = pre
            chat["system_on"] = True
            chat["pre_prompt"] = ""
            chat["pre_on"] = False
        else:
            chat.setdefault("system_prompt", "")
            chat["system_on"] = bool(chat.get("system_prompt", "").strip())
            chat["pre_on"] = bool(pre.strip())
        return True

    def set_prompts(self, prompts):
        """Whole-tree save from the client (mirrors how chats are persisted)."""
        with self._lock:
            sys_t = (prompts or {}).get("system") or []
            pre_t = (prompts or {}).get("pre") or []
            self.prompts = {"system": list(sys_t), "pre": list(pre_t)}
            self.save_prompts()
            return self.prompts

    def merge_prompts_tree(self, kind, groups):
        """Merge imported ``groups`` into the ``kind`` tree. Ancestor groups/categories are
        matched by name (find-or-create); leaf prompts that clash are auto-renamed
        'name (2)', '(3)'… so nothing is ever overwritten."""
        with self._lock:
            kind = "pre" if kind == "pre" else "system"
            tree = self.prompts.setdefault(kind, [])
            for ig in groups or []:
                g = next((x for x in tree if x.get("name") == ig.get("name")), None)
                if g is None:
                    g = core.new_prompt_group(ig.get("name") or "Imported Group")
                    tree.append(g)
                for ic in ig.get("categories", []) or []:
                    c = next((x for x in g["categories"] if x.get("name") == ic.get("name")), None)
                    if c is None:
                        c = core.new_prompt_category(ic.get("name") or "Uncategorized")
                        g["categories"].append(c)
                    existing = {p.get("name") for p in c["prompts"]}
                    for ip in ic.get("prompts", []) or []:
                        name = ip.get("name") or "Untitled"
                        if name in existing:
                            base, n = name, 2
                            while f"{base} ({n})" in existing:
                                n += 1
                            name = f"{base} ({n})"
                        existing.add(name)
                        c["prompts"].append(core.new_prompt_item(name, ip.get("prompt") or ""))
            self.save_prompts()
            return self.prompts

    def slice_prompts(self, kind, group_id, category_id=None, prompt_id=None):
        """Return a list of group dicts representing the most-specific selected subtree
        (a single prompt, a category, or a whole group), each carrying its ancestor path."""
        kind = "pre" if kind == "pre" else "system"
        tree = (self.prompts or {}).get(kind, [])
        g = next((x for x in tree if x.get("id") == group_id), None)
        if not g:
            return []
        if not category_id:
            return [g]
        c = next((x for x in g.get("categories", []) if x.get("id") == category_id), None)
        if not c:
            return []
        if not prompt_id:
            cat = {"id": c["id"], "name": c["name"], "prompts": c.get("prompts", [])}
            return [{"id": g["id"], "name": g["name"], "categories": [cat]}]
        p = next((x for x in c.get("prompts", []) if x.get("id") == prompt_id), None)
        if not p:
            return []
        cat = {"id": c["id"], "name": c["name"], "prompts": [p]}
        return [{"id": g["id"], "name": g["name"], "categories": [cat]}]

    # ---------------- libraries ----------------
    def get_library(self, lib_id):
        with self._lock:
            for l in self.libraries:
                if l.get("id") == lib_id:
                    return l
            return None

    def add_library(self, lib):
        with self._lock:
            core.ensure_item_ids(lib)
            self.libraries.append(lib)
            self.save_libraries()
            return lib

    def upsert_library(self, lib):
        with self._lock:
            core.ensure_item_ids(lib)  # items created in the frontend arrive id-less
            existing = self.get_library(lib.get("id"))
            if existing is None:
                self.libraries.append(lib)
            else:
                idx = self.libraries.index(existing)
                self.libraries[idx] = lib
            self.save_libraries()
            return lib

    def append_library_items(self, lib_id, items):
        """Append items to the LIVE library under the lock and persist.

        The long-running add routes (file parse, URL scrape, YouTube, Brave crawl) run
        for minutes. Holding the dict they read at request start and upserting it at the
        end silently discarded any autosave PUT that landed in between, because
        ``upsert_library`` REPLACES the list entry and detaches every held reference.
        Re-reading here keeps the append atomic against concurrent edits.

        Returns the updated library, or None if it was deleted mid-flight."""
        with self._lock:
            lib = self.get_library(lib_id)
            if lib is None:
                return None
            for it in items:
                if not it.get("id"):
                    it["id"] = uuid.uuid4().hex[:12]
            lib.setdefault("items", []).extend(items)
            lib["updated"] = datetime.utcnow().isoformat()
            self.save_libraries()
            return lib

    def delete_library(self, lib_id):
        with self._lock:
            before = len(self.libraries)
            self.libraries = [l for l in self.libraries if l.get("id") != lib_id]
            self.save_libraries()
            self.prune_library_refs(lib_id)
            return len(self.libraries) < before

    def prune_library_refs(self, lib_id):
        """Drop a library id from every chat's ``library_ids``.

        Without this a deleted library leaves chats pointing at a dead id, which keeps
        RAG permanently active for those chats (``logic.resolve_rag`` only checks that
        the list is non-empty) while retrieving nothing — and the Library button reads
        "none", so there is no way to see why. Returns the number of chats changed."""
        with self._lock:
            changed = 0
            for c in self.chats:
                ids = c.get("library_ids") or []
                if lib_id in ids:
                    c["library_ids"] = [i for i in ids if i != lib_id]
                    changed += 1
            if changed:
                self.save_chats()
            return changed

    def _prune_orphan_library_refs(self):
        """Drop chat ``library_ids`` entries with no matching library. Repairs chats
        broken by deletions that predate ``prune_library_refs``. Returns True if any
        chat was mutated (so the caller can persist)."""
        known = {l.get("id") for l in self.libraries}
        changed = False
        for c in self.chats:
            ids = c.get("library_ids") or []
            kept = [i for i in ids if i in known]
            if len(kept) != len(ids):
                c["library_ids"] = kept
                changed = True
        return changed

    # ---------------- evals (Prompt Validation & Evaluation) ----------------
    def eval_summaries(self):
        with self._lock:
            return [
                {"id": e["id"], "name": e.get("name", "Untitled eval"),
                 "updated": e.get("updated", "")}
                for e in self.evals
            ]

    def get_eval(self, eval_id):
        with self._lock:
            for e in self.evals:
                if e.get("id") == eval_id:
                    return e
            return None

    def add_eval(self, ev):
        with self._lock:
            self.evals.insert(0, ev)
            self.save_evals()
            return ev

    def upsert_eval(self, ev):
        with self._lock:
            existing = self.get_eval(ev.get("id"))
            if existing is None:
                self.evals.insert(0, ev)
            else:
                idx = self.evals.index(existing)
                self.evals[idx] = ev
            self.save_evals()
            return ev

    def delete_eval(self, eval_id):
        with self._lock:
            before = len(self.evals)
            self.evals = [e for e in self.evals if e.get("id") != eval_id]
            self.save_evals()
            return len(self.evals) < before

    # ---------------- batch projects (Batch tab) ----------------
    # A saved Batch config: input sources, the prompt template, and the export
    # settings. Holds no secrets and no item content — sources are re-resolved on
    # every run, so a project stays small no matter how much it processes.
    def batch_project_summaries(self):
        with self._lock:
            return [
                {"id": p["id"], "name": p.get("name", "Untitled batch"),
                 "updated": p.get("updated", "")}
                for p in self.batch_projects
            ]

    def get_batch_project(self, project_id):
        with self._lock:
            for p in self.batch_projects:
                if p.get("id") == project_id:
                    return p
            return None

    def upsert_batch_project(self, proj):
        with self._lock:
            existing = self.get_batch_project(proj.get("id"))
            if existing is None:
                self.batch_projects.insert(0, proj)
            else:
                self.batch_projects[self.batch_projects.index(existing)] = proj
            self.save_batch_projects()
            return proj

    def delete_batch_project(self, project_id):
        with self._lock:
            before = len(self.batch_projects)
            self.batch_projects = [p for p in self.batch_projects
                                   if p.get("id") != project_id]
            self.save_batch_projects()
            return len(self.batch_projects) < before

    # ---------------- database import sessions (Database tab) ----------------
    # NB: these hold only NON-secret metadata (name, source-profile id ref, table,
    # selection mode, column configs). Credentials live in the AES-GCM vault, and
    # the staged rows live in per-session DuckDB files under data/db/staging/.
    def db_project_summaries(self):
        with self._lock:
            return [
                {"id": p["id"], "name": p.get("name", "Untitled import"),
                 "profile_id": p.get("profile_id"), "table": p.get("table"),
                 "status": p.get("status", "new"), "updated": p.get("updated", "")}
                for p in self.db_projects
            ]

    def get_db_project(self, project_id):
        with self._lock:
            for p in self.db_projects:
                if p.get("id") == project_id:
                    return p
            return None

    def add_db_project(self, proj):
        with self._lock:
            self.db_projects.insert(0, proj)
            self.save_db_projects()
            return proj

    def upsert_db_project(self, proj):
        with self._lock:
            existing = self.get_db_project(proj.get("id"))
            if existing is None:
                self.db_projects.insert(0, proj)
            else:
                self.db_projects[self.db_projects.index(existing)] = proj
            self.save_db_projects()
            return proj

    def delete_db_project(self, project_id):
        with self._lock:
            before = len(self.db_projects)
            self.db_projects = [p for p in self.db_projects if p.get("id") != project_id]
            self.save_db_projects()
            return len(self.db_projects) < before
