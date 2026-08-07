#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Flask web server. Serves the single-page browser UI (static/) and a JSON API for
every feature of the original desktop app: chats, streaming generation + stop,
presets, servers/models, libraries (Resources), web search, and folder batch
processing. Filesystem operations use native OS dialogs (app.native_dialog) and
read/write paths directly on the machine — nothing is uploaded.
"""

import base64
import ipaddress
import json
import queue
import shutil
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import requests
from flask import (Flask, request, jsonify, Response, send_from_directory,
                   stream_with_context, session, redirect)

from . import (batch as batch_mod, compile as compile_mod, context_tracker, core, crypto,
               evals, images as images_mod, ingest, logic, memory, migrate, native_dialog,
               parallel, persona as persona_mod, persona_io, persona_store,
               pipeline as pipeline_mod, profiles, providers, rag, rewrite, youtube)
from .database import staging as db_staging
from .database.routes import register_db_routes
from .database.vault import Vault
from .core import (
    APP_NAME, APP_AUTHOR, APP_VERSION, DEFAULT_LOCAL_URL, CONTEXT_LENGTHS,
    MIN_CRAWLED_PAGES, PROVIDER_PRESETS, web_search, strip_markdown, looks_like_reasoning_model,
    library_to_xml_bytes, library_from_xml_file, _new_library, _new_library_item,
    prompts_to_xml_bytes, prompts_from_xml_file,
    _normalize_domain,
)
from .store import Store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _expand_range(rng: str):
    """Expand an IP range into a list of address strings. Accepts CIDR
    (192.168.1.0/24), a hyphen range (192.168.1.10-50 or 192.168.1.10-192.168.1.50),
    or a single address."""
    rng = (rng or "").strip()
    if not rng:
        return []
    if "/" in rng:
        net = ipaddress.ip_network(rng, strict=False)
        hosts = list(net.hosts())
        return [str(h) for h in hosts] or [str(net.network_address)]
    if "-" in rng:
        start, end = [p.strip() for p in rng.split("-", 1)]
        start_ip = ipaddress.ip_address(start)
        if "." not in end:  # short form: 192.168.1.10-50
            prefix = start.rsplit(".", 1)[0]
            end_ip = ipaddress.ip_address(f"{prefix}.{end}")
        else:
            end_ip = ipaddress.ip_address(end)
        if int(end_ip) < int(start_ip):
            start_ip, end_ip = end_ip, start_ip
        return [str(ipaddress.ip_address(i)) for i in range(int(start_ip), int(end_ip) + 1)]
    return [str(ipaddress.ip_address(rng))]


def _store_model_image(frame):
    """Persist an image a model returned and return its record, or None.

    Returns the record rather than the bytes so the SSE frame stays small — the
    browser fetches the picture from /api/images/<id> once, instead of receiving it
    JSON-escaped in the stream and again in the chat it persists afterwards.
    A picture that can't be decoded is dropped: it must not take the answer with it.
    """
    try:
        data = base64.b64decode(frame.get("b64") or "", validate=False)
        if not data:
            return None
        prep = images_mod.prepare(data, frame.get("media_type") or "")
        return images_mod.store_prepared(
            prep, name=f"generated-{frame.get('index', 0) + 1}"
                       f"{images_mod.ext_for(prep['media_type'])}",
            origin="model")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Active-generation registry: maps a run id -> threading.Event so /api/stop can
# cancel an in-flight stream or batch run.
# ---------------------------------------------------------------------------
class RunRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._events = {}

    def new(self, run_id):
        ev = threading.Event()
        with self._lock:
            self._events[run_id] = ev
        return ev

    def stop(self, run_id):
        with self._lock:
            ev = self._events.get(run_id)
        if ev:
            ev.set()
            return True
        return False

    def stop_all(self):
        with self._lock:
            evs = list(self._events.values())
        for ev in evs:
            ev.set()

    def done(self, run_id):
        with self._lock:
            self._events.pop(run_id, None)

    def active(self, run_id):
        """True while ``run_id`` is registered. A liveness probe that, unlike stop(),
        does not cancel the run — the Database tab uses it to tell a genuinely
        in-flight import from one whose client disconnected and left the session
        advertising work that no longer exists."""
        if not run_id:
            return False
        with self._lock:
            return run_id in self._events


def create_app():
    app = Flask(__name__, static_folder=None)
    # Sign session cookies with the persisted secret (creates the encryption keyfile
    # with default admin/admin on first run). The app data is encrypted at rest and is
    # only loaded after the user logs in — see the auth block below.
    app.secret_key = crypto.get_flask_secret(core.APP_KEYFILE)
    # Image uploads are the only request body that isn't small JSON. A ceiling turns
    # "someone dragged a 2 GB scan in" from an out-of-memory server into a 413.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    # Profiles: run first-run migration, then point core's data/settings paths at the
    # persisted active profiles BEFORE building the Store/Vault that read those paths.
    pm = profiles.ProfileManager()
    core.set_active_settings_profile(pm.active_settings_dir())
    core.set_active_data_profile(pm.active_data_dir())
    store = Store()
    runs = RunRegistry()
    # Apply the compile/chunking settings to the shared RAG chunker.
    rag.set_chunk_config(store.config.get("rag_chunker"),
                         store.config.get("rag_chunk_size"),
                         store.config.get("rag_chunk_overlap"))

    def _apply_rag_backend():
        """Select the vector store, honouring an existing DuckDB corpus.

        The default is LanceDB, but an upgrade must not make an existing install look
        empty: if the user has never chosen a backend explicitly and a populated
        ``rag.duckdb`` is present, stay on DuckDB until they migrate. Fresh installs
        (and anyone who has chosen) get what the setting says."""
        chosen = store.config.get("rag_backend")
        if not store.config_has_explicit("rag_backend"):
            try:
                legacy = Path(core.RAG_DB_FILE)
                if legacy.exists() and legacy.stat().st_size > 0:
                    chosen = "duckdb"
            except Exception:
                pass
        rag.set_backend(chosen)
        rag.set_ann_enabled(store.config.get("rag_ann_enabled", True))

    _apply_rag_backend()
    vault = Vault(core.VAULT_FILE)   # Database tab: encrypted connection profiles (per data profile)
    # Cache of model -> capability list per server, so we don't re-hit /api/show.
    caps_cache = {}
    # Persona system (Phase 4-7): CRUD service + live pipeline engines keyed by run_id
    # (kept so a paused/completed run can be edited and re-run from a step).
    psvc = persona_mod.PersonaService()
    kbsvc = persona_store.KnowledgeService()
    memsvc = persona_store.MemoryService()
    # Bounded: each entry pins a persona, an adapter closure, and the run's full state
    # (every step's output, its retrieved excerpts, and any raw model text). Unbounded,
    # a long session of persona chat grew this forever. Only recent runs can plausibly
    # be re-run from a step, so keep the newest few and evict oldest-first.
    PIPELINE_RUNS_MAX = 20
    pipeline_runs = OrderedDict()

    def _remember_run(run_id, eng):
        pipeline_runs[run_id] = eng
        pipeline_runs.move_to_end(run_id)
        while len(pipeline_runs) > PIPELINE_RUNS_MAX:
            pipeline_runs.popitem(last=False)

    def _gc_images():
        """Sweep image files nothing references any more.

        A sweep rather than refcounting: a private chat writes its images long before
        (and usually instead of) being persisted, so no count is ever correct.
        images.gc() spares anything recently written for exactly that reason. Best
        effort — failing to tidy up must never break the thing that triggered it."""
        try:
            live = images_mod.collect_ids(store.chats, store.batch_projects)
            return images_mod.gc(live)
        except Exception:
            return 0

    # ----------------------------- Profile switching ------------------------
    def _switch_data_runtime():
        """Repoint every data-profile collaborator at the CURRENT core.* paths. Assumes
        core.set_active_data_profile()/INCOGNITO has already run. Releases the previous
        profile's file handles first so nothing keeps writing to the old profile."""
        runs.stop_all()
        rag.reset_connection()
        db_staging.close_all()
        vault.set_path(core.VAULT_FILE)   # locks the old vault; re-unlock per profile
        store.reload_data()
        # The new profile has its own stores, so re-decide which backend to open.
        _apply_rag_backend()
        caps_cache.clear()
        # The image caches hold DECRYPTED bytes keyed only by id, so carrying them
        # across a profile switch would leak the previous profile's pictures.
        images_mod.clear_caches()
        _gc_images()

    def _switch_settings_runtime():
        core.set_active_settings_profile(pm.active_settings_dir())
        store.reload_settings()
        _apply_rag_backend()
        caps_cache.clear()

    def sse(event, data):
        """Format one Server-Sent-Events frame."""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def adapter_for(server_url):
        """Return the provider adapter for a base_url (resolved to its server entry)."""
        return providers.get_client(store.resolve_server(server_url))

    def _ollama_caps(server, server_url, model):
        key = (server_url, model)
        if key not in caps_cache:
            caps_cache[key] = providers.get_client(server).model_capabilities(model)
        return caps_cache[key]

    def model_supports_tools(server_url, model):
        """Return True/False/None for native tool support. Only Ollama servers
        advertise tool capabilities; cloud providers use the app-side web-search
        path, so they resolve to None (no native tools)."""
        if not model:
            return None
        server = store.resolve_server(server_url)
        if server.get("type") != "ollama":
            return None
        caps = _ollama_caps(server, server_url, model)
        return ("tools" in caps) if caps else None

    # Cloud model-name markers for image INPUT. Only Ollama actually reports a
    # "vision" capability tag; everywhere else this is the best that can be done.
    _CLOUD_VISION = ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5", "chatgpt-4o",
                     "o3", "o4-mini", "claude-", "gemini-", "grok-2-vision",
                     "grok-3", "grok-4", "pixtral", "llama-3.2-11b", "llama-3.2-90b",
                     "llama-4", "qwen2.5-vl", "qwen3-vl", "internvl", "llava",
                     "-vl", "vision", "moondream", "minicpm-v", "mistral-small-3")
    # Names we are confident have NO vision, so the warning can fire honestly.
    _CLOUD_TEXT_ONLY = ("deepseek-chat", "deepseek-reasoner", "gpt-3.5", "o1-mini",
                        "text-embedding", "whisper", "tts-", "-embed", "embedding")
    # Models that return an image through the chat-completions path.
    _IMAGE_OUTPUT = ("gemini-2.5-flash-image", "gemini-2.0-flash-preview-image",
                     "gemini-3-pro-image", "flash-image", "-image-preview")

    def model_supports_vision(server_url, model):
        """Return True/False/None for image input.

        Ollama reports the real capability tag. Cloud providers report nothing, so
        this falls back to name markers and answers **None** when neither list
        matches — an honest "don't know" is what keeps the UI from warning about a
        model that is perfectly capable, or staying silent about one that isn't."""
        if not model:
            return None
        server = store.resolve_server(server_url)
        if server.get("type") == "ollama":
            caps = _ollama_caps(server, server_url, model)
            return ("vision" in caps) if caps else None
        m = model.lower()
        if any(h in m for h in _CLOUD_VISION):
            return True
        if any(h in m for h in _CLOUD_TEXT_ONLY):
            return False
        return None

    def model_returns_images(server_url, model):
        """True for the handful of models that hand back a picture. Name-only: no
        provider advertises this, and Ollama's chat endpoint cannot do it at all."""
        if not model:
            return False
        if store.resolve_server(server_url).get("type") == "ollama":
            return False
        m = model.lower()
        return any(h in m for h in _IMAGE_OUTPUT)

    # Cloud model-name markers that indicate a reasoning/thinking model.
    _CLOUD_REASONING = ("reasoner", "o1", "o3", "-thinking", "magistral",
                        "qwq", "claude-opus", "claude-sonnet", "glm-4")

    def model_is_reasoning(server_url, model):
        """True if the model produces a separable chain-of-thought we should show.
        Ollama: has the 'thinking' capability (or a reasoning-model name). Cloud:
        name heuristics (best effort)."""
        if not model:
            return False
        server = store.resolve_server(server_url)
        if server.get("type") == "ollama":
            caps = _ollama_caps(server, server_url, model)
            if "thinking" in (caps or []):
                return True
            return looks_like_reasoning_model(model)
        m = model.lower()
        return looks_like_reasoning_model(model) or any(h in m for h in _CLOUD_REASONING)

    # ----------------------------- Static files -----------------------------
    def _no_cache(resp):
        # This is a local single-user app; never let the browser serve a stale
        # index.html / app.js after the code changes.
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/")
    def index():
        return _no_cache(send_from_directory(STATIC_DIR, "index.html"))

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return _no_cache(send_from_directory(STATIC_DIR, filename))

    # ----------------------------- Authentication ---------------------------
    # App data is AES-encrypted at rest; the login password unwraps the Data
    # Encryption Key (crypto.py). Until the user logs in, no data route may run
    # (nothing can be decrypted) and the Store is empty — a successful login activates
    # the key, migrates any legacy plaintext, and (re)loads settings + data.
    _AUTH_ENDPOINTS = {"login_page", "api_login", "api_logout", "api_auth_status"}

    def _activate_after_login():
        rag.reset_connection()
        db_staging.close_all()
        try:
            migrate.run()                 # one-time: encrypt legacy plaintext in place
        except Exception:
            pass
        store.reload_settings()
        store.reload_data()
        rag.set_chunk_config(store.config.get("rag_chunker"),
                             store.config.get("rag_chunk_size"),
                             store.config.get("rag_chunk_overlap"))
        _apply_rag_backend()
        vault.set_path(core.VAULT_FILE)   # re-point the (separate) DB connection vault
        caps_cache.clear()

    @app.errorhandler(413)
    def _payload_too_large(_e):
        """Flask aborts oversized uploads before the route runs; without this the
        client gets an HTML error page where it expects JSON."""
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"That file is too large (limit {mb} MB)."}), 413

    @app.before_request
    def _require_login():
        path = request.path or "/"
        if (path == "/login" or path == "/favicon.ico" or path.startswith("/static/")
                or request.endpoint in _AUTH_ENDPOINTS):
            return None
        if not session.get("authed") or not crypto.is_unlocked():
            if path.startswith("/api/"):
                return jsonify({"error": "Authentication required."}), 401
            return redirect("/login")
        return None

    @app.route("/login")
    def login_page():
        return _no_cache(send_from_directory(STATIC_DIR, "login.html"))

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(force=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not crypto.verify_username(core.APP_KEYFILE, username):
            return jsonify({"error": "Invalid username or password."}), 401
        try:
            dek = crypto.unlock(core.APP_KEYFILE, password)
        except crypto.AuthError:
            return jsonify({"error": "Invalid username or password."}), 401
        crypto.set_active_key(dek)
        session["authed"] = True
        try:
            _activate_after_login()
        except Exception as e:
            crypto.clear_key()
            session.clear()
            return jsonify({"error": f"Failed to open your data: {e}"}), 500
        return jsonify({"ok": True, "auth": crypto.status(core.APP_KEYFILE)})

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        session.clear()
        crypto.clear_key()
        return jsonify({"ok": True})

    @app.route("/api/auth/status")
    def api_auth_status():
        return jsonify({"authed": bool(session.get("authed") and crypto.is_unlocked()),
                        **crypto.status(core.APP_KEYFILE)})

    @app.route("/api/auth/change", methods=["POST"])
    def api_auth_change():
        """Change the login username/password. Re-wraps the same Data Encryption Key,
        so no bulk re-encryption of data is needed."""
        data = request.get_json(force=True) or {}
        try:
            st = crypto.change_credentials(
                core.APP_KEYFILE,
                old_password=data.get("current_password") or "",
                new_username=data.get("new_username") or "",
                new_password=data.get("new_password") or "",
            )
        except crypto.AuthError:
            return jsonify({"error": "Current password is incorrect."}), 400
        return jsonify({"ok": True, "auth": st})

    @app.route("/api/auth/dismiss-warning", methods=["POST"])
    def api_auth_dismiss():
        return jsonify({"ok": True, "auth": crypto.dismiss_default_warning(core.APP_KEYFILE)})

    # ----------------------------- App state --------------------------------
    @app.route("/api/state")
    def api_state():
        return jsonify({
            "app": {"name": APP_NAME, "author": APP_AUTHOR, "version": APP_VERSION},
            "profiles": pm.state(),
            "config": store.masked_config(),
            "servers": store.server_choices(),
            "provider_presets": PROVIDER_PRESETS,
            "presets": store.presets,
            "prompts": store.prompts,
            "libraries": store.libraries,
            "chats": store.chat_summaries(),
            "chat_groups": store.groups(),
            "context_lengths": CONTEXT_LENGTHS,
            "min_crawled_pages": MIN_CRAWLED_PAGES,
            "default_local_url": DEFAULT_LOCAL_URL,
            "default_eval_prompt": logic.DEFAULT_EVAL_PROMPT,
            "evals": store.eval_summaries(),
            "default_criteria": evals.DEFAULT_CRITERIA,
            "batch_projects": store.batch_project_summaries(),
            "default_batch_project": batch_mod.new_project(),
            "batch_source_kinds": batch_mod.SOURCE_KINDS,
            "batch_exts": sorted(ingest.SUPPORTED_EXTS),
            "memory_cores": memory.decorate_all(store.memory_cores_snapshot()),
            "memory_categories": memory.CATEGORIES,
        })

    # ----------------------------- Profiles ---------------------------------
    @app.route("/api/profiles", methods=["GET"])
    def api_profiles_get():
        return jsonify({"profiles": pm.state()})

    # ---- data profiles ----
    @app.route("/api/profiles/data", methods=["POST"])
    def api_profiles_data_create():
        """Create a data profile. Body: {name, seed:'blank'|'clone', activate:bool}."""
        data = request.get_json(force=True) or {}
        seed = "clone" if data.get("seed") == "clone" else "blank"
        prof = pm.create_data(data.get("name", ""), seed)
        if data.get("activate", True):
            pm.set_active_data(prof["id"])
            core.set_active_data_profile(pm.active_data_dir())
            _switch_data_runtime()
        return jsonify({"profile": prof, "profiles": pm.state()})

    @app.route("/api/profiles/data/<profile_id>/activate", methods=["POST"])
    def api_profiles_data_activate(profile_id):
        try:
            pm.set_active_data(profile_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        core.set_active_data_profile(pm.active_data_dir())
        _switch_data_runtime()
        return jsonify({"profiles": pm.state()})

    @app.route("/api/profiles/data/<profile_id>", methods=["PATCH"])
    def api_profiles_data_rename(profile_id):
        data = request.get_json(force=True) or {}
        prof = pm.rename_data(profile_id, data.get("name", ""))
        if not prof:
            return jsonify({"error": "not found"}), 404
        return jsonify({"profile": prof, "profiles": pm.state()})

    @app.route("/api/profiles/data/<profile_id>", methods=["DELETE"])
    def api_profiles_data_delete(profile_id):
        try:
            pm.delete_data(profile_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"profiles": pm.state()})

    # ---- settings profiles ----
    @app.route("/api/profiles/settings", methods=["POST"])
    def api_profiles_settings_create():
        data = request.get_json(force=True) or {}
        prof = pm.create_settings(data.get("name", ""))
        if data.get("activate", True):
            pm.set_active_settings(prof["id"])
            _switch_settings_runtime()
        return jsonify({"profile": prof, "profiles": pm.state()})

    @app.route("/api/profiles/settings/<profile_id>/activate", methods=["POST"])
    def api_profiles_settings_activate(profile_id):
        try:
            pm.set_active_settings(profile_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        _switch_settings_runtime()
        return jsonify({"profiles": pm.state()})

    @app.route("/api/profiles/settings/<profile_id>", methods=["PATCH"])
    def api_profiles_settings_rename(profile_id):
        data = request.get_json(force=True) or {}
        prof = pm.rename_settings(profile_id, data.get("name", ""))
        if not prof:
            return jsonify({"error": "not found"}), 404
        return jsonify({"profile": prof, "profiles": pm.state()})

    @app.route("/api/profiles/settings/<profile_id>", methods=["DELETE"])
    def api_profiles_settings_delete(profile_id):
        try:
            pm.delete_settings(profile_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"profiles": pm.state()})

    # ---- incognito (data axis only) ----
    @app.route("/api/profiles/data/incognito", methods=["POST"])
    def api_profiles_incognito_start():
        """Enter a private session. Body: {seed:'blank'|'clone'}."""
        data = request.get_json(force=True) or {}
        seed = "clone" if data.get("seed") == "clone" else "blank"
        pm.start_incognito(seed)
        core.set_active_data_profile(pm.active_data_dir())   # -> the incognito scratch
        store.incognito = True   # set before reload so blank-session backfills stay in memory
        _switch_data_runtime()
        return jsonify({"profiles": pm.state()})

    @app.route("/api/profiles/data/incognito/save", methods=["POST"])
    def api_profiles_incognito_save():
        """Persist the current private session. Body: {mode:'new'|'merge',
        name (new) | target_id (merge)}."""
        if not pm.incognito:
            return jsonify({"error": "Not in an incognito session."}), 400
        data = request.get_json(force=True) or {}
        mode = data.get("mode")
        scratch = core.INCOGNITO_DIR
        if mode == "new":
            prof = pm.create_data(data.get("name", ""), seed="blank")
            dest = pm.data_dir(prof["id"])
            # Carry over the RAG store, DB staging/audit, the credential vault and any
            # attached images that were written into the scratch during the session.
            for name in ("rag.duckdb", "rag.duckdb.wal", "db", "db_vault.enc", "images"):
                src = scratch / name
                if src.exists():
                    dst = dest / name
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(src), str(dst))
            store.incognito = False
            pm.set_active_data(prof["id"])
            core.set_active_data_profile(dest)
            rag.reset_connection(); db_staging.close_all(); vault.set_path(core.VAULT_FILE)
            store.flush_all()      # write the in-memory session data into the new profile
            store.reload_data()    # reload so ids/backfills match what's on disk
            caps_cache.clear()
            pm.stop_incognito()
            return jsonify({"profiles": pm.state(), "activated": prof["id"]})
        elif mode == "merge":
            target_id = data.get("target_id")
            if target_id not in {p["id"] for p in pm.data_reg["profiles"]}:
                return jsonify({"error": "No such target profile."}), 400
            target_dir = pm.data_dir(target_id)
            store.merge_into(target_dir)       # merges in-memory into a copy of target
            store.incognito = False
            pm.set_active_data(target_id)
            core.set_active_data_profile(target_dir)
            rag.reset_connection(); db_staging.close_all(); vault.set_path(core.VAULT_FILE)
            store.flush_all()                  # write merged result back to the target
            store.reload_data()
            caps_cache.clear()
            pm.stop_incognito()
            return jsonify({"profiles": pm.state(), "activated": target_id})
        return jsonify({"error": "mode must be 'new' or 'merge'."}), 400

    @app.route("/api/profiles/data/incognito/discard", methods=["POST"])
    def api_profiles_incognito_discard():
        """Leave incognito without saving, returning to the last real data profile."""
        pm.stop_incognito()
        store.incognito = False
        core.set_active_data_profile(pm.active_data_dir())
        _switch_data_runtime()
        return jsonify({"profiles": pm.state()})

    # ----------------------------- Servers ----------------------------------
    @app.route("/api/servers", methods=["GET"])
    def api_servers_get():
        # Return masked servers (no api_key values) for the Settings editor.
        return jsonify({
            "servers": store.server_choices(),
            "raw": store.masked_config()["servers"],
        })

    @app.route("/api/servers", methods=["PUT"])
    def api_servers_put():
        """Replace saved servers. Entries with a blank api_key on a provider that
        already had one keep the existing key (so the masked editor doesn't wipe it)."""
        data = request.get_json(force=True) or {}
        incoming = data.get("servers", [])
        existing = {s["base_url"].rstrip("/"): s for s in store.config.get("servers", [])}
        for s in incoming:
            base = (s.get("base_url") or s.get("url") or "").rstrip("/")
            if not s.get("api_key") and base in existing:
                s["api_key"] = existing[base].get("api_key", "")
        clean = store.set_servers(incoming)
        caps_cache.clear()
        masked = [{"name": x["name"], "type": x["type"], "base_url": x["base_url"],
                   "has_key": bool(x.get("api_key"))} for x in clean]
        return jsonify({"servers": store.server_choices(), "raw": masked})

    @app.route("/api/servers/scan", methods=["POST"])
    def api_servers_scan():
        """Scan an IP range for Ollama servers. Body: {range, port}. Returns
        responders as [{base_url, name, model_count}] — the client picks which to add."""
        data = request.get_json(force=True) or {}
        rng = (data.get("range") or "").strip()
        try:
            port = int(data.get("port") or 11434)
        except Exception:
            port = 11434
        try:
            hosts = _expand_range(rng)
        except Exception as e:
            return jsonify({"error": f"Could not parse range: {e}"}), 400
        if not hosts:
            return jsonify({"error": "No addresses to scan."}), 400
        if len(hosts) > 1024:
            return jsonify({"error": "Range too large (limit 1024 addresses)."}), 400

        def probe(ip):
            base = f"http://{ip}:{port}"
            try:
                r = requests.get(f"{base}/api/tags", timeout=1.2)
                r.raise_for_status()
                models = (r.json() or {}).get("models") or []
                return {"base_url": base, "name": ip, "model_count": len(models)}
            except Exception:
                return None

        found = []
        with ThreadPoolExecutor(max_workers=64) as ex:
            for fut in as_completed([ex.submit(probe, h) for h in hosts]):
                res = fut.result()
                if res:
                    found.append(res)
        found.sort(key=lambda x: x["base_url"])
        return jsonify({"found": found, "scanned": len(hosts)})

    # ----------------------------- Settings ---------------------------------
    @app.route("/api/settings", methods=["POST"])
    def api_settings():
        """Save API keys and general settings. Blank secret fields are ignored so
        they don't clobber a stored key."""
        data = request.get_json(force=True) or {}
        allowed = ("brave_token", "brightdata_token", "brightdata_zone",
                   "default_num_ctx", "auto_detect_reasoning", "max_output_tokens",
                   "rag_embed_server_url", "rag_embed_model", "rag_top_k",
                   "rag_retrieval_mode", "rag_contextual_chunking", "rag_context_model",
                   "rag_query_rewrite", "rewrite_model",
                   "rag_chunker", "rag_chunk_size", "rag_chunk_overlap",
                   "rag_embed_batch_size", "rag_embed_concurrency",
                   "rag_embed_parallel", "rag_embed_servers", "rag_ann_enabled",
                   "rag_backend",
                   "memory_weight_influence", "pipeline_max_retries",
                   "provider_context_windows",
                   "image_max_dim", "image_full_res_default")
        patch = {k: data[k] for k in allowed if k in data}
        if "provider_context_windows" in patch:
            raw = patch["provider_context_windows"]
            if isinstance(raw, dict):
                clean = {}
                for k, v in raw.items():
                    try:
                        clean[str(k)] = max(1, int(v))
                    except Exception:
                        continue
                patch["provider_context_windows"] = clean
            else:
                patch.pop("provider_context_windows")
        if "rag_retrieval_mode" in patch and patch["rag_retrieval_mode"] not in (
                "vector", "keyword", "hybrid"):
            patch.pop("rag_retrieval_mode")
        for boolk in ("rag_contextual_chunking", "rag_query_rewrite",
                      "rag_embed_parallel", "rag_ann_enabled",
                      "image_full_res_default"):
            if boolk in patch:
                patch[boolk] = bool(patch[boolk])
        if "image_max_dim" in patch:
            try:
                # Below ~256 an image carries no usable detail; above 8192 nothing
                # accepts it and the base64 alone would swamp the context.
                patch["image_max_dim"] = max(256, min(8192, int(patch["image_max_dim"])))
            except Exception:
                patch.pop("image_max_dim")
        if "rag_embed_servers" in patch:
            # [{base_url, enabled}] — no per-server model on purpose: the whole pool
            # embeds with rag_embed_model, since vectors from different embedding
            # models are not comparable and mixing them would corrupt the index.
            clean, seen = [], set()
            for s in (patch.get("rag_embed_servers") or []):
                url = (str(s.get("base_url") or "")).strip().rstrip("/")
                if not url or url in seen:
                    continue
                seen.add(url)
                clean.append({"base_url": url, "enabled": bool(s.get("enabled", True))})
            patch["rag_embed_servers"] = clean
        if "pipeline_max_retries" in patch:
            try:
                patch["pipeline_max_retries"] = max(1, min(6, int(patch["pipeline_max_retries"])))
            except Exception:
                patch.pop("pipeline_max_retries")
        if "memory_weight_influence" in patch:
            try:
                patch["memory_weight_influence"] = max(0.0, min(1.0, float(patch["memory_weight_influence"])))
            except Exception:
                patch.pop("memory_weight_influence")
        for numk in ("default_num_ctx", "max_output_tokens", "rag_top_k",
                     "rag_chunk_size", "rag_chunk_overlap"):
            if numk in patch:
                try:
                    patch[numk] = int(patch[numk])
                except Exception:
                    patch.pop(numk)
        store.update_config(patch)
        # A chunker change alters chunk boundaries, so re-apply it live; the compile
        # signature then flags previously compiled data as stale until recompiled.
        if any(k in patch for k in ("rag_chunker", "rag_chunk_size", "rag_chunk_overlap")):
            rag.set_chunk_config(store.config.get("rag_chunker"),
                                 store.config.get("rag_chunk_size"),
                                 store.config.get("rag_chunk_overlap"))
        if "rag_backend" in patch:
            # Switching the store takes effect immediately; the other backend's data is
            # left untouched, so this is reversible.
            rag.set_backend(store.config.get("rag_backend"))
        if "rag_ann_enabled" in patch:
            # Turning the ANN index off drops it immediately so the RAM comes back
            # without needing a restart.
            rag.set_ann_enabled(store.config.get("rag_ann_enabled", True))
        return jsonify({"config": store.masked_config()})

    # ----------------------------- Prompt rewrite ---------------------------
    @app.route("/api/rewrite-prompt", methods=["POST"])
    def api_rewrite_prompt():
        """On-demand instruction improvement for the chat input. Body:
        {text, server_url?, model?, messages?}. Rewrites ``text`` into a clearer, more
        effective instruction using the chat's provider/model and returns
        {rewritten, original}. Never throws — returns the original on failure."""
        data = request.get_json(force=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "empty text"}), 400
        server_url = data.get("server_url") or store.config.get("last_server_url") or DEFAULT_LOCAL_URL
        model = store.config.get("rewrite_model") or data.get("model") or ""
        if not model:
            return jsonify({"error": "No model selected for rewriting."}), 400
        try:
            adapter = adapter_for(server_url)
            hist = rewrite.history_text(data.get("messages", []))
            rewritten = rewrite.improve_prompt(adapter, model, text, hist)
            return jsonify({"rewritten": rewritten, "original": text})
        except Exception as e:
            return jsonify({"rewritten": text, "original": text, "error": str(e)})

    # ----------------------------- Models -----------------------------------
    @app.route("/api/models")
    def api_models():
        server_url = request.args.get("server", DEFAULT_LOCAL_URL)
        try:
            models = adapter_for(server_url).list_models()
            err = None
        except Exception as e:
            models, err = [], str(e)
        store.update_config({"last_server_url": server_url})
        return jsonify({"models": models, "error": err})

    @app.route("/api/models/capabilities")
    def api_model_caps():
        server_url = request.args.get("server", DEFAULT_LOCAL_URL)
        model = request.args.get("model", "")
        supported = model_supports_tools(server_url, model)
        return jsonify({
            "tools": supported,
            "reasoning_hint": looks_like_reasoning_model(model),
            # None = unknown (a cloud model we have no marker for). The UI stays
            # quiet on None rather than guessing in either direction.
            "vision": model_supports_vision(server_url, model),
            "image_output": model_returns_images(server_url, model),
        })

    # ----------------------------- Chats ------------------------------------
    @app.route("/api/chats", methods=["GET"])
    def api_chats_list():
        return jsonify({"chats": store.chat_summaries()})

    @app.route("/api/chats/<chat_id>", methods=["GET"])
    def api_chat_get(chat_id):
        chat = store.get_chat(chat_id)
        if not chat:
            return jsonify({"error": "not found"}), 404
        return jsonify({"chat": chat})

    @app.route("/api/chats", methods=["POST"])
    def api_chat_create():
        data = request.get_json(force=True) or {}
        group_id = data.get("group_id")
        # Only honour a group that actually exists; "default"/unknown -> built-in tab.
        if group_id and group_id not in {g["id"] for g in store.groups()}:
            group_id = None
        chat = logic.create_chat_dict(
            title=data.get("title", "Private Chat" if data.get("private") else "New Chat"),
            server_url=data.get("server_url") or store.config.get("last_server_url") or DEFAULT_LOCAL_URL,
            model=data.get("model", ""),
            private=bool(data.get("private", False)),
            default_num_ctx=store.config.get("default_num_ctx", 4096),
            group_id=group_id,
        )
        if not chat["private"]:
            store.add_chat(chat)
        return jsonify({"chat": chat})

    @app.route("/api/chats/<chat_id>", methods=["PATCH"])
    def api_chat_patch(chat_id):
        """Update chat settings/title/messages. For private chats the client
        sends the full chat object (server doesn't persist private chats until
        'save'); for saved chats we patch the stored copy.
        """
        data = request.get_json(force=True) or {}
        chat = data.get("chat")
        if chat and chat.get("id") == chat_id:
            if not chat.get("private"):
                chat["updated"] = datetime.utcnow().isoformat()
                store.upsert_chat(chat)
            return jsonify({"chat": chat})
        # Field-level patch of a saved chat.
        existing = store.get_chat(chat_id)
        if not existing:
            return jsonify({"error": "not found"}), 404
        for k, v in (data.get("patch") or {}).items():
            existing[k] = v
        existing["updated"] = datetime.utcnow().isoformat()
        store.upsert_chat(existing)
        return jsonify({"chat": existing})

    @app.route("/api/chats/<chat_id>/save", methods=["POST"])
    def api_chat_save_private(chat_id):
        """Promote a private chat (sent in body) to a saved chat."""
        data = request.get_json(force=True) or {}
        chat = data.get("chat") or {}
        chat["private"] = False
        chat["id"] = chat.get("id") or uuid.uuid4().hex[:12]
        chat["updated"] = datetime.utcnow().isoformat()
        store.upsert_chat(chat)
        return jsonify({"chat": chat})

    @app.route("/api/chats/<chat_id>", methods=["DELETE"])
    def api_chat_delete(chat_id):
        store.delete_chat(chat_id)
        # Its images are now unreferenced. The age guard in images.gc keeps this from
        # touching anything a still-open chat is using.
        _gc_images()
        return jsonify({"ok": True})

    # -------------------- Chat export / import (native dialogs) -------------
    def _write_chat_export(envelope, default_name):
        """Open a native Save dialog and write the export envelope as JSON."""
        dest = native_dialog.save_file(title="Export chats as JSON",
                                       default_name=default_name, filetypes_key="json")
        if not dest:
            return jsonify({"ok": False, "cancelled": True})
        try:
            if not dest.lower().endswith(".json"):
                dest += ".json"
            Path(dest).write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": dest,
                        "count": len(envelope.get("chats", [])),
                        # An export is deliberately plaintext so it opens anywhere.
                        # When it carries pictures, that is worth saying out loud.
                        "images": len(envelope.get("images") or {}),
                        "warnings": envelope.get("warnings") or []})

    @app.route("/api/chats/export", methods=["POST"])
    def api_chats_export():
        """Export all saved chats (scope='all') or a subset (ids=[...]) to a
        user-chosen JSON file. No secrets are included."""
        data = request.get_json(force=True) or {}
        ids = None if data.get("scope") == "all" else (data.get("ids") or None)
        envelope = store.export_chats(ids)
        default_name = "all_chats.json" if ids is None else "chats.json"
        return _write_chat_export(envelope, default_name)

    @app.route("/api/chats/<chat_id>/export", methods=["POST"])
    def api_chat_export_one(chat_id):
        """Export a single chat to a user-chosen JSON file."""
        chat = store.get_chat(chat_id)
        if not chat:
            return jsonify({"error": "not found"}), 404
        envelope = store.export_chats([chat_id])
        safe = (chat.get("title") or "chat").strip().replace(" ", "_") or "chat"
        return _write_chat_export(envelope, f"{safe}.json")

    @app.route("/api/chats/import", methods=["POST"])
    def api_chats_import():
        """Open a native picker for a chat-export JSON file and import it into a
        new sidebar tab named after the file (renameable later)."""
        paths = native_dialog.pick_files(title="Import chats (JSON)", filetypes_key="json")
        if not paths:
            return jsonify({"ok": False, "cancelled": True})
        path = Path(paths[0])
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return jsonify({"ok": False, "error": f"Could not read file: {e}"}), 400
        try:
            group, count = store.import_chats(envelope, path.stem)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "group": group, "count": count,
                        "chats": store.chat_summaries(), "chat_groups": store.groups()})

    # -------------------- Chat groups (sidebar tabs) ------------------------
    @app.route("/api/chat-groups/<group_id>", methods=["PATCH"])
    def api_chat_group_rename(group_id):
        data = request.get_json(force=True) or {}
        group = store.rename_group(group_id, data.get("name", ""))
        if not group:
            return jsonify({"error": "not found or not renameable"}), 400
        return jsonify({"group": group, "chat_groups": store.groups()})

    @app.route("/api/chat-groups/<group_id>", methods=["DELETE"])
    def api_chat_group_delete(group_id):
        if not store.delete_group(group_id):
            return jsonify({"error": "not found or not deletable"}), 400
        return jsonify({"ok": True, "chats": store.chat_summaries(),
                        "chat_groups": store.groups()})

    # -------------------- Streaming generation + stop -----------------------
    _CONTEXT_SYS = (
        "You situate a text excerpt within its source document. Given a short document "
        "summary and an excerpt, reply with ONE concise sentence (max 25 words) stating "
        "what the excerpt is about and how it fits the document. Output only that sentence.")

    def _contextualizer_for(context_model, embed_url):
        """Return a ``contextualize(chunk, summary)->str`` callable that situates each
        chunk with a local Ollama chat model on the embed host, or None when no model is
        given. Per-chunk failures are swallowed by the store (fall back to the raw chunk).
        Shared by the persona pipeline and the Compile Data step."""
        if not context_model:
            return None
        client = core.OllamaClient(embed_url)

        def contextualize(chunk, summary):
            return client.complete(
                context_model,
                [{"role": "system", "content": _CONTEXT_SYS},
                 {"role": "user", "content":
                    f"Document summary:\n{summary}\n\nExcerpt:\n{chunk}\n\nSituating sentence:"}],
                num_ctx=2048, temperature=0.2, timeout=60)

        return contextualize

    def _embed_lanes(primary_url):
        """[{base_url, name}] for the embedding pool: the primary endpoint plus any
        enabled extra servers from ``rag_embed_servers``. Returns just the primary
        unless fan-out is switched on, so the default single-machine setup is
        unchanged."""
        lanes = [{"base_url": primary_url.rstrip("/"),
                  "name": store.resolve_server(primary_url).get("name") or primary_url}]
        if not store.config.get("rag_embed_parallel"):
            return lanes
        seen = {lanes[0]["base_url"]}
        for s in (store.config.get("rag_embed_servers") or []):
            url = (s.get("base_url") or "").strip().rstrip("/")
            if not url or url in seen or not s.get("enabled", True):
                continue
            seen.add(url)
            lanes.append({"base_url": url,
                          "name": store.resolve_server(url).get("name") or url})
        return lanes

    def _compile_embedder(persona=None):
        """Resolve (embed_pool, embed_model, embed_url) for a compile run, mirroring the
        chat RAG path. Persona compiles honor the persona's embedding_model override.

        The pool fans batches across every enabled embedding server at once; with a
        single server it behaves exactly as the old single-callable path did."""
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_model = store.config.get("rag_embed_model") or "nomic-embed-text"
        if persona:
            embed_model = (persona.get("models", {}).get("embedding_model") or embed_model)
        batch_size, per_server = _compile_tuning()
        pool = rag.EmbedPool(lanes=_embed_lanes(embed_url), model=embed_model,
                             per_server_concurrency=per_server, batch_size=batch_size)
        return pool, embed_model, embed_url

    def _compile_contextualizer(embed_url, want_contextual):
        """Build the compile-time contextualizer from the global model setting when the
        caller asks for contextual chunking (defaults to the global toggle)."""
        if not want_contextual:
            return None
        ctx_model = store.config.get("rag_context_model") or store.config.get("rewrite_model") or ""
        return _contextualizer_for(ctx_model, embed_url)

    def _compile_tuning():
        """(batch_size, max_workers) for a compile run — how many chunks per embed
        request and how many embed requests to run concurrently. Sanitized to sane
        floors so a bad setting can't stall or serialize a compile unexpectedly."""
        try:
            batch = max(1, int(store.config.get("rag_embed_batch_size", 64)))
        except Exception:
            batch = 64
        try:
            workers = max(1, int(store.config.get("rag_embed_concurrency", 3)))
        except Exception:
            workers = 3
        return batch, workers

    def _rag_retrieve(chat):
        """Resolve + run RAG for one generation, shared by interactive/queue
        (generate_one) and sequential batch. Returns (rag_plan, rag_retrieved,
        status_msg): rag_plan is None when RAG is inactive OR degraded to full
        context; a plan with ``blocked=True`` means a selected library isn't
        compiled — the caller must inject NO library context (no full dump) and
        surface status_msg prompting the user to Compile. Indexing is NOT done here:
        libraries/personas must be compiled explicitly via the Compile Data button."""
        rag_plan = logic.resolve_rag(chat, store.config)
        if not (rag_plan and rag_plan.get("active")):
            return None, None, None
        try:
            embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
            embed_model = store.config.get("rag_embed_model") or "nomic-embed-text"
            embed_client = core.OllamaClient(embed_url)
            embed_fn = lambda texts: embed_client.embed(embed_model, texts)
            lib_ids = chat.get("library_ids") or []
            # Compilation is required: block RAG (and the full-text fallback) for any
            # selected library that isn't compiled/fresh, prompting the user to Compile.
            if rag_plan["use_libraries"] and lib_ids:
                uncompiled = [lib.get("name") or lib.get("id")
                              for lib in store.libraries if lib.get("id") in lib_ids
                              and compile_mod.library_status(lib, embed_model).get("state") != "compiled"]
                if uncompiled:
                    names = ", ".join(f'"{n}"' for n in uncompiled)
                    msg = (f"⚠ Library {names} needs compiling before RAG can use it — "
                           f"open Resources → Compile Data.")
                    # Dropping the library also drops the Strict preamble, so a chat set
                    # to answer ONLY from its references answers from general knowledge
                    # this turn. Say so rather than letting it pass silently.
                    if chat.get("library_strict"):
                        msg += (" Strict mode is NOT in effect for this message — the "
                                "answer may come from the model's own knowledge.")
                    return ({"active": True, "blocked": True}, None, msg)
            mode = rag_plan.get("mode", "hybrid")
            queries = rag_plan.get("queries") or ([rag_plan["query"]] if rag_plan["query"] else [])
            # Prompt Reword: expand the message into 2-3 retrieval queries (+ keywords),
            # resolving references from history. Degrades to the base query on any error.
            if rag_plan.get("query_rewrite") and rag_plan.get("query"):
                try:
                    rw_model = (store.config.get("rewrite_model")
                                or chat.get("model") or "")
                    rw_adapter = adapter_for(chat.get("server_url") or DEFAULT_LOCAL_URL)
                    rw = rewrite.rewrite_queries(
                        rw_adapter, rw_model, rag_plan["query"],
                        rewrite.history_text(chat.get("messages", [])))
                    variants = list(rw.get("variants") or [])
                    if rw.get("keywords"):
                        variants.append(" ".join(rw["keywords"]))
                    queries = [q for q in variants if q] or queries
                except Exception:
                    pass
            # Keyword-only mode needs no embeddings at all — retrieve without touching
            # the embed server (so RAG works even with none configured). Chunks were
            # written up front by Compile Data, so nothing is indexed on this path.
            wants_vectors = mode in ("vector", "hybrid")
            qvecs = []
            if wants_vectors and queries:
                qvecs = embed_fn(queries)
            retrieved = []
            if rag_plan["use_libraries"]:
                retrieved += rag.retrieve_libraries(
                    qvecs, lib_ids, embed_model, rag_plan["top_k"],
                    mode=mode, queries=queries)
            if rag_plan["data_text"]:
                retrieved += rag.retrieve_inline(
                    qvecs, rag_plan["data_text"], embed_fn if wants_vectors else None,
                    rag_plan["top_k"], mode=mode, queries=queries,
                    embed_model=embed_model)
            retrieved.sort(key=lambda d: d.get("score", -1.0), reverse=True)
            rag_retrieved = retrieved[:rag_plan["top_k"]]
            return rag_plan, rag_retrieved, f"📚 RAG ({mode}): injected {len(rag_retrieved)} relevant chunk(s)"
        except Exception as e:
            # Degrade to full-context: caller falls back to the full library dump.
            return None, None, f"📚 RAG unavailable ({e}); using full context"

    def generate_one(chat, search_query, stop_event, batch_item_label=None):
        """Core single-generation loop shared by the /send route and the parallel
        engine. Yields (kind, data) tuples — kind in pass_start / status /
        reasoning / chunk / pass_end / context / error. Does NOT touch the run
        registry or emit start/done; the caller wraps those. The chat's last message
        must be the user turn to answer, and chat['server_url']/['model'] select the
        target. ``batch_item_label`` (e.g. a filename) tags the context-usage frames
        and history entry when this call is one item of a batch."""
        server_url = chat.get("server_url") or DEFAULT_LOCAL_URL
        adapter = adapter_for(server_url)
        model = chat.get("model") or ""
        if not model:
            yield ("error", {"message": "No model selected."})
            return

        # Context-usage monitor: resolve the bar denominator + server identity once.
        server = store.resolve_server(server_url)
        server_id = server.get("base_url") or server_url
        server_name = server.get("name") or server_id
        window = context_tracker.resolve_window(chat, server, store.config)
        chat_id = chat.get("id") or ""
        is_isolated = bool(chat.get("isolated"))
        is_private = bool(chat.get("private"))

        options = {
            "num_ctx": chat.get("num_ctx") or store.config.get("default_num_ctx", 4096),
            "max_output_tokens": store.config.get("max_output_tokens", 16000),
        }
        think = model_is_reasoning(server_url, model)
        show_reasoning = not bool(chat.get("hide_thinking", False))
        tools_supported = model_supports_tools(server_url, model)
        # 0 means "send the originals untouched"; otherwise clamp the long edge, which
        # is what keeps a handful of phone photos from eating the whole context window.
        image_max_dim = (0 if chat.get("image_full_res")
                         else int(store.config.get("image_max_dim")
                                  or images_mod.DEFAULT_MAX_DIM))
        pinned_images = logic.attachment_images(chat)

        # Multi-Pass: N refinement rounds after the initial answer.
        multi = bool(chat.get("multi_pass"))
        N = max(0, int(chat.get("passes") or 0)) if multi else 0
        input_prompt = ""
        for m in reversed(chat.get("messages", [])):
            if m.get("role") == "user":
                input_prompt = m.get("content", "")
                break

        # RAG for the whole turn: retrieve top-k relevant chunks once and reuse them
        # across every Multi-Pass round (a selected library is always served via RAG).
        # use_rag is only true when chunks actually came back — otherwise we fall back
        # to the full library dump so the library never silently vanishes.
        rag_plan, rag_retrieved, rag_status = _rag_retrieve(chat)
        use_rag = bool(rag_plan and rag_plan.get("active") and rag_retrieved)
        # Uncompiled library: drop it from the turn so no full-text dump leaks in; the
        # rag_status line tells the user to Compile first.
        if rag_plan and rag_plan.get("blocked"):
            chat = dict(chat)
            chat["library_ids"] = []
        pass_use_system = bool(chat.get("pass_use_system", True))

        last_answer = ""
        try:
            for p in range(N + 1):
                if stop_event.is_set():
                    break
                if N == 0:
                    label, intermediate = "", False
                elif p == 0:
                    label, intermediate = "Initial", True
                elif p == N:
                    label, intermediate = f"Final (pass {N})", False
                else:
                    label, intermediate = f"Pass {p}", True
                yield ("pass_start", {"index": p, "total": N + 1,
                                      "label": label, "intermediate": intermediate})

                if p == 0:
                    if rag_status:
                        yield ("status", {"message": rag_status})
                    if use_rag:
                        messages = logic.build_messages(chat, store.libraries, skip_library_dump=True)
                        # Pinned attachments go in first so the RAG excerpts stay the
                        # message closest to the question.
                        messages = logic.inject_attachments(
                            messages, logic.resolve_attachments(chat, use_rag))
                        messages = logic.inject_rag(messages, rag_retrieved, store.libraries)
                    else:
                        messages = logic.build_messages(chat, store.libraries)
                        messages = logic.inject_attachments(
                            messages, logic.resolve_attachments(chat, use_rag))
                    ws = logic.resolve_web_search(chat, store.config, search_query, tools_supported)
                    if ws["worker_query"]:
                        yield ("status", {"message":
                            f"🌐 Searching the web & crawling ≥{ws['min_pages']} page(s)…"})
                        research = web_search(
                            ws["worker_query"], min_pages=ws["min_pages"],
                            should_stop=lambda: stop_event.is_set(),
                            allowed_domains=ws["allowed_domains"],
                        )
                        if stop_event.is_set():
                            yield ("pass_end", {"label": label, "intermediate": intermediate,
                                                "content": "", "reasoning": ""})
                            break
                        messages = logic.inject_research(messages, ws["worker_query"], research)
                    # What the assistant has learned about this user, if the chat opted
                    # in. Pass 0 only, like RAG — refinement passes rework an answer
                    # that was already written with the memory in view.
                    mem_core = memory.resolve_memory(chat, store.memory_cores_snapshot())
                    if mem_core:
                        messages = logic.inject_memory(messages, memory.render_core(mem_core))
                    pass_tools = ws["tools"]
                    tool_executor = logic.make_tool_executor(
                        stop_event, min_pages=ws["min_pages"], allowed_domains=ws["allowed_domains"])
                else:
                    if use_rag:
                        messages = logic.build_eval_messages(
                            chat, input_prompt, last_answer, store.libraries,
                            pass_use_system, skip_library_dump=True)
                        messages = logic.inject_rag(messages, rag_retrieved, store.libraries)
                    else:
                        messages = logic.build_eval_messages(
                            chat, input_prompt, last_answer, store.libraries, pass_use_system)
                    # Pinned attachments follow into every refinement round. They are
                    # reference material for the whole conversation, and the pass that
                    # writes the FINAL answer is the one that most needs to see them.
                    # (Memory is deliberately pass-0 only — see above — because the
                    # answer being refined was already written with it in view.)
                    messages = logic.inject_attachments(
                        messages, logic.resolve_attachments(chat, use_rag))
                    pass_tools, tool_executor = None, None

                # Pinned images follow into every pass, like the attachment block:
                # material pinned to the conversation is meant to be in view for the
                # whole of it, and the pass writing the final answer needs it most.
                messages = logic.inject_images(messages, pinned_images)

                # Context-usage: pre-call estimate (breakdown + full prompt) so the
                # bar can move while the request is in flight. Isolation is inherently
                # correct here — excluded history was never in `messages`.
                # Measured BEFORE hydration, so the tracker never walks base64.
                breakdown = context_tracker.breakdown_from_messages(messages)
                prompt_estimate = context_tracker.estimate_messages(messages)
                # Attached images cost real prompt tokens the text estimate can't see —
                # without this an image-only turn reports as free and the bar lies.
                image_tokens = images_mod.estimate_message_tokens(messages)
                if image_tokens:
                    prompt_estimate += image_tokens
                    breakdown["user"] = breakdown.get("user", 0) + image_tokens

                # Swap image ids for inline bytes. The one place image data is read
                # during a generation; the adapters reshape it per provider from here.
                messages = images_mod.hydrate_messages(messages, image_max_dim)
                yield ("context", {
                    "phase": "start", "chat_id": chat_id,
                    "server": server_id, "server_name": server_name,
                    "window": window, "prompt_tokens": prompt_estimate,
                    "breakdown": breakdown, "isolation": is_isolated,
                    "batch_item_label": batch_item_label, "exact": False,
                    "pass_index": p, "pass_total": N + 1,
                })

                content, reasoning = "", ""
                pass_images = []
                exact_usage = None
                for kind, text in adapter.chat_stream(
                    model, messages, options, stop_event,
                    think=think, tools=pass_tools, tool_executor=tool_executor,
                ):
                    if kind == "usage":
                        exact_usage = text  # exact provider counts (dict)
                    elif kind == "reasoning":
                        if show_reasoning:
                            reasoning += text
                            yield ("reasoning", {"content": text})
                    elif kind == "image":
                        # Store the bytes and forward only the record: the same
                        # megabytes would otherwise cross the wire twice (once
                        # JSON-escaped into an SSE frame, once in the persist body).
                        rec = _store_model_image(text)
                        if rec:
                            pass_images.append(rec)
                            yield ("image", rec)
                    else:
                        content += text
                        yield ("chunk", {"content": text})
                last_answer = content

                # Context-usage: exact totals become the source of truth (fall back to
                # the estimate when a provider omits usage). Emit a finish frame + one
                # persisted history entry per LLM call (skipped for private chats).
                exact_prompt = (exact_usage or {}).get("prompt_tokens") or prompt_estimate
                exact_completion = ((exact_usage or {}).get("completion_tokens")
                                    or context_tracker.estimate_tokens(content))
                yield ("context", {
                    "phase": "finish", "chat_id": chat_id,
                    "server": server_id, "server_name": server_name,
                    "window": window, "prompt_tokens": exact_prompt,
                    "completion_tokens": exact_completion, "breakdown": breakdown,
                    "isolation": is_isolated, "batch_item_label": batch_item_label,
                    "exact": bool(exact_usage), "pass_index": p, "pass_total": N + 1,
                })
                entry = context_tracker.make_entry(
                    server_id=server_id, server_name=server_name, isolation=is_isolated,
                    breakdown=breakdown, prompt_tokens=exact_prompt,
                    completion_tokens=exact_completion, num_ctx_at_time=window,
                    batch_item_label=batch_item_label,
                    notes=(label or None) if N else None)
                context_tracker.record_call(store, chat_id, entry, private=is_private)

                yield ("pass_end", {"label": label, "intermediate": intermediate,
                                    "content": content,
                                    "reasoning": reasoning if show_reasoning else "",
                                    "images": pass_images})
        except Exception as e:
            yield ("error", {"message": str(e)})

    def _generate_stream(chat, search_query, run_id):
        """SSE wrapper around generate_one for the single-chat /send route."""
        stop_event = runs.new(run_id)
        try:
            yield sse("start", {"run_id": run_id})
            for kind, data in generate_one(chat, search_query, stop_event):
                yield sse(kind, data)
            yield sse("done", {"stopped": stop_event.is_set()})
        finally:
            runs.done(run_id)

    @app.route("/api/chats/<chat_id>/send", methods=["POST"])
    def api_chat_send(chat_id):
        """Body: {chat: <full chat dict incl. the just-added user message>,
                  search_query: str, run_id: str}. Streams the assistant reply.
        The client owns chat state and re-sends it; saved chats are persisted here.
        """
        data = request.get_json(force=True) or {}
        chat = data.get("chat") or store.get_chat(chat_id)
        if not chat:
            return jsonify({"error": "not found"}), 404
        search_query = data.get("search_query", "")
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]

        gen = _generate_stream(chat, search_query, run_id)
        return Response(stream_with_context(gen), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        data = request.get_json(force=True) or {}
        run_id = data.get("run_id")
        if run_id:
            runs.stop(run_id)
        else:
            runs.stop_all()
        return jsonify({"ok": True})

    # Persist a saved chat's messages after streaming completes (client calls this).
    @app.route("/api/chats/<chat_id>/persist", methods=["POST"])
    def api_chat_persist(chat_id):
        data = request.get_json(force=True) or {}
        chat = data.get("chat") or {}
        if chat.get("private"):
            return jsonify({"ok": True, "skipped": "private"})
        chat["updated"] = datetime.utcnow().isoformat()
        store.upsert_chat(chat)
        return jsonify({"ok": True})

    # -------------------- Context-usage history (per chat) ------------------
    @app.route("/api/chats/<chat_id>/context-history", methods=["GET"])
    def api_chat_context_history(chat_id):
        """Return this chat's persisted per-call token-usage history (newest last)."""
        return jsonify({"history": context_tracker.get_history(store, chat_id)})

    @app.route("/api/chats/<chat_id>/context-history", methods=["DELETE"])
    def api_chat_context_history_clear(chat_id):
        """Clear this chat's context-usage history."""
        with store._lock:
            if store.context_history.pop(chat_id, None) is not None:
                store.save_context_history()
        return jsonify({"ok": True})

    # =========================== Personas ===================================
    def _persona_embed(persona):
        """(embed_fn, embed_model) for a persona's stores, via the configured Ollama
        embed host. embed_fn returns [] on failure so keyword paths still work."""
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_model = (persona.get("models", {}).get("embedding_model")
                       or store.config.get("rag_embed_model") or "nomic-embed-text")
        client = core.OllamaClient(embed_url)

        def embed_fn(texts):
            try:
                return client.embed(embed_model, texts)
            except Exception:
                return [[] for _ in texts]
        return embed_fn, embed_model

    def _persona_engine(persona, chat, run_id, variant=""):
        """Assemble a PipelineEngine with real Ollama/provider + persona-store I/O."""
        server_url = chat.get("server_url") or DEFAULT_LOCAL_URL
        adapter = adapter_for(server_url)
        chat_model = persona.get("models", {}).get("chat_model") or chat.get("model") or ""
        embed_fn, embed_model = _persona_embed(persona)
        pid = persona["id"]
        mode = persona.get("stores", {}).get("retrieval", "hybrid")
        reword = persona.get("stores", {}).get("prompt_reword", True)
        top_k = int(store.config.get("rag_top_k") or 6)
        wants_vectors = mode in ("vector", "hybrid")
        try:
            temperature = float(persona.get("models", {}).get("temperature", 0.7))
        except (TypeError, ValueError):
            temperature = 0.7
        # A mutable holder, not the Event itself: each action (start, then any re-run
        # from a step) registers a FRESH Event with `runs`, and llm_stream has to see
        # the current one or Stop stops nothing. _pipeline_sse rebinds holder["event"].
        stop_holder = {"event": runs.new(run_id)}
        # The persona pipeline is otherwise isolated from the chat's context (no
        # libraries, attachments, or web search), but the chat's own system prompt is a
        # standing instruction from the user and applies to a persona answer too.
        chat_system, _pre = logic._resolve_prompts(chat)

        def llm_complete(model, messages, schema):
            # Pass the run's stop event: without it every structured step ran to
            # completion after the user pressed Stop.
            return rewrite.run_completion(adapter, model or chat_model, messages,
                                          num_ctx=chat.get("num_ctx") or 4096,
                                          max_tokens=store.config.get("max_output_tokens", 4096),
                                          fmt=schema, temperature=temperature,
                                          stop=stop_holder["event"])

        def llm_stream(model, messages):
            options = {"num_ctx": chat.get("num_ctx") or 4096,
                       "max_output_tokens": store.config.get("max_output_tokens", 16000),
                       "temperature": temperature}
            for kind, text in adapter.chat_stream(model or chat_model, messages, options,
                                                  stop_holder["event"], think=False):
                if kind == "content":
                    yield text

        def _qvecs(queries):
            if wants_vectors and queries:
                try:
                    return embed_fn(queries)
                except Exception:
                    return []
            return []

        def knowledge_search(queries):
            return kbsvc.search(pid, _qvecs(queries), embed_model, top_k,
                                mode=mode, queries=queries)

        def memory_search(queries):
            return memsvc.search(pid, _qvecs(queries), embed_model, top_k,
                                 mode=mode, queries=queries,
                                 weight_influence=float(store.config.get("memory_weight_influence", 0.35)))

        def rewrite_queries(message, history):
            if not reword:
                return {"variants": [message] if message else [], "keywords": []}
            return rewrite.rewrite_queries(adapter, chat_model, message, history)

        eng = pipeline_mod.PipelineEngine(
            persona, llm_complete=llm_complete, llm_stream=llm_stream,
            knowledge_search=knowledge_search, memory_search=memory_search,
            rewrite_queries=rewrite_queries, emit=None,
            max_retries=int(store.config.get("pipeline_max_retries", 3)),
            should_stop=lambda: stop_holder["event"].is_set(),
            variant=variant, extra_system=chat_system)
        eng._stop_holder = stop_holder
        return eng

    def _pipeline_sse(run_id, action):
        """Run a pipeline action (start / run_from) in a worker thread and multiplex the
        engine's emitted events into one SSE stream."""
        frames = queue.Queue()
        sentinel = object()
        eng = pipeline_runs.get(run_id)
        if eng is None:
            def _missing():
                yield sse("error", {"message": "run not found"})
                yield sse("done", {})
            return _missing()
        eng.emit = lambda ev, data: frames.put((ev, data))
        # Re-register with the stop registry for EVERY action, not just the first.
        # The generator's `finally` calls runs.done(run_id), so after the initial run
        # the engine was holding an Event nobody could reach: a re-run from a step
        # streamed to completion with POST /api/stop answering "not found".
        holder = getattr(eng, "_stop_holder", None)
        if holder is not None:
            holder["event"] = runs.new(run_id)

        def work():
            try:
                action(eng)
            except Exception as e:
                frames.put(("error", {"message": str(e)}))
            finally:
                frames.put(sentinel)

        def gen():
            threading.Thread(target=work, daemon=True).start()
            try:
                yield sse("start", {"run_id": run_id})
                while True:
                    frame = frames.get()
                    if frame is sentinel:
                        break
                    ev, data = frame
                    yield sse(ev, data)
                yield sse("done", {})
            finally:
                runs.done(run_id)
        return gen()

    # ----------------------------- Compile Data (RAG) -----------------------
    def _compile_sse(work_fn, run_id=None):
        """Run a Compile Data job in a worker thread and stream its emit(event, data)
        frames as SSE (mirrors ``_pipeline_sse``). ``work_fn(emit)`` may raise.

        The job registers with the shared ``runs`` stop registry so ``POST /api/stop``
        can cancel it — compiling a shelf of ebooks can run for a long time, and
        previously there was no way to call it off. If the client disconnects we also
        set the stop event, so an abandoned run doesn't keep embedding forever and fill
        an unbounded queue."""
        frames = queue.Queue()
        sentinel = object()
        stop_event = runs.new(run_id) if run_id else None

        def work():
            try:
                work_fn(lambda ev, data: frames.put((ev, data)), stop_event)
            except Exception as e:
                frames.put(("error", {"message": str(e)}))
            finally:
                frames.put(sentinel)

        def gen():
            threading.Thread(target=work, daemon=True).start()
            try:
                yield sse("start", {"run_id": run_id})
                while True:
                    frame = frames.get()
                    if frame is sentinel:
                        break
                    ev, data = frame
                    yield sse(ev, data)
                yield sse("done", {})
            finally:
                # GeneratorExit lands here when the browser goes away mid-compile.
                if stop_event is not None:
                    stop_event.set()
                if run_id:
                    runs.done(run_id)
        return Response(stream_with_context(gen()), mimetype="text/event-stream")

    def _rag_embed_model():
        return store.config.get("rag_embed_model") or "nomic-embed-text"

    @app.route("/api/libraries/<lib_id>/compile-status", methods=["GET"])
    def api_library_compile_status(lib_id):
        lib = next((l for l in store.libraries if l.get("id") == lib_id), None)
        if lib is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(compile_mod.library_status(lib, _rag_embed_model()))

    @app.route("/api/libraries/<lib_id>/compile", methods=["POST"])
    def api_library_compile(lib_id):
        lib = next((l for l in store.libraries if l.get("id") == lib_id), None)
        if lib is None:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(silent=True) or {}
        force = bool(body.get("force"))
        want_ctx = body.get("contextual")
        if want_ctx is None:
            want_ctx = bool(store.config.get("rag_contextual_chunking"))
        pool, embed_model, embed_url = _compile_embedder()
        contextualize = _compile_contextualizer(embed_url, want_ctx)
        batch_size, max_workers = _compile_tuning()
        run_id = body.get("run_id") or f"compile-library-{lib_id}"

        def work(emit, stop_event):
            compile_mod.compile_library(lib, pool, embed_model,
                                        contextualize=contextualize, force=force, emit=emit,
                                        batch_size=batch_size, max_workers=max_workers,
                                        stop_event=stop_event)
        return _compile_sse(work, run_id)

    @app.route("/api/personas/<pid>/compile-status", methods=["GET"])
    def api_persona_compile_status(pid):
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        embed_model = persona.get("models", {}).get("embedding_model") or _rag_embed_model()
        return jsonify(compile_mod.persona_status(persona, embed_model))

    @app.route("/api/personas/<pid>/compile", methods=["POST"])
    def api_persona_compile(pid):
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        body = request.get_json(silent=True) or {}
        force = bool(body.get("force"))
        want_ctx = body.get("contextual")
        if want_ctx is None:
            want_ctx = bool(store.config.get("rag_contextual_chunking"))
        pool, embed_model, embed_url = _compile_embedder(persona)
        contextualize = _compile_contextualizer(embed_url, want_ctx)
        batch_size, max_workers = _compile_tuning()
        run_id = body.get("run_id") or f"compile-persona-{pid}-{uuid.uuid4().hex[:8]}"

        def work(emit, stop_event):
            compile_mod.compile_persona(persona, pool, embed_model,
                                        contextualize=contextualize, force=force, emit=emit,
                                        batch_size=batch_size, max_workers=max_workers,
                                        stop_event=stop_event)
        return _compile_sse(work, run_id)

    @app.route("/api/rag/backend", methods=["GET"])
    def api_rag_backend():
        """Which vector store is live, what each holds, and whether a migration is
        worth offering."""
        from .vectorstore import migrate as vs_migrate
        duck_rows = 0
        try:
            duck_rows = vs_migrate.duckdb_row_count()
        except Exception:
            pass
        return jsonify({
            "backend": rag.backend_name(),
            "explicit": store.config_has_explicit("rag_backend"),
            "status": rag.backend_status(),
            "duckdb_rows": duck_rows,
            "lance_dir": str(core.RAG_LANCE_DIR),
            "duckdb_path": str(core.RAG_DB_FILE),
        })

    @app.route("/api/rag/migrate", methods=["POST"])
    def api_rag_migrate():
        """SSE. Copy the DuckDB vector store into LanceDB. The DuckDB file is left in
        place, so this is reversible by switching the backend back."""
        from .vectorstore import migrate as vs_migrate

        def work(emit, stop_event):
            vs_migrate.migrate_duckdb_to_lance(emit=emit, stop_event=stop_event)

        return _compile_sse(work, "rag-migrate")

    @app.route("/api/rag/embed-health", methods=["GET"])
    def api_rag_embed_health():
        """Is each configured embedding server reachable, and does it actually have the
        embedding model pulled? A server that answers but lacks the model would produce
        failed batches mid-compile, so the settings UI checks up front.

        Also reports the ANN index state so the user can see what retrieval is using."""
        embed_model = _rag_embed_model()
        primary = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        lanes = _embed_lanes(primary)
        # Include disabled/extra servers too, so the UI can show why one is unchecked.
        known = {l["base_url"] for l in lanes}
        for s in (store.config.get("rag_embed_servers") or []):
            url = (s.get("base_url") or "").strip().rstrip("/")
            if url and url not in known:
                known.add(url)
                lanes.append({"base_url": url, "name": url, "enabled": False})

        def probe(lane):
            out = {"base_url": lane["base_url"], "name": lane.get("name") or lane["base_url"],
                   "primary": lane["base_url"] == primary.rstrip("/"),
                   "reachable": False, "has_model": False, "models": 0, "error": ""}
            try:
                models = core.OllamaClient(lane["base_url"]).list_models() or []
                out["reachable"] = True
                out["models"] = len(models)
                base = embed_model.split(":")[0]
                out["has_model"] = any(m == embed_model or m.split(":")[0] == base
                                       for m in models)
            except Exception as e:
                out["error"] = str(e)[:200]
            return out

        with ThreadPoolExecutor(max_workers=max(1, len(lanes))) as ex:
            servers = list(ex.map(probe, lanes))
        return jsonify({"embed_model": embed_model,
                        "parallel": bool(store.config.get("rag_embed_parallel")),
                        "servers": servers, "ann": rag.ann_status()})

    @app.route("/api/personas", methods=["GET"])
    def api_personas_list():
        return jsonify({"personas": psvc.list_all()})

    @app.route("/api/persona-default-pipeline", methods=["GET"])
    def api_persona_default_pipeline():
        return jsonify({"pipeline": persona_mod.default_pipeline()})

    @app.route("/api/persona-health", methods=["GET"])
    def api_persona_health():
        """Is the embedding Ollama reachable and is the embed model pulled? Surfaces a
        setup hint in the Personas tab."""
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_model = store.config.get("rag_embed_model") or "nomic-embed-text"
        out = {"embed_url": embed_url, "embed_model": embed_model,
               "ollama": False, "embed_present": False}
        try:
            models = adapter_for(embed_url).list_models()
            out["ollama"] = True
            stem = embed_model.split(":")[0]
            out["embed_present"] = any(stem in m for m in models)
        except Exception as e:
            out["error"] = str(e)
        return jsonify(out)

    @app.route("/api/personas", methods=["POST"])
    def api_personas_create():
        data = request.get_json(force=True) or {}
        try:
            p = psvc.create(data.get("name", ""), data.get("chat_model", ""),
                            data.get("embedding_model", "nomic-embed-text"))
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"persona": p, "personas": psvc.list_all()})

    @app.route("/api/personas/<pid>", methods=["GET"])
    def api_persona_get(pid):
        try:
            return jsonify({"persona": psvc.load(pid)})
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/api/personas/<pid>", methods=["PUT"])
    def api_persona_update(pid):
        data = request.get_json(force=True) or {}
        p = data.get("persona") or {}
        p["id"] = pid
        try:
            saved = psvc.save(p)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"persona": saved})

    @app.route("/api/personas/<pid>", methods=["DELETE"])
    def api_persona_delete(pid):
        try:
            psvc.delete(pid)
            rag.delete_source(pid)      # drop this persona's knowledge + memory vectors
            compile_mod.forget_persona(pid)   # drop its compile manifest
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "personas": psvc.list_all()})

    @app.route("/api/personas/<pid>/duplicate", methods=["POST"])
    def api_persona_duplicate(pid):
        data = request.get_json(force=True) or {}
        try:
            p = psvc.duplicate(pid, data.get("name", ""))
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"persona": p, "personas": psvc.list_all()})

    def _persona_user_message(chat):
        for m in reversed(chat.get("messages", [])):
            if m.get("role") == "user":
                _d, q = logic.split_inline_data(m.get("content", ""))
                return q or m.get("content", "")
        return ""

    @app.route("/api/personas/<pid>/chat", methods=["POST"])
    def api_persona_chat(pid):
        """Run the persona's chain-of-thought pipeline for the chat's last user message
        and stream step events + the final answer over SSE."""
        data = request.get_json(force=True) or {}
        chat = data.get("chat") or {}
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        user_message = _persona_user_message(chat)
        history = rewrite.history_text(chat.get("messages", []))
        variant = data.get("variant") or chat.get("persona_variant") or ""
        eng = _persona_engine(persona, chat, run_id, variant=variant)
        _remember_run(run_id, eng)
        gen = _pipeline_sse(run_id, lambda e: e.start(user_message, history))
        return Response(stream_with_context(gen), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/runs/<run_id>/rerun", methods=["POST"])
    def api_run_rerun(run_id):
        """Edit a completed/paused run's step output and re-run from that step forward."""
        data = request.get_json(force=True) or {}
        if run_id not in pipeline_runs:
            return jsonify({"error": "run not found (start a new message)"}), 404
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "index must be an integer"}), 400
        # Out of range would otherwise surface as an IndexError inside the SSE stream.
        if not (0 <= index < len(pipeline_runs[run_id].run.steps)):
            return jsonify({"error": "step index out of range"}), 400
        edited = data.get("output", None)
        pipeline_runs.move_to_end(run_id)   # a re-run keeps it clear of the LRU eviction
        gen = _pipeline_sse(run_id, lambda e: e.run_from(index, edited))
        return Response(stream_with_context(gen), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------------- Memories ---------------------------------
    _DRAFT_MEMORY_SCHEMA = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "emotional_weight": {"type": "integer"},
        },
        "required": ["title", "description"],
    }

    def _persona_or_404(pid):
        """(persona, None) or (None, error_response). Every persona sub-resource route
        goes through this: without it an unknown id fell through to the store layer,
        which answered 200 with an empty list (and used to scaffold a folder for it),
        and a malformed id raised PersonaError into a 500."""
        try:
            return psvc.load(pid), None
        except persona_mod.PersonaError as e:
            return None, (jsonify({"error": str(e)}), 404)

    @app.route("/api/personas/<pid>/memories", methods=["GET"])
    def api_persona_memories(pid):
        _p, err = _persona_or_404(pid)
        if err:
            return err
        return jsonify({"memories": memsvc.list_memories(pid)})

    @app.route("/api/personas/<pid>/memories", methods=["POST"])
    def api_persona_memory_create(pid):
        data = request.get_json(force=True) or {}
        mem = data.get("memory") or data
        persona, err = _persona_or_404(pid)
        if err:
            return err
        embed_fn, embed_model = _persona_embed(persona)
        saved = memsvc.save_memory(pid, mem, embed_fn, embed_model)
        return jsonify({"memory": saved, "memories": memsvc.list_memories(pid)})

    @app.route("/api/personas/<pid>/memories/<mem_id>", methods=["DELETE"])
    def api_persona_memory_delete(pid, mem_id):
        _p, err = _persona_or_404(pid)
        if err:
            return err
        memsvc.delete_memory(pid, mem_id)
        return jsonify({"ok": True, "memories": memsvc.list_memories(pid)})

    @app.route("/api/personas/<pid>/draft-memory", methods=["POST"])
    def api_persona_draft_memory(pid):
        """Ask the model to draft {title, description, emotional_weight} from selected
        chat messages, for the user to edit before saving. Never throws."""
        data = request.get_json(force=True) or {}
        messages = data.get("messages") or []
        transcript = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)[:6000]
        server_url = data.get("server_url") or store.config.get("last_server_url") or DEFAULT_LOCAL_URL
        model = store.config.get("rewrite_model") or data.get("model") or ""
        draft = {"title": "", "description": transcript[:400], "emotional_weight": 5}
        if model:
            try:
                adapter = adapter_for(server_url)
                sys = ("Summarize the following exchange as a personal MEMORY for a character. "
                       "Return JSON {title, description, emotional_weight} where description is a "
                       "1-3 sentence first-person recollection and emotional_weight is 1-10.")
                raw = rewrite.run_completion(adapter, model,
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": transcript + "\n\nJSON:"}],
                    num_ctx=4096, max_tokens=400, fmt=_DRAFT_MEMORY_SCHEMA)
                parsed = rewrite._extract_json(raw) or {}
                if parsed.get("title"):
                    draft["title"] = str(parsed["title"])
                if parsed.get("description"):
                    draft["description"] = str(parsed["description"])
                try:
                    draft["emotional_weight"] = max(1, min(10, int(parsed.get("emotional_weight", 5))))
                except Exception:
                    pass
            except Exception:
                pass
        return jsonify({"draft": draft})

    # ----------------------------- Persona knowledge ------------------------
    @app.route("/api/personas/<pid>/knowledge", methods=["GET"])
    def api_persona_kb_list(pid):
        _p, err = _persona_or_404(pid)
        if err:
            return err
        return jsonify({"documents": kbsvc.list_documents(pid)})

    @app.route("/api/personas/<pid>/knowledge/add-files", methods=["POST"])
    def api_persona_kb_add(pid):
        """SSE. Native picker → ingest each document into the persona's knowledge base,
        streaming per-file progress. Each file is parsed AND embedded here, so on large
        documents this is the slow path the progress bar exists for."""
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        paths = native_dialog.pick_files(
            title="Add documents to persona knowledge", filetypes_key="documents")
        # Unique per invocation: a fixed "addfiles-persona-<pid>" meant two concurrent
        # runs for one persona shared a stop event, so cancelling one cancelled both.
        run_id = f"addfiles-persona-{pid}-{uuid.uuid4().hex[:8]}"
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_fn, embed_model = _persona_embed(persona)
        mode = persona.get("stores", {}).get("retrieval", "hybrid")
        wants_vectors = mode in ("vector", "hybrid")
        want_ctx = bool(store.config.get("rag_contextual_chunking"))
        ctx_model = ((store.config.get("rag_context_model")
                      or persona.get("models", {}).get("chat_model", "")) if want_ctx else "")
        contextualize = _contextualizer_for(ctx_model, embed_url)

        def work(emit, stop_event):
            emit("begin", {"total": len(paths), "run_id": run_id,
                           "name": persona.get("profile", {}).get("name", "")})
            added, errors = [], []
            for i, p in enumerate(paths):
                if stop_event is not None and stop_event.is_set():
                    break
                name = Path(p).name
                emit("progress", {"phase": "parse", "done": i, "total": len(paths),
                                  "unit": "files", "name": name})
                try:
                    r = kbsvc.ingest_file(pid, p, embed_fn if wants_vectors else None,
                                          embed_model, contextualize=contextualize)
                    added.append(r)
                except (ingest.IngestError, Exception) as e:
                    errors.append(f"{name}: {e}")
                emit("progress", {"phase": "parse", "done": i + 1, "total": len(paths),
                                  "unit": "files", "name": name})
            # Record which embedding model produced these vectors (re-index on mismatch).
            if added and wants_vectors:
                persona.setdefault("stores", {})["embedding_model_used"] = embed_model
                try:
                    psvc.save(persona)
                except Exception:
                    pass
            emit("complete", {"documents": kbsvc.list_documents(pid),
                              "added": added, "errors": errors})

        return _compile_sse(work, run_id)

    @app.route("/api/personas/<pid>/knowledge/add-text", methods=["POST"])
    def api_persona_kb_add_text(pid):
        data = request.get_json(force=True) or {}
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        embed_fn, embed_model = _persona_embed(persona)
        mode = persona.get("stores", {}).get("retrieval", "hybrid")
        wants_vectors = mode in ("vector", "hybrid")
        r = kbsvc.add_text(pid, data.get("name", ""), data.get("text", ""),
                           embed_fn if wants_vectors else None, embed_model)
        return jsonify({"documents": kbsvc.list_documents(pid), "added": [r]})

    @app.route("/api/personas/<pid>/knowledge/<doc_id>", methods=["DELETE"])
    def api_persona_kb_remove(pid, doc_id):
        _p, err = _persona_or_404(pid)
        if err:
            return err
        kbsvc.remove_document(pid, doc_id)
        return jsonify({"documents": kbsvc.list_documents(pid)})

    # ----------------------------- Import / Export --------------------------
    @app.route("/api/personas/<pid>/export.xml", methods=["GET"])
    def api_persona_export_xml(pid):
        try:
            data = persona_io.export_xml(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        return Response(data, mimetype="application/xml", headers={
            "Content-Disposition": f'attachment; filename="{pid}.xml"'})

    @app.route("/api/personas/<pid>/export.zip", methods=["GET"])
    def api_persona_export_zip(pid):
        try:
            data = persona_io.export_bundle(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        return Response(data, mimetype="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{pid}.zip"'})

    @app.route("/api/personas/import", methods=["POST"])
    def api_persona_import():
        """Native picker for a persona .xml or .zip bundle; validate + import (bundles
        re-ingest sources and re-embed memories locally)."""
        paths = native_dialog.pick_files(title="Import persona (XML or .zip)")
        if not paths:
            return jsonify({"error": "no file selected"})
        path = paths[0]
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_default = store.config.get("rag_embed_model") or "nomic-embed-text"
        try:
            persona = persona_io.import_path(
                path, psvc, kbsvc, memsvc, embed_url=embed_url,
                embed_model_default=embed_default, wants_vectors=True)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"import failed: {e}"}), 400
        return jsonify({"persona": persona, "personas": psvc.list_all()})

    # ----------------------------- User memory cores ------------------------
    # A memory core is the assistant's profile OF THE USER, grown from their chats and
    # editable here. Distinct from persona memories above (in-character recollections,
    # embedded + retrieved); a core is small, plain JSON, and injected whole. See
    # app/memory.py for the model and the extraction prompts.

    def _memory_eligible(chat, mode="auto"):
        """Why a chat may not feed a memory core: '' when it may.

        Incognito never writes anything to disk, and private chats are excluded — but an
        explicit press of the Extract button in a private chat is the user asking for it,
        so 'manual' overrides that one. The two reasons are reported separately because
        they are not the same thing to the user; incognito used to be reported as
        'private', which named the wrong cause."""
        if store.incognito:
            return "incognito"
        if (chat or {}).get("private") and mode != "manual":
            return "private"
        return ""

    # Shown when neither a rewrite model nor a chat model is available to run a pass.
    _NO_MEMORY_MODEL = ("No model available for memory extraction. Set a Rewrite model "
                        "in Settings, or open a chat with a model selected.")

    def _memory_model(chat):
        """(adapter, model) for a memory pass. Reuses the small helper model already
        configured for query rewrite / the Rewrite button, falling back to the chat's
        own model. Returns (None, '') when nothing is usable."""
        model = store.config.get("rewrite_model") or (chat or {}).get("model") or ""
        if not model:
            return None, ""
        server_url = ((chat or {}).get("server_url")
                      or store.config.get("last_server_url") or DEFAULT_LOCAL_URL)
        try:
            return adapter_for(server_url), model
        except Exception:
            return None, ""

    def _empty_summary():
        return {"added": 0, "updated": 0, "merged": 0, "deleted": 0}

    def _memory_ops(adapter, model, messages):
        """One extractor call → ``(ok, operations)``. The single place a memory pass
        talks to a model, so the eval route scores the same call the feature makes.

        ``ok`` is False only when the call itself failed. That is deliberately distinct
        from a call that succeeded and returned nothing: a pass the model never answered
        must not be recorded as a consolidation attempt, or one flaky request would
        suppress compaction until the core grew past its old size.
        """
        try:
            raw = rewrite.run_completion(adapter, model, messages,
                                         num_ctx=8192, max_tokens=1500,
                                         fmt=memory.EXTRACT_SCHEMA)
        except Exception:
            return False, None
        return True, (rewrite._extract_json(raw) or {}).get("operations")

    def _memory_pass(core_id, build_messages, chat=None, after=None):
        """Run one LLM pass and apply the operations it returns. Never raises — a
        failed or unparseable pass is a no-op, because this runs behind an ordinary
        send and must not be able to break it.

        The core is resolved twice by id: once (unlocked) to build the prompt, and again
        under the store lock to apply the result. Holding one reference across the whole
        call let a concurrent entry edit or a delete race the write — and because
        ``delete_memory_entry`` replaced the entries list, an in-flight pass could append
        to a detached one and lose everything it learned. ``build_messages(core)`` and
        the optional ``after(core)`` both run against a freshly resolved core.

        Returns the summary dict; an empty one also covers "the core was deleted"."""
        mc = store.get_memory_core(core_id)
        if not mc:
            return _empty_summary()
        adapter, model = _memory_model(chat)
        if not adapter:
            return _empty_summary()
        # Slow, and deliberately outside the lock: a pass must not block reads.
        ok, ops = _memory_ops(adapter, model, build_messages(mc))
        if not ok:
            return _empty_summary()

        def apply(core):
            summary = memory.apply_operations(core, ops, source_chat=chat)
            if after:
                after(core)
            return summary

        _, summary = store.mutate_memory_core(core_id, apply)
        return summary or _empty_summary()

    def _memory_extract(core_id, chat, mark_built=False):
        """Learn from one chat, then compact the core if that pushed it over the line.
        Both halves go through the same operations vocabulary. Returns (summary, alive)
        — ``alive`` is False once the core has been deleted, which ends a bulk build.

        ``mark_built`` is set only by the retroactive build. The every-N-turns trigger
        deliberately does not mark, because the chat it just read is still going: a later
        build must be free to come back to it once it has grown."""
        marker = (lambda c: _mark_built(c, chat)) if mark_built else None
        transcript = memory.transcript_text(chat)
        if not transcript.strip():
            # An empty chat has nothing to teach — record it as read so a bulk build
            # stops reconsidering it on every run.
            if marker:
                mc, _ = store.mutate_memory_core(core_id, marker)
                return _empty_summary(), mc is not None
            return _empty_summary(), store.get_memory_core(core_id) is not None

        summary = _memory_pass(
            core_id, lambda c: memory.build_extract_messages(c, transcript), chat,
            after=marker)

        mc = store.get_memory_core(core_id)
        if not mc:
            return summary, False
        if memory.needs_consolidation(mc):
            # Record the attempt whether or not it compacts anything — a pass that
            # returns no operations is exactly what the backoff exists to stop repeating.
            extra = _memory_pass(core_id, memory.build_consolidate_messages, chat,
                                 after=memory.mark_consolidated)
            for k, v in extra.items():
                summary[k] = summary.get(k, 0) + v
        return summary, store.get_memory_core(core_id) is not None

    def _mark_built(core, chat):
        """Remember that this core has read this chat, so a re-run of the retroactive
        build doesn't pay for the whole history again."""
        chat_id = (chat or {}).get("id") or ""
        if not chat_id:
            return
        built = core.setdefault("built_chat_ids", [])
        if chat_id not in built:
            built.append(chat_id)

    def _cores_payload():
        """Every core, decorated with what each entry's injection status actually is."""
        return memory.decorate_all(store.memory_cores_snapshot())

    def _core_payload(core_id):
        return memory.decorate_for_client(store.get_memory_core(core_id))

    @app.route("/api/memory/cores", methods=["GET"])
    def api_memory_cores():
        """The tab's refresh path — re-reads on entering Memory, so edits made in
        another browser tab or by a background pass don't leave it stale."""
        return jsonify({"cores": _cores_payload(),
                        "categories": memory.CATEGORIES})

    @app.route("/api/memory/cores", methods=["POST"])
    def api_memory_core_create():
        data = request.get_json(force=True) or {}
        mc = store.add_memory_core(data.get("name", ""))
        return jsonify({"core": memory.decorate_for_client(mc), "cores": _cores_payload()})

    @app.route("/api/memory/cores/<core_id>", methods=["PATCH"])
    def api_memory_core_update(core_id):
        mc = store.update_memory_core(core_id, request.get_json(force=True) or {})
        if not mc:
            return jsonify({"error": "not found"}), 404
        return jsonify({"core": memory.decorate_for_client(mc), "cores": _cores_payload()})

    @app.route("/api/memory/cores/<core_id>", methods=["DELETE"])
    def api_memory_core_delete(core_id):
        if not store.delete_memory_core(core_id):
            return jsonify({"error": "not found"}), 404
        # Deleting a core also clears it off every chat that referenced it, so the
        # browser needs the refreshed summaries, not just the core list.
        return jsonify({"ok": True, "cores": _cores_payload(),
                        "chats": store.chat_summaries()})

    @app.route("/api/memory/cores/<core_id>/entries", methods=["POST"])
    def api_memory_entry_upsert(core_id):
        data = request.get_json(force=True) or {}
        entry, reason = store.upsert_memory_entry(core_id, data.get("entry") or data)
        if entry is None:
            problem = {
                "core": ("memory core not found", 404),
                "gone": ("That memory was deleted somewhere else — nothing to edit.", 409),
                "text": ("A memory needs some text.", 400),
            }.get(reason, ("could not save entry", 400))
            return jsonify({"error": problem[0]}), problem[1]
        return jsonify({"entry": entry, "core": _core_payload(core_id)})

    @app.route("/api/memory/cores/<core_id>/entries/<entry_id>", methods=["DELETE"])
    def api_memory_entry_delete(core_id, entry_id):
        if not store.delete_memory_entry(core_id, entry_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True, "core": _core_payload(core_id)})

    @app.route("/api/memory/extract", methods=["POST"])
    def api_memory_extract():
        """One extraction pass over a chat. ``mode`` is 'auto' (the every-N-turns
        trigger) or 'manual' (the chat button, which may run on a private chat)."""
        data = request.get_json(force=True) or {}
        mode = data.get("mode") or "auto"
        # The browser owns live chat state, so prefer the posted chat over the store's
        # copy — an unsaved private chat has no stored copy at all.
        chat = data.get("chat") or store.get_chat(data.get("chat_id") or "")
        if not chat:
            return jsonify({"error": "chat not found"}), 404
        core_id = data.get("core_id") or ""
        mc = store.get_memory_core(core_id)
        if not mc:
            return jsonify({"error": "memory core not found"}), 404
        skipped = _memory_eligible(chat, mode)
        if skipped:
            return jsonify({"ok": False, "skipped": skipped, "summary": {},
                            "core": memory.decorate_for_client(mc)})
        if not _memory_model(chat)[0]:
            return jsonify({"ok": False, "error": _NO_MEMORY_MODEL,
                            "core": memory.decorate_for_client(mc)})
        summary, alive = _memory_extract(core_id, chat)
        if not alive:
            return jsonify({"ok": False, "error": "That memory core was deleted."}), 404
        # A pass that ran is a completed cycle, so the countdown restarts here rather
        # than in the browser — a reload used to lose the count and two tabs on one chat
        # used to double it.
        stored = store.get_chat((chat or {}).get("id") or "")
        if stored is not None:
            stored["memory_turns_since"] = 0
            store.save_chats()
        return jsonify({"ok": True, "summary": summary,
                        "text": memory.summary_text(summary),
                        "core": _core_payload(core_id)})

    @app.route("/api/memory/cores/<core_id>/consolidate", methods=["POST"])
    def api_memory_consolidate(core_id):
        """Manual refine pass: merge overlapping memories and sharpen wording."""
        mc = store.get_memory_core(core_id)
        if not mc:
            return jsonify({"error": "not found"}), 404
        if store.incognito:
            return jsonify({"ok": False, "skipped": "incognito",
                            "core": memory.decorate_for_client(mc)})
        # No chat is involved in a refine pass, so only the configured helper model
        # (plus the browser's current model, posted as a fallback) can run it.
        data = request.get_json(silent=True) or {}
        chat = {"model": data.get("model") or "",
                "server_url": data.get("server_url")
                or store.config.get("last_server_url") or DEFAULT_LOCAL_URL}
        if not _memory_model(chat)[0]:
            return jsonify({"ok": False, "error": _NO_MEMORY_MODEL,
                            "core": memory.decorate_for_client(mc)})
        summary = _memory_pass(core_id, memory.build_consolidate_messages, chat,
                               after=memory.mark_consolidated)
        if not store.get_memory_core(core_id):
            return jsonify({"error": "That memory core was deleted."}), 404
        return jsonify({"ok": True, "summary": summary,
                        "text": memory.summary_text(summary),
                        "core": _core_payload(core_id)})

    @app.route("/api/memory/cores/<core_id>/build", methods=["POST"])
    def api_memory_build(core_id):
        """SSE. Retroactively grow a core from existing chats — one extraction pass per
        chat, oldest first so later conversations refine what earlier ones established.
        Private chats are skipped (this is the bulk path, never an explicit ask)."""
        mc = store.get_memory_core(core_id)
        if not mc:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True) or {}
        ids = data.get("chat_ids")
        run_id = data.get("run_id")
        chats = [c for c in store.chats if ids is None or c.get("id") in set(ids)]
        chats = [c for c in chats if not _memory_eligible(c, "auto")]
        # Skip what this core has already read unless the user asked to start over —
        # a second run used to re-pay for the entire history.
        skipped = 0
        if not data.get("reread"):
            already = set(mc.get("built_chat_ids") or [])
            before = len(chats)
            chats = [c for c in chats if c.get("id") not in already]
            skipped = before - len(chats)
        chats.sort(key=lambda c: c.get("created") or "")

        def work(emit, stop_event):
            total = len(chats)
            if not total:
                emit("status", {"message": (
                    f"Nothing new to read — all {skipped} chat(s) already learned from."
                    if skipped else "No eligible chats to learn from.")})
                emit("built", {"summary": _empty_summary(), "text": "",
                               "core": _core_payload(core_id)})
                return
            if not _memory_model(chats[0])[0]:
                emit("error", {"message": _NO_MEMORY_MODEL})
                return
            if skipped:
                emit("status", {"message": f"Skipping {skipped} chat(s) already read."})
            # Frame shape matches the shared compile progress UI (makeProgressUI).
            emit("plan", {"phases": [{"id": "chats", "weight": 1}]})
            totals = _empty_summary()
            for i, chat in enumerate(chats):
                if stop_event is not None and stop_event.is_set():
                    break
                emit("status", {"message": f"Reading “{chat.get('title') or 'Untitled'}”…"})
                summary, alive = _memory_extract(core_id, chat, mark_built=True)
                for k, v in summary.items():
                    totals[k] = totals.get(k, 0) + v
                if not alive:
                    # The core was deleted from another tab. Stop rather than spending
                    # the rest of the run on an orphan nobody will ever see.
                    emit("error", {"message": "That memory core was deleted — build stopped."})
                    return
                emit("progress", {"phase": "chats", "label": "reading chats",
                                  "unit": "chats", "done": i + 1, "total": total})
            emit("built", {"summary": totals, "text": memory.summary_text(totals),
                           "core": _core_payload(core_id)})

        return _compile_sse(work, run_id)

    # ------------------- scoring the extractor itself -------------------
    # Everything above tests that the memory pipeline *works*. This measures whether it
    # remembers the RIGHT things — the one question a stub model can never answer.

    def _eval_core_from(text):
        """A throwaway core seeded with one memory per line, as the tab lists them.
        Never enters the store: an eval must not leave anything behind."""
        mc = memory.new_core("evaluation")
        for line in (text or "").splitlines():
            line = line.strip()
            if line:
                mc["entries"].append(memory.new_entry(line, origin="user"))
        return mc

    def _eval_response_text(ops, mc, summary):
        """What the judge is shown: the operations the extractor chose, and the profile
        they produced. Both, because a plausible-looking operation list can still add up
        to a bad profile."""
        return (
            "Operations returned by the extractor:\n"
            + json.dumps(ops if isinstance(ops, list) else [], indent=2)
            + "\n\nResulting memory profile:\n"
            + (memory.render_core_for_prompt(mc) or "(no memories)")
            + f"\n\nCounts: {memory.summary_text(summary) or 'nothing changed'}"
        )

    @app.route("/api/memory/eval", methods=["POST"])
    def api_memory_eval():
        """SSE. Score the memory extractor against a judge model.

        Each row is a transcript plus the memories already held. The row runs through the
        *production* extractor — ``build_extract_messages`` → ``_memory_ops`` (schema and
        all) → ``apply_operations`` — against a scratch core that is never stored, and the
        judge then grades what came out.

        Deliberately not the generic eval runner: that builds a single user message with
        no schema and routes it through ``generate_one``, so it would score a hand-copied
        paraphrase of the extractor prompt that drifts the moment ``_EXTRACT_SYS`` is
        edited. The scoring half of ``app/evals.py`` is reused verbatim.
        """
        data = request.get_json(force=True) or {}
        project = data.get("eval") or {}
        run_id = data.get("run_id")
        rows = project.get("rows") or []
        criteria = project.get("criteria") or evals.MEMORY_CRITERIA
        if not rows:
            return jsonify({"error": "Add at least one transcript to score."}), 400
        if not project.get("grader_model"):
            return jsonify({"error": "Pick a grader model."}), 400
        if not project.get("gen_model"):
            return jsonify({"error": "Pick the model whose extraction you want scored."}), 400
        graded_project = {**project, "criteria": criteria}

        def work(emit, stop_event):
            try:
                gen_adapter = adapter_for(project.get("gen_server_url") or DEFAULT_LOCAL_URL)
                grader_adapter = adapter_for(project.get("grader_server_url")
                                             or project.get("gen_server_url")
                                             or DEFAULT_LOCAL_URL)
            except Exception as e:
                emit("error", {"message": f"Could not reach a model server: {e}"})
                return
            emit("plan", {"phases": [{"id": "rows", "weight": 1}]})
            all_grades = []
            for i, row in enumerate(rows):
                if stop_event is not None and stop_event.is_set():
                    break
                transcript = (row.get("Transcript") or "").strip()
                emit("status", {"message": f"Row {i + 1} of {len(rows)}: extracting…"})
                if not transcript:
                    emit("row_result", {"row": i, "ungraded": True,
                                        "note": "empty transcript"})
                    emit("progress", {"phase": "rows", "label": "rows", "unit": "rows",
                                      "done": i + 1, "total": len(rows)})
                    continue

                mc = _eval_core_from(row.get("ExistingMemories"))
                messages = memory.build_extract_messages(mc, transcript)
                # An extractor that fails is still a result worth grading: it recorded
                # nothing, which the judge scores like any other empty answer.
                _ok, ops = _memory_ops(gen_adapter, project["gen_model"], messages)
                summary = memory.apply_operations(mc, ops)
                task = messages[-1].get("content", "")
                response = _eval_response_text(ops, mc, summary)

                emit("status", {"message": f"Row {i + 1} of {len(rows)}: grading…"})
                try:
                    raw = rewrite.run_completion(
                        grader_adapter, project["grader_model"],
                        evals.build_grader_messages(graded_project, task, response),
                        num_ctx=8192, max_tokens=1200, stop=stop_event)
                    grades = evals.normalize_grades(evals.parse_grader_json(raw), criteria)
                except Exception as e:
                    emit("status", {"message": f"Row {i + 1} could not be graded: {e}"})
                    grades = evals.normalize_grades({}, criteria)

                scored = any(g.get("score") is not None for g in grades.values())
                if scored:
                    all_grades.append(grades)
                emit("row_result", {
                    "row": i, "ungraded": not scored, "grades": grades,
                    "operations": ops if isinstance(ops, list) else [],
                    "summary": summary, "note": row.get("Note", ""),
                    "profile": memory.render_core_for_prompt(mc),
                })
                emit("progress", {"phase": "rows", "label": "rows", "unit": "rows",
                                  "done": i + 1, "total": len(rows)})

            emit("summary", {"aggregate": evals.aggregate(all_grades, criteria),
                             "rows": len(rows)})

        return _compile_sse(work, run_id)

    @app.route("/api/memory/eval/seed", methods=["GET"])
    def api_memory_eval_seed():
        """The starting dataset and criteria for the extractor eval."""
        return jsonify({"rows": [dict(r) for r in evals.MEMORY_SEED_ROWS],
                        "criteria": [dict(c) for c in evals.MEMORY_CRITERIA]})

    @app.route("/api/memory/cores/export", methods=["POST"])
    def api_memory_export():
        """Export all cores (scope='all') or a subset (ids=[...]) to a JSON file the
        user picks. Exports stay plaintext by design, like chat exports."""
        data = request.get_json(force=True) or {}
        ids = None if data.get("scope") == "all" else (data.get("ids") or None)
        envelope = store.export_memory_cores(ids)
        cores = envelope.get("cores") or []
        default_name = ("all_memory_cores.json" if ids is None or len(cores) != 1
                        else (cores[0].get("name") or "memory_core").strip().replace(" ", "_") + ".json")
        dest = native_dialog.save_file(title="Export memory cores as JSON",
                                       default_name=default_name, filetypes_key="json")
        if not dest:
            return jsonify({"ok": False, "cancelled": True})
        try:
            if not dest.lower().endswith(".json"):
                dest += ".json"
            Path(dest).write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": dest, "count": len(cores)})

    @app.route("/api/memory/cores/import", methods=["POST"])
    def api_memory_import():
        """Import cores from a JSON export. Fresh ids throughout, so importing never
        overwrites a core the user already has."""
        paths = native_dialog.pick_files(title="Import memory cores (JSON)",
                                         filetypes_key="json")
        if not paths:
            return jsonify({"ok": False, "cancelled": True})
        path = Path(paths[0])
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return jsonify({"ok": False, "error": f"Could not read file: {e}"}), 400
        try:
            imported, count = store.import_memory_cores(envelope)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "count": count, "cores": _cores_payload(),
                        "imported": [c.get("id") for c in imported]})

    # ------------------- Multi-server parallel processing -------------------
    def _parallel_lanes():
        """Resolve the saved parallel_servers into lane dicts (with credentials)."""
        lanes = []
        for s in (store.config.get("parallel_servers") or []):
            srv = store.resolve_server(s.get("base_url"))
            lanes.append({"base_url": srv["base_url"], "model": (s.get("model") or "").strip(),
                          "name": srv.get("name") or srv["base_url"]})
        return lanes

    def _parallel_sse(items, run_id, on_item_done=None, lanes=None, mode=None,
                      stop_event=None):
        """Shared SSE generator: fan `items` across the saved lanes and multiplex
        every frame into one stream. on_item_done(frame) runs server-side as each
        item finishes (e.g. to write a batch response to disk) before forwarding.

        ``lanes``/``mode`` override the saved multi-server config. The Batch tab uses
        this to run through a SINGLE synthetic lane when multi-server processing is
        off — a one-lane run_parallel is exactly the sequential case, which means batch
        needs no second generation loop of its own. ``stop_event`` lets a caller that
        already registered the run (e.g. to resolve sources first) reuse it instead of
        registering a second time."""
        if stop_event is None:
            stop_event = runs.new(run_id)
        if lanes is None:
            lanes = _parallel_lanes()
        if mode is None:
            mode = store.config.get("parallel_mode", "balanced")
        frames = queue.Queue()
        sentinel = object()

        def work():
            try:
                parallel.run_parallel(items, lanes, mode, stop_event, generate_one, frames.put)
            finally:
                frames.put(sentinel)

        threading.Thread(target=work, daemon=True).start()
        try:
            yield sse("start", {"run_id": run_id, "mode": mode, "total": len(items),
                                "lanes": [{"index": i, "name": l["name"],
                                           "server": l["base_url"], "model": l["model"]}
                                          for i, l in enumerate(lanes)]})
            while True:
                frame = frames.get()
                if frame is sentinel:
                    break
                if frame.get("event") == "item_done" and on_item_done:
                    try:
                        on_item_done(frame)
                    except Exception:
                        pass
                event = frame.pop("event", "message")
                yield sse(event, frame)
            yield sse("done", {"stopped": stop_event.is_set()})
        finally:
            runs.done(run_id)

    @app.route("/api/parallel/config", methods=["GET"])
    def api_parallel_config_get():
        return jsonify({
            "parallel_enabled": store.config.get("parallel_enabled", False),
            "parallel_mode": store.config.get("parallel_mode", "balanced"),
            "parallel_servers": store.config.get("parallel_servers", []),
        })

    @app.route("/api/parallel/config", methods=["PUT"])
    def api_parallel_config_put():
        data = request.get_json(force=True) or {}
        return jsonify(store.set_parallel_config(data))

    @app.route("/api/parallel/common-model")
    def api_parallel_common_model():
        """Return the models installed on ALL of the given servers (comma-separated
        base_urls), for the 'Use common model' button."""
        raw = request.args.get("servers", "")
        urls = [u.strip() for u in raw.split(",") if u.strip()]

        def list_models(url):
            return adapter_for(url).list_models()

        return jsonify({"models": parallel.common_models(urls, list_models)})

    @app.route("/api/parallel/start", methods=["POST"])
    def api_parallel_start():
        """Body: {items:[{item_id, chat, search_query, title}], run_id}. Streams a
        multiplexed SSE (frames tagged lane + item_id) as every queued item is
        processed across the selected servers at once."""
        data = request.get_json(force=True) or {}
        items = data.get("items") or []
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]
        if not _parallel_lanes():
            return jsonify({"error": "No servers are selected for parallel processing (Settings → Multi-Server Processing)."}), 400
        if not items:
            return jsonify({"error": "The queue is empty."}), 400
        return Response(stream_with_context(_parallel_sse(items, run_id)),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------------- Presets ----------------------------------
    @app.route("/api/presets", methods=["GET"])
    def api_presets_get():
        return jsonify({"presets": store.presets})

    @app.route("/api/presets", methods=["POST"])
    def api_preset_upsert():
        data = request.get_json(force=True) or {}
        name = (data.get("name") or "").strip()
        prompt = data.get("prompt", "")
        if not name:
            return jsonify({"error": "name required"}), 400
        return jsonify({"presets": store.upsert_preset(name, prompt)})

    @app.route("/api/presets/<path:name>", methods=["DELETE"])
    def api_preset_delete(name):
        return jsonify({"presets": store.delete_preset(name)})

    # ---------------------- Prompt library (System/Pre) ---------------------
    @app.route("/api/prompts", methods=["GET"])
    def api_prompts_get():
        return jsonify({"prompts": store.prompts})

    @app.route("/api/prompts", methods=["POST"])
    def api_prompts_save():
        """Whole-tree save (the client mutates its local copy and pushes it back)."""
        data = request.get_json(force=True) or {}
        return jsonify({"prompts": store.set_prompts(data.get("prompts") or {})})

    @app.route("/api/prompts/import-xml", methods=["POST"])
    def api_prompts_import_xml():
        """Open a native picker for one or more prompt XML files and merge each into its
        declared tree (auto-renaming clashing prompts so nothing is overwritten)."""
        paths = native_dialog.pick_files(title="Import prompt XML", filetypes_key="xml")
        imported = 0
        errors = []
        for p in paths:
            try:
                kind, groups = prompts_from_xml_file(p)
                store.merge_prompts_tree(kind, groups)
                imported += 1
            except Exception as e:
                errors.append(f"{Path(p).name}: {e}")
        return jsonify({"imported": imported, "errors": errors, "prompts": store.prompts})

    @app.route("/api/prompts/export-xml", methods=["POST"])
    def api_prompts_export_xml():
        """Open a native save dialog and write the selected subtree (group / category /
        single prompt) to the chosen path as XML."""
        data = request.get_json(force=True) or {}
        kind = "pre" if data.get("kind") == "pre" else "system"
        groups = store.slice_prompts(kind, data.get("group_id"),
                                     data.get("category_id"), data.get("prompt_id"))
        if not groups:
            return jsonify({"ok": False, "error": "nothing selected to export"}), 400
        default_name = (data.get("default_name") or "prompts").strip().replace(" ", "_") + ".xml"
        dest = native_dialog.save_file(title="Save prompts as XML",
                                       default_name=default_name, filetypes_key="xml")
        if not dest:
            return jsonify({"ok": False, "cancelled": True})
        try:
            if not dest.lower().endswith(".xml"):
                dest += ".xml"
            Path(dest).write_bytes(prompts_to_xml_bytes(kind, groups))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": dest})

    # --------------------------- Web search config --------------------------
    @app.route("/api/websearch/config", methods=["POST"])
    def api_websearch_config():
        data = request.get_json(force=True) or {}
        patch = {}
        if "approved_domains" in data:
            patch["approved_domains"] = [
                _normalize_domain(d) for d in data["approved_domains"] if _normalize_domain(d)
            ]
        if "restrict_to_approved" in data:
            patch["restrict_to_approved"] = bool(data["restrict_to_approved"])
        if "auto_detect_reasoning" in data:
            patch["auto_detect_reasoning"] = bool(data["auto_detect_reasoning"])
        if "default_num_ctx" in data:
            patch["default_num_ctx"] = int(data["default_num_ctx"])
        store.update_config(patch)
        return jsonify({"config": store.masked_config()})

    # ----------------------------- Libraries --------------------------------
    def _append_items(lib_id, items):
        """Atomically append freshly fetched items to a library, re-reading it under
        the store lock. The add routes run for minutes; writing back the dict they read
        at request start clobbered any autosave that landed in between. Returns the
        updated library, or None if it was deleted mid-flight."""
        if not items:
            return store.get_library(lib_id)
        return store.append_library_items(lib_id, items)

    @app.route("/api/libraries", methods=["GET"])
    def api_libraries_get():
        return jsonify({"libraries": store.libraries})

    @app.route("/api/libraries", methods=["POST"])
    def api_library_create():
        data = request.get_json(force=True) or {}
        lib = _new_library(data.get("name", "New Library"))
        store.add_library(lib)
        return jsonify({"library": lib, "libraries": store.libraries})

    @app.route("/api/libraries/<lib_id>", methods=["PUT"])
    def api_library_update(lib_id):
        data = request.get_json(force=True) or {}
        lib = data.get("library") or {}
        lib["id"] = lib_id
        lib["updated"] = datetime.utcnow().isoformat()
        store.upsert_library(lib)
        return jsonify({"library": lib, "libraries": store.libraries})

    @app.route("/api/libraries/<lib_id>", methods=["DELETE"])
    def api_library_delete(lib_id):
        # delete_library also strips the id from every chat's library_ids — a dangling
        # reference would keep RAG active for those chats with nothing to retrieve.
        store.delete_library(lib_id)
        rag.delete_source(lib_id)          # drop this library's vectors
        compile_mod.forget_library(lib_id) # drop its compile manifest
        return jsonify({"libraries": store.libraries})

    @app.route("/api/libraries/<lib_id>/add-text-files", methods=["POST"])
    def api_library_add_text_files(lib_id):
        """SSE. Open a native multi-file picker, parse the chosen documents in parallel
        and append them as library items. Handles plain text plus PDF/EPUB/DOCX via
        app.ingest.

        Streams ``progress`` frames as files land and a final ``complete`` frame with
        the updated library. Parsing a shelf of ebooks is minutes of CPU, so it runs
        across a process pool and reports as it goes rather than blocking one request
        thread in silence.
        """
        lib = store.get_library(lib_id)
        if not lib:
            return jsonify({"error": "not found"}), 404
        # The native picker must stay on the request thread (tkinter), before the
        # stream starts.
        paths = native_dialog.pick_files(
            title="Add document(s) to library", filetypes_key="documents")
        errors = []
        wanted = []
        for p in paths:
            path = Path(p)
            if not ingest.is_supported(path):
                errors.append(f"{path.name}: unsupported type")
            else:
                wanted.append(str(path))

        # Unique per invocation. A fixed "addfiles-library-<id>" meant two runs on one
        # library shared a registry slot: RunRegistry.new OVERWRITES, so the first run's
        # stop event was orphaned and unreachable, and whichever generator finished
        # first deregistered the other's. Clients read the id off _compile_sse's own
        # `start` frame rather than composing it.
        run_id = f"addfiles-library-{lib_id}-{uuid.uuid4().hex[:8]}"

        def work(emit, stop_event):
            emit("begin", {"total": len(wanted), "name": lib.get("name", "")})
            added, items = [], []
            errs = list(errors)
            if wanted:
                results = ingest.extract_many(
                    wanted,
                    on_progress=lambda d, t, name: emit(
                        "progress", {"phase": "parse", "done": d, "total": t,
                                     "unit": "files", "name": name}),
                    should_stop=lambda: bool(stop_event and stop_event.is_set()))
                for res in results:
                    name = Path(res.get("path", "")).name
                    if not res.get("ok"):
                        errs.append(f"{name}: {res.get('error', 'parse failed')}")
                        continue
                    items.append(_new_library_item(item_type="file", label=name,
                                                   content=res.get("text") or "",
                                                   filename=name))
                    added.append(name)
            saved = _append_items(lib_id, items)
            if saved is None:
                emit("error", {"message": "That library was deleted while the files "
                                          "were being read."})
                return
            emit("complete", {"library": saved, "added": added,
                              "added_items": items, "errors": errs})

        return _compile_sse(work, run_id)

    @app.route("/api/libraries/<lib_id>/add-url", methods=["POST"])
    def api_library_add_url(lib_id):
        """Body: {url}. Scrape the page (Bright Data → Playwright → requests) and
        append the readable text as an editable 'url' library item. The source URL is
        stored in the item's ``filename`` so it round-trips through XML export/import.
        """
        lib = store.get_library(lib_id)
        if not lib:
            return jsonify({"error": "not found"}), 404
        data = request.get_json(force=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "no url"}), 400
        try:
            page = core.fetch_url_text(url)
        except Exception as e:
            return jsonify({"library": lib, "added": [], "errors": [str(e)]})
        item = _new_library_item(item_type="url", label=page["title"],
                                 content=page["text"], filename=page["url"])
        saved = _append_items(lib_id, [item])
        if saved is None:
            return jsonify({"error": "that library was deleted"}), 404
        return jsonify({"library": saved, "added": [page["title"]],
                        "added_items": [item], "errors": [],
                        "via": page.get("via")})

    @app.route("/api/libraries/<lib_id>/brave-search", methods=["GET"])
    def api_library_brave_search(lib_id):
        """SSE. Query params: q (search term), sites (comma-separated, optional),
        max (pages to successfully crawl). Discovers URLs via Brave, crawls each with
        the fallback chain, streams per-page progress, then appends one 'url' item per
        crawled page and emits a final ``complete`` frame with the updated library.
        """
        lib = store.get_library(lib_id)
        if not lib:
            return jsonify({"error": "not found"}), 404
        query = (request.args.get("q") or "").strip()
        sites = [s.strip() for s in (request.args.get("sites") or "").split(",") if s.strip()]
        try:
            max_results = int(request.args.get("max") or 5)
        except ValueError:
            max_results = 5
        # Clamped here as well as in the browser: the <input max> attribute is not
        # enforced against a typed value, and a mistyped 500 is 500 page fetches.
        max_results = max(1, min(core.MAX_CRAWL_PAGES, max_results))

        def work(emit, stop_event):
            pages, errors, attempted = [], [], 0
            for ev in core.crawl_search(query, sites=sites, max_results=max_results,
                                        should_stop=lambda: bool(stop_event and stop_event.is_set())):
                if ev.get("type") == "progress":
                    emit("progress", ev)
                elif ev.get("type") == "result":
                    pages = ev.get("pages") or []
                    errors = ev.get("errors") or []
                    attempted = ev.get("attempted") or 0
            added, items = [], []
            for p in pages:
                items.append(_new_library_item(item_type="url", label=p["title"],
                                               content=p["text"], filename=p["url"]))
                added.append(p["title"])
            saved = _append_items(lib_id, items)
            if saved is None:
                emit("error", {"message": "That library was deleted while the crawl "
                                          "was running."})
                return
            emit("complete", {"library": saved, "added": added, "added_items": items,
                              "attempted": attempted, "errors": errors})
        # A run_id is what gives this stream a stop event at all: without one
        # _compile_sse passes stop_event=None, so should_stop above was permanently
        # False and neither the Stop button nor a client disconnect could end the crawl.
        # Unique per invocation (see the add-files route) — the client takes the id from
        # the `start` frame instead of composing it from the library id.
        return _compile_sse(work, f"brave-library-{lib_id}-{uuid.uuid4().hex[:8]}")

    @app.route("/api/libraries/import-xml", methods=["POST"])
    def api_library_import_xml():
        """Open a native picker for one or more XML files and import each."""
        paths = native_dialog.pick_files(title="Load library XML", filetypes_key="xml")
        imported = []
        errors = []
        for p in paths:
            try:
                lib = library_from_xml_file(p)
                store.add_library(lib)
                imported.append(lib)
            except Exception as e:
                errors.append(f"{Path(p).name}: {e}")
        return jsonify({"imported": imported, "errors": errors, "libraries": store.libraries})

    @app.route("/api/libraries/<lib_id>/export-xml", methods=["POST"])
    def api_library_export_xml(lib_id):
        """Open a native save dialog and write the library XML to the chosen path."""
        data = request.get_json(force=True) or {}
        lib = data.get("library")
        if not lib or lib.get("id") != lib_id:
            lib = store.get_library(lib_id)
        if not lib:
            return jsonify({"error": "not found"}), 404
        default_name = (lib.get("name") or "library").strip().replace(" ", "_") + ".xml"
        dest = native_dialog.save_file(title="Save library as XML",
                                       default_name=default_name, filetypes_key="xml")
        if not dest:
            return jsonify({"ok": False, "cancelled": True})
        try:
            if not dest.lower().endswith(".xml"):
                dest += ".xml"
            Path(dest).write_bytes(library_to_xml_bytes(lib))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": dest})

    # ----------------------------- YouTube ----------------------------------
    def _youtube_params():
        """Read the shared query params for both YouTube routes.
        Returns (url, include_comments, max_comments)."""
        url = (request.args.get("url") or "").strip()
        include_comments = (request.args.get("comments") or "1") not in ("0", "false", "")
        try:
            max_comments = int(request.args.get("max") or youtube.DEFAULT_MAX_COMMENTS)
        except ValueError:
            max_comments = youtube.DEFAULT_MAX_COMMENTS
        return url, include_comments, max_comments

    def _youtube_fetch_work(url, include_comments, max_comments, on_complete):
        """Build the SSE worker shared by the library and chat YouTube routes.

        ``on_complete(emit, result)`` decides what happens with the fetched video —
        appending it to a library, or just handing the text back to the composer.
        """
        def work(emit, stop_event):
            emit("begin", {"url": url, "comments": include_comments,
                           "max": max_comments})
            result = youtube.fetch_video(
                url,
                include_comments=include_comments,
                max_comments=max_comments,
                on_progress=lambda phase, **fields: emit(
                    "progress", {"phase": phase, **fields}),
                should_stop=lambda: bool(stop_event and stop_event.is_set()),
            )
            on_complete(emit, result)
        return work

    @app.route("/api/libraries/<lib_id>/add-youtube", methods=["GET"])
    def api_library_add_youtube(lib_id):
        """SSE (EventSource is GET-only). Query params: url, comments (0/1), max.

        Fetches a video's transcript and comments and appends them as a single
        'youtube' library item. The watch URL goes in the item's ``filename`` exactly
        like 'url' items do, so it round-trips through XML export/import.
        """
        lib = store.get_library(lib_id)
        if not lib:
            return jsonify({"error": "not found"}), 404
        url, include_comments, max_comments = _youtube_params()
        if not url:
            return jsonify({"error": "no url"}), 400

        def on_complete(emit, result):
            item = _new_library_item(item_type="youtube", label=result["title"],
                                     content=result["text"], filename=result["url"])
            saved = _append_items(lib_id, [item])
            if saved is None:
                emit("error", {"message": "That library was deleted while the video "
                                          "was being fetched."})
                return
            emit("complete", {"library": saved, "added": [result["title"]],
                              "added_items": [item],
                              "via": result.get("via"), "errors": result.get("errors") or [],
                              "comment_count": len(result.get("comments") or []),
                              "transcript_chars": len(result.get("transcript") or "")})

        # Unique per invocation (see the add-files route); the client reads the id off
        # the `start` frame rather than composing it from the library id, which it could
        # only do from whichever library happened to be selected when Cancel was pressed.
        return _compile_sse(
            _youtube_fetch_work(url, include_comments, max_comments, on_complete),
            f"youtube-library-{lib_id}-{uuid.uuid4().hex[:8]}")

    @app.route("/api/youtube/fetch", methods=["GET"])
    def api_youtube_fetch():
        """SSE. Same params as the library route, but writes nothing — the composer
        stages the returned text as a chat attachment instead."""
        url, include_comments, max_comments = _youtube_params()
        if not url:
            return jsonify({"error": "no url"}), 400

        def on_complete(emit, result):
            emit("complete", {"title": result["title"], "url": result["url"],
                              "text": result["text"], "via": result.get("via"),
                              "errors": result.get("errors") or [],
                              "comment_count": len(result.get("comments") or []),
                              "transcript_chars": len(result.get("transcript") or "")})

        return _compile_sse(
            _youtube_fetch_work(url, include_comments, max_comments, on_complete),
            f"youtube-chat-{uuid.uuid4().hex[:8]}")

    # ------------------ Chat-scoped sources (no library write) --------------
    # The composer offers the same sources as the Resources tab, for material that
    # belongs to one conversation rather than a reusable library. Each route is its
    # library counterpart with the item-append tail removed.

    @app.route("/api/fetch-url", methods=["POST"])
    def api_fetch_url():
        """Body: {url}. Scrape a page to readable text and return it un-stored."""
        url = ((request.get_json(force=True) or {}).get("url") or "").strip()
        if not url:
            return jsonify({"error": "no url"}), 400
        try:
            page = core.fetch_url_text(url)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(page)

    @app.route("/api/extract-files", methods=["POST"])
    def api_extract_files():
        """SSE. Native multi-file picker + parallel document parsing, returning the
        extracted text rather than writing it anywhere."""
        paths = native_dialog.pick_files(
            title="Attach document(s) to this chat", filetypes_key="documents")
        errors = []
        wanted = []
        for p in paths:
            path = Path(p)
            if not ingest.is_supported(path):
                errors.append(f"{path.name}: unsupported type")
            else:
                wanted.append(str(path))

        def work(emit, stop_event):
            emit("begin", {"total": len(wanted)})
            docs = []
            errs = list(errors)
            if wanted:
                results = ingest.extract_many(
                    wanted,
                    on_progress=lambda d, t, name: emit(
                        "progress", {"phase": "parse", "done": d, "total": t,
                                     "unit": "files", "name": name}))
                for res in results:
                    name = Path(res.get("path", "")).name
                    if not res.get("ok"):
                        errs.append(f"{name}: {res.get('error', 'parse failed')}")
                        continue
                    docs.append({"title": res.get("title") or name,
                                 "filename": name,
                                 "text": res.get("text") or ""})
            emit("complete", {"docs": docs, "errors": errs})

        return _compile_sse(work, f"extract-files-{uuid.uuid4().hex[:8]}")

    @app.route("/api/brave-search-text", methods=["GET"])
    def api_brave_search_text():
        """SSE. Query params: q, sites, max. Same discovery + crawl as the library
        Brave search, returning the pages instead of storing them."""
        query = (request.args.get("q") or "").strip()
        sites = [s.strip() for s in (request.args.get("sites") or "").split(",") if s.strip()]
        try:
            max_results = int(request.args.get("max") or 5)
        except ValueError:
            max_results = 5
        # Same clamp as the Resources-tab crawl, and for the same reason: the input's
        # `max` attribute is not enforced against a typed value, and crawl_search only
        # bounds this from below.
        max_results = max(1, min(core.MAX_CRAWL_PAGES, max_results))

        def work(emit, stop_event):
            pages, errors, attempted = [], [], 0
            for ev in core.crawl_search(query, sites=sites, max_results=max_results,
                                        should_stop=lambda: bool(stop_event and stop_event.is_set())):
                if ev.get("type") == "progress":
                    emit("progress", ev)
                elif ev.get("type") == "result":
                    pages = ev.get("pages") or []
                    errors = ev.get("errors") or []
                    attempted = ev.get("attempted") or 0
            emit("complete", {"pages": pages, "attempted": attempted, "errors": errors})

        return _compile_sse(work, f"brave-chat-{uuid.uuid4().hex[:8]}")

    # ----------------------------- Native dialogs ---------------------------
    @app.route("/api/pick-folder", methods=["POST"])
    def api_pick_folder():
        data = request.get_json(silent=True) or {}
        path = native_dialog.pick_folder(title=data.get("title", "Choose a folder"))
        return jsonify({"path": path})

    # ----------------------------- Images -----------------------------------
    # Images are the one thing the browser actually uploads. Everything else in this
    # app exchanges paths and lets the server read the disk, but a screenshot pasted
    # from the clipboard and an image dragged onto the composer have no path to send
    # — so /api/images/upload takes bytes, and /api/images/pick keeps the native
    # dialog available for picking files that do exist on disk.

    def _upload_max_dim():
        """Long-edge clamp for an upload: whatever the client asked for, else the
        configured default. Storing a downscaled copy is not the same as sending one
        — this is the user explicitly choosing not to keep the full-size original."""
        raw = (request.form.get("max_dim") or "").strip()
        if raw:
            try:
                return max(0, min(8192, int(raw)))
            except ValueError:
                pass
        return 0        # keep the original; the send-time clamp still applies

    @app.route("/api/images/upload", methods=["POST"])
    def api_images_upload():
        """Body: multipart/form-data with a repeated ``files`` field (+ optional
        ``max_dim``). One bad file contributes an error string rather than failing
        the whole drop."""
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files in the request."}), 400
        max_dim = _upload_max_dim()
        records, errors = [], []
        for f in files:
            name = f.filename or "pasted-image"
            try:
                data = f.read()
                if not data:
                    raise images_mod.ImageError("file is empty")
                prep = images_mod.prepare(data, f.mimetype or "", max_dim)
                rec = images_mod.store_prepared(prep, name=name)
                if prep.get("note"):
                    rec["note"] = prep["note"]
                records.append(rec)
            except images_mod.ImageError as e:
                errors.append(f"{name}: {e}")
            except Exception as e:
                errors.append(f"{name}: {e}")
        return jsonify({"images": records, "errors": errors})

    @app.route("/api/images/pick", methods=["POST"])
    def api_images_pick():
        """SSE. Native multi-file picker + prepare/store, mirroring
        /api/extract-files. Body: {max_dim} (optional)."""
        body = request.get_json(silent=True) or {}
        try:
            max_dim = max(0, min(8192, int(body.get("max_dim") or 0)))
        except (TypeError, ValueError):
            max_dim = 0
        paths = native_dialog.pick_files(
            title="Attach image(s) to this chat", filetypes_key="images")
        errors = []
        wanted = []
        for p in paths:
            path = Path(p)
            if not images_mod.is_supported(path):
                errors.append(f"{path.name}: not an image type we can read")
            else:
                wanted.append(path)

        def work(emit, stop_event):
            emit("begin", {"total": len(wanted)})
            records, errs = [], list(errors)
            for n, path in enumerate(wanted, 1):
                if stop_event and stop_event.is_set():
                    break
                emit("progress", {"phase": "images", "done": n, "total": len(wanted),
                                  "unit": "images", "name": path.name})
                try:
                    records.append(images_mod.store_file(path, max_dim=max_dim))
                except Exception as e:
                    errs.append(f"{path.name}: {e}")
            emit("complete", {"images": records, "errors": errs})

        return _compile_sse(work, f"pick-images-{uuid.uuid4().hex[:8]}")

    @app.route("/api/images/<image_id>", methods=["GET"])
    def api_image_get(image_id):
        """Raw image bytes (``?thumb=1`` for a small PNG).

        Reads through core.read_bytes so the stored file is decrypted on the way
        out — send_file would hand the browser ciphertext."""
        try:
            if request.args.get("thumb"):
                data, media_type = images_mod.thumb(image_id)
            else:
                data, media_type = images_mod.load(image_id)
        except images_mod.ImageError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        resp = Response(data, mimetype=media_type)
        # Ids are minted per stored image and never reused, so the bytes behind one
        # can't change. Private, because this is somebody's photo.
        resp.headers["Cache-Control"] = "private, max-age=31536000, immutable"
        return resp

    @app.route("/api/images/<image_id>/save", methods=["POST"])
    def api_image_save(image_id):
        """Write an image to a user-chosen path via the native Save dialog."""
        data = request.get_json(silent=True) or {}
        try:
            _bytes, media_type = images_mod.load(image_id)
        except images_mod.ImageError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        default_name = (data.get("default_name") or "image").strip()
        ext = images_mod.ext_for(media_type)
        if not default_name.lower().endswith(ext):
            default_name = Path(default_name).stem + ext
        dest = native_dialog.save_file(title="Save image as", default_name=default_name)
        if not dest:
            return jsonify({"ok": False, "cancelled": True})
        if not Path(dest).suffix:
            dest = dest + ext
        try:
            path = images_mod.write_out(image_id, dest)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": str(path)})

    # ----------------------------- Batch ------------------------------------
    @app.route("/api/batch/start", methods=["POST"])
    def api_batch_start():
        """Body: {chat: <full chat dict for settings>, folder: <path>, run_id}.
        Streams per-file progress + prompt/response pairs. Writes each cleaned
        response to <folder>/responses/<name> on disk.
        """
        data = request.get_json(force=True) or {}
        chat = data.get("chat") or {}
        folder = data.get("folder") or ""
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]

        folder_path = Path(folder) if folder else None
        if not folder_path or not folder_path.is_dir():
            return jsonify({"error": "Folder does not exist."}), 400
        if not chat.get("model"):
            return jsonify({"error": "No model selected."}), 400

        files = sorted(
            (p for p in folder_path.iterdir()
             if p.is_file() and p.suffix.lower() in (".txt", ".md")),
            key=lambda p: p.name.lower(),
        )
        if not files:
            return jsonify({"error": "No .txt or .md prompt files were found in that folder."}), 400

        try:
            (folder_path / "responses").mkdir(exist_ok=True)
        except Exception as e:
            return jsonify({"error": f"Could not create responses folder: {e}"}), 500

        # Parallel batch: distribute the files across the selected servers instead
        # of the sequential loop below. Each file is an independent single-turn
        # chat (no shared history), reusing generate_one; responses are written to
        # <folder>/responses/<name> as each item completes.
        if store.config.get("parallel_enabled") and _parallel_lanes():
            responses_dir = folder_path / "responses"
            items = []
            for path in files:
                try:
                    prompt = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    prompt = ""
                if not prompt.strip():
                    continue
                item_chat = dict(chat)
                item_chat["messages"] = [{"role": "user", "content": prompt}]
                item_chat["isolated"] = True   # each file is independent in parallel mode
                items.append({"item_id": path.name, "title": path.name,
                              "search_query": "", "chat": item_chat})
            if not items:
                return jsonify({"error": "No non-empty .txt or .md prompt files were found."}), 400

            def _write_response(frame):
                name = frame.get("item_id") or ""
                raw = frame.get("content") or ""
                clean = strip_markdown(raw) or raw
                try:
                    (responses_dir / name).write_text(clean, encoding="utf-8")
                except Exception as e:
                    clean = clean + f"\n\n[Warning: could not write response file: {e}]"
                frame["response"] = clean
                frame["responses_dir"] = str(responses_dir)

            return Response(
                stream_with_context(_parallel_sse(items, run_id, on_item_done=_write_response)),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # Snapshot settings (mirrors _on_batch_process).
        server_url = chat.get("server_url") or DEFAULT_LOCAL_URL
        tools_supported = model_supports_tools(server_url, chat.get("model"))
        web_on = bool(chat.get("web_search", False))
        allowed_domains = None
        if store.config.get("restrict_to_approved") and store.config.get("approved_domains"):
            allowed_domains = list(store.config["approved_domains"])
        snap_sys, snap_pre = logic._resolve_prompts(chat)
        snap = {
            "model": chat.get("model"),
            "system_prompt": snap_sys,
            "pre": snap_pre,
            "lib_ctx": logic.build_library_context(chat, store.libraries),      # full XML dump (fallback)
            "lib_manifest": logic.build_library_manifest(chat, store.libraries),  # XML index (RAG path)
            "isolated": bool(chat.get("isolated", False)),
            "think": model_is_reasoning(server_url, chat.get("model")),
            "num_ctx": chat.get("num_ctx") or store.config.get("default_num_ctx", 4096),
            "tools": ([core.WEB_SEARCH_TOOL] if (web_on and tools_supported is not False) else None),
            "min_pages": int(chat.get("crawl_pages") or MIN_CRAWLED_PAGES),
            "allowed_domains": allowed_domains,
            "max_output_tokens": store.config.get("max_output_tokens", 16000),
            "multi_pass": bool(chat.get("multi_pass")),
            "passes": max(0, int(chat.get("passes") or 0)) if chat.get("multi_pass") else 0,
            "pass_use_system": bool(chat.get("pass_use_system", True)),
        }

        def batch_gen():
            stop_event = runs.new(run_id)
            adapter = adapter_for(server_url)
            tool_executor = logic.make_tool_executor(
                stop_event, min_pages=snap["min_pages"], allowed_domains=snap["allowed_domains"])
            responses_dir = folder_path / "responses"
            history = []
            done = 0
            total = len(files)
            # Context-usage monitor: resolve the bar denominator + server identity once
            # for the whole (sequential) batch; each file emits its own start/finish.
            b_server = store.resolve_server(server_url)
            b_sid = b_server.get("base_url") or server_url
            b_sname = b_server.get("name") or b_sid
            b_chat_id = chat.get("id") or ""
            b_private = bool(chat.get("private"))
            b_window = context_tracker.resolve_window({"num_ctx": snap["num_ctx"]}, b_server, store.config)
            try:
                yield sse("start", {"run_id": run_id, "total": total,
                                    "responses_dir": str(responses_dir)})
                for idx, path in enumerate(files, 1):
                    if stop_event.is_set():
                        break
                    try:
                        prompt = path.read_text(encoding="utf-8", errors="replace")
                    except Exception as e:
                        yield sse("file", {"index": idx, "total": total, "name": path.name,
                                           "prompt": "(could not read file)",
                                           "response": f"[Error reading {path.name}: {e}]"})
                        continue
                    if not prompt.strip():
                        yield sse("file", {"index": idx, "total": total, "name": path.name,
                                           "prompt": "(empty file — skipped)",
                                           "response": "[Skipped: the prompt file was empty.]"})
                        continue

                    yield sse("progress", {"index": idx, "total": total, "name": path.name})

                    opts = {"num_ctx": snap["num_ctx"], "max_output_tokens": snap["max_output_tokens"]}
                    # RAG for this file: a selected library is always served via RAG
                    # (manifest + retrieved excerpts); degrade to the full XML dump when
                    # embeddings are unavailable or nothing is retrieved.
                    item_chat = dict(chat)
                    item_chat["messages"] = (
                        ([] if snap["isolated"] else list(history))
                        + [{"role": "user", "content": prompt}])
                    rag_plan, rag_retrieved, _rag_status = _rag_retrieve(item_chat)
                    use_rag = bool(rag_plan and rag_plan.get("active") and rag_retrieved)
                    blocked = bool(rag_plan and rag_plan.get("blocked"))
                    if use_rag:
                        messages = logic.build_batch_messages(
                            prompt, history, snap, lib_block=snap["lib_manifest"])
                        messages = logic.inject_rag(messages, rag_retrieved, store.libraries)
                    elif blocked:
                        # Uncompiled library: inject no library context (no full dump).
                        messages = logic.build_batch_messages(prompt, history, snap, lib_block="")
                    else:
                        messages = logic.build_batch_messages(prompt, history, snap)
                    # build_batch_messages knows nothing about attachments, so without
                    # this a batch run saw the chat's pinned material only when the
                    # parallel branch (which goes through generate_one) happened to be
                    # active — the same run, two different contexts.
                    messages = logic.inject_attachments(
                        messages, logic.resolve_attachments(chat, use_rag))

                    breakdown = context_tracker.breakdown_from_messages(messages)
                    yield sse("context", {
                        "phase": "start", "chat_id": b_chat_id,
                        "server": b_sid, "server_name": b_sname, "window": b_window,
                        "prompt_tokens": context_tracker.estimate_messages(messages),
                        "breakdown": breakdown, "isolation": snap["isolated"],
                        "batch_item_label": path.name, "exact": False,
                    })

                    raw = ""
                    file_usage = None
                    try:
                        for kind, text in adapter.chat_stream(
                            snap["model"], messages, opts, stop_event,
                            think=snap["think"], tools=snap["tools"], tool_executor=tool_executor,
                        ):
                            if kind == "content":   # batch writes only the answer, not reasoning
                                raw += text
                            elif kind == "usage":
                                file_usage = text
                        # Multi-Pass refinement: feed the last answer back N times.
                        for _p in range(snap["passes"]):
                            if stop_event.is_set():
                                break
                            if use_rag:
                                eval_msgs = logic.build_eval_messages(
                                    chat, prompt, raw, store.libraries,
                                    snap["pass_use_system"], skip_library_dump=True)
                                eval_msgs = logic.inject_rag(eval_msgs, rag_retrieved, store.libraries)
                            else:
                                # When blocked (uncompiled library), drop the library so no
                                # full-text dump leaks into the refinement pass.
                                eval_chat = dict(chat, library_ids=[]) if blocked else chat
                                eval_msgs = logic.build_eval_messages(
                                    eval_chat, prompt, raw, store.libraries, snap["pass_use_system"])
                            refined = ""
                            for kind, text in adapter.chat_stream(
                                snap["model"], eval_msgs, opts, stop_event,
                                think=snap["think"], tools=None, tool_executor=None,
                            ):
                                if kind == "content":
                                    refined += text
                                elif kind == "usage":
                                    file_usage = text
                            if refined.strip():
                                raw = refined
                    except Exception as e:
                        raw = f"[Generation error for {path.name}: {e}]"

                    if stop_event.is_set() and not raw:
                        break

                    clean = strip_markdown(raw) or raw
                    try:
                        (responses_dir / path.name).write_text(clean, encoding="utf-8")
                    except Exception as e:
                        clean = clean + f"\n\n[Warning: could not write response file: {e}]"

                    yield sse("file", {"index": idx, "total": total, "name": path.name,
                                       "prompt": prompt, "response": clean})

                    # Context-usage: exact counts (or estimate fallback) + history entry.
                    exact_prompt = ((file_usage or {}).get("prompt_tokens")
                                    or context_tracker.estimate_messages(messages))
                    exact_completion = ((file_usage or {}).get("completion_tokens")
                                        or context_tracker.estimate_tokens(raw))
                    yield sse("context", {
                        "phase": "finish", "chat_id": b_chat_id,
                        "server": b_sid, "server_name": b_sname, "window": b_window,
                        "prompt_tokens": exact_prompt, "completion_tokens": exact_completion,
                        "breakdown": breakdown, "isolation": snap["isolated"],
                        "batch_item_label": path.name, "exact": bool(file_usage),
                    })
                    context_tracker.record_call(store, b_chat_id, context_tracker.make_entry(
                        server_id=b_sid, server_name=b_sname, isolation=snap["isolated"],
                        breakdown=breakdown, prompt_tokens=exact_prompt,
                        completion_tokens=exact_completion, num_ctx_at_time=b_window,
                        batch_item_label=path.name), private=b_private)

                    if not snap["isolated"]:
                        history.append({"role": "user", "content": prompt})
                        history.append({"role": "assistant", "content": clean})
                    done += 1

                yield sse("done", {"count": done, "stopped": stop_event.is_set(),
                                   "responses_dir": str(responses_dir)})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(batch_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------------- Batch tab --------------------------------
    # Distinct from /api/batch/start above, which is the chat composer's older
    # "every file IS a prompt" button. Here an input item is CONTENT and one prompt
    # template runs against each item. Source resolution, naming and export live in
    # app/batch.py; generation reuses generate_one through the parallel engine.
    @app.route("/api/batch/projects", methods=["GET"])
    def api_batch_projects_get():
        return jsonify({"projects": store.batch_project_summaries()})

    @app.route("/api/batch/projects", methods=["POST"])
    def api_batch_projects_post():
        proj = request.get_json(force=True) or {}
        if not proj.get("id"):
            proj["id"] = uuid.uuid4().hex[:12]
        proj["updated"] = datetime.now().isoformat(timespec="seconds")
        proj.setdefault("created", proj["updated"])
        store.upsert_batch_project(proj)
        return jsonify({"project": proj, "projects": store.batch_project_summaries()})

    @app.route("/api/batch/projects/<project_id>", methods=["GET"])
    def api_batch_project_get(project_id):
        proj = store.get_batch_project(project_id)
        if proj is None:
            return jsonify({"error": "not found"}), 404
        return jsonify({"project": proj})

    @app.route("/api/batch/projects/<project_id>", methods=["DELETE"])
    def api_batch_project_delete(project_id):
        store.delete_batch_project(project_id)
        return jsonify({"ok": True, "projects": store.batch_project_summaries()})

    def _batch_lanes(project):
        """Lanes for a batch run: the saved multi-server lanes when parallel processing
        is on, otherwise ONE synthetic lane pinned to the project's own server/model.
        A single-lane run_parallel is the sequential case, so both paths share one
        generation loop."""
        if store.config.get("parallel_enabled") and _parallel_lanes():
            return _parallel_lanes(), store.config.get("parallel_mode", "balanced")
        server_url = project.get("server_url") or DEFAULT_LOCAL_URL
        srv = store.resolve_server(server_url)
        return ([{"base_url": srv["base_url"], "model": project.get("model") or "",
                  "name": srv.get("name") or srv["base_url"]}], "balanced")

    def _batch_item_chat(project, item, run_id):
        """One item's chat dict. The prompt template is already rendered into the user
        turn, so pre_on is forced off — leaving it on would inject the template twice.

        The project's reference images come first and this item's own image (if the
        source was a folder of pictures) last, so the reference reads as background
        and the item as the thing being asked about."""
        turn = {"role": "user",
                "content": batch_mod.render_prompt(project.get("pre_prompt"), item)}
        refs = [{"id": r["id"]} for r in (project.get("reference_images") or [])
                if isinstance(r, dict) and r.get("id")]
        refs += [{"id": i} for i in (item.get("image_ids") or [])]
        if refs:
            turn["images"] = refs
        return {
            "id": f"batch-{run_id}",
            "private": True,
            "isolated": True,
            "messages": [turn],
            "image_full_res": bool(project.get("image_full_res")),
            "server_url": project.get("server_url") or DEFAULT_LOCAL_URL,
            "model": project.get("model") or "",
            "num_ctx": project.get("num_ctx") or store.config.get("default_num_ctx", 4096),
            "system_prompt": project.get("system_prompt") or "",
            "system_on": bool(project.get("system_on")),
            "pre_prompt": "",
            "pre_on": False,
            "library_ids": list(project.get("library_ids") or []),
            "library_strict": bool(project.get("library_strict")),
            "multi_pass": bool(project.get("multi_pass")),
            "passes": int(project.get("passes") or 0),
            "pass_use_system": bool(project.get("pass_use_system", True)),
            "eval_prompt": project.get("eval_prompt") or "",
            "web_search": False,
            "hide_thinking": True,
        }

    # Naming only needs enough of the document to describe it; sending a whole
    # transcript to pick six words wastes a full context window per item.
    _BATCH_TITLE_CHARS = 4000

    def _batch_llm_title(project, item, response, stop_event):
        """Ask the model for a filename based on what it just wrote. Falls back to the
        item's own title on any failure — a naming hiccup must not lose the response."""
        server_url = project.get("server_url") or DEFAULT_LOCAL_URL
        prompt = batch_mod.render_prompt(
            project.get("name_prompt") or batch_mod.DEFAULT_FILENAME_PROMPT,
            {**item, "content": (response or item.get("content") or "")[:_BATCH_TITLE_CHARS]})
        try:
            out = ""
            for kind, text in adapter_for(server_url).chat_stream(
                project.get("model"), [{"role": "user", "content": prompt}],
                {"num_ctx": project.get("num_ctx") or store.config.get("default_num_ctx", 4096),
                 "max_output_tokens": 200},
                stop_event, think=False, tools=None, tool_executor=None,
            ):
                if kind == "content":
                    out += text
            # Models like to explain themselves; take the first non-empty line only.
            for line in (out or "").splitlines():
                if line.strip():
                    return line.strip()
        except Exception:
            pass
        return item.get("title") or ""

    @app.route("/api/batch/reference-images", methods=["POST"])
    def api_batch_reference_images():
        """Native picker for a project's reference images (sent with every item).
        Synchronous rather than SSE: this is a handful of files, not a folder walk."""
        paths = native_dialog.pick_files(
            title="Choose reference image(s) for every item", filetypes_key="images")
        records, errors = [], []
        for p in paths:
            path = Path(p)
            if not images_mod.is_supported(path):
                errors.append(f"{path.name}: not an image type we can read")
                continue
            try:
                records.append(images_mod.store_file(path))
            except Exception as e:
                errors.append(f"{path.name}: {e}")
        return jsonify({"images": records, "errors": errors})

    @app.route("/api/batch/preview", methods=["POST"])
    def api_batch_preview():
        """Resolve a project's sources WITHOUT generating, so the user can see exactly
        what would run. Streams the same resolve progress frames as a real run."""
        data = request.get_json(force=True) or {}
        project = data.get("project") or {}
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]

        def work(emit, stop_event):
            items, errors = batch_mod.resolve_sources(
                project, emit=emit, should_stop=lambda: stop_event and stop_event.is_set())
            emit("items", {
                "total": len(items),
                "errors": errors,
                # Titles and sizes only — the browser never needs the content.
                "items": [{"item_id": i["item_id"], "title": i["title"], "kind": i["kind"],
                           "chars": i["chars"], "source_path": i["source_path"],
                           "source_url": i["source_url"],
                           "image_ids": list(i.get("image_ids") or [])}
                          for i in items],
            })

        return _compile_sse(work, run_id)

    @app.route("/api/batch/run", methods=["POST"])
    def api_batch_run():
        """Body: {project, run_id}. Resolves sources, runs each item through
        generate_one via the parallel engine, and exports per the project's settings.
        Frames: resolve progress / plan / item_start / chunk / item_done / export."""
        data = request.get_json(force=True) or {}
        project = data.get("project") or {}
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]

        problems = batch_mod.validate_project(project)
        if problems:
            return jsonify({"error": " ".join(problems)}), 400

        def work(emit, stop_event):
            stopped = lambda: bool(stop_event and stop_event.is_set())

            items, errors = batch_mod.resolve_sources(
                project, emit=emit, should_stop=stopped)
            if not items:
                raise batch_mod.BatchError(
                    "No items could be read from the chosen sources."
                    + (" " + errors[0] if errors else ""))

            by_id = {i["item_id"]: i for i in items}
            lanes, mode = _batch_lanes(project)
            emit("plan", {
                "total": len(items), "mode": mode, "resolve_errors": errors,
                "lanes": [{"index": n, "name": l["name"], "server": l["base_url"],
                           "model": l["model"]} for n, l in enumerate(lanes)],
                "items": [{"item_id": i["item_id"], "title": i["title"],
                           "chars": i["chars"]} for i in items],
            })

            write_files = (project.get("output_mode") or "both") in ("files", "both")
            per_item = (project.get("export_mode") or "per_item") != "combined"
            results = []
            written = []

            def on_frame(frame):
                """run_parallel calls this from its worker threads. item_done is where
                naming + the per-item file write happen, so the file is on disk before
                the browser is told the item finished."""
                if frame.get("event") != "item_done":
                    emit(frame.pop("event", "message"), frame)
                    return
                item = by_id.get(frame.get("item_id")) or {}
                prompt = batch_mod.render_prompt(project.get("pre_prompt"), item)
                response = frame.get("content") or ""
                results.append({"item": item, "prompt": prompt, "response": response})

                if write_files and per_item and not stopped():
                    title = ""
                    if project.get("name_mode") == "llm":
                        title = _batch_llm_title(project, item, response, stop_event)
                    try:
                        path = batch_mod.write_item(item, prompt, response, project, title)
                        written.append(str(path))
                        frame["out_path"] = str(path)
                    except Exception as e:
                        frame["export_error"] = str(e)
                    # Pictures the model produced land beside the text, under the
                    # same prefix/suffix/uniqueness rules.
                    paths = batch_mod.write_item_images(
                        item, frame.get("images") or [], project, title)
                    if paths:
                        written.extend(str(p) for p in paths)
                        frame["out_image_paths"] = [str(p) for p in paths]
                frame["title"] = item.get("title") or frame.get("title") or ""
                emit(frame.pop("event", "message"), frame)

            parallel.run_parallel(
                [{"item_id": i["item_id"], "title": i["title"], "search_query": "",
                  "chat": _batch_item_chat(project, i, run_id)} for i in items],
                lanes, mode, stop_event, generate_one, on_frame)

            # on_frame appends from the lane worker threads, so `results` is in
            # completion order. Restore the source order before exporting.
            order = {i["item_id"]: n for n, i in enumerate(items)}
            results.sort(key=lambda r: order.get((r["item"] or {}).get("item_id"), 0))

            combined_path = ""
            if write_files and not per_item and results:
                combined_path = str(batch_mod.write_combined(results, project))
                written.append(combined_path)

            emit("export", {"files": written, "combined": combined_path,
                            "count": len(results), "stopped": stopped(),
                            "resolve_errors": errors})

        return _compile_sse(work, run_id)

    # ---------------- Prompt Validation & Evaluation ----------------
    @app.route("/api/evals", methods=["GET"])
    def api_evals_get():
        return jsonify({"evals": store.eval_summaries()})

    @app.route("/api/evals", methods=["POST"])
    def api_eval_upsert():
        """Create or update a full eval project. Body: {eval: <project dict>}."""
        data = request.get_json(force=True) or {}
        project = data.get("eval") or {}
        if not project.get("id"):
            # Drop a blank id so the freshly generated one isn't overwritten.
            project.pop("id", None)
            project = {**evals.create_eval_dict(), **project}
        project["updated"] = datetime.now(timezone.utc).isoformat()
        store.upsert_eval(project)
        return jsonify({"eval": project, "evals": store.eval_summaries()})

    @app.route("/api/evals/<eval_id>", methods=["GET"])
    def api_eval_get_one(eval_id):
        project = store.get_eval(eval_id)
        if not project:
            return jsonify({"error": "not found"}), 404
        return jsonify({"eval": project})

    @app.route("/api/evals/<eval_id>", methods=["DELETE"])
    def api_eval_delete(eval_id):
        store.delete_eval(eval_id)
        return jsonify({"ok": True, "evals": store.eval_summaries()})

    @app.route("/api/evals/import", methods=["POST"])
    def api_eval_import():
        """Open a native file picker and parse the chosen file. Body:
        {kind:"csv"|"txt", delimiter, is_regex}. CSV -> {columns, rows};
        TXT -> {cells} (the client picks which column to populate)."""
        data = request.get_json(force=True) or {}
        kind = (data.get("kind") or "csv").lower()
        title = "Choose a CSV file" if kind == "csv" else "Choose a text file"
        paths = native_dialog.pick_files(title=title)
        if not paths:
            return jsonify({"ok": False, "cancelled": True})
        path = Path(paths[0])
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if kind == "csv":
            parsed = evals.parse_csv(raw)
            return jsonify({"ok": True, "name": path.name, "columns": parsed["columns"],
                            "rows": parsed["rows"]})
        cells = evals.split_text(raw, data.get("delimiter", ""), bool(data.get("is_regex")))
        return jsonify({"ok": True, "name": path.name, "cells": cells})

    @app.route("/api/evals/run", methods=["POST"])
    def api_eval_run():
        """Run a prompt-evaluation. Body: {eval: <project>, run_id, batch:bool}.
        Streams gen_progress + row_result frames per model, model_done aggregates,
        and a final summary. Generation fans rows across the parallel lanes when
        Parallel Processing is enabled; grading uses the project's grader model so
        scores stay comparable across models under test."""
        data = request.get_json(force=True) or {}
        project = data.get("eval") or {}
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]
        batch = bool(data.get("batch"))

        rows = project.get("rows") or []
        criteria = [c for c in (project.get("criteria") or []) if (c.get("label") or "").strip()]
        if not rows:
            return jsonify({"error": "The spreadsheet has no rows to evaluate."}), 400
        if not (project.get("prompt_template") or "").strip():
            return jsonify({"error": "Enter a prompt to evaluate."}), 400
        if not criteria:
            return jsonify({"error": "Add at least one grading criterion."}), 400

        grader_url = project.get("grader_server_url") or project.get("gen_server_url") or DEFAULT_LOCAL_URL
        grader_model = project.get("grader_model") or project.get("gen_model")
        if not grader_model:
            return jsonify({"error": "No grader model selected."}), 400

        if batch:
            gen_models = [m for m in (project.get("batch_models") or []) if m.get("model")]
            if not gen_models:
                return jsonify({"error": "Add at least one model to the batch list."}), 400
        else:
            if not project.get("gen_model"):
                return jsonify({"error": "No model selected to run the prompt."}), 400
            gen_models = [{"server_url": project.get("gen_server_url"), "model": project.get("gen_model")}]

        input_cols = project.get("input_columns") or project.get("columns") or []
        template = project.get("prompt_template") or ""
        num_ctx = project.get("num_ctx") or store.config.get("default_num_ctx", 4096)
        opts = {"num_ctx": num_ctx, "max_output_tokens": store.config.get("max_output_tokens", 16000)}
        filled = [evals.fill_prompt(template, r, input_cols) for r in rows]

        # One model-list lookup per server for the whole run — lane selection asks the
        # same question once per batch model, never once per row.
        _models_on = {}

        def _hosts_model(url, model):
            """Does this server actually have `model` installed? Unknown/unreachable
            servers answer False so work is never sent somewhere it can only fail."""
            if url not in _models_on:
                try:
                    _models_on[url] = set(adapter_for(url).list_models() or [])
                except Exception:
                    _models_on[url] = set()
            return model in _models_on[url]

        def _lanes_for(g_url, g_model):
            """The lanes that may run `g_model`, with the model under test forced onto
            each. Only servers that actually host the model qualify — the model's own
            server always leads the list — so a batch entry's server choice is honoured
            instead of being replaced by whatever the global parallel list happens to
            hold. Returns [] when parallel processing shouldn't be used."""
            if not store.config.get("parallel_enabled"):
                return []
            picked, seen = [], set()
            for ln in ([{"base_url": g_url, "name": g_url}] + _parallel_lanes()):
                url = ln.get("base_url")
                if not url or url in seen or not _hosts_model(url, g_model):
                    continue
                seen.add(url)
                picked.append({**ln, "model": g_model})
            # A single lane is just the sequential loop with extra machinery.
            return picked if len(picked) > 1 else []

        def _generate_all(g_url, g_model, stop_event):
            """Stream gen_progress frames, then yield ("__responses__", responses,
            errors, servers) — the per-row response strings, the per-row generation
            error messages (empty string when the row succeeded), and the servers that
            actually did the work. Fans across the qualifying parallel lanes when there
            is more than one, else runs a sequential loop against g_url."""
            responses = [""] * len(rows)
            errors = [""] * len(rows)

            def base_chat(i):
                return {"server_url": g_url, "model": g_model, "num_ctx": num_ctx,
                        "isolated": True, "hide_thinking": True,
                        "messages": [{"role": "user", "content": filled[i]}]}

            lanes = _lanes_for(g_url, g_model)
            if lanes:
                servers = [ln.get("base_url") for ln in lanes]
                yield sse("status", {"message": f"⚡ {g_model} across "
                                                f"{len(lanes)} server(s): {', '.join(servers)}"})
                items = [{"item_id": i, "title": f"Row {i + 1}", "search_query": "",
                          "chat": base_chat(i)} for i in range(len(rows))]
                frames = queue.Queue()
                sentinel = object()

                def work():
                    try:
                        parallel.run_parallel(items, lanes,
                                              store.config.get("parallel_mode", "balanced"),
                                              stop_event, generate_one, frames.put)
                    finally:
                        frames.put(sentinel)

                threading.Thread(target=work, daemon=True).start()
                done = 0
                while True:
                    fr = frames.get()
                    if fr is sentinel:
                        break
                    idx = fr.get("item_id")
                    in_range = isinstance(idx, int) and 0 <= idx < len(responses)
                    if fr.get("event") == "error" and in_range:
                        errors[idx] = fr.get("message", "") or "generation failed"
                    elif fr.get("event") == "item_done":
                        if in_range:
                            responses[idx] = fr.get("content", "") or ""
                        done += 1
                        yield sse("gen_progress", {"done": done, "total": len(rows)})
            else:
                servers = [g_url]
                for i in range(len(rows)):
                    if stop_event.is_set():
                        break
                    content = ""
                    for kind, d in generate_one(base_chat(i), "", stop_event):
                        if kind == "chunk":
                            content += (d or {}).get("content", "")
                        elif kind == "pass_end":
                            content = (d or {}).get("content", content)
                        elif kind == "error":
                            # Kept out of `content`: a failure is an ungraded row, not a
                            # response for the grader to score.
                            errors[i] = (d or {}).get("message", "") or "generation failed"
                    responses[i] = content
                    yield sse("gen_progress", {"done": i + 1, "total": len(rows)})
            yield ("__responses__", responses, errors, servers)

        def eval_gen():
            stop_event = runs.new(run_id)
            try:
                yield sse("start", {"run_id": run_id, "batch": batch, "total_rows": len(rows),
                                    "models": [{"server": m.get("server_url"), "model": m.get("model")}
                                               for m in gen_models],
                                    "criteria": [{"label": c.get("label"), "mode": c.get("mode", "score"),
                                                  "min": c.get("min", 1), "max": c.get("max", 10)}
                                                 for c in criteria]})
                grader_adapter = adapter_for(grader_url)
                grader_think = model_is_reasoning(grader_url, grader_model)
                per_model = []

                for mi, gm in enumerate(gen_models):
                    if stop_event.is_set():
                        break
                    g_url = gm.get("server_url") or project.get("gen_server_url") or DEFAULT_LOCAL_URL
                    g_model = gm.get("model") or project.get("gen_model")
                    yield sse("model_start", {"index": mi, "total": len(gen_models),
                                              "server": g_url, "model": g_model})

                    # ---- generation stage ----
                    responses = [""] * len(rows)
                    errors = [""] * len(rows)
                    servers = [g_url]
                    for frame in _generate_all(g_url, g_model, stop_event):
                        if isinstance(frame, tuple) and frame[0] == "__responses__":
                            _, responses, errors, servers = frame
                        else:
                            yield frame

                    # ---- grading stage ----
                    row_grades = []
                    for i in range(len(rows)):
                        if stop_event.is_set():
                            break
                        resp = responses[i]
                        if not (resp or "").strip():
                            # Nothing to grade — an empty or failed generation must not
                            # be handed to the grader, or its complaint about the error
                            # text would land in the average as a real score.
                            grades = evals.normalize_grades({}, criteria)
                            row_grades.append(grades)
                            yield sse("row_result", {"model_index": mi, "index": i,
                                                     "response": resp, "grades": grades,
                                                     "ungraded": True, "error": errors[i]})
                            continue
                        gmsgs = evals.build_grader_messages(project, filled[i], resp)
                        gtext = ""
                        try:
                            for kind, text in grader_adapter.chat_stream(
                                grader_model, gmsgs, opts, stop_event, think=grader_think):
                                if kind == "content":
                                    gtext += text
                        except Exception as e:
                            gtext = ""
                            yield sse("status", {"message": f"Grader error on row {i + 1}: {e}"})
                        grades = evals.normalize_grades(evals.parse_grader_json(gtext), criteria)
                        row_grades.append(grades)
                        yield sse("row_result", {"model_index": mi, "index": i,
                                                 "response": resp, "grades": grades})

                    agg = evals.aggregate(row_grades, criteria)
                    # `servers` is where the work really ran, which is not always g_url:
                    # a parallel run fans across every lane that hosts this model.
                    label = ", ".join(servers) if servers else g_url
                    per_model.append({"server": label, "servers": servers,
                                      "model": g_model, "aggregate": agg})
                    yield sse("model_done", {"index": mi, "server": label, "servers": servers,
                                             "model": g_model, "aggregate": agg})

                yield sse("summary", {"batch": batch, "models": per_model,
                                      "criteria": [c.get("label") for c in criteria]})
                yield sse("done", {"stopped": stop_event.is_set()})
            except Exception as e:
                yield sse("error", {"message": str(e)})
                # The client re-enables its buttons off `done`; without this an error
                # would leave the tab wedged until a page reload.
                yield sse("done", {"stopped": True, "error": True})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(eval_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/evals/gen-data", methods=["POST"])
    def api_eval_gen_data():
        """Synthesize dummy test rows with the LLM. Body:
        {eval: <project>, run_id, num_rows}. For each requested row the generation
        model is asked to invent one JSON record keyed by the target columns (every
        column except the output column, restricted to those with an instruction).
        Rows fan across the parallel lanes when Parallel Processing is enabled, else
        run sequentially. Streams a row_result per finished row, gen_progress, done."""
        data = request.get_json(force=True) or {}
        project = data.get("eval") or {}
        run_id = data.get("run_id") or uuid.uuid4().hex[:12]

        columns = project.get("columns") or []
        output_column = project.get("output_column") or ""
        gen_instructions = project.get("gen_instructions") or {}
        targets = evals.gen_target_columns(columns, gen_instructions, output_column)
        try:
            num_rows = int(data.get("num_rows") or project.get("gen_num_rows") or 0)
        except (TypeError, ValueError):
            num_rows = 0

        if not targets:
            return jsonify({"error": "Add a generation instruction to at least one "
                                     "non-output column first."}), 400
        if num_rows < 1:
            return jsonify({"error": "Enter how many rows to generate."}), 400

        g_url = project.get("gen_server_url") or DEFAULT_LOCAL_URL
        g_model = project.get("gen_model")
        if not g_model:
            return jsonify({"error": "No model selected to generate data."}), 400

        num_ctx = project.get("num_ctx") or store.config.get("default_num_ctx", 4096)

        def base_chat(i):
            msgs = evals.build_gen_prompt(columns, gen_instructions, output_column,
                                          row_index=i, seed=run_id)
            return {"server_url": g_url, "model": g_model, "num_ctx": num_ctx,
                    "isolated": True, "hide_thinking": True, "messages": msgs}

        def _row_has_content(row):
            """False when a row came back with every cell empty — the generation failed
            or its JSON didn't parse, and appending it would just add a blank line."""
            return any((v or "").strip() for v in (row or {}).values())

        def _gen_lanes(url, model):
            """Lanes that may generate rows: only servers that actually host the model,
            the chosen server first. [] means run sequentially."""
            if not store.config.get("parallel_enabled"):
                return []
            picked, seen = [], set()
            for ln in ([{"base_url": url, "name": url}] + _parallel_lanes()):
                base = ln.get("base_url")
                if not base or base in seen:
                    continue
                try:
                    installed = set(adapter_for(base).list_models() or [])
                except Exception:
                    installed = set()
                if model not in installed:
                    continue
                seen.add(base)
                picked.append({**ln, "base_url": base, "model": model})
            return picked if len(picked) > 1 else []

        def gen_gen():
            stop_event = runs.new(run_id)
            try:
                yield sse("start", {"run_id": run_id, "total": num_rows,
                                    "columns": targets, "server": g_url, "model": g_model})

                lanes = _gen_lanes(g_url, g_model)
                if lanes:
                    yield sse("status", {"message": f"⚡ {g_model} across {len(lanes)} "
                                         f"server(s): {', '.join(ln['base_url'] for ln in lanes)}"})
                    items = [{"item_id": i, "title": f"Row {i + 1}", "search_query": "",
                              "chat": base_chat(i)} for i in range(num_rows)]
                    frames = queue.Queue()
                    sentinel = object()

                    def work():
                        try:
                            parallel.run_parallel(items, lanes,
                                                  store.config.get("parallel_mode", "balanced"),
                                                  stop_event, generate_one, frames.put)
                        finally:
                            frames.put(sentinel)

                    threading.Thread(target=work, daemon=True).start()
                    done = 0
                    while True:
                        fr = frames.get()
                        if fr is sentinel:
                            break
                        if fr.get("event") == "item_done":
                            idx = fr.get("item_id")
                            row = evals.parse_gen_row(fr.get("content", "") or "", targets)
                            done += 1
                            yield sse("row_result", {"index": idx, "row": row,
                                                     "ok": _row_has_content(row)})
                            yield sse("gen_progress", {"done": done, "total": num_rows})
                else:
                    for i in range(num_rows):
                        if stop_event.is_set():
                            break
                        content = ""
                        for kind, d in generate_one(base_chat(i), "", stop_event):
                            if kind == "chunk":
                                content += (d or {}).get("content", "")
                            elif kind == "pass_end":
                                content = (d or {}).get("content", content)
                            elif kind == "error":
                                yield sse("status", {"message": f"Row {i + 1}: "
                                                     f"{(d or {}).get('message', '')}"})
                        row = evals.parse_gen_row(content, targets)
                        yield sse("row_result", {"index": i, "row": row,
                                                 "ok": _row_has_content(row)})
                        yield sse("gen_progress", {"done": i + 1, "total": num_rows})

                yield sse("done", {"stopped": stop_event.is_set()})
            except Exception as e:
                yield sse("error", {"message": str(e)})
                yield sse("done", {"stopped": True, "error": True})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(gen_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------- Database Processing tab ----------------------
    # Register the /api/db/* routes, handing over the shared collaborators the
    # DB feature reuses (LLM engine, parallel SSE, run registry, sse helper).
    db_ctx = SimpleNamespace(
        store=store, vault=vault, runs=runs, sse=sse,
        generate_one=generate_one, adapter_for=adapter_for,
        parallel_sse=_parallel_sse, parallel_lanes=_parallel_lanes,
    )
    register_db_routes(app, db_ctx)

    return app
