#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Flask web server. Serves the single-page browser UI (static/) and a JSON API for
every feature of the original desktop app: chats, streaming generation + stop,
presets, servers/models, libraries (Resources), web search, and folder batch
processing. Filesystem operations use native OS dialogs (app.native_dialog) and
read/write paths directly on the machine — nothing is uploaded.
"""

import ipaddress
import json
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import requests
from flask import (Flask, request, jsonify, Response, send_from_directory,
                   stream_with_context, session, redirect)

from . import (compile as compile_mod, context_tracker, core, crypto, evals, ingest,
               logic, migrate, native_dialog, parallel, persona as persona_mod,
               persona_io, persona_store, pipeline as pipeline_mod, profiles, providers,
               rag, rewrite)
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


def create_app():
    app = Flask(__name__, static_folder=None)
    # Sign session cookies with the persisted secret (creates the encryption keyfile
    # with default admin/admin on first run). The app data is encrypted at rest and is
    # only loaded after the user logs in — see the auth block below.
    app.secret_key = crypto.get_flask_secret(core.APP_KEYFILE)
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
    pipeline_runs = {}

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
            # Carry over the RAG store, DB staging/audit and the credential vault that
            # were written into the scratch during the session.
            for name in ("rag.duckdb", "rag.duckdb.wal", "db", "db_vault.enc"):
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
                   "provider_context_windows")
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
                      "rag_embed_parallel", "rag_ann_enabled"):
            if boolk in patch:
                patch[boolk] = bool(patch[boolk])
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
        return jsonify({"ok": True, "path": dest, "count": len(envelope.get("chats", []))})

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
                    return ({"active": True, "blocked": True}, None,
                            f"⚠ Library {names} needs compiling before RAG can use it — "
                            f"open Resources → Compile Data.")
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
                    rag_plan["top_k"], mode=mode, queries=queries)
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
                        messages = logic.inject_rag(messages, rag_retrieved, store.libraries)
                    else:
                        messages = logic.build_messages(chat, store.libraries)
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
                    pass_tools, tool_executor = None, None

                # Context-usage: pre-call estimate (breakdown + full prompt) so the
                # bar can move while the request is in flight. Isolation is inherently
                # correct here — excluded history was never in `messages`.
                breakdown = context_tracker.breakdown_from_messages(messages)
                prompt_estimate = context_tracker.estimate_messages(messages)
                yield ("context", {
                    "phase": "start", "chat_id": chat_id,
                    "server": server_id, "server_name": server_name,
                    "window": window, "prompt_tokens": prompt_estimate,
                    "breakdown": breakdown, "isolation": is_isolated,
                    "batch_item_label": batch_item_label, "exact": False,
                    "pass_index": p, "pass_total": N + 1,
                })

                content, reasoning = "", ""
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
                                    "reasoning": reasoning if show_reasoning else ""})
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

    def _persona_engine(persona, chat, run_id):
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
        stop_event = runs.new(run_id)

        def llm_complete(model, messages, schema):
            return rewrite.run_completion(adapter, model or chat_model, messages,
                                          num_ctx=chat.get("num_ctx") or 4096,
                                          max_tokens=store.config.get("max_output_tokens", 4096),
                                          fmt=schema)

        def llm_stream(model, messages):
            options = {"num_ctx": chat.get("num_ctx") or 4096,
                       "max_output_tokens": store.config.get("max_output_tokens", 16000)}
            for kind, text in adapter.chat_stream(model or chat_model, messages, options,
                                                  stop_event, think=False):
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
            max_retries=int(store.config.get("pipeline_max_retries", 3)))
        eng._stop_event = stop_event
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
        run_id = body.get("run_id") or f"compile-persona-{pid}"

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
        eng = _persona_engine(persona, chat, run_id)
        pipeline_runs[run_id] = eng
        gen = _pipeline_sse(run_id, lambda e: e.start(user_message, history))
        return Response(stream_with_context(gen), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/runs/<run_id>/rerun", methods=["POST"])
    def api_run_rerun(run_id):
        """Edit a completed/paused run's step output and re-run from that step forward."""
        data = request.get_json(force=True) or {}
        if run_id not in pipeline_runs:
            return jsonify({"error": "run not found (start a new message)"}), 404
        index = int(data.get("index", 0))
        edited = data.get("output", None)
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

    @app.route("/api/personas/<pid>/memories", methods=["GET"])
    def api_persona_memories(pid):
        return jsonify({"memories": memsvc.list_memories(pid)})

    @app.route("/api/personas/<pid>/memories", methods=["POST"])
    def api_persona_memory_create(pid):
        data = request.get_json(force=True) or {}
        mem = data.get("memory") or data
        try:
            persona = psvc.load(pid)
        except persona_mod.PersonaError as e:
            return jsonify({"error": str(e)}), 404
        embed_fn, embed_model = _persona_embed(persona)
        saved = memsvc.save_memory(pid, mem, embed_fn, embed_model)
        return jsonify({"memory": saved, "memories": memsvc.list_memories(pid)})

    @app.route("/api/personas/<pid>/memories/<mem_id>", methods=["DELETE"])
    def api_persona_memory_delete(pid, mem_id):
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
        embed_url = store.config.get("rag_embed_server_url") or DEFAULT_LOCAL_URL
        embed_fn, embed_model = _persona_embed(persona)
        mode = persona.get("stores", {}).get("retrieval", "hybrid")
        wants_vectors = mode in ("vector", "hybrid")
        want_ctx = bool(store.config.get("rag_contextual_chunking"))
        ctx_model = ((store.config.get("rag_context_model")
                      or persona.get("models", {}).get("chat_model", "")) if want_ctx else "")
        contextualize = _contextualizer_for(ctx_model, embed_url)

        def work(emit, stop_event):
            emit("begin", {"total": len(paths), "name": persona.get("profile", {}).get("name", "")})
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

        return _compile_sse(work, f"addfiles-persona-{pid}")

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

    # ------------------- Multi-server parallel processing -------------------
    def _parallel_lanes():
        """Resolve the saved parallel_servers into lane dicts (with credentials)."""
        lanes = []
        for s in (store.config.get("parallel_servers") or []):
            srv = store.resolve_server(s.get("base_url"))
            lanes.append({"base_url": srv["base_url"], "model": (s.get("model") or "").strip(),
                          "name": srv.get("name") or srv["base_url"]})
        return lanes

    def _parallel_sse(items, run_id, on_item_done=None):
        """Shared SSE generator: fan `items` across the saved lanes and multiplex
        every frame into one stream. on_item_done(frame) runs server-side as each
        item finishes (e.g. to write a batch response to disk) before forwarding."""
        stop_event = runs.new(run_id)
        lanes = _parallel_lanes()
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

        def work(emit, stop_event):
            emit("begin", {"total": len(wanted), "name": lib.get("name", "")})
            added = []
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
                    item = _new_library_item(item_type="file", label=name,
                                             content=res.get("text") or "",
                                             filename=name)
                    lib.setdefault("items", []).append(item)
                    added.append(name)
            lib["updated"] = datetime.utcnow().isoformat()
            store.upsert_library(lib)
            emit("complete", {"library": lib, "added": added, "errors": errs})

        return _compile_sse(work, f"addfiles-library-{lib_id}")

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
        lib.setdefault("items", []).append(item)
        lib["updated"] = datetime.utcnow().isoformat()
        store.upsert_library(lib)
        return jsonify({"library": lib, "added": [page["title"]], "errors": [],
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

        def work(emit):
            pages, errors, attempted = [], [], 0
            for ev in core.crawl_search(query, sites=sites, max_results=max_results):
                if ev.get("type") == "progress":
                    emit("progress", ev)
                elif ev.get("type") == "result":
                    pages = ev.get("pages") or []
                    errors = ev.get("errors") or []
                    attempted = ev.get("attempted") or 0
            added = []
            for p in pages:
                item = _new_library_item(item_type="url", label=p["title"],
                                         content=p["text"], filename=p["url"])
                lib.setdefault("items", []).append(item)
                added.append(p["title"])
            if pages:
                lib["updated"] = datetime.utcnow().isoformat()
                store.upsert_library(lib)
            emit("complete", {"library": lib, "added": added,
                              "attempted": attempted, "errors": errors})
        return _compile_sse(work)

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

    # ----------------------------- Native dialogs ---------------------------
    @app.route("/api/pick-folder", methods=["POST"])
    def api_pick_folder():
        data = request.get_json(silent=True) or {}
        path = native_dialog.pick_folder(title=data.get("title", "Choose a folder"))
        return jsonify({"path": path})

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
        project["updated"] = datetime.utcnow().isoformat()
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

        def _generate_all(g_url, g_model, stop_event):
            """Return a list of response strings (one per row), streaming gen_progress
            frames. Uses the parallel lanes when enabled, else a sequential loop."""
            responses = [""] * len(rows)

            def base_chat(i):
                return {"server_url": g_url, "model": g_model, "num_ctx": num_ctx,
                        "isolated": True, "hide_thinking": True,
                        "messages": [{"role": "user", "content": filled[i]}]}

            if store.config.get("parallel_enabled") and _parallel_lanes():
                # Fan rows across every lane, forcing the model under test on each.
                lanes = [{**ln, "model": g_model} for ln in _parallel_lanes()]
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
                    if fr.get("event") == "item_done":
                        idx = fr.get("item_id")
                        if isinstance(idx, int) and 0 <= idx < len(responses):
                            responses[idx] = fr.get("content", "") or ""
                        done += 1
                        yield sse("gen_progress", {"done": done, "total": len(rows)})
            else:
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
                            content = content or f"[Generation error: {(d or {}).get('message', '')}]"
                    responses[i] = content
                    yield sse("gen_progress", {"done": i + 1, "total": len(rows)})
            yield ("__responses__", responses)

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
                    for frame in _generate_all(g_url, g_model, stop_event):
                        if isinstance(frame, tuple) and frame[0] == "__responses__":
                            responses = frame[1]
                        else:
                            yield frame

                    # ---- grading stage ----
                    row_grades = []
                    for i in range(len(rows)):
                        if stop_event.is_set():
                            break
                        resp = responses[i]
                        if not (resp or "").strip():
                            grades = evals.normalize_grades({}, criteria)
                            row_grades.append(grades)
                            yield sse("row_result", {"model_index": mi, "index": i,
                                                     "response": resp, "grades": grades,
                                                     "ungraded": True})
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
                    per_model.append({"server": g_url, "model": g_model, "aggregate": agg})
                    yield sse("model_done", {"index": mi, "server": g_url, "model": g_model,
                                             "aggregate": agg})

                yield sse("summary", {"batch": batch, "models": per_model,
                                      "criteria": [c.get("label") for c in criteria]})
                yield sse("done", {"stopped": stop_event.is_set()})
            except Exception as e:
                yield sse("error", {"message": str(e)})
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

        def gen_gen():
            stop_event = runs.new(run_id)
            try:
                yield sse("start", {"run_id": run_id, "total": num_rows,
                                    "columns": targets, "server": g_url, "model": g_model})

                if store.config.get("parallel_enabled") and _parallel_lanes():
                    lanes = [{**ln, "model": g_model} for ln in _parallel_lanes()]
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
                            yield sse("row_result", {"index": idx, "row": row})
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
                        yield sse("row_result", {"index": i, "row": row})
                        yield sse("gen_progress", {"done": i + 1, "total": num_rows})

                yield sse("done", {"stopped": stop_event.is_set()})
            except Exception as e:
                yield sse("error", {"message": str(e)})
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
