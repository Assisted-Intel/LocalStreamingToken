#!/usr/bin/env python3
"""
Flask routes for the Database Processing tab, registered onto the main app by
``register_db_routes`` (called from ``app.server.create_app``). Kept in its own
module so ``server.py`` stays focused; it receives the shared collaborators
(store, vault, run registry, the ``sse`` helper, ``generate_one`` and the parallel
SSE plumbing) via a context object rather than importing them, avoiding cycles.

All routes are under ``/api/db``. Vault + profiles + session listing are live from
Phase 1; import/staging/process/dry-run/conflict/write-back return a clear 501 until
their phase lands.
"""

from __future__ import annotations

from datetime import datetime

from flask import jsonify, request, Response, stream_with_context

from .. import core, evals
from .conflict import ConflictDetector
from .connections import ConnectionManager, ConnectionError_
from .dryrun import DryRunEngine
from .importer import StreamingImporter, DEFAULT_CHUNK
from .models import ColumnDef, ColumnType, ConnectionProfile, ImportSession, SelectionSpec
from .processing import StagingProcessor
from .staging import StagingManager
from .types_map import sanitize_ddl_type
from .vault import VaultError, VaultLocked
from .writeback import WriteBackEngine


def register_db_routes(app, ctx):
    """Attach every /api/db/* route to the main Flask app.

    Called once from app.server.create_app(). Taking the shared services as ``ctx``
    rather than importing app.server keeps the dependency one-way (server -> database)
    and avoids a circular import; it also means the DB tab reuses the app's single
    generation loop instead of growing its own.

    ctx attributes used: store, vault, runs, sse, generate_one, adapter_for,
    parallel_sse, parallel_lanes.

    These routes sit behind the app-wide login gate, and everything that touches a
    real database additionally requires the connection vault to be unlocked - which is
    a SEPARATE password. Locked-vault responses use HTTP 423 with {"locked": true} so
    the UI can prompt for it."""
    store = ctx.store
    vault = ctx.vault
    runs = ctx.runs
    sse = ctx.sse
    conns = ConnectionManager()
    importer = StreamingImporter(conns)

    def _todo(phase):
        """Placeholder response for an endpoint that isn't built yet."""
        return jsonify({"error": f"Not implemented yet (arrives in Phase {phase})."}), 501

    def _int_arg(name, default, *, lo=0, hi=None):
        """A non-negative int from the query string. Garbage returns the default rather
        than raising ValueError out of the route (which surfaced as a 500)."""
        try:
            v = int(request.args.get(name, default))
        except (TypeError, ValueError):
            return default
        v = max(lo, v)
        return min(v, hi) if hi is not None else v

    def _resolve_stale_status(proj):
        """Downgrade a status left mid-flight by a disconnected client.

        ``importing``/``processing`` are set before the SSE stream starts and cleared
        by the generator's own exit. If the browser goes away the generator is
        abandoned, so the session would advertise work that is no longer running,
        forever. ``runs.active`` distinguishes that from a genuinely in-flight run
        (e.g. another tab watching the same session), which must be left alone."""
        if proj.get("status") in ("importing", "processing") \
                and not runs.active(proj.get("run_id")):
            proj["status"] = "staged" if proj.get("row_count") else "new"
            proj["run_id"] = ""
            store.upsert_db_project(proj)
        return proj

    def _resolve_profile(body) -> ConnectionProfile:
        """Return a full ConnectionProfile (with secrets) from a request body that
        carries either a saved ``profile_id`` or an inline ``profile`` dict. For an
        inline profile that references a saved id, blank secret fields are filled
        from the stored profile so the user needn't retype a password to test."""
        pid = body.get("profile_id")
        if pid:
            return vault.get_profile(pid)          # may raise VaultLocked/VaultError
        raw = body.get("profile") or body
        prof = ConnectionProfile.from_dict(raw)
        if prof.id:
            try:
                stored = vault.get_profile(prof.id)
                for f in ConnectionProfile.SECRET_FIELDS:
                    if not getattr(prof, f, ""):
                        setattr(prof, f, getattr(stored, f, ""))
            except VaultLocked:
                # VaultLocked subclasses VaultError, so the broad catch below used to
                # swallow it and connect with a BLANK password — the user then got an
                # authentication failure from their own database instead of being told
                # to unlock the vault.
                raise
            except VaultError:
                # No such saved profile (e.g. an id from a deleted one). Fine: use the
                # inline fields as given.
                pass
        return prof

    # ------------------------------- state -------------------------------
    @app.route("/api/db/state")
    def db_state():
        """Everything the DB tab needs on load: vault lock state + session list."""
        return jsonify({
            "vault": vault.status(),
            "projects": store.db_project_summaries(),
        })

    # ------------------------------- vault -------------------------------
    @app.route("/api/db/vault/unlock", methods=["POST"])
    def db_vault_unlock():
        """Unlock the connection vault, deriving its key from the master password.
        The key is held in memory only."""
        data = request.get_json(force=True) or {}
        try:
            status = vault.unlock(data.get("password") or "")
        except VaultError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"vault": status})

    @app.route("/api/db/vault/lock", methods=["POST"])
    def db_vault_lock():
        """Wipe the vault key from memory. Stored credentials become unreadable until
        the next unlock."""
        return jsonify({"vault": vault.lock()})

    @app.route("/api/db/vault/change-password", methods=["POST"])
    def db_vault_change_pw():
        """Change the vault master password, re-encrypting the stored profiles."""
        data = request.get_json(force=True) or {}
        try:
            status = vault.change_password(data.get("old") or "", data.get("new") or "")
        except VaultError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"vault": status})

    # ------------------------------ profiles ------------------------------
    @app.route("/api/db/profiles", methods=["GET"])
    def db_profiles_get():
        """List saved connection profiles, MASKED - passwords never leave the server,
        only a has_password flag."""
        try:
            return jsonify({"profiles": vault.list_profiles()})
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423

    @app.route("/api/db/profiles", methods=["POST"])
    def db_profiles_post():
        """Create or update one connection profile in the encrypted vault."""
        data = request.get_json(force=True) or {}
        try:
            prof = vault.upsert_profile(data.get("profile") or data)
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        except VaultError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"profile": prof})

    @app.route("/api/db/profiles/<profile_id>", methods=["DELETE"])
    def db_profiles_delete(profile_id):
        """Delete a saved connection profile. Existing import sessions are unaffected -
        their staged copy is independent of the source connection."""
        try:
            ok = vault.delete_profile(profile_id)
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        return jsonify({"ok": ok})

    @app.route("/api/db/test-connection", methods=["POST"])
    def db_test_connection():
        """Try to connect and report the result. Accepts either a saved profile_id or
        discrete connection fields, so the dialog can test before saving."""
        body = request.get_json(force=True) or {}
        try:
            prof = _resolve_profile(body)
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        # test_connection never raises — it returns {ok, message, ...}.
        return jsonify(conns.test_connection(prof))

    @app.route("/api/db/tables", methods=["POST"])
    def db_tables():
        """List the source database's tables, for the import picker."""
        body = request.get_json(force=True) or {}
        try:
            prof = _resolve_profile(body)
            return jsonify({"tables": conns.list_tables(prof)})
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        except (ConnectionError_, Exception) as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/db/columns", methods=["POST"])
    def db_columns():
        """List one table's columns with their source SQL types."""
        body = request.get_json(force=True) or {}
        table = body.get("table") or ""
        if not table:
            return jsonify({"error": "No table specified."}), 400
        try:
            prof = _resolve_profile(body)
            return jsonify({"columns": conns.list_columns(prof, table)})
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        except (ConnectionError_, Exception) as e:
            return jsonify({"error": str(e)}), 400

    # ------------------------------ sessions ------------------------------
    @app.route("/api/db/sessions", methods=["GET"])
    def db_sessions_get():
        """Summaries of every import session (no secrets - credentials live in the vault)."""
        return jsonify({"sessions": store.db_project_summaries()})

    @app.route("/api/db/session/<session_id>", methods=["GET"])
    def db_session_get(session_id):
        """Full metadata for one import session, including its column configuration."""
        proj = store.get_db_project(session_id)
        if not proj:
            return jsonify({"error": "not found"}), 404
        return jsonify({"session": _resolve_stale_status(proj)})

    @app.route("/api/db/session/<session_id>", methods=["DELETE"])
    def db_session_delete(session_id):
        """Delete a session: drop its DuckDB staging file and forget its metadata.
        The SOURCE database is untouched."""
        from .staging import StagingManager
        StagingManager(session_id).drop()
        store.delete_db_project(session_id)
        return jsonify({"ok": True})

    # -------------------- import / staging / processing --------------------
    @app.route("/api/db/import", methods=["POST"])
    def db_import():
        """Body: {profile_id, table, name?, selection:{mode,n,where,order_by},
        key_columns?, chunk?}. Streams import progress over SSE and creates a staged
        session backed by a DuckDB file."""
        body = request.get_json(force=True) or {}
        table = (body.get("table") or "").strip()
        if not body.get("profile_id"):
            return jsonify({"error": "Choose a saved connection to import from."}), 400
        if not table:
            return jsonify({"error": "No table selected."}), 400
        try:
            profile = vault.get_profile(body["profile_id"])
        except VaultLocked as e:
            return jsonify({"error": str(e), "locked": True}), 423
        except VaultError as e:
            return jsonify({"error": str(e)}), 400

        sel = SelectionSpec(**{k: v for k, v in (body.get("selection") or {}).items()
                               if k in SelectionSpec.__dataclass_fields__})
        chunk = int(body.get("chunk") or DEFAULT_CHUNK)
        run_id = body.get("run_id") or None
        import uuid as _uuid
        run_id = run_id or _uuid.uuid4().hex[:12]

        def gen():
            stop = runs.new(run_id)
            session = None
            try:
                yield sse("start", {"run_id": run_id, "table": table})
                cols = importer.capture_columns(profile, table)
                key_cols = body.get("key_columns") or [c["name"] for c in cols if c.get("primary_key")]
                total = importer.count_rows(profile, table, sel)
                yield sse("status", {"message": f"Importing up to {total} rows…", "total": total})

                session = ImportSession(
                    name=(body.get("name") or f"{profile.name}:{table}").strip(),
                    profile_id=profile.id, table=table, selection=sel.to_dict(),
                    key_columns=key_cols, status="importing",
                    columns=[ColumnDef(name=c["name"], ctype=ColumnType.SOURCE.value,
                                       source_type=c["source_type"], duckdb_type=c["duckdb_type"]).to_dict()
                             for c in cols])
                # Recorded so a session left "importing" by a disconnected client can be
                # told apart from one whose import is genuinely still running.
                rec = session.to_dict()
                rec["run_id"] = run_id
                store.add_db_project(rec)

                sm = StagingManager(session.id, session.staging_table)
                sm.create_table(cols)
                done = 0
                for batch in importer.stream(profile, table, sel, chunk=chunk):
                    if stop.is_set():
                        break
                    done += sm.insert_chunk(batch, cols, key_cols)
                    yield sse("progress", {"done": done, "total": total})

                stopped = stop.is_set()
                session.row_count = done
                session.status = "cancelled" if stopped else "staged"
                session.source_fingerprint = {"count": done, "captured_at": datetime.utcnow().isoformat()}
                session.updated = datetime.utcnow().isoformat()
                rec = session.to_dict()
                rec["run_id"] = ""
                store.upsert_db_project(rec)
                yield sse("done", {"session_id": session.id, "row_count": done, "stopped": stopped})
            except Exception as e:
                if session is not None:
                    session.status = "error"
                    rec = session.to_dict()
                    rec["run_id"] = ""
                    store.upsert_db_project(rec)
                yield sse("error", {"message": str(e)})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/db/session/<session_id>/rows", methods=["GET"])
    def db_session_rows(session_id):
        """One page of staged rows for the grid. ``limit`` is capped at 500 so a bad
        client cannot ask the server to materialize an unbounded page."""
        proj = store.get_db_project(session_id)
        if not proj:
            return jsonify({"error": "not found"}), 404
        _resolve_stale_status(proj)
        offset = _int_arg("offset", 0)
        limit = _int_arg("limit", 100, lo=1, hi=500)
        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        try:
            return jsonify(sm.get_page(offset=offset, limit=limit))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/db/session/<session_id>/cell", methods=["PUT"])
    def db_session_cell(session_id):
        """Edit one staged cell. Marks it dirty and captures the original value, which
        is what later feeds the dry-run preview and the audit log."""
        proj = store.get_db_project(session_id)
        if not proj:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(force=True) or {}
        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        try:
            res = sm.update_cell(body.get("rowid"), body.get("column"), body.get("value"))
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/db/session/<session_id>/columns", methods=["POST"])
    def db_session_add_column(session_id):
        """Add a column to the staging table and record its role on the session.
        The role (source / web_source / prompt / output) decides how the processing
        step fills it. Columns added here exist only in staging until write-back."""
        proj = store.get_db_project(session_id)
        if not proj:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(force=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Column name required."}), 400
        ctype = body.get("ctype") or ColumnType.OUTPUT.value
        # The type lands in DDL, which cannot be parameterized — validate, don't trust.
        ddl_type = sanitize_ddl_type(body.get("duckdb_type"))
        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        try:
            sm.add_column(name, ddl_type)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        # Track the new column (+ any prompt/web config) on the session record.
        cols = proj.get("columns", [])
        if not any(c.get("name") == name for c in cols):
            cols.append(ColumnDef(name=name, ctype=ctype, duckdb_type=ddl_type,
                                  prompt_template=body.get("prompt_template", ""),
                                  input_columns=body.get("input_columns", []),
                                  search_query=body.get("search_query", ""),
                                  domains=body.get("domains", [])).to_dict())
            proj["columns"] = cols
            store.upsert_db_project(proj)
        return jsonify({"ok": True, "columns": proj["columns"]})

    @app.route("/api/db/session/<session_id>/process", methods=["POST"])
    def db_session_process(session_id):
        """Body: {server_url, model, num_ctx?, columns:[{name, ctype, prompt_template,
        input_columns, search_query, domains, duckdb_type?}], run_id?}. Ensures each
        AI column exists in staging + saves its config on the session, then streams
        row-by-row processing (reusing the app's generate_one + web_search)."""
        proj = store.get_db_project(session_id)
        if not proj:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(force=True) or {}
        model = (body.get("model") or "").strip()
        if not model:
            return jsonify({"error": "Choose a model to process with."}), 400
        server_url = body.get("server_url") or core.DEFAULT_LOCAL_URL
        num_ctx = int(body.get("num_ctx") or store.config.get("default_num_ctx", 4096))
        run_id = body.get("run_id") or None
        import uuid as _uuid
        run_id = run_id or _uuid.uuid4().hex[:12]

        run_cols = [c for c in (body.get("columns") or [])
                    if c.get("name") and c.get("ctype") in ("web_source", "prompt", "output")]
        if not run_cols:
            return jsonify({"error": "No AI columns configured."}), 400

        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        # Ensure each configured column exists in staging + persist its config.
        cols = proj.get("columns", [])
        by_name = {c["name"]: c for c in cols}
        for rc in run_cols:
            ddl_type = sanitize_ddl_type(rc.get("duckdb_type"))
            sm.add_column(rc["name"], ddl_type)
            cfg = ColumnDef(name=rc["name"], ctype=rc["ctype"],
                            duckdb_type=ddl_type,
                            prompt_template=rc.get("prompt_template", ""),
                            input_columns=rc.get("input_columns", []),
                            search_query=rc.get("search_query", ""),
                            domains=rc.get("domains", [])).to_dict()
            by_name[rc["name"]] = cfg
        proj["columns"] = list(by_name.values())
        proj["status"] = "processing"
        proj["run_id"] = run_id
        store.upsert_db_project(proj)

        processor = StagingProcessor(
            sm, fill_prompt=evals.fill_prompt, generate_one=ctx.generate_one,
            web_search=core.web_search)

        def gen():
            stop = runs.new(run_id)
            try:
                yield sse("start", {"run_id": run_id})
                for kind, data in processor.process(
                        run_cols, server_url=server_url, model=model, num_ctx=num_ctx,
                        web_min_pages=core.MIN_CRAWLED_PAGES, stop_event=stop):
                    yield sse(kind, data)
                proj["status"] = "cancelled" if stop.is_set() else "ready"
                proj["run_id"] = ""
                store.upsert_db_project(proj)
                yield sse("done", {"stopped": stop.is_set()})
            except Exception as e:
                proj["status"] = "error"
                proj["run_id"] = ""
                store.upsert_db_project(proj)
                yield sse("error", {"message": str(e)})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------- safety / write-back -----------------------
    def _session_and_profile(session_id):
        """Return (proj_dict, ConnectionProfile) or (error_response, None)."""
        proj = store.get_db_project(session_id)
        if not proj:
            return (jsonify({"error": "not found"}), 404), None
        try:
            profile = vault.get_profile(proj.get("profile_id"))
        except VaultLocked as e:
            return (jsonify({"error": str(e), "locked": True}), 423), None
        except VaultError as e:
            return (jsonify({"error": str(e)}), 400), None
        return proj, profile

    @app.route("/api/db/session/<session_id>/dry-run", methods=["POST"])
    def db_session_dry_run(session_id):
        """Render the exact UPDATE statements the pending edits would run, without
        executing anything. Capped by max_rows/max_cols so the preview stays readable."""
        proj, profile = _session_and_profile(session_id)
        if profile is None:
            return proj
        body = request.get_json(force=True) or {}
        max_rows = int(body.get("max_rows", 50))
        max_cols = int(body.get("max_cols", 20))
        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        try:
            changes = sm.dirty_changes()
            result = DryRunEngine(conns).preview(profile, proj, changes,
                                                 max_rows=max_rows, max_cols=max_cols)
            return jsonify(result.to_dict())
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/db/session/<session_id>/check-conflicts", methods=["POST"])
    def db_session_conflicts(session_id):
        """Compare the source rows against the fingerprints taken at import time and
        report what is unchanged / changed / new / deleted - i.e. whether someone else
        edited the source while you were working."""
        proj, profile = _session_and_profile(session_id)
        if profile is None:
            return proj
        try:
            report = ConflictDetector(conns).check(profile, proj)
            return jsonify(report.to_dict())
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/db/session/<session_id>/writeback", methods=["POST"])
    def db_session_writeback(session_id):
        """Apply the staged edits to the real database, streaming progress over SSE.

        This is the ONLY endpoint that opens the source writable. Transactional,
        conflict-guarded (``on_conflict``), audited, and interruptible through the run
        registry. ``mode`` is "bulk" or row-by-row."""
        proj, profile = _session_and_profile(session_id)
        if profile is None:
            return proj
        body = request.get_json(force=True) or {}
        mode = body.get("mode", "bulk")
        # Approval must be explicit in THIS request. It used to fall back to a
        # persisted proj["auto_approve"], which nothing in the app ever sets and which
        # upsert_db_project would store from any client-supplied key — an arbitrary
        # field could switch off the only guard on the one endpoint that opens the
        # source database writable.
        approved = bool(body.get("approved"))
        on_conflict = body.get("on_conflict", "abort")
        continue_on_error = bool(body.get("continue_on_error"))
        import uuid as _uuid
        run_id = body.get("run_id") or _uuid.uuid4().hex[:12]
        sm = StagingManager(session_id, proj.get("staging_table", "staged"))
        engine = WriteBackEngine(conns)

        def gen():
            stop = runs.new(run_id)
            try:
                yield sse("start", {"run_id": run_id})
                changes = sm.dirty_changes()
                applied = 0
                applied_pairs = []
                for kind, data in engine.run(profile, proj, changes, mode=mode,
                                             approved=approved, on_conflict=on_conflict,
                                             continue_on_error=continue_on_error, stop_event=stop):
                    if kind == "done":
                        applied = data.get("applied_rows", 0)
                        applied_pairs = data.get("applied_pairs") or []
                        # Not part of the wire contract — it exists to drive the
                        # retirement below and would only bloat the SSE frame.
                        data = {k: v for k, v in data.items() if k != "applied_pairs"}
                    yield sse(kind, data)
                if applied:
                    # Retire the cells that actually committed, then re-read the source
                    # so the rows we just wrote become the new conflict baseline. Without
                    # this the session stays dirty forever and the next conflict check
                    # flags our own writes as third-party changes.
                    try:
                        sm.retire_changes(applied_pairs)
                        sm.rebaseline(ConflictDetector(conns).current_fingerprints(profile, proj))
                    except Exception as e:
                        yield sse("warn", {"message": f"Write-back applied, but the staging "
                                                      f"baseline could not be refreshed: {e}"})
                    proj["status"] = "written"
                    proj["updated"] = datetime.utcnow().isoformat()
                    store.upsert_db_project(proj)
            except Exception as e:
                yield sse("error", {"message": str(e)})
            finally:
                runs.done(run_id)

        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/db/session/<session_id>/audit", methods=["GET"])
    def db_session_audit(session_id):
        """Read the session's append-only audit log, one page at a time."""
        from .audit import AuditLog
        log = AuditLog(session_id)
        limit = _int_arg("limit", 500, lo=1, hi=5000)
        offset = _int_arg("offset", 0)
        return jsonify({"entries": log.read(limit=limit, offset=offset), "total": log.count()})
