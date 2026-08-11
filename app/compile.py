#!/usr/bin/env python3
"""
Local Streaming Token — "Compile Data" orchestration.

The RAG store (``rag.py``) can chunk + embed a single item, but historically that
happened *implicitly* (libraries re-indexed on every send, persona docs indexed on
add). This module turns compilation into an **explicit, resumable, pre-warming** step
shared by Libraries and Personas: it walks every item, skips ones whose content and
build signature are unchanged (unless ``force``), embeds the rest through the existing
``rag.upsert_item`` (token-accurate semantic chunking + optional per-chunk LLM
contextualization), prunes deleted items, and records a **manifest** so the UI can show
Compiled / Stale / Not-compiled and so chat can *require* compilation before using RAG.

Manifests are a rebuildable cache (like ``rag.duckdb``): one JSON file per data profile
at ``core.COMPILED_FILE``, keyed ``"library:<id>"`` / ``"persona:<id>"``. They are never
exported with a library/persona.

An ``emit(event, data)`` callback streams progress for the SSE routes; pass ``None`` for
a silent run.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from . import core, rag, persona_store, ingest, logic
from .persona import PersonaService

LIBRARY = "library"
PERSONA_KNOWLEDGE = persona_store.ST_KNOWLEDGE   # "persona_knowledge"
PERSONA_MEMORY = persona_store.ST_MEMORY         # "persona_memory"

# A chat's own corpus. Unlike libraries and personas there is no Compile button and no
# compile gate: a conversation changes every turn, so ``sync_chat`` keeps the index fresh
# on the send path instead, embedding only what is new or edited.
CHAT = "chat"                    # manifest kind
CHAT_THREAD = "chat_thread"      # source_type — the message bodies
CHAT_ATTACH = "chat_attach"      # source_type — pinned attachments + staged <Data>

_MANIFEST_LOCK = threading.RLock()
# One lock per chat: the parallel engine can run several lanes against the same chat id,
# and two concurrent syncs would interleave the backend's delete+insert waves.
_CHAT_LOCKS = {}


# --------------------------- manifest persistence ---------------------------

def _load_all() -> dict:
    # core.load_json transparently decrypts (compiled.json is encrypted at rest).
    return core.load_json(Path(core.COMPILED_FILE), {}) or {}


def _save_all(data: dict) -> None:
    core.save_json(Path(core.COMPILED_FILE), data)


def _manifest_key(kind: str, obj_id: str) -> str:
    """Manifest key for ``kind:id`` under the ACTIVE vector-store backend.

    Both backends can hold data at once, so a single key would let the badge claim a
    library is compiled when only the *other* store actually has its vectors. Keys are
    therefore namespaced ``library:<id>@lance`` / ``@duckdb``, giving each backend an
    independent compiled-state.

    Deliberately not folded into ``signature()``: that would mark everything stale on
    every switch, even when the target backend's vectors are perfectly valid."""
    return f"{kind}:{obj_id}@{rag.backend_name()}"


def _get_manifest(key: str) -> dict:
    with _MANIFEST_LOCK:
        data = _load_all()
        found = data.get(key)
        if found is not None:
            return found
        # Manifests written before backends existed describe the DuckDB store.
        if key.endswith("@duckdb"):
            legacy = data.get(key[: -len("@duckdb")])
            if legacy is not None:
                return legacy
        return {}


def _put_manifest(key: str, manifest: dict) -> None:
    with _MANIFEST_LOCK:
        data = _load_all()
        data[key] = manifest
        _save_all(data)


def _drop_manifests_all_backends(kind: str, obj_id: str) -> None:
    """Drop ``kind:id`` manifests under EVERY backend suffix, plus the pre-backend
    legacy key. ``_manifest_key`` namespaces by the active backend, so deleting while
    LanceDB was active used to leave the DuckDB manifest (and vice versa) on disk
    forever, still describing content the user asked to remove."""
    with _MANIFEST_LOCK:
        data = _load_all()
        prefix = f"{kind}:{obj_id}"
        doomed = [k for k in data if k == prefix or k.startswith(prefix + "@")]
        if doomed:
            for k in doomed:
                del data[k]
            _save_all(data)


def forget_library(lib_id: str) -> None:
    """Drop a library's compile manifest (called on library delete)."""
    _drop_manifests_all_backends(LIBRARY, lib_id)


def forget_persona(persona_id: str) -> None:
    """Drop a persona's compile manifest (called on persona delete)."""
    _drop_manifests_all_backends("persona", persona_id)


def forget_chat(chat_id: str) -> None:
    """Drop a chat's compile manifest (called on chat delete and RAG forget). The
    vectors themselves go via ``rag.delete_source``."""
    _drop_manifests_all_backends(CHAT, chat_id)


# --------------------------- signature / fingerprints ---------------------------

def signature(embed_model: str) -> dict:
    """The build identity a compiled item must match to be considered fresh: the
    embedding model plus the active chunker and its size/overlap. A change to any of
    these invalidates previously compiled chunks."""
    cfg = rag.chunk_config()
    return {
        "embedding_model": embed_model or "",
        "chunker": rag.active_chunker_id(),
        "chunk_size": cfg.get("size"),
        "chunk_overlap": cfg.get("overlap"),
    }


def _file_fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return ""


def _content_fingerprint(text: str) -> str:
    return rag._hash(text or "")


def _emit(emit, event, **data):
    if emit is not None:
        try:
            emit(event, data)
        except Exception:
            pass


# --------------------------- progress / ETA plumbing ---------------------------
#
# The bar is denominated in CHUNKS, not items: a library can be three items and
# 40,000 chunks, and the old item-denominated bar sat frozen at 0% for the entire
# embed run. Each phase reports its own counters and carries a weight, so the client
# can turn several differently-measured phases into one honest overall fraction.
#
# Deliberately, the server emits raw counters + elapsed and NOT an ETA: the client
# keeps a smoothed rate and re-renders the countdown every second, so the number ticks
# down between frames instead of lurching whenever a frame happens to arrive.

# Phase weights, as a rough share of wall time. Contextual chunking is one LLM call
# per chunk and dwarfs everything else when it's on, so the split changes with it.
_WEIGHTS_PLAIN = {"chunk": 0.12, "embed": 0.88}
_WEIGHTS_CONTEXTUAL = {"chunk": 0.04, "context": 0.76, "embed": 0.20}
_WEIGHTS_PLAIN_PARSE = {"parse": 0.15, "chunk": 0.10, "embed": 0.75}
_WEIGHTS_CONTEXTUAL_PARSE = {"parse": 0.06, "chunk": 0.03, "context": 0.71, "embed": 0.20}
_PHASE_LABELS = {"parse": "Reading documents", "chunk": "Chunking",
                 "context": "Contextualizing", "embed": "Embedding"}
_PHASE_UNITS = {"parse": "files", "chunk": "items", "context": "chunks",
                "embed": "chunks"}


class Progress:
    """Turns fine-grained phase callbacks into throttled ``progress`` SSE frames.

    Throttling matters: a 50k-chunk compile would otherwise push tens of thousands of
    frames through the SSE queue. Transitions and phase completions always emit, so
    the bar never stalls short of the end of a phase."""

    def __init__(self, emit, contextual: bool = False, with_parse: bool = False,
                 min_interval: float = 0.25):
        self.emit = emit
        self.min_interval = min_interval
        if with_parse:
            weights = _WEIGHTS_CONTEXTUAL_PARSE if contextual else _WEIGHTS_PLAIN_PARSE
        else:
            weights = _WEIGHTS_CONTEXTUAL if contextual else _WEIGHTS_PLAIN
        self.phases = [{"id": pid, "label": _PHASE_LABELS.get(pid, pid),
                        "unit": _PHASE_UNITS.get(pid, ""), "weight": w}
                       for pid, w in weights.items()]
        self.started = time.time()
        self._last_at = 0.0
        self._last_key = None

    def plan(self, **totals):
        _emit(self.emit, "plan", phases=self.phases, totals=totals)

    def update(self, phase, done, total, **extra):
        now = time.time()
        key = (phase, done >= total)
        force = (done >= total) or (key != self._last_key)
        if not force and (now - self._last_at) < self.min_interval:
            return
        self._last_at = now
        self._last_key = key
        _emit(self.emit, "progress", phase=phase, done=int(done), total=int(total),
              unit=_PHASE_UNITS.get(phase, ""),
              elapsed_s=round(now - self.started, 2), **extra)

    def callback(self):
        """The ``on_progress(phase, done, total, **extra)`` hook rag.upsert_items wants."""
        return lambda phase, done, total, **extra: self.update(phase, done, total, **extra)


# --------------------------- status (cheap, no embedding) ---------------------------

def _state_from(manifest: dict, sig: dict, current_items: dict, stored_chunks: int) -> dict:
    """Decide compiled / stale / none from a manifest vs the current items + signature.
    ``current_items`` maps item_id -> fingerprint. Never embeds or parses heavy files
    beyond the cheap fingerprints the caller already gathered."""
    # Keyed on the manifest's absence, not on an empty items map: a library whose items
    # are all empty compiles to zero items, and treating that as "never compiled" made
    # the badge read "Not compiled" right after the toast said "Compiled: 0 chunks".
    if not manifest:
        return {"state": "none", "reasons": ["not compiled yet"]}
    if not manifest.get("items") and current_items:
        return {"state": "none", "reasons": ["not compiled yet"]}
    reasons = []
    if manifest.get("signature") != sig:
        reasons.append("embedding model or chunker changed")
    stored = manifest.get("items") or {}
    if set(stored.keys()) != set(current_items.keys()):
        reasons.append("items added or removed")
    else:
        for iid, fp in current_items.items():
            if (stored.get(iid) or {}).get("fingerprint") != fp:
                reasons.append("content edited")
                break
    if stored_chunks <= 0 and current_items:
        reasons.append("vector store empty (recompile)")
    state = "stale" if reasons else "compiled"
    out = {
        "state": state,
        "reasons": reasons,
        "items": len(stored),
        "chunks": manifest.get("chunks", stored_chunks),
        "embedding_model": (manifest.get("signature") or {}).get("embedding_model", ""),
        "chunker": (manifest.get("signature") or {}).get("chunker", ""),
        "compiled_at": manifest.get("compiled_at", ""),
    }
    return out


def _library_items(lib: dict) -> list:
    """[(item_id, content, meta)] for a library's non-empty items, deriving a stable-ish
    id, so retrieval scopes line up with what was written."""
    out = []
    lib_id = lib.get("id")
    for idx, it in enumerate(lib.get("items", [])):
        content = (it.get("content") or "").strip()
        if not content:
            continue
        item_id = it.get("id") or f"{lib_id}:{idx}"
        meta = {"label": rag._item_label(it), "type": it.get("type", "")}
        out.append((item_id, content, meta))
    return out


def library_status(lib: dict, embed_model: str) -> dict:
    lib_id = lib.get("id")
    sig = signature(embed_model)
    current = {iid: _content_fingerprint(content) for iid, content, _ in _library_items(lib)}
    stored_chunks = rag.count_chunks(LIBRARY, lib_id)
    return _state_from(_get_manifest(_manifest_key(LIBRARY, lib_id)), sig, current, stored_chunks)


def _chat_lock(chat_id: str):
    with _MANIFEST_LOCK:
        lock = _CHAT_LOCKS.get(chat_id)
        if lock is None:
            lock = _CHAT_LOCKS[chat_id] = threading.RLock()
        return lock


def _chat_sources(scope: str) -> list:
    """[(source_type, manifest_prefix, builder)] for the corpora a scope covers."""
    out = []
    if scope in ("attachments", "both"):
        out.append((CHAT_ATTACH, "a", logic.attachment_items))
    if scope in ("thread", "both"):
        out.append((CHAT_THREAD, "t", logic.thread_items))
    return out


def chat_status(chat: dict, embed_model: str) -> dict:
    """compiled / stale / none for a chat's RAG index. Cheap fingerprints only — never
    embeds. Mostly for tests and diagnostics: there is no compile gate on chats, since
    ``sync_chat`` keeps them fresh on the send path."""
    chat_id = (chat or {}).get("id") or ""
    scope = logic.rag_scope(chat)
    current, stored_chunks = {}, 0
    for source_type, prefix, build in _chat_sources(scope):
        for item_id, content, _meta in build(chat):
            current[f"{prefix}:{item_id}"] = _content_fingerprint(content)
        stored_chunks += rag.count_chunks(source_type, chat_id)
    return _state_from(_get_manifest(_manifest_key(CHAT, chat_id)),
                       signature(embed_model), current, stored_chunks)


def sync_chat(chat: dict, embed_fn, embed_model: str, contextualize=None,
              force: bool = False, on_status=None, stop_event=None,
              batch_size: int = 64, max_workers: int = 1) -> dict:
    """Incrementally index a chat's in-scope corpora into the vector store.

    The send-path counterpart to ``compile_library``. A library is compiled once by an
    explicit button press and RAG refuses to touch it until that happens; a conversation
    changes with every message, so the same contract would mean a Compile click per turn.
    Freshness is maintained continuously instead, and only new or edited items are
    embedded — an unchanged chat costs one store query per corpus and no embedding at all.

    Item identity comes from ``logic.thread_items`` / ``logic.attachment_items``; see
    those for why a raw message index is a stable key here.

    Returns ``{"scope", "indexed", "skipped", "chunks", "sources"}``.
    """
    chat_id = (chat or {}).get("id") or ""
    scope = logic.rag_scope(chat)
    sources = _chat_sources(scope)
    summary = {"scope": scope, "indexed": 0, "skipped": 0, "chunks": 0,
               "sources": [s[0] for s in sources]}
    if not chat_id or not sources:
        return summary

    key = _manifest_key(CHAT, chat_id)
    sig = signature(embed_model)
    with _chat_lock(chat_id):
        prev = _get_manifest(key)
        prev_items = prev.get("items") or {}
        same_sig = prev.get("signature") == sig
        new_items = dict(prev_items) if not force else {}

        pending = []          # [(source_type, changed_items, all_ids)]
        for source_type, prefix, build in sources:
            items = build(chat)
            # ONE scoped query per corpus, rather than compile_library's per-item
            # stored_hashes call: that is N store round-trips per send on a long thread.
            stored = {d.get("item_id") for d in rag.list_items(source_type, chat_id)}
            changed = []
            for item_id, content, meta in items:
                mkey = f"{prefix}:{item_id}"
                fp = _content_fingerprint(content)
                if (same_sig and not force
                        and (prev_items.get(mkey) or {}).get("fingerprint") == fp
                        and item_id in stored):
                    summary["skipped"] += 1
                    continue
                changed.append((item_id, content, meta, mkey, fp))
            pending.append((source_type, changed, [i for i, _c, _m in items]))

        if on_status is not None:
            total_changed = sum(len(c) for _s, c, _a in pending)
            if total_changed:
                try:
                    on_status(f"📚 Indexing {total_changed} new/edited item(s) "
                              f"for retrieval…")
                except Exception:
                    pass

        for source_type, changed, all_ids in pending:
            if changed:
                counts = rag.upsert_items(
                    source_type, chat_id,
                    [(i, c, m) for i, c, m, _k, _f in changed],
                    embed_fn, embed_model, contextualize=contextualize,
                    batch_size=batch_size, max_workers=max_workers,
                    stop_event=stop_event)
                complete = (counts.get("__meta__") or {}).get("complete") or set()
                for item_id, _c, _m, mkey, fp in changed:
                    if item_id in complete:
                        # Anything that did not embed cleanly is left OUT of the
                        # manifest, so it retries next send instead of being recorded
                        # as done (same rule as compile_library).
                        new_items[mkey] = {"fingerprint": fp,
                                           "chunks": counts.get(item_id, 0)}
                        summary["indexed"] += 1
            # Prune against what the chat STILL has, which is what self-heals a
            # regenerate (tail ids gone) or a cleared thread with no client cooperation.
            rag.prune_items(source_type, chat_id, all_ids)
            summary["chunks"] += rag.count_chunks(source_type, chat_id)

        # Only the synced corpora are reconciled. A scope the user switched away from
        # keeps its rows: they are never retrieved, and they are exactly what
        # cached_vectors reuses if the scope is switched back.
        live = {f"{p}:{i}" for st, p, b in sources for i, _c, _m in b(chat)}
        stale_prefixes = tuple(f"{p}:" for _st, p, _b in sources)
        new_items = {k: v for k, v in new_items.items()
                     if k in live or not k.startswith(stale_prefixes)}

        _put_manifest(key, {"signature": sig, "items": new_items,
                            "chunks": summary["chunks"],
                            "compiled_at": datetime.utcnow().isoformat()})
    return summary


def _persona_source_items(persona_id: str) -> list:
    """[(doc_id, path, fingerprint)] for the persona's source documents."""
    ksvc = persona_store.KnowledgeService()
    d = ksvc.sources_dir(persona_id)
    out = []
    for p in sorted(d.glob("*")):
        if p.is_file():
            out.append((p.name, p, _file_fingerprint(p)))
    return out


def _persona_memory_items(persona_id: str) -> list:
    """[(mem_id, mem_dict, fingerprint)] for the persona's memories."""
    msvc = persona_store.MemoryService()
    out = []
    for mem in msvc.list_memories(persona_id):
        fp = f"{mem.get('updated', '')}"
        out.append((mem.get("id"), mem, fp))
    return out


def persona_status(persona: dict, embed_model: str) -> dict:
    pid = persona.get("id")
    sig = signature(embed_model)
    current = {}
    for doc_id, _p, fp in _persona_source_items(pid):
        current[f"k:{doc_id}"] = fp
    for mem_id, _m, fp in _persona_memory_items(pid):
        current[f"m:{mem_id}"] = fp
    stored_chunks = rag.count_chunks(PERSONA_KNOWLEDGE, pid) + rag.count_chunks(PERSONA_MEMORY, pid)
    return _state_from(_get_manifest(_manifest_key("persona", pid)), sig, current, stored_chunks)


# --------------------------- compile (embeds) ---------------------------

def compile_library(lib: dict, embed_fn, embed_model: str, contextualize=None,
                    force: bool = False, emit=None, batch_size: int = 64,
                    max_workers: int = 1, stop_event=None) -> dict:
    """Chunk + embed every item of ``lib`` into the store, skipping unchanged items
    (unless ``force``), prune removed items, and write the manifest. Returns a summary.

    Changed items are embedded in ONE bulk ``rag.upsert_items`` call — chunks pooled
    across items, embedded in bounded waves across every configured embedding server,
    reusing any vector already computed for identical text.

    Items that did not embed cleanly (a dead embed server, or a run the user cancelled)
    are deliberately LEFT OUT of the manifest, so the library reads as stale and those
    items recompile next time instead of being silently recorded as done."""
    lib_id = lib.get("id")
    key = _manifest_key(LIBRARY, lib_id)
    sig = signature(embed_model)
    prev = _get_manifest(key)
    prev_items = prev.get("items") or {}
    same_sig = prev.get("signature") == sig
    items = _library_items(lib)
    total = len(items)
    prog = Progress(emit, contextual=contextualize is not None)
    _emit(emit, "begin", total=total, kind="library", name=lib.get("name", ""))

    new_items = {}
    embedded = skipped = 0
    changed = []          # [(item_id, content, meta)] to embed in bulk
    changed_pos = {}      # item_id -> (index, label, fingerprint)
    for i, (item_id, content, meta) in enumerate(items):
        fp = _content_fingerprint(content)
        label = meta.get("label") or item_id
        _emit(emit, "item_start", index=i + 1, total=total, name=label)
        unchanged = (same_sig and not force
                     and (prev_items.get(item_id) or {}).get("fingerprint") == fp
                     and rag.stored_hashes(LIBRARY, lib_id, item_id, embed_model))
        if unchanged:
            n = (prev_items.get(item_id) or {}).get("chunks", 0)
            skipped += 1
            _emit(emit, "item_done", index=i + 1, total=total, name=label, chunks=n, skipped=True)
            new_items[item_id] = {"fingerprint": fp, "chunks": n}
        else:
            changed.append((item_id, content, meta))
            changed_pos[item_id] = (i + 1, label, fp)

    meta_out = {}
    if changed:
        prog.plan(items=total, changed=len(changed))
        counts = rag.upsert_items(LIBRARY, lib_id, changed, embed_fn, embed_model,
                                  contextualize=contextualize, batch_size=batch_size,
                                  max_workers=max_workers, on_progress=prog.callback(),
                                  stop_event=stop_event)
        meta_out = counts.get("__meta__") or {}
        complete = meta_out.get("complete") or set()
        for item_id, _c, _m in changed:
            n = counts.get(item_id, 0)
            idx, label, fp = changed_pos[item_id]
            if item_id in complete:
                embedded += 1
                new_items[item_id] = {"fingerprint": fp, "chunks": n}
                _emit(emit, "item_done", index=idx, total=total, name=label,
                      chunks=n, skipped=False)
            else:
                # Left out of the manifest on purpose — see the docstring.
                _emit(emit, "warn", name=label,
                      message="not fully embedded — will recompile")

    # Prune against every item the library STILL HAS, not against the manifest.
    # ``new_items`` deliberately omits anything that didn't embed cleanly, so pruning
    # by it deleted the stored chunks of items a stopped or half-failed run never
    # reached — with Force full rebuild that is every item, i.e. the whole library's
    # vectors. Items dropped from the library are still pruned, which is the point.
    # (This matches how compile_persona has always pruned; see below.)
    rag.prune_items(LIBRARY, lib_id, [item_id for item_id, _c, _m in items])
    failed = int(meta_out.get("failed") or 0)
    stopped = bool(meta_out.get("stopped"))
    manifest = {
        "signature": sig,
        "items": new_items,
        "chunks": rag.count_chunks(LIBRARY, lib_id),
        "compiled_at": datetime.utcnow().isoformat(),
    }
    _put_manifest(key, manifest)
    summary = {"state": "partial" if (failed or stopped) else "compiled",
               "items": len(new_items), "embedded": embedded,
               "skipped": skipped, "chunks": manifest["chunks"],
               "compiled_at": manifest["compiled_at"],
               "embedding_model": embed_model, "chunker": sig["chunker"],
               "failed": failed, "stopped": stopped,
               "cached": int(meta_out.get("cached") or 0),
               "lanes": meta_out.get("lanes") or []}
    _emit(emit, "compiled", **summary)
    return summary


def compile_persona(persona: dict, embed_fn, embed_model: str, contextualize=None,
                    force: bool = False, emit=None, batch_size: int = 64,
                    max_workers: int = 1, stop_event=None) -> dict:
    """Chunk + embed a persona's knowledge sources and memories, skip unchanged, prune
    removed, record ``stores.embedding_model_used`` on the persona, and write the
    manifest. Returns a summary.

    Changed knowledge documents are embedded in one bulk ``rag.upsert_items`` call
    (chunks pooled + batched/concurrent per ``batch_size``/``max_workers``). Memories
    are few and stay per-item."""
    pid = persona.get("id")
    key = _manifest_key("persona", pid)
    sig = signature(embed_model)
    prev = _get_manifest(key)
    prev_items = prev.get("items") or {}
    same_sig = prev.get("signature") == sig

    msvc = persona_store.MemoryService()
    sources = _persona_source_items(pid)
    memories = _persona_memory_items(pid)
    total = len(sources) + len(memories)
    prog = Progress(emit, contextual=contextualize is not None, with_parse=True)
    _emit(emit, "begin", total=total, kind="persona", name=persona.get("profile", {}).get("name", ""))

    new_items = {}
    embedded = skipped = 0
    idx = 0

    # --- Knowledge documents (changed ones parsed in parallel, embedded in bulk) ---
    to_parse = []         # [(doc_id, path, fp, index)]
    changed_pos = {}      # doc_id -> (index, fp)
    for doc_id, path, fp in sources:
        idx += 1
        mkey = f"k:{doc_id}"
        _emit(emit, "item_start", index=idx, total=total, name=doc_id)
        unchanged = (same_sig and not force
                     and (prev_items.get(mkey) or {}).get("fingerprint") == fp
                     and rag.stored_hashes(PERSONA_KNOWLEDGE, pid, doc_id, embed_model))
        if unchanged:
            n = (prev_items.get(mkey) or {}).get("chunks", 0)
            skipped += 1
            _emit(emit, "item_done", index=idx, total=total, name=doc_id, chunks=n, skipped=True)
            new_items[mkey] = {"fingerprint": fp, "chunks": n}
        else:
            to_parse.append((doc_id, path, fp, idx))
            changed_pos[doc_id] = (idx, fp)

    prog.plan(items=total, changed=len(to_parse))

    # Parsing PDFs/EPUBs is CPU-bound and was the other serial half of a slow compile;
    # extract_many spreads it over a process pool. Files are already in the persona's
    # sources/ dir (that's what we globbed), so nothing is copied.
    changed = []          # [(doc_id, text, meta)]
    if to_parse:
        by_path = {str(p): (doc_id, fp) for doc_id, p, fp, _i in to_parse}
        results = ingest.extract_many(
            [str(p) for _d, p, _f, _i in to_parse],
            on_progress=lambda d, t, name: prog.update("parse", d, t))
        for res in results:
            doc_id, fp = by_path.get(res.get("path"), (None, ""))
            if doc_id is None:
                continue
            if res.get("ok"):
                changed.append((doc_id, res.get("text") or "",
                                {"name": doc_id, "tags": []}))
            else:
                _emit(emit, "warn", name=doc_id, message=res.get("error", "parse failed"))

    meta_out = {}
    if changed:
        counts = rag.upsert_items(PERSONA_KNOWLEDGE, pid, changed, embed_fn, embed_model,
                                  contextualize=contextualize, batch_size=batch_size,
                                  max_workers=max_workers, on_progress=prog.callback(),
                                  stop_event=stop_event)
        meta_out = counts.get("__meta__") or {}
        complete = meta_out.get("complete") or set()
        for doc_id, _t, _m in changed:
            n = counts.get(doc_id, 0)
            didx, fp = changed_pos[doc_id]
            if doc_id in complete:
                embedded += 1
                new_items[f"k:{doc_id}"] = {"fingerprint": fp, "chunks": n}
                _emit(emit, "item_done", index=didx, total=total, name=doc_id,
                      chunks=n, skipped=False)
            else:
                # Left out of the manifest so it recompiles rather than reading clean.
                _emit(emit, "warn", name=doc_id,
                      message="not fully embedded — will recompile")

    # --- Memories ---
    for mem_id, mem, fp in memories:
        if stop_event is not None and stop_event.is_set():
            break
        idx += 1
        mkey = f"m:{mem_id}"
        title = mem.get("title") or mem_id
        _emit(emit, "item_start", index=idx, total=total, name=title)
        unchanged = (same_sig and not force
                     and (prev_items.get(mkey) or {}).get("fingerprint") == fp
                     and rag.stored_hashes(PERSONA_MEMORY, pid, mem_id, embed_model))
        if unchanged:
            n = (prev_items.get(mkey) or {}).get("chunks", 0)
            skipped += 1
            _emit(emit, "item_done", index=idx, total=total, name=title, chunks=n, skipped=True)
        else:
            try:
                # save_memory bumps the memory's ``updated`` timestamp, so record the
                # POST-save fingerprint or the very next status check reads as stale.
                saved = msvc.save_memory(pid, mem, embed_fn, embed_model)
                fp = f"{saved.get('updated', '')}"
                n = len(rag.stored_hashes(PERSONA_MEMORY, pid, mem_id, embed_model)) or 1
                embedded += 1
                _emit(emit, "item_done", index=idx, total=total, name=title, chunks=n, skipped=False)
            except Exception as e:
                # Left out of the manifest, like a knowledge doc that didn't embed:
                # recording it would make persona_status read "compiled" while this
                # memory has no vectors.
                _emit(emit, "warn", name=title, message=str(e))
                continue
        new_items[mkey] = {"fingerprint": fp, "chunks": n}

    # Prune deleted knowledge/memory rows from the vector store.
    rag.prune_items(PERSONA_KNOWLEDGE, pid, [d for d, _p, _f in sources])
    rag.prune_items(PERSONA_MEMORY, pid, [m for m, _x, _f in memories])

    failed = int(meta_out.get("failed") or 0)
    stopped = bool(meta_out.get("stopped")) or bool(
        stop_event is not None and stop_event.is_set())
    manifest = {
        "signature": sig,
        "items": new_items,
        "chunks": rag.count_chunks(PERSONA_KNOWLEDGE, pid) + rag.count_chunks(PERSONA_MEMORY, pid),
        "compiled_at": datetime.utcnow().isoformat(),
    }
    _put_manifest(key, manifest)

    # Record the embedding model used so the mismatch banner clears (persona.xml).
    try:
        persona.setdefault("stores", {})["embedding_model_used"] = embed_model
        PersonaService().save(persona)
    except Exception:
        pass

    summary = {"state": "partial" if (failed or stopped) else "compiled",
               "items": len(new_items), "embedded": embedded,
               "skipped": skipped, "chunks": manifest["chunks"],
               "compiled_at": manifest["compiled_at"],
               "embedding_model": embed_model, "chunker": sig["chunker"],
               "failed": failed, "stopped": stopped,
               "cached": int(meta_out.get("cached") or 0),
               "lanes": meta_out.get("lanes") or []}
    _emit(emit, "compiled", **summary)
    return summary
