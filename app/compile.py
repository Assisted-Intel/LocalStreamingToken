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
from datetime import datetime
from pathlib import Path

from . import core, rag, persona_store, ingest
from .persona import PersonaService

LIBRARY = "library"
PERSONA_KNOWLEDGE = persona_store.ST_KNOWLEDGE   # "persona_knowledge"
PERSONA_MEMORY = persona_store.ST_MEMORY         # "persona_memory"

_MANIFEST_LOCK = threading.RLock()


# --------------------------- manifest persistence ---------------------------

def _load_all() -> dict:
    # core.load_json transparently decrypts (compiled.json is encrypted at rest).
    return core.load_json(Path(core.COMPILED_FILE), {}) or {}


def _save_all(data: dict) -> None:
    core.save_json(Path(core.COMPILED_FILE), data)


def _get_manifest(key: str) -> dict:
    with _MANIFEST_LOCK:
        return _load_all().get(key) or {}


def _put_manifest(key: str, manifest: dict) -> None:
    with _MANIFEST_LOCK:
        data = _load_all()
        data[key] = manifest
        _save_all(data)


def _drop_manifest(key: str) -> None:
    with _MANIFEST_LOCK:
        data = _load_all()
        if key in data:
            del data[key]
            _save_all(data)


def forget_library(lib_id: str) -> None:
    """Drop a library's compile manifest (called on library delete)."""
    _drop_manifest(f"{LIBRARY}:{lib_id}")


def forget_persona(persona_id: str) -> None:
    """Drop a persona's compile manifest (called on persona delete)."""
    _drop_manifest(f"persona:{persona_id}")


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


# --------------------------- status (cheap, no embedding) ---------------------------

def _state_from(manifest: dict, sig: dict, current_items: dict, stored_chunks: int) -> dict:
    """Decide compiled / stale / none from a manifest vs the current items + signature.
    ``current_items`` maps item_id -> fingerprint. Never embeds or parses heavy files
    beyond the cheap fingerprints the caller already gathered."""
    if not manifest or not manifest.get("items"):
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
    id the same way ``rag.index_libraries`` does so retrieval scopes line up."""
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
    return _state_from(_get_manifest(f"{LIBRARY}:{lib_id}"), sig, current, stored_chunks)


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
    return _state_from(_get_manifest(f"persona:{pid}"), sig, current, stored_chunks)


# --------------------------- compile (embeds) ---------------------------

def compile_library(lib: dict, embed_fn, embed_model: str, contextualize=None,
                    force: bool = False, emit=None, batch_size: int = 64,
                    max_workers: int = 1) -> dict:
    """Chunk + embed every item of ``lib`` into the store, skipping unchanged items
    (unless ``force``), prune removed items, and write the manifest. Returns a summary.

    Changed items are embedded in ONE bulk ``rag.upsert_items`` call (chunks pooled
    across items, embedded in batches of ``batch_size`` with up to ``max_workers``
    concurrent requests) rather than one blocking HTTP round-trip per item."""
    lib_id = lib.get("id")
    key = f"{LIBRARY}:{lib_id}"
    sig = signature(embed_model)
    prev = _get_manifest(key)
    prev_items = prev.get("items") or {}
    same_sig = prev.get("signature") == sig
    items = _library_items(lib)
    total = len(items)
    _emit(emit, "begin", total=total, kind="library", name=lib.get("name", ""))

    new_items = {}
    embedded = skipped = 0
    changed = []          # [(item_id, content, meta)] to embed in bulk
    changed_pos = {}      # item_id -> (index, label) for progress events
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
            changed_pos[item_id] = (i + 1, label)
            new_items[item_id] = {"fingerprint": fp, "chunks": 0}

    if changed:
        counts = rag.upsert_items(LIBRARY, lib_id, changed, embed_fn, embed_model,
                                  contextualize=contextualize, batch_size=batch_size,
                                  max_workers=max_workers)
        for item_id, _c, _m in changed:
            n = counts.get(item_id, 0)
            embedded += 1
            new_items[item_id]["chunks"] = n
            idx, label = changed_pos[item_id]
            _emit(emit, "item_done", index=idx, total=total, name=label, chunks=n, skipped=False)

    rag.prune_items(LIBRARY, lib_id, list(new_items.keys()))
    manifest = {
        "signature": sig,
        "items": new_items,
        "chunks": rag.count_chunks(LIBRARY, lib_id),
        "compiled_at": datetime.utcnow().isoformat(),
    }
    _put_manifest(key, manifest)
    summary = {"state": "compiled", "items": len(new_items), "embedded": embedded,
               "skipped": skipped, "chunks": manifest["chunks"],
               "compiled_at": manifest["compiled_at"],
               "embedding_model": embed_model, "chunker": sig["chunker"]}
    _emit(emit, "compiled", **summary)
    return summary


def compile_persona(persona: dict, embed_fn, embed_model: str, contextualize=None,
                    force: bool = False, emit=None, batch_size: int = 64,
                    max_workers: int = 1) -> dict:
    """Chunk + embed a persona's knowledge sources and memories, skip unchanged, prune
    removed, record ``stores.embedding_model_used`` on the persona, and write the
    manifest. Returns a summary.

    Changed knowledge documents are embedded in one bulk ``rag.upsert_items`` call
    (chunks pooled + batched/concurrent per ``batch_size``/``max_workers``). Memories
    are few and stay per-item."""
    pid = persona.get("id")
    key = f"persona:{pid}"
    sig = signature(embed_model)
    prev = _get_manifest(key)
    prev_items = prev.get("items") or {}
    same_sig = prev.get("signature") == sig

    msvc = persona_store.MemoryService()
    sources = _persona_source_items(pid)
    memories = _persona_memory_items(pid)
    total = len(sources) + len(memories)
    _emit(emit, "begin", total=total, kind="persona", name=persona.get("profile", {}).get("name", ""))

    new_items = {}
    embedded = skipped = 0
    idx = 0

    # --- Knowledge documents (changed ones embedded in bulk) ---
    changed = []          # [(doc_id, text, meta)]
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
            try:
                # File is already in the persona's sources/ dir (that's what we globbed),
                # so just extract text — no copy — and defer embedding to the bulk call.
                text = ingest.extract_text(Path(path))
                changed.append((doc_id, text, {"name": doc_id, "tags": []}))
                changed_pos[doc_id] = (idx, fp)
                new_items[mkey] = {"fingerprint": fp, "chunks": 0}
            except Exception as e:
                _emit(emit, "warn", name=doc_id, message=str(e))
                new_items[mkey] = {"fingerprint": fp, "chunks": 0}

    if changed:
        counts = rag.upsert_items(PERSONA_KNOWLEDGE, pid, changed, embed_fn, embed_model,
                                  contextualize=contextualize, batch_size=batch_size,
                                  max_workers=max_workers)
        for doc_id, _t, _m in changed:
            n = counts.get(doc_id, 0)
            embedded += 1
            new_items[f"k:{doc_id}"]["chunks"] = n
            didx, _fp = changed_pos[doc_id]
            _emit(emit, "item_done", index=didx, total=total, name=doc_id, chunks=n, skipped=False)

    # --- Memories ---
    for mem_id, mem, fp in memories:
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
                n = 0
                _emit(emit, "warn", name=title, message=str(e))
        new_items[mkey] = {"fingerprint": fp, "chunks": n}

    # Prune deleted knowledge/memory rows from the vector store.
    rag.prune_items(PERSONA_KNOWLEDGE, pid, [d for d, _p, _f in sources])
    rag.prune_items(PERSONA_MEMORY, pid, [m for m, _x, _f in memories])

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

    summary = {"state": "compiled", "items": len(new_items), "embedded": embedded,
               "skipped": skipped, "chunks": manifest["chunks"],
               "compiled_at": manifest["compiled_at"],
               "embedding_model": embed_model, "chunker": sig["chunker"]}
    _emit(emit, "compiled", **summary)
    return summary
