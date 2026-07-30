#!/usr/bin/env python3
"""
Local Streaming Token — persona knowledge & memory stores.

Both corpora reuse the shared DuckDB retrieval layer in ``rag.py`` (hybrid vector +
keyword + RRF, multi-query, contextual chunking) — no retrieval logic is duplicated
here. They are namespaced in the shared ``rag.duckdb`` by ``source_id = persona_id`` and
``source_type``:

    persona_knowledge  — the persona's domain documents (PDF/EPUB/DOCX/TXT via app.ingest)
    persona_memory     — life-like memories, grown from chat ("Save as memory")

Portable source of truth stays in the persona folder (``sources/`` originals,
``memories/entries/*.json``); DuckDB rows are a rebuildable cache and are never exported.
"""

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from . import core, ingest, rag
from .persona import persona_path

ST_KNOWLEDGE = "persona_knowledge"
ST_MEMORY = "persona_memory"


# --------------------------- Knowledge ---------------------------

class KnowledgeService:
    """Ingests documents into a persona's knowledge base and searches it."""

    def sources_dir(self, persona_id: str) -> Path:
        """Where original documents are kept so they can be re-exported and, on another
        machine, re-ingested with that machine's embedding model. Created if missing."""
        d = persona_path(persona_id) / "sources"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ingest_file(self, persona_id: str, file_path, embed_fn, model: str,
                    contextualize=None, tags=None) -> dict:
        """Copy the original into the persona's ``sources/``, extract its text, and
        chunk+embed it under source_type=persona_knowledge. Returns
        {doc_id, chunks, name}. Raises ingest.IngestError on unreadable files."""
        src = Path(file_path)
        text = ingest.extract_text(src)                 # may raise IngestError
        dest = self.sources_dir(persona_id) / src.name
        try:
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        except Exception:
            pass
        doc_id = src.name
        meta = {"name": src.name, "tags": tags or []}
        n = rag.upsert_item(ST_KNOWLEDGE, persona_id, doc_id, text, embed_fn, model,
                            contextualize=contextualize, meta=meta)
        return {"doc_id": doc_id, "chunks": n, "name": src.name}

    def add_text(self, persona_id: str, name: str, text: str, embed_fn, model: str,
                 contextualize=None) -> dict:
        """Add a write-in knowledge document. The text is also saved into ``sources/``
        as a .txt/.md file so it travels in an export bundle and re-ingests on import."""
        base = (name or f"note-{uuid.uuid4().hex[:8]}").strip()
        fname = base if base.lower().endswith((".txt", ".md")) else base + ".txt"
        try:
            core.write_text(self.sources_dir(persona_id) / fname, text)
        except Exception:
            pass
        doc_id = fname
        n = rag.upsert_item(ST_KNOWLEDGE, persona_id, doc_id, text, embed_fn, model,
                            contextualize=contextualize, meta={"name": fname})
        return {"doc_id": doc_id, "chunks": n, "name": fname}

    def list_documents(self, persona_id: str) -> list:
        """Summaries of this persona's knowledge documents, from the store's index."""
        return rag.list_items(ST_KNOWLEDGE, persona_id)

    def remove_document(self, persona_id: str, doc_id: str) -> None:
        """Drop a document's chunks and, best-effort, its original file. The vector rows
        are the part that matters — a leftover source file is harmless, so failing to
        delete it is not raised."""
        rag.delete_item(ST_KNOWLEDGE, persona_id, doc_id)
        # Best-effort remove the original source file too.
        f = self.sources_dir(persona_id) / doc_id
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass

    def search(self, persona_id: str, query_vecs, model: str, top_k: int,
               mode: str = "hybrid", queries=None) -> list:
        """Retrieve top-k knowledge chunks for this persona only. ``queries`` carries the
        raw query strings for the BM25 half of hybrid mode; ``query_vecs`` the embedded
        form for the vector half."""
        return rag.retrieve(ST_KNOWLEDGE, [persona_id], query_vecs, model, top_k,
                            mode=mode, queries=queries)


# --------------------------- Memories ---------------------------

def _entries_dir(persona_id: str) -> Path:
    """One JSON file per memory — the portable source of truth. Created if missing."""
    d = persona_path(persona_id) / "memories" / "entries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _memory_text(mem: dict) -> str:
    """Text indexed for a memory: title + description together (so both feed BM25)."""
    return f"{(mem.get('title') or '').strip()}\n{(mem.get('description') or '').strip()}".strip()


class MemoryService:
    """Persona memories: JSON on disk (portable) + embedded in the shared store.
    Retrieval blends the base retrieval score with the memory's emotional weight so
    high-weight memories surface more readily."""

    def save_memory(self, persona_id: str, mem: dict, embed_fn, model: str) -> dict:
        """Create or update one memory, writing both representations: the JSON entry on
        disk and the embedded row in the shared store. Fills in an id and timestamps,
        and clamps ``emotional_weight`` to 1-10 (defaulting to 5) so a bad value from
        the UI or an imported bundle can't distort retrieval ranking."""
        mem = dict(mem)
        mem_id = mem.get("id") or uuid.uuid4().hex[:12]
        mem["id"] = mem_id
        mem.setdefault("created", datetime.utcnow().isoformat())
        mem["updated"] = datetime.utcnow().isoformat()
        try:
            mem["emotional_weight"] = max(1, min(10, int(mem.get("emotional_weight", 5))))
        except Exception:
            mem["emotional_weight"] = 5
        mem.setdefault("narrative_time", "")
        mem.setdefault("tags", [])
        # 1) portable source of truth (encrypted at rest)
        core.write_text(_entries_dir(persona_id) / f"{mem_id}.json",
                        json.dumps(mem, indent=2))
        # 2) embed into the shared store (weight travels in meta for retrieval blend)
        rag.upsert_item(ST_MEMORY, persona_id, mem_id, _memory_text(mem), embed_fn, model,
                        contextualize=None,
                        meta={"weight": mem["emotional_weight"],
                              "narrative_time": mem.get("narrative_time", ""),
                              "created": mem["created"], "title": mem.get("title", "")})
        return mem

    def list_memories(self, persona_id: str) -> list:
        """All memories, newest first. Unreadable entries are skipped rather than
        failing the whole listing."""
        out = []
        d = _entries_dir(persona_id)
        for f in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(core.read_text(f)))
            except Exception:
                continue
        out.sort(key=lambda m: m.get("created", ""), reverse=True)
        return out

    def get_memory(self, persona_id: str, mem_id: str) -> dict:
        """One memory by id, or None if there's no entry for it."""
        f = _entries_dir(persona_id) / f"{mem_id}.json"
        return json.loads(core.read_text(f)) if f.is_file() else None

    def delete_memory(self, persona_id: str, mem_id: str) -> None:
        """Remove a memory from both the store and disk."""
        rag.delete_item(ST_MEMORY, persona_id, mem_id)
        f = _entries_dir(persona_id) / f"{mem_id}.json"
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass

    def search(self, persona_id: str, query_vecs, model: str, top_k: int,
               mode: str = "hybrid", queries=None, weight_influence: float = 0.35) -> list:
        """Retrieve memories, then blend the retrieval score with emotional weight:
        blended = base * (1 + influence*(weight-5)/5). High-weight memories rise, very
        low-weight memories are gently demoted. Over-fetch before blending so weighting
        can reorder the final top-k."""
        raw = rag.retrieve(ST_MEMORY, [persona_id], query_vecs, model, max(top_k * 3, top_k),
                           mode=mode, queries=queries)
        for r in raw:
            weight = 5
            try:
                weight = int((json.loads(r.get("meta") or "{}")).get("weight", 5))
            except Exception:
                pass
            base = r.get("score", 0.0)
            r["base_score"] = base
            r["weight"] = weight
            r["score"] = base * (1 + weight_influence * (weight - 5) / 5.0)
        raw.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return raw[:int(top_k)]
