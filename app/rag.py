#!/usr/bin/env python3
"""
Local Streaming Token — RAG vector store.

A DuckDB-backed store for Retrieval Augmented Generation. Library documents are
chunked, embedded (via a caller-supplied ``embed_fn``) and persisted in
``data/rag.duckdb``; at query time the top-k chunks most similar to the user's
question are retrieved and injected into the prompt in place of the full text.

Design notes
- Persistent corpus = the user's selected **Libraries**. Re-embedding is skipped
  when an item's chunk set is unchanged (content-hash compared), so indexing is
  idempotent and cheap to call on every send.
- Transient corpus = the inline ``<Data>`` block staged for one message. These are
  embedded on the fly and scored in memory — never written to the store.
- Similarity is cosine. DuckDB's built-in ``list_cosine_similarity`` is used when
  available, with a pure-Python fallback so the feature works on any DuckDB build
  and needs no VSS extension download (an HNSW/VSS index can be added later purely
  as a scale-up).

The DuckDB connection is cached process-wide with a reentrant lock, mirroring
``app/database/staging.py``.
"""

import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import core

# Process-wide (connection, lock). Lazy — only opened when RAG is first used.
_CONN = None
_REG_LOCK = threading.Lock()


def reset_connection():
    """Close and drop the cached DuckDB connection so the next use reopens against
    the current ``core.RAG_DB_FILE``. Called when the active data profile changes —
    otherwise RAG would keep reading/writing the previous profile's vector store."""
    global _CONN
    with _REG_LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:
                pass
            _CONN = None


# --------------------------- Chunking ---------------------------

def chunk_text(text: str, size_words: int = 200, overlap_words: int = 40) -> list:
    """Split text into overlapping word windows. Returns [] for empty text.

    This is the legacy/fallback chunker: it needs no dependencies and is used when
    chonkie is unavailable. The primary chunker is ``chunk_text_semantic``."""
    words = (text or "").split()
    if not words:
        return []
    size = max(1, int(size_words))
    overlap = max(0, min(int(overlap_words), size - 1))
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# --- Semantic (token-accurate) chunking, backed by chonkie ------------------
#
# ``chunk_text_semantic`` is the ONE chunker used by every persistent indexer
# (index_libraries + upsert_item), so compiled data stays internally consistent.
# It uses chonkie's RecursiveChunker (paragraphs → sentences → tokens) when the
# dependency is importable and works offline, and transparently falls back to the
# word-window ``chunk_text`` otherwise. ``active_chunker_id()`` reports which one is
# live so the compile signature invalidates chunks when the chunker changes.

CHUNKER_RECURSIVE = "chonkie-recursive-v1"   # algorithm identity when chonkie is live
CHUNKER_FALLBACK = "wordwindow-v1"           # algorithm identity for the fallback

_CHUNK_CFG = {"chunker": "recursive", "size": 512, "overlap": 64}
_CHUNKER = None          # cached chonkie chunker instance
_CHUNKER_SIG = None      # (chunker, size) the cached instance was built for
_CHONKIE_OK = None       # None=untried, True/False after the first smoke test


def set_chunk_config(chunker: str = None, size=None, overlap=None):
    """Update chunking parameters (called from settings). Drops the cached chunker so
    the next chunk call rebuilds it. ``None`` values are left unchanged."""
    global _CHUNKER, _CHUNKER_SIG
    if chunker:
        _CHUNK_CFG["chunker"] = str(chunker)
    if size:
        try:
            _CHUNK_CFG["size"] = max(64, int(size))
        except Exception:
            pass
    if overlap is not None:
        try:
            _CHUNK_CFG["overlap"] = max(0, int(overlap))
        except Exception:
            pass
    _CHUNKER = None
    _CHUNKER_SIG = None


def chunk_config() -> dict:
    return dict(_CHUNK_CFG)


def _get_chunker():
    """Build/cache the chonkie chunker for the current config, smoke-testing it once so
    a broken/absent dependency degrades cleanly. Returns None to signal the fallback."""
    global _CHUNKER, _CHUNKER_SIG, _CHONKIE_OK
    if _CHONKIE_OK is False:
        return None
    sig = (_CHUNK_CFG["chunker"], _CHUNK_CFG["size"])
    if _CHUNKER is not None and _CHUNKER_SIG == sig:
        return _CHUNKER
    try:
        from chonkie import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=int(_CHUNK_CFG["size"]))
        chunker.chunk("Smoke test. Second sentence.")   # verify it runs offline
        _CHUNKER = chunker
        _CHUNKER_SIG = sig
        _CHONKIE_OK = True
        return _CHUNKER
    except Exception:
        _CHONKIE_OK = False
        return None


def active_chunker_id() -> str:
    """CHUNKER_ID for the live chunker — part of the compile signature."""
    return CHUNKER_RECURSIVE if _get_chunker() is not None else CHUNKER_FALLBACK


def chunk_text_semantic(text: str) -> list:
    """Primary chunker: token-accurate, structure-aware chunks via chonkie, with a
    word-window fallback. Returns [] for empty text."""
    text = (text or "").strip()
    if not text:
        return []
    chunker = _get_chunker()
    if chunker is None:
        return chunk_text(text, size_words=_CHUNK_CFG["size"], overlap_words=_CHUNK_CFG["overlap"])
    try:
        pieces = chunker.chunk(text)
        out = [((getattr(p, "text", None) or str(p)) or "").strip() for p in pieces]
        out = [c for c in out if c]
        return out or chunk_text(text)
    except Exception:
        return chunk_text(text)


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def item_hashes(content: str) -> list:
    """Ordered content-hashes of an item's chunks under the live chunker. Used by the
    compile step to detect staleness (item added/edited) without embedding."""
    return [_hash(c) for c in chunk_text_semantic(content)]


def stored_hashes(source_type: str, source_id: str, item_id: str, model: str) -> list:
    """Ordered content-hashes currently stored for an item under ``model`` (empty if
    none). Compared against ``item_hashes`` to decide whether re-embedding is needed."""
    conn = _conn()
    with _REG_LOCK:
        rows = conn.execute(
            "SELECT content_hash FROM rag_chunks WHERE source_type = ? AND source_id = ? "
            "AND item_id = ? AND model = ? ORDER BY chunk_index",
            [source_type, source_id, item_id, model]).fetchall()
    return [r[0] for r in rows]


def prune_items(source_type: str, source_id: str, keep_item_ids: list) -> int:
    """Delete stored chunks for items no longer present. Returns rows deleted."""
    conn = _conn()
    keep = [i for i in (keep_item_ids or []) if i]
    with _REG_LOCK:
        before = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ?",
            [source_type, source_id]).fetchone()[0]
        if keep:
            placeholders = ", ".join("?" for _ in keep)
            conn.execute(
                f"DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ? "
                f"AND item_id NOT IN ({placeholders})",
                [source_type, source_id, *keep])
        else:
            conn.execute(
                "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ?",
                [source_type, source_id])
        after = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ?",
            [source_type, source_id]).fetchone()[0]
    return int(before - after)


def count_chunks(source_type: str, source_id: str) -> int:
    """Total stored chunks for a scope (across all its items)."""
    conn = _conn()
    with _REG_LOCK:
        return int(conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ?",
            [source_type, source_id]).fetchone()[0])


# --------------------------- Connection / schema ---------------------------

def _conn():
    """Return the cached DuckDB connection, creating the schema on first use."""
    global _CONN
    with _REG_LOCK:
        if _CONN is None:
            conn = core.duckdb_connect(core.RAG_DB_FILE)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id VARCHAR,
                    source_type VARCHAR,
                    source_id VARCHAR,
                    item_id VARCHAR,
                    chunk_index INTEGER,
                    content VARCHAR,
                    content_hash VARCHAR,
                    embedding FLOAT[],
                    model VARCHAR,
                    updated VARCHAR,
                    context VARCHAR,
                    meta VARCHAR
                )
                """
            )
            _migrate_schema(conn)
            _CONN = conn
        return _CONN


def _migrate_schema(conn):
    """In-place upgrades for stores created by older versions. Adds any column that
    the current schema expects but an existing ``rag.duckdb`` predates."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info('rag_chunks')").fetchall()}
    if "context" not in cols:
        # Situating context from contextual chunking (Phase 2). Empty for old rows.
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN context VARCHAR")
    if "meta" not in cols:
        # Per-chunk JSON metadata (Phase 5): memory weight/narrative_time, etc.
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN meta VARCHAR")


# --------------------------- Similarity ---------------------------

def _cosine(a, b) -> float:
    """Standard cosine similarity between two equal-length float lists."""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --------------------------- Keyword (BM25) scoring ---------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def _bm25_search(rows: list, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list:
    """Rank ``rows`` (dicts with 'content' and optional 'context') against ``query``
    using Okapi BM25 over the in-scope candidate set. Pure Python — needs no DuckDB
    extension, so keyword/hybrid retrieval works offline and with no embedding model.
    Returns rows (copied, with a 'score') sorted by descending BM25 score; only rows
    with score > 0 are returned."""
    q_terms = [t for t in set(_tokenize(query)) if t]
    if not rows or not q_terms:
        return []
    docs = [_tokenize((r.get("context") or "") + " " + (r.get("content") or "")) for r in rows]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    scored = []
    for r, d in zip(rows, docs):
        if not d:
            continue
        dl = len(d)
        tf = {}
        for term in d:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            ni = df.get(term, 0)
            idf = math.log(1 + (n - ni + 0.5) / (ni + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            rr = dict(r)
            rr["score"] = score
            scored.append(rr)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:int(top_k)]


def _result_key(r: dict):
    """Stable identity for a retrieved chunk, used to dedupe across queries/modes."""
    return r.get("id") or (r.get("source_id"), r.get("item_id"), r.get("content"))


def _rrf_merge(result_lists: list, top_k: int, k: int = 60) -> list:
    """Reciprocal-rank-fusion merge of several ranked result lists (each already
    sorted best-first). The fused score replaces the per-list score; ties broken by
    insertion order. Used to combine multiple query variants and to combine the
    vector and keyword result sets in hybrid mode."""
    fused = {}
    keep = {}
    for lst in result_lists:
        for rank, r in enumerate(lst):
            kk = _result_key(r)
            fused[kk] = fused.get(kk, 0.0) + 1.0 / (k + rank + 1)
            if kk not in keep:
                keep[kk] = r
    merged = []
    for kk, s in fused.items():
        rr = dict(keep[kk])
        rr["score"] = s
        merged.append(rr)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:int(top_k)]


def _as_vec_list(query_vecs) -> list:
    """Normalize the query-vector argument to a list of vectors. Accepts a single
    vector (list of floats), a list of vectors, or None/empty."""
    if not query_vecs:
        return []
    first = query_vecs[0]
    if isinstance(first, (int, float)):   # a single flat vector
        return [query_vecs]
    return [v for v in query_vecs if v]


def _as_query_list(queries) -> list:
    """Normalize the keyword-query argument to a list of non-empty strings."""
    if not queries:
        return []
    if isinstance(queries, str):
        queries = [queries]
    return [q for q in queries if q and q.strip()]


# --------------------------- Library indexing ---------------------------

def _item_label(it: dict) -> str:
    return (it.get("label") or it.get("filename")
            or ("Write-in" if it.get("type") == "write" else "File"))


def _embed_chunks(embed_fn, chunks: list) -> list:
    """Embed ``chunks`` with ``embed_fn`` (list[str] -> list[list[float]]). When no
    embedder is supplied (keyword-only mode) or embedding fails (e.g. no embed server
    reachable), return empty vectors so the chunks are still stored and available to
    keyword/BM25 retrieval — vector retrieval simply won't match them until re-indexed."""
    if not chunks:
        return []
    if embed_fn is None:
        return [[] for _ in chunks]
    try:
        return embed_fn(chunks)
    except Exception:
        return [[] for _ in chunks]


def _embed_batched(embed_fn, inputs: list, batch_size: int = 64, max_workers: int = 1) -> list:
    """Embed ``inputs`` (list[str]) -> one vector per input **in the same order**.

    Splits the work into batches of ``batch_size`` (each batch is one ``embed_fn``
    call, so a huge document can't build a single 5000-input request) and, when
    ``max_workers > 1``, runs up to that many batches concurrently. This is the hot
    path for compile: it is **lock-free** (pure network/CPU) and MUST be called
    outside ``_REG_LOCK`` so the network wait never blocks the shared DuckDB
    connection. A failed batch degrades to empty vectors for its inputs (mirroring
    ``_embed_chunks``) so the chunks still persist for keyword/BM25 retrieval.

    Concurrency is a backend-sensitive knob: a win on any GPU (CUDA/Metal/ROCm/
    Intel), neutral-to-harmful on CPU-only. Callers pass ``max_workers=1`` there.
    """
    if not inputs:
        return []
    if embed_fn is None:
        return [[] for _ in inputs]
    try:
        batch_size = max(1, int(batch_size))
    except Exception:
        batch_size = 64
    try:
        max_workers = max(1, int(max_workers))
    except Exception:
        max_workers = 1
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]

    def _run(batch):
        try:
            vecs = embed_fn(batch)
            if vecs and len(vecs) == len(batch):
                return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass
        return [[] for _ in batch]

    if max_workers > 1 and len(batches) > 1:
        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as ex:
            futures = {ex.submit(_run, b): i for i, b in enumerate(batches)}
            for fut, i in futures.items():
                results[i] = fut.result()
    else:
        results = [_run(b) for b in batches]

    out = []
    for r in results:
        out.extend(r)
    return out


def _contextualize_all(prepared: list, contextualize, max_workers: int = 1) -> None:
    """Fill each prepared item's ``contexts`` list with LLM-written situating sentences,
    running the per-chunk ``contextualize(chunk, doc_summary)`` calls concurrently (up to
    ``max_workers``). Mutates ``prepared`` in place; a failed call degrades to '' so the
    raw chunk is still embedded. Lock-free — call outside ``_REG_LOCK``."""
    tasks = []  # (item_index, chunk_index, chunk_text, doc_summary)
    for pi, p in enumerate(prepared):
        if not p["chunks"]:
            continue
        summary = _doc_summary(p["content"])
        for ci, ch in enumerate(p["chunks"]):
            tasks.append((pi, ci, ch, summary))
    if not tasks:
        return

    def _run(t):
        pi, ci, ch, summary = t
        try:
            return pi, ci, (contextualize(ch, summary) or "").strip()
        except Exception:
            return pi, ci, ""

    workers = max(1, int(max_workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            for pi, ci, ctx in ex.map(_run, tasks):
                prepared[pi]["contexts"][ci] = ctx
    else:
        for t in tasks:
            pi, ci, ctx = _run(t)
            prepared[pi]["contexts"][ci] = ctx


def _doc_summary(content: str, max_words: int = 120) -> str:
    """A cheap 'what this document is about' blurb (its opening ~120 words) used to
    situate each chunk during contextual chunking. Avoids a second LLM summarization
    call per document."""
    words = (content or "").split()
    return " ".join(words[:max_words])


def index_libraries(libraries: list, lib_ids: list, embed_fn, model: str,
                    contextualize=None) -> dict:
    """Ensure the selected libraries' items are chunked + embedded in the store.

    Only items whose chunk set changed are re-embedded; items removed from a
    library are pruned. ``embed_fn(list[str]) -> list[list[float]]`` supplies the
    vectors (pass ``None`` for keyword-only indexing).

    ``contextualize(chunk, doc_summary) -> str`` (optional): when supplied, each chunk
    is passed through it to obtain 1-2 sentences of situating context, which is stored
    in the ``context`` column and prepended to the chunk for embedding + keyword
    indexing (markedly improves retrieval of out-of-context chunks). One LLM call per
    chunk, so it's opt-in; failures fall back to indexing the raw chunk.

    Returns {"embedded": n_chunks_embedded, "items": n_items_indexed}.
    """
    ids = [i for i in (lib_ids or []) if i]
    if not ids:
        return {"embedded": 0, "items": 0}
    libs = [l for l in (libraries or []) if l.get("id") in ids]
    conn = _conn()
    embedded = 0
    items = 0
    now = datetime.utcnow().isoformat()
    with _REG_LOCK:
        for lib in libs:
            lib_id = lib.get("id")
            present_item_ids = []
            for idx, it in enumerate(lib.get("items", [])):
                content = (it.get("content") or "").strip()
                if not content:
                    continue
                # Items may not carry their own id; fall back to a stable index key.
                item_id = it.get("id") or f"{lib_id}:{idx}"
                present_item_ids.append(item_id)
                items += 1
                chunks = chunk_text_semantic(content)
                hashes = [_hash(c) for c in chunks]
                existing = [r[0] for r in conn.execute(
                    "SELECT content_hash FROM rag_chunks WHERE source_id = ? AND item_id = ? "
                    "AND model = ? ORDER BY chunk_index",
                    [lib_id, item_id, model],
                ).fetchall()]
                if existing == hashes:
                    continue  # unchanged — skip re-embedding
                # Changed (or new): replace this item's chunks wholesale.
                conn.execute(
                    "DELETE FROM rag_chunks WHERE source_id = ? AND item_id = ?",
                    [lib_id, item_id],
                )
                if not chunks:
                    continue
                # Contextual chunking (optional): 1-2 sentences situating each chunk.
                if contextualize is not None:
                    summary = _doc_summary(content)
                    contexts = []
                    for ch in chunks:
                        try:
                            contexts.append((contextualize(ch, summary) or "").strip())
                        except Exception:
                            contexts.append("")  # degrade to raw chunk on any failure
                else:
                    contexts = ["" for _ in chunks]
                # Embed (and later BM25-index) context + chunk together when present.
                embed_inputs = [((contexts[i] + "\n" + ch).strip() if contexts[i] else ch)
                                for i, ch in enumerate(chunks)]
                vectors = _embed_chunks(embed_fn, embed_inputs)
                for ci, (chunk, vec, h, ctx) in enumerate(zip(chunks, vectors, hashes, contexts)):
                    conn.execute(
                        "INSERT INTO rag_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [f"{item_id}:{ci}", "library", lib_id, item_id, ci,
                         chunk, h, [float(x) for x in vec], model, now, ctx, ""],
                    )
                    embedded += 1
            # Prune chunks for items that no longer exist in this library.
            if present_item_ids:
                placeholders = ", ".join("?" for _ in present_item_ids)
                conn.execute(
                    f"DELETE FROM rag_chunks WHERE source_id = ? AND item_id NOT IN ({placeholders})",
                    [lib_id, *present_item_ids],
                )
            else:
                conn.execute("DELETE FROM rag_chunks WHERE source_id = ?", [lib_id])
    return {"embedded": embedded, "items": items}


# --------------------------- Retrieval ---------------------------
#
# All retrieval honors a ``mode``:
#   vector  — embedding cosine similarity only (needs query vectors)
#   keyword — BM25 over chunk text only (needs no embeddings; works offline)
#   hybrid  — both, merged with reciprocal rank fusion (default)
# and accepts MULTIPLE query variants (from Prompt Reword, Phase 3). Each variant is
# retrieved independently and the ranked lists are RRF-merged, so one poor rewording
# can't sink retrieval.


def _vector_rows(conn, source_type: str, ids: list, model: str, qv: list, top_k: int) -> list:
    """One vector query over chunks of ``source_type`` in ``ids``. DuckDB
    ``list_cosine_similarity`` with a pure-Python fallback."""
    placeholders = ", ".join("?" for _ in ids)
    qv = [float(x) for x in qv]
    try:
        rows = conn.execute(
            f"""
            SELECT content, source_id, item_id, id, meta,
                   list_cosine_similarity(embedding, CAST(? AS FLOAT[])) AS score
            FROM rag_chunks
            WHERE source_type = ? AND model = ? AND source_id IN ({placeholders})
              AND len(embedding) > 0
            ORDER BY score DESC NULLS LAST
            LIMIT ?
            """,
            [qv, source_type, model, *ids, int(top_k)],
        ).fetchall()
        return [{"content": r[0], "source_id": r[1], "item_id": r[2], "id": r[3],
                 "meta": r[4], "score": float(r[5]) if r[5] is not None else -1.0} for r in rows]
    except Exception:
        rows = conn.execute(
            f"SELECT content, source_id, item_id, id, meta, embedding FROM rag_chunks "
            f"WHERE source_type = ? AND model = ? AND source_id IN ({placeholders})",
            [source_type, model, *ids],
        ).fetchall()
    scored = [{"content": r[0], "source_id": r[1], "item_id": r[2], "id": r[3],
               "meta": r[4], "score": _cosine(qv, r[5])} for r in rows]
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:int(top_k)]


def _keyword_candidates(conn, source_type: str, ids: list) -> list:
    """All in-scope chunks (text only, no model filter) for BM25 scoring."""
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT content, source_id, item_id, id, context, meta FROM rag_chunks "
        f"WHERE source_type = ? AND source_id IN ({placeholders})",
        [source_type, *ids],
    ).fetchall()
    return [{"content": r[0], "source_id": r[1], "item_id": r[2], "id": r[3],
             "context": r[4], "meta": r[5]} for r in rows]


def retrieve(source_type: str, ids: list, query_vecs, model: str, top_k: int,
             mode: str = "hybrid", queries=None) -> list:
    """Generic top-k retrieval over a scope (source_type + source ids) under ``mode``.
    Shared by libraries and persona knowledge/memory — the ONE retrieval path.

    ``query_vecs`` is one vector or a list (vector/hybrid); ``queries`` is one string
    or a list (keyword/hybrid). Returns [{content, source_id, item_id, id, meta, score}].
    """
    ids = [i for i in (ids or []) if i]
    if not ids:
        return []
    mode = (mode or "hybrid").lower()
    vecs = _as_vec_list(query_vecs)
    qstrs = _as_query_list(queries)
    conn = _conn()
    with _REG_LOCK:
        vec_lists = ([_vector_rows(conn, source_type, ids, model, v, top_k) for v in vecs]
                     if mode in ("vector", "hybrid") else [])
        kw_lists = []
        if mode in ("keyword", "hybrid") and qstrs:
            cands = _keyword_candidates(conn, source_type, ids)
            kw_lists = [_bm25_search(cands, q, top_k) for q in qstrs]

    if mode == "vector":
        return _rrf_merge(vec_lists, top_k) if len(vec_lists) != 1 else vec_lists[0][:int(top_k)]
    if mode == "keyword":
        return _rrf_merge(kw_lists, top_k) if len(kw_lists) != 1 else (kw_lists[0][:int(top_k)] if kw_lists else [])
    return _rrf_merge(vec_lists + kw_lists, top_k)


def retrieve_libraries(query_vecs, lib_ids: list, model: str, top_k: int,
                       mode: str = "hybrid", queries=None) -> list:
    """Top-k library chunks (thin wrapper over ``retrieve`` for source_type='library')."""
    return retrieve("library", lib_ids, query_vecs, model, top_k, mode=mode, queries=queries)


def retrieve_inline(query_vecs, data_text: str, embed_fn, top_k: int,
                    mode: str = "hybrid", queries=None, label: str = "Data") -> list:
    """Chunk the transient inline ``<Data>`` block and rank it under ``mode``. Nothing
    is persisted. Vector scoring embeds the chunks on the fly (skipped in keyword mode
    or when no embedder is available)."""
    chunks = chunk_text(data_text or "")
    if not chunks:
        return []
    mode = (mode or "hybrid").lower()
    vecs = _as_vec_list(query_vecs)
    qstrs = _as_query_list(queries)

    vec_lists = []
    if mode in ("vector", "hybrid") and vecs and embed_fn is not None:
        try:
            cvecs = embed_fn(chunks)
        except Exception:
            cvecs = None
        if cvecs:
            for qv in vecs:
                qv = [float(x) for x in qv]
                scored = [{"content": c, "source_id": "inline", "item_id": label,
                           "id": f"inline:{label}:{i}", "score": _cosine(qv, v)}
                          for i, (c, v) in enumerate(zip(chunks, cvecs))]
                scored.sort(key=lambda d: d["score"], reverse=True)
                vec_lists.append(scored[:int(top_k)])

    kw_lists = []
    if mode in ("keyword", "hybrid") and qstrs:
        cands = [{"content": c, "source_id": "inline", "item_id": label,
                  "id": f"inline:{label}:{i}"} for i, c in enumerate(chunks)]
        kw_lists = [_bm25_search(cands, q, top_k) for q in qstrs]

    if mode == "vector":
        return _rrf_merge(vec_lists, top_k) if len(vec_lists) != 1 else (vec_lists[0][:int(top_k)] if vec_lists else [])
    if mode == "keyword":
        return _rrf_merge(kw_lists, top_k) if len(kw_lists) != 1 else (kw_lists[0][:int(top_k)] if kw_lists else [])
    return _rrf_merge(vec_lists + kw_lists, top_k)


# --------------------------- Generic per-item upsert (persona knowledge & memory) ---------------------------

def upsert_items(source_type: str, source_id: str, items: list, embed_fn, model: str,
                 contextualize=None, batch_size: int = 64, max_workers: int = 1) -> dict:
    """Chunk + (optionally contextualize) + embed MANY items of one source in bulk and
    replace their chunks in the store. ``items`` is ``[(item_id, content, meta_dict)]``.

    The point of the bulk path (vs. calling ``upsert_item`` in a loop) is throughput:
    chunks from every changed item are pooled into ONE flat list and embedded with a
    few large, optionally concurrent ``/api/embed`` calls **outside ``_REG_LOCK``** —
    the network wait no longer holds the shared DuckDB connection. Only the per-item
    DELETE and a single ``executemany`` INSERT run under the lock.

    Returns ``{item_id: chunk_count}``. Embedding failures degrade to empty vectors so
    the chunks still persist (keyword/BM25 retrieval keeps working)."""
    conn = _conn()  # acquires+releases _REG_LOCK internally — must be before our lock
    now = datetime.utcnow().isoformat()

    # 1. Chunk every item (cheap CPU) — no lock, no network yet.
    prepared = []
    for item_id, content, imeta in items:
        content = (content or "").strip()
        chunks = chunk_text_semantic(content) if content else []
        prepared.append({
            "item_id": item_id,
            "meta_json": json.dumps(imeta or {}),
            "content": content,
            "chunks": chunks,
            "contexts": ["" for _ in chunks],
        })

    # 2. Optional contextual chunking (LLM per chunk) — concurrent, no lock.
    if contextualize is not None:
        _contextualize_all(prepared, contextualize, max_workers)

    # 3. Build one flat embed-input list across all items (context + chunk).
    flat_inputs = []
    ranges = []
    for p in prepared:
        start = len(flat_inputs)
        for ci, ch in enumerate(p["chunks"]):
            ctx = p["contexts"][ci]
            flat_inputs.append((ctx + "\n" + ch).strip() if ctx else ch)
        ranges.append((start, len(flat_inputs)))

    # 4. Embed everything OUTSIDE the lock (batched + optionally concurrent).
    vectors = _embed_batched(embed_fn, flat_inputs, batch_size, max_workers)

    # 5. Assemble rows, then delete-and-bulk-insert under the lock.
    rows = []
    counts = {}
    for p, (start, end) in zip(prepared, ranges):
        item_id = p["item_id"]
        item_vecs = vectors[start:end]
        for ci, ch in enumerate(p["chunks"]):
            vec = item_vecs[ci] if ci < len(item_vecs) else []
            ctx = p["contexts"][ci]
            rows.append([f"{source_type}:{source_id}:{item_id}:{ci}", source_type,
                         source_id, item_id, ci, ch, _hash(ch),
                         [float(x) for x in vec], model, now, ctx, p["meta_json"]])
        counts[item_id] = len(p["chunks"])

    with _REG_LOCK:
        for p in prepared:
            conn.execute(
                "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ? AND item_id = ?",
                [source_type, source_id, p["item_id"]])
        if rows:
            conn.executemany(
                "INSERT INTO rag_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return counts


def upsert_item(source_type: str, source_id: str, item_id: str, content: str,
                embed_fn, model: str, contextualize=None, meta: dict = None,
                batch_size: int = 64, max_workers: int = 1) -> int:
    """Chunk + (optionally contextualize) + embed one logical item and replace its
    existing chunks in the store. Used for persona knowledge documents and memories
    (both namespaced by ``source_id`` = persona id). ``meta`` is a small JSON dict
    persisted per chunk (e.g. a memory's emotional weight). Returns chunks written.

    Thin wrapper over ``upsert_items`` so the single-item callers (persona ingest) and
    the bulk compile path share one code path."""
    counts = upsert_items(source_type, source_id, [(item_id, content, meta)],
                          embed_fn, model, contextualize=contextualize,
                          batch_size=batch_size, max_workers=max_workers)
    return counts.get(item_id, 0)


def delete_item(source_type: str, source_id: str, item_id: str) -> None:
    conn = _conn()
    with _REG_LOCK:
        conn.execute(
            "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ? AND item_id = ?",
            [source_type, source_id, item_id])


def delete_source(source_id: str) -> None:
    """Remove every chunk for a source id across all source types (persona deletion)."""
    conn = _conn()
    with _REG_LOCK:
        conn.execute("DELETE FROM rag_chunks WHERE source_id = ?", [source_id])


def list_items(source_type: str, source_id: str) -> list:
    """Distinct item ids currently stored for a scope, with chunk counts."""
    conn = _conn()
    with _REG_LOCK:
        rows = conn.execute(
            "SELECT item_id, COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ? "
            "GROUP BY item_id", [source_type, source_id]).fetchall()
    return [{"item_id": r[0], "chunks": r[1]} for r in rows]
