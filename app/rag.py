#!/usr/bin/env python3
"""
Local Streaming Token — RAG vector store.

Retrieval Augmented Generation: chunk → embed → store → retrieve. Library documents
are chunked, embedded (via ``EmbedPool``) and persisted; at query time the top-k chunks
most relevant to the user's question are retrieved and injected into the prompt in
place of the full text.

This module owns everything **storage-independent**. The store itself lives behind
``app/vectorstore``, which offers two backends — encrypted DuckDB and plaintext
LanceDB — selectable in Settings → RAG. Nothing here should know which is active.

Design notes
- Persistent corpus = the user's selected **Libraries** (and persona knowledge and
  memories, namespaced by ``source_type``). Re-embedding is skipped when an item's
  content is unchanged, and vectors are reused for identical text even when it moves,
  so editing one paragraph of a long book re-embeds only what actually changed.
- Transient corpus = the inline ``<Data>`` block staged for one message. Embedded on
  the fly and scored in memory — never written to the store.
- Retrieval accepts multiple query variants and fuses the ranked lists with RRF, which
  merges by *rank* and so works across backends whose raw scores are not comparable.
- Writes stream in bounded waves so a shelf of ebooks never has to be held in memory
  at once.
"""

import hashlib
import json
import queue as _queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import core
from . import vectorstore
from .vectorstore import scoring


def _backend():
    """The active vector-store backend (see ``app/vectorstore``)."""
    return vectorstore.get_backend()


def reset_connection():
    """Drop the cached store handle so the next use reopens against the current data
    profile. Called when the active profile changes — otherwise RAG would keep
    reading/writing the previous profile's vector store."""
    vectorstore.reset()
    _clear_inline_cache()


def set_backend(name) -> str:
    """Settings hook: choose the vector-store backend ('lance' or 'duckdb')."""
    return vectorstore.set_backend(name)


def backend_name() -> str:
    return vectorstore.active_name()


def backend_status() -> dict:
    try:
        return _backend().status()
    except Exception as e:
        return {"backend": vectorstore.active_name(), "error": str(e)}


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
    return _backend().hashes(source_type, source_id, item_id, model)


def prune_items(source_type: str, source_id: str, keep_item_ids: list) -> int:
    """Delete stored chunks for items no longer present. Returns rows deleted."""
    return _backend().prune(source_type, source_id, keep_item_ids)


def count_chunks(source_type: str, source_id: str) -> int:
    """Total stored chunks for a scope (across all its items)."""
    return _backend().count(source_type, source_id)


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


class _Lane:
    """One embedding endpoint in the pool. Tracks throughput so the UI can show which
    box is doing the work, and health so a dead host stops being handed batches."""

    def __init__(self, call, name: str, base_url: str = ""):
        self._call = call
        self.name = name
        self.base_url = base_url
        self.batches = 0
        self.inputs = 0
        self.seconds = 0.0
        self.failures = 0
        self.down = False

    def embed(self, batch: list) -> list:
        t0 = time.time()
        vecs = self._call(batch)
        if not vecs or len(vecs) != len(batch):
            raise RuntimeError(f"embedder returned {len(vecs) if vecs else 0} vectors "
                               f"for {len(batch)} inputs")
        out = [[float(x) for x in v] for v in vecs]
        self.seconds += time.time() - t0
        self.batches += 1
        self.inputs += len(batch)
        return out

    def stats(self) -> dict:
        rate = (self.inputs / self.seconds) if self.seconds > 0 else 0.0
        return {"name": self.name, "base_url": self.base_url, "chunks": self.inputs,
                "rate": round(rate, 2), "failures": self.failures, "down": self.down}


class EmbedPool:
    """Embeds text across one or more Ollama hosts at once.

    Batches are dispatched from a shared queue to ``per_server_concurrency`` worker
    threads **per lane**, so a faster machine simply pulls more work ("balanced" mode
    in ``app/parallel.py`` terms). One embedding MODEL is used for the whole pool by
    design: vectors from different models live in different spaces and are not
    comparable, so mixing them would silently corrupt the index.

    This is the hot path for compile. It is lock-free (pure network) and MUST be
    called outside the store's write lock so a network wait never blocks readers.

    A batch that fails is retried on a *different* lane before being given up on; the
    lane that failed is marked down after ``max_retries`` failures so it stops
    receiving work. Unrecoverable inputs get empty vectors — the chunk still persists
    for keyword/BM25 retrieval — and are COUNTED and returned, so the caller can
    refuse to record a half-embedded corpus as cleanly compiled.
    """

    def __init__(self, lanes=None, model: str = "", embed_fn=None,
                 per_server_concurrency: int = 3, batch_size: int = 64,
                 max_retries: int = 2):
        self.model = model
        self.batch_size = max(1, _int_or(batch_size, 64))
        self.per_server_concurrency = max(1, _int_or(per_server_concurrency, 3))
        self.max_retries = max(1, _int_or(max_retries, 2))
        self.lanes = []
        for spec in (lanes or []):
            url = (spec.get("base_url") or "").strip().rstrip("/")
            if not url:
                continue
            client = core.OllamaClient(url)
            self.lanes.append(_Lane(
                (lambda c: lambda batch: c.embed(model, batch))(client),
                spec.get("name") or url, url))
        if not self.lanes and embed_fn is not None:
            # Single-callable mode: the existing embed_fn callers (persona ingest,
            # memory save) keep working unchanged.
            self.lanes.append(_Lane(embed_fn, "embedder", ""))
        # No lanes AND no callable means the caller deliberately asked for keyword-only
        # indexing. That is not a failure — chunks are stored with empty vectors and
        # BM25 retrieval works — so it must not be reported as one.
        self.keyword_only = not self.lanes

    @property
    def active(self) -> bool:
        return bool(self.lanes)

    def lane_stats(self) -> list:
        return [l.stats() for l in self.lanes]

    def embed_many(self, inputs: list, on_progress=None, stop_event=None):
        """Embed ``inputs`` -> ``(vectors_in_input_order, n_failed_inputs)``.

        ``on_progress(done, total)`` is called as batches land (from worker threads —
        it must be thread-safe and cheap). ``stop_event`` aborts between batches.
        """
        if not inputs:
            return [], 0
        if not self.lanes:
            # Keyword-only indexing — empty vectors are the intended outcome.
            return [[] for _ in inputs], 0

        bs = self.batch_size
        batches = [(i, inputs[s:s + bs])
                   for i, s in enumerate(range(0, len(inputs), bs))]
        results = [None] * len(batches)
        total = len(inputs)

        pending = _queue.Queue()
        for idx, batch in batches:
            pending.put((idx, batch, 0))          # (batch index, texts, attempts)

        state_lock = threading.Lock()
        done = [0]
        failed = [0]

        def tick(n):
            with state_lock:
                done[0] += n
                cur = done[0]
            if on_progress is not None:
                try:
                    on_progress(cur, total)
                except Exception:
                    pass

        # Batches still owed a result. A worker may only exit on an empty queue once
        # nothing is in flight — otherwise the last worker standing can drain the queue,
        # see Empty, and leave while another thread is about to requeue a failed batch,
        # which then gets written off as failed with healthy lanes sitting idle.
        inflight = [0]

        def worker(lane):
            while True:
                # A down lane must stop pulling work entirely. Letting it keep grabbing
                # batches just to reject them would burn each batch's retry budget and
                # fail work a healthy lane could have done.
                if lane.down:
                    return
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    idx, batch, attempts = pending.get_nowait()
                except _queue.Empty:
                    with state_lock:
                        busy = inflight[0]
                    if not busy:
                        return
                    time.sleep(0.01)    # a peer may still hand work back
                    continue
                with state_lock:
                    inflight[0] += 1
                try:
                    results[idx] = lane.embed(batch)
                    tick(len(batch))
                except Exception:
                    lane.failures += 1
                    if lane.failures >= self.max_retries and len(self.lanes) > 1:
                        lane.down = True
                    alive = [l for l in self.lanes if not l.down]
                    if attempts + 1 < self.max_retries and alive:
                        # Hand it back to the queue — another lane will pick it up.
                        pending.put((idx, batch, attempts + 1))
                    else:
                        results[idx] = [[] for _ in batch]
                        with state_lock:
                            failed[0] += len(batch)
                        tick(len(batch))
                finally:
                    with state_lock:
                        inflight[0] -= 1
                    pending.task_done()

        threads = []
        for lane in self.lanes:
            for _ in range(self.per_server_concurrency):
                t = threading.Thread(target=worker, args=(lane,), daemon=True)
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

        # Every lane died (or we were stopped) with work still queued — account for it
        # rather than silently returning empty vectors.
        while True:
            try:
                idx, batch, _a = pending.get_nowait()
            except _queue.Empty:
                break
            if results[idx] is None:
                results[idx] = [[] for _ in batch]
                if stop_event is None or not stop_event.is_set():
                    failed[0] += len(batch)

        out = []
        for i, (_idx, batch) in enumerate(batches):
            r = results[i]
            out.extend(r if r is not None else [[] for _ in batch])
        # A stop can leave trailing batches unprocessed; pad so the caller's
        # chunk<->vector zip stays aligned.
        if len(out) < total:
            out.extend([[] for _ in range(total - len(out))])
        return out[:total], failed[0]


def _int_or(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_pool(embed, model: str, batch_size: int = 64, max_workers: int = 1) -> EmbedPool:
    """Accept either an ``EmbedPool`` or a bare ``embed_fn`` callable (or None) and
    return a pool. Lets every existing ``embed_fn`` caller keep working while the
    compile path passes a real multi-server pool."""
    if isinstance(embed, EmbedPool):
        return embed
    return EmbedPool(model=model, embed_fn=embed, batch_size=batch_size,
                     per_server_concurrency=max_workers)


def _contextualize_all(prepared: list, contextualize, max_workers: int = 1,
                       on_progress=None) -> None:
    """Fill each prepared item's ``contexts`` list with LLM-written situating sentences,
    running the per-chunk ``contextualize(chunk, doc_summary)`` calls concurrently (up to
    ``max_workers``). Mutates ``prepared`` in place; a failed call degrades to '' so the
    raw chunk is still embedded. Lock-free — call outside the store's write lock.

    This is ONE LLM call per chunk and is by far the most expensive thing a compile can
    do — on an ebook-sized corpus it dominates everything else, which is why it reports
    its own progress phase rather than hiding inside the embed step."""
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

    done = 0
    total = len(tasks)

    def note():
        if on_progress is not None:
            try:
                on_progress(done, total)
            except Exception:
                pass

    workers = max(1, int(max_workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            for pi, ci, ctx in ex.map(_run, tasks):
                prepared[pi]["contexts"][ci] = ctx
                done += 1
                note()
    else:
        for t in tasks:
            pi, ci, ctx = _run(t)
            prepared[pi]["contexts"][ci] = ctx
            done += 1
            note()


def _doc_summary(content: str, max_words: int = 120) -> str:
    """A cheap 'what this document is about' blurb (its opening ~120 words) used to
    situate each chunk during contextual chunking. Avoids a second LLM summarization
    call per document."""
    words = (content or "").split()
    return " ".join(words[:max_words])


# --------------------------- Retrieval ---------------------------
#
# All retrieval honors a ``mode``:
#   vector  — embedding cosine similarity only (needs query vectors)
#   keyword — BM25 over chunk text only (needs no embeddings; works offline)
#   hybrid  — both, merged with reciprocal rank fusion (default)
# and accepts MULTIPLE query variants (from Prompt Reword, Phase 3). Each variant is
# retrieved independently and the ranked lists are RRF-merged, so one poor rewording
# can't sink retrieval.


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
    be = _backend()

    # Each query variant is retrieved independently; RRF fuses them by RANK, so the
    # backends' non-comparable score scales (cosine vs Lance distance, Python BM25 vs
    # tantivy) merge cleanly without normalisation.
    vec_lists = ([be.search_vector(source_type, ids, model, v, top_k) for v in vecs]
                 if mode in ("vector", "hybrid") else [])
    kw_lists = ([be.search_keyword(source_type, ids, q, top_k) for q in qstrs]
                if mode in ("keyword", "hybrid") and qstrs else [])

    if mode == "vector":
        return (scoring.rrf_merge(vec_lists, top_k) if len(vec_lists) != 1
                else vec_lists[0][:int(top_k)])
    if mode == "keyword":
        return (scoring.rrf_merge(kw_lists, top_k) if len(kw_lists) != 1
                else (kw_lists[0][:int(top_k)] if kw_lists else []))
    return scoring.rrf_merge(vec_lists + kw_lists, top_k)


def retrieve_libraries(query_vecs, lib_ids: list, model: str, top_k: int,
                       mode: str = "hybrid", queries=None) -> list:
    """Top-k library chunks (thin wrapper over ``retrieve`` for source_type='library')."""
    return retrieve("library", lib_ids, query_vecs, model, top_k, mode=mode, queries=queries)


def fuse(result_lists: list, top_k: int) -> list:
    """RRF-merge several independently-retrieved rankings into one.

    Each corpus (libraries, a chat's attachments, its thread) is retrieved separately and
    comes back with scores from its own space — a Lance distance, a cosine, a BM25 score,
    or an RRF score from an earlier merge. Sorting those together by raw value compares
    numbers that mean different things, and in practice let one corpus crowd out the
    rest. RRF ranks instead, so every corpus's best hit is weighted alike.

    Exposed here so callers outside this module don't reach into the store package."""
    return scoring.rrf_merge([lst for lst in (result_lists or []) if lst], top_k)


# Chunks+vectors for the transient corpus, keyed by (embed model, sha1 of the text).
# The inline corpus now includes a chat's PINNED ATTACHMENTS, which are re-sent every
# turn and can be 200k characters each (a YouTube transcript). Embedding that from
# scratch on every send made each message in such a chat pay the full cost again, so
# an unchanged corpus is embedded once and reused. Small and bounded: this is a cache,
# never a store — nothing here is persisted.
_INLINE_CACHE_MAX = 8
_INLINE_LOCK = threading.Lock()
_INLINE_CACHE = OrderedDict()


def _clear_inline_cache():
    with _INLINE_LOCK:
        _INLINE_CACHE.clear()


def _inline_vectors(chunks: list, data_text: str, embed_fn, embed_model: str):
    """Embed the transient chunks, reusing the last few corpora. Returns None when the
    embedder is unavailable or fails, which the caller treats as "no vector half"."""
    key = (embed_model or "", _hash(data_text))
    with _INLINE_LOCK:
        hit = _INLINE_CACHE.get(key)
        if hit is not None and len(hit) == len(chunks):
            _INLINE_CACHE.move_to_end(key)
            return hit
    try:
        cvecs = embed_fn(chunks)
    except Exception:
        return None
    if cvecs:
        with _INLINE_LOCK:
            _INLINE_CACHE[key] = cvecs
            _INLINE_CACHE.move_to_end(key)
            while len(_INLINE_CACHE) > _INLINE_CACHE_MAX:
                _INLINE_CACHE.popitem(last=False)
    return cvecs


def retrieve_inline(query_vecs, data_text: str, embed_fn, top_k: int,
                    mode: str = "hybrid", queries=None, label: str = "Data",
                    embed_model: str = "") -> list:
    """Chunk the transient inline ``<Data>`` block and rank it under ``mode``. Nothing
    is persisted. Vector scoring embeds the chunks on the fly (skipped in keyword mode
    or when no embedder is available); an unchanged corpus reuses the previous
    embedding rather than paying for it again — see ``_inline_vectors``."""
    chunks = chunk_text(data_text or "")
    if not chunks:
        return []
    mode = (mode or "hybrid").lower()
    vecs = _as_vec_list(query_vecs)
    qstrs = _as_query_list(queries)

    vec_lists = []
    if mode in ("vector", "hybrid") and vecs and embed_fn is not None:
        cvecs = _inline_vectors(chunks, data_text or "", embed_fn, embed_model)
        if cvecs:
            for qv in vecs:
                qv = [float(x) for x in qv]
                scored = [{"content": c, "source_id": "inline", "item_id": label,
                           "id": f"inline:{label}:{i}", "score": scoring.cosine(qv, v)}
                          for i, (c, v) in enumerate(zip(chunks, cvecs))]
                scored.sort(key=lambda d: d["score"], reverse=True)
                vec_lists.append(scored[:int(top_k)])

    kw_lists = []
    if mode in ("keyword", "hybrid") and qstrs:
        cands = [{"content": c, "source_id": "inline", "item_id": label,
                  "id": f"inline:{label}:{i}"} for i, c in enumerate(chunks)]
        kw_lists = [scoring.bm25_search(cands, q, top_k) for q in qstrs]

    if mode == "vector":
        return scoring.rrf_merge(vec_lists, top_k) if len(vec_lists) != 1 else (vec_lists[0][:int(top_k)] if vec_lists else [])
    if mode == "keyword":
        return scoring.rrf_merge(kw_lists, top_k) if len(kw_lists) != 1 else (kw_lists[0][:int(top_k)] if kw_lists else [])
    return scoring.rrf_merge(vec_lists + kw_lists, top_k)


# --------------------------- Generic per-item upsert (persona knowledge & memory) ---------------------------

def _chunk_all(items: list, max_workers: int = 1, on_progress=None) -> list:
    """Chunk every item into ``prepared`` dicts, preserving order. Runs across a thread
    pool because chonkie's chunker is backed by Rust tokenizers that release the GIL,
    so this actually scales (unlike PDF parsing, which needs processes)."""
    items = list(items)
    prepared = [None] * len(items)
    done = [0]
    lock = threading.Lock()

    def one(i):
        item_id, content, imeta = items[i]
        content = (content or "").strip()
        chunks = chunk_text_semantic(content) if content else []
        prepared[i] = {
            "item_id": item_id,
            "meta_json": json.dumps(imeta or {}),
            "content": content,
            "chunks": chunks,
            "contexts": ["" for _ in chunks],
        }
        if on_progress is not None:
            with lock:
                done[0] += 1
                cur = done[0]
            try:
                on_progress(cur, len(items))
            except Exception:
                pass

    workers = max(1, int(max_workers or 1))
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
            list(ex.map(one, range(len(items))))
    else:
        for i in range(len(items)):
            one(i)
    return prepared


def upsert_items(source_type: str, source_id: str, items: list, embed_fn, model: str,
                 contextualize=None, batch_size: int = 64, max_workers: int = 1,
                 on_progress=None, stop_event=None, wave_size: int = 2000) -> dict:
    """Chunk + (optionally contextualize) + embed MANY items of one source and replace
    their chunks in the store. ``items`` is ``[(item_id, content, meta_dict)]``.
    ``embed_fn`` may be a plain ``list[str] -> list[list[float]]`` callable or an
    :class:`EmbedPool` fanning batches out across several hosts.

    Embedding and writing run in **bounded waves** of ~``wave_size`` chunks rather than
    pooling the whole corpus: several ebooks would otherwise hold every vector in
    memory as Python floats (a 768-float list is ~6 KB, so 50k chunks ≈ 300 MB before
    overhead, and measurably ~1.7 GB in practice) before a single row was written.
    Waves also bound how long the store's write lock is held, and give the UI something
    to move a progress bar with.

    Vectors already computed for identical text under the same model are reused from
    the store rather than re-embedded (``backend.cached_vectors``).

    Lock discipline (load-bearing): chunking, contextualization and embedding all happen
    OUTSIDE any store lock — only the backend's delete+insert is serialised, because on
    DuckDB that lock guards the single shared connection every reader uses. Never hold
    it across a network wait.

    ``on_progress(phase, done, total, **extra)`` reports "chunk"/"context"/"embed"
    progress; ``stop_event`` aborts between waves, leaving written items intact.

    Returns ``{item_id: chunk_count}`` plus a ``__meta__`` entry carrying
    ``{"failed", "cached", "stopped", "lanes"}``."""
    be = _backend()
    now = datetime.utcnow().isoformat()
    pool = _as_pool(embed_fn, model, batch_size, max_workers)

    def report(phase, done, total, **extra):
        if on_progress is not None:
            try:
                on_progress(phase, done, total, **extra)
            except Exception:
                pass

    # 1. Chunk every item (cheap CPU, parallel) — no lock, no network yet. Doing this
    #    up front is what makes a chunk-denominated progress bar possible at all.
    prepared = _chunk_all(items, max_workers, lambda d, t: report("chunk", d, t))

    # 2. Optional contextual chunking (one LLM call per chunk) — concurrent, no lock.
    if contextualize is not None:
        _contextualize_all(prepared, contextualize, max_workers,
                           lambda d, t: report("context", d, t))

    # 3. Flatten to (item index, chunk index, embed input) work units.
    units = []
    counts = {}
    for pi, p in enumerate(prepared):
        counts[p["item_id"]] = len(p["chunks"])
        for ci, ch in enumerate(p["chunks"]):
            ctx = p["contexts"][ci]
            units.append((pi, ci, (ctx + "\n" + ch).strip() if ctx else ch))
    total_chunks = len(units)
    report("embed", 0, total_chunks)

    if not units:
        # Nothing to embed, but the items may still have stale rows to clear.
        be.delete_items(source_type, source_id, [p["item_id"] for p in prepared])
        counts["__meta__"] = {"failed": 0, "cached": 0, "stopped": False, "lanes": [],
                              "complete": {p["item_id"] for p in prepared}}
        return counts

    # 4. Reuse anything already embedded under this model (one query for the corpus).
    embed_hashes = [_hash(text) for _pi, _ci, text in units]
    cache = be.cached_vectors(model, embed_hashes)

    # 5. Embed + write in bounded waves. An item's DELETE is issued with the first wave
    #    carrying its chunks, so a reader never observes it half-replaced.
    deleted = set()
    embedded_done = 0
    failed_total = 0
    cached_total = 0
    stopped = False
    wave = max(1, int(wave_size or 2000))
    # Per-item bookkeeping so the caller can tell which items are FULLY and cleanly
    # written. A stopped run or a dead embed server must not let compile record a
    # half-embedded document as cleanly compiled.
    written = {}
    tainted = set()

    for start in range(0, len(units), wave):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        part = units[start:start + wave]
        part_hashes = embed_hashes[start:start + wave]

        miss_idx = [i for i, h in enumerate(part_hashes) if h not in cache]
        cached_total += len(part) - len(miss_idx)
        base = embedded_done + (len(part) - len(miss_idx))

        vecs = [cache.get(h, []) for h in part_hashes]
        if miss_idx:
            fresh, failed = pool.embed_many(
                [part[i][2] for i in miss_idx],
                on_progress=lambda d, _t, _b=base: report(
                    "embed", _b + d, total_chunks,
                    cached=cached_total, failed=failed_total),
                stop_event=stop_event)
            for slot, v in zip(miss_idx, fresh):
                vecs[slot] = v
                if v:
                    cache[part_hashes[slot]] = v
            failed_total += failed
        embedded_done += len(part)
        report("embed", embedded_done, total_chunks,
               cached=cached_total, failed=failed_total)

        rows = []
        for (pi, ci, _text), vec, ehash in zip(part, vecs, part_hashes):
            p = prepared[pi]
            ch = p["chunks"][ci]
            written[p["item_id"]] = written.get(p["item_id"], 0) + 1
            if not vec and not pool.keyword_only:
                tainted.add(p["item_id"])   # embedding failed for this chunk
            rows.append({
                "id": f"{source_type}:{source_id}:{p['item_id']}:{ci}",
                "source_type": source_type, "source_id": source_id,
                "item_id": p["item_id"], "chunk_index": ci,
                "content": ch, "content_hash": _hash(ch), "vector": vec,
                "model": model, "updated": now, "context": p["contexts"][ci],
                "meta": p["meta_json"], "embed_hash": ehash,
            })

        # Delete each touched item's old chunks with the FIRST wave that carries it,
        # in the same locked operation as the insert, so a reader never observes an
        # item half-replaced.
        touched = sorted({pi for pi, _ci, _t in part} - deleted)
        be.replace_wave(source_type, source_id,
                        [prepared[pi]["item_id"] for pi in touched], rows)
        deleted.update(touched)

    # Items that chunked to nothing never appear in ``units`` — clear their old rows.
    empty = [p["item_id"] for pi, p in enumerate(prepared)
             if not p["chunks"] and pi not in deleted]
    if empty:
        be.delete_items(source_type, source_id, empty)

    # Build/refresh the store's indexes once per run rather than per wave — index
    # maintenance is the expensive part, and mid-compile the index would only be
    # rebuilt again by the next wave. No-op on backends without durable indexes.
    #
    # This runs even when the user stopped the compile. Whatever waves DID land are
    # already in the store and are searched from now on; skipping the index rebuild
    # left them reachable only through the slow full-scan fallback until some later
    # compile happened to finish.
    try:
        optimize = getattr(be, "optimize", None)
        if optimize is not None:
            optimize()
    except Exception:
        pass

    complete = {p["item_id"] for p in prepared
                if written.get(p["item_id"], 0) == len(p["chunks"])
                and p["item_id"] not in tainted}
    counts["__meta__"] = {"failed": failed_total, "cached": cached_total,
                          "stopped": stopped, "lanes": pool.lane_stats(),
                          "complete": complete}
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
    _backend().delete_items(source_type, source_id, [item_id])


def _backend_has_store(name: str) -> bool:
    """Does this backend's store already exist on disk? Guards the sweep below so
    deleting a library never *creates* an empty store for a backend never used."""
    target = core.RAG_LANCE_DIR if name == vectorstore.LANCE else core.RAG_DB_FILE
    try:
        return Path(target).exists()
    except Exception:
        return False


def delete_source(source_id: str) -> None:
    """Remove every chunk for a source id across all source types and BOTH vector-store
    backends (library / persona deletion).

    Both backends can hold vectors at once, so deleting while LanceDB was active used to
    leave the DuckDB copy of the user's content on disk indefinitely — and vice versa.
    A backend that has never been opened (no store file) is skipped rather than created.
    """
    active = vectorstore.active_name()
    for name in vectorstore.BACKENDS:
        if name != active and not _backend_has_store(name):
            continue
        try:
            vectorstore.get_backend(name).delete_source(source_id)
        except Exception:
            pass          # a store we cannot open has nothing to leak
    if vectorstore.active_name() != active:
        vectorstore.get_backend(active)   # restore the cached instance


def list_items(source_type: str, source_id: str) -> list:
    """Distinct item ids currently stored for a scope, with chunk counts."""
    return _backend().list_items(source_type, source_id)


def set_ann_enabled(enabled: bool) -> None:
    """Settings hook for the DuckDB backend's in-memory HNSW sidecar. A no-op on
    LanceDB, whose vector index is durable and always on."""
    be = _backend()
    fn = getattr(be, "set_ann_enabled", None)
    if fn is None:
        # Import the module directly so the setting still applies to the DuckDB
        # backend when Lance happens to be the active one.
        from .vectorstore import duckdb_backend
        duckdb_backend.set_ann_enabled(enabled)
    else:
        fn(enabled)


def ann_status() -> dict:
    """Vector-index state for diagnostics (shape differs per backend)."""
    return backend_status()
