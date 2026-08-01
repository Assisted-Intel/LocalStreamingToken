#!/usr/bin/env python3
"""
Local Streaming Token — DuckDB vector-store backend (encrypted at rest).

This is the original store, moved out of ``rag.py`` behind the backend interface with
its behaviour intact. Its distinguishing property is **encryption**: the whole file is
AES-256 encrypted by DuckDB, keyed by the DEK that the login password unwraps
(``core.duckdb_connect``). Nothing readable hits disk.

The costs of that guarantee, and the reason the LanceDB backend exists:

* **No durable vector index.** DuckDB's ``vss`` HNSW cannot be created inside an
  encrypted database at all (verified: internal allocator assertion), and even
  unencrypted its persistence is experimental with documented data-loss risk on unclean
  shutdown. So the index here is an in-memory sidecar in an attached ``:memory:``
  catalog, rebuilt on demand and invalidated by every write.
* **No inverted index.** Keyword search materialises every in-scope chunk into Python
  and scores it with BM25, so its cost grows with corpus size rather than match count.

State is module-level (connection, lock, sidecar) because there is exactly one store per
process; the class is a thin facade over it.
"""

import threading

from .. import core
from . import scoring

# Process-wide (connection, lock). Lazy — only opened when RAG is first used.
_CONN = None
_REG_LOCK = threading.Lock()


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
                    meta VARCHAR,
                    embed_hash VARCHAR
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
    if "embed_hash" not in cols:
        # Hash of the text that was actually EMBEDDED (context + chunk when contextual
        # chunking is on, else the raw chunk) — the key for the embedding reuse cache.
        # NULL on rows written before this column existed: they simply never produce a
        # cache hit, which is correct, just not free.
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN embed_hash VARCHAR")


def _vec_literal(vec) -> str:
    """Render a vector as a DuckDB list literal, e.g. ``[0.1,0.2]``.

    Binding a Python list to a ``FLOAT[]`` parameter makes DuckDB run its per-value
    object-conversion path, which is dramatically slower than binding a string and
    letting SQL do ``CAST(? AS FLOAT[])`` — ~30x on a healthy interpreter here, and far
    more when an optional dependency is broken (see
    ``core._neutralize_broken_optional_imports``). ``repr`` round-trips a float exactly,
    and the column is single-precision anyway."""
    if not vec:
        return "[]"
    return "[" + ",".join(map(repr, (float(x) for x in vec))) + "]"


# --------------------------- ANN index (in-memory HNSW sidecar) ---------------------------
#
# See the module docstring for why this cannot live on disk. ``rag_chunks`` stays the
# source of truth; the sidecar is a rebuildable cache, like the compile manifest. Every
# write path calls ``_ann_invalidate()``; the next retrieval rebuilds it.
#
# The index lives in a dedicated in-memory catalog attached to the same connection
# (``ATTACH ':memory:' AS annmem``) rather than the implicit ``memory`` one, because
# ``core.duckdb_connect`` opens a plain file connection when the app is locked, where no
# ``memory`` catalog exists.

_ANN_DB = "annmem"
_ANN_OK = None          # None = untried, True/False after the first LOAD vss
_ANN_STATE = None       # {"model": str, "dim": int, "rows": int} when built
_ANN_DIRTY = True
_ANN_DISABLED = False


def set_ann_enabled(enabled: bool) -> None:
    """Settings hook: turn the ANN sidecar on/off. Disabling drops it immediately so
    the RAM comes back without a restart."""
    global _ANN_DISABLED
    _ANN_DISABLED = not bool(enabled)
    if _ANN_DISABLED:
        _ann_drop()


def _ann_invalidate() -> None:
    """Mark the sidecar stale. Called after any write to ``rag_chunks``."""
    global _ANN_DIRTY
    _ANN_DIRTY = True


def _ann_drop() -> None:
    global _ANN_STATE, _ANN_DIRTY
    conn = _CONN
    if conn is not None:
        try:
            with _REG_LOCK:
                conn.execute(f"DROP TABLE IF EXISTS {_ANN_DB}.rag_ann")
        except Exception:
            pass
    _ANN_STATE = None
    _ANN_DIRTY = True


def _ann_load(conn) -> bool:
    """Load the vss extension once. Returns False (permanently) when unavailable — the
    extension is fetched from the network on first install, which an offline machine
    won't have, and retrieval must keep working regardless."""
    global _ANN_OK
    if _ANN_OK is not None:
        return _ANN_OK
    try:
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        try:
            conn.execute("SET hnsw_enable_experimental_persistence = true")
        except Exception:
            pass
        conn.execute(f"ATTACH IF NOT EXISTS ':memory:' AS {_ANN_DB}")
        _ANN_OK = True
    except Exception:
        _ANN_OK = False
    return _ANN_OK


def _ann_ensure(conn, model: str):
    """Build/refresh the in-memory HNSW sidecar for ``model``. Returns its dimension,
    or None when unavailable (caller falls back to the scan). Assumes ``_REG_LOCK`` is
    NOT held; takes it around the DuckDB work."""
    global _ANN_STATE, _ANN_DIRTY
    if _ANN_DISABLED:
        return None
    state = _ANN_STATE
    if state is not None and not _ANN_DIRTY and state.get("model") == model:
        return state.get("dim")
    if not _ann_load(conn):
        return None
    try:
        with _REG_LOCK:
            row = conn.execute(
                "SELECT len(embedding) FROM rag_chunks WHERE model = ? AND len(embedding) > 0 "
                "LIMIT 1", [model]).fetchone()
            if not row or not row[0]:
                _ANN_STATE = None
                _ANN_DIRTY = False
                return None
            dim = int(row[0])
            conn.execute(f"DROP TABLE IF EXISTS {_ANN_DB}.rag_ann")
            conn.execute(
                f"CREATE TABLE {_ANN_DB}.rag_ann (id VARCHAR, source_type VARCHAR, "
                f"source_id VARCHAR, item_id VARCHAR, vec FLOAT[{dim}])")
            conn.execute(
                f"INSERT INTO {_ANN_DB}.rag_ann "
                f"SELECT id, source_type, source_id, item_id, embedding::FLOAT[{dim}] "
                f"FROM rag_chunks WHERE model = ? AND len(embedding) = ?", [model, dim])
            n = conn.execute(f"SELECT COUNT(*) FROM {_ANN_DB}.rag_ann").fetchone()[0]
            conn.execute(f"CREATE INDEX rag_ann_hnsw ON {_ANN_DB}.rag_ann "
                         f"USING HNSW (vec) WITH (metric = 'cosine')")
        _ANN_STATE = {"model": model, "dim": dim, "rows": int(n)}
        _ANN_DIRTY = False
        return dim
    except Exception:
        # Any failure here is non-fatal: drop back to the exact-scan path.
        try:
            with _REG_LOCK:
                conn.execute(f"DROP TABLE IF EXISTS {_ANN_DB}.rag_ann")
        except Exception:
            pass
        _ANN_STATE = None
        _ANN_DIRTY = False
        return None


def ann_status() -> dict:
    """What the sidecar currently holds (for diagnostics/settings display)."""
    st = dict(_ANN_STATE or {})
    st["available"] = bool(_ANN_OK) and not _ANN_DISABLED
    st["enabled"] = not _ANN_DISABLED
    st["stale"] = _ANN_DIRTY
    return st


def _ann_rows(conn, source_type: str, ids: list, dim: int, qv: list, top_k: int):
    """Top-k via the in-memory HNSW sidecar, or None to signal "fall back to the scan".

    The scope filter is applied AFTER an unfiltered nearest-neighbour fetch, because
    DuckDB only uses the HNSW index scan for an unfiltered ``ORDER BY … LIMIT``. We
    over-fetch and then filter; if that didn't surface enough in-scope chunks the result
    would be silently short, so we return None and let the exact scan run. (LanceDB
    supports a filtered ANN search directly and needs none of this.)"""
    if dim != len(qv):
        return None
    fetch = max(int(top_k) * 20, 200)
    try:
        rows = conn.execute(
            f"SELECT id, source_type, source_id, "
            f"       array_cosine_distance(vec, CAST(? AS FLOAT[{dim}])) AS dist "
            f"FROM {_ANN_DB}.rag_ann ORDER BY dist LIMIT ?",
            [_vec_literal(qv), fetch]).fetchall()
    except Exception:
        return None
    scope = set(ids)
    hits = [(r[0], 1.0 - float(r[3])) for r in rows
            if r[1] == source_type and r[2] in scope][:int(top_k)]
    if len(hits) < int(top_k):
        return None
    by_id = {cid: score for cid, score in hits}
    placeholders = ", ".join("?" for _ in by_id)
    meta = conn.execute(
        f"SELECT id, content, source_id, item_id, meta FROM rag_chunks "
        f"WHERE id IN ({placeholders})", list(by_id.keys())).fetchall()
    out = [{"content": m[1], "source_id": m[2], "item_id": m[3], "id": m[0],
            "meta": m[4], "score": by_id.get(m[0], -1.0)} for m in meta]
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:int(top_k)]


def _vector_rows(conn, source_type: str, ids: list, model: str, qv: list, top_k: int,
                 ann_dim: int = None) -> list:
    """One vector query over chunks of ``source_type`` in ``ids``: the HNSW sidecar when
    it's built and covers the query, else DuckDB ``list_cosine_similarity``, else a
    pure-Python cosine fallback."""
    placeholders = ", ".join("?" for _ in ids)
    qv = [float(x) for x in qv]
    if ann_dim:
        hit = _ann_rows(conn, source_type, ids, ann_dim, qv, top_k)
        if hit is not None:
            return hit
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
            [_vec_literal(qv), source_type, model, *ids, int(top_k)],
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
               "meta": r[4], "score": scoring.cosine(qv, r[5])} for r in rows]
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


# --------------------------- Backend ---------------------------

class DuckDBBackend:
    """Encrypted DuckDB store. See the module docstring for its trade-offs."""

    name = "duckdb"
    encrypted = True

    # -- lifecycle ---------------------------------------------------------
    def close(self):
        global _CONN, _ANN_STATE, _ANN_DIRTY
        with _REG_LOCK:
            if _CONN is not None:
                try:
                    _CONN.close()
                except Exception:
                    pass
                _CONN = None
        # The ANN sidecar lived in that connection's in-memory catalog — it went with it.
        _ANN_STATE = None
        _ANN_DIRTY = True

    # -- writes ------------------------------------------------------------
    def replace_wave(self, source_type: str, source_id: str, delete_item_ids: list,
                     rows: list) -> None:
        """Delete the named items' existing chunks and insert ``rows`` atomically.

        One locked block so a reader never observes an item half-replaced. Chunking and
        embedding happen outside this call — the lock guards the single shared
        connection every reader uses, so it must never be held across a network wait."""
        conn = _conn()
        payload = [
            [r["id"], r["source_type"], r["source_id"], r["item_id"], r["chunk_index"],
             r["content"], r["content_hash"], _vec_literal(r["vector"]), r["model"],
             r["updated"], r["context"], r["meta"], r["embed_hash"]]
            for r in rows
        ]
        with _REG_LOCK:
            for item_id in delete_item_ids or []:
                conn.execute(
                    "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ? "
                    "AND item_id = ?", [source_type, source_id, item_id])
            if payload:
                # CAST(? AS FLOAT[]) rather than binding a Python list — see _vec_literal.
                conn.executemany(
                    "INSERT INTO rag_chunks VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, CAST(? AS FLOAT[]), ?, ?, ?, ?, ?)", payload)
        if payload or delete_item_ids:
            _ann_invalidate()

    def delete_items(self, source_type: str, source_id: str, item_ids: list) -> None:
        conn = _conn()
        ids = [i for i in (item_ids or []) if i]
        if not ids:
            return
        with _REG_LOCK:
            for item_id in ids:
                conn.execute(
                    "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ? "
                    "AND item_id = ?", [source_type, source_id, item_id])
        _ann_invalidate()

    def delete_source(self, source_id: str) -> None:
        conn = _conn()
        with _REG_LOCK:
            conn.execute("DELETE FROM rag_chunks WHERE source_id = ?", [source_id])
        _ann_invalidate()

    def prune(self, source_type: str, source_id: str, keep_item_ids: list) -> int:
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
        if before != after:
            _ann_invalidate()
        return int(before - after)

    # -- reads -------------------------------------------------------------
    def count(self, source_type: str, source_id: str) -> int:
        conn = _conn()
        with _REG_LOCK:
            return int(conn.execute(
                "SELECT COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ?",
                [source_type, source_id]).fetchone()[0])

    def hashes(self, source_type: str, source_id: str, item_id: str, model: str) -> list:
        conn = _conn()
        with _REG_LOCK:
            rows = conn.execute(
                "SELECT content_hash FROM rag_chunks WHERE source_type = ? AND source_id = ? "
                "AND item_id = ? AND model = ? ORDER BY chunk_index",
                [source_type, source_id, item_id, model]).fetchall()
        return [r[0] for r in rows]

    def list_items(self, source_type: str, source_id: str) -> list:
        conn = _conn()
        with _REG_LOCK:
            rows = conn.execute(
                "SELECT item_id, COUNT(*) FROM rag_chunks WHERE source_type = ? AND source_id = ? "
                "GROUP BY item_id", [source_type, source_id]).fetchall()
        return [{"item_id": r[0], "chunks": r[1]} for r in rows]

    def cached_vectors(self, model: str, embed_hashes: list) -> dict:
        """{embed_hash: vector} for inputs already embedded under ``model``."""
        uniq = [h for h in dict.fromkeys(embed_hashes) if h]
        if not uniq:
            return {}
        conn = _conn()
        found = {}
        step = 2000   # keep the IN list to a sane size at ebook scale
        with _REG_LOCK:
            for i in range(0, len(uniq), step):
                part = uniq[i:i + step]
                placeholders = ", ".join("?" for _ in part)
                rows = conn.execute(
                    f"SELECT embed_hash, ANY_VALUE(embedding) FROM rag_chunks "
                    f"WHERE model = ? AND embed_hash IN ({placeholders}) "
                    f"AND len(embedding) > 0 GROUP BY embed_hash",
                    [model, *part]).fetchall()
                for h, vec in rows:
                    if vec:
                        found[h] = [float(x) for x in vec]
        return found

    # -- search ------------------------------------------------------------
    def search_vector(self, source_type: str, ids: list, model: str, qv: list,
                      top_k: int) -> list:
        conn = _conn()
        # Build/refresh the sidecar BEFORE taking the lock — _ann_ensure takes it
        # itself, and _REG_LOCK is not reentrant.
        ann_dim = _ann_ensure(conn, model)
        with _REG_LOCK:
            return _vector_rows(conn, source_type, ids, model, qv, top_k, ann_dim)

    def search_keyword(self, source_type: str, ids: list, query: str, top_k: int) -> list:
        conn = _conn()
        with _REG_LOCK:
            cands = _keyword_candidates(conn, source_type, ids)
        return scoring.bm25_search(cands, query, top_k)

    # -- diagnostics -------------------------------------------------------
    def status(self) -> dict:
        return {"backend": self.name, "encrypted": True,
                "path": str(core.RAG_DB_FILE), "ann": ann_status()}
