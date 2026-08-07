#!/usr/bin/env python3
"""
Local Streaming Token — LanceDB vector-store backend (plaintext, indexed).

The fast store. Unlike the DuckDB backend it has a **durable on-disk vector index** and
a **native full-text (tantivy) index**, so neither retrieval path degrades into a full
scan as the corpus grows. It also supports a filtered ANN search, so scoping a query to
a set of libraries needs none of the over-fetch-then-filter workaround the DuckDB HNSW
sidecar requires.

**This store is NOT encrypted.** Chunk text and embeddings sit in plaintext on disk,
readable by anyone with file access and independent of the login password. That is the
deliberate trade for the tantivy index, which cannot operate on ciphertext. Users who
need the original guarantee select the DuckDB backend in Settings → RAG; the two stores
are independent, so switching never destroys the other's data.

Everything here goes through Arrow directly. **Never pandas** — it is an optional
LanceDB dependency that may be installed-but-broken, in which case
``core._neutralize_broken_optional_imports`` blocks it outright.
"""

import threading

from .. import core
from . import scoring

# Vector width when we cannot infer one (a keyword-only index, where every chunk is
# stored with a null vector). Matches nomic-embed-text, the default embedding model, so
# the table is usually right anyway if embeddings arrive later.
_FALLBACK_DIM = 768

# Below this many rows an ANN index costs more than it saves and Lance's brute-force
# search is already fast; above it, build one.
_ANN_MIN_ROWS = 5000

_LOCK = threading.RLock()


def _sql_str(value) -> str:
    """A single-quoted SQL literal with quotes escaped, for Lance filter predicates."""
    return "'" + str(value if value is not None else "").replace("'", "''") + "'"


def _scope_predicate(source_type: str, ids: list, model: str = None) -> str:
    """Filter predicate for a retrieval scope.

    ``model`` is required for VECTOR search and must be omitted for keyword search
    (BM25 is model-independent). Tables are keyed by vector WIDTH, so two different
    embedding models of the same width — 768 covers nomic-embed-text, bge-base and
    gte-base — share one table. Without the model filter a scope still holding vectors
    from the previous model gets ranked against the new model's query vector, which
    returns confident nonsense instead of nothing. The DuckDB backend has always
    filtered on model; this keeps the two stores in agreement."""
    inner = ", ".join(_sql_str(i) for i in ids)
    pred = f"source_type = {_sql_str(source_type)} AND source_id IN ({inner})"
    if model is not None:
        pred += f" AND model = {_sql_str(model)}"
    return pred


class LanceBackend:
    """Plaintext LanceDB store. See the module docstring for its trade-offs."""

    name = "lance"
    encrypted = False

    def __init__(self):
        self._db = None
        self._tables = {}          # dim -> table handle
        self._fts_ready = set()    # dims whose FTS index exists
        self._ann_ready = set()    # dims whose vector index exists

    # -- lifecycle ---------------------------------------------------------
    def _conn(self):
        if self._db is None:
            import lancedb
            path = str(core.RAG_LANCE_DIR)
            self._db = lancedb.connect(path)
        return self._db

    def close(self):
        with _LOCK:
            self._db = None
            self._tables.clear()
            self._fts_ready.clear()
            self._ann_ready.clear()

    # -- schema ------------------------------------------------------------
    def _schema(self, dim: int):
        import pyarrow as pa
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("item_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            # Nullable: a chunk whose embedding failed, or keyword-only indexing, is
            # still stored so BM25/FTS can find it — it simply never matches a vector
            # query. This is the Lance equivalent of DuckDB's empty FLOAT[].
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=True),
            pa.field("content", pa.string()),
            pa.field("context", pa.string()),
            pa.field("meta", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("embed_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("updated", pa.string()),
        ])

    def _table_name(self, dim: int) -> str:
        # One table per embedding width: Lance needs a fixed-size vector column, and a
        # model change (which changes the width) already forces a full recompile via
        # compile.signature.
        return f"chunks_{int(dim)}"

    def _table_names(self) -> list:
        """Table names, across both lancedb generations.

        ``list_tables()`` superseded the deprecated ``table_names()``, but it is NOT a
        drop-in: it returns a paginated ``ListTablesResponse`` whose ``.tables`` holds the
        names (iterating the response itself yields field tuples, not names). The page
        loop is bounded because this app only ever creates one table per embedding
        width — it exists so a future change can't silently truncate the list."""
        db = self._conn()
        lister = getattr(db, "list_tables", None)
        if lister is None:
            return list(db.table_names())           # older lancedb
        names, token = [], None
        for _ in range(100):
            resp = lister(page_token=token)
            names.extend(getattr(resp, "tables", None) or [])
            token = getattr(resp, "page_token", None)
            if not token:
                break
        return names

    def _existing_dims(self) -> list:
        try:
            names = self._table_names()
        except Exception:
            return []
        dims = []
        for n in names:
            if n.startswith("chunks_"):
                try:
                    dims.append(int(n.split("_", 1)[1]))
                except ValueError:
                    continue
        return sorted(dims)

    def _table(self, dim: int, create: bool = False):
        dim = int(dim)
        with _LOCK:
            if dim in self._tables:
                return self._tables[dim]
            db = self._conn()
            name = self._table_name(dim)
            try:
                tbl = db.open_table(name)
            except Exception:
                if not create:
                    return None
                tbl = db.create_table(name, schema=self._schema(dim))
            self._tables[dim] = tbl
            return tbl

    def _any_table(self):
        """The single populated table, when there is exactly one width in play (the
        normal case). Returns (dim, table) or (None, None)."""
        dims = self._existing_dims()
        if not dims:
            return None, None
        if len(dims) == 1:
            return dims[0], self._table(dims[0])
        # Several widths means a model changed without a recompile; prefer the largest
        # populated one rather than guessing silently.
        best, best_n = None, -1
        for d in dims:
            t = self._table(d)
            try:
                n = t.count_rows() if t is not None else 0
            except Exception:
                n = 0
            if n > best_n:
                best, best_n = d, n
        # Parenthesised deliberately: `return best, x if best else (None, None)` binds
        # the conditional to the second element only, so a falsy `best` returned a
        # TUPLE as the table handle — truthy, so it sailed past every `if tbl is None`
        # guard and blew up on the first `.search()`.
        if best is None:
            return None, None
        return best, self._table(best)

    # -- writes ------------------------------------------------------------
    def replace_wave(self, source_type: str, source_id: str, delete_item_ids: list,
                     rows: list) -> None:
        """Delete the named items' existing chunks and insert ``rows``."""
        import pyarrow as pa
        dim = None
        for r in rows:
            if r.get("vector"):
                dim = len(r["vector"])
                break
        if dim is None:
            dim = self._existing_dims()[0] if self._existing_dims() else _FALLBACK_DIM
        tbl = self._table(dim, create=True)

        with _LOCK:
            if delete_item_ids:
                inner = ", ".join(_sql_str(i) for i in delete_item_ids)
                tbl.delete(f"source_type = {_sql_str(source_type)} AND "
                           f"source_id = {_sql_str(source_id)} AND item_id IN ({inner})")
            if rows:
                cols = {
                    "id": [r["id"] for r in rows],
                    "source_type": [r["source_type"] for r in rows],
                    "source_id": [r["source_id"] for r in rows],
                    "item_id": [r["item_id"] for r in rows],
                    "chunk_index": [int(r["chunk_index"]) for r in rows],
                    # None (not []) for a missing embedding — see the schema comment.
                    "vector": [([float(x) for x in r["vector"]] if r.get("vector") else None)
                               for r in rows],
                    "content": [r["content"] or "" for r in rows],
                    "context": [r["context"] or "" for r in rows],
                    "meta": [r["meta"] or "{}" for r in rows],
                    "content_hash": [r["content_hash"] or "" for r in rows],
                    "embed_hash": [r["embed_hash"] or "" for r in rows],
                    "model": [r["model"] or "" for r in rows],
                    "updated": [r["updated"] or "" for r in rows],
                }
                tbl.add(pa.Table.from_pydict(cols, schema=self._schema(dim)))
            # New rows invalidate the FTS index until it is rebuilt in optimize().
            self._fts_ready.discard(dim)

    def delete_items(self, source_type: str, source_id: str, item_ids: list) -> None:
        ids = [i for i in (item_ids or []) if i]
        if not ids:
            return
        inner = ", ".join(_sql_str(i) for i in ids)
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            with _LOCK:
                tbl.delete(f"source_type = {_sql_str(source_type)} AND "
                           f"source_id = {_sql_str(source_id)} AND item_id IN ({inner})")

    def delete_source(self, source_id: str) -> None:
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            with _LOCK:
                tbl.delete(f"source_id = {_sql_str(source_id)}")

    def prune(self, source_type: str, source_id: str, keep_item_ids: list) -> int:
        keep = [i for i in (keep_item_ids or []) if i]
        removed = 0
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            with _LOCK:
                before = self._count_in(tbl, source_type, source_id)
                if keep:
                    inner = ", ".join(_sql_str(i) for i in keep)
                    tbl.delete(f"source_type = {_sql_str(source_type)} AND "
                               f"source_id = {_sql_str(source_id)} AND "
                               f"item_id NOT IN ({inner})")
                else:
                    tbl.delete(f"source_type = {_sql_str(source_type)} AND "
                               f"source_id = {_sql_str(source_id)}")
                removed += before - self._count_in(tbl, source_type, source_id)
        return int(removed)

    def optimize(self) -> None:
        """Build/refresh the vector and full-text indexes. Called once after a compile
        rather than per wave, because index maintenance is the expensive part."""
        from lancedb.index import FTS
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            try:
                n = tbl.count_rows()
            except Exception:
                continue
            if not n:
                continue
            with _LOCK:
                try:
                    tbl.optimize()
                except Exception:
                    pass
                if dim not in self._fts_ready:
                    try:
                        tbl.create_index("content", config=FTS(), replace=True)
                        self._fts_ready.add(dim)
                    except Exception:
                        pass        # keyword search falls back to Python BM25
                if n >= _ANN_MIN_ROWS and dim not in self._ann_ready:
                    try:
                        # Unified config API (lancedb >= 0.25); the metric= form is
                        # deprecated. Fall back for older versions.
                        try:
                            from lancedb.index import IvfPq
                            tbl.create_index("vector",
                                             config=IvfPq(distance_type="cosine"),
                                             replace=True)
                        except (ImportError, TypeError):
                            tbl.create_index(metric="cosine",
                                             vector_column_name="vector", replace=True)
                        self._ann_ready.add(dim)
                    except Exception:
                        pass        # brute-force vector search still works

    # -- reads -------------------------------------------------------------
    @staticmethod
    def _count_in(tbl, source_type: str, source_id: str) -> int:
        try:
            return int(tbl.count_rows(
                f"source_type = {_sql_str(source_type)} AND source_id = {_sql_str(source_id)}"))
        except Exception:
            return 0

    def count(self, source_type: str, source_id: str) -> int:
        total = 0
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is not None:
                total += self._count_in(tbl, source_type, source_id)
        return total

    def hashes(self, source_type: str, source_id: str, item_id: str, model: str) -> list:
        out = []
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            try:
                res = (tbl.search()
                       .where(f"source_type = {_sql_str(source_type)} AND "
                              f"source_id = {_sql_str(source_id)} AND "
                              f"item_id = {_sql_str(item_id)} AND "
                              f"model = {_sql_str(model)}")
                       .select(["chunk_index", "content_hash"])
                       .limit(1_000_000).to_arrow())
            except Exception:
                continue
            pairs = sorted(zip(res.column("chunk_index").to_pylist(),
                               res.column("content_hash").to_pylist()))
            out.extend(h for _i, h in pairs)
        return out

    def list_items(self, source_type: str, source_id: str) -> list:
        counts = {}
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            try:
                res = (tbl.search()
                       .where(f"source_type = {_sql_str(source_type)} AND "
                              f"source_id = {_sql_str(source_id)}")
                       .select(["item_id"]).limit(1_000_000).to_arrow())
            except Exception:
                continue
            for iid in res.column("item_id").to_pylist():
                counts[iid] = counts.get(iid, 0) + 1
        return [{"item_id": k, "chunks": v} for k, v in counts.items()]

    def cached_vectors(self, model: str, embed_hashes: list) -> dict:
        """{embed_hash: vector} for inputs already embedded under ``model``."""
        uniq = [h for h in dict.fromkeys(embed_hashes) if h]
        if not uniq:
            return {}
        found = {}
        step = 2000
        for dim in self._existing_dims():
            tbl = self._table(dim)
            if tbl is None:
                continue
            for i in range(0, len(uniq), step):
                part = uniq[i:i + step]
                inner = ", ".join(_sql_str(h) for h in part)
                try:
                    res = (tbl.search()
                           .where(f"model = {_sql_str(model)} AND embed_hash IN ({inner})")
                           .select(["embed_hash", "vector"])
                           .limit(len(part)).to_arrow())
                except Exception:
                    continue
                for h, v in zip(res.column("embed_hash").to_pylist(),
                                res.column("vector").to_pylist()):
                    if h and v and h not in found:
                        found[h] = [float(x) for x in v]
        return found

    # -- search ------------------------------------------------------------
    @staticmethod
    def _rows_from(res, score_col: str, invert: bool) -> list:
        ids = res.column("id").to_pylist()
        contents = res.column("content").to_pylist()
        sids = res.column("source_id").to_pylist()
        iids = res.column("item_id").to_pylist()
        metas = res.column("meta").to_pylist() if "meta" in res.schema.names else [""] * len(ids)
        if score_col in res.schema.names:
            raw = res.column(score_col).to_pylist()
        else:
            raw = [0.0] * len(ids)
        out = []
        for i in range(len(ids)):
            s = raw[i] if raw[i] is not None else 0.0
            # _distance is smaller-is-better; convert to a similarity so callers and
            # RRF see the same orientation as the DuckDB backend.
            out.append({"content": contents[i], "source_id": sids[i], "item_id": iids[i],
                        "id": ids[i], "meta": metas[i],
                        "score": (1.0 - float(s)) if invert else float(s)})
        return out

    def search_vector(self, source_type: str, ids: list, model: str, qv: list,
                      top_k: int) -> list:
        dim = len(qv or [])
        tbl = self._table(dim) if dim else None
        if tbl is None:
            return []
        try:
            # metric("cosine") is NOT optional: Lance defaults to L2, which ranks
            # differently from the cosine similarity the rest of the app (and the
            # DuckDB backend) uses, so omitting it silently changes retrieval results.
            res = (tbl.search([float(x) for x in qv])
                   .metric("cosine")
                   .where(_scope_predicate(source_type, ids, model))
                   .limit(int(top_k)).to_arrow())
        except Exception:
            return []
        # Lance reports cosine DISTANCE (1 - similarity); flip it so scores share the
        # DuckDB backend's orientation (higher = better).
        return self._rows_from(res, "_distance", invert=True)

    def _has_fts(self, dim: int, tbl) -> bool:
        """Whether ``tbl`` has a usable full-text index on ``content``.

        ``_fts_ready`` is only ever populated by ``optimize()``, i.e. by a compile in
        THIS process. The index itself is durable, so trusting the in-memory set alone
        meant that after a restart with no recompile every keyword and hybrid search
        silently fell back to the Python BM25 full scan — the slow path this backend
        exists to avoid. So ask the table once and remember the answer."""
        if dim in self._fts_ready:
            return True
        lister = getattr(tbl, "list_indices", None)
        if lister is None:
            return False
        try:
            found = False
            for idx in (lister() or []):
                cols = (getattr(idx, "columns", None)
                        or (idx.get("columns") if isinstance(idx, dict) else None) or [])
                kind = str(getattr(idx, "index_type", "")
                           or (idx.get("index_type") if isinstance(idx, dict) else "")).upper()
                if "content" in cols and "FTS" in kind:
                    found = True
                    break
            if found:
                self._fts_ready.add(dim)
            return found
        except Exception:
            return False

    def search_keyword(self, source_type: str, ids: list, query: str, top_k: int) -> list:
        dim, tbl = self._any_table()
        if tbl is None:
            return []
        # No model filter: BM25 scores text, and chunk text does not depend on which
        # embedding model produced the vectors alongside it.
        pred = _scope_predicate(source_type, ids)
        if self._has_fts(dim, tbl):
            try:
                res = (tbl.search(query, query_type="fts")
                       .where(pred).limit(int(top_k)).to_arrow())
                return self._rows_from(res, "_score", invert=False)
            except Exception:
                pass    # fall through to BM25 below
        # No FTS index yet (or the query upset tantivy) — materialise and score in
        # Python, exactly as the DuckDB backend does. Correct, just not fast.
        try:
            res = (tbl.search().where(pred)
                   .select(["id", "content", "source_id", "item_id", "context", "meta"])
                   .limit(1_000_000).to_arrow())
        except Exception:
            return []
        cands = [{"id": a, "content": b, "source_id": c, "item_id": d,
                  "context": e, "meta": f}
                 for a, b, c, d, e, f in zip(res.column("id").to_pylist(),
                                             res.column("content").to_pylist(),
                                             res.column("source_id").to_pylist(),
                                             res.column("item_id").to_pylist(),
                                             res.column("context").to_pylist(),
                                             res.column("meta").to_pylist())]
        return scoring.bm25_search(cands, query, top_k)

    # -- diagnostics -------------------------------------------------------
    def status(self) -> dict:
        dims = self._existing_dims()
        rows = 0
        for d in dims:
            t = self._table(d)
            try:
                rows += t.count_rows() if t is not None else 0
            except Exception:
                pass
        return {"backend": self.name, "encrypted": False,
                "path": str(core.RAG_LANCE_DIR), "dims": dims, "rows": rows,
                "fts_indexed": sorted(self._fts_ready),
                "ann_indexed": sorted(self._ann_ready)}
