#!/usr/bin/env python3
"""
Local Streaming Token — copy an existing DuckDB vector store into LanceDB.

Re-embedding a shelf of ebooks costs real time and, on a metered embedding host, real
money — so switching backends copies the vectors across rather than recomputing them.

The DuckDB file is **left exactly where it is**: it remains the DuckDB backend's live
store, so switching back in Settings → RAG is instant and keeps its existing data. This
is a copy, never a move.

Note the trade being made: the source is AES-encrypted, the destination is not. After
migrating, the same chunk text exists in plaintext on disk. That is the documented
purpose of the Lance backend, but it is worth being deliberate about.
"""

from .. import core
from . import DUCKDB, LANCE, get_backend


def _emit(emit, event, **data):
    if emit is not None:
        try:
            emit(event, data)
        except Exception:
            pass


def duckdb_row_count() -> int:
    """Rows in the DuckDB store (0 if it doesn't exist / can't be opened)."""
    try:
        from .duckdb_backend import _conn, _REG_LOCK
        conn = _conn()
        with _REG_LOCK:
            return int(conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
    except Exception:
        return 0


def migrate_duckdb_to_lance(emit=None, batch: int = 5000, stop_event=None) -> dict:
    """Copy every chunk from the DuckDB store into LanceDB.

    Streams ``progress`` frames shaped like the compile ones, so the existing progress
    bar and ETA widget drive it with no frontend changes. Returns a summary dict."""
    from .duckdb_backend import _conn, _REG_LOCK

    src = get_backend(DUCKDB)          # noqa: F841 — ensures schema/migration has run
    dst = get_backend(LANCE)

    conn = _conn()
    with _REG_LOCK:
        total = int(conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
    _emit(emit, "begin", total=total, name="DuckDB → LanceDB")
    _emit(emit, "plan", phases=[{"id": "migrate", "label": "Copying vectors",
                                 "unit": "chunks", "weight": 1.0}],
          totals={"chunks": total})

    if not total:
        _emit(emit, "complete", moved=0, total=0, verified=True,
              message="Nothing to migrate — the DuckDB store is empty.")
        return {"moved": 0, "total": 0, "verified": True}

    moved = 0
    stopped = False
    offset = 0
    # Ordered by id so the pagination is stable across batches.
    while offset < total:
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        with _REG_LOCK:
            rows = conn.execute(
                "SELECT id, source_type, source_id, item_id, chunk_index, content, "
                "       content_hash, embedding, model, updated, context, meta, embed_hash "
                "FROM rag_chunks ORDER BY id LIMIT ? OFFSET ?",
                [batch, offset]).fetchall()
        if not rows:
            break
        payload = []
        for r in rows:
            payload.append({
                "id": r[0], "source_type": r[1], "source_id": r[2], "item_id": r[3],
                "chunk_index": int(r[4] or 0), "content": r[5] or "",
                "content_hash": r[6] or "",
                # An empty DuckDB FLOAT[] becomes a null Lance vector — same meaning:
                # stored for keyword search, never matches a vector query.
                "vector": [float(x) for x in (r[7] or [])],
                "model": r[8] or "", "updated": r[9] or "",
                "context": r[10] or "", "meta": r[11] or "{}",
                "embed_hash": r[12] or "",
            })
        # No delete list: this is a fresh copy into an empty destination, and passing
        # item ids here would delete rows an earlier batch just wrote.
        dst.replace_wave(payload[0]["source_type"], payload[0]["source_id"], [], payload)
        moved += len(payload)
        offset += len(rows)
        _emit(emit, "progress", phase="migrate", done=moved, total=total, unit="chunks")

    # Build the vector + full-text indexes once at the end.
    _emit(emit, "progress", phase="migrate", done=moved, total=total, unit="chunks")
    try:
        dst.optimize()
    except Exception as e:
        _emit(emit, "warn", name="index", message=str(e))

    # Verify by scope rather than in bulk: a global count can match while individual
    # libraries are wrong.
    verified, mismatches = True, []
    with _REG_LOCK:
        scopes = conn.execute(
            "SELECT source_type, source_id, COUNT(*) FROM rag_chunks "
            "GROUP BY source_type, source_id").fetchall()
    for st, sid, n in scopes:
        got = dst.count(st, sid)
        if got != int(n):
            verified = False
            mismatches.append({"source_type": st, "source_id": sid,
                               "duckdb": int(n), "lance": got})
    summary = {"moved": moved, "total": total, "verified": verified and not stopped,
               "stopped": stopped, "mismatches": mismatches,
               "duckdb_path": str(core.RAG_DB_FILE),
               "lance_path": str(core.RAG_LANCE_DIR)}
    if mismatches:
        _emit(emit, "warn", name="verify",
              message=f"{len(mismatches)} scope(s) have differing counts")
    _emit(emit, "complete", **summary)
    return summary
