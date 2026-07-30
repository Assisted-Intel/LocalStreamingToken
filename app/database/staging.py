#!/usr/bin/env python3
"""
DuckDB staging layer — the isolated local copy the user edits and AI-enriches. One
DuckDB file per import session (``data/db/staging/staging_<id>.duckdb``), fully
separate from the source so the source is never touched until write-back.

Bookkeeping columns on every staging table:
  __rowid     stable integer key for the staged row (cell edits / write-back)
  __src_key   JSON of the source primary-key values (write-back matching)
  __row_hash  fingerprint of the imported source row (conflict detection)
  __dirty     1 if any cell changed locally (manual edit or AI output)
A side table ``__cell_orig(rowid, col, old)`` captures each cell's ORIGINAL value
the first time it is changed, so dry-run and audit can show old→new.

DuckDB is imported lazily. One shared connection per file is cached process-wide and
guarded by a lock (this is a single-user local app; that keeps writes serialized and
avoids cross-request conflicts on the same file).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from decimal import Decimal
from typing import Iterator, Optional

from .. import core

BOOKKEEPING = ("__rowid", "__src_key", "__row_hash", "__dirty")

# Process-wide connection cache: path -> (duckdb connection, reentrant lock).
_CONNS: dict = {}
_REG_LOCK = threading.Lock()


def staging_path(session_id: str):
    return core.DB_STAGING_DIR / f"staging_{session_id}.duckdb"


def _get_conn(path):
    key = str(path)
    with _REG_LOCK:
        entry = _CONNS.get(key)
        if entry is None:
            entry = (core.duckdb_connect(path), threading.RLock())
            _CONNS[key] = entry
        return entry


def close_all():
    """Close every cached staging connection and clear the cache. Called when the
    active data profile changes so the previous profile's ``staging_*.duckdb`` files
    are released before their paths are repointed."""
    with _REG_LOCK:
        for conn, _lock in _CONNS.values():
            try:
                conn.close()
            except Exception:
                pass
        _CONNS.clear()


def _qi(name: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _jsonsafe(v):
    """Make a value JSON-serialisable for the browser grid (types are still fully
    preserved inside DuckDB — this is display only)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (date, datetime, Decimal)):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return bytes(v).decode("utf-8", "replace")
        except Exception:
            return str(v)
    return str(v)


def _coerce_in(v):
    """Coerce a source value for DuckDB insertion (dict/list -> JSON text)."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return v


def compute_src_key(row: dict, key_columns: list, rowid=None) -> str:
    """Stable JSON identity for a source row. Uses the primary-key columns when
    available, else falls back to the staging __rowid. Must be computed identically
    at import and at conflict-check time."""
    if key_columns:
        return json.dumps({k: row.get(k) for k in key_columns}, sort_keys=True, default=str)
    return json.dumps({"__rowid": rowid})


def compute_row_hash(row: dict, source_cols: list) -> str:
    """Fingerprint of a source row over its source columns (order-independent),
    used for conflict detection. Shared by import and conflict-check."""
    return hashlib.sha1(
        json.dumps({c: row.get(c) for c in source_cols}, sort_keys=True, default=str)
        .encode("utf-8")).hexdigest()


class StagingManager:
    def __init__(self, session_id: str, staging_table: str = "staged"):
        self.session_id = session_id
        self.path = staging_path(session_id)
        self.staging_table = staging_table
        self._next_rowid: Optional[int] = None

    # ------------------------------ connection ------------------------------
    def _conn(self):
        return _get_conn(self.path)

    def _display_columns(self, con) -> list:
        """All non-bookkeeping columns (source + AI-output), in table order."""
        rows = con.execute(f"PRAGMA table_info({_qi(self.staging_table)})").fetchall()
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
        return [r[1] for r in rows if not str(r[1]).startswith("__")]

    # ------------------------------ lifecycle ------------------------------
    def create_table(self, columns: list) -> None:
        """CREATE the staging table (dropping any prior) with DuckDB types + the
        bookkeeping columns and the __cell_orig side table."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        col_defs = ", ".join(f"{_qi(c['name'])} {c.get('duckdb_type', 'VARCHAR')}" for c in columns)
        with lock:
            con.execute(f"DROP TABLE IF EXISTS {st}")
            con.execute("DROP TABLE IF EXISTS __cell_orig")
            con.execute(
                f"CREATE TABLE {st} ("
                f"__rowid BIGINT, __src_key VARCHAR, __row_hash VARCHAR, __dirty INTEGER DEFAULT 0"
                + (", " + col_defs if col_defs else "") + ")")
            con.execute("CREATE TABLE __cell_orig (rowid BIGINT, col VARCHAR, old VARCHAR)")
        self._next_rowid = 0

    def insert_chunk(self, rows: list, columns: list, key_columns: list) -> int:
        """Append a chunk of source rows, computing __src_key + __row_hash and a
        sequential __rowid. Returns the number of rows inserted."""
        if not rows:
            return 0
        con, lock = self._conn()
        src_cols = [c["name"] for c in columns]
        st = _qi(self.staging_table)
        placeholders = ", ".join(["?"] * (4 + len(src_cols)))
        collist = ", ".join([_qi("__rowid"), _qi("__src_key"), _qi("__row_hash"), _qi("__dirty")]
                            + [_qi(c) for c in src_cols])
        sql = f"INSERT INTO {st} ({collist}) VALUES ({placeholders})"

        with lock:
            if self._next_rowid is None:
                mx = con.execute(f"SELECT COALESCE(MAX(__rowid), -1) FROM {st}").fetchone()[0]
                self._next_rowid = int(mx) + 1
            data = []
            for row in rows:
                rid = self._next_rowid
                self._next_rowid += 1
                src_key = compute_src_key(row, key_columns, rid)
                row_hash = compute_row_hash(row, src_cols)
                data.append([rid, src_key, row_hash, 0] + [_coerce_in(row.get(c)) for c in src_cols])
            con.executemany(sql, data)
        return len(rows)

    def row_count(self) -> int:
        con, lock = self._conn()
        with lock:
            return int(con.execute(f"SELECT COUNT(*) FROM {_qi(self.staging_table)}").fetchone()[0])

    # ------------------------------ read (Phase 4) ------------------------------
    def get_page(self, offset: int = 0, limit: int = 100) -> dict:
        """Return {columns, rows, total} for a page of the staging table. Each row
        includes __rowid and __dirty plus the display columns (JSON-safe)."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        with lock:
            cols = self._display_columns(con)
            sel = ", ".join([_qi("__rowid"), _qi("__dirty")] + [_qi(c) for c in cols])
            cur = con.execute(
                f"SELECT {sel} FROM {st} ORDER BY __rowid LIMIT ? OFFSET ?", [int(limit), int(offset)])
            names = [d[0] for d in cur.description]
            fetched = cur.fetchall()
            total = self.row_count()
        rows = []
        for r in fetched:
            d = {names[i]: _jsonsafe(v) for i, v in enumerate(r)}
            rows.append(d)
        return {"columns": cols, "rows": rows, "total": total}

    def iter_rows(self, only_dirty: bool = False) -> Iterator[dict]:
        """Yield staged rows as dicts keyed by column name (display columns +
        __rowid) for AI processing / write-back. Part of the AI-integration surface."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        with lock:
            cols = self._display_columns(con)
            sel = ", ".join([_qi("__rowid")] + [_qi(c) for c in cols])
            where = " WHERE __dirty = 1" if only_dirty else ""
            cur = con.execute(f"SELECT {sel} FROM {st}{where} ORDER BY __rowid")
            names = [d[0] for d in cur.description]
            fetched = cur.fetchall()
        for r in fetched:
            yield {names[i]: v for i, v in enumerate(r)}

    # ------------------------------ write (Phase 4) ------------------------------
    def update_cell(self, rowid: int, column: str, value) -> dict:
        """Set one cell, capturing its ORIGINAL value once for dry-run/audit and
        marking __dirty. Returns {rowid, column, old, new}."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        rowid = int(rowid)
        with lock:
            cols = self._display_columns(con)
            if column not in cols:
                raise ValueError(f"Unknown column {column!r}.")
            # Capture original once.
            already = con.execute(
                "SELECT COUNT(*) FROM __cell_orig WHERE rowid = ? AND col = ?",
                [rowid, column]).fetchone()[0]
            cur_val = con.execute(
                f"SELECT {_qi(column)} FROM {st} WHERE __rowid = ?", [rowid]).fetchone()
            old = _jsonsafe(cur_val[0]) if cur_val else None
            if not already:
                con.execute("INSERT INTO __cell_orig (rowid, col, old) VALUES (?, ?, ?)",
                            [rowid, column, None if old is None else str(old)])
            con.execute(
                f"UPDATE {st} SET {_qi(column)} = ?, __dirty = 1 WHERE __rowid = ?",
                [value, rowid])
        return {"rowid": rowid, "column": column, "old": old, "new": _jsonsafe(value)}

    def add_column(self, name: str, duckdb_type: str = "VARCHAR") -> None:
        """Add a new (e.g. AI-output) column to the staging table (idempotent)."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        with lock:
            existing = {r[1] for r in con.execute(f"PRAGMA table_info({st})").fetchall()}
            if name in existing:
                return
            con.execute(f"ALTER TABLE {st} ADD COLUMN {_qi(name)} {duckdb_type}")

    def dirty_changes(self) -> list:
        """Return [{rowid, src_key, column, old, new}] for every changed cell — the
        input to the dry-run and write-back engines. One query per changed column."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        out = []
        with lock:
            changed_cols = [r[0] for r in con.execute(
                "SELECT DISTINCT col FROM __cell_orig").fetchall()]
            for col in changed_cols:
                cur = con.execute(
                    f"SELECT s.__rowid, s.__src_key, o.old, s.{_qi(col)} "
                    f"FROM __cell_orig o JOIN {st} s ON o.rowid = s.__rowid "
                    f"WHERE o.col = ?", [col])
                for rid, src_key, old, new in cur.fetchall():
                    try:
                        parsed_key = json.loads(src_key) if src_key else {}
                    except Exception:
                        parsed_key = {}
                    out.append({"rowid": int(rid), "src_key": parsed_key, "column": col,
                                "old": old, "new": _jsonsafe(new)})
        return out

    def fingerprints(self) -> dict:
        """Return {src_key: row_hash} captured at import — the baseline for conflict
        detection against the current source."""
        con, lock = self._conn()
        st = _qi(self.staging_table)
        with lock:
            rows = con.execute(f"SELECT __src_key, __row_hash FROM {st}").fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------ teardown ------------------------------
    def close(self) -> None:
        key = str(self.path)
        with _REG_LOCK:
            entry = _CONNS.pop(key, None)
        if entry:
            try:
                entry[0].close()
            except Exception:
                pass

    def drop(self) -> None:
        """Delete the staging DuckDB file (on session delete)."""
        self.close()
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
