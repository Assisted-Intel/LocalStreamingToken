#!/usr/bin/env python3
"""
Streaming importer. Reads the user-selected rows from a source table in **bounded
chunks** (never the whole table into RAM) and yields them for the staging layer to
persist into DuckDB. Row-selection modes: first N / random sample N / full table /
custom WHERE.

Identifiers (table name) are quoted with the dialect's preparer. WHERE / ORDER BY
fragments are passed through verbatim — this is a local, single-user tool pointed
at the user's own database, so raw SQL fragments are trusted by design.

Phase status: relational streaming implemented in Phase 3; Mongo in Phase 9.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .connections import ConnectionManager
from .models import ConnectionProfile, SelectionMode, SelectionSpec
from .types_map import duckdb_type_for

# Rows pulled per round-trip. Modest for memory; the route may override.
DEFAULT_CHUNK = 1000

# Keys per WHERE ... OR ... batch in stream_by_keys. Each key costs one bound
# parameter per key column, so 200 keys stays far inside every driver's limit
# (SQLite's default SQLITE_MAX_VARIABLE_NUMBER is the tightest at 999).
KEY_BATCH = 200

# Dialect-specific RANDOM() function for random sampling.
_RANDOM_FN = {"sqlite": "RANDOM()", "postgresql": "random()", "mysql": "RAND()",
              "mssql": "NEWID()", "oracle": "DBMS_RANDOM.VALUE"}


class StreamingImporter:
    def __init__(self, conns: Optional[ConnectionManager] = None):
        self.conns = conns or ConnectionManager()

    # ------------------------------------------------------------------ #
    def _qtable(self, engine, table: str) -> str:
        return engine.dialect.identifier_preparer.quote(table)

    def _where_clause(self, selection: SelectionSpec) -> str:
        w = (selection.where or "").strip()
        return f" WHERE {w}" if w else ""

    def build_select(self, engine, table: str, selection: SelectionSpec) -> str:
        """Render the SELECT for a selection spec (dialect-aware)."""
        qt = self._qtable(engine, table)
        where = self._where_clause(selection)
        order = f" ORDER BY {selection.order_by.strip()}" if (selection.order_by or "").strip() else ""
        mode = selection.mode
        n = max(1, int(selection.n or 0))
        eng = engine.dialect.name

        if mode == SelectionMode.RANDOM_N.value:
            rnd = _RANDOM_FN.get(eng, "RANDOM()")
            return f"SELECT * FROM {qt}{where} ORDER BY {rnd} LIMIT {n}"
        if mode == SelectionMode.FIRST_N.value:
            return f"SELECT * FROM {qt}{where}{order} LIMIT {n}"
        # FULL / CUSTOM -> everything matching (custom supplies the WHERE).
        return f"SELECT * FROM {qt}{where}{order}"

    def count_rows(self, profile: ConnectionProfile, table: str,
                   selection: SelectionSpec) -> int:
        """Effective import size for the progress bar."""
        from sqlalchemy import text
        eng = self.conns.build_engine(profile, readonly=True)
        try:
            qt = self._qtable(eng, table)
            where = self._where_clause(selection)
            with eng.connect() as conn:
                total = conn.execute(text(f"SELECT COUNT(*) FROM {qt}{where}")).scalar_one()
            if selection.mode in (SelectionMode.FIRST_N.value, SelectionMode.RANDOM_N.value):
                return min(int(total), max(1, int(selection.n or 0)))
            return int(total)
        finally:
            eng.dispose()

    def capture_columns(self, profile: ConnectionProfile, table: str) -> list:
        """Return [{name, source_type, duckdb_type, primary_key, nullable}] with
        types preserved via types_map (uses the live type object for DECIMAL
        precision, falling back to the stringified type)."""
        from sqlalchemy import inspect
        eng = self.conns.build_engine(profile, readonly=True)
        try:
            insp = inspect(eng)
            try:
                pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            except Exception:
                pk = set()
            out = []
            for c in insp.get_columns(table):
                t = c.get("type")
                out.append({
                    "name": c["name"],
                    "source_type": str(t),
                    "duckdb_type": duckdb_type_for(t),
                    "primary_key": c["name"] in pk,
                    "nullable": bool(c.get("nullable", True)),
                })
            return out
        finally:
            eng.dispose()

    def stream_by_keys(self, profile: ConnectionProfile, table: str, key_columns: list,
                       keys: list, chunk: int = KEY_BATCH) -> Iterator[list]:
        """Yield lists of row-dicts for EXACTLY the rows named by ``keys``.

        ``keys`` is [{key_col: value}]. Used by conflict detection, which must ask
        about the rows that were actually staged rather than re-running the session's
        selection — a ``random_n`` selection returns a different sample every time it
        runs, so re-issuing it compares the baseline against unrelated rows.

        Emits batched ``WHERE (k1 = :p0_0 AND k2 = :p0_1) OR (...)`` reads. Values are
        bound as parameters (never interpolated) and identifiers go through the
        dialect preparer. ``chunk`` bounds keys per round-trip, keeping the statement
        inside driver parameter limits."""
        from sqlalchemy import text
        if not key_columns or not keys:
            return
        eng = self.conns.build_engine(profile, readonly=True)
        try:
            qt = self._qtable(eng, table)
            prep = eng.dialect.identifier_preparer
            qcols = [prep.quote(c) for c in key_columns]
            with eng.connect() as conn:
                for start in range(0, len(keys), chunk):
                    batch = keys[start:start + chunk]
                    clauses, params = [], {}
                    for i, key in enumerate(batch):
                        parts = []
                        for j, kc in enumerate(key_columns):
                            p = f"p{i}_{j}"
                            parts.append(f"{qcols[j]} = :{p}")
                            params[p] = key.get(kc)
                        clauses.append("(" + " AND ".join(parts) + ")")
                    sql = f"SELECT * FROM {qt} WHERE {' OR '.join(clauses)}"
                    result = conn.execute(text(sql), params)
                    rows = [dict(r) for r in result.mappings()]
                    if rows:
                        yield rows
        finally:
            eng.dispose()

    def stream(self, profile: ConnectionProfile, table: str, selection: SelectionSpec,
               chunk: int = DEFAULT_CHUNK) -> Iterator[list]:
        """Yield lists of row-dicts (one list per chunk) using a server-side cursor
        (``stream_results=True`` + ``yield_per``) so large tables never fully load."""
        from sqlalchemy import text
        eng = self.conns.build_engine(profile, readonly=True)
        try:
            sql = self.build_select(eng, table, selection)
            with eng.connect().execution_options(stream_results=True, yield_per=chunk) as conn:
                result = conn.execute(text(sql))
                for partition in result.mappings().partitions(chunk):
                    yield [dict(r) for r in partition]
        finally:
            eng.dispose()
