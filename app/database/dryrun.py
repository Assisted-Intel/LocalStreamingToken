#!/usr/bin/env python3
"""
Dry-run engine. Renders the EXACT statements that write-back WOULD execute against
the source — both the parameterized form (``UPDATE t SET c=:p WHERE k=:k``) and a
human-readable literal-rendered preview — without touching the source. The preview
can be limited to a configurable number of rows and columns.

Only columns that actually exist in the source table are written; staging-only
enrichment columns (e.g. an AI ``Summary`` the source has no column for) are
reported as ``skipped_columns`` rather than silently dropped. Write-back needs a
key: with no primary key / key columns, no statement can safely match a source row,
so the result carries a warning and no statements.

Phase status: implemented in Phase 6. The row-grouping helper is shared with
write-back (Phase 8).
"""

from __future__ import annotations

from .connections import ConnectionManager
from .models import DryRunResult


def _literal(v) -> str:
    """Render a value as a SQL literal for the (display-only) preview."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def group_row_updates(changes: list, source_cols: set) -> tuple:
    """Group dirty changes into per-row updates, keeping only columns present in the
    source. Returns (rows_by_id, skipped_columns).

    rows_by_id: {rowid: {"src_key": {..}, "cols": {col: new}}}"""
    rows: dict = {}
    skipped = set()
    for ch in changes:
        col = ch["column"]
        if col not in source_cols:
            skipped.add(col)
            continue
        entry = rows.setdefault(ch["rowid"], {"src_key": ch.get("src_key") or {}, "cols": {}})
        entry["cols"][col] = ch["new"]
    return rows, sorted(skipped)


class DryRunEngine:
    def __init__(self, conns: ConnectionManager | None = None):
        self.conns = conns or ConnectionManager()

    def preview(self, profile, session: dict, changes: list, *,
                max_rows: int = 50, max_cols: int = 20) -> DryRunResult:
        table = session.get("table")
        key_columns = session.get("key_columns") or []

        eng = self.conns.build_engine(profile, readonly=True)
        dialect = eng.dialect
        prep = dialect.identifier_preparer
        try:
            source_cols = {c["name"] for c in self.conns.list_columns(profile, table)}
        finally:
            eng.dispose()

        rows, skipped = group_row_updates(changes, source_cols)
        warnings = []
        if not key_columns:
            warnings.append("The source table has no primary key / key columns, so write-back "
                            "cannot match rows. No statements generated.")
            return DryRunResult(statements=[], total_rows=len(rows), total_statements=0,
                                truncated=False, dialect=dialect.name,
                                skipped_columns=skipped, warnings=warnings)

        qtable = prep.quote(table)
        statements = []
        truncated = False
        for i, (rowid, info) in enumerate(rows.items()):
            if i >= max_rows:
                truncated = True
                break
            col_items = list(info["cols"].items())
            if len(col_items) > max_cols:
                truncated = True
                col_items = col_items[:max_cols]

            set_sql, set_lit, params = [], [], {}
            for j, (col, val) in enumerate(col_items):
                p = f"v{j}"
                set_sql.append(f"{prep.quote(col)} = :{p}")
                set_lit.append(f"{prep.quote(col)} = {_literal(val)}")
                params[p] = val

            skey = info["src_key"]
            where_sql, where_lit = [], []
            for k, kc in enumerate(key_columns):
                pk = f"k{k}"
                where_sql.append(f"{prep.quote(kc)} = :{pk}")
                where_lit.append(f"{prep.quote(kc)} = {_literal(skey.get(kc))}")
                params[pk] = skey.get(kc)

            sql = f"UPDATE {qtable} SET {', '.join(set_sql)} WHERE {' AND '.join(where_sql)}"
            literal = f"UPDATE {qtable} SET {', '.join(set_lit)} WHERE {' AND '.join(where_lit)}"
            statements.append({"sql": sql, "params": params, "literal": literal, "key": skey})

        return DryRunResult(statements=statements, total_rows=len(rows),
                            total_statements=len(statements), truncated=truncated,
                            dialect=dialect.name, skipped_columns=skipped, warnings=warnings)
