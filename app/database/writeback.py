#!/usr/bin/env python3
"""
Write-back engine. Applies the staged changes to the source database inside a
transaction. This is the ONLY place the source is ever opened writable.

Two modes:
  * BULK (preferred) — group rows by their changed-column set and ``executemany``
    each group inside a single transaction (all-or-nothing).
  * ROW — one transaction per row, each audited individually, with optional
    continue-on-error.

Safety:
  * Requires ``approved`` (the route enforces it too).
  * A conflict re-check runs immediately before writing. If the source changed
    under us, behaviour follows ``on_conflict``: ``abort`` (default, write nothing),
    ``skip`` (write only non-conflicted rows), or ``overwrite`` (write anyway).
  * Only columns that exist in the source are written; staging-only enrichment
    columns are reported as skipped, never invented on the source.
  * Every applied cell is recorded in the append-only audit log.

Emits ``(kind, data)`` progress tuples like ``generate_one`` so the SSE route can
forward them: guard / progress / applied / error / done.

Phase status: implemented in Phase 8 (SQL); Mongo in Phase 9.
"""

from __future__ import annotations

import json
from typing import Iterator

from .audit import AuditLog
from .conflict import ConflictDetector
from .connections import ConnectionManager
from .dryrun import group_row_updates
from .models import AuditEntry, ConflictReport, WriteMode


def _key_json(src_key: dict) -> str:
    return json.dumps(src_key, sort_keys=True, default=str)


class WriteBackEngine:
    def __init__(self, conns: ConnectionManager | None = None):
        self.conns = conns or ConnectionManager()

    def run(self, profile, session: dict, changes: list, *,
            mode: str = WriteMode.BULK.value, approved: bool = False,
            on_conflict: str = "abort", continue_on_error: bool = False,
            batch_size: int = 500, stop_event=None) -> Iterator[tuple]:
        from sqlalchemy import text

        if not approved:
            yield ("error", {"message": "Write-back not approved."})
            return

        table = session["table"]
        key_columns = session.get("key_columns") or []
        if not key_columns:
            yield ("error", {"message": "No key columns — cannot match source rows for write-back."})
            return

        audit = AuditLog(session["id"])
        olds = {(ch["rowid"], ch["column"]): ch.get("old") for ch in changes}

        # Resolve source columns + build the row update set.
        eng = self.conns.build_engine(profile, readonly=False)
        prep = eng.dialect.identifier_preparer
        qtable = prep.quote(table)
        try:
            source_cols = {c["name"] for c in self.conns.list_columns(profile, table)}
        except Exception as e:
            eng.dispose()
            yield ("error", {"message": f"Could not read source schema: {e}"})
            return

        rows, skipped = group_row_updates(changes, source_cols)
        if not rows:
            eng.dispose()
            yield ("done", {"applied_rows": 0, "applied_cells": 0,
                            "skipped_columns": skipped, "stopped": False,
                            "message": "Nothing to write (no changed source columns)."})
            return

        # Conflict re-check immediately before writing.
        report = ConflictDetector(self.conns).check(profile, session)
        conflicted = set(report.changed) | set(report.deleted)
        yield ("guard", {**report.to_dict(), "on_conflict": on_conflict})
        if report.has_conflict:
            if on_conflict == "abort":
                eng.dispose()
                yield ("error", {"message": "Source changed since import — write-back aborted. "
                                            "Re-check conflicts and choose skip/overwrite.",
                                 "conflict": report.to_dict()})
                return
            if on_conflict == "skip":
                rows = {rid: info for rid, info in rows.items()
                        if _key_json(info["src_key"]) not in conflicted}

        def build_sql(cols_order):
            set_sql = ", ".join(f"{prep.quote(c)} = :v{j}" for j, c in enumerate(cols_order))
            where_sql = " AND ".join(f"{prep.quote(kc)} = :k{k}" for k, kc in enumerate(key_columns))
            return f"UPDATE {qtable} SET {set_sql} WHERE {where_sql}"

        def row_params(cols_order, col_map, src_key):
            p = {f"v{j}": col_map[c] for j, c in enumerate(cols_order)}
            for k, kc in enumerate(key_columns):
                p[f"k{k}"] = src_key.get(kc)
            return p

        def audit_row(rowid, cols_order, col_map, src_key, sql, status="ok", err=""):
            audit.append_many([
                AuditEntry(session_id=session["id"], table=table, key=src_key, column=c,
                           old=olds.get((rowid, c)), new=col_map[c], statement=sql,
                           mode=mode, status=status, error=err)
                for c in cols_order])

        total = len(rows)
        applied_rows = applied_cells = 0
        stopped = False
        try:
            if mode == WriteMode.ROW.value:
                # One transaction per row; continue-on-error optional.
                for i, (rowid, info) in enumerate(rows.items()):
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        break
                    cols_order = list(info["cols"].keys())
                    sql = build_sql(cols_order)
                    params = row_params(cols_order, info["cols"], info["src_key"])
                    try:
                        with eng.begin() as conn:
                            conn.execute(text(sql), params)
                        audit_row(rowid, cols_order, info["cols"], info["src_key"], sql)
                        applied_rows += 1
                        applied_cells += len(cols_order)
                        yield ("applied", {"rowid": rowid, "cells": len(cols_order)})
                    except Exception as e:
                        audit_row(rowid, cols_order, info["cols"], info["src_key"], sql,
                                  status="error", err=str(e))
                        yield ("error", {"rowid": rowid, "message": str(e)})
                        if not continue_on_error:
                            return
                    yield ("progress", {"done": i + 1, "total": total})
            else:
                # BULK: group by changed-column set, executemany per group in one txn.
                groups: dict = {}
                for rowid, info in rows.items():
                    groups.setdefault(tuple(sorted(info["cols"].keys())), []).append((rowid, info))
                done = 0
                with eng.begin() as conn:
                    for cols_order, members in groups.items():
                        cols_order = list(cols_order)
                        sql = build_sql(cols_order)
                        for start in range(0, len(members), batch_size):
                            if stop_event is not None and stop_event.is_set():
                                stopped = True
                                raise _Stop()
                            batch = members[start:start + batch_size]
                            conn.execute(text(sql),
                                         [row_params(cols_order, info["cols"], info["src_key"])
                                          for _, info in batch])
                            for rowid, info in batch:
                                audit_row(rowid, cols_order, info["cols"], info["src_key"], sql)
                                applied_cells += len(cols_order)
                            applied_rows += len(batch)
                            done += len(batch)
                            yield ("progress", {"done": done, "total": total})
        except _Stop:
            pass  # stop_event during bulk — the txn rolled back cleanly
        except Exception as e:
            yield ("error", {"message": str(e)})
            eng.dispose()
            return
        finally:
            eng.dispose()

        yield ("done", {"applied_rows": applied_rows, "applied_cells": applied_cells,
                        "skipped_columns": skipped, "stopped": stopped,
                        "audited": audit.count()})


class _Stop(Exception):
    """Internal signal to unwind the bulk transaction on cancellation."""
