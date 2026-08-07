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
  * An UPDATE whose WHERE matches nothing is NOT counted as applied. It executes
    cleanly, so counting it would retire the pending edit (silently losing it) and
    log a write that never happened.

Whatever happens, the run ends with exactly one ``done`` frame carrying
``applied_pairs`` — the cells that actually committed. The caller retires those from
staging and re-baselines them; skipping the frame on an error would leave rows that
DID commit marked dirty, and the next conflict check would read our own writes as
third-party drift and refuse to ever write again.

Emits ``(kind, data)`` progress tuples like ``generate_one`` so the SSE route can
forward them: guard / progress / applied / warn / error / done.

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


# Audit entries buffered before a flush. Each flush rewrites and re-encrypts the
# whole log file, so flushing per row made a large row-mode run quadratic.
AUDIT_FLUSH = 250


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

        # Everything that can fail BEFORE the writable engine exists happens first, so
        # no failure path can leak it. (The schema read and the conflict re-check both
        # open their own short-lived read-only engines.)
        try:
            source_cols = {c["name"] for c in self.conns.list_columns(profile, table)}
        except Exception as e:
            yield ("error", {"message": f"Could not read source schema: {e}"})
            return

        rows, skipped = group_row_updates(changes, source_cols)
        if not rows:
            yield ("done", {"applied_rows": 0, "applied_cells": 0,
                            "skipped_columns": skipped, "stopped": False, "applied_pairs": [],
                            "message": "Nothing to write (no changed source columns)."})
            return

        # Conflict re-check immediately before writing.
        try:
            report = ConflictDetector(self.conns).check(profile, session)
        except Exception as e:
            yield ("error", {"message": f"Could not re-check the source for conflicts: {e}"})
            return
        conflicted = set(report.changed) | set(report.deleted)
        yield ("guard", {**report.to_dict(), "on_conflict": on_conflict})
        if report.has_conflict:
            if on_conflict == "abort":
                yield ("error", {"message": "Source changed since import — write-back aborted. "
                                            "Re-check conflicts and choose skip/overwrite.",
                                 "conflict": report.to_dict()})
                return
            if on_conflict == "skip":
                rows = {rid: info for rid, info in rows.items()
                        if _key_json(info["src_key"]) not in conflicted}
                if not rows:
                    yield ("done", {"applied_rows": 0, "applied_cells": 0,
                                    "skipped_columns": skipped, "stopped": False,
                                    "applied_pairs": [],
                                    "message": "Every changed row conflicts; nothing written."})
                    return

        # From here on the source is open WRITABLE — the only place in the app that is
        # true. The try/finally below guarantees it is disposed.
        eng = self.conns.build_engine(profile, readonly=False)
        prep = eng.dialect.identifier_preparer
        qtable = prep.quote(table)
        # executemany rowcount is only meaningful on dialects that report it; pysqlite
        # does not, so a bulk short-count check there would be pure noise.
        multi_rowcount = bool(getattr(eng.dialect, "supports_sane_multi_rowcount", False))

        def build_sql(cols_order):
            set_sql = ", ".join(f"{prep.quote(c)} = :v{j}" for j, c in enumerate(cols_order))
            where_sql = " AND ".join(f"{prep.quote(kc)} = :k{k}" for k, kc in enumerate(key_columns))
            return f"UPDATE {qtable} SET {set_sql} WHERE {where_sql}"

        def row_params(cols_order, col_map, src_key):
            p = {f"v{j}": col_map[c] for j, c in enumerate(cols_order)}
            for k, kc in enumerate(key_columns):
                p[f"k{k}"] = src_key.get(kc)
            return p

        # Each append rewrites (and re-encrypts) the whole log file, so appending once
        # per row made a large row-mode run quadratic in bytes written. Buffer instead
        # and flush in batches — still only ever after the owning transaction committed.
        buffered: list = []

        def audit_row(rowid, cols_order, col_map, src_key, sql, status="ok", err=""):
            buffered.extend(
                AuditEntry(session_id=session["id"], table=table, key=src_key, column=c,
                           old=olds.get((rowid, c)), new=col_map[c], statement=sql,
                           mode=mode, status=status, error=err)
                for c in cols_order)
            if len(buffered) >= AUDIT_FLUSH:
                flush_audit()

        def flush_audit():
            if buffered:
                audit.append_many(buffered)
                buffered.clear()

        total = len(rows)
        applied_rows = applied_cells = 0
        stopped = False
        partial = False
        error_message = ""
        # Rows whose UPDATE executed fine but matched zero source rows — nothing written.
        unmatched = 0
        # (rowid, column) pairs that reached the source and COMMITTED, so the caller can
        # retire them from the staging table's change log.
        applied_pairs: list = []
        try:
            if mode == WriteMode.ROW.value:
                # One transaction per row; continue-on-error optional. Each row is
                # audited only after its own transaction has committed.
                for i, (rowid, info) in enumerate(rows.items()):
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        break
                    cols_order = list(info["cols"].keys())
                    sql = build_sql(cols_order)
                    params = row_params(cols_order, info["cols"], info["src_key"])
                    try:
                        with eng.begin() as conn:
                            matched = conn.execute(text(sql), params).rowcount
                        if matched == 0:
                            # The statement succeeded but its WHERE matched no source
                            # row — the row was deleted upstream, or its key no longer
                            # compares equal. Counting this as applied would retire the
                            # pending edit (silently discarding it) and record a write
                            # in the audit log that never happened.
                            audit_row(rowid, cols_order, info["cols"], info["src_key"], sql,
                                      status="nomatch",
                                      err="No source row matched the key; nothing written.")
                            unmatched += 1
                            yield ("warn", {"rowid": rowid, "message":
                                            f"Key {_key_json(info['src_key'])} matched no "
                                            "source row — that edit was NOT written."})
                        else:
                            audit_row(rowid, cols_order, info["cols"], info["src_key"], sql)
                            applied_rows += 1
                            applied_cells += len(cols_order)
                            applied_pairs.extend((rowid, c) for c in cols_order)
                            yield ("applied", {"rowid": rowid, "cells": len(cols_order)})
                    except Exception as e:
                        audit_row(rowid, cols_order, info["cols"], info["src_key"], sql,
                                  status="error", err=str(e))
                        yield ("error", {"rowid": rowid, "message": str(e)})
                        if not continue_on_error:
                            # Break, never return. Earlier rows committed in their own
                            # transactions and are already in the source; returning here
                            # skipped the `done` frame, so the route never learned which
                            # cells to retire or re-baseline. The session then stayed
                            # dirty and its own writes came back as third-party drift on
                            # the next conflict check, locking it out of writing again.
                            partial = True
                            error_message = str(e)
                            break
                    yield ("progress", {"done": i + 1, "total": total})
            else:
                # BULK: group by changed-column set, executemany per group in ONE txn.
                #
                # Nothing is audited or counted from inside that transaction. A stop (or
                # any error) rolls the whole thing back, so writing the audit log as we
                # went would record rows as applied that the database never kept — and
                # the audit log is the record of what touched the user's real data.
                # Progress frames still stream live; they are a bar, not a claim.
                groups: dict = {}
                for rowid, info in rows.items():
                    groups.setdefault(tuple(sorted(info["cols"].keys())), []).append((rowid, info))
                done = 0
                pending = []            # (rowid, cols_order, col_map, src_key, sql)
                short = 0               # rows the driver says no statement matched
                committed = False
                try:
                    with eng.begin() as conn:
                        for cols_order, members in groups.items():
                            cols_order = list(cols_order)
                            sql = build_sql(cols_order)
                            for start in range(0, len(members), batch_size):
                                if stop_event is not None and stop_event.is_set():
                                    stopped = True
                                    raise _Stop()
                                batch = members[start:start + batch_size]
                                matched = conn.execute(
                                    text(sql),
                                    [row_params(cols_order, info["cols"], info["src_key"])
                                     for _, info in batch]).rowcount
                                if multi_rowcount and matched is not None and matched >= 0:
                                    short += max(0, len(batch) - matched)
                                    unmatched += max(0, len(batch) - matched)
                                pending.extend((rowid, cols_order, info["cols"],
                                                info["src_key"], sql)
                                               for rowid, info in batch)
                                done += len(batch)
                                yield ("progress", {"done": done, "total": total})
                    committed = True
                finally:
                    if committed:
                        for rowid, cols_order, col_map, src_key, sql in pending:
                            audit_row(rowid, cols_order, col_map, src_key, sql)
                            applied_rows += 1
                            applied_cells += len(cols_order)
                            applied_pairs.extend((rowid, c) for c in cols_order)
                if committed and short:
                    # Bulk is one transaction, so we cannot tell WHICH rows missed
                    # without re-querying; say plainly that the counts are optimistic
                    # rather than reporting a clean run. Row mode names each one.
                    yield ("warn", {"message":
                                    f"{short} of {total} row(s) matched no source row — "
                                    "those edits were not written. Re-check conflicts, or "
                                    "use row-by-row mode to see exactly which."})
        except _Stop:
            pass  # stop_event during bulk — the txn rolled back, nothing was audited
        except Exception as e:
            flush_audit()
            yield ("error", {"message": str(e)})
            return
        finally:
            eng.dispose()

        flush_audit()
        yield ("done", {"applied_rows": applied_rows, "applied_cells": applied_cells,
                        "skipped_columns": skipped, "stopped": stopped,
                        "partial": partial, "error": error_message,
                        "unmatched_rows": unmatched,
                        "applied_pairs": applied_pairs, "audited": audit.count()})


class _Stop(Exception):
    """Internal signal to unwind the bulk transaction on cancellation."""
