#!/usr/bin/env python3
"""
Conflict detection. At import we stored, per staged row, a ``__src_key`` (its
primary-key identity) and a ``__row_hash`` (a fingerprint of the imported source
row). To check for conflicts we re-read the current source rows, recompute the same
fingerprints, and diff:

  * key present now but absent at import      -> NEW in source
  * key present in both, hash differs         -> CHANGED in source
  * key present at import but absent now       -> DELETED from source
  * key present in both, hash equal            -> UNCHANGED

If the source changed underneath us we NEVER silently overwrite — the caller
surfaces the choice (process only new additions, or re-process + merge). Detection
requires primary-key columns; without them every row hashes to the same synthetic
key, so we return an empty report with an explanatory note instead.

How the "current" side is read depends on the session's selection mode. A FULL
import is re-scanned in one pass — that covers every staged key and additionally
reveals rows added since import. A SAMPLED import (first_n / random_n) must NOT be
re-run: ``random_n`` renders ``ORDER BY RANDOM() LIMIT n`` and draws a different
sample every call, so re-issuing the selection diffed the baseline against
unrelated rows and reported nearly all of them as deleted. Those modes re-read
exactly the staged keys instead (``StagingManager.staged_keys`` ->
``StreamingImporter.stream_by_keys``), which makes changed/deleted exact at the
cost of not seeing newly-added source rows.

Phase status: implemented in Phase 7.
"""

from __future__ import annotations

from .connections import ConnectionManager
from .importer import StreamingImporter
from .models import ConflictReport, SelectionMode, SelectionSpec
from .staging import StagingManager, compute_row_hash, compute_src_key


def fingerprint_rows(rows: list, key_columns: list, source_cols: list) -> dict:
    """{src_key: row_hash} for a batch of source rows."""
    return {compute_src_key(r, key_columns): compute_row_hash(r, source_cols) for r in rows}


class ConflictDetector:
    def __init__(self, conns: ConnectionManager | None = None,
                 importer: StreamingImporter | None = None):
        self.conns = conns or ConnectionManager()
        self.importer = importer or StreamingImporter(self.conns)

    def current_fingerprints(self, profile, session: dict) -> dict:
        """Re-read the source and return {src_key: row_hash} as it is RIGHT NOW.

        Shared by ``check`` (which diffs it against the import-time baseline) and by the
        write-back route, which uses it to re-baseline the staging table so the rows it
        just wrote don't read as third-party conflicts on the next check.

        For a FULL import the session's selection is the whole table, so one scan both
        covers every staged key and reveals rows added since import. For any SAMPLED
        selection it is NOT re-runnable: ``random_n`` renders ``ORDER BY RANDOM() LIMIT
        n`` and returns a different sample each call, and ``first_n`` without an
        ORDER BY is unordered on Postgres/MySQL. Re-issuing it there compared the
        baseline against unrelated rows — every staged key looked deleted, so
        ``has_conflict`` was permanently true and a sampled import could never be
        written back. Those modes instead re-read exactly the staged keys."""
        key_columns = session.get("key_columns") or []
        source_cols = [c["name"] for c in session.get("columns", [])
                       if c.get("ctype") == "source"]
        if not key_columns:
            return {}
        current: dict = {}
        for chunk in self._read_current(profile, session, key_columns):
            for r in chunk:
                current[compute_src_key(r, key_columns)] = compute_row_hash(r, source_cols)
        return current

    def _read_current(self, profile, session: dict, key_columns: list):
        """Yield chunks of current source rows: a full scan for a FULL selection,
        otherwise a keyed re-read of exactly the staged rows."""
        sel = _selection_of(session)
        if sel.mode == SelectionMode.FULL.value:
            return self.importer.stream(profile, session["table"], sel)
        staging = StagingManager(session["id"], session.get("staging_table", "staged"))
        return self.importer.stream_by_keys(
            profile, session["table"], key_columns, staging.staged_keys())

    def check(self, profile, session: dict) -> ConflictReport:
        """Diff the current source against the import-time snapshot for ``session``
        (a db-project dict)."""
        key_columns = session.get("key_columns") or []

        if not key_columns:
            return ConflictReport(
                note="No primary key on the source table — per-row conflict detection is "
                     "unavailable. Row counts alone are compared.")

        staging = StagingManager(session["id"], session.get("staging_table", "staged"))
        baseline = staging.fingerprints()   # {src_key: row_hash} captured at import

        sel = _selection_of(session)
        full = sel.mode == SelectionMode.FULL.value
        current = self.current_fingerprints(profile, session)

        changed, new, deleted, unchanged = [], [], [], 0
        for sk, h in current.items():
            if sk not in baseline:
                new.append(sk)
            elif baseline[sk] != h:
                changed.append(sk)
            else:
                unchanged += 1
        for sk in baseline:
            if sk not in current:
                deleted.append(sk)

        note = ""
        if not full:
            # A keyed re-read only ever returns staged keys, so it cannot see rows
            # added to the source since import. Say so rather than reporting an empty
            # list as if the table had been scanned.
            new = []
            note = ("Selection was a sample, so only the staged rows were re-read: "
                    "'changed' and 'deleted' are exact, but rows ADDED to the source "
                    "since import are not visible here. Re-import to pick those up.")
        # Only changed/deleted are conflicts. 'new' never was one — those rows are not
        # staged, so write-back would not touch them.
        has = bool(changed or deleted)
        return ConflictReport(changed=changed, new=new, deleted=deleted, unchanged=unchanged,
                              has_conflict=has, note=note)


def _selection_of(session: dict) -> SelectionSpec:
    """The session's SelectionSpec, ignoring any keys the dataclass doesn't define."""
    return SelectionSpec(**{k: v for k, v in (session.get("selection") or {}).items()
                            if k in SelectionSpec.__dataclass_fields__})
