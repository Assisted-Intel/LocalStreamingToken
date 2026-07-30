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

Phase status: implemented in Phase 7.
"""

from __future__ import annotations

from .connections import ConnectionManager
from .importer import StreamingImporter
from .models import ConflictReport, SelectionSpec
from .staging import StagingManager, compute_row_hash, compute_src_key


def fingerprint_rows(rows: list, key_columns: list, source_cols: list) -> dict:
    """{src_key: row_hash} for a batch of source rows."""
    return {compute_src_key(r, key_columns): compute_row_hash(r, source_cols) for r in rows}


class ConflictDetector:
    def __init__(self, conns: ConnectionManager | None = None,
                 importer: StreamingImporter | None = None):
        self.conns = conns or ConnectionManager()
        self.importer = importer or StreamingImporter(self.conns)

    def check(self, profile, session: dict) -> ConflictReport:
        """Diff the current source against the import-time snapshot for ``session``
        (a db-project dict)."""
        table = session["table"]
        key_columns = session.get("key_columns") or []
        source_cols = [c["name"] for c in session.get("columns", [])
                       if c.get("ctype") == "source"]

        if not key_columns:
            return ConflictReport(
                note="No primary key on the source table — per-row conflict detection is "
                     "unavailable. Row counts alone are compared.")

        staging = StagingManager(session["id"], session.get("staging_table", "staged"))
        baseline = staging.fingerprints()   # {src_key: row_hash} captured at import

        sel = SelectionSpec(**{k: v for k, v in (session.get("selection") or {}).items()
                               if k in SelectionSpec.__dataclass_fields__})

        current: dict = {}
        for chunk in self.importer.stream(profile, table, sel):
            for r in chunk:
                sk = compute_src_key(r, key_columns)
                current[sk] = compute_row_hash(r, source_cols)

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
        if sel.mode in ("first_n", "random_n"):
            note = ("Selection was a sample (first/random N); 'new'/'deleted' reflect only the "
                    "re-sampled rows and may not be exhaustive. 'changed' is reliable per key.")
        has = bool(changed or deleted)
        return ConflictReport(changed=changed, new=new, deleted=deleted, unchanged=unchanged,
                              has_conflict=has, note=note)
