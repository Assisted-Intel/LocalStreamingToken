#!/usr/bin/env python3
"""
Local Streaming Token — Database Processing feature.

A self-contained sub-package that adds the "Database" tab: import selected rows
from a real database into an isolated **DuckDB** staging file, edit/AI-enrich the
staged data, preview the exact write statements (dry-run), detect if the source
changed underneath us, then write back (bulk or row-by-row) with a full audit log.

Design rules (see the plan for the full spec):
  * The **source database is read-only** until an explicit (or auto-approved)
    write-back. Staging is a separate DuckDB file, so the source never sees a
    change during editing/processing.
  * **Offline after install** — the only network is the DB itself and the
    already-configured web-search credentials.
  * **Memory-frugal** — imports stream in chunks; writes batch. Never load a whole
    table into RAM.
  * **Reuse, don't rebuild** — AI processing reuses the app's ``evals.fill_prompt``
    + ``generate_one`` + ``parallel.run_parallel``; web-source columns reuse
    ``core.web_search``. See ``processing.py``.

Heavy third-party imports (SQLAlchemy, DuckDB, cryptography, drivers) are done
**lazily inside functions**, never at module import time, so the rest of the app
boots even before these optional dependencies are installed.
"""

from .models import (  # noqa: F401  (re-exported for callers)
    ColumnType, SelectionMode, WriteMode, ConflictKind,
    ConnectionProfile, ColumnDef, SelectionSpec, ImportSession,
    DryRunResult, ConflictReport, AuditEntry,
)

__all__ = [
    "ColumnType", "SelectionMode", "WriteMode", "ConflictKind",
    "ConnectionProfile", "ColumnDef", "SelectionSpec", "ImportSession",
    "DryRunResult", "ConflictReport", "AuditEntry",
]
