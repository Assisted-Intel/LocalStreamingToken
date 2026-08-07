#!/usr/bin/env python3
"""
Append-only audit log. Every cell that is actually written back to a source
database is recorded as one JSON line in ``data/db/audit/<session>.jsonl``. The
log is never rewritten in place — new events are appended — so it is a durable,
tamper-evident record of what changed, when, and via which statement.
"""

from __future__ import annotations

import json
import os
import threading

from .. import core
from .models import AuditEntry

# Process-wide path -> RLock registry. The lock MUST be shared by every AuditLog
# for the same file, not held per instance: each call site builds its own
# AuditLog(session_id) (the write-back engine, the audit route), and an append is
# a read-decrypt-modify-encrypt-write of the WHOLE file. A per-instance lock
# guards nothing across instances, so two concurrent appends — or a "View audit"
# landing mid-write-back — silently drop entries from the one record of what
# touched the user's real data. Mirrors staging._CONNS/_REG_LOCK.
_LOCKS: dict = {}
_REG_LOCK = threading.Lock()


def _lock_for(path) -> threading.RLock:
    key = str(path)
    with _REG_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


class AuditLog:
    """One log file per import session. Thread-safe appends."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = core.DB_AUDIT_DIR / f"{session_id}.jsonl"
        self._lock = _lock_for(self._path)

    # The log is encrypted at rest (AES-GCM), which is not append-friendly, so each
    # append rewrites the whole file. Audit logs are small (one line per written-back
    # cell), so this is cheap and keeps the "durable record" semantics intact.
    def _read_text(self) -> str:
        if not os.path.exists(self._path):
            return ""
        return core.read_text(self._path) or ""

    def append(self, entry: AuditEntry) -> None:
        entry.session_id = entry.session_id or self.session_id
        line = json.dumps(entry.to_dict(), default=str)
        with self._lock:
            core.write_text(self._path, self._read_text() + line + "\n")

    def append_many(self, entries) -> None:
        new = []
        for e in entries:
            e.session_id = e.session_id or self.session_id
            new.append(json.dumps(e.to_dict(), default=str))
        if not new:
            return
        with self._lock:
            core.write_text(self._path, self._read_text() + "\n".join(new) + "\n")

    def read(self, limit: int = 500, offset: int = 0) -> list:
        """Return the most recent ``limit`` entries (newest first)."""
        with self._lock:
            lines = self._read_text().splitlines()
        rows = []
        for ln in reversed(lines):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
        return rows[offset:offset + limit]

    def count(self) -> int:
        with self._lock:
            return sum(1 for ln in self._read_text().splitlines() if ln.strip())
