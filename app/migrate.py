#!/usr/bin/env python3
"""
Local Streaming Token — one-time encryption migration.

Runs immediately after the first successful login of an install that already has
plaintext data on disk. It walks ``data/`` and ``settings/`` (every profile) and, for
each file the app owns, rewrites any *plaintext* file as an AES-encrypted one under the
now-active Data Encryption Key, deleting the plaintext in the same atomic replace.

It is **idempotent**: files that already carry the encryption header (or DuckDB files
that already refuse a plain open) are skipped, so it is safe to run on every launch.

Carve-outs (must stay readable BEFORE login, or are already encrypted / are templates):
  * ``data/profiles.json`` and ``settings/profiles.json`` — the profile registries the
    ProfileManager reads at boot to know which profile is active.
  * ``settings/network.json`` — the bind host and port (app/netconfig.py). main.py reads
    it to open the listening socket, which happens before any login can supply the key.
    Encrypting it does not fail loudly: netconfig falls back to its defaults, so the app
    silently reverts to localhost on 8756 and the user simply cannot reach it any more.
  * ``settings/app_key.enc`` and any ``*.enc`` (e.g. the DB connection vault) — already
    encrypted with their own scheme.
  * ``settings/settings.example.json`` — a plaintext template that ships in the repo.
  * **Anything inside a ``*.lance`` directory** — the LanceDB vector store. It is a
    deliberately plaintext store owned by a third-party engine (see
    ``app/vectorstore/lance_backend.py``), made of many internal files: data fragments,
    manifests, transaction logs and index segments. Encrypting them in place makes the
    store unopenable — Lance reports ``LanceError(IO): file size is too small`` because
    every file has grown by the 36-byte header/nonce/tag. This carve-out is load-bearing:
    without it the sweep destroys the vector store on the very next login.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import core, crypto

# Files that must NOT be encrypted (see module docstring).
_SKIP_NAMES = {"profiles.json", "network.json", "app_key.enc", "settings.example.json"}
_SKIP_SUFFIXES = {".enc", ".tmp", ".encmig"}
# Directory suffix marking a third-party store we must not touch the insides of.
_OPAQUE_STORE_SUFFIX = ".lance"


def _in_opaque_store(path: Path) -> bool:
    """True for anything at or beneath a ``*.lance`` directory. Checked against every
    path component, since the store nests its own ``*.lance`` data fragments."""
    return any(part.lower().endswith(_OPAQUE_STORE_SUFFIX) for part in path.parts)


def _encrypt_file_in_place(path: Path) -> bool:
    """Encrypt a plaintext file in place (atomic). Returns True if it re-wrote it."""
    try:
        raw = path.read_bytes()
    except Exception:
        return False
    if crypto.is_encrypted(raw):
        return False                      # already encrypted
    core.write_bytes(path, raw)           # writes encrypted (app is unlocked) + atomic replace
    return True


def _migrate_duckdb(path: Path) -> bool:
    """Rewrite a plaintext DuckDB file as an encrypted one. Skips files that can't be
    opened plaintext (already encrypted, or not a DB)."""
    import duckdb
    p = str(path)
    try:
        con = duckdb.connect(p, read_only=True)
        con.close()
    except Exception:
        return False                      # can't open plaintext -> assume already encrypted
    tmp = p + ".encmig"
    if os.path.exists(tmp):
        os.remove(tmp)
    lit, tlit = p.replace("'", "''"), tmp.replace("'", "''")
    key = crypto.dek_hex()
    c = duckdb.connect()
    try:
        c.execute(f"ATTACH '{lit}' AS src (READ_ONLY)")
        c.execute(f"ATTACH '{tlit}' AS dst (ENCRYPTION_KEY '{key}')")
        c.execute("COPY FROM DATABASE src TO dst")
        c.execute("DETACH src")
        c.execute("DETACH dst")
    finally:
        c.close()
    os.replace(tmp, path)
    return True


def _should_skip(path: Path) -> bool:
    return (path.name in _SKIP_NAMES
            or path.suffix.lower() in _SKIP_SUFFIXES
            or _in_opaque_store(path))


def run() -> dict:
    """Encrypt every plaintext app file under data/ and settings/. Returns a small
    summary. Best-effort: a failure on one file never aborts the migration."""
    if not crypto.is_unlocked():
        return {"skipped": "locked"}
    files = enc = dbs = errors = 0
    for base in (core.DATA_DIR, core.SETTINGS_DIR):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or _should_skip(path):
                continue
            files += 1
            try:
                if path.suffix.lower() == ".duckdb":
                    if _migrate_duckdb(path):
                        dbs += 1
                elif _encrypt_file_in_place(path):
                    enc += 1
            except Exception:
                errors += 1
    return {"files_seen": files, "encrypted": enc, "duckdb_encrypted": dbs, "errors": errors}


def wipe_encrypted() -> int:
    """Delete every encrypted app file under data/ and settings/ plus all DuckDB stores.

    Used ONLY by the CLI forgot-password reset: once the login password is lost the
    Data Encryption Key can never be recovered, so the encrypted data is unreadable
    garbage — this clears it for a clean start. Plaintext registries/templates and the
    separately-encrypted ``*.enc`` files (incl. the DB connection vault) are left alone;
    the keyfile itself is removed by the caller. Returns the number of files removed.

    LanceDB stores are removed **wholesale**, and that is the point: they are plaintext,
    so unlike the encrypted files they would remain perfectly readable after a "wipe my
    data" reset. Leaving them would silently break the promise this command makes. They
    are also a rebuildable cache, so nothing unique is lost."""
    removed = 0
    for base in (core.DATA_DIR, core.SETTINGS_DIR):
        if not base.exists():
            continue
        # Plaintext vector stores go first, as whole directories — _should_skip()
        # deliberately protects their insides from the per-file sweep below.
        for store in list(base.rglob(f"*{_OPAQUE_STORE_SUFFIX}")):
            if not store.is_dir():
                continue
            try:
                n = sum(1 for p in store.rglob("*") if p.is_file())
                shutil.rmtree(store, ignore_errors=True)
                if not store.exists():
                    removed += n
            except OSError:
                pass
        for path in base.rglob("*"):
            if not path.is_file() or _should_skip(path):
                continue
            try:
                # .duckdb stores are a rebuildable cache — always removable. Other files
                # are only wiped if they carry our encryption header.
                if path.suffix.lower() == ".duckdb" or crypto.is_encrypted(path.read_bytes()):
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
