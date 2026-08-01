#!/usr/bin/env python3
"""
Local Streaming Token — pluggable vector-store backends.

``rag.py`` owns everything that is storage-independent (chunking, the multi-server
``EmbedPool``, contextual chunking, the bounded-wave write loop, RRF merging). This
package owns the part that actually persists and searches vectors, behind one small
interface so the two supported stores can coexist:

``duckdb``
    The original store: a single AES-256-encrypted DuckDB file whose key is unwrapped
    by the login password. Vector search is an exact scan, optionally accelerated by an
    in-memory HNSW sidecar; keyword search is BM25 scored in Python. **Encrypted at
    rest**, and slower.

``lance``
    A LanceDB dataset with a durable on-disk vector index and a native full-text
    (tantivy) index. Faster, and the keyword path stops materialising the whole corpus
    into Python. **Not encrypted** — chunk text and embeddings sit in plaintext on disk,
    readable by anyone with file access and independent of the login password.

That trade-off is the user's to make; it is surfaced in Settings → RAG and the two
stores are independent, so switching back and forth never destroys the other's data.

Rows crossing this interface are plain dicts (see ``ROW_FIELDS``) so neither backend's
storage quirks leak into ``rag.py``.
"""

import threading

# The neutral row shape. Backends translate this into their own storage layout.
ROW_FIELDS = ("id", "source_type", "source_id", "item_id", "chunk_index",
              "content", "content_hash", "vector", "model", "updated",
              "context", "meta", "embed_hash")

DUCKDB = "duckdb"
LANCE = "lance"
BACKENDS = (LANCE, DUCKDB)

_LOCK = threading.RLock()
_ACTIVE = None                  # cached backend instance
_ACTIVE_NAME = None             # which backend that instance is
_CONFIGURED = LANCE             # what settings asked for

# Settings are PUSHED in (mirroring rag.set_chunk_config / rag.set_ann_enabled) rather
# than pulled: there is no module-level Store singleton — server.py constructs one per
# app — so a pull would create an import cycle for no benefit.


def normalize_name(name) -> str:
    name = (str(name or "")).strip().lower()
    return name if name in BACKENDS else LANCE


def set_backend(name) -> str:
    """Select the backend. Drops any cached instance so the next use opens the new
    store. Returns the normalized name actually applied."""
    global _CONFIGURED
    name = normalize_name(name)
    with _LOCK:
        if name != _CONFIGURED:
            _CONFIGURED = name
            reset()
        else:
            _CONFIGURED = name
    return name


def active_name() -> str:
    """Name of the backend in use, without forcing one to open."""
    return _ACTIVE_NAME or _CONFIGURED


def get_backend(name=None):
    """Return the backend instance for ``name`` (default: the configured one). Cached,
    so the DuckDB connection / Lance handle is opened once."""
    global _ACTIVE, _ACTIVE_NAME
    name = normalize_name(name if name is not None else _CONFIGURED)
    with _LOCK:
        if _ACTIVE is not None and _ACTIVE_NAME == name:
            return _ACTIVE
        if _ACTIVE is not None:
            try:
                _ACTIVE.close()
            except Exception:
                pass
            _ACTIVE = None
        if name == LANCE:
            from .lance_backend import LanceBackend
            _ACTIVE = LanceBackend()
        else:
            from .duckdb_backend import DuckDBBackend
            _ACTIVE = DuckDBBackend()
        _ACTIVE_NAME = name
        return _ACTIVE


def reset():
    """Close and drop the cached backend, so the next use reopens against the current
    data profile / backend. Called on profile switch and on backend change."""
    global _ACTIVE, _ACTIVE_NAME
    with _LOCK:
        if _ACTIVE is not None:
            try:
                _ACTIVE.close()
            except Exception:
                pass
        _ACTIVE = None
        _ACTIVE_NAME = None
