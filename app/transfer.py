#!/usr/bin/env python3
"""
Local Streaming Token — moving files between the browser and this machine.

Every file feature in this app was built around a native OS dialog: a route calls
``native_dialog.pick_files()``, tkinter opens a window **on the machine running the
server**, and the route gets back a list of local paths. That is the right design when
the browser and the server are the same computer. It is useless the moment they are not —
from a phone the button appears to hang, because the dialog is waiting for someone to
click it on a desktop nobody is looking at.

This module supplies the missing halves so the same routes work either way:

* **Uploads.** ``POST /api/uploads`` writes what the browser sent into a staging
  directory and returns the paths. A route then receives those paths in its JSON body and
  proceeds exactly as if a picker had produced them — no route needed to learn what a
  multipart request is.
* **Downloads.** A route that would have written to a user-chosen destination writes to a
  staging file instead and hands back a single-use token; the browser fetches it from
  ``/api/download/<token>`` and the phone saves it wherever phones save things.

Staging lives in the OS temp directory, not under ``data/``: these files are in transit,
belong to no profile, and must not be swept into the encrypted store.

Nothing here weakens the local case. When the request comes from the machine itself and
the client asked for a dialog, the dialog is what it gets.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

# One staging root per process, removed and recreated at startup so a crash cannot leave
# yesterday's uploads readable.
STAGING_ROOT = Path(tempfile.gettempdir()) / "local-streaming-token-transfer"

# token -> (path, filename, created_at). Downloads are single-use; this only has to
# outlive the round trip between the JSON response and the browser fetching it.
_downloads: dict[str, tuple[Path, str, float]] = {}
_DOWNLOAD_TTL = 3600.0


def reset_staging():
    """Clear and recreate the staging root. Called once from create_app()."""
    try:
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    except OSError:
        pass
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    _downloads.clear()


def _new_dir() -> Path:
    d = STAGING_ROOT / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------- uploads ---------------------------

def stage_uploads(files) -> list[str]:
    """Write ``files`` (a list of Werkzeug FileStorage) into a fresh staging directory
    and return their paths.

    Each batch gets its own directory, so two files with the same name from different
    uploads cannot collide — and ``secure_filename`` keeps a hostile name from escaping
    it. The extension is preserved because every consumer downstream (ingest, images,
    transcribe) dispatches on it.
    """
    out, seen = [], set()
    d = _new_dir()
    for f in files:
        raw = f.filename or "upload"
        name = secure_filename(raw) or "upload"
        if "." not in name and "." in raw:
            # secure_filename can strip an extension that was the only ASCII part of a
            # non-Latin filename; without it ingest cannot tell a PDF from a PNG.
            name += raw[raw.rindex("."):].lower()
        stem, dot, ext = name.rpartition(".")
        n = 1
        while name.lower() in seen:
            name = f"{stem or 'upload'}-{n}{dot}{ext}"
            n += 1
        seen.add(name.lower())
        dest = d / name
        f.save(str(dest))
        out.append(str(dest))
    return out


def is_staged(path) -> bool:
    """True when ``path`` is inside the staging root.

    Paths that arrive from a browser are only honoured when this holds. Without the
    check, a signed-in client could name any file on the machine and have the server read
    it back — a folder path typed into the Batch tab is a deliberate feature, but a path
    smuggled in where a file picker was expected is not.
    """
    try:
        Path(path).resolve().relative_to(STAGING_ROOT.resolve())
        return True
    except (ValueError, OSError):
        return False


def accept_paths(paths) -> list[str]:
    """Filter client-supplied paths down to the ones that really are staged uploads."""
    return [str(p) for p in (paths or []) if p and is_staged(p)]


# --------------------------- downloads ---------------------------

def offer(filename: str) -> tuple[str, Path]:
    """Reserve a staging file for the caller to write, and return ``(token, path)``.

    The caller writes to ``path`` exactly as it would have written to a destination
    chosen in a Save dialog, then returns the token to the browser.
    """
    _sweep()
    token = uuid.uuid4().hex
    name = secure_filename(filename) or "download"
    path = _new_dir() / name
    _downloads[token] = (path, name, time.time())
    return token, path


def rebind(token: str, path):
    """Re-point a token at the file that was actually written.

    Several export routes append an extension after they are handed a destination
    (``dest += ".json"``). The token has to follow, or the browser is offered a path
    nothing was written to.
    """
    entry = _downloads.get(token)
    if not entry:
        return
    _, name, at = entry
    path = Path(path)
    _downloads[token] = (path, path.name or name, at)


def take(token: str):
    """``(path, filename)`` for a token, or None. Single use: the token is consumed, so a
    link cannot be replayed by anything that happens to see the URL."""
    entry = _downloads.pop(token, None)
    if not entry:
        return None
    path, name, _ = entry
    if not path.exists():
        return None
    return path, name


def _sweep():
    """Drop tokens nobody collected. A download the user never accepted would otherwise
    keep an unencrypted copy of their data in temp for the life of the process."""
    cutoff = time.time() - _DOWNLOAD_TTL
    for token in [t for t, (_, _, at) in _downloads.items() if at < cutoff]:
        path, _, _ = _downloads.pop(token)
        try:
            shutil.rmtree(path.parent, ignore_errors=True)
        except OSError:
            pass
