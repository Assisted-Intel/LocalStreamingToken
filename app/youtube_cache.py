#!/usr/bin/env python3
"""
Local Streaming Token — persistent cache of fetched YouTube videos.

Crawling a video is the most expensive ingestion the app does: a watch-page fetch, a
caption download, and up to 2000 comments paged 15 at a time, sometimes through
yt-dlp. Nothing about that result changes minute to minute, yet before this module
every path re-fetched from scratch — a Batch preview followed by the run it was
previewing paid for the whole playlist twice.

So: one entry per video, per data profile, on disk, and it **never expires**. The user
clears it from Settings, or ticks "Refresh (ignore cache)" for a single fetch.

Storage — one encrypted ``<video_id>.json`` per video under ``core.YOUTUBE_CACHE_DIR``,
and no index file:

  * Not one shared blob, for the reason core.py already gives about IMAGES_DIR: a file
    that is rewritten and re-encrypted *in full* on every write must not hold megabytes.
    A transcript runs to 200k characters, and caching a 40-video playlist would rewrite
    a shared file 40 times.
  * No index beside the files either. The video id IS the filename, so a lookup is a
    path join, and ``stats()`` needs only a glob and st_size — neither decrypts
    anything. An index would be a second source of truth that can drift from the files,
    for no gain. (A future "browse the cache" UI is when one would earn its place.)

An entry holds the two halves of a fetch *separately*, each with its own provenance,
because they fail independently — see ``youtube.fetch_video``, which falls through its
transport chain per half. It does NOT hold the rendered ``text``: that is re-rendered
by ``format_video_text`` at read time, because ``include_comments`` and ``max_comments``
differ per caller and a stored rendering would be wrong for most of them.

``comment_target`` records how many comments were *asked* for, not just how many came
back. Without it, a thread with 87 comments fetched under an ask of 500 would look
short of every future ask of 500 and refetch forever — and with no TTL, forever is
literal.

There is deliberately NO in-memory cache here. That is what lets profile switching need
no hook: nothing survives in the process. Adding an LRU later would incur the same
``clear_caches()`` obligation ``app/images.py`` has, and for the same reason (decrypted
content keyed only by id) — wire it into ``_switch_data_runtime`` if you ever do.

Public API:
    CacheError
    get(video_id) -> dict | None
    have(entry, include_comments, max_comments) -> (have_transcript, have_comments)
    as_result(entry, include_comments, max_comments) -> dict
    put(result, include_comments, max_comments, stopped, comment_error) -> dict | None
    delete(video_id) -> bool
    stats() -> {entries, bytes, dir}
    clear() -> {removed, bytes}
"""

import re
import threading
from datetime import datetime
from pathlib import Path

from . import core

# Bumped when the entry shape changes incompatibly; a record from another version reads
# as a miss rather than as corrupt data, so an upgrade just re-crawls.
ENTRY_VERSION = 1

# Same alphabet as youtube._VIDEO_ID. Validated on every path build because the id
# arrives from a URL — a bare join would be a path-traversal hole (see images._path_for).
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Serialises the read-modify-write in put(). core.write_bytes composes its temp path as
# a *fixed* "<name>.tmp", so two threads upgrading the same video would interleave into
# one temp file. Realistically that is two browser tabs, or a preview overlapping a run
# — resolve_sources is sequential and the parallel engine runs after resolution — but
# the lock is free. Cross-process contention is out of scope: this is a local
# single-user app, and a lost upgrade costs one refetch, nothing worse.
_LOCK = threading.RLock()


class CacheError(Exception):
    """Raised for a malformed video id. Cache misses are not errors — they return None."""


def _path_for(video_id: str) -> Path:
    if not _ID_RE.match(video_id or ""):
        raise CacheError(f"Bad YouTube video id: {video_id!r}")
    return Path(core.YOUTUBE_CACHE_DIR) / f"{video_id}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------- Read ---------------------------

def get(video_id: str):
    """The stored entry, or None when it is absent, unreadable or a foreign version.

    Every failure mode collapses to None on purpose: a cache that raises would turn a
    corrupt file into a broken fetch, when the right answer is simply to crawl again.
    """
    try:
        path = _path_for(video_id)
    except CacheError:
        return None
    entry = core.load_json(path, None)      # returns the default on any read/parse error
    if not isinstance(entry, dict) or entry.get("v") != ENTRY_VERSION:
        return None
    return entry


def have(entry, include_comments: bool, max_comments: int):
    """(have_transcript, have_comments) for THIS request against a cached entry.

    Comments count as covered when the ask is no larger than what the last fetch either
    returned or asked for — see ``comment_target`` in the module docstring. An entry
    with no ``comments`` key was never asked for them and is never covered; one holding
    ``[]`` under a target was asked and came back empty, which is a real answer.
    """
    if not entry:
        return False, False
    have_t = bool(entry.get("transcript"))
    if not include_comments:
        return have_t, True                  # nothing to serve, nothing to fetch
    if "comments" not in entry:
        return have_t, False
    covered = max(int(entry.get("comment_count") or 0),
                  int(entry.get("comment_target") or 0))
    return have_t, int(max_comments) <= covered


def as_result(entry, include_comments: bool, max_comments: int) -> dict:
    """The entry reshaped as ``fetch_video``'s working dict, so it can seed the
    transport loop directly. Never carries ``text`` — the caller renders that itself."""
    comments = list(entry.get("comments") or []) if include_comments else []
    return {
        "video_id": entry.get("video_id") or "",
        "url": entry.get("url") or "",
        "title": entry.get("title") or "",
        "channel": entry.get("channel") or "",
        "published": entry.get("published") or "",
        "views": entry.get("views") or "",
        "transcript": entry.get("transcript") or "",
        "comments": comments[:max(0, int(max_comments))],
        "via": "cache",
    }


# --------------------------- Write ---------------------------

def put(result: dict, include_comments: bool, max_comments: int,
        stopped: bool = False, comment_error: str = ""):
    """Write or upgrade the entry for ``result['video_id']``. Returns it, or None when
    there was nothing worth storing.

    Merges rather than replaces: a half this call did not successfully fetch is left
    exactly as it was on disk. Every rule below exists because the cache never expires,
    so anything written once is served indefinitely:

      1. An empty transcript is never stored. A rung blocked today must not harden into
         "this video has no captions".
      2. Comments are stored only when they were asked for, arrived without error, and
         the run was not stopped — a cancelled run's list is truncated and, once on
         disk, indistinguishable from a complete one.
      3. ``max_comments`` is recorded as ``comment_target`` even when fewer came back.
    """
    video_id = (result or {}).get("video_id") or ""
    try:
        path = _path_for(video_id)
    except CacheError:
        return None

    with _LOCK:
        entry = get(video_id) or {"v": ENTRY_VERSION, "created": _now()}
        entry["v"] = ENTRY_VERSION
        entry["video_id"] = video_id
        changed = False

        # Metadata: keep whatever the freshest successful fetch saw, but never let a
        # blank or placeholder title overwrite a real one already on disk.
        for key in ("url", "channel", "published", "views"):
            value = result.get(key)
            if value:
                changed = changed or entry.get(key) != value
                entry[key] = value
        title = (result.get("title") or "").strip()
        # A real title always wins; the "Unknown Video" placeholder only fills a blank
        # slot, so one bad fetch can't pin a bad title on the entry forever.
        if title and (title != "Unknown Video" or not entry.get("title")):
            changed = changed or entry.get("title") != title
            entry["title"] = title

        transcript = result.get("transcript") or ""
        if transcript and transcript != entry.get("transcript"):
            entry["transcript"] = transcript
            entry["transcript_via"] = result.get("via") or ""
            entry["transcript_fetched"] = _now()
            changed = True

        if include_comments and not comment_error and not stopped:
            comments = list(result.get("comments") or [])
            target = max(int(max_comments or 0), int(entry.get("comment_target") or 0))
            # Only an upgrade: a 20-comment fetch must not shrink a stored 500.
            if len(comments) >= int(entry.get("comment_count") or 0) or "comments" not in entry:
                entry["comments"] = comments
                entry["comment_count"] = len(comments)
                entry["comments_via"] = result.get("via") or ""
                entry["comments_fetched"] = _now()
                changed = True
            if target != entry.get("comment_target"):
                entry["comment_target"] = target
                changed = True

        # Metadata alone is not worth a file: it would read as a full miss on every
        # future request anyway, so all it buys is clutter in the Settings count.
        if not (entry.get("transcript") or "comments" in entry):
            return None
        if not changed:
            return entry
        entry["updated"] = _now()
        core.save_json(path, entry)
        return entry


def delete(video_id: str) -> bool:
    try:
        path = _path_for(video_id)
    except CacheError:
        return False
    with _LOCK:
        if not path.exists():
            return False
        path.unlink()
        return True


# --------------------------- Maintenance ---------------------------

def _entry_files():
    d = Path(core.YOUTUBE_CACHE_DIR)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def stats() -> dict:
    """{entries, bytes, dir} for the Settings card. Reads file sizes only — nothing is
    decrypted, so this stays cheap with a thousand cached videos."""
    total = 0
    files = _entry_files()
    for p in files:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return {"entries": len(files), "bytes": total, "dir": str(core.YOUTUBE_CACHE_DIR)}


def clear() -> dict:
    """Drop every cached video for the active profile. Rebuildable by definition, so
    there is no confirmation gate here — the UI asks."""
    removed, freed = 0, 0
    with _LOCK:
        for p in _entry_files():
            try:
                size = p.stat().st_size
                p.unlink()
                removed += 1
                freed += size
            except OSError:
                pass
    return {"removed": removed, "bytes": freed}
