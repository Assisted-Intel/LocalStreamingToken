#!/usr/bin/env python3
"""
Local Streaming Token — persistent cache of fetched RSS/podcast material.

A port of the ``youtube_cache.py`` contract, with one structural difference that drives
almost everything below. A YouTube video has two halves (transcript, comments) that fail
independently. A podcast episode has ONE half with three possible provenances:

    published  — the feed advertised a <podcast:transcript> and we fetched it. Free.
    whisper    — nobody published one, so we downloaded the audio and transcribed it.
                 Minutes of GPU per episode.
    none       — no transcript exists yet.

Ranking those (``SOURCE_RANK``) is what the whole write path is about: a whisper
transcript must never overwrite a published one, and a published one must always be
allowed to replace a whisper one. That asymmetry is not fussiness — the two cost three
orders of magnitude apart, and the cache never expires, so a wrong write is permanent.

Storage — two subdirectories under ``core.RSS_CACHE_DIR``, both one encrypted JSON file
per record and no index:

    episodes/<episode_id>.json   never expires; user-cleared from Settings
    feeds/<feed_id>.json         a LISTING, refreshed by conditional GET (ETag /
                                 Last-Modified). A feed's whole purpose is to change, so
                                 caching it forever would break "check for new episodes";
                                 caching it not at all would re-download 2 MB of XML to
                                 discover nothing happened.

Two directories rather than a filename prefix, so ``stats()`` stays two globs over
``st_size`` and decrypts nothing — the property youtube_cache.py's docstring calls out at
lines 20-24, kept here for the same reason (a thousand cached episodes must not make the
Settings card expensive).

A feed entry deliberately holds NO transcripts. It is rewritten in full on every
non-304 fetch, and a 226-episode feed carrying its transcripts would turn a routine
"anything new?" check into a 45 MB re-encrypt.

Entries hold no rendered ``text`` either: ``rss.format_episode_text`` rebuilds it at read
time, because ``include_notes`` and the truncation cap differ per caller and a stored
rendering would be wrong for most of them.

Public API:
    CacheError
    get(episode_id) / have(entry, want_whisper, advertised_url) / as_result(entry)
    put(result, want_whisper=, stopped=, transcript_error=, refresh=) -> dict | None
    delete(episode_id) -> bool
    get_feed(feed_id) / put_feed(entry) / delete_feed(feed_id)
    stats() -> {episodes, feeds, transcribed, bytes, dir}
    clear(what="all") -> {removed, bytes}
"""

import re
import threading
from datetime import datetime
from pathlib import Path

from . import core

# Bumped when the entry shape changes incompatibly; a record from another version reads
# as a miss rather than as corrupt data, so an upgrade just re-fetches.
#
# ENTRY_VERSION stays at 1 deliberately, and should be very hard to talk anyone into
# bumping: an episode record can hold a Whisper transcript that cost minutes of GPU, and
# discarding a shelf of them to pick up a new metadata field is not a trade worth making.
# Prefer tolerating a missing key in rss.py.
ENTRY_VERSION = 1
# 2: items and feeds carry `categories`. The bump is what makes the category filter
# correct on an existing install — a v1 listing has no such key, so every filter would
# match nothing, and "run it again with Refresh ticked" is not a discoverable cure. Costs
# one full-body GET per feed on upgrade (no cached validators to send with).
FEED_ENTRY_VERSION = 2

# How good a transcript is, by where it came from. Only ever compared, never displayed.
SOURCE_RANK = {"": 0, "whisper": 1, "published": 2}

# Ids are hashes we compute ourselves (rss.episode_id / rss.feed_id), so they cannot be
# malformed by accident — but delete() is reachable from a route with a user-supplied id,
# and a bare join would be a path-traversal hole. Same guard as youtube_cache._path_for.
_ID_RE = re.compile(r"^[0-9a-f]{40}$")

# Serialises the read-modify-write in put(). core.write_bytes composes its temp path as a
# FIXED "<name>.tmp", so two threads upgrading one episode would interleave into a single
# temp file. Unlike the YouTube case this is genuinely reachable: a composer fetch and a
# Batch preview of the same feed legitimately overlap.
_LOCK = threading.RLock()


class CacheError(Exception):
    """Raised for a malformed id. Cache misses are not errors — they return None."""


def _episodes_dir() -> Path:
    return Path(core.RSS_CACHE_DIR) / "episodes"


def _feeds_dir() -> Path:
    return Path(core.RSS_CACHE_DIR) / "feeds"


def _path_for(episode_id: str) -> Path:
    if not _ID_RE.match(episode_id or ""):
        raise CacheError(f"Bad episode id: {episode_id!r}")
    return _episodes_dir() / f"{episode_id}.json"


def _feed_path_for(feed_id: str) -> Path:
    if not _ID_RE.match(feed_id or ""):
        raise CacheError(f"Bad feed id: {feed_id!r}")
    return _feeds_dir() / f"{feed_id}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------- Episodes: read ---------------------------

def get(episode_id: str):
    """The stored entry, or None when absent, unreadable or a foreign version.

    Every failure mode collapses to None on purpose: a cache that raises would turn a
    corrupt file into a broken fetch, when the right answer is simply to fetch again.
    """
    try:
        path = _path_for(episode_id)
    except CacheError:
        return None
    entry = core.load_json(path, None)       # returns the default on any read/parse error
    if not isinstance(entry, dict) or entry.get("v") != ENTRY_VERSION:
        return None
    return entry


def have(entry, want_whisper: bool, advertised_url: str = "") -> bool:
    """Does this cached entry satisfy THIS request's transcript ask?

    The whole coverage matrix, in order:

      * The publisher has since shipped a transcript at a URL we have never fetched, and
        what we hold is only a whisper run — take theirs. It is better and free. (A
        published transcript already in hand is not re-fetched for a changed URL: it is
        already the best rank, and re-downloading every episode because a CDN rotated a
        path is not worth it.)
      * Any stored transcript covers any ask. In particular a *published* transcript
        covers ``want_whisper=True``: the acquisition chain prefers published anyway, so
        re-transcribing would spend forty GPU-minutes to produce a worse answer.
      * "This episode publishes no transcript" is a real, cacheable fact — it stops every
        future free run from re-hitting a 404. It covers a request that does NOT want
        whisper, and it must NOT cover one that does, or ticking the checkbox would
        silently do nothing for every episode already seen. That single line is what
        makes the feature work on a feed imported once with the box unticked.
      * Never looked → never covered.
    """
    if not entry:
        return False

    if (advertised_url
            and advertised_url != (entry.get("advertised_transcript_url") or "")
            and entry.get("transcript_source") != "published"):
        return False

    if entry.get("transcript"):
        return True

    if entry.get("published_transcript_missing"):
        return not want_whisper

    return False


def as_result(entry) -> dict:
    """The entry reshaped as ``rss.fetch_episode``'s working dict, so it can seed the
    acquisition chain directly. Never carries ``text`` — the caller renders that."""
    return {
        "episode_id": entry.get("episode_id") or "",
        "feed_id": entry.get("feed_id") or "",
        "episode_key": entry.get("episode_key") or "",
        "title": entry.get("title") or "",
        "feed_title": entry.get("feed_title") or "",
        "link": entry.get("link") or "",
        "published": entry.get("published") or "",
        "duration": entry.get("duration") or "",
        "episode": entry.get("episode") or "",
        "season": entry.get("season") or "",
        "enclosure_url": entry.get("enclosure_url") or "",
        "enclosure_type": entry.get("enclosure_type") or "",
        "persons": list(entry.get("persons") or []),
        "chapters": list(entry.get("chapters") or []),
        "chapters_url": entry.get("chapters_url") or "",
        "summary": entry.get("summary") or "",
        "summary_source": entry.get("summary_source") or "",
        "transcript": entry.get("transcript") or "",
        "transcript_source": entry.get("transcript_source") or "",
        "transcript_format": entry.get("transcript_format") or "",
        "transcript_url": entry.get("transcript_url") or "",
        "transcript_speakers": bool(entry.get("transcript_speakers")),
        "via": "cache",
    }


# --------------------------- Episodes: write ---------------------------

_META_KEYS = ("feed_id", "episode_key", "id_source", "title", "feed_title", "link",
              "published", "duration", "episode", "season", "enclosure_url",
              "enclosure_type", "enclosure_bytes", "chapters_url")
_LIST_KEYS = ("persons", "chapters")


def put(result: dict, *, want_whisper: bool = False, stopped: bool = False,
        transcript_error: str = "", refresh: bool = False):
    """Write or upgrade the entry for ``result['episode_id']``. Returns it, or None when
    there was nothing worth storing.

    Merges rather than replaces. Every rule exists because the cache never expires, so
    anything written once is served indefinitely:

      1. An empty transcript is never stored. A transcript URL that 403'd today must not
         harden into "this episode has none".
      2. A transcript is written only when it ranks at least as high as the stored one.
         Published upgrades whisper and flips ``transcript_source``; whisper NEVER
         overwrites published. At equal rank, a published one is rewritten when its URL
         changed (the publisher replaced the file), and a whisper one only when it came
         out longer — or when the caller passed ``refresh``, which is read-bypass and
         write-through by definition.
      3. A stopped transcription writes nothing, whatever its length. faster-whisper's
         segment generator is consumed incrementally, so a cancelled run holds the first
         N minutes and, once on disk, is indistinguishable from a complete transcript.
      4. ``published_transcript_missing`` is set only from a DEFINITIVE negative — the
         item advertised no usable <podcast:transcript> at all. A URL that 404'd, timed
         out or parsed to nothing is a transport failure: it arrives as
         ``transcript_error`` and the flag is left alone. The flag is cleared the moment
         a transcript of any provenance lands.
      5. Metadata merges; a non-empty value wins and a blank never clears a stored one.
      6. Metadata alone is not worth a file — but a ``summary`` alone IS, unlike
         youtube_cache. For a plain non-podcast item the notes ARE the document, and
         getting them may have cost a full page crawl.
    """
    episode_id = (result or {}).get("episode_id") or ""
    try:
        path = _path_for(episode_id)
    except CacheError:
        return None

    with _LOCK:
        entry = get(episode_id) or {"v": ENTRY_VERSION, "created": _now()}
        entry["v"] = ENTRY_VERSION
        entry["episode_id"] = episode_id
        changed = False

        for key in _META_KEYS:
            value = result.get(key)
            if value not in (None, "", 0):
                changed = changed or entry.get(key) != value
                entry[key] = value
        for key in _LIST_KEYS:
            value = result.get(key)
            if value:
                changed = changed or entry.get(key) != value
                entry[key] = list(value)

        summary = result.get("summary") or ""
        if summary and summary != entry.get("summary"):
            entry["summary"] = summary
            entry["summary_source"] = result.get("summary_source") or ""
            changed = True

        # What the feed advertised on THIS pass, used or not. have() compares against it
        # to notice a publisher who has since shipped a transcript.
        advertised = result.get("advertised_transcript_url")
        if advertised is not None and advertised != entry.get("advertised_transcript_url"):
            entry["advertised_transcript_url"] = advertised
            changed = True

        transcript = result.get("transcript") or ""
        source = result.get("transcript_source") or ""
        # Rule 3, before any length comparison: a truncated run must not even be
        # considered as an upgrade.
        if transcript and stopped and source == "whisper":
            transcript = ""

        if transcript and _wins(entry, transcript, source, result, refresh):
            entry["transcript"] = transcript
            entry["transcript_source"] = source
            entry["transcript_format"] = result.get("transcript_format") or ""
            entry["transcript_url"] = result.get("transcript_url") or ""
            entry["transcript_speakers"] = bool(result.get("transcript_speakers"))
            entry["transcript_chars"] = len(transcript)
            entry["transcript_fetched"] = _now()
            if source == "whisper":
                for k in ("whisper_model", "whisper_device", "whisper_compute",
                          "whisper_language"):
                    if result.get(k):
                        entry[k] = result[k]
            # Rule 4: any transcript clears the negative fact.
            entry.pop("published_transcript_missing", None)
            changed = True

        # Rule 4: only a parse-level negative sets the flag, and only when we do not
        # already hold a transcript.
        if (result.get("published_transcript_missing") and not transcript_error
                and not entry.get("transcript")):
            if not entry.get("published_transcript_missing"):
                entry["published_transcript_missing"] = True
                changed = True
            entry["published_transcript_checked"] = _now()

        # Rule 6.
        if not (entry.get("transcript") or entry.get("published_transcript_missing")
                or entry.get("summary")):
            return None
        if not changed:
            return entry
        entry["updated"] = _now()
        _episodes_dir().mkdir(parents=True, exist_ok=True)
        core.save_json(path, entry)
        return entry


def _wins(entry, transcript, source, result, refresh) -> bool:
    """Rule 2, isolated so the ranking is readable on its own."""
    old_rank = SOURCE_RANK.get(entry.get("transcript_source") or "", 0)
    new_rank = SOURCE_RANK.get(source, 0)
    if not entry.get("transcript"):
        return True
    if new_rank > old_rank:
        return True
    if new_rank < old_rank:
        return False                       # whisper never overwrites published
    if refresh:
        return True
    if source == "published":
        # Same rank: only a genuinely different file is worth rewriting 200 KB.
        return (result.get("transcript_url") or "") != (entry.get("transcript_url") or "")
    return len(transcript) > len(entry.get("transcript") or "")


def delete(episode_id: str) -> bool:
    try:
        path = _path_for(episode_id)
    except CacheError:
        return False
    with _LOCK:
        if not path.exists():
            return False
        path.unlink()
        return True


# --------------------------- Feeds ---------------------------

def get_feed(feed_id: str):
    """The stored listing, or None. Same collapse-to-None contract as ``get``."""
    try:
        path = _feed_path_for(feed_id)
    except CacheError:
        return None
    entry = core.load_json(path, None)
    if not isinstance(entry, dict) or entry.get("v") != FEED_ENTRY_VERSION:
        return None
    return entry


def put_feed(entry: dict):
    """Store a feed listing. Replaces rather than merges — unlike an episode, the whole
    point of a listing is that the new one supersedes the old."""
    feed_id = (entry or {}).get("feed_id") or ""
    try:
        path = _feed_path_for(feed_id)
    except CacheError:
        return None
    with _LOCK:
        record = dict(entry)
        record["v"] = FEED_ENTRY_VERSION
        record["fetched"] = _now()
        _feeds_dir().mkdir(parents=True, exist_ok=True)
        core.save_json(path, record)
        return record


def delete_feed(feed_id: str) -> bool:
    try:
        path = _feed_path_for(feed_id)
    except CacheError:
        return False
    with _LOCK:
        if not path.exists():
            return False
        path.unlink()
        return True


# --------------------------- Maintenance ---------------------------

def _files(d: Path):
    return sorted(d.glob("*.json")) if d.is_dir() else []


def stats() -> dict:
    """Counts and bytes for the Settings card.

    ``transcribed`` — how many episodes hold a locally produced transcript — is the one
    figure here that costs a decrypt per episode, and it earns it: the Clear button
    otherwise throws away GPU-hours without saying so.
    """
    eps = _files(_episodes_dir())
    feeds = _files(_feeds_dir())
    total = 0
    for p in eps + feeds:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    transcribed = 0
    for p in eps:
        entry = core.load_json(p, None)
        if isinstance(entry, dict) and entry.get("transcript_source") == "whisper":
            transcribed += 1
    return {"episodes": len(eps), "feeds": len(feeds), "transcribed": transcribed,
            "bytes": total, "dir": str(core.RSS_CACHE_DIR)}


def clear(what: str = "all") -> dict:
    """Drop cached RSS data for the active profile.

    ``what="feeds"`` is the cheap one — it forces the next run to re-read every listing
    while keeping every transcript, which is what somebody who just wants fresh episodes
    actually means. ``"episodes"`` and ``"all"`` are the destructive ones; the UI warns
    with the ``transcribed`` count before calling them.
    """
    dirs = []
    if what in ("all", "feeds"):
        dirs.append(_feeds_dir())
    if what in ("all", "episodes"):
        dirs.append(_episodes_dir())
    removed, freed = 0, 0
    with _LOCK:
        for d in dirs:
            for p in _files(d):
                try:
                    size = p.stat().st_size
                    p.unlink()
                    removed += 1
                    freed += size
                except OSError:
                    pass
    return {"removed": removed, "bytes": freed}
