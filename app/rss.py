#!/usr/bin/env python3
"""
Local Streaming Token — RSS / Atom feeds, with Podcasting 2.0 transcripts.

The ``youtube.py`` analogue: enumerate a feed, then turn each item into one plain-text
document. Much shorter transport chain, though — podcast feeds are static XML on a CDN
with nobody trying to stop you, so there is no Bright Data / Playwright ladder here. What
replaces it is a *format* chain: an episode's transcript may be published as JSON, VTT,
SRT, plain text or HTML, and a 404 on the first must fall through to the next rather than
dropping straight into a forty-minute local transcription.

Shape of the API, deliberately different from youtube's ``fetch_playlist`` →
``fetch_video`` pair: ``fetch_episode`` takes the ALREADY-PARSED feed and item rather
than a URL, so one feed request serves every episode. The YouTube pair re-resolves each
video because it has no choice; here we do.

--- feedparser is not enough, and that is not a style preference ---

feedparser 6.0.11 collapses repeated namespaced elements to a single last-one-wins dict.
Measured against the No Agenda feed: ``entry.podcast_person`` returns only John C.
Dvorak — Adam Curry, listed first, is simply gone. An episode advertising transcripts in
JSON *and* SRT would keep one at random, defeating the whole preference chain below.

So the ``podcast:*`` elements are parsed out of the raw XML with ``xml.etree`` and keyed
back to items by guid; feedparser is used only for the well-trodden RSS/Atom core (dates
in eight formats, Atom vs RSS, entity soup), which it is very good at.

Matching is on the element's LOCAL NAME, ignoring the namespace URI. Also not a style
preference: the sample feed declares the namespace as
``https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md`` while the
spec text says ``https://podcastindex.org/namespace/1.0``. Both are in the wild.

--- Where the text comes from ---

An item WITH an audio/video enclosure is a podcast episode: the transcript is the
document and the show notes are supporting matter. An item WITHOUT one is an article:
the notes ARE the document, so a teaser shorter than ``rss_notes_min_chars`` escalates to
a full page crawl of the item's ``<link>`` via ``core.fetch_url_text``. An episode never
escalates — importing the 226-item sample feed would otherwise fire 226 Playwright
crawls to fetch credits we already have.

Public API:
    RSSError
    feed_id(url) / episode_key(item) / episode_id(feed_id, raw_key)
    fetch_feed(url, limit=, refresh=, should_stop=, on_progress=) -> dict
    fetch_episode(feed, item, ...) -> dict
    format_episode_text(meta, ...) -> str
    episode_doc_name(feed, meta) -> str
    choose_transcripts(links, language=) -> list
    cues_from_srt / _vtt / _podcast_json / _text / _html
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from . import core, transcribe

# Transcript types we can parse, best first. JSON outranks document order because it is
# the only format carrying speaker attribution as structured data, and keeping speakers
# is the point of the flowing step.
TRANSCRIPT_TYPES = ("application/json", "text/vtt", "application/srt", "text/srt",
                    "application/x-subrip", "text/plain", "text/html")

# Enclosure MIME prefixes that mean "there is audio here to transcribe".
_MEDIA_TYPES = ("audio/", "video/")

DEFAULT_NOTES_MIN_CHARS = 600
_FEED_TIMEOUT = 45
# A transcript file is text; 20 MB is far past any real one and stops a mislabelled
# enclosure URL from being pulled into memory as a "transcript".
_MAX_TRANSCRIPT_BYTES = 20 * 1024 * 1024


class RSSError(Exception):
    """Raised with a user-facing, actionable message when a feed can't be read."""


# --------------------------- ids ---------------------------

def _normalize_feed_url(url: str) -> str:
    """Scheme and host lowercased, default port dropped, fragment dropped. The path and
    query are kept verbatim — a query string is very often the feed's identity."""
    parts = urlsplit((url or "").strip())
    host = (parts.hostname or "").lower()
    if parts.port and not ((parts.scheme == "http" and parts.port == 80)
                           or (parts.scheme == "https" and parts.port == 443)):
        host = f"{host}:{parts.port}"
    return urlunsplit(((parts.scheme or "https").lower(), host, parts.path,
                       parts.query, ""))


def feed_id(url: str) -> str:
    return hashlib.sha256(_normalize_feed_url(url).encode("utf-8")).hexdigest()[:40]


def episode_key(item) -> tuple:
    """``(raw_key, source)``. Preference: guid, enclosure url, link.

    ``<guid>`` first because it is the only field a publisher promises is stable. A CDN
    can rotate an enclosure host and a site can restructure its permalinks, and either
    would read as a brand-new back catalogue — which, with Whisper enabled, means
    re-transcribing every episode.
    """
    for key, source in (("guid", "guid"), ("enclosure_url", "enclosure"),
                        ("link", "link")):
        value = ((item or {}).get(key) or "").strip()
        if value:
            return value, source
    return "", ""


def episode_id(fid: str, raw_key: str) -> str:
    """Namespaced per feed. Guids are supposed to be globally unique and routinely are
    not — ``<guid>1</guid>`` is real and common — so a global keyspace would let one
    feed be served another feed's transcript."""
    return hashlib.sha256(f"{fid}\x00{raw_key}".encode("utf-8")).hexdigest()[:40]


# --------------------------- HTTP ---------------------------

def _http_get(url, *, headers=None, timeout=_FEED_TIMEOUT, max_bytes=None):
    """GET returning ``(status, headers, bytes)``. 304 comes back with empty content.

    Deliberately not ``feedparser.parse(url)``: that does its own urllib fetch with NO
    timeout parameter, which would let a hung feed pin an SSE worker forever, and applies
    its own proxy handling. Doing the conditional GET by hand costs six lines.
    """
    h = {"User-Agent": core._DEFAULT_UA}
    h.update(headers or {})
    resp = requests.get(url, headers=h, timeout=timeout, stream=bool(max_bytes))
    if resp.status_code == 304:
        resp.close()
        return 304, dict(resp.headers), b""
    resp.raise_for_status()
    if not max_bytes:
        return resp.status_code, dict(resp.headers), resp.content
    chunks, total = [], 0
    for chunk in resp.iter_content(1 << 18):
        total += len(chunk)
        if total > max_bytes:
            resp.close()
            raise RSSError(f"That file is larger than {max_bytes // 1024**2} MB.")
        chunks.append(chunk)
    resp.close()
    return resp.status_code, dict(resp.headers), b"".join(chunks)


def _header(headers, name: str) -> str:
    """Case-insensitive header lookup.

    ``dict(resp.headers)`` throws away requests' CaseInsensitiveDict, and HTTP/2
    lowercases header names on the wire — so a plain ``.get("ETag")`` silently missed
    against every HTTP/2 origin, and the conditional GET never fired.
    """
    if not headers:
        return ""
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return value or ""
    return ""


def _content_hash(raw: bytes) -> str:
    return hashlib.sha256(raw or b"").hexdigest()


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------- podcast:* from raw XML ---------------------------

def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _podcast_elements(raw: bytes) -> dict:
    """``{guid_or_link: {transcripts: [...], persons: [...], chapters_url: str}}``.

    Everything feedparser flattens away. Keyed by the item's own ``<guid>``, falling back
    to ``<enclosure url>`` then ``<link>`` — the same ladder ``episode_key`` uses, so the
    two always agree on which item is which.
    """
    out = {}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return out                      # feedparser is lenient; let it carry the feed

    items = [el for el in root.iter() if _local(el.tag) == "item"]
    if not items:
        items = [el for el in root.iter() if _local(el.tag) == "entry"]   # Atom

    for item in items:
        key, transcripts, persons, chapters = "", [], [], ""
        enclosure, link = "", ""
        for child in item:
            name = _local(child.tag)
            if name == "guid" and (child.text or "").strip():
                key = key or (child.text or "").strip()
            elif name == "enclosure":
                enclosure = enclosure or (child.get("url") or "").strip()
            elif name == "link" and not link:
                link = ((child.get("href") or child.text) or "").strip()
            elif name == "transcript" and child.get("url"):
                transcripts.append({
                    "url": (child.get("url") or "").strip(),
                    "type": (child.get("type") or "").strip().lower(),
                    "rel": (child.get("rel") or "").strip().lower(),
                    "language": (child.get("language") or "").strip().lower(),
                })
            elif name == "person":
                persons.append({
                    "name": (child.text or "").strip(),
                    "role": (child.get("role") or "").strip(),
                    "group": (child.get("group") or "").strip(),
                    "href": (child.get("href") or "").strip(),
                })
            elif name == "chapters" and child.get("url"):
                chapters = chapters or (child.get("url") or "").strip()
        key = key or enclosure or link
        if not key:
            continue
        out[key] = {"transcripts": transcripts, "persons": persons,
                    "chapters_url": chapters}
    return out


# --------------------------- feed parsing ---------------------------

def _import_feedparser():
    try:
        import feedparser
    except Exception as e:
        raise RSSError("Reading a feed needs feedparser. Install it with:  "
                       f"pip install feedparser\n({e})")
    return feedparser


def _first_media_enclosure(entry):
    """``(url, type, bytes)`` for the first audio/video enclosure, or blanks.

    Checks the MIME type first and the extension second, because a publisher who writes
    ``type="application/octet-stream"`` on an .mp3 is common enough to matter.
    """
    for enc in (entry.get("enclosures") or []):
        href = (enc.get("href") or "").strip()
        ctype = (enc.get("type") or "").strip().lower()
        if not href:
            continue
        if ctype.startswith(_MEDIA_TYPES) or transcribe.is_supported(urlsplit(href).path):
            try:
                length = int(enc.get("length") or 0)
            except (TypeError, ValueError):
                length = 0
            return href, ctype, length
    return "", "", 0


def _published_date(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6]).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass
    return (entry.get("published") or entry.get("updated") or "").strip()[:40]


def _duration(value) -> str:
    """Normalise ``<itunes:duration>`` to HH:MM:SS.

    The tag is specified as either a clock string or a bare number of seconds, and both
    are common — the sample feed uses seconds, which would otherwise render in the
    document header as a bare "9241".
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return _clock(int(raw))
    parts = raw.split(":")
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        return f"00:{int(parts[0]):02d}:{int(parts[1]):02d}"
    return raw[:20]


def _int_or_blank(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return ""


def _normalize_entry(entry, extras) -> dict:
    """One feedparser entry plus its raw-XML ``podcast:*`` extras, as a plain dict.

    Plain dicts rather than feedparser objects on purpose: this is what gets cached under
    ``FEED_ENTRY_VERSION``, and a library's object model is not a storage format.
    """
    enc_url, enc_type, enc_bytes = _first_media_enclosure(entry)
    guid = (entry.get("id") or entry.get("guid") or "").strip()
    link = (entry.get("link") or "").strip()
    extra = extras.get(guid) or extras.get(enc_url) or extras.get(link) or {}

    body, body_source = "", ""
    for candidate in (entry.get("content") or []):
        value = (candidate.get("value") or "").strip()
        if value:
            body, body_source = value, "content"
            break
    if not body and (entry.get("summary") or "").strip():
        body, body_source = entry["summary"].strip(), "description"

    return {
        "guid": guid,
        "title": (entry.get("title") or "").strip(),
        "link": link,
        "published": _published_date(entry),
        "duration": _duration(entry.get("itunes_duration")),
        "episode": _int_or_blank(entry.get("itunes_episode")),
        "season": _int_or_blank(entry.get("itunes_season")),
        "author": (entry.get("author") or "").strip(),
        "enclosure_url": enc_url,
        "enclosure_type": enc_type,
        "enclosure_bytes": enc_bytes,
        "body_html": body,
        "body_source": body_source,
        "transcripts": extra.get("transcripts") or [],
        "persons": extra.get("persons") or [],
        "chapters_url": extra.get("chapters_url") or "",
    }


def fetch_feed(url, *, limit=0, refresh=False, should_stop=None, on_progress=None) -> dict:
    """Read a feed, newest-first in feed order, and return its listing.

    Cached with an HTTP conditional GET rather than forever: a feed's whole purpose is to
    change, so a never-expiring listing would break "check for new episodes" — while no
    caching at all would re-download 2 MB of XML to discover nothing happened. A 304 is
    served from disk with zero parse cost, and IS the fast path.

    A network failure with a cached listing in hand serves the cache with a warning; only
    a failure with nothing cached raises.
    """
    from . import rss_cache

    url = (url or "").strip()
    if not url:
        raise RSSError("No feed URL was given.")
    if not re.match(r"(?i)^https?://", url):
        url = "https://" + url

    fid = feed_id(url)
    cached = None if refresh else rss_cache.get_feed(fid)
    warnings = []

    headers = {}
    if cached:
        if cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        if cached.get("modified"):
            headers["If-Modified-Since"] = cached["modified"]

    if on_progress:
        on_progress("feed", url=url)

    entry, from_cache = None, False
    try:
        status, resp_headers, raw = _http_get(url, headers=headers)
        if status == 304 and cached:
            entry, from_cache = cached, True
        elif cached and _content_hash(raw) == (cached.get("content_hash") or "_"):
            # The origin answered 200 with byte-identical content. Very common: this
            # feed's ETag is weak and its CDN won't match it, and RFC 7232 makes
            # If-None-Match take precedence over the If-Modified-Since it WOULD have
            # honoured. Rather than guess which validator each origin respects, notice
            # that nothing changed. Bandwidth is spent either way; the parse of 2 MB of
            # XML and the re-encrypt of a 400 KB listing are not.
            entry, from_cache = cached, True
        else:
            entry = _parse_feed_bytes(url, fid, raw, resp_headers)
            rss_cache.put_feed(entry)
    except RSSError:
        raise
    except Exception as e:
        if not cached:
            raise RSSError(f"Could not read the feed: {e}")
        warnings.append(f"Using the cached listing — the feed could not be read ({e}).")
        entry, from_cache = cached, True

    items = list(entry.get("items") or [])
    if cached and not from_cache:
        warnings.extend(_warn_on_unstable_ids(cached, entry))
    if limit and limit > 0:
        items = items[:int(limit)]

    if on_progress:
        on_progress("feed", total=len(items), title=entry.get("title") or "",
                    from_cache=from_cache)

    return {"feed_id": fid, "url": url, "final_url": entry.get("final_url") or url,
            "title": entry.get("title") or "", "author": entry.get("author") or "",
            "link": entry.get("link") or "", "description": entry.get("description") or "",
            "image": entry.get("image") or "", "language": entry.get("language") or "",
            "items": items, "item_count": len(items),
            "total_available": len(entry.get("items") or []),
            "from_cache": from_cache, "warnings": warnings}


def _parse_feed_bytes(url, fid, raw, resp_headers) -> dict:
    feedparser = _import_feedparser()
    parsed = feedparser.parse(raw)
    entries = parsed.get("entries") or []
    if not entries and not (parsed.get("feed") or {}).get("title"):
        raise RSSError("That URL didn't parse as an RSS or Atom feed. Check it points at "
                       "the feed itself rather than the show's web page.")

    extras = _podcast_elements(raw)
    items = [_normalize_entry(e, extras) for e in entries]
    # An item with no guid, no enclosure and no link cannot be identified across runs, so
    # it could never be cached — skip it rather than re-fetching it forever.
    items = [i for i in items if episode_key(i)[0]]

    info = parsed.get("feed") or {}
    image = ((info.get("image") or {}).get("href")
             or (info.get("image") or {}).get("url") or "")
    return {
        "feed_id": fid, "url": url,
        "final_url": (parsed.get("href") or url),
        "title": (info.get("title") or "").strip(),
        "author": (info.get("author") or "").strip(),
        "link": (info.get("link") or "").strip(),
        "description": core._html_to_text(info.get("subtitle") or
                                          info.get("description") or "")[:2000],
        "image": image,
        "language": (info.get("language") or "").strip().lower(),
        "etag": _header(resp_headers, "ETag"),
        "modified": _header(resp_headers, "Last-Modified"),
        "content_hash": _content_hash(raw),
        "items": items, "item_count": len(items),
    }


def _warn_on_unstable_ids(old, new):
    """Warn when a feed appears to have regenerated its episode ids.

    Some feeds mint a fresh ``<guid>`` on every republish, or carry a rotating CDN token
    in the enclosure URL. Every episode then misses the cache forever — and with Whisper
    enabled that means silently re-transcribing the entire back catalogue on every run.
    Say so rather than doing it.
    """
    old_keys = {episode_key(i)[0] for i in (old.get("items") or [])}
    new_keys = {episode_key(i)[0] for i in (new.get("items") or [])}
    if len(old_keys) < 4 or not new_keys:
        return []
    kept = len(old_keys & new_keys)
    if kept < len(old_keys) * 0.5:
        return ["This feed's episode ids changed since the last check, so cached "
                "transcripts can't be matched to it. Re-importing will fetch them again."]
    return []


# --------------------------- transcript parsers ---------------------------

def choose_transcripts(links, language="") -> list:
    """The item's transcript links, best first.

    Ranked by type (JSON's structured speakers first), then ``rel="captions"``, then a
    language match, then document order. Unknown types are dropped rather than guessed
    at — a ``type="application/pdf"`` transcript is real and we cannot read it.
    """
    ranked = []
    for i, link in enumerate(links or []):
        ctype = (link.get("type") or "").strip().lower()
        if ctype not in TRANSCRIPT_TYPES or not link.get("url"):
            continue
        lang = (link.get("language") or "").lower()
        ranked.append((
            TRANSCRIPT_TYPES.index(ctype),
            0 if link.get("rel") == "captions" else 1,
            0 if (not lang or not language or lang.startswith(language[:2])) else 1,
            i, link,
        ))
    return [link for *_, link in sorted(ranked, key=lambda r: r[:4])]


_SRT_TIME = re.compile(r"^\s*[\d:,.]+\s*-->\s*[\d:,.]+")
_VTT_TAG = re.compile(r"<(\d{2}:\d{2}[:.][\d.]+|/?[cibuv](?:\.[^>]*)?)>", re.IGNORECASE)
_VOICE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>", re.IGNORECASE)
_SPEAKER_PREFIX = re.compile(r"^([A-Z][\w .'&-]{1,40}):\s+")
# Subtitle positioning/styling overrides. Text, not speech.
_ASS_OVERRIDE = re.compile(r"\{\\[^}]*\}")


def _strip_markup(text: str) -> str:
    text = _ASS_OVERRIDE.sub("", text or "")
    text = _VTT_TAG.sub("", text)
    return re.sub(r"</?[a-zA-Z][^>]*>", "", text).strip()


def cues_from_srt(text: str) -> list:
    """``[{text, speaker}]`` from SRT. Index lines and timing lines are dropped."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text or ""):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines = lines[1:]
        lines = [ln for ln in lines if not _SRT_TIME.match(ln)]
        body = _strip_markup(" ".join(lines))
        if body:
            cues.append({"text": body, "speaker": ""})
    return cues


def cues_from_vtt(text: str) -> list:
    """``[{text, speaker}]`` from WebVTT.

    Handles the header, ``NOTE``/``STYLE``/``REGION`` blocks, optional cue identifier
    lines, ``<v Name>`` voice spans and inline karaoke timestamps.
    """
    cues = []
    body = re.sub(r"^\s*WEBVTT[^\n]*\n", "", text or "", count=1)
    for block in re.split(r"\r?\n\s*\r?\n", body):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[0].strip().upper()
        if head.startswith(("NOTE", "STYLE", "REGION")):
            continue
        timed = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timed is None:
            continue                     # no timing line: not a cue
        payload = lines[timed + 1:]
        speaker = ""
        voice = _VOICE.search(" ".join(payload))
        if voice:
            speaker = voice.group(1).strip()
        cue = _strip_markup(" ".join(payload))
        if cue:
            cues.append({"text": cue, "speaker": speaker})
    return cues


def cues_from_podcast_json(payload) -> list:
    """``[{text, speaker}]`` from the podcast-index transcript JSON — the only published
    format carrying speaker attribution as structured data."""
    if isinstance(payload, (bytes, str)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return []
    segments = (payload or {}).get("segments")
    if not isinstance(segments, list):
        return []
    cues = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        body = (seg.get("body") or seg.get("text") or "").strip()
        if body:
            cues.append({"text": body, "speaker": (seg.get("speaker") or "").strip()})
    return cues


def cues_from_text(text: str) -> list:
    """``[{text, speaker}]`` from plain text: one cue per non-blank line."""
    return [{"text": ln.strip(), "speaker": ""}
            for ln in (text or "").splitlines() if ln.strip()]


def cues_from_html(html: str) -> list:
    return cues_from_text(core._html_to_text(html or ""))


def _apply_speaker_prefixes(cues) -> list:
    """Promote leading ``Name: `` prefixes to the speaker field.

    Only when at least three DISTINCT prefixes appear across the document. Without that
    guard a transcript containing one sentence beginning "Note: " would acquire a speaker
    called Note, and every following cue would be attributed to nobody by contrast.
    """
    hits = {}
    for cue in cues:
        m = _SPEAKER_PREFIX.match(cue["text"])
        if m:
            hits[m.group(1)] = hits.get(m.group(1), 0) + 1
    if len(hits) < 3:
        return cues
    out = []
    for cue in cues:
        m = _SPEAKER_PREFIX.match(cue["text"])
        if m:
            out.append({"text": cue["text"][m.end():], "speaker": m.group(1)})
        else:
            out.append(cue)
    return out


def flow_cues(cues) -> tuple:
    """``(text, had_speakers)``. Timing dropped, speaker turns become paragraphs.

    The actual flowing is ``transcribe.flow_paragraphs``, shared with the Whisper path so
    a published transcript and a locally produced one read identically.
    """
    if not cues:
        return "", False
    cues = _apply_speaker_prefixes([c for c in cues if (c.get("text") or "").strip()])
    had_speakers = any((c.get("speaker") or "").strip() for c in cues)
    return transcribe.flow_paragraphs(cues), had_speakers


def _parse_transcript(ctype: str, raw: bytes) -> list:
    text = _decode(raw)
    if ctype == "application/json":
        return cues_from_podcast_json(text)
    if ctype == "text/vtt":
        return cues_from_vtt(text)
    if ctype in ("application/srt", "text/srt", "application/x-subrip"):
        return cues_from_srt(text)
    if ctype == "text/html":
        return cues_from_html(text)
    # text/plain, but publishers mislabel constantly — if it looks like SRT, read it as
    # SRT rather than emitting three thousand cues of timestamps.
    if "-->" in text[:4000]:
        return cues_from_srt(text)
    return cues_from_text(text)


# --------------------------- notes & chapters ---------------------------

def episode_notes(item, *, min_chars=DEFAULT_NOTES_MIN_CHARS, allow_page_fetch=True,
                  has_media=False, on_progress=None) -> tuple:
    """``(text, source, error)`` for an item's show notes or article body.

    An item WITH media never escalates to a page fetch: its transcript is the document,
    and crawling every episode's web page to re-read the credits would cost a full
    Bright Data → Playwright → requests chain per episode.
    """
    body = core._html_to_text(item.get("body_html") or "")
    source = item.get("body_source") or ""
    if has_media or not allow_page_fetch:
        return body, source, ""
    if len(body) >= max(0, int(min_chars or 0)):
        return body, source, ""
    link = (item.get("link") or "").strip()
    if not link:
        return body, source, ""
    if on_progress:
        on_progress("page", url=link)
    try:
        page = core.fetch_url_text(link)
    except Exception as e:
        # A short body beats no body: keep the teaser and say what went wrong.
        return body, source, f"Page fetch failed: {e}"
    text = (page.get("text") or "").strip()
    if len(text) <= len(body):
        return body, source, ""
    return text, "page", ""


def fetch_chapters(url, *, timeout=20) -> list:
    """``[{start_time, title}]`` from a podcast:chapters JSON file.

    Chapters are garnish, not the document — every failure returns an empty list rather
    than costing the caller its episode.
    """
    try:
        _, _, raw = _http_get(url, timeout=timeout, max_bytes=2 * 1024 * 1024)
        payload = json.loads(_decode(raw))
    except Exception:
        return []
    out = []
    for ch in (payload or {}).get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        title = (ch.get("title") or "").strip()
        if not title or ch.get("toc") is False:
            continue
        try:
            start = float(ch.get("startTime") or 0)
        except (TypeError, ValueError):
            start = 0.0
        out.append({"start_time": start, "title": title})
    return out


# --------------------------- rendering ---------------------------

def _clock(seconds) -> str:
    s = max(0, int(float(seconds or 0)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _people_line(persons, cap=20) -> str:
    seen, names = set(), []
    for p in persons or []:
        name = (p.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        role = (p.get("role") or "").strip()
        names.append(f"{name} ({role})" if role else name)
    if len(names) > cap:
        return ", ".join(names[:cap]) + f", +{len(names) - cap} more"
    return ", ".join(names)


def format_episode_text(meta, *, include_notes=True, include_chapters=True,
                        include_persons=True) -> str:
    """Render one episode as the plain-text document that becomes a library item.

    The transcript goes LAST — the reverse of ``youtube.format_video_text``, and
    deliberately so. This document is routinely truncated at ``LIBRARY_PAGE_CHARS`` and
    the transcript is the only part that ever overflows; the header, people, chapters and
    notes are cheap, high-signal, and must survive the cut. It also means the ``[:6000]``
    slice the memory-drafting route takes lands on the right 6000 characters.
    """
    title = meta.get("title") or "Untitled episode"
    number = meta.get("episode")
    head_bits = [f"Episode {number}: {title}" if number else title]
    for key in ("feed_title", "published"):
        if meta.get(key):
            head_bits.append(meta[key])
    if meta.get("duration"):
        head_bits.append(meta["duration"])
    parts = [" | ".join(head_bits)]

    link = meta.get("link") or meta.get("enclosure_url") or ""
    if link:
        parts.append(link)

    if include_persons:
        people = _people_line(meta.get("persons"))
        if people:
            parts.append(f"\n--- People ---\n{people}")

    if include_chapters and meta.get("chapters"):
        lines = [f"{_clock(c.get('start_time'))}  {c.get('title')}"
                 for c in meta["chapters"]]
        parts.append("\n--- Chapters ---\n" + "\n".join(lines))

    if include_notes and (meta.get("summary") or "").strip():
        parts.append("\n--- Show notes ---\n" + meta["summary"].strip())

    transcript = (meta.get("transcript") or "").strip()
    parts.append("\n--- Transcript ---\n" +
                 (transcript if transcript else "(No transcript available)"))
    return "\n".join(parts)


_SLUG = re.compile(r"[^A-Za-z0-9]+")


def _slug(text, cap=40) -> str:
    return _SLUG.sub("-", (text or "")).strip("-").lower()[:cap].strip("-")


def episode_doc_name(feed, meta) -> str:
    """A stable, unique, filesystem-safe persona document name.

    Stability is load-bearing and it is the reason the EPISODE TITLE is not in here.
    ``KnowledgeService.add_text`` uses this verbatim as both the ``doc_id`` and the
    filename, and ``rag.upsert_item`` keys on the doc_id — so a stable name makes a
    re-import REPLACE an episode's chunks (the check-for-new-episodes workflow on the
    persona side), while a name that moves silently duplicates every episode, with no
    bulk-delete UI to recover. Publishers retitle episodes constantly (typo fixes,
    "(rerun)", sponsor changes), so a title slug here would do exactly that.

    Legibility is not lost: the rendered document's first line is the full title, and
    that is what the knowledge list shows.
    """
    short = (meta.get("episode_id") or "")[:12] or "000000000000"
    feed_part = _slug(feed.get("title"), 40)
    return "-".join(x for x in (feed_part, short) if x)[:120] + ".txt"


# --------------------------- the episode fetch ---------------------------

def fetch_episode(feed, item, *, want_whisper=False, include_notes=True,
                  settings=None, on_progress=None, should_stop=None,
                  refresh=False) -> dict:
    """Fetch one episode: cache → published transcripts → (optionally) Whisper.

    Mirrors ``youtube.fetch_video``: seed from cache, fall through a chain, and write
    back only when this call actually fetched something. Returns the working dict plus a
    rendered ``text``, ``truncated`` and ``full_chars``.
    """
    from . import rss_cache

    def stopped():
        return bool(should_stop and should_stop())

    raw_key, id_source = episode_key(item)
    if not raw_key:
        raise RSSError("That feed item carries no guid, enclosure or link, so it can't "
                       "be identified.")
    fid = feed.get("feed_id") or feed_id(feed.get("url") or "")
    eid = episode_id(fid, raw_key)

    ranked = choose_transcripts(item.get("transcripts"), feed.get("language") or "")
    advertised = ranked[0]["url"] if ranked else ""

    meta = {
        "episode_id": eid, "feed_id": fid, "episode_key": raw_key,
        "id_source": id_source,
        "title": item.get("title") or "",
        "feed_title": feed.get("title") or "",
        "link": item.get("link") or "",
        "published": item.get("published") or "",
        "duration": item.get("duration") or "",
        "episode": item.get("episode") or "", "season": item.get("season") or "",
        "enclosure_url": item.get("enclosure_url") or "",
        "enclosure_type": item.get("enclosure_type") or "",
        "enclosure_bytes": item.get("enclosure_bytes") or 0,
        "persons": list(item.get("persons") or []),
        "chapters_url": item.get("chapters_url") or "",
        "chapters": [],
        "summary": "", "summary_source": "",
        "transcript": "", "transcript_source": "", "transcript_format": "",
        "transcript_url": "", "transcript_speakers": False,
        "advertised_transcript_url": advertised,
    }
    has_media = bool(meta["enclosure_url"])
    errors, contributors = [], []
    transcript_error, missing = "", False

    cached = None if refresh else rss_cache.get(eid)
    covered = bool(cached) and rss_cache.have(cached, want_whisper, advertised)
    if cached:
        prior = rss_cache.as_result(cached)
        for key, value in prior.items():
            if value not in ("", 0, [], False, None) and key != "via":
                meta[key] = value
        contributors.append("cache")
        if on_progress:
            on_progress("cache", transcript=bool(meta["transcript"]),
                        source=meta.get("transcript_source") or "",
                        need_transcript=not covered)
        if covered:
            return _finish(meta, include_notes, contributors, errors, truncate=True)
        # Not covered: the ask has outgrown what is stored. Clear the transcript so the
        # chain below runs, exactly as fetch_video blanks a missing half.
        meta["transcript"] = ""
        meta["transcript_source"] = ""

    # --- notes ---
    if include_notes and not meta.get("summary"):
        notes, notes_source, notes_err = episode_notes(
            item,
            min_chars=(settings or {}).get("rss_notes_min_chars",
                                           DEFAULT_NOTES_MIN_CHARS),
            allow_page_fetch=(settings or {}).get("rss_fetch_pages", True),
            has_media=has_media, on_progress=on_progress)
        meta["summary"], meta["summary_source"] = notes, notes_source
        if notes_err:
            errors.append(notes_err)
        if notes:
            contributors.append(notes_source or "notes")

    # --- chapters (garnish: never fatal, only fetched on a miss) ---
    if meta["chapters_url"] and not meta["chapters"] and not stopped():
        meta["chapters"] = fetch_chapters(meta["chapters_url"])

    # --- published transcripts, best type first ---
    # A 404 on the JSON must reach the SRT rather than dropping straight into a
    # forty-minute Whisper run. Direct analogue of youtube's per-half fallthrough.
    for link in ranked:
        if stopped() or meta["transcript"]:
            break
        try:
            _, _, raw = _http_get(link["url"], max_bytes=_MAX_TRANSCRIPT_BYTES)
            cues = _parse_transcript(link["type"], raw)
            text, speakers = flow_cues(cues)
        except Exception as e:
            transcript_error = f"{link['type']}: {e}"
            errors.append(f"Transcript ({link['type']}): {e}")
            continue
        if not text:
            transcript_error = f"{link['type']}: parsed to nothing"
            errors.append(f"Transcript ({link['type']}) was empty.")
            continue
        meta.update({"transcript": text, "transcript_source": "published",
                     "transcript_format": _format_name(link["type"]),
                     "transcript_url": link["url"],
                     "transcript_speakers": speakers})
        contributors.append("published")
        if on_progress:
            on_progress("transcript", format=meta["transcript_format"],
                        chars=len(text), speakers=speakers)

    # A definitive negative — the item advertised nothing we can read — as opposed to a
    # transport failure, which must not harden into "this episode has none".
    if not meta["transcript"] and not ranked:
        missing = True
        if on_progress:
            on_progress("transcript", chars=0, missing=True)

    if not meta["transcript"] and want_whisper and has_media and not stopped():
        _whisper_rung(meta, settings, on_progress, should_stop, errors, contributors)

    if contributors and any(c != "cache" for c in contributors):
        try:
            rss_cache.put(meta, want_whisper=want_whisper, stopped=stopped(),
                          transcript_error=transcript_error, refresh=refresh)
        except Exception:
            pass          # a cache write must never cost the caller its fetch
    if missing and not meta["transcript"]:
        try:
            rss_cache.put({**meta, "published_transcript_missing": True},
                          want_whisper=want_whisper, stopped=stopped(),
                          transcript_error=transcript_error)
        except Exception:
            pass

    return _finish(meta, include_notes, contributors, errors, truncate=True,
                   stopped=stopped())


def _whisper_rung(meta, settings, on_progress, should_stop, errors, contributors):
    """Last rung: download the enclosure and transcribe it locally.

    Only reached when the caller opted in AND no published transcript could be had — it
    is minutes of GPU and a 100 MB download per episode, which is why every earlier rung
    gets a full chance first.

    A failure here is recorded and returned like any other rung failure rather than
    raised: the episode still has its title, notes and chapters, and an import of forty
    episodes must not die because one enclosure 404s.
    """
    if not transcribe.is_available():
        errors.append("Local transcription needs faster-whisper "
                      "(pip install faster-whisper).")
        return
    try:
        out = transcribe.transcribe_url(meta["enclosure_url"], settings=settings,
                                        on_progress=on_progress, should_stop=should_stop)
    except Exception as e:
        errors.append(f"Transcription failed: {e}")
        return
    text = (out.get("text") or "").strip()
    if not text:
        errors.append("Transcription produced no text.")
        return
    meta.update({
        "transcript": text,
        "transcript_source": "whisper",
        "transcript_format": "",
        "transcript_url": "",
        # faster-whisper has no diarization. <podcast:person> names are deliberately NOT
        # synthesised into labels here: attributing lines to a host we did not detect
        # would be a fabrication the model then treats as fact.
        "transcript_speakers": False,
        "whisper_model": out.get("model") or "",
        "whisper_device": out.get("device") or "",
        "whisper_compute": out.get("compute_type") or "",
        "whisper_language": out.get("language") or "",
    })
    contributors.append("whisper")


def _format_name(ctype: str) -> str:
    return {"application/json": "json", "text/vtt": "vtt", "text/html": "html",
            "text/plain": "text"}.get(ctype, "srt")


def _finish(meta, include_notes, contributors, errors, *, truncate=True, stopped=False):
    """Render, truncate for display, and report what was lost.

    The CACHE holds the full transcript — the slice is a rendering concern, so raising
    LIBRARY_PAGE_CHARS later costs nothing.
    """
    text = format_episode_text(meta, include_notes=include_notes)
    full = len(text)
    truncated = truncate and full > core.LIBRARY_PAGE_CHARS
    if truncated:
        text = text[:core.LIBRARY_PAGE_CHARS] + "\n\n[… truncated …]"
    out = dict(meta)
    out.update({"text": text, "full_chars": full, "truncated": truncated,
                "via": "+".join(contributors) if contributors else "",
                "errors": errors, "stopped": stopped,
                "from_cache": contributors == ["cache"]})
    return out
