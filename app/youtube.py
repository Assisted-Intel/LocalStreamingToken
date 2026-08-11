#!/usr/bin/env python3
"""
Local Streaming Token — YouTube ingestion.

Turns a YouTube URL into one plain-text document containing the video's transcript
and (optionally) its comments, so a video can become a Library item or a chat
attachment exactly like a scraped web page.

This is a server-side port of the "YT Copy All" browser extension. The extension
runs inside youtube.com and so gets same-origin cookies for free; here we have to
fetch the watch page ourselves, which is what the fetch chain below is for:

    Bright Data Web Unlocker  (only when a token is configured)
        -> plain requests     (works from most residential IPs)
            -> yt-dlp         (optional dependency, lazy-imported)

The first two rungs both yield watch-page HTML and share all the parsing code; the
yt-dlp rung is a separate path that produces the same result dict.

Public API:
    YouTubeError
    parse_video_id(url_or_id) -> str | None
    parse_playlist_id(url_or_id) -> str | None
    fetch_video(url, include_comments, max_comments, on_progress, should_stop,
                refresh) -> dict
    fetch_playlist(url, limit, on_progress, should_stop) -> [{video_id, url, title}]
    format_video_text(meta, transcript, comments) -> str
"""

import json
import math
import re
from urllib.parse import parse_qs, urlparse

import requests

from . import core
from . import transcribe
from . import youtube_cache

WATCH_URL = "https://www.youtube.com/watch?v={vid}"
INNERTUBE_NEXT = "https://www.youtube.com/youtubei/v1/next"

# The extension's options clamp; mirrored so both entry points agree.
DEFAULT_MAX_COMMENTS = 100
MIN_MAX_COMMENTS = 5
MAX_MAX_COMMENTS = 2000

# YouTube serves ~15-20 comments per continuation. The slack matches the extension's
# `Math.ceil(maxComments / 15) + 12` so a sparse thread can't loop forever.
_COMMENTS_PER_PAGE = 15
_PAGE_SLACK = 12

# Consent/region interstitials otherwise replace the player JSON with a cookie wall.
_COOKIES = {"CONSENT": "YES+1", "SOCS": "CAI"}
_HEADERS = {
    "User-Agent": core._DEFAULT_UA,
    "Accept-Language": "en-US,en;q=0.9",
}


class YouTubeError(Exception):
    """Raised with a user-facing, actionable message when a video can't be read."""


# --------------------------- URL parsing ---------------------------

# A video id is exactly 11 chars of the URL-safe base64 alphabet.
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_FORMS = ("/shorts/", "/live/", "/embed/", "/v/")


def parse_video_id(url_or_id: str):
    """Extract the 11-character video id from any common YouTube URL form, or from a
    bare id. Returns None when the input isn't recognisably a video reference.

    Handles watch?v=, youtu.be/, /shorts/, /live/, /embed/ and /v/, with or without a
    scheme, and ignores extra query parameters (playlists, timestamps, tracking).
    """
    raw = (url_or_id or "").strip()
    if not raw:
        return None
    if _VIDEO_ID.match(raw):
        return raw
    if not re.match(r"(?i)^https?://", raw):
        raw = "https://" + raw
    try:
        parts = urlparse(raw)
    except Exception:
        return None
    # str.lstrip takes a SET of characters, not a prefix: "wwwyoutube.com".lstrip("www.")
    # is "youtube.com", which sailed straight through the allowlist below.
    host = (parts.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host not in ("youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
                    "youtube-nocookie.com"):
        return None

    if host == "youtu.be":
        candidate = (parts.path or "/").strip("/").split("/")[0]
        return candidate if _VIDEO_ID.match(candidate) else None

    candidate = (parse_qs(parts.query or "").get("v") or [""])[0]
    if _VIDEO_ID.match(candidate):
        return candidate

    path = parts.path or ""
    for form in _PATH_FORMS:
        if path.startswith(form):
            candidate = path[len(form):].split("/")[0]
            return candidate if _VIDEO_ID.match(candidate) else None
    return None


# Playlist ids are longer and use a different alphabet than video ids. PL = user
# playlist, UU/UL = channel uploads, OL/RD/FL = generated mixes, LL = liked videos.
# "WL" (Watch Later) is deliberately excluded — it needs the owner's cookies.
_PLAYLIST_ID = re.compile(r"^(?:PL|UU|UL|OL|RD|FL|LL)[A-Za-z0-9_-]{10,}$")


def parse_playlist_id(url_or_id: str):
    """Extract a playlist id from a YouTube URL, or from a bare id. Returns None when
    the input isn't recognisably a playlist reference.

    Note ``parse_video_id`` deliberately ignores the ``list=`` parameter so that a
    "video in a playlist" URL still resolves to the single video; this is the opposite
    lookup, used when the user explicitly asks to batch a whole playlist.
    """
    raw = (url_or_id or "").strip()
    if not raw:
        return None
    if _PLAYLIST_ID.match(raw):
        return raw
    if not re.match(r"(?i)^https?://", raw):
        raw = "https://" + raw
    try:
        parts = urlparse(raw)
    except Exception:
        return None
    host = (parts.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host not in ("youtube.com", "m.youtube.com", "music.youtube.com",
                    "youtube-nocookie.com"):
        return None
    candidate = (parse_qs(parts.query or "").get("list") or [""])[0]
    return candidate if _PLAYLIST_ID.match(candidate) else None


# --------------------------- Watch-page JSON ---------------------------

def _balanced_json(text: str, start: int):
    """Read one complete JSON object out of ``text`` starting at the '{' at ``start``.

    A plain non-greedy regex stops at the first '};' — which is inside the payload for
    any video whose metadata happens to contain that sequence. Walking the braces while
    tracking string state is the only reliable way to find the real end.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_assignment(html: str, name: str):
    """Parse a ``var <name> = {...};`` blob out of watch-page HTML into a dict.
    Returns None when the variable isn't present or doesn't parse."""
    m = re.search(re.escape(name) + r"\s*=\s*\{", html or "")
    if not m:
        return None
    blob = _balanced_json(html, m.end() - 1)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def _extract_string(html: str, key: str, default: str = "") -> str:
    """Pull a bare ``"KEY":"value"`` out of watch-page HTML (config keys only)."""
    m = re.search(r'"' + re.escape(key) + r'":"([^"]+)"', html or "")
    return m.group(1) if m else default


def _page_config(html: str) -> dict:
    """Everything the InnerTube comment API needs from the watch page."""
    return {
        "initial_data": _extract_assignment(html, "ytInitialData"),
        "api_key": _extract_string(html, "INNERTUBE_API_KEY"),
        "client_version": _extract_string(html, "INNERTUBE_CLIENT_VERSION",
                                          "2.20240101.00.00"),
        "visitor_data": _extract_string(html, "VISITOR_DATA"),
    }


# --------------------------- Metadata ---------------------------

def _metadata(player: dict, video_id: str, url: str) -> dict:
    """Video title/channel/date/views from the player response. Every field is
    optional — a live stream or an age-gated video may carry only some of them."""
    details = (player or {}).get("videoDetails") or {}
    micro = (((player or {}).get("microformat") or {})
             .get("playerMicroformatRenderer") or {})
    views = details.get("viewCount") or micro.get("viewCount") or ""
    try:
        views = f"{int(views):,}"
    except (TypeError, ValueError):
        views = str(views or "")
    published = (micro.get("publishDate") or micro.get("uploadDate") or "")[:10]
    return {
        "video_id": video_id,
        "url": url,
        "title": (details.get("title") or "").strip() or "Unknown Video",
        "channel": (details.get("author") or micro.get("ownerChannelName") or "").strip(),
        "published": published,
        "views": views,
    }


# --------------------------- Transcript ---------------------------

def _pick_caption_track(tracks: list):
    """Choose a caption track, preferring a human-written English one.

    Order (the extension's): manual English -> any English -> a track whose name
    mentions English -> the first track available. ``vssId`` starting with 'a.' marks
    an auto-generated track, which is noticeably worse than a manual one.
    """
    if not tracks:
        return None
    manual_en = [t for t in tracks
                 if t.get("languageCode") == "en"
                 and not str(t.get("vssId") or "").startswith("a.")]
    if manual_en:
        return manual_en[0]
    any_en = [t for t in tracks if t.get("languageCode") == "en"]
    if any_en:
        return any_en[0]
    named_en = [t for t in tracks
                if "english" in str((t.get("name") or {}).get("simpleText") or "").lower()]
    if named_en:
        return named_en[0]
    return tracks[0]


# Shared with the comment cleaner below, which has the same missing-space problem.
_ensure_punctuation_spacing = transcribe.ensure_punctuation_spacing


def _clean_transcript(raw: str) -> str:
    """Tidy a transcript assembled from the timedtext API.

    The implementation moved to ``transcribe.clean_transcript`` when podcast SRTs turned
    out to need exactly the same scrubbing — ``[MUSIC]``, ``(laughs)``, the missing space
    after a full stop. Behaviour is unchanged; this keeps the name so the rest of this
    module and its tests read as they always did.
    """
    return transcribe.clean_transcript(raw)


def _transcript_from_json3(payload: dict) -> str:
    """Flatten a timedtext ``fmt=json3`` response into one paragraph.

    Segments within an event are joined with a space and events with a space too —
    timestamps are dropped, matching the extension's output.
    """
    parts = []
    for event in (payload or {}).get("events") or []:
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        seg_text = " ".join(s.get("utf8") or "" for s in segs).strip()
        if seg_text:
            parts.append(seg_text)
    return _clean_transcript(" ".join(parts))


# --------------------------- Comments ---------------------------

def _walk_comment_continuations(node, depth=0, found=None):
    """Collect continuation tokens whose surrounding subtree mentions comments.

    The watch page carries continuation tokens for several shelves (related videos,
    chips, comments); serialising the subtree and looking for "comment" is how the
    extension tells them apart, and it is good enough because a comments continuation
    always sits next to comment-specific renderer names.
    """
    if found is None:
        found = []
    if node is None or depth > 30:
        return found
    if isinstance(node, dict):
        for key in ("continuationCommand", "continuationEndpoint"):
            sub = node.get(key) or {}
            token = (sub.get("token")
                     or ((sub.get("continuationCommand") or {}).get("token")))
            if token:
                try:
                    context = json.dumps(node).lower()
                except (TypeError, ValueError):
                    context = ""
                if "comment" in context:
                    found.append(token)
        for value in node.values():
            _walk_comment_continuations(value, depth + 1, found)
    elif isinstance(node, list):
        for value in node:
            _walk_comment_continuations(value, depth + 1, found)
    return found


def _next_continuation(payload: dict):
    """The token for the next page of comments.

    Takes the *last* continuation item in the response: YouTube appends the
    "load more" token after the batch of comments, whereas earlier entries in the
    list can be per-thread reply tokens.
    """
    token = None

    def scan(items):
        nonlocal token
        for item in items or []:
            renderer = (item or {}).get("continuationItemRenderer") or {}
            candidate = (
                ((renderer.get("continuationEndpoint") or {}).get("continuationCommand") or {}).get("token")
                or ((((renderer.get("button") or {}).get("buttonRenderer") or {})
                     .get("command") or {}).get("continuationCommand") or {}).get("token")
            )
            if candidate:
                token = candidate

    for endpoint in (payload or {}).get("onResponseReceivedEndpoints") or []:
        scan((endpoint.get("reloadContinuationItemsCommand") or {}).get("continuationItems"))
        scan((endpoint.get("appendContinuationItemsAction") or {}).get("continuationItems"))
    for action in (payload or {}).get("onResponseReceivedActions") or []:
        scan((action.get("appendContinuationItemsAction") or {}).get("continuationItems"))
        scan((action.get("reloadContinuationItemsCommand") or {}).get("continuationItems"))

    if not token:
        tokens = _walk_comment_continuations(payload)
        if tokens:
            token = tokens[-1]
    return token


_COMMENT_TRAILING_NOISE = re.compile(
    r"\s*(Reply|Show replies|Hide replies|Read more|Show more)\s*$", re.IGNORECASE)


def _clean_comment(raw: str) -> str:
    """Collapse a comment to a single line and drop trailing UI affordances."""
    text = re.sub(r"\s+", " ", raw or "").strip()
    text = _COMMENT_TRAILING_NOISE.sub("", text).strip()
    return _ensure_punctuation_spacing(text).strip()


def _parse_renderer_comment(node: dict):
    """Parse the classic ``commentRenderer`` / ``commentThreadRenderer`` shape."""
    thread = node.get("commentThreadRenderer") or node
    comment = (((thread.get("comment") or {}).get("commentRenderer"))
               or thread.get("commentRenderer")
               or node.get("commentRenderer"))
    if not comment:
        return None
    author = ((comment.get("authorText") or {}).get("simpleText")
              or (comment.get("authorName") or {}).get("simpleText") or "Unknown")
    content = comment.get("contentText") or {}
    if content.get("runs"):
        text = "".join(r.get("text") or "" for r in content["runs"])
    else:
        text = content.get("simpleText") or ""
    text = _clean_comment(text)
    if len(text) < 2:
        return None
    return {
        "author": (author or "Unknown").strip(),
        "text": text,
        "likes": (comment.get("voteCount") or {}).get("simpleText") or "",
        "published": (comment.get("publishedTimeText") or {}).get("simpleText") or "",
    }


def _parse_entity_comment(payload: dict):
    """Parse the modern ``commentEntityPayload`` shape.

    Current InnerTube responses deliver comment bodies here, in
    ``frameworkUpdates.entityBatchUpdate.mutations[]``, rather than in a renderer. The
    extension never reads this shape, which is why it so often falls back to scraping
    the DOM. It is also the only place like counts and publish times are available.
    """
    entity = (payload or {}).get("commentEntityPayload")
    if not entity:
        return None
    props = entity.get("properties") or {}
    text = _clean_comment((props.get("content") or {}).get("content") or "")
    if len(text) < 2:
        return None
    author = ((entity.get("author") or {}).get("displayName") or "Unknown").strip()
    toolbar = entity.get("toolbar") or {}
    return {
        "author": author,
        "text": text,
        "likes": toolbar.get("likeCountNotliked") or toolbar.get("likeCountLiked") or "",
        "published": props.get("publishedTime") or "",
    }


def _harvest_comments(node, out: list, seen: set, limit: int, depth: int = 0):
    """Recursively pull every comment out of an InnerTube payload.

    Both payload shapes are collected, deduped against each other on author + text
    prefix (the same key the extension uses), because a single response can describe
    one comment as a renderer *and* as an entity mutation.
    """
    if node is None or depth > 30 or len(out) >= limit:
        return
    if isinstance(node, dict):
        parsed = None
        if "commentEntityPayload" in node:
            parsed = _parse_entity_comment(node)
        elif ("commentThreadRenderer" in node or "commentRenderer" in node
                or "commentViewModel" in node):
            parsed = _parse_renderer_comment(node)
        if parsed:
            key = parsed["author"] + "|" + parsed["text"][:85]
            if key not in seen:
                seen.add(key)
                out.append(parsed)
                if len(out) >= limit:
                    return
        for value in node.values():
            _harvest_comments(value, out, seen, limit, depth + 1)
    elif isinstance(node, list):
        for value in node:
            _harvest_comments(value, out, seen, limit, depth + 1)


# --------------------------- Fetch rungs ---------------------------

def _requests_get(url: str, timeout: int) -> str:
    resp = requests.get(url, headers=_HEADERS, cookies=_COOKIES, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _requests_post_json(url: str, body: dict, timeout: int) -> dict:
    resp = requests.post(url, headers={**_HEADERS, "Content-Type": "application/json"},
                         cookies=_COOKIES, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class _Transport:
    """How to reach youtube.com for this run: either through Bright Data's Web
    Unlocker or straight out of this machine. Both rungs need a GET (watch page,
    timedtext) and a POST (InnerTube), so they're paired behind one object rather
    than threaded through every call as flags."""

    def __init__(self, name, timeout=45):
        self.name = name
        self.timeout = timeout

    def get(self, url: str) -> str:
        if self.name == "Bright Data":
            return core.brightdata_request(url, timeout=self.timeout)
        return _requests_get(url, self.timeout)

    def post_json(self, url: str, body: dict) -> dict:
        if self.name == "Bright Data":
            raw = core.brightdata_request(
                url, method="POST", body=json.dumps(body),
                headers={"Content-Type": "application/json"}, timeout=self.timeout)
            return json.loads(raw)
        return _requests_post_json(url, body, self.timeout)


def _fetch_via_html(video_id: str, url: str, transport: "_Transport",
                    include_transcript: bool, include_comments: bool,
                    max_comments: int, on_progress, should_stop) -> dict:
    """Extraction from watch-page HTML. Shared by the Bright Data and requests rungs —
    they differ only in the transport used for every request.

    Raises YouTubeError when the page yields no player response at all (a bot wall or
    a removed video). A half that fails is reported in ``transcript_error`` /
    ``comment_error`` rather than raised, so the caller can retry just that half on
    the next rung. ``include_transcript``/``include_comments`` let it skip work the
    caller already has.
    """
    def progress(phase, **fields):
        if on_progress:
            on_progress(phase, **fields)

    def stopped():
        return bool(should_stop and should_stop())

    progress("page", via=transport.name)
    html = transport.get(url)
    player = _extract_assignment(html, "ytInitialPlayerResponse")
    if not player:
        raise YouTubeError(
            "The watch page did not include player data (YouTube may have served a "
            "consent or bot-check page).")

    meta = _metadata(player, video_id, url)
    transcript, transcript_error = "", ""

    if include_transcript:
        tracks = (((player.get("captions") or {})
                   .get("playerCaptionsTracklistRenderer") or {}).get("captionTracks") or [])
        track = _pick_caption_track(tracks)
        if not track or not track.get("baseUrl"):
            transcript_error = "No captions are available for this video."
        else:
            try:
                base = track["baseUrl"]
                sep = "&" if "?" in base else "?"
                raw = transport.get(base + sep + "fmt=json3")
                if not (raw or "").strip():
                    # YouTube answers 200 with an *empty body* when the request needs
                    # a proof-of-origin token that a server can't mint. Name it rather
                    # than letting it surface as a JSON parse error, so the failure is
                    # legible and the chain knows to retry this half elsewhere.
                    transcript_error = ("YouTube returned no caption data for this "
                                        "request (proof-of-origin required).")
                else:
                    transcript = _transcript_from_json3(json.loads(raw))
            except Exception as e:
                transcript_error = f"Transcript fetch failed: {e}"
        progress("transcript", chars=len(transcript))

    comments, comment_error = [], ""
    if include_comments and not stopped():
        try:
            comments = _fetch_comments(html, transport, max_comments, progress, stopped)
        except Exception as e:
            comment_error = f"Comments fetch failed: {e}"

    return {**meta, "transcript": transcript, "comments": comments,
            "via": transport.name, "transcript_error": transcript_error,
            "comment_error": comment_error}


def _fetch_comments(html: str, transport: "_Transport", max_comments: int,
                    progress, stopped) -> list:
    """Page the InnerTube comments API until ``max_comments`` are collected."""
    config = _page_config(html)
    if not config["api_key"]:
        raise YouTubeError("Could not read YouTube's API key from the page.")

    comments = []
    seen = set()
    initial = config["initial_data"]
    if initial:
        _harvest_comments(initial, comments, seen, max_comments)

    tokens = _walk_comment_continuations(initial) if initial else []
    continuation = None if len(comments) >= max_comments else (tokens[0] if tokens else None)

    client = {"clientName": "WEB", "clientVersion": config["client_version"],
              "hl": "en", "gl": "US"}
    if config["visitor_data"]:
        client["visitorData"] = config["visitor_data"]

    url = f"{INNERTUBE_NEXT}?key={config['api_key']}"
    max_pages = math.ceil(max_comments / _COMMENTS_PER_PAGE) + _PAGE_SLACK
    used = set()
    pages = 0
    barren = 0

    while continuation and len(comments) < max_comments and pages < max_pages:
        if stopped() or continuation in used:
            break
        used.add(continuation)
        pages += 1
        before = len(comments)

        payload = transport.post_json(url, {"context": {"client": client},
                                            "continuation": continuation})
        _harvest_comments(payload, comments, seen, max_comments)
        progress("comments", done=len(comments), target=max_comments)
        continuation = _next_continuation(payload)

        if len(comments) == before:
            barren += 1
            # Two dry pages with nowhere left to go means the thread is exhausted.
            if barren >= 2 and not continuation:
                break
        else:
            barren = 0

    progress("comments", done=len(comments), target=max_comments)
    return comments[:max_comments]


def _fetch_via_ytdlp(video_id: str, url: str, include_transcript: bool,
                     include_comments: bool, max_comments: int, on_progress) -> dict:
    """Last-resort rung using yt-dlp, which maintains its own workarounds for the
    signature and consent changes that break the direct rungs. Optional dependency:
    absent, this raises an actionable message and the chain reports it like any other
    rung failure."""
    try:
        import yt_dlp
    except Exception:
        raise YouTubeError(
            "yt-dlp is not installed. Install it with:  pip install yt-dlp")

    if on_progress:
        on_progress("page", via="yt-dlp")

    opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    if include_transcript:
        opts.update({"writesubtitles": True, "writeautomaticsub": True,
                     "subtitleslangs": ["en"]})
    if include_comments:
        opts["getcomments"] = True
        opts["extractor_args"] = {"youtube": {"max_comments": [str(max_comments)]}}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    transcript, transcript_error = "", ""
    if include_transcript:
        # Prefer a human-written track over yt-dlp's automatic captions, as elsewhere.
        for bucket in ("subtitles", "automatic_captions"):
            for fmt in (info.get(bucket) or {}).get("en") or []:
                if fmt.get("ext") != "json3" or not fmt.get("url"):
                    continue
                try:
                    transcript = _transcript_from_json3(_requests_get_json(fmt["url"]))
                except Exception as e:
                    transcript_error = f"Transcript fetch failed: {e}"
                break
            if transcript:
                break
        if not transcript and not transcript_error:
            transcript_error = "No captions are available for this video."
        if on_progress:
            on_progress("transcript", chars=len(transcript))

    comments = []
    if include_comments:
        for c in (info.get("comments") or [])[:max_comments]:
            text = _clean_comment(c.get("text") or "")
            if len(text) < 2:
                continue
            likes = c.get("like_count")
            comments.append({
                "author": (c.get("author") or "Unknown").strip(),
                "text": text,
                "likes": str(likes) if likes else "",
                "published": "",
            })
        if on_progress:
            on_progress("comments", done=len(comments), target=max_comments)

    views = info.get("view_count")
    upload = str(info.get("upload_date") or "")
    published = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else ""
    return {
        "video_id": video_id,
        "url": url,
        "title": (info.get("title") or "").strip() or "Unknown Video",
        "channel": (info.get("uploader") or info.get("channel") or "").strip(),
        "published": published,
        "views": f"{views:,}" if isinstance(views, int) else "",
        "transcript": transcript,
        "comments": comments,
        "via": "yt-dlp",
        "transcript_error": transcript_error,
        "comment_error": "",
    }


def _requests_get_json(url: str, timeout: int = 45) -> dict:
    return json.loads(_requests_get(url, timeout))


# --------------------------- Playlists ---------------------------

def fetch_playlist(url: str, limit: int = 0, on_progress=None, should_stop=None) -> list:
    """Enumerate a playlist into ``[{video_id, url, title}]``, newest-first as YouTube
    orders it. ``limit`` caps the count (0 = every video).

    Uses yt-dlp's flat extraction, which lists a playlist's entries from one request
    without resolving each video — the per-video transcript fetch happens later, through
    the normal ``fetch_video`` chain, so a playlist costs exactly one extra call. Unlike
    the video path there is no requests/Bright Data rung to fall back to: scraping the
    playlist page yields a continuation-token walk that yt-dlp already does better.

    Private and deleted entries are skipped rather than raising — a 200-video playlist
    with three dead videos should still batch the other 197.
    """
    playlist_id = parse_playlist_id(url)
    if not playlist_id:
        raise YouTubeError(
            "That doesn't look like a YouTube playlist URL. Expected something like "
            "https://www.youtube.com/playlist?list=PL…")
    try:
        import yt_dlp
    except Exception:
        raise YouTubeError(
            "Reading a playlist needs yt-dlp. Install it with:  pip install yt-dlp")

    if on_progress:
        on_progress("playlist", playlist_id=playlist_id)

    opts = {
        # "in_playlist" returns entries without a network round-trip per video.
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    if limit and limit > 0:
        opts["playlistend"] = int(limit)

    target = f"https://www.youtube.com/playlist?list={playlist_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as e:
        raise YouTubeError(f"Could not read playlist '{playlist_id}': {e}")

    videos = []
    for entry in ((info or {}).get("entries") or []):
        if should_stop and should_stop():
            break
        if not entry:                       # ignoreerrors leaves None for dead entries
            continue
        vid = entry.get("id") or ""
        if not _VIDEO_ID.match(vid):
            continue
        title = (entry.get("title") or "").strip()
        # yt-dlp reports unavailable entries with these placeholder titles.
        if title in ("[Private video]", "[Deleted video]", "[Unavailable video]"):
            continue
        videos.append({
            "video_id": vid,
            "url": WATCH_URL.format(vid=vid),
            "title": title or vid,
        })
        if limit and len(videos) >= int(limit):
            break

    if not videos:
        raise YouTubeError(
            f"Playlist '{playlist_id}' returned no playable videos. It may be private "
            f"or empty.")
    if on_progress:
        on_progress("playlist", total=len(videos))
    return videos


# --------------------------- Output formatting ---------------------------

def format_video_text(meta: dict, transcript: str, comments: list,
                      include_comments: bool = True) -> str:
    """Render a fetched video as the plain-text document stored in a library item.

    Follows the extension's clipboard layout — flat transcript, ``--- Comments ---``
    separator, numbered ``N. Author: text`` — with a metadata header and per-comment
    likes/age added, since an LLM reading this later benefits from knowing whose video
    it is, when it aired, and which comments carried weight. Absent fields are simply
    left out rather than rendered empty.
    """
    header = " | ".join(p for p in (meta.get("title"), meta.get("channel"),
                                    meta.get("published"),
                                    f"{meta['views']} views" if meta.get("views") else "")
                        if p)
    lines = [header] if header else []
    if meta.get("url"):
        lines.append(meta["url"])
    out = "\n".join(lines)
    if out:
        out += "\n\n"

    out += transcript.strip() if (transcript or "").strip() else "(No transcript available)"

    if not include_comments:
        return out

    out += "\n\n--- Comments ---\n\n"
    if comments:
        rendered = []
        for i, c in enumerate(comments, 1):
            notes = []
            if c.get("likes"):
                notes.append(f"{c['likes']} likes")
            if c.get("published"):
                notes.append(c["published"])
            suffix = f" ({', '.join(notes)})" if notes else ""
            rendered.append(f"{i}. {c.get('author') or 'Unknown'}{suffix}: {c.get('text') or ''}")
        out += "\n\n".join(rendered)
    else:
        out += "(No comments available)"
    return out


# --------------------------- Public entry point ---------------------------

def fetch_video(url: str, include_comments: bool = True,
                max_comments: int = DEFAULT_MAX_COMMENTS,
                on_progress=None, should_stop=None, timeout: int = 45,
                refresh: bool = False) -> dict:
    """Fetch a YouTube video's transcript and comments as one plain-text document.

    Tries each transport in turn — Bright Data when configured, then plain requests,
    then yt-dlp — and falls through **per half**. That matters in practice: the watch
    page reliably yields comments but YouTube now answers server-side caption requests
    with an empty body unless they carry a proof-of-origin token, so the common
    outcome is comments from an early rung and the transcript from yt-dlp. Each rung
    is only asked for the half still missing, and the chain stops as soon as both are
    in hand.

    The persistent cache rides on exactly that structure: a hit seeds the working dict
    *before* the loop, so a request for more comments than were cached leaves the
    transcript half satisfied and only the comment half is re-fetched — by the code
    that was already there. ``refresh`` is read-bypass, write-through: it ignores what
    is on disk but still upgrades the entry with whatever it fetched, which is what
    "Refresh (ignore cache)" means to a user.

    ``on_progress`` is called as ``(phase, **fields)`` with phase in
    {cache, page, transcript, comments}; ``should_stop`` is polled between comment
    pages so an SSE client can cancel.

    Returns {video_id, url, title, channel, published, views, transcript, comments,
    text, via, errors}. Raises YouTubeError when the URL isn't a video or no transport
    could read the page at all.
    """
    video_id = parse_video_id(url)
    if not video_id:
        raise YouTubeError(f"'{url}' is not a recognisable YouTube video URL.")
    watch_url = WATCH_URL.format(vid=video_id)
    max_comments = max(MIN_MAX_COMMENTS, min(MAX_MAX_COMMENTS, int(max_comments or DEFAULT_MAX_COMMENTS)))

    attempts = []
    if core.BRIGHTDATA_TOKEN:
        attempts.append("Bright Data")
    attempts.append("requests")
    attempts.append("yt-dlp")

    merged = None            # metadata + whichever halves have been filled so far
    contributors = []        # rung names that supplied something, for `via`
    failures = []            # rungs that couldn't read the page at all
    last_transcript_error = ""
    last_comment_error = ""

    # Seed from disk. Blanking a half we don't have is what hands it to the loop below:
    # its want_transcript/want_comments conditions then do the right thing unchanged.
    cached = None if refresh else youtube_cache.get(video_id)
    if cached:
        have_transcript, have_comments = youtube_cache.have(
            cached, include_comments, max_comments)
        if have_transcript or have_comments:
            merged = youtube_cache.as_result(cached, include_comments, max_comments)
            if not have_transcript:
                merged["transcript"] = ""
            if not have_comments:
                merged["comments"] = []   # a short cached list can't be paged up from
            contributors.append("cache")
            if on_progress:
                on_progress("cache", transcript=have_transcript,
                            comments=len(merged["comments"]),
                            need_transcript=not have_transcript,
                            need_comments=include_comments and not have_comments)

    for name in attempts:
        want_transcript = not (merged and merged["transcript"])
        want_comments = include_comments and not (merged and merged["comments"])
        if not want_transcript and not want_comments:
            break
        if should_stop and should_stop():
            break
        try:
            if name == "yt-dlp":
                result = _fetch_via_ytdlp(video_id, watch_url, want_transcript,
                                          want_comments, max_comments, on_progress)
            else:
                result = _fetch_via_html(video_id, watch_url,
                                         _Transport(name, timeout=timeout),
                                         want_transcript, want_comments,
                                         max_comments, on_progress, should_stop)
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue

        gave = False
        if merged is None:
            merged = result
            gave = bool(result["transcript"] or result["comments"])
        else:
            if want_transcript and result["transcript"]:
                merged["transcript"] = result["transcript"]
                gave = True
            if want_comments and result["comments"]:
                merged["comments"] = result["comments"]
                gave = True
            # A rung that reached the watch page has better metadata than a cached
            # entry whose first fetch failed before it could read a title.
            if result.get("title") and merged.get("title") in ("", "Unknown Video"):
                for key in ("title", "channel", "published", "views"):
                    if result.get(key):
                        merged[key] = result[key]
        if gave:
            contributors.append(name)
        if want_transcript and result.get("transcript_error"):
            last_transcript_error = f"{name}: {result['transcript_error']}"
        if want_comments and result.get("comment_error"):
            last_comment_error = f"{name}: {result['comment_error']}"

    if merged is None:
        raise YouTubeError(
            f"Could not read '{watch_url}'. Tried: " + " | ".join(failures))

    # Only surface the problems that actually cost the user something: a half that a
    # later rung filled in doesn't need explaining.
    errors = list(failures)
    if not merged["transcript"] and last_transcript_error:
        errors.append(last_transcript_error)
    if include_comments and not merged["comments"] and last_comment_error:
        errors.append(last_comment_error)

    merged["via"] = " + ".join(contributors) if contributors else merged.get("via", "")
    merged["errors"] = errors

    # Write back only when this call actually fetched something: a pure cache hit must
    # not re-encrypt a 300KB file for having been read. Best effort — a cache write
    # must never cost the caller the fetch it has already paid for.
    if any(c != "cache" for c in contributors):
        try:
            youtube_cache.put(merged, include_comments, max_comments,
                              stopped=bool(should_stop and should_stop()),
                              comment_error=last_comment_error)
        except Exception:
            pass

    # Rendered here rather than stored, because include_comments/max_comments differ
    # per caller — which is exactly why the cache keeps the two halves and not `text`.
    merged["text"] = format_video_text(merged, merged["transcript"],
                                       merged["comments"], include_comments)[
                                           :core.LIBRARY_PAGE_CHARS]
    return merged
