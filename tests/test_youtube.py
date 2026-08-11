#!/usr/bin/env python3
"""Tests for YouTube ingestion (app/youtube.py) and the chat-attachment injector
(app/logic.py). All network access is monkeypatched, so these run offline and need
neither Ollama nor a Bright Data account."""

import json

import pytest

import conftest
from app import core, logic, youtube, youtube_cache


@pytest.fixture(autouse=True)
def real_ytdlp_rung(monkeypatch):
    """Block the yt-dlp rung by default, and hand the real one to whoever wants it.

    yt-dlp is an optional dependency, so whether it happens to be installed decides
    whether the fetch chain reaches out to the real network — which would make these
    tests slow and machine-dependent. The test that exercises the rung itself asks
    for this fixture and calls the function it returns.
    """
    original = youtube._fetch_via_ytdlp

    def refuse(*a, **k):
        raise youtube.YouTubeError("yt-dlp is not installed")

    monkeypatch.setattr(youtube, "_fetch_via_ytdlp", refuse)
    return original


# ------------------------------ URL parsing ------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&t=42s", "dQw4w9WgXcQ"),
    ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?si=abcdef", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
])
def test_parse_video_id_accepts(url, expected):
    assert youtube.parse_video_id(url) == expected


@pytest.mark.parametrize("url", [
    "", None, "   ",
    "https://example.com/watch?v=dQw4w9WgXcQ",   # right shape, wrong host
    "https://www.youtube.com/watch?v=tooshort",  # not 11 chars
    "https://www.youtube.com/@somechannel",      # a channel, not a video
    "https://youtu.be/",
    "just some text",
])
def test_parse_video_id_rejects(url):
    assert youtube.parse_video_id(url) is None


# ------------------------------ playlists ------------------------------
# parse_video_id deliberately ignores `list=` so a "video in a playlist" URL still
# resolves to the single video; parse_playlist_id is the opposite lookup.

@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/playlist?list=PLabcdefghij12345", "PLabcdefghij12345"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabcdefghij12345",
     "PLabcdefghij12345"),
    ("www.youtube.com/playlist?list=UUabcdefghij12345", "UUabcdefghij12345"),
    ("PLabcdefghij12345", "PLabcdefghij12345"),
])
def test_parse_playlist_id_accepts(url, expected):
    assert youtube.parse_playlist_id(url) == expected


@pytest.mark.parametrize("url", [
    "", None, "   ",
    "https://example.com/playlist?list=PLabcdefghij12345",   # wrong host
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",           # a video, not a playlist
    "https://www.youtube.com/playlist?list=WLxxxxxxxxxxxx",  # Watch Later needs cookies
    "https://www.youtube.com/playlist?list=PLshort",         # too short
])
def test_parse_playlist_id_rejects(url):
    assert youtube.parse_playlist_id(url) is None


def test_parse_video_id_still_ignores_a_playlist_parameter():
    """Regression guard: adding playlist support must not change the video lookup."""
    assert youtube.parse_video_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabcdefghij12345"
    ) == "dQw4w9WgXcQ"


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL as a context manager."""
    info = None
    seen = {}

    def __init__(self, opts):
        _FakeYDL.seen = dict(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return _FakeYDL.info


@pytest.fixture
def fake_ytdlp(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    return _FakeYDL


def test_fetch_playlist_lists_videos(fake_ytdlp):
    fake_ytdlp.info = {"entries": [
        {"id": "a" * 11, "title": "First"},
        {"id": "b" * 11, "title": "Second"},
    ]}
    got = youtube.fetch_playlist("https://www.youtube.com/playlist?list=PLabcdefghij12345")
    assert [v["title"] for v in got] == ["First", "Second"]
    assert got[0]["url"] == "https://www.youtube.com/watch?v=" + "a" * 11
    # Flat extraction is the whole point — one request, no per-video resolution.
    assert fake_ytdlp.seen["extract_flat"] == "in_playlist"


def test_fetch_playlist_skips_dead_entries(fake_ytdlp):
    """ignoreerrors leaves None entries, and unavailable videos get placeholder titles.
    A 200-video playlist with three dead videos should still batch the rest."""
    fake_ytdlp.info = {"entries": [
        {"id": "a" * 11, "title": "Good"},
        None,
        {"id": "b" * 11, "title": "[Private video]"},
        {"id": "not-an-id", "title": "Malformed"},
        {"id": "c" * 11, "title": "Also good"},
    ]}
    got = youtube.fetch_playlist("https://www.youtube.com/playlist?list=PLabcdefghij12345")
    assert [v["title"] for v in got] == ["Good", "Also good"]


def test_fetch_playlist_honours_the_limit(fake_ytdlp):
    fake_ytdlp.info = {"entries": [{"id": chr(97 + i) * 11, "title": f"V{i}"}
                                   for i in range(10)]}
    got = youtube.fetch_playlist("https://www.youtube.com/playlist?list=PLabcdefghij12345",
                                 limit=3)
    assert len(got) == 3
    assert fake_ytdlp.seen["playlistend"] == 3


def test_fetch_playlist_rejects_a_non_playlist_url(fake_ytdlp):
    with pytest.raises(youtube.YouTubeError, match="playlist URL"):
        youtube.fetch_playlist("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_fetch_playlist_raises_when_empty(fake_ytdlp):
    fake_ytdlp.info = {"entries": []}
    with pytest.raises(youtube.YouTubeError, match="no playable videos"):
        youtube.fetch_playlist("https://www.youtube.com/playlist?list=PLabcdefghij12345")


def test_fetch_playlist_without_ytdlp_is_actionable(monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "yt_dlp":
            raise ImportError("no yt_dlp")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(youtube.YouTubeError, match="pip install yt-dlp"):
        youtube.fetch_playlist("https://www.youtube.com/playlist?list=PLabcdefghij12345")


# ------------------------------ watch-page JSON ------------------------------

def test_balanced_json_survives_embedded_brace_semicolon():
    """A non-greedy regex would stop at the first '};' inside the payload."""
    html = 'var ytInitialPlayerResponse = {"a": "trap };", "b": {"c": 1}};</script>'
    got = youtube._extract_assignment(html, "ytInitialPlayerResponse")
    assert got == {"a": "trap };", "b": {"c": 1}}


def test_extract_assignment_missing_returns_none():
    assert youtube._extract_assignment("<html></html>", "ytInitialData") is None


def test_page_config_reads_innertube_keys():
    html = ('var ytInitialData = {"ok": true};'
            '"INNERTUBE_API_KEY":"AIzaTESTKEY","INNERTUBE_CLIENT_VERSION":"2.2025",'
            '"VISITOR_DATA":"CgtWSVNJVE9S"')
    cfg = youtube._page_config(html)
    assert cfg["initial_data"] == {"ok": True}
    assert cfg["api_key"] == "AIzaTESTKEY"
    assert cfg["client_version"] == "2.2025"
    assert cfg["visitor_data"] == "CgtWSVNJVE9S"


def test_page_config_defaults_client_version():
    assert youtube._page_config("")["client_version"] == "2.20240101.00.00"


# ------------------------------ transcript ------------------------------

def _track(lang, vss=None, name=None, url="https://t/base"):
    t = {"languageCode": lang, "baseUrl": url}
    if vss:
        t["vssId"] = vss
    if name:
        t["name"] = {"simpleText": name}
    return t


def test_caption_track_prefers_manual_english_over_auto():
    tracks = [_track("de"), _track("en", vss="a.en"), _track("en", vss=".en")]
    assert youtube._pick_caption_track(tracks)["vssId"] == ".en"


def test_caption_track_falls_back_to_auto_english():
    tracks = [_track("de"), _track("en", vss="a.en")]
    assert youtube._pick_caption_track(tracks)["languageCode"] == "en"


def test_caption_track_falls_back_to_named_english():
    tracks = [_track("de"), _track("mul", name="English (auto)")]
    assert youtube._pick_caption_track(tracks)["languageCode"] == "mul"


def test_caption_track_falls_back_to_first():
    tracks = [_track("de"), _track("fr")]
    assert youtube._pick_caption_track(tracks)["languageCode"] == "de"


def test_caption_track_empty_is_none():
    assert youtube._pick_caption_track([]) is None


def test_transcript_from_json3_joins_and_cleans():
    payload = {"events": [
        {"segs": [{"utf8": "Hello"}, {"utf8": "world."}]},
        {"segs": [{"utf8": "[Music]"}]},
        {"segs": [{"utf8": "(sound effect)"}]},
        {"segs": [{"utf8": "Next.Sentence"}]},
        {"nosegs": True},
    ]}
    got = youtube._transcript_from_json3(payload)
    assert got == "Hello world. Next. Sentence"


def test_transcript_strips_note_glyphs_but_keeps_lyrics():
    """Matching the extension: ♪ markers go, the words between them stay."""
    payload = {"events": [{"segs": [{"utf8": "♪ never gonna give you up ♪"}]}]}
    assert youtube._transcript_from_json3(payload) == "never gonna give you up"


def test_transcript_keeps_words_the_extension_would_scrub():
    """The extension strips "Comments"/"Description"/"Search" anywhere in the text to
    clean its DOM fallback. The API path has no such chrome, so real speech survives."""
    payload = {"events": [{"segs": [{"utf8": "Read the description and the comments below."}]}]}
    got = youtube._transcript_from_json3(payload)
    assert got == "Read the description and the comments below."


# ------------------------------ comments ------------------------------

def _renderer_payload(author, text):
    return {"commentThreadRenderer": {"comment": {"commentRenderer": {
        "authorText": {"simpleText": author},
        "contentText": {"runs": [{"text": text}]},
        "voteCount": {"simpleText": "12"},
        "publishedTimeText": {"simpleText": "3 days ago"},
    }}}}


_entity_payload = conftest.yt_entity_payload


def test_harvest_reads_the_modern_entity_shape():
    """The extension's parser misses commentEntityPayload, which is where current
    InnerTube responses put comment bodies (and the only source of like counts)."""
    payload = {"frameworkUpdates": {"entityBatchUpdate": {"mutations": [
        {"payload": _entity_payload("Ada", "First!")},
    ]}}}
    out = []
    youtube._harvest_comments(payload, out, set(), 100)
    assert out == [{"author": "Ada", "text": "First!",
                    "likes": "412", "published": "2 months ago"}]


def test_harvest_reads_the_classic_renderer_shape():
    out = []
    youtube._harvest_comments(_renderer_payload("Grace", "Nice video"), out, set(), 100)
    assert out == [{"author": "Grace", "text": "Nice video",
                    "likes": "12", "published": "3 days ago"}]


def test_harvest_dedupes_the_two_shapes_against_each_other():
    """A single response can describe the same comment as a renderer AND as an entity
    mutation; it must land in the output once."""
    payload = {
        "contents": [_renderer_payload("Ada", "Same words here")],
        "frameworkUpdates": {"entityBatchUpdate": {"mutations": [
            {"payload": _entity_payload("Ada", "Same words here")},
        ]}},
    }
    out = []
    youtube._harvest_comments(payload, out, set(), 100)
    assert len(out) == 1


def test_harvest_respects_the_limit():
    payload = {"items": [_renderer_payload(f"A{i}", f"comment {i}") for i in range(10)]}
    out = []
    youtube._harvest_comments(payload, out, set(), 3)
    assert len(out) == 3


def test_harvest_skips_empty_and_too_short():
    payload = {"items": [_entity_payload("Ada", ""), _entity_payload("Bob", "x"),
                         _entity_payload("Cy", "ok")]}
    out = []
    youtube._harvest_comments(payload, out, set(), 100)
    assert [c["author"] for c in out] == ["Cy"]


def test_next_continuation_takes_the_last_item():
    """Reply tokens come first; the 'load more top-level comments' token comes last."""
    payload = {"onResponseReceivedEndpoints": [{"appendContinuationItemsAction": {
        "continuationItems": [
            {"continuationItemRenderer": {"continuationEndpoint": {
                "continuationCommand": {"token": "reply-token"}}}},
            {"continuationItemRenderer": {"continuationEndpoint": {
                "continuationCommand": {"token": "more-comments"}}}},
        ]}}]}
    assert youtube._next_continuation(payload) == "more-comments"


def test_next_continuation_none_when_exhausted():
    assert youtube._next_continuation({"onResponseReceivedActions": []}) is None


def test_walk_finds_only_comment_continuations():
    data = {
        "related": {"continuationCommand": {"token": "related-token"}},
        "comments": {"commentsHeaderRenderer": {},
                     "continuationCommand": {"token": "comment-token"}},
    }
    assert youtube._walk_comment_continuations(data) == ["comment-token"]


# ------------------------------ output format ------------------------------

_META = {"title": "How Engines Work", "channel": "Garage Lab",
         "published": "2024-03-12", "views": "1,240,113",
         "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}


def test_format_full():
    comments = [
        {"author": "Ada", "text": "Great explainer", "likes": "412", "published": "2 months ago"},
        {"author": "Bob", "text": "Second", "likes": "", "published": ""},
    ]
    got = youtube.format_video_text(_META, "The engine turns.", comments)
    assert got == (
        "How Engines Work | Garage Lab | 2024-03-12 | 1,240,113 views\n"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
        "\n"
        "The engine turns.\n"
        "\n"
        "--- Comments ---\n"
        "\n"
        "1. Ada (412 likes, 2 months ago): Great explainer\n"
        "\n"
        "2. Bob: Second"
    )


def test_format_omits_missing_header_fields():
    meta = {"title": "Untitled", "channel": "", "published": "", "views": "", "url": ""}
    got = youtube.format_video_text(meta, "Body.", [])
    assert got.startswith("Untitled\n\nBody.")


def test_format_empty_states_match_the_extension():
    got = youtube.format_video_text(_META, "", [])
    assert "(No transcript available)" in got
    assert "(No comments available)" in got


def test_format_without_comments_has_no_comments_section():
    got = youtube.format_video_text(_META, "Body.", [], include_comments=False)
    assert "--- Comments ---" not in got
    assert got.endswith("Body.")


# ------------------------------ fetch chain ------------------------------

# The payloads and the network fake live in conftest.py, because test_batch.py drives
# the same chain to prove Preview and Run share one crawl.
_PLAYER = conftest._YT_PLAYER
_JSON3 = conftest._YT_JSON3
_INITIAL_DATA = conftest._YT_INITIAL_DATA
_watch_html = conftest.yt_watch_html


@pytest.fixture
def fake_net(youtube_net):
    """This suite's long-standing name for conftest's ``youtube_net``."""
    return youtube_net


def test_fetch_video_via_requests(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    got = youtube.fetch_video("https://youtu.be/dQw4w9WgXcQ", include_comments=True)
    assert got["via"] == "requests"
    assert got["title"] == "How Engines Work"
    assert got["views"] == "1,240,113"
    assert got["transcript"] == "The engine turns."
    assert [c["author"] for c in got["comments"]] == ["Ada"]
    assert "--- Comments ---" in got["text"]
    assert got["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_fetch_video_skips_comments_when_not_wanted(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=False)
    assert got["comments"] == []
    assert fake_net["post"] == []
    assert "--- Comments ---" not in got["text"]


def test_fetch_video_prefers_bright_data_when_configured(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "bd-token")
    seen = []

    def fake_bd(url, method="GET", body=None, headers=None, timeout=60):
        seen.append((url, method))
        if "timedtext" in url:
            return json.dumps(_JSON3)
        if "youtubei" in url:
            return json.dumps({})
        return _watch_html()

    monkeypatch.setattr(core, "brightdata_request", fake_bd)
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=False)
    assert got["via"] == "Bright Data"
    assert fake_net["get"] == []          # the direct rung was never reached
    assert seen and seen[0][1] == "GET"


def test_fetch_video_falls_through_a_blocked_rung(monkeypatch):
    """A consent wall yields no player JSON; the chain must keep going."""
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "bd-token")
    monkeypatch.setattr(core, "brightdata_request",
                        lambda *a, **k: "<html>Before you continue to YouTube</html>")
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: json.dumps(_JSON3) if "timedtext" in url
                        else _watch_html())
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=False)
    assert got["via"] == "requests"
    assert any("Bright Data" in e for e in got["errors"])


def test_fetch_video_reports_every_rung_when_all_fail(monkeypatch):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: "<html>nope</html>")
    with pytest.raises(youtube.YouTubeError) as exc:
        youtube.fetch_video("dQw4w9WgXcQ")
    assert "requests:" in str(exc.value)
    assert "yt-dlp" in str(exc.value)


def test_ytdlp_rung_normalises_into_the_same_result(monkeypatch, real_ytdlp_rung):
    """The optional third rung takes a completely different path; it must still hand
    back the shape the callers expect."""
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {
                "title": "How Engines Work",
                "uploader": "Garage Lab",
                "upload_date": "20240312",
                "view_count": 1240113,
                "subtitles": {"en": [{"ext": "json3", "url": "https://timedtext/x"}]},
                "comments": [{"author": "Ada", "text": "Great explainer", "like_count": 412}],
            }

    monkeypatch.setitem(__import__("sys").modules, "yt_dlp",
                        type("m", (), {"YoutubeDL": FakeYDL}))
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: json.dumps(_JSON3))
    got = real_ytdlp_rung("dQw4w9WgXcQ",
                          "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                          True, True, 100, None)
    assert got["via"] == "yt-dlp"
    assert got["title"] == "How Engines Work"
    assert got["channel"] == "Garage Lab"
    assert got["published"] == "2024-03-12"
    assert got["views"] == "1,240,113"
    assert got["transcript"] == "The engine turns."
    assert got["comments"] == [{"author": "Ada", "text": "Great explainer",
                                "likes": "412", "published": ""}]


def test_transcript_falls_through_while_comments_are_kept(monkeypatch, fake_net):
    """The real-world case: YouTube answers server-side caption requests with an empty
    200 body, so the watch page yields comments only and yt-dlp supplies the
    transcript. Both halves must end up in one result."""
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: "" if "timedtext" in url else _watch_html())

    calls = []

    def fake_ytdlp(vid, url, want_transcript, want_comments, max_comments, on_progress):
        calls.append((want_transcript, want_comments))
        return {"video_id": vid, "url": url, "title": "How Engines Work",
                "channel": "Garage Lab", "published": "2024-03-12", "views": "1",
                "transcript": "The engine turns.", "comments": [], "via": "yt-dlp",
                "transcript_error": "", "comment_error": ""}

    monkeypatch.setattr(youtube, "_fetch_via_ytdlp", fake_ytdlp)
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    # yt-dlp was asked for the transcript only — the comments were already in hand.
    assert calls == [(True, False)]
    assert got["transcript"] == "The engine turns."
    assert [c["author"] for c in got["comments"]] == ["Ada"]
    assert got["via"] == "requests + yt-dlp"
    assert got["errors"] == []          # a half that was recovered isn't a problem
    assert "The engine turns." in got["text"] and "Ada" in got["text"]


def test_chain_stops_once_both_halves_are_in_hand(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    monkeypatch.setattr(youtube, "_fetch_via_ytdlp",
                        lambda *a, **k: pytest.fail("yt-dlp should not have been reached"))
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert got["via"] == "requests"


def test_empty_caption_body_is_reported_as_a_transcript_error(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: "" if "timedtext" in url else _watch_html())
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert got["transcript"] == ""
    assert any("proof-of-origin" in e for e in got["errors"])
    assert got["comments"]              # the half that worked is still returned


def test_fetch_video_rejects_a_non_video_url():
    with pytest.raises(youtube.YouTubeError):
        youtube.fetch_video("https://example.com/nope")


def test_fetch_video_reports_missing_captions_without_failing(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    player = dict(_PLAYER)
    player.pop("captions")
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: _watch_html(player=player))
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert got["transcript"] == ""
    assert any("No captions" in e for e in got["errors"])
    assert got["comments"]                       # comments still collected
    assert "(No transcript available)" in got["text"]


def test_fetch_video_returns_metadata_when_a_video_has_neither(monkeypatch):
    """Captions off and comments disabled is a real video, not a failed fetch — the
    title and channel are still worth attaching."""
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    player = dict(_PLAYER)
    player.pop("captions")
    monkeypatch.setattr(
        youtube, "_requests_get",
        lambda url, timeout=45: _watch_html(player=player, initial={"none": True}))
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert got["title"] == "How Engines Work"
    assert got["transcript"] == "" and got["comments"] == []
    assert "(No transcript available)" in got["text"]
    assert "(No comments available)" in got["text"]


def test_fetch_video_clamps_max_comments(monkeypatch):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    seen = {}
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: json.dumps(_JSON3) if "timedtext" in url
                        else _watch_html())

    def spy(html, transport, max_comments, progress, stopped):
        seen["max"] = max_comments
        return []

    monkeypatch.setattr(youtube, "_fetch_comments", spy)
    youtube.fetch_video("dQw4w9WgXcQ", max_comments=99999)
    assert seen["max"] == youtube.MAX_MAX_COMMENTS


def test_fetch_video_emits_progress_phases(monkeypatch, fake_net):
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    phases = []
    youtube.fetch_video("dQw4w9WgXcQ", include_comments=True,
                        on_progress=lambda phase, **kw: phases.append(phase))
    assert phases[0] == "page"
    assert "transcript" in phases
    assert "comments" in phases


def test_comment_paging_stops_on_a_repeated_token(monkeypatch):
    """A continuation that hands back itself must not loop forever."""
    posts = []

    class T:
        name = "requests"

        def get(self, url):
            return ""

        def post_json(self, url, body):
            posts.append(body["continuation"])
            return {
                "frameworkUpdates": {"entityBatchUpdate": {"mutations": [
                    {"payload": _entity_payload(f"A{len(posts)}", f"c{len(posts)}")}]}},
                "onResponseReceivedActions": [{"appendContinuationItemsAction": {
                    "continuationItems": [{"continuationItemRenderer": {
                        "continuationEndpoint": {"continuationCommand": {"token": "loop"}}}}]}}],
            }

    html = ('var ytInitialData = ' + json.dumps({
        "comments": {"commentsHeader": {}, "continuationCommand": {"token": "loop"}}}) + ";"
        '"INNERTUBE_API_KEY":"AIzaTESTKEY"')
    got = youtube._fetch_comments(html, T(), 100, lambda *a, **k: None, lambda: False)
    assert posts == ["loop"]        # one page, then the repeat is refused
    assert len(got) == 1


def test_comment_paging_honours_should_stop(monkeypatch):
    class T:
        name = "requests"

        def post_json(self, url, body):
            raise AssertionError("should have stopped before requesting a page")

    html = ('var ytInitialData = ' + json.dumps({
        "comments": {"commentsHeader": {}, "continuationCommand": {"token": "t1"}}}) + ";"
        '"INNERTUBE_API_KEY":"AIzaTESTKEY"')
    got = youtube._fetch_comments(html, T(), 100, lambda *a, **k: None, lambda: True)
    assert got == []


def test_fetch_comments_without_an_api_key_raises():
    with pytest.raises(youtube.YouTubeError):
        youtube._fetch_comments("<html></html>", None, 10,
                                lambda *a, **k: None, lambda: False)


# ------------------------------ cache ------------------------------
# The cache seeds fetch_video's working dict before the transport loop, so most of the
# behaviour here is the loop's existing per-half fall-through doing its job — that is
# the point of wiring it in there rather than bolting a second code path alongside.

@pytest.fixture
def requests_rung(monkeypatch, fake_net):
    """The requests rung and nothing above it. Every cache test asserts on `via`, so a
    stray Bright Data token would rename the contributor and break them all."""
    monkeypatch.setattr(core, "BRIGHTDATA_TOKEN", "")
    return fake_net


def _prime(max_comments=100, include_comments=True):
    """Populate the cache the way a real fetch does, and return the result."""
    return youtube.fetch_video("dQw4w9WgXcQ", include_comments=include_comments,
                               max_comments=max_comments)


def test_fetch_video_writes_the_cache(requests_rung):
    _prime(max_comments=100)
    entry = youtube_cache.get("dQw4w9WgXcQ")
    assert entry["transcript"] == "The engine turns."
    assert entry["comment_count"] == 1
    assert entry["comment_target"] == 100
    assert entry["title"] == "How Engines Work"


def test_a_full_cache_hit_makes_no_network_calls(monkeypatch, requests_rung):
    _prime()
    requests_rung["get"].clear(); requests_rung["post"].clear()

    def boom(*a, **k):
        raise AssertionError("a cache hit must not touch the network")

    monkeypatch.setattr(youtube, "_requests_get", boom)
    monkeypatch.setattr(youtube, "_requests_post_json", boom)
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True, max_comments=100)
    assert got["via"] == "cache"
    assert got["transcript"] == "The engine turns."
    assert [c["author"] for c in got["comments"]] == ["Ada"]
    assert "How Engines Work" in got["text"]


def test_a_cache_hit_rerenders_text_for_this_call(requests_rung):
    """`text` is rendered at read time, never stored — otherwise a fetch cached with
    comments would keep serving them to a caller that asked for none."""
    _prime(max_comments=100)
    with_comments = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    without = youtube.fetch_video("dQw4w9WgXcQ", include_comments=False)
    assert "--- Comments ---" in with_comments["text"]
    assert "--- Comments ---" not in without["text"]
    assert without["via"] == "cache"


def test_a_smaller_comment_ask_is_a_full_hit(monkeypatch, requests_rung):
    _prime(max_comments=100)
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    assert youtube.fetch_video("dQw4w9WgXcQ", max_comments=20)["via"] == "cache"


def test_more_comments_refetches_only_the_comments_half(requests_rung):
    """The headline behaviour: the transcript half stays satisfied by the cache, so
    only the comment pager runs. No timedtext request is made a second time."""
    _prime(max_comments=100)
    requests_rung["get"].clear(); requests_rung["post"].clear()
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True, max_comments=500)
    assert got["via"] == "cache + requests"
    assert not any("timedtext" in u for u in requests_rung["get"])
    assert requests_rung["post"], "the comment pager should have run"
    assert got["transcript"] == "The engine turns."
    # …and the upgrade is recorded, so the next ask of 500 is free.
    assert youtube_cache.get("dQw4w9WgXcQ")["comment_target"] == 500


def test_refresh_bypasses_the_cache_but_still_writes(monkeypatch, requests_rung):
    _prime()
    player = json.loads(json.dumps(_PLAYER))
    player["videoDetails"]["title"] = "How Engines Work (2024 remaster)"
    monkeypatch.setattr(youtube, "_requests_get",
                        lambda url, timeout=45: json.dumps(_JSON3) if "timedtext" in url
                        else _watch_html(player=player))
    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True, refresh=True)
    assert "cache" not in (got["via"] or "")
    assert got["title"] == "How Engines Work (2024 remaster)"
    assert youtube_cache.get("dQw4w9WgXcQ")["title"] == "How Engines Work (2024 remaster)"


def test_refresh_without_comments_keeps_the_cached_comments(requests_rung):
    _prime(max_comments=100)
    youtube.fetch_video("dQw4w9WgXcQ", include_comments=False, refresh=True)
    assert youtube_cache.get("dQw4w9WgXcQ")["comment_count"] == 1


def test_a_stopped_run_does_not_cache_comments(monkeypatch, requests_rung):
    calls = {"n": 0}

    def stop_after_the_page():
        calls["n"] += 1
        return calls["n"] > 3

    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True,
                              should_stop=stop_after_the_page)
    entry = youtube_cache.get("dQw4w9WgXcQ")
    # The transcript either parsed whole or not at all, so it is safe to keep. A
    # truncated comment list is not: once stored it would be served forever.
    assert got["transcript"] == "The engine turns."
    assert entry["transcript"] == "The engine turns."
    assert "comments" not in entry


def test_a_failed_comment_half_is_not_cached_and_is_retried(monkeypatch, requests_rung):
    def boom(html, transport, max_comments, progress, stopped):
        raise RuntimeError("InnerTube said no")

    # A nested context, not monkeypatch.undo(): undo() reverts every patch on this
    # instance, including the fake_net stubs requests_rung installed, so the retry
    # below went to the real youtube.com and came back with its actual comments.
    with monkeypatch.context() as broken:
        broken.setattr(youtube, "_fetch_comments", boom)
        youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert "comments" not in youtube_cache.get("dQw4w9WgXcQ")

    got = youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert [c["author"] for c in got["comments"]] == ["Ada"]


def test_a_missing_transcript_is_not_cached_as_empty(monkeypatch, requests_rung):
    """"No captions exist" and "the rung was blocked today" look identical from here,
    so a blank transcript must never harden into a stored answer."""
    player = json.loads(json.dumps(_PLAYER))
    player.pop("captions")
    # Nested context rather than monkeypatch.undo(), for the reason given in
    # test_a_failed_comment_half_is_not_cached_and_is_retried.
    with monkeypatch.context() as no_captions:
        no_captions.setattr(youtube, "_requests_get",
                            lambda url, timeout=45: _watch_html(player=player))
        youtube.fetch_video("dQw4w9WgXcQ", include_comments=True)
    assert "transcript" not in youtube_cache.get("dQw4w9WgXcQ")

    assert youtube.fetch_video("dQw4w9WgXcQ")["transcript"] == "The engine turns."


def test_a_cache_hit_emits_a_cache_progress_phase(requests_rung):
    _prime()
    phases = []
    youtube.fetch_video("dQw4w9WgXcQ", include_comments=True,
                        on_progress=lambda phase, **kw: phases.append((phase, kw)))
    assert phases[0][0] == "cache"
    assert phases[0][1]["transcript"] is True
    assert phases[0][1]["need_comments"] is False


def test_a_partial_hit_reports_what_it_still_needs(requests_rung):
    _prime(max_comments=100)
    phases = []
    youtube.fetch_video("dQw4w9WgXcQ", include_comments=True, max_comments=500,
                        on_progress=lambda phase, **kw: phases.append((phase, kw)))
    assert phases[0][0] == "cache"
    assert phases[0][1]["need_transcript"] is False
    assert phases[0][1]["need_comments"] is True
    assert "comments" in [p for p, _ in phases]


# ------------------------------ library round-trip ------------------------------

def test_youtube_item_survives_xml_export_and_import(tmp_path):
    """A library item carries only id/type/label/content/filename through XML, so the
    watch URL rides in `filename` the way 'url' items do rather than in a new field."""
    meta = {"title": "How Engines Work", "channel": "Garage Lab", "published": "2024-03-12",
            "views": "1,240,113", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    text = youtube.format_video_text(
        meta, "The engine turns.",
        [{"author": "Ada", "text": "Great explainer", "likes": "412", "published": "2 months ago"}])
    lib = core._new_library("Videos")
    lib["items"].append(core._new_library_item(
        item_type="youtube", label=meta["title"], content=text, filename=meta["url"]))

    path = tmp_path / "videos.xml"
    path.write_bytes(core.library_to_xml_bytes(lib))
    back = core.library_from_xml_file(str(path))

    item = back["items"][0]
    assert item["type"] == "youtube"
    assert item["label"] == "How Engines Work"
    assert item["filename"] == meta["url"]
    assert "--- Comments ---" in item["content"]
    assert "1. Ada (412 likes, 2 months ago): Great explainer" in item["content"]


# ------------------------------ chat attachments ------------------------------

def _chat(**over):
    chat = {"messages": [{"role": "user", "content": "What did they say?"}],
            "attachments": []}
    chat.update(over)
    return chat


def test_attachment_block_empty_without_attachments():
    assert logic.build_attachment_block(_chat()) == ""
    assert logic.build_attachment_block({}) == ""


def test_attachment_block_skips_empty_content():
    chat = _chat(attachments=[{"id": "a", "type": "url", "label": "Blank", "content": "  "}])
    assert logic.build_attachment_block(chat) == ""


def test_attachment_block_renders_items_with_source():
    chat = _chat(attachments=[
        {"id": "a", "type": "youtube", "label": "How Engines Work",
         "content": "The engine turns.", "source": "https://youtu.be/x"},
    ])
    block = logic.build_attachment_block(chat)
    assert 'type="youtube"' in block
    assert 'label="How Engines Work"' in block
    assert 'source="https://youtu.be/x"' in block
    assert "The engine turns." in block


def test_attachment_block_escapes_xml():
    chat = _chat(attachments=[{"id": "a", "type": "write", "label": "L",
                               "content": "a < b & c"}])
    block = logic.build_attachment_block(chat)
    assert "a &lt; b &amp; c" in block


def test_inject_attachments_sits_before_the_last_user_turn():
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"}]
    out = logic.inject_attachments(messages, "BLOCK")
    assert out[-1]["content"] == "second"
    assert out[-2] == {"role": "system", "content": "BLOCK"}
    assert len(out) == len(messages) + 1


def test_inject_attachments_empty_block_is_a_noop():
    messages = [{"role": "user", "content": "hi"}]
    assert logic.inject_attachments(messages, "") is messages


def test_resolve_attachments_hands_material_to_rag_regardless_of_size():
    """Under RAG, retrieval owns the pinned material outright — at any size.

    There used to be a size threshold below which the block was ALSO sent whole, which
    spent the context twice on text that collect_rag_inputs had already put in the
    corpus. Both halves of this are the contract now.
    """
    for content in ("short", "x" * 50_000):
        chat = _chat(attachments=[{"id": "a", "type": "file", "label": "Doc",
                                   "content": content}])
        assert logic.resolve_attachments(chat, rag_active=True) == ""
        assert content in logic.resolve_attachments(chat, rag_active=False)


def test_resolve_attachments_empty_when_nothing_pinned():
    assert logic.resolve_attachments(_chat(), rag_active=False) == ""


def test_pinned_attachments_join_the_rag_corpus():
    chat = _chat(attachments=[{"id": "a", "type": "url", "label": "Page",
                               "content": "pinned body"}])
    question, data_text = logic.collect_rag_inputs(chat)
    assert question == "What did they say?"
    assert "pinned body" in data_text


def test_pinned_attachments_survive_isolation():
    """Isolation scopes history, not what the user pinned to the chat."""
    chat = _chat(isolated=True,
                 attachments=[{"id": "a", "type": "url", "label": "Page",
                               "content": "pinned body"}])
    _q, data_text = logic.collect_rag_inputs(chat)
    assert "pinned body" in data_text


def test_pinned_attachments_reach_rag_before_the_first_message():
    chat = {"messages": [], "attachments": [
        {"id": "a", "type": "url", "label": "Page", "content": "pinned body"}]}
    question, data_text = logic.collect_rag_inputs(chat)
    assert question == ""
    assert data_text == "Page\npinned body"


def test_inline_data_and_pinned_attachments_combine():
    chat = _chat(messages=[{"role": "user",
                            "content": "<Data>\n  <Notes>inline</Notes>\n</Data>\n\nQ?"}],
                 attachments=[{"id": "a", "type": "url", "label": "Page",
                               "content": "pinned body"}])
    question, data_text = logic.collect_rag_inputs(chat)
    assert question == "Q?"
    assert "inline" in data_text and "pinned body" in data_text


def test_new_chat_has_an_attachments_list():
    assert logic.create_chat_dict()["attachments"] == []
