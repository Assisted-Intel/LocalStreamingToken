#!/usr/bin/env python3
"""The /api/youtube/* routes.

Two things are worth pinning here that the unit tests can't reach: the playlist route's
frame-per-video shape (the composer stages one attachment per `video` frame, so an
all-or-nothing `complete` would defeat the feature), and that a dead video inside a
playlist is survivable.
"""

import pytest

from app import core, youtube, youtube_cache
from conftest import all_of, events, first, sse_frames

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLabcdefghij12345"


def _video(n, title=None):
    vid = chr(ord("a") + n) * 11
    return {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title or f"Episode {n}"}


def _result(vid, title, text="body", via="requests"):
    return {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title, "text": text, "via": via, "errors": [],
            "comments": [{"author": "Ada", "text": "hi"}],
            "transcript": "the transcript"}


@pytest.fixture
def fake_youtube(monkeypatch):
    """Replace both fetch entry points, recording the kwargs each route passed."""
    seen = {"playlist": [], "videos": []}

    def fake_playlist(url, limit=0, on_progress=None, should_stop=None):
        seen["playlist"].append({"url": url, "limit": limit})
        return [_video(0), _video(1), _video(2)]

    def fake_video(url, **kw):
        seen["videos"].append({"url": url, **kw})
        vid = youtube.parse_video_id(url) or "0" * 11
        return _result(vid, f"Title for {vid[:3]}")

    monkeypatch.setattr(youtube, "fetch_playlist", fake_playlist)
    monkeypatch.setattr(youtube, "fetch_video", fake_video)
    return seen


# --------------------------- playlist route ---------------------------

def test_playlist_route_emits_one_video_frame_per_video(client, fake_youtube):
    frames = sse_frames(client.get(f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"))
    assert events(frames) == ["start", "begin", "playlist",
                              "video", "video", "video", "complete", "done"]

    assert first(frames, "playlist")["total"] == 3
    videos = all_of(frames, "video")
    assert [v["index"] for v in videos] == [1, 2, 3]
    assert all(v["total"] == 3 for v in videos)
    # Each frame carries its own text — that is what makes one chip per video possible.
    assert len({v["url"] for v in videos}) == 3
    assert all(v["text"] for v in videos)

    done = first(frames, "complete")
    assert (done["ok"], done["failed"], done["stopped"]) == (3, 0, False)


def test_playlist_route_survives_a_dead_video(client, fake_youtube, monkeypatch):
    """One private or region-blocked video must not cost the user the rest."""
    def sometimes(url, **kw):
        if url.endswith("b" * 11):
            raise youtube.YouTubeError("Video unavailable")
        return _result(youtube.parse_video_id(url), "ok")

    monkeypatch.setattr(youtube, "fetch_video", sometimes)
    frames = sse_frames(client.get(f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"))

    assert len(all_of(frames, "video")) == 2
    bad, = all_of(frames, "video_error")
    assert bad["index"] == 2
    assert "unavailable" in bad["message"].lower()

    done = first(frames, "complete")
    assert (done["ok"], done["failed"]) == (2, 1)
    assert done["errors"]


def test_playlist_route_forwards_limit_comments_max_and_refresh(client, fake_youtube):
    sse_frames(client.get(
        f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"
        "&limit=2&comments=0&max=250&refresh=1"))
    assert fake_youtube["playlist"][0]["limit"] == 2
    first_video = fake_youtube["videos"][0]
    assert first_video["include_comments"] is False
    assert first_video["max_comments"] == 250
    assert first_video["refresh"] is True


def test_playlist_route_defaults_to_the_whole_playlist_and_the_cache(client, fake_youtube):
    sse_frames(client.get(f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"))
    assert fake_youtube["playlist"][0]["limit"] == 0        # 0 = every video
    assert fake_youtube["videos"][0]["refresh"] is False    # cache on unless asked


def test_playlist_route_rejects_a_url_with_no_list_id(client, fake_youtube):
    r = client.get("/api/youtube/fetch-playlist?url=https://youtu.be/dQw4w9WgXcQ")
    assert r.status_code == 400
    assert "playlist id" in r.get_json()["error"]


def test_playlist_route_requires_a_url(client, fake_youtube):
    assert client.get("/api/youtube/fetch-playlist").status_code == 400


def test_playlist_route_reports_a_missing_ytdlp_as_an_error_frame(client, monkeypatch):
    """Playlists are yt-dlp-only, so this is the one dependency failure a user will
    actually hit — it has to arrive as a legible message, not a dead button."""
    def refuse(*a, **kw):
        raise youtube.YouTubeError("Listing a playlist needs yt-dlp: pip install yt-dlp")

    monkeypatch.setattr(youtube, "fetch_playlist", refuse)
    frames = sse_frames(client.get(f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"))
    assert "pip install yt-dlp" in first(frames, "error")["message"]


def test_playlist_route_announces_a_unique_run_id(client, fake_youtube):
    """Cancel targets the id the run announces; a composed one would name nothing."""
    ids = set()
    for _ in range(2):
        frames = sse_frames(client.get(f"/api/youtube/fetch-playlist?url={PLAYLIST_URL}"))
        run_id = first(frames, "start")["run_id"]
        assert run_id.startswith("youtube-playlist-")
        ids.add(run_id)
    assert len(ids) == 2


# --------------------------- refresh passthrough ---------------------------

def test_single_video_route_forwards_refresh(client, fake_youtube):
    sse_frames(client.get(
        "/api/youtube/fetch?url=https://youtu.be/dQw4w9WgXcQ&refresh=1"))
    assert fake_youtube["videos"][0]["refresh"] is True


def test_single_video_route_defaults_to_using_the_cache(client, fake_youtube):
    frames = sse_frames(client.get("/api/youtube/fetch?url=https://youtu.be/dQw4w9WgXcQ"))
    assert fake_youtube["videos"][0]["refresh"] is False
    assert first(frames, "begin")["refresh"] is False


def test_single_video_route_reports_a_cache_hit(client, monkeypatch):
    monkeypatch.setattr(youtube, "fetch_video",
                        lambda url, **kw: _result("dQw4w9WgXcQ", "V", via="cache"))
    frames = sse_frames(client.get("/api/youtube/fetch?url=https://youtu.be/dQw4w9WgXcQ"))
    assert first(frames, "complete")["from_cache"] is True


def test_library_route_forwards_refresh(client, fake_youtube):
    lib_id = client.post("/api/libraries", json={"name": "Videos"}).get_json()["library"]["id"]
    sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube?url=https://youtu.be/dQw4w9WgXcQ&refresh=1"))
    assert fake_youtube["videos"][0]["refresh"] is True


# --------------------------- library playlist route ---------------------------
#
# The library used to take single videos only, so a playlist URL either silently
# collapsed to the one embedded video or failed outright. These pin the thing that
# makes the feature worth having: one ITEM per video, written as each lands.

def _new_lib(client, name="Videos"):
    return client.post("/api/libraries", json={"name": name}).get_json()["library"]["id"]


def test_library_playlist_route_adds_one_item_per_video(client, fake_youtube):
    lib_id = _new_lib(client)
    frames = sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"))

    assert events(frames) == ["start", "begin", "playlist",
                              "video", "video", "video", "complete", "done"]
    done = first(frames, "complete")
    assert len(done["added_items"]) == 3
    assert (done["ok"], done["failed"]) == (3, 0)

    items = client.get("/api/libraries").get_json()["libraries"]
    lib, = [l for l in items if l["id"] == lib_id]
    assert len(lib["items"]) == 3
    assert all(it["type"] == "youtube" for it in lib["items"])
    # The watch URL rides in `filename` exactly like a 'url' item, so it round-trips
    # through XML export/import and renders as a clickable link.
    assert len({it["filename"] for it in lib["items"]}) == 3
    assert all(it["content"] for it in lib["items"])


def test_library_playlist_route_appends_incrementally(client, fake_youtube, monkeypatch):
    """A long playlist that dies at video 38 must keep the first 37. Items are appended
    per video, so each fetch observes the items the PREVIOUS ones already wrote —
    rather than everything landing in one batch at the end."""
    lib_id = _new_lib(client)
    seen = []

    def spy(url, **kw):
        lib, = [l for l in client.get("/api/libraries").get_json()["libraries"]
                if l["id"] == lib_id]
        seen.append(len(lib["items"]))
        vid = youtube.parse_video_id(url) or "0" * 11
        return _result(vid, f"Title for {vid[:3]}")

    monkeypatch.setattr(youtube, "fetch_video", spy)
    sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"))

    assert seen == [0, 1, 2]


def test_library_playlist_route_survives_a_dead_video(client, fake_youtube, monkeypatch):
    lib_id = _new_lib(client)

    def sometimes(url, **kw):
        if url.endswith("b" * 11):
            raise youtube.YouTubeError("Video unavailable")
        return _result(youtube.parse_video_id(url), "ok")

    monkeypatch.setattr(youtube, "fetch_video", sometimes)
    frames = sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"))

    bad, = all_of(frames, "video_error")
    assert "unavailable" in bad["message"].lower()
    done = first(frames, "complete")
    assert (done["ok"], done["failed"]) == (2, 1)

    lib, = [l for l in client.get("/api/libraries").get_json()["libraries"]
            if l["id"] == lib_id]
    assert len(lib["items"]) == 2      # the survivors were kept


def test_library_playlist_route_does_not_echo_transcripts_twice(client, monkeypatch):
    """Each transcript may be 200k chars. Repeating it on the per-video frame, and then
    shipping the whole library alongside `added_items` on `complete`, put tens of
    megabytes on a single SSE `data:` line for a long playlist — enough to hang the tab
    right before the user clicks Compile."""
    lib_id = _new_lib(client)
    big = "x" * 200_000

    monkeypatch.setattr(youtube, "fetch_playlist",
                        lambda url, **kw: [_video(0), _video(1), _video(2)])
    monkeypatch.setattr(youtube, "fetch_video",
                        lambda url, **kw: _result(youtube.parse_video_id(url), "T", text=big))

    frames = sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"))

    for v in all_of(frames, "video"):
        assert "item" not in v and "text" not in v
        assert v["transcript_chars"]          # still reports size, just not the payload

    done = first(frames, "complete")
    assert "library" not in done              # the whole library is never echoed back
    # added_items DOES keep its content: makeLibItem renders it into an editable
    # textarea, and sending it empty would let the next autosave write the emptiness back.
    assert [len(it["content"]) for it in done["added_items"]] == [len(big)] * 3


def test_library_playlist_route_forwards_limit_comments_max_and_refresh(client, fake_youtube):
    lib_id = _new_lib(client)
    sse_frames(client.get(
        f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"
        "&limit=2&comments=0&max=250&refresh=1"))
    assert fake_youtube["playlist"][0]["limit"] == 2
    first_video = fake_youtube["videos"][0]
    assert first_video["include_comments"] is False
    assert first_video["max_comments"] == 250
    assert first_video["refresh"] is True


def test_library_playlist_route_rejects_a_url_with_no_list_id(client, fake_youtube):
    lib_id = _new_lib(client)
    r = client.get(f"/api/libraries/{lib_id}/add-youtube-playlist"
                   "?url=https://youtu.be/dQw4w9WgXcQ")
    assert r.status_code == 400
    assert "playlist id" in r.get_json()["error"]


def test_library_playlist_route_404s_for_an_unknown_library(client, fake_youtube):
    r = client.get(f"/api/libraries/nope/add-youtube-playlist?url={PLAYLIST_URL}")
    assert r.status_code == 404


def test_library_playlist_route_announces_a_unique_run_id(client, fake_youtube):
    lib_id = _new_lib(client)
    ids = set()
    for _ in range(2):
        frames = sse_frames(client.get(
            f"/api/libraries/{lib_id}/add-youtube-playlist?url={PLAYLIST_URL}"))
        run_id = first(frames, "start")["run_id"]
        assert run_id.startswith("youtube-libplaylist-")
        ids.add(run_id)
    assert len(ids) == 2


# --------------------------- cache routes ---------------------------

def test_cache_stats_start_empty(client):
    body = client.get("/api/youtube/cache").get_json()
    assert body["entries"] == 0 and body["bytes"] == 0


def test_cache_stats_and_clear_round_trip(client):
    youtube_cache.put(
        {"video_id": "dQw4w9WgXcQ", "url": "u", "title": "V",
         "transcript": "words", "comments": [], "via": "requests"},
        include_comments=True, max_comments=100)

    body = client.get("/api/youtube/cache").get_json()
    assert body["entries"] == 1
    assert body["bytes"] > 0
    assert body["dir"] == str(core.YOUTUBE_CACHE_DIR)

    cleared = client.delete("/api/youtube/cache").get_json()
    assert cleared["removed"] == 1
    assert client.get("/api/youtube/cache").get_json()["entries"] == 0
