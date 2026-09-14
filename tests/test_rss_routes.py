#!/usr/bin/env python3
"""Route-level tests for the RSS/podcast source (app/server.py).

``rss_net`` (conftest) stands in for the network, so these drive the REAL rss.py and
rss_cache.py rather than a stub of them — the frame sequence, the library append and the
cache behaviour are all under test together, which is what the composer actually
exercises.
"""

import json

import pytest

from app import rss, rss_cache
from conftest import SRT_BODY, all_of, events, feed_xml, first, podcast_item, sse_frames

URL = "https://feeds.test/show.xml"
Q = f"url={URL}"


@pytest.fixture
def feed3(rss_net):
    """A three-episode feed where every episode has a working SRT."""
    rss_net["routes"][URL] = feed_xml([podcast_item(i) for i in (1, 2, 3)])
    for i in (1, 2, 3):
        rss_net["routes"][f"ep{i}.srt"] = SRT_BODY.encode()
    return rss_net


# ------------------------------ the listing route ------------------------------

def test_feed_route_returns_a_slim_manifest(client, feed3):
    body = client.get(f"/api/rss/feed?{Q}").get_json()
    assert body["title"] == "Test Show"
    assert len(body["items"]) == 3
    item = body["items"][0]
    assert item["has_transcript"] is True and item["has_media"] is True
    # The heavy fields must not ride along — bodies and transcript lists are megabytes
    # across a real feed and the caller wants a manifest.
    assert "body_html" not in item and "transcripts" not in item


def test_feed_route_honours_the_limit(client, feed3):
    body = client.get(f"/api/rss/feed?{Q}&limit=2").get_json()
    assert len(body["items"]) == 2
    assert body["total_available"] == 3


def test_feed_route_needs_a_url(client):
    assert client.get("/api/rss/feed").status_code == 400


def test_feed_route_reports_a_non_feed_as_a_400_with_a_real_message(client, rss_net):
    rss_net["routes"][URL] = b"<html>not a feed</html>"
    r = client.get(f"/api/rss/feed?{Q}")
    assert r.status_code == 400
    assert "RSS or Atom" in r.get_json()["error"]


# ------------------------------ fetch-feed ------------------------------

def test_fetch_feed_emits_one_episode_frame_each(client, feed3):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    assert events(frames)[:3] == ["start", "begin", "progress"]
    assert first(frames, "feed")["total"] == 3
    eps = all_of(frames, "episode")
    assert [e["title"] for e in eps] == ["Episode 1", "Episode 2", "Episode 3"]
    assert all("Gitmo Nation" in e["text"] for e in eps)
    assert first(frames, "complete")["ok"] == 3


def test_a_dead_transcript_url_still_yields_a_document(client, feed3):
    feed3["routes"]["ep2.srt"] = RuntimeError("503")
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    # A dead transcript URL is not a dead EPISODE — it still yields a document with the
    # notes and an error note, which is the behaviour we want.
    assert len(all_of(frames, "episode")) == 3
    assert any(e["errors"] for e in all_of(frames, "episode"))


def test_a_genuinely_failing_episode_becomes_an_episode_error(client, feed3, monkeypatch):
    real = rss.fetch_episode
    calls = {"n": 0}

    def flaky(feed, item, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise rss.RSSError("that episode is gone")
        return real(feed, item, **kw)

    monkeypatch.setattr(rss, "fetch_episode", flaky)
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    assert len(all_of(frames, "episode")) == 2
    assert first(frames, "episode_error")["title"] == "Episode 2"
    done = first(frames, "complete")
    assert done["ok"] == 2 and done["failed"] == 1


def test_the_limit_is_forwarded(client, feed3):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=1"))
    assert len(all_of(frames, "episode")) == 1


def test_whisper_defaults_off(client, feed3, monkeypatch):
    """Nobody may opt into minutes of GPU per episode by omitting a query parameter."""
    seen = {}
    real = rss.fetch_episode
    monkeypatch.setattr(rss, "fetch_episode",
                        lambda f, i, **kw: (seen.update(kw), real(f, i, **kw))[1])
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=1"))
    assert seen["want_whisper"] is False


def test_whisper_notes_and_refresh_are_forwarded(client, feed3, monkeypatch):
    seen = {}
    real = rss.fetch_episode
    monkeypatch.setattr(rss, "fetch_episode",
                        lambda f, i, **kw: (seen.update(kw), real(f, i, **kw))[1])
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=1&whisper=1&notes=0&refresh=1"))
    assert seen["want_whisper"] is True
    assert seen["include_notes"] is False
    assert seen["refresh"] is True


def test_progress_frames_use_count_for_the_episode_position(client, feed3):
    """`total` is reserved for a phase's own units (seconds of audio, bytes)."""
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    positioned = [p for p in all_of(frames, "progress") if "index" in p]
    assert all(p["count"] == 3 for p in positioned)


def test_a_warning_frame_is_emitted_for_unstable_ids(client, rss_net):
    rss_net["routes"][URL] = feed_xml([podcast_item(i, guid=f"old-{i}", transcripts=())
                                       for i in range(6)])
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    rss_net["routes"][URL] = feed_xml([podcast_item(i, guid=f"new-{i}", transcripts=())
                                       for i in range(6)])
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    assert "episode ids changed" in first(frames, "warning")["message"]


def test_the_run_id_is_unique_and_prefixed(client, feed3):
    a = first(sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=1")), "start")["run_id"]
    b = first(sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=1")), "start")["run_id"]
    assert a != b and a.startswith("rss-feed-")


def test_fetch_feed_needs_a_url(client):
    assert client.get("/api/rss/fetch-feed").status_code == 400


def test_a_missing_feedparser_surfaces_as_an_error_frame(client, rss_net, monkeypatch):
    """EventSource cannot read a 500 body, so this has to arrive as a frame or the
    button is simply dead."""
    monkeypatch.setattr(rss, "_import_feedparser", lambda: (_ for _ in ()).throw(
        rss.RSSError("Reading a feed needs feedparser. Install it with:  "
                     "pip install feedparser")))
    rss_net["routes"][URL] = feed_xml()
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    assert "pip install feedparser" in first(frames, "error")["message"]


def test_from_cache_is_reported_on_a_second_run(client, feed3):
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    assert all(e["from_cache"] for e in all_of(frames, "episode"))
    assert first(frames, "feed")["from_cache"] is True


def test_widening_the_limit_refetches_only_the_new_ones(client, rss_net):
    """The check-for-new-episodes workflow: two cache hits and one real fetch."""
    rss_net["routes"][URL] = feed_xml([podcast_item(i) for i in (1, 2, 3)])
    for i in (1, 2, 3):
        rss_net["routes"][f"ep{i}.srt"] = SRT_BODY.encode()
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=2"))
    rss_net["get"].clear()
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&limit=3"))
    assert [e["from_cache"] for e in all_of(frames, "episode")] == [True, True, False]
    # Exactly one transcript download: the third episode's.
    assert [u for u in rss_net["get"] if u.endswith(".srt")] == \
        ["https://cdn.test/ep3.srt"]


# ------------------------------ single episode ------------------------------

def test_fetch_one_episode_by_guid(client, feed3):
    frames = sse_frames(client.get(
        f"/api/rss/fetch?{Q}&guid=https://show.test/ep2"))
    eps = all_of(frames, "episode")
    assert len(eps) == 1 and eps[0]["title"] == "Episode 2"


def test_fetch_needs_a_guid(client, feed3):
    assert client.get(f"/api/rss/fetch?{Q}").status_code == 400


def test_a_guid_outside_the_window_is_an_actionable_error(client, feed3):
    frames = sse_frames(client.get(f"/api/rss/fetch?{Q}&guid=https://show.test/ep99"))
    assert "episode limit" in first(frames, "error")["message"]


# ------------------------------ library append ------------------------------

def test_library_add_rss_appends_one_item_per_episode(client, feed3):
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    frames = sse_frames(client.get(f"/api/libraries/{lib['id']}/add-rss?{Q}"))
    done = first(frames, "complete")
    items = done["library"]["items"]
    assert len(items) == 3
    assert {i["type"] for i in items} == {"rss"}
    assert items[0]["label"] == "Episode 1"
    # The episode link goes in filename, so it round-trips through XML export/import.
    assert items[0]["filename"] == "https://show.test/ep1"


def test_library_items_are_appended_as_they_land_not_in_one_batch(client, feed3,
                                                                  monkeypatch):
    """A 40-episode Whisper import runs for hours; a crash at episode 38 must not throw
    away the first 37."""
    real = rss.fetch_episode
    calls = {"n": 0}

    def blow_up_on_the_third(feed, item, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise rss.RSSError("boom")
        return real(feed, item, **kw)

    monkeypatch.setattr(rss, "fetch_episode", blow_up_on_the_third)
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    sse_frames(client.get(f"/api/libraries/{lib['id']}/add-rss?{Q}"))
    saved = client.get("/api/libraries").get_json()["libraries"][0]
    assert len(saved["items"]) == 2


def test_library_add_rss_404s_on_a_missing_library(client, feed3):
    assert client.get(f"/api/libraries/nope/add-rss?{Q}").status_code == 404


def test_library_run_id_names_the_library(client, feed3):
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    run = first(sse_frames(
        client.get(f"/api/libraries/{lib['id']}/add-rss?{Q}&limit=1")), "start")["run_id"]
    assert run.startswith(f"rss-library-{lib['id']}-")


# ------------------------------ category / keyword filters ------------------------------

@pytest.fixture
def mixed(rss_net):
    """Five episodes alternating News / Sport, each with a working SRT."""
    cats = ["News", "Sport", "News", "Sport", "News"]
    rss_net["routes"][URL] = feed_xml(
        [podcast_item(i, categories=[cats[i - 1]]) for i in range(1, 6)],
        categories=[("News", "Politics")])
    for i in range(1, 6):
        rss_net["routes"][f"ep{i}.srt"] = SRT_BODY.encode()
    return rss_net


def test_the_listing_route_filters_and_reports_what_matched(client, mixed):
    body = client.get(f"/api/rss/feed?{Q}&categories=News").get_json()
    assert [i["title"] for i in body["items"]] == ["Episode 1", "Episode 3", "Episode 5"]
    assert body["matched"] == 3
    assert body["total_available"] == 5


def test_the_slim_manifest_carries_the_categories_to_filter_on(client, mixed):
    body = client.get(f"/api/rss/feed?{Q}").get_json()
    assert body["items"][0]["categories"] == ["News"]
    # The show's own, so the UI can offer them even when no episode is tagged.
    assert body["categories"] == ["News", "Politics"]


def test_fetch_feed_forwards_the_filter_from_the_query_string(client, mixed):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&categories=Sport"))
    assert [e["title"] for e in all_of(frames, "episode")] == ["Episode 2", "Episode 4"]


def test_the_begin_frame_echoes_the_filter_it_was_given(client, mixed):
    """The only record of what a run was actually asked to narrow on — a mistyped key
    would otherwise be an invisible no-op."""
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&categories=News&match=all"))
    assert first(frames, "begin")["filters"] == {
        "categories": ["News"], "keywords": [], "exclude": [], "match": "all"}
    assert first(frames, "feed")["matched"] == 3


def test_empty_filter_boxes_narrow_nothing(client, mixed):
    """sourceStream sends `categories=` for an untouched input, on every single import."""
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&categories=&keywords=&exclude="))
    assert len(all_of(frames, "episode")) == 5
    assert first(frames, "begin")["filters"] == {}


def test_the_filter_runs_before_the_limit_through_the_route(client, mixed):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&categories=News&limit=2"))
    assert [e["title"] for e in all_of(frames, "episode")] == ["Episode 1", "Episode 3"]


def test_a_zero_match_run_explains_itself_and_is_not_an_error(client, mixed):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&categories=Cooking"))
    assert all_of(frames, "episode") == []
    assert all_of(frames, "episode_error") == []
    assert "matched none of the 5 item(s)" in first(frames, "warning")["message"]
    assert first(frames, "complete")["total"] == 0


def test_a_filter_never_blocks_a_named_episode(client, mixed):
    """/api/rss/fetch narrows by guid AFTER the listing, so honouring a filter here would
    answer "raise the episode limit" about an episode that is right there."""
    frames = sse_frames(client.get(
        f"/api/rss/fetch?{Q}&guid=https://show.test/ep2&categories=Cooking"))
    eps = all_of(frames, "episode")
    assert len(eps) == 1 and eps[0]["title"] == "Episode 2"


def test_library_add_rss_imports_only_the_matching_episodes(client, mixed):
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    frames = sse_frames(
        client.get(f"/api/libraries/{lib['id']}/add-rss?{Q}&categories=Sport"))
    items = first(frames, "complete")["library"]["items"]
    assert [i["label"] for i in items] == ["Episode 2", "Episode 4"]


def test_a_keyword_filter_works_on_the_title(client, mixed):
    frames = sse_frames(client.get(f"/api/rss/fetch-feed?{Q}&keywords=Episode 4"))
    assert [e["title"] for e in all_of(frames, "episode")] == ["Episode 4"]


# ------------------------------ the cache routes ------------------------------

def test_cache_stats_round_trip(client, feed3):
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    body = client.get("/api/rss/cache").get_json()
    assert body["episodes"] == 3 and body["feeds"] == 1
    assert body["transcribed"] == 0
    assert body["bytes"] > 0


def test_clearing_only_feeds_keeps_the_episodes(client, feed3):
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    client.delete("/api/rss/cache?what=feeds")
    body = client.get("/api/rss/cache").get_json()
    assert body["feeds"] == 0 and body["episodes"] == 3


def test_clearing_episodes_keeps_the_listing(client, feed3):
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    client.delete("/api/rss/cache?what=episodes")
    body = client.get("/api/rss/cache").get_json()
    assert body["episodes"] == 0 and body["feeds"] == 1


def test_an_unknown_what_falls_back_to_all(client, feed3):
    sse_frames(client.get(f"/api/rss/fetch-feed?{Q}"))
    client.delete("/api/rss/cache?what=nonsense")
    body = client.get("/api/rss/cache").get_json()
    assert body["episodes"] == 0 and body["feeds"] == 0
