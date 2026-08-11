#!/usr/bin/env python3
"""Tests for the persistent YouTube cache (app/youtube_cache.py).

The rules under test all exist because the cache NEVER expires: anything written once
is served indefinitely, so a truncated comment list or a transcript that failed to load
today must not harden into a permanent answer.

``conftest.isolate_youtube_cache`` is autouse, so every test here starts with an empty
cache dir under tmp_path.
"""

import json

import pytest

from app import core, crypto, youtube_cache
from conftest import isolate_paths

VID = "dQw4w9WgXcQ"


def _result(video_id=VID, title="How Engines Work", transcript="The engine turns.",
            comments=None, via="requests"):
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title, "channel": "Garage Lab",
        "published": "2024-03-12", "views": "1,240,113",
        "transcript": transcript,
        "comments": comments if comments is not None else [
            {"author": "Ada", "text": "Great explainer", "likes": "412", "published": "2mo"}],
        "via": via,
    }


def _comments(n):
    return [{"author": f"U{i}", "text": f"c{i}", "likes": "", "published": ""}
            for i in range(n)]


# ------------------------------ round trip ------------------------------

def test_put_then_get_round_trips_both_halves():
    youtube_cache.put(_result(), include_comments=True, max_comments=100)
    entry = youtube_cache.get(VID)
    assert entry["video_id"] == VID
    assert entry["title"] == "How Engines Work"
    assert entry["transcript"] == "The engine turns."
    assert entry["comment_count"] == 1
    assert entry["comment_target"] == 100


def test_get_is_none_for_an_unknown_video():
    assert youtube_cache.get("aaaaaaaaaaa") is None


def test_text_is_never_stored():
    """`text` depends on this caller's include_comments/max_comments, so storing it
    would serve a rendering that is wrong for the next caller."""
    result = _result()
    result["text"] = "rendered document"
    youtube_cache.put(result, include_comments=True, max_comments=100)
    assert "text" not in youtube_cache.get(VID)


def test_a_corrupt_entry_reads_as_a_miss_rather_than_raising():
    core.write_bytes(youtube_cache._path_for(VID), b"not json at all")
    assert youtube_cache.get(VID) is None


def test_a_foreign_version_reads_as_a_miss():
    core.save_json(youtube_cache._path_for(VID), {"v": 99, "transcript": "old"})
    assert youtube_cache.get(VID) is None


# ------------------------------ id validation ------------------------------

@pytest.mark.parametrize("bad", [
    "../../../etc/passwd",
    "..",
    "",
    "short",
    "waaaaaaytoolongforanid",
    "has/slash11",
])
def test_bad_ids_are_rejected(bad):
    """The id arrives from a URL, so a bare path join would be a traversal hole."""
    with pytest.raises(youtube_cache.CacheError):
        youtube_cache._path_for(bad)


def test_a_legal_id_resolves_inside_the_cache_dir():
    path = youtube_cache._path_for(VID)
    assert path.parent == core.YOUTUBE_CACHE_DIR
    assert path.name == f"{VID}.json"


def test_put_ignores_a_result_with_no_usable_id():
    assert youtube_cache.put(_result(video_id="nope"), True, 100) is None


# ------------------------------ encryption at rest ------------------------------

def test_entries_are_encrypted_at_rest(tmp_path, monkeypatch):
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", tmp_path / "ytcache")
    youtube_cache.put(_result(), include_comments=True, max_comments=100)
    raw = youtube_cache._path_for(VID).read_bytes()
    assert crypto.is_encrypted(raw)
    with pytest.raises(Exception):
        json.loads(raw.decode("utf-8", "ignore"))
    # …and still reads back through the normal path.
    assert youtube_cache.get(VID)["transcript"] == "The engine turns."


# ------------------------------ have(): coverage rules ------------------------------

def test_a_smaller_ask_than_the_cached_count_is_covered():
    youtube_cache.put(_result(comments=_comments(100)), True, 100)
    entry = youtube_cache.get(VID)
    assert youtube_cache.have(entry, True, 50) == (True, True)


def test_a_larger_ask_than_was_cached_is_not_covered():
    youtube_cache.put(_result(comments=_comments(100)), True, 100)
    entry = youtube_cache.get(VID)
    assert youtube_cache.have(entry, True, 500) == (True, False)


def test_a_thread_shorter_than_the_ask_stays_covered_for_that_ask():
    """87 comments fetched under an ask of 500 means the thread is exhausted. Without
    comment_target this would refetch on every future ask of 500 — forever, since
    nothing here expires."""
    youtube_cache.put(_result(comments=_comments(87)), True, 500)
    entry = youtube_cache.get(VID)
    assert entry["comment_count"] == 87
    assert entry["comment_target"] == 500
    assert youtube_cache.have(entry, True, 300) == (True, True)
    assert youtube_cache.have(entry, True, 500) == (True, True)
    assert youtube_cache.have(entry, True, 800) == (True, False)


def test_comments_off_is_always_covered():
    youtube_cache.put(_result(comments=[]), include_comments=False, max_comments=100)
    entry = youtube_cache.get(VID)
    assert youtube_cache.have(entry, False, 2000)[1] is True


def test_never_asked_for_comments_is_not_covered():
    youtube_cache.put(_result(), include_comments=False, max_comments=100)
    entry = youtube_cache.get(VID)
    assert "comments" not in entry
    assert youtube_cache.have(entry, True, 10) == (True, False)


def test_asked_and_genuinely_empty_is_covered():
    """`[]` under a target is a real answer — the video has no comments — and must not
    be confused with "never asked"."""
    youtube_cache.put(_result(comments=[]), include_comments=True, max_comments=100)
    entry = youtube_cache.get(VID)
    assert entry["comments"] == []
    assert youtube_cache.have(entry, True, 100) == (True, True)


def test_have_on_no_entry_is_all_false():
    assert youtube_cache.have(None, True, 100) == (False, False)


# ------------------------------ put(): what may be stored ------------------------------

def test_an_empty_transcript_is_never_stored():
    """A rung blocked today must not become a permanent "this video has no captions"."""
    youtube_cache.put(_result(transcript=""), True, 100)
    assert "transcript" not in (youtube_cache.get(VID) or {})


def test_a_stopped_run_stores_the_transcript_but_not_the_comments():
    youtube_cache.put(_result(comments=_comments(30)), True, 100, stopped=True)
    entry = youtube_cache.get(VID)
    assert entry["transcript"] == "The engine turns."
    assert "comments" not in entry


def test_a_failed_comment_half_is_not_stored():
    youtube_cache.put(_result(comments=[]), True, 100, comment_error="requests: boom")
    entry = youtube_cache.get(VID)
    assert "comments" not in entry


def test_metadata_alone_writes_nothing():
    assert youtube_cache.put(_result(transcript="", comments=[]),
                             include_comments=False, max_comments=100) is None
    assert youtube_cache.get(VID) is None


# ------------------------------ put(): merging ------------------------------

def test_a_comments_off_fetch_leaves_stored_comments_intact():
    youtube_cache.put(_result(comments=_comments(40)), True, 100)
    youtube_cache.put(_result(comments=[]), include_comments=False, max_comments=100)
    entry = youtube_cache.get(VID)
    assert entry["comment_count"] == 40


def test_a_shorter_comment_fetch_does_not_shrink_a_longer_stored_one():
    youtube_cache.put(_result(comments=_comments(200)), True, 200)
    youtube_cache.put(_result(comments=_comments(20)), True, 20)
    assert youtube_cache.get(VID)["comment_count"] == 200


def test_an_upgrade_keeps_the_transcript_byte_identical():
    youtube_cache.put(_result(comments=_comments(20)), True, 20)
    before = youtube_cache.get(VID)["transcript"]
    youtube_cache.put(_result(transcript="", comments=_comments(200)), True, 200)
    after = youtube_cache.get(VID)
    assert after["transcript"] == before
    assert after["comment_count"] == 200


def test_a_placeholder_title_never_overwrites_a_real_one():
    youtube_cache.put(_result(), True, 100)
    youtube_cache.put(_result(title="Unknown Video", comments=_comments(200)), True, 200)
    assert youtube_cache.get(VID)["title"] == "How Engines Work"


def test_a_real_title_replaces_a_placeholder():
    youtube_cache.put(_result(title="Unknown Video"), True, 100)
    youtube_cache.put(_result(title="How Engines Work"), True, 100)
    assert youtube_cache.get(VID)["title"] == "How Engines Work"


# ------------------------------ as_result ------------------------------

def test_as_result_slices_to_max_comments_and_flags_the_cache():
    youtube_cache.put(_result(comments=_comments(100)), True, 100)
    got = youtube_cache.as_result(youtube_cache.get(VID), True, 10)
    assert len(got["comments"]) == 10
    assert got["via"] == "cache"
    assert got["transcript"] == "The engine turns."
    assert "text" not in got


def test_as_result_drops_comments_when_they_were_not_asked_for():
    youtube_cache.put(_result(comments=_comments(100)), True, 100)
    got = youtube_cache.as_result(youtube_cache.get(VID), False, 100)
    assert got["comments"] == []


# ------------------------------ maintenance ------------------------------

def test_delete_removes_one_entry():
    youtube_cache.put(_result(), True, 100)
    assert youtube_cache.delete(VID) is True
    assert youtube_cache.get(VID) is None
    assert youtube_cache.delete(VID) is False


def test_stats_counts_entries_and_bytes():
    assert youtube_cache.stats()["entries"] == 0
    youtube_cache.put(_result(), True, 100)
    youtube_cache.put(_result(video_id="abcdefghijk"), True, 100)
    stats = youtube_cache.stats()
    assert stats["entries"] == 2
    assert stats["bytes"] > 0


def test_stats_ignores_non_entry_files():
    youtube_cache.put(_result(), True, 100)
    (core.YOUTUBE_CACHE_DIR / "notes.txt").write_text("scratch")
    assert youtube_cache.stats()["entries"] == 1


def test_stats_and_clear_survive_a_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", tmp_path / "never-made")
    assert youtube_cache.stats() == {"entries": 0, "bytes": 0,
                                     "dir": str(tmp_path / "never-made")}
    assert youtube_cache.clear() == {"removed": 0, "bytes": 0}


def test_clear_empties_the_cache():
    youtube_cache.put(_result(), True, 100)
    youtube_cache.put(_result(video_id="abcdefghijk"), True, 100)
    out = youtube_cache.clear()
    assert out["removed"] == 2
    assert out["bytes"] > 0
    assert youtube_cache.stats()["entries"] == 0
    assert youtube_cache.get(VID) is None


def test_a_cache_is_per_data_profile(tmp_path, monkeypatch):
    """Switching profiles must not expose one profile's crawls to another — and an
    incognito session's cache dies with its scratch dir for the same reason."""
    first, second = tmp_path / "p1", tmp_path / "p2"
    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", first)
    youtube_cache.put(_result(), True, 100)
    assert youtube_cache.get(VID) is not None

    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", second)
    assert youtube_cache.get(VID) is None

    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", first)
    assert youtube_cache.get(VID) is not None


def test_set_active_data_profile_points_the_cache_at_the_profile(tmp_path, monkeypatch):
    core.set_active_data_profile(tmp_path / "profile-a")
    assert core.YOUTUBE_CACHE_DIR == tmp_path / "profile-a" / "youtube_cache"
    assert core.YOUTUBE_CACHE_DIR.is_dir()
