#!/usr/bin/env python3
"""Tests for the persistent RSS/podcast cache (app/rss_cache.py).

Like the YouTube cache, every rule here exists because episode entries NEVER expire: a
transcript URL that 403'd today must not harden into "this episode has none", and a
cancelled forty-minute transcription must not become the permanent answer.

Unlike the YouTube cache, the interesting axis is PROVENANCE rather than halves. A
published transcript is free and good; a whisper one costs GPU-minutes and is worse. The
coverage matrix in the middle of this file is the reason the module exists — in
particular the row where a cached "no transcript published" must NOT cover a request that
has the Whisper checkbox ticked, which is the difference between that checkbox working
and doing nothing.

``conftest.isolate_rss_cache`` is autouse, so every test starts with an empty cache dir.
"""

import pytest

from app import core, crypto, rss_cache
from conftest import isolate_paths

EP = "a" * 40
EP2 = "b" * 40
FEED = "f" * 40


def _result(episode_id=EP, transcript="Adam: In the morning.", source="published",
            **over):
    out = {
        "episode_id": episode_id,
        "feed_id": FEED,
        "episode_key": "http://1892.noagendanotes.com",
        "id_source": "guid",
        "title": '1892 - "Kill Switch"',
        "feed_title": "No Agenda Show",
        "link": "http://1892.noagendanotes.com",
        "published": "2026-08-06",
        "duration": "03:12:44",
        "enclosure_url": "https://mp3s.example.com/NA-1892.mp3",
        "enclosure_type": "audio/mpeg",
        "persons": [{"name": "Adam Curry", "role": "host", "group": "cast", "href": ""}],
        "transcript": transcript,
        "transcript_source": source if transcript else "",
        "transcript_format": "srt" if source == "published" else "",
        "transcript_url": ("https://mp3s.example.com/NA-1892.srt"
                           if source == "published" else ""),
        "transcript_speakers": True,
    }
    out.update(over)
    return out


# ------------------------------ round trip ------------------------------

def test_put_then_get_round_trips():
    rss_cache.put(_result())
    entry = rss_cache.get(EP)
    assert entry["transcript"] == "Adam: In the morning."
    assert entry["transcript_source"] == "published"
    assert entry["title"] == '1892 - "Kill Switch"'
    assert entry["persons"][0]["name"] == "Adam Curry"


def test_a_miss_is_none_not_an_error():
    assert rss_cache.get(EP) is None


def test_the_rendered_text_is_never_stored():
    result = _result()
    result["text"] = "the whole rendered document"
    rss_cache.put(result)
    assert "text" not in rss_cache.get(EP)


def test_a_corrupt_file_reads_as_a_miss():
    rss_cache._episodes_dir().mkdir(parents=True, exist_ok=True)
    core.write_bytes(rss_cache._path_for(EP), b"not json at all")
    assert rss_cache.get(EP) is None


def test_a_foreign_version_reads_as_a_miss():
    rss_cache._episodes_dir().mkdir(parents=True, exist_ok=True)
    core.save_json(rss_cache._path_for(EP), {"v": 99, "transcript": "old"})
    assert rss_cache.get(EP) is None


@pytest.mark.parametrize("bad", ["", "../../etc/passwd", "A" * 40, "a" * 39, "zz" * 20])
def test_path_for_rejects_anything_that_is_not_a_hash(bad):
    with pytest.raises(rss_cache.CacheError):
        rss_cache._path_for(bad)


def test_a_bad_id_makes_put_a_no_op_rather_than_an_exception():
    assert rss_cache.put(_result(episode_id="nope")) is None


def test_entries_are_encrypted_at_rest(tmp_path, monkeypatch):
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    monkeypatch.setattr(core, "RSS_CACHE_DIR", tmp_path / "rsscache")
    rss_cache.put(_result())
    raw = rss_cache._path_for(EP).read_bytes()
    assert b"In the morning" not in raw
    assert crypto.is_unlocked()
    assert rss_cache.get(EP)["transcript"] == "Adam: In the morning."


# --------------------- have(): the coverage matrix ---------------------
# One test per row. This is the file's reason to exist.

def _stored(source, transcript="text", missing=False, advertised=""):
    rss_cache.put(_result(transcript=transcript, source=source,
                          advertised_transcript_url=advertised,
                          published_transcript_missing=missing))
    return rss_cache.get(EP)


def test_published_covers_a_request_that_does_not_want_whisper():
    assert rss_cache.have(_stored("published"), want_whisper=False) is True


def test_published_covers_a_request_that_does_want_whisper():
    """Re-transcribing an episode that already has a publisher transcript would spend
    forty GPU-minutes to produce a worse answer."""
    assert rss_cache.have(_stored("published"), want_whisper=True) is True


def test_published_stays_covered_even_when_the_advertised_url_changed():
    entry = _stored("published", advertised="https://x/old.srt")
    assert rss_cache.have(entry, want_whisper=False,
                          advertised_url="https://x/new.srt") is True


def test_whisper_covers_a_request_that_does_not_want_whisper():
    assert rss_cache.have(_stored("whisper"), want_whisper=False) is True


def test_whisper_covers_a_request_that_wants_whisper():
    assert rss_cache.have(_stored("whisper"), want_whisper=True) is True


def test_whisper_is_NOT_covered_once_the_publisher_ships_a_transcript():
    """A new advertised URL over a whisper transcript means take theirs: better, free."""
    entry = _stored("whisper", advertised="")
    assert rss_cache.have(entry, want_whisper=False,
                          advertised_url="https://x/new.srt") is False


def test_a_known_missing_transcript_covers_a_free_run():
    entry = _stored("", transcript="", missing=True)
    assert rss_cache.have(entry, want_whisper=False) is True


def test_a_known_missing_transcript_does_NOT_cover_a_whisper_run():
    """The row that makes the checkbox work. Without it, ticking "Transcribe missing
    episodes" on a feed already imported once would silently do nothing."""
    entry = _stored("", transcript="", missing=True)
    assert rss_cache.have(entry, want_whisper=True) is False


def test_an_entry_that_was_never_looked_at_is_never_covered():
    rss_cache.put(_result(transcript="", source="", summary="Some show notes."))
    entry = rss_cache.get(EP)
    assert rss_cache.have(entry, want_whisper=False) is False
    assert rss_cache.have(entry, want_whisper=True) is False


def test_no_entry_at_all_is_not_covered():
    assert rss_cache.have(None, want_whisper=False) is False
    assert rss_cache.have({}, want_whisper=True) is False


# ------------------------------ put() rules ------------------------------

def test_an_empty_transcript_is_never_stored():
    rss_cache.put(_result(transcript="", summary="notes"))
    assert not (rss_cache.get(EP) or {}).get("transcript")


def test_whisper_never_overwrites_published():
    rss_cache.put(_result(transcript="the good one", source="published"))
    rss_cache.put(_result(transcript="a much longer machine transcription of the show",
                          source="whisper"))
    entry = rss_cache.get(EP)
    assert entry["transcript"] == "the good one"
    assert entry["transcript_source"] == "published"


def test_published_does_overwrite_whisper_and_flips_the_source():
    rss_cache.put(_result(transcript="a long machine transcription", source="whisper"))
    rss_cache.put(_result(transcript="short", source="published"))
    entry = rss_cache.get(EP)
    assert entry["transcript"] == "short"
    assert entry["transcript_source"] == "published"


def test_a_shorter_whisper_run_does_not_shrink_a_longer_stored_one():
    rss_cache.put(_result(transcript="a" * 500, source="whisper"))
    rss_cache.put(_result(transcript="a" * 20, source="whisper"))
    assert len(rss_cache.get(EP)["transcript"]) == 500


def test_a_longer_whisper_run_does_upgrade():
    rss_cache.put(_result(transcript="a" * 20, source="whisper"))
    rss_cache.put(_result(transcript="a" * 500, source="whisper"))
    assert len(rss_cache.get(EP)["transcript"]) == 500


def test_refresh_rewrites_at_equal_rank():
    rss_cache.put(_result(transcript="a" * 500, source="whisper"))
    rss_cache.put(_result(transcript="b" * 20, source="whisper"), refresh=True)
    assert rss_cache.get(EP)["transcript"] == "b" * 20


def test_a_republished_transcript_at_a_new_url_is_taken():
    rss_cache.put(_result(transcript="v1", source="published",
                          transcript_url="https://x/a.srt"))
    rss_cache.put(_result(transcript="v2", source="published",
                          transcript_url="https://x/b.srt"))
    assert rss_cache.get(EP)["transcript"] == "v2"


def test_the_same_published_url_is_not_rewritten():
    rss_cache.put(_result(transcript="v1", source="published",
                          transcript_url="https://x/a.srt"))
    before = rss_cache.get(EP)["updated"]
    rss_cache.put(_result(transcript="v1-refetched", source="published",
                          transcript_url="https://x/a.srt"))
    entry = rss_cache.get(EP)
    assert entry["transcript"] == "v1"
    assert entry["updated"] == before      # nothing was re-encrypted


def test_a_cancelled_transcription_writes_nothing():
    """A stopped run holds the first N minutes and, once on disk, is indistinguishable
    from a complete transcript."""
    rss_cache.put(_result(transcript="the first ten minutes only", source="whisper",
                          summary="notes"), stopped=True)
    assert not (rss_cache.get(EP) or {}).get("transcript")


def test_a_cancelled_run_does_not_clobber_a_stored_transcript():
    rss_cache.put(_result(transcript="a" * 200, source="whisper"))
    rss_cache.put(_result(transcript="b" * 5000, source="whisper"), stopped=True)
    assert rss_cache.get(EP)["transcript"] == "a" * 200


def test_a_stopped_flag_does_not_block_a_published_transcript():
    """`stopped` is about a truncated Whisper generator. A published transcript arrived
    whole or not at all, so a run cancelled afterwards must not discard it."""
    rss_cache.put(_result(transcript="published text", source="published"), stopped=True)
    assert rss_cache.get(EP)["transcript"] == "published text"


def test_missing_is_set_only_from_a_parse_level_negative():
    rss_cache.put(_result(transcript="", source="", published_transcript_missing=True))
    assert rss_cache.get(EP)["published_transcript_missing"] is True


def test_a_transport_failure_does_not_record_the_episode_as_having_none():
    """A 404 on the transcript URL is a transport failure, not proof the publisher
    ships nothing — recording it as the latter would be permanent."""
    rss_cache.put(_result(transcript="", source="", summary="notes",
                          published_transcript_missing=True),
                  transcript_error="404 on https://x/a.srt")
    assert "published_transcript_missing" not in rss_cache.get(EP)


def test_missing_is_cleared_once_a_transcript_lands():
    rss_cache.put(_result(transcript="", source="", published_transcript_missing=True))
    assert rss_cache.get(EP)["published_transcript_missing"] is True
    rss_cache.put(_result(transcript="machine text", source="whisper"))
    entry = rss_cache.get(EP)
    assert "published_transcript_missing" not in entry
    assert entry["transcript"] == "machine text"


def test_metadata_merges_and_a_blank_never_clears():
    rss_cache.put(_result())
    rss_cache.put(_result(title="", duration=""))
    entry = rss_cache.get(EP)
    assert entry["title"] == '1892 - "Kill Switch"'
    assert entry["duration"] == "03:12:44"


def test_metadata_alone_is_not_worth_a_file():
    assert rss_cache.put(_result(transcript="", source="")) is None
    assert rss_cache.get(EP) is None


def test_a_summary_alone_IS_worth_a_file():
    """Unlike a YouTube entry: for a plain non-podcast item the notes ARE the document,
    and getting them may have cost a full page crawl."""
    assert rss_cache.put(_result(transcript="", source="",
                                 summary="The whole blog post.",
                                 summary_source="content")) is not None
    assert rss_cache.get(EP)["summary"] == "The whole blog post."


def test_whisper_provenance_is_recorded():
    rss_cache.put(_result(transcript="machine", source="whisper",
                          whisper_model="large-v3", whisper_device="cuda"))
    entry = rss_cache.get(EP)
    assert entry["whisper_model"] == "large-v3"
    assert entry["whisper_device"] == "cuda"


def test_as_result_reshapes_without_the_text():
    rss_cache.put(_result())
    got = rss_cache.as_result(rss_cache.get(EP))
    assert got["via"] == "cache"
    assert got["transcript"] == "Adam: In the morning."
    assert "text" not in got


def test_delete_removes_the_entry():
    rss_cache.put(_result())
    assert rss_cache.delete(EP) is True
    assert rss_cache.get(EP) is None
    assert rss_cache.delete(EP) is False


# ------------------------------ feeds ------------------------------

def _feed(feed_id=FEED, etag='W/"abc"', items=None):
    return {"feed_id": feed_id, "url": "https://feeds.example.com/show.xml",
            "title": "No Agenda Show", "etag": etag, "modified": "Thu, 06 Aug 2026 22:12:48 GMT",
            "items": items if items is not None else [{"guid": "g1", "title": "Ep 1"}],
            "item_count": 1}


def test_feed_round_trips_with_its_validators():
    rss_cache.put_feed(_feed())
    got = rss_cache.get_feed(FEED)
    assert got["etag"] == 'W/"abc"'
    assert got["modified"].startswith("Thu, 06 Aug")
    assert got["items"][0]["guid"] == "g1"
    assert got["fetched"]


def test_put_feed_replaces_rather_than_merges():
    """A listing's whole point is that the new one supersedes the old — a merge would
    resurrect episodes the publisher has pulled."""
    rss_cache.put_feed(_feed(items=[{"guid": "g1"}, {"guid": "g2"}]))
    rss_cache.put_feed(_feed(items=[{"guid": "g3"}]))
    assert [i["guid"] for i in rss_cache.get_feed(FEED)["items"]] == ["g3"]


def test_a_feed_miss_is_none():
    assert rss_cache.get_feed(FEED) is None


def test_delete_feed():
    rss_cache.put_feed(_feed())
    assert rss_cache.delete_feed(FEED) is True
    assert rss_cache.get_feed(FEED) is None


def test_feeds_and_episodes_live_in_separate_namespaces():
    """The same 40-hex string can legitimately be both — stats() must not double-count
    and clear(what=...) must not cross over."""
    rss_cache.put(_result(episode_id=FEED))
    rss_cache.put_feed(_feed(feed_id=FEED))
    assert rss_cache.get(FEED) is not None
    assert rss_cache.get_feed(FEED) is not None
    st = rss_cache.stats()
    assert st["episodes"] == 1 and st["feeds"] == 1


# ------------------------------ maintenance ------------------------------

def test_stats_counts_both_namespaces_and_the_transcribed_ones():
    rss_cache.put(_result(episode_id=EP, source="published"))
    rss_cache.put(_result(episode_id=EP2, transcript="machine", source="whisper"))
    rss_cache.put_feed(_feed())
    st = rss_cache.stats()
    assert st["episodes"] == 2
    assert st["feeds"] == 1
    assert st["transcribed"] == 1          # what the Clear warning quotes
    assert st["bytes"] > 0
    assert st["dir"] == str(core.RSS_CACHE_DIR)


def test_stats_on_a_directory_that_was_never_created():
    assert rss_cache.stats()["episodes"] == 0
    assert rss_cache.clear() == {"removed": 0, "bytes": 0}


def test_clear_feeds_keeps_every_episode():
    """The cheap Clear: force a fresh listing without throwing away GPU-hours."""
    rss_cache.put(_result())
    rss_cache.put_feed(_feed())
    out = rss_cache.clear(what="feeds")
    assert out["removed"] == 1
    assert rss_cache.get_feed(FEED) is None
    assert rss_cache.get(EP) is not None


def test_clear_episodes_keeps_the_listings():
    rss_cache.put(_result())
    rss_cache.put_feed(_feed())
    rss_cache.clear(what="episodes")
    assert rss_cache.get(EP) is None
    assert rss_cache.get_feed(FEED) is not None


def test_clear_all_removes_both():
    rss_cache.put(_result())
    rss_cache.put_feed(_feed())
    out = rss_cache.clear()
    assert out["removed"] == 2
    assert rss_cache.stats()["episodes"] == 0
    assert rss_cache.stats()["feeds"] == 0


def test_stray_files_are_ignored_by_stats():
    rss_cache.put(_result())
    (rss_cache._episodes_dir() / "notes.txt").write_text("scratch")
    assert rss_cache.stats()["episodes"] == 1


# ------------------------------ profile scoping ------------------------------

def test_entries_are_scoped_to_the_active_profile(tmp_path, monkeypatch):
    first, second = tmp_path / "p1", tmp_path / "p2"
    monkeypatch.setattr(core, "RSS_CACHE_DIR", first)
    rss_cache.put(_result())
    assert rss_cache.get(EP) is not None

    monkeypatch.setattr(core, "RSS_CACHE_DIR", second)
    assert rss_cache.get(EP) is None

    monkeypatch.setattr(core, "RSS_CACHE_DIR", first)
    assert rss_cache.get(EP) is not None


def test_set_active_data_profile_points_the_cache_dir(tmp_path):
    core.set_active_data_profile(tmp_path / "profile-a")
    assert core.RSS_CACHE_DIR == tmp_path / "profile-a" / "rss_cache"
    assert core.RSS_CACHE_DIR.is_dir()
