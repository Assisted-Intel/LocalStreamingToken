#!/usr/bin/env python3
"""Tests for RSS / podcast ingestion (app/rss.py).

Nothing here touches the network: ``rss_net`` (conftest) replaces ``rss._http_get`` and
fails loudly on an unregistered URL, so a forgotten stub surfaces as an error rather than
a live request.

Two properties get disproportionate attention because they are the ones that would fail
silently in production:

  * ``podcast:*`` elements are parsed from raw XML, not from feedparser, which collapses
    repeated namespaced elements to a single last-one-wins dict. The fixtures emit two
    <podcast:person> and sometimes three <podcast:transcript> per item precisely so a
    regression to feedparser's view is caught.
  * An item WITH an enclosure never escalates to a page crawl. Without that rule,
    importing a 226-episode feed fires 226 Playwright crawls.
"""

import json

import pytest

from app import core, rss, rss_cache
from conftest import SRT_BODY, feed_xml, podcast_item

URL = "https://feeds.test/show.xml"


def _load(rss_net, items=None, **kw):
    rss_net["routes"][URL] = feed_xml(items, **kw)
    return rss.fetch_feed(URL)


# ------------------------------ ids ------------------------------

def test_episode_key_prefers_guid_then_enclosure_then_link():
    assert rss.episode_key({"guid": "g", "enclosure_url": "e", "link": "l"}) == ("g", "guid")
    assert rss.episode_key({"enclosure_url": "e", "link": "l"}) == ("e", "enclosure")
    assert rss.episode_key({"link": "l"}) == ("l", "link")
    assert rss.episode_key({}) == ("", "")


def test_the_same_guid_in_two_feeds_does_not_collide():
    """<guid>1</guid> is real and common. A global keyspace would let one feed be
    served another feed's transcript."""
    a, b = rss.feed_id("https://a.test/f.xml"), rss.feed_id("https://b.test/f.xml")
    assert rss.episode_id(a, "1") != rss.episode_id(b, "1")


def test_feed_id_normalises_scheme_host_and_fragment_but_keeps_the_query():
    same = rss.feed_id("https://Feeds.Test:443/show.xml?u=9#top")
    assert same == rss.feed_id("https://feeds.test/show.xml?u=9")
    assert same != rss.feed_id("https://feeds.test/show.xml?u=8")


def test_an_item_with_no_identity_at_all_is_skipped_not_fatal(rss_net):
    feed = _load(rss_net, [podcast_item(1),
                           podcast_item(2, guid="", link="", enclosure=False)])
    assert feed["item_count"] == 1


# ------------------------------ feed parsing ------------------------------

def test_repeated_podcast_persons_all_survive(rss_net):
    """feedparser returns only the LAST <podcast:person>; the raw-XML pass must not."""
    feed = _load(rss_net, [podcast_item(1, persons=2)])
    assert [p["name"] for p in feed["items"][0]["persons"]] == ["Adam Curry", "John C Dvorak"]


def test_multiple_transcripts_all_survive(rss_net):
    feed = _load(rss_net, [podcast_item(1, transcripts=[("json", "captions"),
                                                        ("srt", None), ("vtt", None)])])
    assert len(feed["items"][0]["transcripts"]) == 3


def test_the_feed_channel_metadata_is_read(rss_net):
    feed = _load(rss_net, title="No Agenda Show")
    assert feed["title"] == "No Agenda Show"
    assert feed["language"] == "en"
    assert feed["link"] == "https://show.test/"


def test_enclosure_and_duration_are_normalised(rss_net):
    feed = _load(rss_net, [podcast_item(1, duration="9241")])
    item = feed["items"][0]
    assert item["enclosure_url"] == "https://cdn.test/ep1.mp3"
    assert item["enclosure_type"] == "audio/mpeg"
    # Bare seconds is a legal <itunes:duration> and the sample feed uses it; rendering
    # "9241" in the header would be gibberish.
    assert item["duration"] == "02:34:01"


def test_a_clock_duration_is_left_alone(rss_net):
    feed = _load(rss_net, [podcast_item(1, duration="01:02:03")])
    assert feed["items"][0]["duration"] == "01:02:03"


def test_a_non_feed_url_is_an_actionable_error(rss_net):
    rss_net["routes"][URL] = b"<html><body>not a feed</body></html>"
    with pytest.raises(rss.RSSError) as e:
        rss.fetch_feed(URL)
    assert "RSS or Atom feed" in str(e.value)


def test_a_dead_feed_with_no_cache_raises(rss_net):
    rss_net["routes"][URL] = RuntimeError("connection refused")
    with pytest.raises(rss.RSSError):
        rss.fetch_feed(URL)


def test_a_dead_feed_with_a_cache_serves_the_cache_with_a_warning(rss_net):
    _load(rss_net, [podcast_item(1)])
    rss_net["routes"][URL] = RuntimeError("connection refused")
    feed = rss.fetch_feed(URL)
    assert feed["from_cache"] is True
    assert feed["item_count"] == 1
    assert any("cached listing" in w for w in feed["warnings"])


def test_a_304_serves_the_cached_listing(rss_net):
    _load(rss_net, [podcast_item(1), podcast_item(2)])
    rss_net["routes"][URL] = 304
    feed = rss.fetch_feed(URL)
    assert feed["from_cache"] is True
    assert feed["item_count"] == 2


def test_the_conditional_validators_are_sent_on_the_second_read(rss_net):
    _load(rss_net, [podcast_item(1)])
    rss_net["headers"].clear()
    rss.fetch_feed(URL)
    assert rss_net["headers"][0].get("If-None-Match") == 'W/"x"'


def test_identical_content_short_circuits_even_without_a_304(rss_net):
    """Real origins serve 200 with a byte-identical body — this feed's CDN won't match
    its own weak ETag. Noticing that skips the parse and the re-encrypt."""
    feed = _load(rss_net, [podcast_item(1)])
    again = rss.fetch_feed(URL)          # same bytes, still a 200
    assert again["from_cache"] is True
    assert again["item_count"] == feed["item_count"]


def test_changed_content_is_reparsed(rss_net):
    _load(rss_net, [podcast_item(1)])
    rss_net["routes"][URL] = feed_xml([podcast_item(1), podcast_item(2)])
    feed = rss.fetch_feed(URL)
    assert feed["from_cache"] is False
    assert feed["item_count"] == 2


def test_refresh_ignores_the_cache(rss_net):
    _load(rss_net, [podcast_item(1)])
    feed = rss.fetch_feed(URL, refresh=True)
    assert feed["from_cache"] is False


def test_limit_takes_the_first_n_in_feed_order(rss_net):
    """Feed order, NOT date order — the items below are deliberately out of
    chronological sequence and must not be re-sorted."""
    items = [podcast_item(1, pubdate="Mon, 03 Aug 2026 00:00:00 +0000"),
             podcast_item(2, pubdate="Fri, 07 Aug 2026 00:00:00 +0000"),
             podcast_item(3, pubdate="Wed, 05 Aug 2026 00:00:00 +0000")]
    rss_net["routes"][URL] = feed_xml(items)
    feed = rss.fetch_feed(URL, limit=2)
    assert [i["title"] for i in feed["items"]] == ["Episode 1", "Episode 2"]
    assert feed["total_available"] == 3


def test_unstable_episode_ids_are_warned_about_not_silently_refetched(rss_net):
    """A feed that regenerates its guids would otherwise re-transcribe its whole back
    catalogue on every run, silently."""
    rss_net["routes"][URL] = feed_xml([podcast_item(i, guid=f"old-{i}") for i in range(6)])
    rss.fetch_feed(URL)
    rss_net["routes"][URL] = feed_xml([podcast_item(i, guid=f"new-{i}") for i in range(6)])
    feed = rss.fetch_feed(URL)
    assert any("episode ids changed" in w for w in feed["warnings"])


def test_a_stable_feed_produces_no_warning(rss_net):
    rss_net["routes"][URL] = feed_xml([podcast_item(i) for i in range(6)])
    rss.fetch_feed(URL)
    rss_net["routes"][URL] = feed_xml([podcast_item(i) for i in range(7)])
    assert rss.fetch_feed(URL)["warnings"] == []


def test_a_missing_feedparser_is_an_actionable_message(rss_net, monkeypatch):
    monkeypatch.setattr(rss, "_import_feedparser",
                        lambda: (_ for _ in ()).throw(
                            rss.RSSError("Reading a feed needs feedparser. Install it "
                                         "with:  pip install feedparser")))
    rss_net["routes"][URL] = feed_xml()
    with pytest.raises(rss.RSSError) as e:
        rss.fetch_feed(URL)
    assert "pip install feedparser" in str(e.value)


# ------------------------------ categories ------------------------------

def test_every_category_flavour_lands_in_one_flat_list(rss_net):
    """The whole filter design rests on feedparser folding four different elements into
    ``entry.tags``. If it ever stops, this is the test that says so."""
    feed = _load(rss_net, [podcast_item(1, categories=["True Crime", "Interviews"],
                                       domain_category="Taxonomy Code",
                                       itunes_category="Crime",
                                       keywords="murder, cold case")])
    assert feed["items"][0]["categories"] == [
        "True Crime", "Interviews", "Taxonomy Code", "Crime", "murder", "cold case"]


def test_show_level_categories_are_read_including_the_nested_subcategory(rss_net):
    rss_net["routes"][URL] = feed_xml([podcast_item(1)],
                                      categories=[("News", "Politics"), "Society & Culture"])
    feed = rss.fetch_feed(URL)
    assert feed["categories"] == ["News", "Politics", "Society & Culture"]


def test_categories_are_deduped_case_insensitively_keeping_the_first_spelling(rss_net):
    feed = _load(rss_net, [podcast_item(1, categories=["True Crime", "TRUE CRIME"],
                                        keywords="true crime")])
    assert feed["items"][0]["categories"] == ["True Crime"]


def test_the_category_list_is_capped(rss_net):
    """An SEO feed shipping 200 keywords per item would otherwise bloat every cached
    listing, which is re-encrypted and rewritten whenever the feed changes."""
    feed = _load(rss_net, [podcast_item(1, keywords=",".join(f"kw{i}" for i in range(200)))])
    assert len(feed["items"][0]["categories"]) == rss._MAX_CATEGORIES


def test_an_item_with_no_categories_gets_a_list_not_a_missing_key(rss_net):
    """This dict is the cached listing shape; a filter reading a missing key would
    silently match nothing."""
    assert _load(rss_net, [podcast_item(1)])["items"][0]["categories"] == []


# ------------------------------ parse_filters ------------------------------

def test_parse_filters_accepts_a_comma_string_and_a_list():
    assert rss.parse_filters({"categories": "News, True Crime"})["categories"] == \
        ["News", "True Crime"]
    assert rss.parse_filters({"keywords": ["a", "b"]})["keywords"] == ["a", "b"]


@pytest.mark.parametrize("src", [
    None, {}, {"categories": ""}, {"categories": "  ", "keywords": ""},
    {"categories": "", "keywords": "  ", "exclude": "", "match": "all"},
])
def test_an_empty_filter_is_off(src):
    """sourceStream sends `categories=` for an untouched box — an empty string must not
    narrow anything, or every unfiltered import would break."""
    assert rss.parse_filters(src) == {}


def test_exclude_alone_is_a_real_filter():
    assert rss.parse_filters({"exclude": "ads"})["exclude"] == ["ads"]


def test_an_unknown_match_mode_falls_back_to_any():
    """These round-trip through a hand-editable batch project; a typo must not fail a
    run."""
    assert rss.parse_filters({"categories": "x", "match": "wat"})["match"] == "any"
    assert rss.parse_filters({"categories": "x", "match": "ALL"})["match"] == "all"


# ------------------------------ filter_items ------------------------------

def _items(*specs):
    return [{"title": t, "categories": list(c), "author": a, "body_html": b}
            for t, c, a, b in specs]


def _filter(items, **kw):
    kept, stats = rss.filter_items(items, rss.parse_filters(kw))
    return [i["title"] for i in kept], stats


ITEMS = _items(
    ("Arts and crafts", ["Arts"], "Ann", "<p class='audio-player'>Hello there</p>"),
    ("Murder most foul", ["True Crime", "Society & Culture"], "Bob", ""),
    ("Sponsored ad read", ["True Crime"], "Cat", ""),
    ("Untagged post", [], "Dee", ""),
)


def test_a_category_matches_the_whole_term_not_a_substring():
    """"art" must not select "Arts" — nor, in the wild, "Martial Arts" or "Heart Health".
    Substring matching belongs in `keywords`, and does happen there."""
    assert _filter(ITEMS, categories="art")[0] == []
    assert _filter(ITEMS, categories="Arts")[0] == ["Arts and crafts"]


def test_category_matching_absorbs_punctuation_and_casing():
    assert _filter(ITEMS, categories="society and culture")[0] == ["Murder most foul"]


def test_a_keyword_matches_the_title_the_categories_or_the_author():
    assert _filter(ITEMS, keywords="crafts")[0] == ["Arts and crafts"]
    assert _filter(ITEMS, keywords="true crime")[0] == ["Murder most foul",
                                                        "Sponsored ad read"]
    assert _filter(ITEMS, keywords="dee")[0] == ["Untagged post"]


def test_a_keyword_searches_show_notes_as_text_not_as_markup():
    """Without the tag stripper, "audio" matches class="audio-player" and "img" matches
    every episode with an image — a filter that looks wrong rather than broken."""
    assert _filter(ITEMS, keywords="audio")[0] == []
    assert _filter(ITEMS, keywords="hello there")[0] == ["Arts and crafts"]


def test_exclude_vetoes_an_otherwise_matching_item():
    assert _filter(ITEMS, categories="True Crime", exclude="sponsored")[0] == \
        ["Murder most foul"]


def test_exclude_works_with_no_include_terms_at_all():
    assert _filter(ITEMS, exclude="murder")[0] == ["Arts and crafts", "Sponsored ad read",
                                                   "Untagged post"]


def test_match_all_wants_every_term_and_any_wants_one():
    both = "True Crime, Society & Culture"
    assert _filter(ITEMS, categories=both, match="all")[0] == ["Murder most foul"]
    assert _filter(ITEMS, categories=both, match="any")[0] == ["Murder most foul",
                                                               "Sponsored ad read"]


def test_an_empty_filter_never_touches_the_show_notes(monkeypatch):
    """The unfiltered path must not run the tag stripper over 226 items' notes."""
    monkeypatch.setattr(core, "_html_to_text",
                        lambda *a, **k: pytest.fail("stripped notes with no filter"))
    kept, _ = rss.filter_items(ITEMS, {})
    assert len(kept) == 4


def test_the_filter_never_reads_the_transcript(rss_net):
    """A transcript costs a download, or minutes of GPU — exactly what the filter exists
    to avoid spending. So a word that appears ONLY in the transcript cannot match, and no
    transcript may be fetched to find that out."""
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    rss_net["routes"][URL] = feed_xml([podcast_item(1, categories=["News"])])
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"keywords": "Gitmo Nation"}))
    assert feed["items"] == []
    assert not [u for u in rss_net["get"] if u.endswith(".srt")]


# ------------------------------ filters through fetch_feed ------------------------------

def _mixed_feed():
    return feed_xml([podcast_item(1, categories=["News"]),
                     podcast_item(2, categories=["Sport"]),
                     podcast_item(3, categories=["News"]),
                     podcast_item(4, categories=["Sport"]),
                     podcast_item(5, categories=["News"])])


def test_the_filter_runs_before_the_limit(rss_net):
    """The load-bearing ordering. Slice-then-filter would return Episode 1 alone here,
    and no limit could ever reach Episodes 3 and 5."""
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL, limit=2, filters={"categories": ["News"], "keywords": [],
                                                 "exclude": [], "match": "any"})
    assert [i["title"] for i in feed["items"]] == ["Episode 1", "Episode 3"]


def test_the_three_counts_mean_three_different_things(rss_net):
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL, limit=2, filters=rss.parse_filters({"categories": "News"}))
    assert feed["total_available"] == 5      # the whole feed
    assert feed["matched"] == 3              # what the filter accepted
    assert feed["item_count"] == 2           # what the limit then took


def test_an_unfiltered_read_reports_no_narrowing(rss_net):
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL)
    assert feed["matched"] == feed["total_available"] == 5
    assert feed["filters"] == {}


def test_a_filter_matching_nothing_warns(rss_net):
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "Cooking"}))
    assert feed["items"] == []
    assert any("matched none of the 5 item(s)" in w for w in feed["warnings"])


def test_the_zero_match_warning_names_the_show_categories(rss_net):
    """The commonest confusion by far: a podcast categorises the SHOW, the user types the
    category they can plainly see, and gets nothing. Say why, and say what does work."""
    rss_net["routes"][URL] = feed_xml([podcast_item(1), podcast_item(2)],
                                      categories=[("News", "Politics")])
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "News"}))
    warning = " ".join(feed["warnings"])
    assert "show level (News, Politics)" in warning
    assert "keyword filter" in warning


def test_the_zero_match_warning_offers_the_categories_that_do_exist(rss_net):
    """When the items ARE tagged and the term simply isn't one of them, naming the real
    ones turns a dead end into a one-word correction."""
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "new"}))
    warning = " ".join(feed["warnings"])
    assert "categories these items do publish are: News, Sport" in warning
    # And it says why "new" missed "News", which is the mistake it was most likely made by.
    assert "match in full" in warning


def test_the_offered_categories_lead_with_the_ones_that_partition_the_feed(rss_net):
    """A facet shared by many episodes is worth typing; per-episode <itunes:keywords>
    noise is not, even though both are equally matchable."""
    rss_net["routes"][URL] = feed_xml([
        podcast_item(i, categories=["News"], keywords=f"filler{i}") for i in range(1, 5)])
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "absent"}))
    offered = " ".join(feed["warnings"]).split("publish are: ")[1]
    assert offered.startswith("News, ")


def test_a_keyword_only_miss_does_not_lecture_about_categories(rss_net):
    rss_net["routes"][URL] = _mixed_feed()
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"keywords": "nothing here"}))
    assert "categories these items do publish" not in " ".join(feed["warnings"])


def test_a_feed_with_no_categories_anywhere_says_so(rss_net):
    rss_net["routes"][URL] = feed_xml([podcast_item(1)])
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "News"}))
    assert any("publish any categories at all" in w for w in feed["warnings"])


def test_filtering_does_not_narrow_what_gets_cached(rss_net):
    """The filter is a view, not a fetch parameter. Filtering ``entry["items"]`` in place
    would poison every later read of this feed with one run's filter."""
    rss_net["routes"][URL] = _mixed_feed()
    rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "News"}))
    again = rss.fetch_feed(URL)
    assert again["from_cache"] is True
    assert again["item_count"] == 5


def test_a_listing_cached_before_categories_existed_is_reparsed(rss_net, monkeypatch):
    """Without the FEED_ENTRY_VERSION bump a v1 listing has no `categories` key, so every
    filter would match nothing and "tick Refresh" would be the only cure."""
    current = rss_cache.FEED_ENTRY_VERSION
    monkeypatch.setattr(rss_cache, "FEED_ENTRY_VERSION", 1)
    rss_net["routes"][URL] = _mixed_feed()
    rss.fetch_feed(URL)
    assert rss_cache.get_feed(rss.feed_id(URL)) is not None
    # Not monkeypatch.undo(): the rss_net fixture shares this monkeypatch instance, so an
    # undo would unstub _http_get and send the next line at the real network.
    monkeypatch.setattr(rss_cache, "FEED_ENTRY_VERSION", current)
    feed = rss.fetch_feed(URL, filters=rss.parse_filters({"categories": "News"}))
    assert [i["title"] for i in feed["items"]] == ["Episode 1", "Episode 3", "Episode 5"]


# ------------------------------ transcript choice ------------------------------

def _links(*specs):
    types = {"srt": "application/srt", "vtt": "text/vtt", "json": "application/json",
             "text": "text/plain", "html": "text/html", "pdf": "application/pdf"}
    return [{"url": f"https://cdn.test/x.{k}", "type": types[k], "rel": rel,
             "language": lang} for k, rel, lang in specs]


def test_json_outranks_srt_because_it_carries_speakers():
    ranked = rss.choose_transcripts(_links(("srt", "", ""), ("json", "", "")))
    assert ranked[0]["type"] == "application/json"


def test_vtt_outranks_srt_and_srt_outranks_plain_text():
    ranked = rss.choose_transcripts(_links(("text", "", ""), ("srt", "", ""),
                                           ("vtt", "", "")))
    assert [r["type"] for r in ranked] == ["text/vtt", "application/srt", "text/plain"]


def test_captions_wins_within_a_type():
    ranked = rss.choose_transcripts(_links(("srt", "", ""), ("srt", "captions", "")))
    assert ranked[0]["rel"] == "captions"


def test_a_matching_language_wins_within_a_type():
    ranked = rss.choose_transcripts(_links(("srt", "", "de"), ("srt", "", "en")),
                                    language="en")
    assert ranked[0]["language"] == "en"


def test_an_unreadable_type_is_dropped_not_guessed_at():
    assert rss.choose_transcripts(_links(("pdf", "", ""))) == []


def test_a_link_with_no_url_is_dropped():
    assert rss.choose_transcripts([{"type": "application/srt", "url": ""}]) == []


# ------------------------------ cue parsers ------------------------------

def test_srt_drops_indices_and_timings():
    cues = rss.cues_from_srt(SRT_BODY)
    assert len(cues) == 3
    assert "00:00:00" not in " ".join(c["text"] for c in cues)
    assert cues[0]["text"].startswith("He's full of shit")


def test_srt_strips_positioning_overrides_and_italics():
    cues = rss.cues_from_srt("1\n00:00:01,000 --> 00:00:02,000\n"
                             "{\\an8}<i>Hello there</i>\n")
    assert cues[0]["text"] == "Hello there"


def test_vtt_skips_the_header_and_note_and_style_blocks():
    vtt = ("WEBVTT\n\nNOTE this is a comment\n\nSTYLE\n::cue { color: red }\n\n"
           "1\n00:00:01.000 --> 00:00:02.000\nReal speech here\n")
    cues = rss.cues_from_vtt(vtt)
    assert [c["text"] for c in cues] == ["Real speech here"]


def test_vtt_reads_voice_spans_as_speakers():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Adam Curry>In the morning\n\n"
           "00:00:02.000 --> 00:00:03.000\n<v.loud John>Hello\n")
    cues = rss.cues_from_vtt(vtt)
    assert [(c["speaker"], c["text"]) for c in cues] == [
        ("Adam Curry", "In the morning"), ("John", "Hello")]


def test_vtt_strips_inline_karaoke_timestamps():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nOne<00:00:01.500>two three\n"
    assert rss.cues_from_vtt(vtt)[0]["text"] == "Onetwo three"


def test_a_vtt_block_with_no_timing_line_is_not_a_cue():
    assert rss.cues_from_vtt("WEBVTT\n\njust a stray line\n") == []


def test_podcast_json_reads_speakers():
    payload = {"segments": [{"speaker": "Adam", "body": "One."},
                            {"speaker": "John", "body": "Two."}]}
    cues = rss.cues_from_podcast_json(json.dumps(payload))
    assert [(c["speaker"], c["text"]) for c in cues] == [("Adam", "One."), ("John", "Two.")]


def test_podcast_json_without_speakers_still_parses():
    cues = rss.cues_from_podcast_json({"segments": [{"body": "Solo."}]})
    assert cues == [{"text": "Solo.", "speaker": ""}]


def test_malformed_json_is_no_cues_not_an_exception():
    assert rss.cues_from_podcast_json("{{{ not json") == []
    assert rss.cues_from_podcast_json({"nope": 1}) == []


def test_html_transcripts_go_through_the_tag_stripper():
    cues = rss.cues_from_html("<p>Hello <b>there</b></p>")
    assert "Hello there" in " ".join(c["text"] for c in cues)


# ------------------------------ flowing ------------------------------

def test_flowing_drops_timestamps_and_rejoins_the_hard_wraps():
    text, speakers = rss.flow_cues(rss.cues_from_srt(SRT_BODY))
    assert "-->" not in text
    assert speakers is False
    # The 42-char wrap mid-sentence must come back as prose, and [MUSIC] must go.
    assert "It's Thursday, August 6th." in text
    assert "MUSIC" not in text


def test_a_speaker_change_opens_a_paragraph():
    cues = [{"text": "One.", "speaker": "Adam"}, {"text": "Two.", "speaker": "Adam"},
            {"text": "Three.", "speaker": "John"}]
    text, speakers = rss.flow_cues(cues)
    assert speakers is True
    assert text == "Adam: One. Two.\n\nJohn: Three."


def test_name_colon_prefixes_are_promoted_when_there_are_at_least_three():
    cues = [{"text": f"{who}: line", "speaker": ""}
            for who in ("Adam", "John", "Guest", "Adam")]
    text, speakers = rss.flow_cues(cues)
    assert speakers is True
    assert text.startswith("Adam: line")
    assert "Adam: Adam:" not in text


def test_a_lone_note_prefix_is_not_mistaken_for_a_speaker():
    """The guard: fewer than three distinct prefixes means this is prose, not dialogue."""
    cues = [{"text": "Note: the show is late.", "speaker": ""},
            {"text": "Then it started.", "speaker": ""}]
    text, speakers = rss.flow_cues(cues)
    assert speakers is False
    assert text == "Note: the show is late. Then it started."


def test_no_cues_flows_to_nothing():
    assert rss.flow_cues([]) == ("", False)


# ------------------------------ notes ------------------------------

def test_a_substantial_feed_body_is_used_verbatim(rss_net):
    feed = _load(rss_net, [podcast_item(1, body="<p>" + "word " * 300 + "</p>",
                                        enclosure=False, transcripts=())])
    rss_net["routes"]["ep1.mp3"] = b""
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["summary_source"] == "content"
    assert "word word" in ep["summary"]


def test_a_teaser_escalates_to_a_page_fetch(rss_net, monkeypatch):
    fetched = []

    def fake(url, timeout=45):
        fetched.append(url)
        return {"url": url, "title": "T", "text": "The full article body. " * 60,
                "via": "requests"}

    monkeypatch.setattr(core, "fetch_url_text", fake)
    feed = _load(rss_net, [podcast_item(1, body="<p>Short teaser.</p>",
                                        enclosure=False, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert fetched == ["https://show.test/ep1"]      # exactly once, with the item link
    assert ep["summary_source"] == "page"


def test_an_item_with_an_enclosure_NEVER_escalates(rss_net, monkeypatch):
    """The rule that stops a 226-episode import firing 226 Playwright crawls."""
    fetched = []
    monkeypatch.setattr(core, "fetch_url_text",
                        lambda url, timeout=45: fetched.append(url) or {"text": "x"})
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1, body="<p>Short.</p>")])
    rss.fetch_episode(feed, feed["items"][0])
    assert fetched == []


def test_the_page_escalation_can_be_turned_off(rss_net, monkeypatch):
    fetched = []
    monkeypatch.setattr(core, "fetch_url_text",
                        lambda url, timeout=45: fetched.append(url) or {"text": "x"})
    feed = _load(rss_net, [podcast_item(1, body="<p>Short.</p>", enclosure=False,
                                        transcripts=())])
    rss.fetch_episode(feed, feed["items"][0], settings={"rss_fetch_pages": False})
    assert fetched == []


def test_a_failed_page_fetch_keeps_the_teaser_and_reports(rss_net, monkeypatch):
    monkeypatch.setattr(core, "fetch_url_text",
                        lambda url, timeout=45: (_ for _ in ()).throw(RuntimeError("403")))
    feed = _load(rss_net, [podcast_item(1, body="<p>Short teaser.</p>",
                                        enclosure=False, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["summary"] == "Short teaser."
    assert any("Page fetch failed" in e for e in ep["errors"])


# ------------------------------ the episode fetch ------------------------------

def test_a_published_srt_is_fetched_and_flowed(rss_net):
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1)])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["transcript_source"] == "published"
    assert ep["transcript_format"] == "srt"
    assert "Gitmo Nation" in ep["transcript"]


def test_a_404_on_the_json_falls_through_to_the_srt(rss_net):
    """Per-TYPE fallthrough. Without it a dead JSON link would drop straight into a
    forty-minute Whisper run with a perfectly good SRT sitting right there."""
    rss_net["routes"]["ep1.json"] = RuntimeError("404")
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1, transcripts=[("json", None), ("srt", None)])])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["transcript_format"] == "srt"
    assert ep["transcript"]
    assert any("json" in e for e in ep["errors"])


def test_an_empty_parse_also_falls_through(rss_net):
    rss_net["routes"]["ep1.json"] = b'{"segments": []}'
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1, transcripts=[("json", None), ("srt", None)])])
    assert rss.fetch_episode(feed, feed["items"][0])["transcript_format"] == "srt"


def test_no_advertised_transcript_records_the_negative(rss_net):
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["transcript"] == ""
    entry = rss_cache.get(ep["episode_id"])
    assert entry["published_transcript_missing"] is True


def test_a_transport_failure_does_NOT_record_the_negative(rss_net):
    """A 404 is not proof the publisher ships nothing, and the cache never expires."""
    rss_net["routes"]["ep1.srt"] = RuntimeError("503 from the CDN")
    feed = _load(rss_net, [podcast_item(1)])
    ep = rss.fetch_episode(feed, feed["items"][0])
    entry = rss_cache.get(ep["episode_id"]) or {}
    assert "published_transcript_missing" not in entry


def test_a_second_fetch_is_served_from_cache(rss_net):
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1)])
    first = rss.fetch_episode(feed, feed["items"][0])
    rss_net["get"].clear()
    again = rss.fetch_episode(feed, feed["items"][0])
    assert again["from_cache"] is True
    assert again["transcript"] == first["transcript"]
    assert rss_net["get"] == []          # not one request


def test_chapters_are_fetched_once_and_cached(rss_net):
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    rss_net["routes"]["chapters.json"] = json.dumps(
        {"chapters": [{"startTime": 0, "title": "Intro"},
                      {"startTime": 252, "title": "Fauci"}]}).encode()
    feed = _load(rss_net, [podcast_item(1, chapters=True)])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert [c["title"] for c in ep["chapters"]] == ["Intro", "Fauci"]
    rss_net["get"].clear()
    rss.fetch_episode(feed, feed["items"][0])
    assert rss_net["get"] == []


def test_a_broken_chapters_file_is_garnish_not_a_failure(rss_net):
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    rss_net["routes"]["chapters.json"] = RuntimeError("500")
    feed = _load(rss_net, [podcast_item(1, chapters=True)])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["chapters"] == []
    assert ep["transcript"]


def test_an_unidentifiable_item_raises_rather_than_being_cached_wrong(rss_net):
    feed = _load(rss_net, [podcast_item(1)])
    with pytest.raises(rss.RSSError):
        rss.fetch_episode(feed, {"title": "orphan"})


# ------------------------------ the whisper rung ------------------------------

@pytest.fixture
def whisper(monkeypatch):
    """A stub local transcriber. conftest's autouse ``no_whisper`` makes the real one
    raise, so a test that wants the rung to succeed installs this instead."""
    from app import transcribe

    seen = {"urls": [], "text": "Machine transcription of the whole show."}

    def fake(url, settings=None, on_progress=None, should_stop=None):
        seen["urls"].append(url)
        if seen.get("raises"):
            raise transcribe.TranscribeError(seen["raises"])
        if on_progress:
            on_progress("whisper", done=30.0, total=60.0, unit="seconds", device="cuda")
        return {"text": seen["text"], "language": "en", "duration": 60.0,
                "device": "cuda", "compute_type": "float16", "model": "large-v3",
                "fallback": False, "stopped": bool(seen.get("stopped")),
                "chars": len(seen["text"]), "url": url}

    monkeypatch.setattr(transcribe, "transcribe_url", fake)
    monkeypatch.setattr(transcribe, "is_available", lambda: True)
    return seen


def test_whisper_is_never_reached_when_it_was_not_asked_for(rss_net, monkeypatch):
    called = []
    monkeypatch.setattr(rss, "_whisper_rung", lambda *a, **k: called.append(1))
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    rss.fetch_episode(feed, feed["items"][0], want_whisper=False)
    assert called == []


def test_whisper_runs_when_asked_and_nothing_is_published(rss_net, whisper):
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert ep["transcript_source"] == "whisper"
    assert ep["transcript"] == whisper["text"]
    assert whisper["urls"] == ["https://cdn.test/ep1.mp3"]
    assert ep["whisper_model"] == "large-v3"


def test_a_published_transcript_wins_and_whisper_is_not_run(rss_net, whisper):
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed = _load(rss_net, [podcast_item(1)])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert ep["transcript_source"] == "published"
    assert whisper["urls"] == []


def test_whisper_needs_an_enclosure(rss_net, whisper):
    feed = _load(rss_net, [podcast_item(1, transcripts=(), enclosure=False)])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert ep["transcript"] == ""
    assert whisper["urls"] == []


def test_a_whisper_failure_is_recorded_not_raised(rss_net, whisper):
    """One dead enclosure must not cost an import of forty episodes."""
    whisper["raises"] = "the decoder gave up"
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert ep["transcript"] == ""
    assert any("Transcription failed" in e for e in ep["errors"])


def test_a_missing_faster_whisper_is_reported_on_the_episode(rss_net, monkeypatch):
    from app import transcribe
    monkeypatch.setattr(transcribe, "is_available", lambda: False)
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert any("pip install faster-whisper" in e for e in ep["errors"])


def test_whisper_output_never_claims_speakers(rss_net, whisper):
    """faster-whisper has no diarization, and synthesising labels from
    <podcast:person> would attribute lines to a host we never detected."""
    feed = _load(rss_net, [podcast_item(1, transcripts=(), persons=2)])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert ep["transcript_speakers"] is False
    assert "Adam Curry:" not in ep["transcript"]


def test_a_whisper_transcript_is_cached_and_reused(rss_net, whisper):
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    whisper["urls"].clear()
    again = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert again["from_cache"] is True
    assert whisper["urls"] == []          # not re-transcribed


def test_ticking_the_box_after_a_free_run_DOES_transcribe(rss_net, whisper):
    """The end-to-end version of the cache's key coverage row: importing a feed with the
    box unticked records "no transcript published", and ticking it later must not be
    silently satisfied by that record."""
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    first = rss.fetch_episode(feed, feed["items"][0], want_whisper=False)
    assert first["transcript"] == ""
    assert whisper["urls"] == []

    second = rss.fetch_episode(feed, feed["items"][0], want_whisper=True)
    assert second["transcript_source"] == "whisper"
    assert whisper["urls"] == ["https://cdn.test/ep1.mp3"]


def test_a_later_published_transcript_replaces_the_whisper_one(rss_net, whisper):
    """The publisher shipped one after we spent the GPU time. Theirs is better and free."""
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    rss.fetch_episode(feed, feed["items"][0], want_whisper=True)

    rss_net["routes"][URL] = feed_xml([podcast_item(1)])          # now advertises an SRT
    rss_net["routes"]["ep1.srt"] = SRT_BODY.encode()
    feed2 = rss.fetch_feed(URL)
    ep = rss.fetch_episode(feed2, feed2["items"][0], want_whisper=True)
    assert ep["transcript_source"] == "published"
    assert "Gitmo Nation" in ep["transcript"]
    assert rss_cache.get(ep["episode_id"])["transcript_source"] == "published"


def test_a_cancelled_transcription_is_not_cached(rss_net, whisper):
    whisper["stopped"] = True
    whisper["text"] = "only the first ten minutes"
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    ep = rss.fetch_episode(feed, feed["items"][0], want_whisper=True,
                           should_stop=lambda: True)
    entry = rss_cache.get(ep["episode_id"]) or {}
    assert not entry.get("transcript")


def test_whisper_progress_frames_reach_the_caller(rss_net, whisper):
    frames = []
    feed = _load(rss_net, [podcast_item(1, transcripts=())])
    rss.fetch_episode(feed, feed["items"][0], want_whisper=True,
                      on_progress=lambda p, **f: frames.append((p, f)))
    assert any(p == "whisper" for p, _ in frames)


# ------------------------------ rendering ------------------------------

def _meta(**over):
    base = {"title": "Kill Switch", "episode": 1892, "feed_title": "No Agenda",
            "published": "2026-08-06", "duration": "03:12:44",
            "link": "https://show.test/1892",
            "persons": [{"name": "Adam Curry", "role": "host"},
                        {"name": "John C Dvorak", "role": "host"}],
            "chapters": [{"start_time": 0, "title": "Intro"}],
            "summary": "The show notes.", "transcript": "Adam: In the morning."}
    base.update(over)
    return base


def test_the_header_composes_every_present_field():
    head = rss.format_episode_text(_meta()).splitlines()[0]
    assert head == "Episode 1892: Kill Switch | No Agenda | 2026-08-06 | 03:12:44"


def test_absent_sections_are_omitted_not_rendered_empty():
    text = rss.format_episode_text(_meta(persons=[], chapters=[], summary=""))
    assert "--- People ---" not in text
    assert "--- Chapters ---" not in text
    assert "--- Show notes ---" not in text


def test_the_transcript_is_rendered_LAST():
    """Reverse of format_video_text, and load-bearing: the transcript is the only
    section that ever overflows LIBRARY_PAGE_CHARS, so everything cheap and high-signal
    must sit above the cut."""
    text = rss.format_episode_text(_meta())
    assert text.index("--- People ---") < text.index("--- Transcript ---")
    assert text.index("--- Show notes ---") < text.index("--- Transcript ---")
    assert text.rstrip().endswith("Adam: In the morning.")


def test_a_missing_transcript_says_so_rather_than_rendering_blank():
    assert "(No transcript available)" in rss.format_episode_text(_meta(transcript=""))


def test_notes_and_people_can_be_switched_off():
    text = rss.format_episode_text(_meta(), include_notes=False, include_persons=False)
    assert "--- Show notes ---" not in text and "--- People ---" not in text


def test_a_long_people_list_is_capped():
    persons = [{"name": f"Person {i}", "role": ""} for i in range(30)]
    text = rss.format_episode_text(_meta(persons=persons))
    assert "+10 more" in text


def test_an_episode_over_the_page_cap_is_truncated_but_the_cache_keeps_it_all(rss_net,
                                                                             monkeypatch):
    monkeypatch.setattr(core, "LIBRARY_PAGE_CHARS", 2000)
    big = "\n\n".join(f"{i}\n00:00:0{i % 10},000 --> 00:00:0{(i + 1) % 10},000\n"
                      f"Sentence number {i} of the show."
                      for i in range(400))
    rss_net["routes"]["ep1.srt"] = big.encode()
    feed = _load(rss_net, [podcast_item(1)])
    ep = rss.fetch_episode(feed, feed["items"][0])
    assert ep["truncated"] is True
    assert len(ep["text"]) < ep["full_chars"]
    # The slice is a rendering concern: raising the cap later must cost nothing.
    assert len(rss_cache.get(ep["episode_id"])["transcript"]) > 2000


# ------------------------------ persona doc name ------------------------------

def test_the_doc_name_is_stable_across_a_retitle():
    """Publishers retitle constantly. If the name moved, add_text's doc_id would move
    with it and every re-import would duplicate the document instead of replacing it."""
    feed = {"title": "No Agenda"}
    a = rss.episode_doc_name(feed, {"episode_id": "a" * 40, "title": "Kill Switch"})
    b = rss.episode_doc_name(feed, {"episode_id": "a" * 40, "title": "Kill Switch (fixed)"})
    assert a == b


def test_the_doc_name_carries_no_title_at_all():
    name = rss.episode_doc_name({"title": "No Agenda"},
                                {"episode_id": "a" * 40, "title": "Kill Switch"})
    assert "kill" not in name.lower()
    assert name == "no-agenda-" + "a" * 12 + ".txt"


def test_the_doc_name_is_unique_per_episode():
    feed = {"title": "No Agenda"}
    a = rss.episode_doc_name(feed, {"episode_id": "a" * 40, "title": "One"})
    b = rss.episode_doc_name(feed, {"episode_id": "b" * 40, "title": "One"})
    assert a != b


def test_the_doc_name_is_filesystem_safe_and_length_capped():
    name = rss.episode_doc_name({"title": "A/Show: With\\Slashes?"},
                                {"episode_id": "c" * 40, "title": "x" * 300})
    assert not set(name) & set('/\\:*?"<>|')
    assert len(name) <= 124
    assert name.endswith(".txt")
