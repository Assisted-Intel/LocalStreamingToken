#!/usr/bin/env python3
"""Shared test harness.

Every test file used to carry its own verbatim copy of the eight ``core.*`` path
monkeypatches plus a logged-in Flask client, and two files hand-rolled their own
(subtly different) SSE parsers. This is that harness, once.

Three things live here:

* ``isolate_paths`` / the ``store`` and ``client`` fixtures — point the app's data and
  settings at a throwaway tree so no real profile is ever touched;
* ``sse_frames`` — one SSE parser;
* ``StubAdapter`` / ``use_adapter`` — the seam for routes that call a model.

Domain-specific fakes stay in the file that needs them (``FakeAdapter`` in
test_evals_routes.py routes on a grader marker, which only that suite cares about).
"""

import json

import pytest

from app import core, crypto, profiles, providers


# --------------------------- isolation ---------------------------

@pytest.fixture(autouse=True)
def isolate_youtube_cache(tmp_path, monkeypatch):
    """Give every test its own empty YouTube cache dir.

    Autouse and unconditional, unlike the rest of the isolation here, because
    ``youtube.fetch_video`` writes to this path on any successful fetch — including
    from suites that never call ``isolate_paths`` (test_youtube.py drives the fetch
    chain directly). Without this they would cache into the developer's real data
    profile, and the second test in a run would then be served by the first one's cache
    instead of exercising the transport chain it means to test.
    """
    monkeypatch.setattr(core, "YOUTUBE_CACHE_DIR", tmp_path / "youtube_cache")


@pytest.fixture(autouse=True)
def isolate_rss_cache(tmp_path, monkeypatch):
    """Give every test its own empty RSS/podcast cache dir.

    Autouse and unconditional for the same reason as ``isolate_youtube_cache``:
    ``rss.fetch_episode`` writes on any successful fetch, including from suites that
    drive it directly and never call ``isolate_paths``.
    """
    monkeypatch.setattr(core, "RSS_CACHE_DIR", tmp_path / "rss_cache")


@pytest.fixture(autouse=True)
def no_whisper(monkeypatch):
    """Make local transcription raise in every test that has not opted in.

    faster-whisper downloads ~3 GB of weights on first use and then pins a GPU for
    minutes. No test may reach it by accident — a suite that means to exercise the
    transcription path installs its own stub (see tests/test_transcribe.py, which
    injects a fake ``faster_whisper`` module) and overrides this.
    """
    from app import transcribe

    def refuse(*a, **k):
        raise transcribe.TranscribeError("transcription is disabled in tests")

    monkeypatch.setattr(transcribe, "transcribe_file", refuse)
    monkeypatch.setattr(transcribe, "transcribe_url", refuse)


class RealNetworkAttempted(BaseException):
    """A test reached for the actual internet.

    Deliberately not an ``Exception``: the app is full of ``except Exception`` blocks
    (``server.py``'s LAN probe, for one) that would swallow this and turn a leaked call
    back into the silence this fixture exists to end.
    """


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Make any real HTTP call raise, in every test.

    Every outbound request in the app goes through the module-level ``requests``
    helpers, and all five modules that make them (core, rss, server, transcribe,
    youtube) share one ``requests`` module object, so patching it here covers all of
    them at once. Suites that exercise a transport chain stub it a rung higher —
    ``youtube._requests_get``, ``rss._requests_get`` — and never reach this.

    This is here because two cache tests in test_youtube.py called ``monkeypatch.undo()``
    to drop a single patch. ``undo()`` reverts *every* patch on the instance, including
    the fixture's HTTP stubs, so both quietly fetched the real youtube.com and asserted
    against whatever it returned. They failed on the *content* rather than on the act of
    calling out, which is a slow and confusing way to find out; a leak that happened to
    agree with the assertion would never have been noticed at all.
    """
    import requests

    def refuse(name):
        def blocked(*a, **k):
            where = a[0] if a else k.get("url", "?")
            raise RealNetworkAttempted(
                f"test tried to reach the network: requests.{name} {where}. "
                "Stub the transport instead (see the youtube_net / rss_net fixtures).")
        return blocked

    for name in ("get", "post", "head", "put", "patch", "delete", "request"):
        monkeypatch.setattr(requests, name, refuse(name))
    monkeypatch.setattr(requests.Session, "request", refuse("Session.request"))


def isolate_paths(tmp_path, monkeypatch, unlock=False):
    """Redirect every data/settings path at ``tmp_path`` and create the keyfile.

    Must run BEFORE ``server.create_app()`` — ``Store()`` reads ``core.*`` at
    construction, which is why every caller imports ``app.server`` lazily afterwards.

    ``unlock`` activates the data encryption key directly, for tests that build a
    ``Store`` without going through ``/api/login`` (the route is what normally
    activates it).
    """
    data, settings = tmp_path / "data", tmp_path / "settings"
    monkeypatch.setattr(core, "DATA_DIR", data)
    monkeypatch.setattr(core, "SETTINGS_DIR", settings)
    monkeypatch.setattr(core, "DATA_PROFILES_DIR", data / "profiles")
    monkeypatch.setattr(core, "SETTINGS_PROFILES_DIR", settings / "profiles")
    monkeypatch.setattr(core, "DATA_REGISTRY_FILE", data / "profiles.json")
    monkeypatch.setattr(core, "SETTINGS_REGISTRY_FILE", settings / "profiles.json")
    monkeypatch.setattr(core, "INCOGNITO_DIR", data / "profiles" / ".incognito")
    monkeypatch.setattr(core, "APP_KEYFILE", settings / "app_key.enc")
    # Plaintext and app-wide, so it is NOT covered by the profile redirection above and
    # a test that saves network settings would otherwise write into the developer's real
    # settings/ folder — and change where their app binds.
    monkeypatch.setattr(core, "NETWORK_FILE", settings / "network.json")
    for d in (data, settings):
        d.mkdir(parents=True, exist_ok=True)
    crypto.create_keyfile(core.APP_KEYFILE)          # default admin/admin
    if unlock:
        # The store's data files are encrypted at rest and unreadable until the DEK is
        # active — the same gate /api/login passes through.
        crypto.set_active_key(crypto.unlock(core.APP_KEYFILE, "admin"))

    pm = profiles.ProfileManager()
    core.set_active_settings_profile(pm.active_settings_dir())
    core.set_active_data_profile(pm.active_data_dir())


def make_client(tmp_path, monkeypatch):
    """Build and log into a test client. Split out from the fixture so a suite needing
    extra setup or teardown (test_db_routes releases its staging file) can reuse it."""
    isolate_paths(tmp_path, monkeypatch)
    from app import server
    app = server.create_app()
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200, r.get_data(as_text=True)
    return c


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real Store whose data lives entirely under tmp_path."""
    from app import store as store_mod
    isolate_paths(tmp_path, monkeypatch, unlock=True)
    return store_mod.Store()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A logged-in test client for an app whose data lives entirely under tmp_path.
    No model server is reachable, which is exactly the state these routes must behave
    sanely in — install a stub with ``use_adapter`` when one is needed."""
    return make_client(tmp_path, monkeypatch)


# --------------------------- YouTube network ---------------------------
# Lives here rather than in test_youtube.py because test_batch.py needs it too: the
# preview-then-run cache test has to drive the REAL fetch chain, so monkeypatching
# youtube.fetch_video (which the other batch tests do) would defeat the point.

_YT_PLAYER = {
    "videoDetails": {"title": "How Engines Work", "author": "Garage Lab",
                     "viewCount": "1240113"},
    "microformat": {"playerMicroformatRenderer": {"publishDate": "2024-03-12"}},
    "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": [
        {"languageCode": "en", "vssId": ".en", "baseUrl": "https://timedtext/x"}]}},
}
_YT_JSON3 = {"events": [{"segs": [{"utf8": "The engine turns."}]}]}

# ytInitialData needs a comments continuation token, or the comment pager correctly
# concludes the video has no comment section and never calls the API.
_YT_INITIAL_DATA = {"engagementPanels": {
    "commentsEntryPointHeaderRenderer": {},
    "continuationCommand": {"token": "seed-token"},
}}


def yt_entity_payload(author, text, likes="412", published="2 months ago"):
    return {"commentEntityPayload": {
        "properties": {"content": {"content": text}, "publishedTime": published},
        "author": {"displayName": author},
        "toolbar": {"likeCountNotliked": likes},
    }}


def yt_watch_html(player=None, initial=None):
    return ("var ytInitialPlayerResponse = " + json.dumps(player or _YT_PLAYER) + ";"
            "var ytInitialData = " + json.dumps(initial or _YT_INITIAL_DATA) + ";"
            '"INNERTUBE_API_KEY":"AIzaTESTKEY"')


@pytest.fixture
def youtube_net(monkeypatch):
    """Stand in for every HTTP call youtube.py makes, recording what was requested."""
    from app import youtube

    calls = {"get": [], "post": []}

    def fake_get(url, timeout=45):
        calls["get"].append(url)
        if "timedtext" in url:
            return json.dumps(_YT_JSON3)
        return yt_watch_html()

    def fake_post(url, body, timeout=45):
        calls["post"].append((url, body))
        return {"frameworkUpdates": {"entityBatchUpdate": {"mutations": [
            {"payload": yt_entity_payload("Ada", "Great explainer")}]}}}

    monkeypatch.setattr(youtube, "_requests_get", fake_get)
    monkeypatch.setattr(youtube, "_requests_post_json", fake_post)
    return calls


@pytest.fixture
def no_ytdlp(monkeypatch):
    """Block the optional yt-dlp rung so a fetch can't reach the real network.

    test_youtube.py has its own autouse version that also hands back the real function;
    this is the plain block, for suites that just need the chain to stop at requests.
    """
    from app import youtube

    def refuse(*a, **k):
        raise youtube.YouTubeError("yt-dlp is not installed")

    monkeypatch.setattr(youtube, "_fetch_via_ytdlp", refuse)


# --------------------------- RSS / podcast feeds ---------------------------
# Lives here rather than in test_rss.py because test_rss_routes.py, test_batch.py and
# test_personas.py all need to build a feed, the same way yt_watch_html is shared.

def podcast_item(n=1, *, guid=None, title=None, transcripts=(("srt", "captions"),),
                 enclosure=True, body="", link=None, pubdate=None, persons=2,
                 chapters=False, duration="3600", categories=(), domain_category="",
                 keywords="", itunes_category=""):
    """One <item>. ``transcripts`` is [(kind, rel)]; kind in srt|vtt|json|text|html|pdf.

    Emits REAL Podcasting 2.0 markup — repeated <podcast:person> and multiple
    <podcast:transcript> elements — because that is precisely what feedparser flattens
    away and app/rss.py parses out of the raw XML instead.

    The four category flavours are separate parameters rather than one list because
    ``rss._terms`` rests on feedparser folding all of them into a single flat
    ``entry.tags``: plain <category>, <category domain=…>, <itunes:category text=…> and
    comma-separated <itunes:keywords>. A test that only emitted the plain form would not
    notice feedparser changing its mind about the others.
    """
    types = {"srt": "application/srt", "vtt": "text/vtt", "json": "application/json",
             "text": "text/plain", "html": "text/html", "pdf": "application/pdf"}
    guid = f"https://show.test/ep{n}" if guid is None else guid
    link = f"https://show.test/ep{n}" if link is None else link
    parts = [f"    <title>{title if title is not None else f'Episode {n}'}</title>"]
    if guid:
        parts.append(f"    <guid isPermaLink='true'>{guid}</guid>")
    if link:
        parts.append(f"    <link>{link}</link>")
    parts.append(f"    <pubDate>{pubdate or 'Thu, 06 Aug 2026 22:05:47 +0000'}</pubDate>")
    if duration:
        parts.append(f"    <itunes:duration>{duration}</itunes:duration>")
    if body:
        parts.append(f"    <content:encoded><![CDATA[{body}]]></content:encoded>")
    for cat in categories:
        parts.append(f"    <category>{cat}</category>")
    if domain_category:
        parts.append(f"    <category domain='http://test/tax'>{domain_category}</category>")
    if itunes_category:
        parts.append(f"    <itunes:category text='{itunes_category}' />")
    if keywords:
        parts.append(f"    <itunes:keywords>{keywords}</itunes:keywords>")
    if enclosure:
        parts.append(f"    <enclosure url='https://cdn.test/ep{n}.mp3' "
                     f"type='audio/mpeg' length='1000' />")
    for kind, rel in transcripts:
        relattr = f" rel='{rel}'" if rel else ""
        parts.append(f"    <podcast:transcript url='https://cdn.test/ep{n}.{kind}' "
                     f"type='{types[kind]}'{relattr} />")
    for i in range(persons):
        who = ["Adam Curry", "John C Dvorak", "Guest Three"][i % 3]
        parts.append(f"    <podcast:person role='host' group='cast'>{who}</podcast:person>")
    if chapters:
        parts.append(f"    <podcast:chapters url='https://cdn.test/ep{n}.chapters.json' "
                     f"type='application/json' />")
    return "  <item>\n" + "\n".join(parts) + "\n  </item>"


def feed_xml(items=None, *, title="Test Show", language="en", categories=()):
    """A Podcasting 2.0 RSS document. ``items`` is a list of podcast_item() strings.

    ``categories`` is show-level <itunes:category>. A tuple entry emits the NESTED
    subcategory form — <itunes:category text="News"><itunes:category text="Politics"/>
    </itunes:category> — which is what real podcast feeds publish and what the zero-match
    warning quotes back when a show is categorised but its episodes are not.
    """
    if items is None:
        items = [podcast_item(1)]
    chan = []
    for cat in categories:
        if isinstance(cat, (list, tuple)):
            subs = "".join(f"<itunes:category text='{s}' />" for s in cat[1:])
            chan.append(f"  <itunes:category text='{cat[0]}'>{subs}</itunes:category>\n")
        else:
            chan.append(f"  <itunes:category text='{cat}' />\n")
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<rss version='2.0'\n"
        "  xmlns:itunes='http://www.itunes.com/dtds/podcast-1.0.dtd'\n"
        "  xmlns:content='http://purl.org/rss/1.0/modules/content/'\n"
        # Deliberately the GitHub-docs URL the sample feed really uses, NOT the one the
        # spec text gives — app/rss.py must match on local name, not namespace URI.
        "  xmlns:podcast='https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md'>\n"
        "<channel>\n"
        f"  <title>{title}</title>\n"
        f"  <language>{language}</language>\n"
        "  <link>https://show.test/</link>\n"
        "  <description>A test show.</description>\n"
        + "".join(chan)
        + "\n".join(items) +
        "\n</channel>\n</rss>\n"
    ).encode("utf-8")


SRT_BODY = (
    "1\n00:00:00,280 --> 00:00:06,163\nHe's full of shit. It's Thursday,\n\n"
    "2\n00:00:06,304 --> 00:00:09,966\nAugust 6th. This is your Gitmo Nation\n\n"
    "3\n00:00:09,986 --> 00:00:15,770\nMedia Assassination. [MUSIC]\n"
)


@pytest.fixture
def rss_net(monkeypatch):
    """Stand in for every HTTP call app/rss.py makes, recording what was requested.

    ``routes`` maps a URL substring to bytes, an int status, or an Exception to raise.
    Anything unmatched is a 404, so a test that forgets to register a URL fails loudly
    rather than reaching the network.

    That promise covers two rungs, not one. Feed and enclosure traffic goes through
    ``rss._http_get``, but the show-notes escalation — a body under ``rss_notes_min_chars``
    falls back to crawling the item's own ``<link>`` page — goes through
    ``core.fetch_url_text`` instead, which starts with a Playwright launch. Stubbing only
    the first left tests that never think about show notes (an item with no enclosure
    escalates by definition) quietly crawling the real ``show.test`` domain, and
    ``rss.episode_notes`` catches Exception around it, so it showed up as nothing worse
    than a "Page fetch failed" string in the result. Tests that mean to exercise the
    escalation still override this with their own stub.
    """
    from app import rss

    calls = {"get": [], "routes": {}, "headers": [], "pages": []}

    def fake_get(url, *, headers=None, timeout=45, max_bytes=None):
        calls["get"].append(url)
        calls["headers"].append(dict(headers or {}))
        for needle, value in calls["routes"].items():
            if needle in url:
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, int):
                    return value, {}, b""
                return 200, {"ETag": 'W/"x"'}, value
        raise RuntimeError(f"404 for {url}")

    def fake_page(url, timeout=45):
        calls["pages"].append(url)
        for needle, value in calls["routes"].items():
            if needle in url and isinstance(value, (bytes, str)):
                body = value.decode() if isinstance(value, bytes) else value
                return {"url": url, "title": "", "text": body, "via": "requests"}
        raise RuntimeError(f"404 for {url}")

    monkeypatch.setattr(rss, "_http_get", fake_get)
    monkeypatch.setattr(core, "fetch_url_text", fake_page)
    return calls


# --------------------------- SSE ---------------------------

def sse_frames(resp):
    """Parse an SSE response body into ``[(event, data-dict), ...]``.

    The body HAS to be consumed for the route to do anything: ``stream_with_context``
    is lazy, so the handler's generator does not run until something reads it. Calling
    this is what runs the route.
    """
    out = []
    for block in resp.get_data(as_text=True).split("\n\n"):
        event, data = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event and data:
            out.append((event, json.loads(data)))
    return out


def events(frames):
    return [e for e, _ in frames]


def first(frames, name):
    return next(d for e, d in frames if e == name)


def all_of(frames, name):
    return [d for e, d in frames if e == name]


# --------------------------- the model seam ---------------------------

class StubAdapter:
    """A provider adapter replaying canned answers, in call order.

    Deliberately has NO ``client`` attribute. ``rewrite.run_completion`` short-circuits
    to ``client.complete`` whenever ``fmt`` is set and the adapter exposes one
    (rewrite.py:51) — and every memory and persona pass sends a schema — so an adapter
    with a mock client would never reach ``chat_stream`` and the stub would be silently
    bypassed. Do not add one.

    A reply may be an ``Exception`` instance, which is raised instead of yielded; that
    is how a failing model call is injected. It may also be a *list* of ``(kind,
    payload)`` frames, which are yielded verbatim — the way an ``("image", ...)`` or
    ``("reasoning", ...)`` stream is injected.
    """

    def __init__(self, replies=(), models=("test-model",)):
        self.replies = list(replies)
        self._models = list(models)
        self.seen = []          # [(model, messages, options)] for every call

    # --- assertion helpers ---
    @property
    def calls(self):
        return len(self.seen)

    def prompt(self, i=0):
        """The i-th call's messages flattened to one string."""
        return "\n".join(m.get("content", "") for m in self.seen[i][1])

    def system(self, i=0):
        for m in self.seen[i][1]:
            if m.get("role") == "system":
                return m.get("content", "")
        return ""

    # --- adapter interface ---
    def list_models(self):
        return list(self._models)

    def model_capabilities(self, model):
        return []               # not a reasoning model, no tool support

    def chat_stream(self, model, messages, options, stop_event, **kw):
        self.seen.append((model, messages, options))
        reply = self.replies.pop(0) if self.replies else ""
        if callable(reply):
            reply = reply(self)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            yield from reply
            return
        yield ("content", reply)


def use_adapter(monkeypatch, adapter):
    """Point every provider lookup at one adapter.

    ``adapter_for`` is a closure inside ``create_app``, so the seam is
    ``providers.get_client`` — which also covers the one place (``_ollama_caps``) that
    calls it directly rather than through ``adapter_for``.
    """
    monkeypatch.setattr(providers, "get_client", lambda server: adapter)
    return adapter


def stub_embeddings(monkeypatch, dims=768):
    """Give a route test a deterministic embedder instead of a live Ollama.

    ``use_adapter`` covers the *chat* provider, but RAG embedding does not go through
    ``providers.get_client`` — ``server._rag_retrieve_frames`` and the persona knowledge
    service each build a ``core.OllamaClient`` directly and call ``.embed``, so a route
    test with ``rag_enabled`` on, or one that adds persona knowledge, reaches
    ``127.0.0.1:11434`` for real. Some of that happens on ``rag.py`` worker threads, where
    the failure surfaces only as a PytestUnhandledThreadExceptionWarning.

    Vectors are unit-length and content-derived, so identical text embeds identically and
    retrieval is stable, without asserting anything about what the real model would say.
    """
    def embed(self, model, texts):
        out = []
        for t in texts:
            h = hash(t)
            v = [((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dims)]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    monkeypatch.setattr(core.OllamaClient, "embed", embed)
    return embed
