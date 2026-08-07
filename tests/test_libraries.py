#!/usr/bin/env python3
"""Tests for the Resources tab's libraries (app/store.py, app/core.py, app/ingest.py).

These pin down the two bugs that lost user data silently:

* the long-running add routes (file parse, URL scrape, YouTube, Brave crawl) held the
  library dict they read at request start and wrote it back minutes later. Because
  ``upsert_library`` REPLACES the list entry, that snapshot was detached and the
  write-back erased every autosave that had landed in between;
* deleting a library left its id in ``chat.library_ids``, which keeps RAG active for
  those chats forever with nothing to retrieve, and invisibly — the Library button
  filters unknown ids out, so it just reads "none".

Plus the smaller guarantees with no coverage at all: XML export/import round-tripping,
cancellable document parsing, and a library that compiles to zero items reporting
"compiled" rather than contradicting the toast that just said so.

Everything runs against a throwaway tree; no real profile is touched.
"""

from app import compile as compile_mod
from app import core, ingest, rag


# --------------------------- harness ---------------------------

# `store` and `client` come from tests/conftest.py.

def _lib_with_item(store, name="Refs", content="original"):
    lib = core._new_library(name)
    lib["items"] = [core._new_library_item(item_type="write", label="note",
                                           content=content)]
    store.add_library(lib)
    return lib


# --------------------------- concurrent append (A1) ---------------------------

def test_append_library_items_keeps_a_concurrent_edit(store):
    """The exact shape of the bug: a route reads the library, an autosave PUT lands
    while it works, and the route then appends. The edit must survive."""
    lib = _lib_with_item(store, content="original")
    lib_id = lib["id"]

    held = store.get_library(lib_id)                      # what the route grabbed
    assert held is not None

    # ...meanwhile the browser autosaves an edit through PUT /api/libraries/<id>.
    edited = {**held, "items": [{**held["items"][0], "content": "user typed this"}]}
    store.upsert_library(edited)

    # ...and only now does the crawl finish and append its page.
    saved = store.append_library_items(
        lib_id, [core._new_library_item(item_type="url", label="Page",
                                        content="scraped", filename="https://x/y")])

    assert saved is not None
    contents = [i["content"] for i in saved["items"]]
    assert contents == ["user typed this", "scraped"]
    # And it is what actually reached disk, not just what we were handed back.
    assert [i["content"] for i in store.get_library(lib_id)["items"]] == contents


def test_the_old_snapshot_pattern_would_have_lost_the_edit(store):
    """Pins WHY append_library_items exists. This is what the add routes used to do;
    if anyone reverts to it, the assertion below documents exactly what breaks."""
    lib = _lib_with_item(store, content="original")
    held = store.get_library(lib["id"])

    store.upsert_library({**held, "items": [{**held["items"][0], "content": "user typed this"}]})

    held.setdefault("items", []).append(core._new_library_item(content="scraped"))
    store.upsert_library(held)                            # the stale write-back

    assert [i["content"] for i in store.get_library(lib["id"])["items"]] \
        == ["original", "scraped"]                        # the edit is gone


def test_append_library_items_returns_none_when_deleted_midflight(store):
    lib_id = _lib_with_item(store)["id"]
    store.delete_library(lib_id)
    assert store.append_library_items(lib_id, [core._new_library_item()]) is None


def test_append_library_items_assigns_ids(store):
    lib_id = _lib_with_item(store)["id"]
    saved = store.append_library_items(lib_id, [{"type": "write", "content": "x"}])
    assert saved["items"][-1]["id"]


# --------------------------- orphan chat refs (A4) ---------------------------

def test_delete_library_prunes_chat_references(store):
    lib_id = _lib_with_item(store)["id"]
    store.chats = [
        {"id": "c1", "library_ids": [lib_id, "other"], "library_strict": True},
        {"id": "c2", "library_ids": []},
    ]
    store.save_chats()

    store.delete_library(lib_id)

    assert store.chats[0]["library_ids"] == ["other"]
    assert store.chats[1]["library_ids"] == []


def test_load_repairs_chats_orphaned_by_an_older_delete(store):
    """Chats broken before prune_library_refs existed are fixed on the next load."""
    kept = _lib_with_item(store, name="Kept")
    store.chats = [{"id": "c1", "library_ids": [kept["id"], "deleted-long-ago"]}]
    store.save_chats()

    store._load_data_collections()

    assert store.chats[0]["library_ids"] == [kept["id"]]


# --------------------------- route wiring ---------------------------

def test_add_url_route_appends_atomically_and_returns_the_new_item(client, monkeypatch):
    """The add routes must return `added_items` (the browser appends those instead of
    adopting a whole server copy of the library, which discarded unsaved edits) and must
    not clobber a PUT that landed while the scrape was running."""
    lib_id = client.post("/api/libraries", json={"name": "Refs"}).get_json()["library"]["id"]

    scraped = {"url": "https://example.com/a", "title": "A Page",
               "text": "page body", "via": "requests"}

    def fake_fetch(url, timeout=45):
        # Stand in for the minutes a real scrape takes: the browser autosaves an edit
        # partway through, exactly as it would over a slow crawl.
        client.put(f"/api/libraries/{lib_id}",
                   json={"library": {"id": lib_id, "name": "Refs",
                                     "items": [{"type": "write", "content": "typed while waiting"}]}})
        return scraped

    monkeypatch.setattr(core, "fetch_url_text", fake_fetch)

    r = client.post(f"/api/libraries/{lib_id}/add-url", json={"url": "https://example.com/a"})
    body = r.get_json()

    assert r.status_code == 200
    assert [i["label"] for i in body["added_items"]] == ["A Page"]
    assert body["added_items"][0]["id"]        # ids are assigned before the reply
    contents = [i["content"] for i in body["library"]["items"]]
    assert contents == ["typed while waiting", "page body"]

    fresh = client.get("/api/libraries").get_json()["libraries"]
    assert [i["content"] for i in fresh[0]["items"]] == contents


def test_add_url_route_404s_when_the_library_vanished(client, monkeypatch):
    lib_id = client.post("/api/libraries", json={"name": "Doomed"}).get_json()["library"]["id"]

    def fake_fetch(url, timeout=45):
        client.delete(f"/api/libraries/{lib_id}")
        return {"url": url, "title": "T", "text": "x", "via": "requests"}

    monkeypatch.setattr(core, "fetch_url_text", fake_fetch)
    r = client.post(f"/api/libraries/{lib_id}/add-url", json={"url": "https://example.com"})
    assert r.status_code == 404


def test_streaming_add_routes_mint_a_unique_run_id(client, monkeypatch):
    """Each streaming add route must announce its OWN run id in the `start` frame.

    These used to be fixed strings composed from the library id, which is broken twice
    over. ``RunRegistry.new`` overwrites, so two runs on one library orphaned each
    other's stop event and whichever generator finished first deregistered the survivor;
    and the YouTube Cancel button composed the id from whatever library happened to be
    selected at the time, so after a switch it stopped nothing while the fetch still
    appended. The clients now read the id from `start` instead of composing it.
    """
    from tests.conftest import first, sse_frames

    lib_id = client.post("/api/libraries", json={"name": "Refs"}).get_json()["library"]["id"]
    monkeypatch.setattr(core, "crawl_search",
                        lambda *a, **kw: iter([{"type": "result", "pages": [],
                                                "errors": [], "attempted": 0}]))
    monkeypatch.setattr("app.youtube.fetch_video",
                        lambda url, **kw: {"url": url, "title": "V", "text": "t",
                                           "via": "requests", "errors": [],
                                           "comments": [], "transcript": "t"})
    monkeypatch.setattr("app.native_dialog.pick_files", lambda **kw: [])

    calls = [
        ("post", f"/api/libraries/{lib_id}/add-text-files", "addfiles-library-"),
        ("get", f"/api/libraries/{lib_id}/brave-search?q=hi", "brave-library-"),
        ("get", f"/api/libraries/{lib_id}/add-youtube?url=https://youtu.be/abcdefghijk",
         "youtube-library-"),
    ]
    seen = set()
    for verb, path, prefix in calls:
        for _ in range(2):        # twice, to prove the ids differ per invocation
            resp = client.post(path, json={}) if verb == "post" else client.get(path)
            run_id = first(sse_frames(resp), "start")["run_id"]
            assert run_id.startswith(f"{prefix}{lib_id}-"), run_id
            assert run_id not in seen, f"{path} reused {run_id}"
            seen.add(run_id)


def test_both_crawl_routes_clamp_the_page_count(client, monkeypatch):
    """``core.crawl_search`` only bounds max_results from below, and an <input max="20">
    is not enforced against a typed value — so every entry point has to clamp. The
    Resources route always did; the composer's ``/api/brave-search-text`` did not, and a
    mistyped 500 was 500 page fetches against a metered API."""
    from tests.conftest import sse_frames

    asked = []

    def fake_crawl(query, sites=None, max_results=5, should_stop=None):
        asked.append(max_results)
        yield {"type": "result", "pages": [], "errors": [], "attempted": 0}
    monkeypatch.setattr(core, "crawl_search", fake_crawl)

    lib_id = client.post("/api/libraries", json={"name": "Refs"}).get_json()["library"]["id"]
    for path in (f"/api/libraries/{lib_id}/brave-search?q=hi&max=500",
                 "/api/brave-search-text?q=hi&max=500"):
        sse_frames(client.get(path))
        assert asked[-1] == core.MAX_CRAWL_PAGES, path

    # Garbage falls back to the default rather than raising out of the route.
    sse_frames(client.get("/api/brave-search-text?q=hi&max=lots"))
    assert asked[-1] == 5


def test_delete_route_prunes_the_reference_from_chats(client):
    lib_id = client.post("/api/libraries", json={"name": "Refs"}).get_json()["library"]["id"]
    chat = client.post("/api/chats", json={"title": "T"}).get_json()["chat"]
    r = client.patch(f"/api/chats/{chat['id']}",
                     json={"patch": {"library_ids": [lib_id], "library_strict": True}})
    assert r.get_json()["chat"]["library_ids"] == [lib_id]

    client.delete(f"/api/libraries/{lib_id}")

    after = client.get(f"/api/chats/{chat['id']}").get_json()["chat"]
    assert after["library_ids"] == []


# --------------------------- cancellable parsing (B3) ---------------------------

def test_extract_many_stops_when_asked(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"contents {i}", encoding="utf-8")
        paths.append(str(p))

    results = ingest.extract_many(paths, workers=1, should_stop=lambda: True)

    assert len(results) == len(paths)
    assert all(not r["ok"] and r["error"] == "cancelled" for r in results)


def test_extract_many_without_should_stop_parses_everything(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world", encoding="utf-8")
    results = ingest.extract_many([str(p)], workers=1)
    assert len(results) == 1 and results[0]["ok"]
    assert "hello world" in results[0]["text"]


# --------------------------- XML round-trip ---------------------------

def test_library_xml_round_trip_preserves_items(tmp_path):
    lib = core._new_library("Research")
    lib["items"] = [
        core._new_library_item(item_type="write", label="Notes", content="alpha\nbeta"),
        core._new_library_item(item_type="url", label="Page", content="<b>&amp;</b> raw",
                               filename="https://example.com/a?x=1&y=2"),
    ]
    path = tmp_path / "lib.xml"
    path.write_bytes(core.library_to_xml_bytes(lib))

    back = core.library_from_xml_file(path)

    assert back["name"] == "Research"
    assert back["id"] != lib["id"]                # imports get a fresh library id
    assert len(back["items"]) == 2
    for original, restored in zip(lib["items"], back["items"]):
        assert restored["id"] == original["id"]   # item ids survive, so RAG keys line up
        assert restored["type"] == original["type"]
        assert restored["label"] == original["label"]
        assert restored["content"] == original["content"]
        assert restored["filename"] == original["filename"]


def test_library_xml_export_survives_control_characters(tmp_path):
    lib = core._new_library("Odd")
    lib["items"] = [core._new_library_item(content="ok\x0ctext\x00here")]
    path = tmp_path / "lib.xml"
    path.write_bytes(core.library_to_xml_bytes(lib))
    assert core.library_from_xml_file(path)["items"][0]["content"] == "oktexthere"


# --------------------------- compile status (D2) ---------------------------

def test_empty_library_status_is_compiled_not_none(store, monkeypatch):
    """A library whose items are all empty compiles to zero items. Reporting that as
    'not compiled' made the badge contradict the toast that had just said otherwise."""
    lib = core._new_library("Empty")
    lib["items"] = [core._new_library_item(item_type="write", content="   ")]
    store.add_library(lib)

    monkeypatch.setattr(rag, "count_chunks", lambda *a, **k: 0)
    key = compile_mod._manifest_key(compile_mod.LIBRARY, lib["id"])
    compile_mod._put_manifest(key, {
        "signature": compile_mod.signature("nomic-embed-text"),
        "items": {}, "chunks": 0, "compiled_at": "2026-01-01T00:00:00",
    })

    assert compile_mod.library_status(lib, "nomic-embed-text")["state"] == "compiled"


def test_never_compiled_library_status_is_none(store, monkeypatch):
    lib = _lib_with_item(store, name="Fresh", content="real content")
    monkeypatch.setattr(rag, "count_chunks", lambda *a, **k: 0)
    assert compile_mod.library_status(lib, "nomic-embed-text")["state"] == "none"


def test_forget_library_drops_manifests_for_every_backend(store):
    lib_id = "abc123"
    compile_mod._put_manifest(f"{compile_mod.LIBRARY}:{lib_id}@lance", {"items": {"a": {}}})
    compile_mod._put_manifest(f"{compile_mod.LIBRARY}:{lib_id}@duckdb", {"items": {"a": {}}})
    compile_mod._put_manifest(f"{compile_mod.LIBRARY}:keep-me@lance", {"items": {"a": {}}})

    compile_mod.forget_library(lib_id)

    remaining = compile_mod._load_all()
    assert not [k for k in remaining if k.startswith(f"{compile_mod.LIBRARY}:{lib_id}")]
    assert f"{compile_mod.LIBRARY}:keep-me@lance" in remaining
