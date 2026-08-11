#!/usr/bin/env python3
"""Per-chat RAG scope: what RAG searches besides the selected libraries.

Three things are worth pinning here. The corpus builders' item identity, because the
whole incremental story rests on a raw message index being a stable key. The privacy
carve-out, because the default vector store is plaintext on disk and a private chat's
messages must never reach it. And the incremental sync itself — a second send over an
unchanged chat must embed nothing at all, or thread scope would re-pay for the whole
conversation on every turn.

Embedders are fake and deterministic (sha256-seeded), as in test_vectorstore.py, so
nothing here needs Ollama or a network.
"""

import hashlib
import random

import pytest

from app import compile as compile_mod
from app import core, crypto, logic, rag
from conftest import StubAdapter, events, sse_frames, use_adapter

DIM = 32
MODEL = "fake-embed"


# ------------------------------ fixtures / helpers ------------------------------

@pytest.fixture
def vecstore(tmp_path):
    """A throwaway vector store, unlocked so the DuckDB path is genuinely encrypted."""
    prev = (core.RAG_DB_FILE, core.RAG_LANCE_DIR, core.COMPILED_FILE)
    kf = tmp_path / "app_key.enc"
    crypto.create_keyfile(kf, password="admin")
    crypto.set_active_key(crypto.unlock(kf, "admin"))
    core.RAG_DB_FILE = tmp_path / "rag.duckdb"
    core.RAG_LANCE_DIR = tmp_path / "rag.lance"
    core.COMPILED_FILE = tmp_path / "compiled.json"
    rag.reset_connection()
    rag.set_backend("duckdb")       # always available; Lance is optional
    yield
    rag.reset_connection()
    core.RAG_DB_FILE, core.RAG_LANCE_DIR, core.COMPILED_FILE = prev
    crypto.clear_key()


class CountingEmbedder:
    """Deterministic pseudo-embedder that records how much it was asked to embed.

    The call count IS the assertion for most of these tests: "an unchanged chat costs
    no embedding" is only checkable by watching this stay at zero.
    """

    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        out = []
        for t in texts:
            self.texts.append(t)
            rnd = random.Random(hashlib.sha256(t.encode()).digest())
            out.append([rnd.uniform(-1, 1) for _ in range(DIM)])
        return out

    @property
    def calls(self):
        return len(self.texts)

    def reset(self):
        self.texts = []


def chat(messages=(), **kw):
    """A chat dict with the fields these tests care about."""
    base = {"id": "chat123456ab", "messages": list(messages), "rag_scope": "both"}
    base.update(kw)
    return base


def turn(role, text):
    return {"role": role, "content": text}


LONG = ("The harbour lantern swung against the storm while the captain read the "
        "chart and the anchor rope went taut across the deck.")
LONG2 = ("Signal flags rose over the tide as the engine turned and the ship came "
         "about towards the far channel marker beyond the point.")


# ------------------------------ scope normalisation ------------------------------

def test_rag_scope_defaults_to_attachments():
    """Chats saved before the control existed must keep behaving as they always did."""
    assert logic.rag_scope({}) == "attachments"
    assert logic.rag_scope({"rag_scope": None}) == "attachments"
    assert logic.rag_scope({"rag_scope": "nonsense"}) == "attachments"


def test_rag_scope_normalises_case_and_whitespace():
    assert logic.rag_scope({"rag_scope": " Thread "}) == "thread"
    assert logic.rag_scope({"rag_scope": "BOTH"}) == "both"


# ------------------------------ thread_items ------------------------------

def test_thread_items_excludes_the_final_user_turn():
    """The last user turn IS the query — indexing it makes retrieval return the
    question as its own best-matching excerpt."""
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "and then?")])
    assert [i for i, _c, _m in logic.thread_items(c)] == ["m0", "m1"]
    # ...but it is available when explicitly asked for.
    assert len(logic.thread_items(c, exclude_last_user=False)) == 2  # "and then?" too short


def test_thread_items_excludes_intermediate_multipass_drafts():
    c = chat([turn("user", LONG),
              dict(turn("assistant", LONG2), intermediate=True),
              turn("assistant", LONG2 + " Final."),
              turn("user", "next")])
    assert [i for i, _c, _m in logic.thread_items(c)] == ["m0", "m2"]


def test_thread_items_ids_are_raw_indexes_so_they_survive_an_intermediate():
    """Ids index the RAW list, not the filtered one — otherwise dropping an
    intermediate would renumber every later turn and strand its chunks."""
    c = chat([turn("user", LONG),
              dict(turn("assistant", "skip"), intermediate=True),
              turn("assistant", LONG2),
              turn("user", "next")])
    assert [i for i, _c, _m in logic.thread_items(c)] == ["m0", "m2"]


def test_thread_items_strips_data_blocks_but_keeps_the_question():
    question = "What does the chart say about the channel marker beyond the point?"
    c = chat([turn("user", f"<Data>\n{LONG}\n</Data>\n\n{question}"),
              turn("assistant", LONG2),
              turn("user", "next")])
    body = dict((i, t) for i, t, _m in logic.thread_items(c))["m0"]
    assert body == question
    assert "harbour lantern" not in body      # that belongs to the attachment corpus


def test_a_turn_that_is_only_data_drops_out_of_the_thread_corpus():
    """Once the <Data> block is stripped, "summarise this" has nothing retrievable left
    — and the data itself is already indexed under the attachment corpus, so keeping the
    husk would only add a chunk that matches everything and says nothing."""
    c = chat([turn("user", f"<Data>\n{LONG}\n</Data>\n\nSummarise."),
              turn("assistant", LONG2),
              turn("user", "next")])
    assert [i for i, _c, _m in logic.thread_items(c)] == ["m1"]


def test_thread_items_skips_short_acknowledgements():
    """"ok" carries nothing retrievable but scores well on keyword search, where it
    would displace a real excerpt."""
    c = chat([turn("user", LONG), turn("assistant", "ok"), turn("user", "next")])
    assert [i for i, _c, _m in logic.thread_items(c)] == ["m0"]


def test_thread_items_labels_carry_the_turn_and_role():
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    labels = [m["label"] for _i, _c, m in logic.thread_items(c)]
    assert labels == ["Turn 1 (user)", "Turn 2 (assistant)"]


def test_an_edit_keeps_the_item_id_and_changes_only_the_content():
    """This is what lets an edited turn re-embed in place instead of orphaning chunks."""
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    before = dict((i, t) for i, t, _m in logic.thread_items(c))
    c["messages"][0]["content"] = LONG + " Rewritten."
    after = dict((i, t) for i, t, _m in logic.thread_items(c))
    assert set(before) == set(after)
    assert before["m0"] != after["m0"]


# ------------------------------ attachment_items ------------------------------

def test_attachment_items_uses_the_attachments_own_ids():
    c = chat(attachments=[{"id": "att1", "label": "Report.pdf", "type": "file",
                           "content": LONG}])
    (item_id, content, meta), = logic.attachment_items(c)
    assert item_id == "att1"
    assert content == LONG
    assert meta["label"] == "Report.pdf"


def test_attachment_items_includes_staged_data_blocks():
    c = chat([turn("user", f"<Data>\n{LONG}\n</Data>\n\nSummarise this.")])
    ids = [i for i, _c, _m in logic.attachment_items(c)]
    assert ids == ["data:0"]


def test_attachment_items_honours_isolation():
    """An isolated chat scopes <Data> to the current turn, exactly as
    collect_rag_inputs does — but pinned attachments belong to the chat, not a turn."""
    msgs = [turn("user", f"<Data>\n{LONG}\n</Data>\n\nfirst"),
            turn("assistant", LONG2),
            turn("user", f"<Data>\n{LONG2}\n</Data>\n\nsecond")]
    pinned = [{"id": "att1", "label": "Pinned", "type": "file", "content": LONG}]

    shared = chat(msgs, isolated=False, attachments=pinned)
    assert [i for i, _c, _m in logic.attachment_items(shared)] == ["att1", "data:0", "data:2"]

    lonely = chat(msgs, isolated=True, attachments=pinned)
    assert [i for i, _c, _m in logic.attachment_items(lonely)] == ["att1", "data:2"]


def test_attachment_items_skips_empty_and_id_less_entries():
    c = chat(attachments=[{"id": "a", "content": "   "},
                          {"label": "no id", "content": LONG},
                          {"id": "b", "content": LONG}])
    assert [i for i, _c, _m in logic.attachment_items(c)] == ["b"]


# ------------------------------ incremental sync ------------------------------

def test_sync_indexes_the_thread_and_the_attachments(vecstore):
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")],
             attachments=[{"id": "att1", "label": "R", "content": LONG2}])
    out = compile_mod.sync_chat(c, embedder, MODEL)

    assert out["scope"] == "both"
    assert out["indexed"] == 3          # two turns + one attachment
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) > 0
    assert rag.count_chunks(compile_mod.CHAT_ATTACH, c["id"]) > 0


def test_an_unchanged_chat_embeds_nothing_on_the_second_sync(vecstore):
    """The core efficiency claim: without this, thread scope re-pays for the whole
    conversation on every single send."""
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)
    assert embedder.calls > 0

    embedder.reset()
    out = compile_mod.sync_chat(c, embedder, MODEL)
    assert embedder.calls == 0
    assert out["indexed"] == 0
    assert out["skipped"] == 2


def test_editing_one_turn_re_embeds_only_that_turn(vecstore):
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)

    embedder.reset()
    c["messages"][0]["content"] = LONG + " An entirely new closing sentence appears."
    out = compile_mod.sync_chat(c, embedder, MODEL)
    assert out["indexed"] == 1
    assert out["skipped"] == 1


def test_a_regenerate_prunes_the_dropped_turn(vecstore):
    """Regenerate pops the assistant turn off the tail; its chunks must go with it."""
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)
    assert {i["item_id"] for i in rag.list_items(compile_mod.CHAT_THREAD, c["id"])} == {"m0", "m1"}

    c["messages"].pop(1)
    compile_mod.sync_chat(c, embedder, MODEL)
    assert {i["item_id"] for i in rag.list_items(compile_mod.CHAT_THREAD, c["id"])} == {"m0"}


def test_clearing_the_thread_prunes_everything(vecstore):
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)

    c["messages"] = []
    compile_mod.sync_chat(c, embedder, MODEL)
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) == 0


def test_changing_the_embedding_model_marks_the_chat_stale_and_re_embeds(vecstore):
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)
    assert compile_mod.chat_status(c, MODEL)["state"] == "compiled"
    assert compile_mod.chat_status(c, "other-model")["state"] == "stale"

    embedder.reset()
    out = compile_mod.sync_chat(c, embedder, "other-model")
    assert out["indexed"] == 2
    assert embedder.calls > 0


def test_switching_scope_away_keeps_the_other_corpus_rows(vecstore):
    """Those rows are never retrieved out of scope, and they are exactly what the
    store's cached-vector reuse needs if the user switches back. Pruning them would
    make flip-flopping the control re-embed the whole thread each time."""
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")],
             attachments=[{"id": "att1", "label": "R", "content": LONG2}])
    compile_mod.sync_chat(c, embedder, MODEL)
    thread_chunks = rag.count_chunks(compile_mod.CHAT_THREAD, c["id"])
    assert thread_chunks > 0

    c["rag_scope"] = "attachments"
    compile_mod.sync_chat(c, embedder, MODEL)
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) == thread_chunks

    # ...and switching back costs no embedding at all.
    embedder.reset()
    c["rag_scope"] = "both"
    compile_mod.sync_chat(c, embedder, MODEL)
    assert embedder.calls == 0


def test_attachments_scope_never_touches_the_thread_corpus(vecstore):
    embedder = CountingEmbedder()
    c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "next")],
             rag_scope="attachments",
             attachments=[{"id": "att1", "label": "R", "content": LONG2}])
    out = compile_mod.sync_chat(c, embedder, MODEL)
    assert out["sources"] == [compile_mod.CHAT_ATTACH]
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) == 0


def test_a_chat_with_no_id_is_never_indexed(vecstore):
    embedder = CountingEmbedder()
    out = compile_mod.sync_chat(chat([turn("user", LONG), turn("user", "x")], id=""),
                                embedder, MODEL)
    assert out["indexed"] == 0
    assert embedder.calls == 0


# ------------------------------ privacy ------------------------------

def test_private_and_synthetic_chats_are_transient():
    """The default LanceDB store keeps chunk text in PLAINTEXT on disk, so a private
    chat's messages must never reach it — and there is no reliable close hook to clean
    up after, so it is never written in the first place."""
    assert logic.rag_is_transient({"private": True, "id": "x"}) is True
    assert logic.rag_is_transient({"rag_ephemeral": True, "id": "x"}) is True
    assert logic.rag_is_transient({"id": ""}) is True
    assert logic.rag_is_transient({"id": "x"}) is False


# ------------------------------ injection seams ------------------------------

def test_resolve_attachments_returns_the_block_under_thread_scope():
    """Under thread scope the attachments are NOT in the corpus, so dropping the block
    would make the user's pinned material vanish — retrieved by nothing, sent by
    nobody."""
    c = chat(attachments=[{"id": "a", "label": "R", "type": "file", "content": LONG}])
    assert logic.resolve_attachments(c, rag_active=True, scope="thread") != ""
    assert logic.resolve_attachments(c, rag_active=True, scope="attachments") == ""
    assert logic.resolve_attachments(c, rag_active=True, scope="both") == ""


def test_resolve_attachments_without_a_scope_keeps_the_old_behaviour():
    c = chat(attachments=[{"id": "a", "label": "R", "type": "file", "content": LONG}])
    assert logic.resolve_attachments(c, rag_active=True) == ""
    assert logic.resolve_attachments(c, rag_active=False) != ""


def test_inject_rag_can_leave_the_data_block_alone():
    msgs = [turn("user", f"<Data>\n{LONG}\n</Data>\n\nSummarise.")]
    kept = logic.inject_rag(msgs, [], strip_data=False)
    assert "harbour lantern" in kept[0]["content"]
    stripped = logic.inject_rag(msgs, [], strip_data=True)
    assert "harbour lantern" not in stripped[0]["content"]
    assert "Summarise." in stripped[0]["content"]


def test_excerpt_labels_fall_back_to_the_chunks_own_meta():
    """A chat excerpt has no library item, so without this it would read as a raw
    'm3' in the prompt."""
    import json as _json
    assert logic._excerpt_label({"item_id": "m3", "meta": {"label": "Turn 4 (assistant)"}},
                                {}) == "Turn 4 (assistant)"
    # Backends hand meta back as a JSON string.
    assert logic._excerpt_label({"item_id": "m3",
                                 "meta": _json.dumps({"label": "Turn 4 (user)"})},
                                {}) == "Turn 4 (user)"
    # A live library label still wins over the one frozen into the chunk.
    assert logic._excerpt_label({"item_id": "lib1", "meta": {"label": "old"}},
                                {"lib1": "Renamed.pdf"}) == "Renamed.pdf"
    assert logic._excerpt_label({"item_id": "zz"}, {}) == "zz"


# ------------------------------ history window ------------------------------

def test_build_messages_caps_the_history_under_thread_scope():
    msgs = [turn("user", f"q{i}") if i % 2 == 0 else turn("assistant", f"a{i}")
            for i in range(10)]
    c = chat(msgs + [turn("user", "final question")])
    windowed = logic.build_messages(c, [], history_window=4)
    assert [m["content"] for m in windowed] == ["a7", "q8", "a9", "final question"]


def test_build_messages_without_a_window_sends_everything():
    msgs = [turn("user", f"q{i}") for i in range(6)]
    c = chat(msgs)
    assert len(logic.build_messages(c, [], history_window=None)) == 6


def test_the_history_window_never_drops_the_final_user_turn():
    c = chat([turn("user", "a"), turn("assistant", "b"), turn("user", "the question")])
    assert logic.build_messages(c, [], history_window=0)[-1]["content"] == "the question"


def test_resolve_rag_sets_a_window_only_under_thread_scope():
    c = chat([turn("user", LONG)], rag_enabled=True, rag_scope="attachments")
    assert logic.resolve_rag(c, {})["history_window"] is None
    for scope in ("thread", "both"):
        plan = logic.resolve_rag(chat([turn("user", LONG)], rag_enabled=True,
                                      rag_scope=scope), {"rag_thread_window": 5})
        assert plan["history_window"] == 5


def test_resolve_rag_reports_the_scope_flags():
    def plan(scope):
        return logic.resolve_rag(chat([turn("user", LONG)], rag_enabled=True,
                                      rag_scope=scope), {})
    assert (plan("attachments")["use_attachments"], plan("attachments")["use_thread"]) == (True, False)
    assert (plan("thread")["use_attachments"], plan("thread")["use_thread"]) == (False, True)
    assert (plan("both")["use_attachments"], plan("both")["use_thread"]) == (True, True)


# ------------------------------ rank fusion ------------------------------

def test_fuse_gives_every_corpus_a_seat():
    """The old code concatenated the per-corpus lists and sorted by raw score, which
    compares numbers from different spaces — a corpus whose scores merely happened to
    be larger took every slot."""
    big = [{"id": f"lib{i}", "item_id": f"lib{i}", "content": "x", "score": 900 - i}
           for i in range(6)]
    small = [{"id": f"chat{i}", "item_id": f"chat{i}", "content": "y", "score": 0.9 - i / 10}
             for i in range(6)]
    fused = rag.fuse([big, small], 6)
    assert any(r["item_id"].startswith("chat") for r in fused)
    assert any(r["item_id"].startswith("lib") for r in fused)
    # Each corpus's rank-1 result is weighted alike, so both lead. A raw sort would
    # have handed all six slots to `big` purely because its numbers are larger.
    assert {fused[0]["item_id"], fused[1]["item_id"]} == {"lib0", "chat0"}


def test_fuse_ignores_empty_lists():
    one = [{"id": "a", "item_id": "a", "content": "x", "score": 1.0}]
    assert len(rag.fuse([one, [], None], 5)) == 1


# ------------------------------ lifecycle routes ------------------------------

def test_deleting_a_chat_drops_its_indexed_corpus(client, monkeypatch):
    created = client.post("/api/chats", json={"title": "T"}).get_json()["chat"]
    embedder = CountingEmbedder()
    c = dict(created, rag_scope="both",
             messages=[turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, embedder, MODEL)
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) > 0

    client.delete(f"/api/chats/{c['id']}")
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) == 0


def test_the_forget_route_clears_the_corpus(client):
    created = client.post("/api/chats", json={"title": "T"}).get_json()["chat"]
    c = dict(created, rag_scope="both",
             messages=[turn("user", LONG), turn("assistant", LONG2), turn("user", "next")])
    compile_mod.sync_chat(c, CountingEmbedder(), MODEL)
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) > 0

    assert client.delete(f"/api/chats/{c['id']}/rag/forget").status_code == 200
    assert rag.count_chunks(compile_mod.CHAT_THREAD, c["id"]) == 0
    assert compile_mod.chat_status(c, MODEL)["state"] == "none"


def test_deleting_a_tab_clears_every_chat_in_it(client, tmp_path, monkeypatch):
    """delete_group used to return a bare bool, so the caller had no way to clean up
    what it had just deleted and those chats' vectors outlived them.

    Tabs are only created by importing chats, so that is how this builds one.
    """
    from app import native_dialog

    export = tmp_path / "Work.json"
    export.write_text('{"chats": [{"title": "C0"}, {"title": "C1"}]}', encoding="utf-8")
    monkeypatch.setattr(native_dialog, "pick_files", lambda **kw: [str(export)])

    body = client.post("/api/chats/import").get_json()
    group_id = body["group"]["id"]
    ids = [c["id"] for c in body["chats"] if c.get("group_id") == group_id]
    assert len(ids) == 2

    for chat_id in ids:
        c = chat([turn("user", LONG), turn("assistant", LONG2), turn("user", "x")],
                 id=chat_id)
        compile_mod.sync_chat(c, CountingEmbedder(), MODEL)
        assert rag.count_chunks(compile_mod.CHAT_THREAD, chat_id) > 0

    assert client.delete(f"/api/chat-groups/{group_id}").status_code == 200
    for chat_id in ids:
        assert rag.count_chunks(compile_mod.CHAT_THREAD, chat_id) == 0


def test_the_default_tab_still_cannot_be_deleted(client):
    assert client.delete("/api/chat-groups/default").status_code == 400


# ------------------------------ the sources frame ------------------------------
# What RAG picked has to reach the browser, or the Sources panel has nothing to draw.
# A PRIVATE chat drives this end to end without a store or a network: it retrieves in
# memory (rag_is_transient), and keyword mode means no embedding server is consulted.

def send_private_rag_chat(client, monkeypatch, question, data, **kw):
    adapter = use_adapter(monkeypatch, StubAdapter(["answered"]))
    chat = client.post("/api/chats", json={"model": "test-model"}).get_json()["chat"]
    chat.update({"private": True, "rag_enabled": True, "rag_scope": "attachments",
                 "rag_retrieval_mode": "keyword",   # no embed server needed
                 "rag_query_rewrite": False,        # no rewrite call to the stub
                 "messages": [turn("user", f"<Data>{data}</Data>{question}")]})
    chat.update(kw)
    r = client.post(f"/api/chats/{chat['id']}/send",
                    json={"chat": chat, "run_id": "test-run"})
    return sse_frames(r), adapter


def test_send_reports_the_chunks_it_retrieved(client, monkeypatch):
    frames, _ = send_private_rag_chat(client, monkeypatch, "what about the anchor?",
                                      LONG + "\n\n" + LONG2)
    seq = events(frames)
    assert seq.count("sources") == 1
    # Before the answer starts, so the panel is on screen while the text streams in.
    assert seq.index("sources") < seq.index("chunk")
    items = next(d for e, d in frames if e == "sources")["items"]
    assert items and all(s["content"] for s in items)
    # Nothing was indexed, so there is no chunk id to navigate back to.
    assert {s["kind"] for s in items} == {"inline"}


def test_no_sources_frame_when_rag_is_off(client, monkeypatch):
    frames, _ = send_private_rag_chat(client, monkeypatch, "hello", "some data",
                                      rag_enabled=False, rag_auto=False)
    assert "sources" not in events(frames)


def test_sources_are_reported_once_for_a_whole_multi_pass_turn(client, monkeypatch):
    """Every pass re-injects the same retrieval, so it is described once."""
    use_adapter(monkeypatch, StubAdapter(["draft", "final"]))
    chat = client.post("/api/chats", json={"model": "test-model"}).get_json()["chat"]
    chat.update({"private": True, "rag_enabled": True, "rag_scope": "attachments",
                 "rag_retrieval_mode": "keyword", "rag_query_rewrite": False,
                 "multi_pass": True, "passes": 1,
                 # The question has to share terms with the data: keyword mode is BM25,
                 # and an empty retrieval would take the no-RAG path instead.
                 "messages": [turn("user", f"<Data>{LONG}</Data>what about the anchor?")]})
    frames = sse_frames(client.post(f"/api/chats/{chat['id']}/send",
                                    json={"chat": chat, "run_id": "test-run"}))
    assert events(frames).count("sources") == 1
    assert events(frames).count("pass_start") == 2


def test_rag_scope_survives_a_settings_round_trip(client):
    r = client.post("/api/settings", json={"rag_thread_window": 12})
    assert r.status_code == 200
    assert r.get_json()["config"]["rag_thread_window"] == 12
    # Clamped, so a 0 can't window away the turn being answered.
    assert client.post("/api/settings", json={"rag_thread_window": 0}
                       ).get_json()["config"]["rag_thread_window"] == 1
