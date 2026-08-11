#!/usr/bin/env python3
"""Citations: turning retrieved chunks into something the UI can show and link.

``logic.describe_sources`` is the seam between retrieval and the Sources panel. What
matters here is identity, not prose: every row has to say which *kind* of corpus it came
from and carry enough of the owning library/item/turn for the client to navigate back to
it. Two details do most of the work and are easy to regress —

  * ``meta`` comes back from both vector-store backends as a JSON *string*, and rows
    written before ``meta`` existed have none at all;
  * ``source_type`` and ``chunk_index`` are not columns the retrieval path selects, so
    they are read off the chunk id that ``rag.upsert_items`` mints.

No store and no embedder: these are pure dict transforms.
"""

import json

from app import logic


LIB = {
    "id": "lib1",
    "name": "Handbook",
    "items": [
        {"id": "it1", "type": "file", "label": "Onboarding.pdf", "content": "hello"},
        {"id": "it2", "type": "url", "label": "Policy", "content": "world"},
    ],
}


def row(chunk_id, content="excerpt text", **kw):
    """A retrieval result as ``rag.retrieve`` returns one."""
    src_type, src_id, item_id, _ci = chunk_id.split(":")
    out = {"id": chunk_id, "source_id": src_id, "item_id": item_id,
           "content": content, "meta": None, "score": 0.5}
    out.update(kw)
    return out


# ------------------------------ shape per corpus ------------------------------

def test_library_row_carries_its_library():
    [s] = logic.describe_sources([row("library:lib1:it1:3")], [LIB])
    assert s["kind"] == "library"
    assert s["library_id"] == "lib1"
    assert s["library_name"] == "Handbook"
    assert s["item_id"] == "it1"
    assert s["chunk_index"] == 3
    assert s["label"] == "Onboarding.pdf"
    assert s["content"] == "excerpt text"
    assert s["score"] == 0.5


def test_attachment_row():
    r = row("chat_attach:c1:att9:0",
            meta=json.dumps({"label": "Transcript", "type": "youtube",
                             "kind": "attachment"}))
    [s] = logic.describe_sources([r], [LIB])
    assert s["kind"] == "attachment"
    assert s["item_id"] == "att9"
    assert s["label"] == "Transcript"
    assert "library_id" not in s


def test_thread_row_exposes_the_message_index():
    r = row("chat_thread:c1:m4:0",
            meta=json.dumps({"label": "Turn 5 (assistant)", "role": "assistant",
                             "index": 4, "kind": "thread"}))
    [s] = logic.describe_sources([r], [LIB])
    assert s["kind"] == "thread"
    assert s["message_index"] == 4
    assert s["label"] == "Turn 5 (assistant)"


def test_inline_row_has_no_navigable_identity():
    """Private/ephemeral chats retrieve in memory and mint no chunk id, so there is
    nothing to link to — the row still has to render rather than blow up."""
    r = {"content": "in-memory chunk", "item_id": "", "source_id": "",
         "meta": {"label": "Attached data"}, "score": None}
    [s] = logic.describe_sources([r], [LIB])
    assert s["kind"] == "inline"
    assert s["label"] == "Attached data"
    assert s["chunk_index"] is None
    assert s["score"] is None


# ------------------------------ label + meta handling ------------------------------

def test_live_library_label_beats_the_one_frozen_at_compile_time():
    """An item can be renamed after it was compiled; the panel must show today's name."""
    r = row("library:lib1:it2:0", meta=json.dumps({"label": "Old Name"}))
    [s] = logic.describe_sources([r], [LIB])
    assert s["label"] == "Policy"


def test_meta_may_be_a_dict_a_string_or_broken():
    ok = row("chat_attach:c1:a1:0", meta={"label": "As dict"})
    bad = row("chat_attach:c1:a2:0", meta="{not json")
    missing = row("chat_attach:c1:a3:0")
    labels = [s["label"] for s in logic.describe_sources([ok, bad, missing], [LIB])]
    assert labels == ["As dict", "a2", "a3"]   # unparseable/absent falls back to the id


def test_deleted_library_still_renders():
    """The chunk outlives the library it was compiled from until the next prune. Keep
    the row — the client decides it can't be linked."""
    [s] = logic.describe_sources([row("library:gone:it1:0")], [LIB])
    assert s["kind"] == "library"
    assert s["library_id"] == "gone"
    assert s["library_name"] == ""


# ------------------------------ id parsing ------------------------------

def test_source_type_and_index_come_off_the_chunk_id():
    """Neither is a column the retrieval path selects, so both are read back off the id
    that upsert_items built as source_type:source_id:item_id:index."""
    [s] = logic.describe_sources([row("chat_thread:c1:m7:12")], [])
    assert s["source_type"] == "chat_thread"
    assert s["chunk_index"] == 12


def test_explicit_columns_win_over_the_id():
    r = row("library:lib1:it1:3", source_type="library", chunk_index=9)
    [s] = logic.describe_sources([r], [LIB])
    assert s["chunk_index"] == 9


def test_item_ids_containing_colons_survive():
    """A staged <Data> block keys on ``data:<turn>``, which puts a colon in the middle
    of the chunk id — reading from the ends is what keeps that intact."""
    r = {"id": "chat_attach:c1:data:2:0", "source_id": "c1", "item_id": "data:2",
         "content": "x", "meta": None, "score": 0.1}
    [s] = logic.describe_sources([r], [])
    assert s["source_type"] == "chat_attach"
    assert s["chunk_index"] == 0
    assert s["item_id"] == "data:2"


# ------------------------------ degenerate input ------------------------------

def test_nothing_retrieved():
    assert logic.describe_sources([], [LIB]) == []
    assert logic.describe_sources(None, None) == []
