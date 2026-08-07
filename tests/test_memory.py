#!/usr/bin/env python3
"""Tests for the Memory tab's user memory cores (app/memory.py, the core half of
app/store.py, and the /api/memory routes).

These pin down the defects that silently destroyed or hid a user's memories:

* ``apply_operations`` handled a merge whose ``ids`` repeated the same entry by writing
  the merged text into it and then removing it — the memory vanished and the pass
  reported "1 merged";
* the same merge read its inherited importance with ``op.get("importance", <max>)``,
  which hands back an explicit ``null`` instead of the default, so a model emitting one
  collapsed a pair of importance-10 memories to 5;
* extraction passes resolved a core once and mutated it minutes later with no lock, while
  ``delete_memory_entry`` REPLACED the entries list — so an in-flight pass appended to a
  detached list and lost everything it had just learned (the same shape as the library
  write-back bug in test_libraries.py);
* deleting a core left its id on every chat that used it, so those chats read
  ``memory_enabled`` with a core that no longer resolves: the toggle says memory is on
  while nothing is ever injected;
* a consolidation pass that compacted nothing re-ran on every single extraction, forever,
  because the trigger only looked at the entry count and never at whether the last
  attempt had achieved anything.

Plus the guarantees that had no coverage at all: injection selection and placement,
normalisation of hand-edited files, export/import round-tripping, and the route contracts.

Everything runs against a throwaway tree; no real profile is touched.
"""

import json
import re
import threading

import pytest

from app import evals, logic, memory, store as store_mod

from conftest import StubAdapter, all_of, events, first, sse_frames, use_adapter


# --------------------------- harness ---------------------------
# `store` and `client` come from tests/conftest.py.

def _core_with(*entries):
    """A core holding the given entries, each a dict of overrides for ``new_entry``."""
    mc = memory.new_core("Test core")
    for spec in entries:
        e = memory.new_entry(spec.get("text", "something"), spec.get("category", "facts"),
                             spec.get("importance", 5), origin=spec.get("origin", "ai"),
                             pinned=spec.get("pinned", False))
        if "id" in spec:
            e["id"] = spec["id"]
        if "created" in spec:
            e["created"] = spec["created"]
        mc["entries"].append(e)
    return mc


def _ids(mc):
    return [e["id"] for e in mc["entries"]]


def _texts(mc):
    return [e["text"] for e in mc["entries"]]


# =====================================================================
# apply_operations — the data-loss bugs
# =====================================================================

def test_merge_with_a_repeated_id_does_not_delete_the_merged_entry():
    """The bug: ids ["a","a"] built targets [A, A], so the removal loop deleted the very
    entry the merged text had just been written into. One memory in, none out."""
    mc = _core_with({"id": "a", "text": "likes tea"})
    summary = memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "a"], "text": "likes tea", "importance": 7},
    ])
    # A single distinct target is not a merge — nothing happens, and nothing is lost.
    assert _ids(mc) == ["a"]
    assert summary["merged"] == 0


def test_merge_with_a_repeated_id_among_real_ones_keeps_both_survivors():
    mc = _core_with({"id": "a", "text": "one"}, {"id": "b", "text": "two"},
                    {"id": "c", "text": "three"})
    summary = memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "a", "b"], "text": "one and two"},
    ])
    assert summary["merged"] == 1
    assert _texts(mc) == ["one and two", "three"]     # 'a' kept, 'b' absorbed
    assert _ids(mc) == ["a", "c"]


def test_merge_inherits_importance_when_the_model_sends_an_explicit_null():
    """``op.get("importance", <max>)`` returns the stored None, not the default — so a
    merge of two 10s used to land on _clamp_importance's own fallback of 5."""
    mc = _core_with({"id": "a", "text": "one", "importance": 10},
                    {"id": "b", "text": "two", "importance": 10})
    memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "b"], "text": "both", "importance": None},
    ])
    assert mc["entries"][0]["importance"] == 10


def test_merge_inherits_importance_when_the_key_is_absent():
    mc = _core_with({"id": "a", "text": "one", "importance": 3},
                    {"id": "b", "text": "two", "importance": 9})
    memory.apply_operations(mc, [{"op": "merge", "ids": ["a", "b"], "text": "both"}])
    assert mc["entries"][0]["importance"] == 9


def test_merge_honours_an_explicit_importance():
    mc = _core_with({"id": "a", "text": "one", "importance": 10},
                    {"id": "b", "text": "two", "importance": 10})
    memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "b"], "text": "both", "importance": 2},
    ])
    assert mc["entries"][0]["importance"] == 2


def test_merge_clamps_a_nonsense_importance_to_the_inherited_max():
    mc = _core_with({"id": "a", "text": "one", "importance": 8},
                    {"id": "b", "text": "two", "importance": 4})
    memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "b"], "text": "both", "importance": "high"},
    ])
    assert mc["entries"][0]["importance"] == 8


# --------------------------- protected entries ---------------------------

@pytest.mark.parametrize("guard", [{"origin": "user"}, {"pinned": True}])
def test_delete_never_removes_a_protected_entry(guard):
    mc = _core_with({"id": "a", "text": "mine", **guard})
    summary = memory.apply_operations(mc, [{"op": "delete", "id": "a"}])
    assert _ids(mc) == ["a"]
    assert summary["deleted"] == 0


@pytest.mark.parametrize("guard", [{"origin": "user"}, {"pinned": True}])
def test_merge_leaves_a_protected_entry_standing_and_merges_the_rest(guard):
    mc = _core_with({"id": "a", "text": "mine", **guard},
                    {"id": "b", "text": "two"}, {"id": "c", "text": "three"})
    summary = memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "b", "c"], "text": "two and three"},
    ])
    assert summary["merged"] == 1
    assert _texts(mc) == ["mine", "two and three"]


def test_merge_of_only_protected_entries_is_a_no_op():
    mc = _core_with({"id": "a", "text": "mine", "origin": "user"},
                    {"id": "b", "text": "also mine", "pinned": True})
    summary = memory.apply_operations(mc, [
        {"op": "merge", "ids": ["a", "b"], "text": "both"},
    ])
    assert _texts(mc) == ["mine", "also mine"]
    assert summary["merged"] == 0


def test_update_may_still_edit_a_protected_entry():
    """Protection is against *removal*. A refinement of the wording is still welcome."""
    mc = _core_with({"id": "a", "text": "vague", "origin": "user"})
    memory.apply_operations(mc, [{"op": "update", "id": "a", "text": "precise"}])
    assert _texts(mc) == ["precise"]


# --------------------------- add / update / malformed ---------------------------

def test_add_records_origin_and_source_chat():
    mc = memory.new_core()
    memory.apply_operations(mc, [
        {"op": "add", "category": "goals", "text": "Shipping a memory tab.", "importance": 8},
    ], source_chat={"id": "c1", "title": "Planning"})
    e = mc["entries"][0]
    assert (e["origin"], e["category"], e["importance"]) == ("ai", "goals", 8)
    assert (e["source_chat_id"], e["source_chat_title"]) == ("c1", "Planning")


def test_update_changes_text_category_and_importance():
    mc = _core_with({"id": "a", "text": "old", "category": "facts", "importance": 3})
    summary = memory.apply_operations(mc, [
        {"op": "update", "id": "a", "text": "new", "category": "style", "importance": 9},
    ])
    e = mc["entries"][0]
    assert (e["text"], e["category"], e["importance"]) == ("new", "style", 9)
    assert summary["updated"] == 1


def test_delete_removes_an_ai_entry():
    mc = _core_with({"id": "a", "text": "wrong"}, {"id": "b", "text": "right"})
    summary = memory.apply_operations(mc, [{"op": "delete", "id": "a"}])
    assert _ids(mc) == ["b"] and summary["deleted"] == 1


@pytest.mark.parametrize("ops", [
    None, "not a list", 42, {},
    [None, 7, "x"],                                    # non-dict operations
    [{}],                                              # no op key
    [{"op": "add", "text": "   "}],                    # blank text
    [{"op": "add"}],                                   # no text at all
    [{"op": "update", "id": "nope", "text": "x"}],     # unknown id
    [{"op": "update", "text": "x"}],                   # no id
    [{"op": "update", "id": "a"}],                     # no text
    [{"op": "delete", "id": "nope"}],
    [{"op": "delete"}],                                # no id
    [{"op": "merge", "ids": ["a"], "text": "x"}],      # only one target
    [{"op": "merge", "ids": [], "text": "x"}],
    [{"op": "merge", "text": "x"}],                    # no ids
    [{"op": "nonsense", "text": "x"}],
])
def test_malformed_operations_are_skipped_without_raising(ops):
    mc = _core_with({"id": "a", "text": "keep me"})
    summary = memory.apply_operations(mc, ops)
    assert _texts(mc) == ["keep me"]
    assert summary == {"added": 0, "updated": 0, "merged": 0, "deleted": 0}


def test_a_mixed_batch_applies_every_valid_operation():
    mc = _core_with({"id": "a", "text": "one"}, {"id": "b", "text": "two"},
                    {"id": "c", "text": "three"}, {"id": "d", "text": "four"})
    summary = memory.apply_operations(mc, [
        {"op": "add", "text": "five", "category": "interests"},
        {"op": "update", "id": "a", "text": "ONE"},
        {"op": "merge", "ids": ["b", "c"], "text": "two and three"},
        {"op": "delete", "id": "d"},
        {"op": "garbage"},
    ])
    assert summary == {"added": 1, "updated": 1, "merged": 1, "deleted": 1}
    assert _texts(mc) == ["ONE", "two and three", "five"]


def test_a_no_op_batch_leaves_updated_untouched():
    mc = _core_with({"id": "a", "text": "one"})
    before = mc["updated"]
    memory.apply_operations(mc, [])
    assert mc["updated"] == before


# =====================================================================
# selection, rendering and injection
# =====================================================================

def test_selected_entries_caps_at_inject_limit_by_importance():
    mc = _core_with(*[{"text": f"e{i}", "importance": i} for i in range(1, 11)])
    mc["inject_limit"] = 3
    assert [e["text"] for e in memory.selected_entries(mc)] == ["e10", "e9", "e8"]


def test_pinned_entries_are_exempt_from_the_cap():
    mc = _core_with({"text": "pinned", "importance": 1, "pinned": True},
                    {"text": "loud", "importance": 10},
                    {"text": "quiet", "importance": 2})
    mc["inject_limit"] = 1
    assert [e["text"] for e in memory.selected_entries(mc)] == ["pinned", "loud"]


def test_selection_breaks_importance_ties_by_age():
    mc = _core_with({"text": "older", "importance": 5, "created": "2024-01-01"},
                    {"text": "newer", "importance": 5, "created": "2025-01-01"})
    mc["inject_limit"] = 1
    assert [e["text"] for e in memory.selected_entries(mc)] == ["older"]


def test_render_core_is_empty_for_an_empty_core():
    """The caller's signal to skip injection entirely."""
    assert memory.render_core(memory.new_core()) == ""


def test_render_core_groups_by_category_and_escapes():
    mc = _core_with({"text": "likes <b>bold</b> & tea", "category": "preferences"},
                    {"text": "works in Ohio", "category": "facts"})
    mc["name"] = 'The "main" core'
    block = memory.render_core(mc)
    assert "<preferences>" in block and "<facts>" in block
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; tea" in block
    assert "<b>" not in block
    assert 'core=\'The "main" core\'' in block       # quoteattr switches quote style
    # Category order follows CATEGORIES, not insertion order.
    assert block.index("<preferences>") < block.index("<facts>")


def test_render_core_only_renders_selected_entries():
    mc = _core_with({"text": "sent", "importance": 9}, {"text": "shelved", "importance": 1})
    mc["inject_limit"] = 1
    block = memory.render_core(mc)
    assert "sent" in block and "shelved" not in block


def test_render_core_does_not_reorder_the_stored_entries():
    mc = _core_with({"text": "quiet", "importance": 1, "category": "facts"},
                    {"text": "loud", "importance": 9, "category": "facts"})
    memory.render_core(mc)
    assert _texts(mc) == ["quiet", "loud"]


# --------------------------- resolve_memory ---------------------------

def test_resolve_memory_returns_the_selected_core():
    mc = _core_with({"text": "something"})
    chat = {"memory_enabled": True, "memory_core_id": mc["id"]}
    assert memory.resolve_memory(chat, [mc]) is mc


@pytest.mark.parametrize("chat", [
    {},                                                     # nothing set
    {"memory_enabled": False, "memory_core_id": "known"},   # toggle off
    {"memory_enabled": True, "memory_core_id": ""},         # no core picked
    {"memory_enabled": True, "memory_core_id": "gone"},     # core deleted
])
def test_resolve_memory_returns_none_when_memory_is_not_usable(chat):
    mc = _core_with({"text": "something"})
    mc["id"] = "known"
    if chat.get("memory_core_id") == "known":
        assert memory.resolve_memory(chat, [mc]) is None    # only the toggle-off case
    else:
        assert memory.resolve_memory(chat, [mc]) is None


def test_resolve_memory_returns_none_for_a_core_with_nothing_to_inject():
    mc = memory.new_core()
    chat = {"memory_enabled": True, "memory_core_id": mc["id"]}
    assert memory.resolve_memory(chat, [mc]) is None


# --------------------------- inject_memory ---------------------------

def test_inject_memory_places_the_block_before_the_last_user_turn():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}]
    out = logic.inject_memory(msgs, "<user_memory/>")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "system", "user"]
    assert out[3]["content"].startswith(memory.MEMORY_PREAMBLE)
    assert out[4]["content"] == "second"


def test_inject_memory_appends_when_there_is_no_user_turn():
    out = logic.inject_memory([{"role": "system", "content": "sys"}], "<user_memory/>")
    assert out[-1]["role"] == "system" and "<user_memory/>" in out[-1]["content"]


def test_inject_memory_is_a_no_op_for_an_empty_block():
    msgs = [{"role": "user", "content": "hi"}]
    assert logic.inject_memory(msgs, "") == msgs


def test_inject_memory_does_not_mutate_the_caller_list():
    msgs = [{"role": "user", "content": "hi"}]
    logic.inject_memory(msgs, "<user_memory/>")
    assert len(msgs) == 1


# =====================================================================
# normalisation — a hand-edited or third-party file must not break the tab
# =====================================================================

def test_normalize_core_fills_in_a_bare_dict():
    mc = memory.normalize_core({})
    assert mc["id"] and mc["name"] == "Untitled core"
    assert mc["extract_every"] == memory.DEFAULT_EXTRACT_EVERY
    assert mc["inject_limit"] == memory.DEFAULT_INJECT_LIMIT
    assert mc["entries"] == [] and mc["built_chat_ids"] == []
    assert mc["last_consolidated_at"] == -1


@pytest.mark.parametrize("given,expected", [(0, 1), (-5, 1), (9999, 100), ("x", 6), (None, 6)])
def test_normalize_core_clamps_extract_every(given, expected):
    assert memory.normalize_core({"extract_every": given})["extract_every"] == expected


@pytest.mark.parametrize("given,expected", [(0, 1), (-1, 1), (10000, 500), ("x", 40)])
def test_normalize_core_clamps_inject_limit(given, expected):
    assert memory.normalize_core({"inject_limit": given})["inject_limit"] == expected


@pytest.mark.parametrize("given,expected", [(0, 1), (99, 10), (-3, 1), ("x", 5), (None, 5)])
def test_normalize_core_clamps_entry_importance(given, expected):
    mc = memory.normalize_core({"entries": [{"text": "t", "importance": given}]})
    assert mc["entries"][0]["importance"] == expected


def test_normalize_core_drops_junk_entries():
    mc = memory.normalize_core({"entries": [
        {"text": "keep"}, {"text": "   "}, {}, "not a dict", None, 42,
    ]})
    assert _texts(mc) == ["keep"]


def test_normalize_core_coerces_an_unknown_category_and_origin():
    mc = memory.normalize_core({"entries": [
        {"text": "t", "category": "wharrgarbl", "origin": "somewhere"},
    ]})
    assert mc["entries"][0]["category"] == "other"
    assert mc["entries"][0]["origin"] == "ai"


def test_normalize_core_accepts_a_valid_category_case_insensitively():
    mc = memory.normalize_core({"entries": [{"text": "t", "category": "  GOALS "}]})
    assert mc["entries"][0]["category"] == "goals"


def test_normalize_core_gives_every_entry_an_id():
    mc = memory.normalize_core({"entries": [{"text": "a"}, {"text": "b"}]})
    assert all(e["id"] for e in mc["entries"])


def test_normalize_core_does_not_mutate_its_argument():
    src = {"entries": [{"text": "a"}]}
    memory.normalize_core(src)
    assert "id" not in src and src["entries"][0] == {"text": "a"}


def test_normalize_core_drops_the_client_only_injected_flag():
    """``decorate_for_client`` adds it for the tab; it must never reach disk."""
    decorated = memory.decorate_for_client(_core_with({"text": "t"}))
    assert "injected" in decorated["entries"][0]
    assert "injected" not in memory.normalize_core(decorated)["entries"][0]


# =====================================================================
# consolidation backoff
# =====================================================================

def test_needs_consolidation_is_false_below_the_threshold():
    mc = _core_with(*[{"text": f"e{i}"} for i in range(5)])
    mc["inject_limit"] = 10
    assert not memory.needs_consolidation(mc)


def test_needs_consolidation_is_true_once_well_past_the_inject_limit():
    mc = _core_with(*[{"text": f"e{i}"} for i in range(16)])
    mc["inject_limit"] = 10                      # 16 > 10 * 1.5
    assert memory.needs_consolidation(mc)


def test_a_consolidation_that_compacts_nothing_does_not_run_again():
    """The retry storm: an over-limit core whose refine pass returns no operations used
    to pay for a second model call on every extraction, forever."""
    mc = _core_with(*[{"text": f"e{i}"} for i in range(16)])
    mc["inject_limit"] = 10
    assert memory.needs_consolidation(mc)
    memory.mark_consolidated(mc)                 # the pass ran and changed nothing
    assert not memory.needs_consolidation(mc)


def test_consolidation_runs_again_once_the_core_has_grown():
    mc = _core_with(*[{"text": f"e{i}"} for i in range(16)])
    mc["inject_limit"] = 10
    memory.mark_consolidated(mc)
    memory.apply_operations(mc, [{"op": "add", "text": "something new"}])
    assert memory.needs_consolidation(mc)


def test_consolidation_that_compacts_below_the_line_resets_cleanly():
    mc = _core_with(*[{"text": f"e{i}"} for i in range(16)])
    mc["inject_limit"] = 10
    memory.apply_operations(mc, [{"op": "delete", "id": e["id"]}
                                 for e in list(mc["entries"])[:8]])
    memory.mark_consolidated(mc)
    assert mc["last_consolidated_at"] == 8
    assert not memory.needs_consolidation(mc)


# =====================================================================
# decorate_for_client — the tab reads the server's answer, not its own
# =====================================================================

def test_decorate_for_client_flags_exactly_what_is_injected():
    mc = _core_with({"text": "sent", "importance": 9}, {"text": "shelved", "importance": 1})
    mc["inject_limit"] = 1
    out = memory.decorate_for_client(mc)
    assert [(e["text"], e["injected"]) for e in out["entries"]] \
        == [("sent", True), ("shelved", False)]


def test_decorate_for_client_counts_the_preamble_in_its_estimate():
    mc = _core_with({"text": "x" * 400})
    out = memory.decorate_for_client(mc)
    assert out["est_tokens"] == memory.estimate_tokens(mc)
    assert out["est_tokens"] > len(memory.MEMORY_PREAMBLE) // 4


def test_decorate_for_client_leaves_the_stored_core_untouched():
    mc = _core_with({"text": "t"})
    memory.decorate_for_client(mc)
    assert "injected" not in mc["entries"][0] and "est_tokens" not in mc


def test_estimate_tokens_is_zero_for_an_empty_core():
    assert memory.estimate_tokens(memory.new_core()) == 0


# =====================================================================
# transcript
# =====================================================================

def test_transcript_skips_multipass_intermediates_and_system_turns():
    chat = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "draft", "intermediate": True},
        {"role": "assistant", "content": "final"},
    ]}
    text = memory.transcript_text(chat)
    assert "sys" not in text and "draft" not in text
    assert "User: q" in text and "Assistant: final" in text


def test_transcript_keeps_the_tail_when_truncating():
    chat = {"messages": [{"role": "user", "content": "old " * 4000},
                         {"role": "assistant", "content": "the newest thing"}]}
    text = memory.transcript_text(chat, max_chars=200)
    assert len(text) == 200 and text.endswith("the newest thing")


def test_transcript_of_an_empty_chat_is_empty():
    assert memory.transcript_text({"messages": []}) == ""
    assert memory.transcript_text({}) == ""


# =====================================================================
# summary_text
# =====================================================================

def test_summary_text_lists_only_what_happened():
    assert memory.summary_text({"added": 2, "updated": 1, "merged": 0, "deleted": 0}) \
        == "2 added, 1 refined"


def test_summary_text_is_empty_when_nothing_changed():
    assert memory.summary_text({"added": 0, "updated": 0, "merged": 0, "deleted": 0}) == ""


# =====================================================================
# Store — concurrency (the point of the locking rework)
# =====================================================================

def test_delete_memory_entry_keeps_the_list_object_identity(store):
    """The detachment at the root of the lost-update bug: rebinding mc["entries"] left an
    in-flight pass appending to a list nothing pointed at any more."""
    mc = store.add_memory_core("c")
    store.upsert_memory_entry(mc["id"], {"text": "one"})
    store.upsert_memory_entry(mc["id"], {"text": "two"})
    held = store.get_memory_core(mc["id"])["entries"]     # what a pass would be holding
    victim = held[0]["id"]

    assert store.delete_memory_entry(mc["id"], victim)

    assert store.get_memory_core(mc["id"])["entries"] is held
    assert [e["text"] for e in held] == ["two"]


def test_an_extraction_pass_survives_a_concurrent_entry_delete(store):
    """The exact shape of the bug: a pass resolves the core, an entry is deleted while
    the model is thinking, and the pass then applies its operations. The new memory must
    land in the core the store actually saves."""
    mc = store.add_memory_core("c")
    store.upsert_memory_entry(mc["id"], {"text": "doomed"})
    core_id = mc["id"]
    doomed = store.get_memory_core(core_id)["entries"][0]["id"]

    # The pass grabs the core, as _memory_pass does to build its prompt...
    store.get_memory_core(core_id)
    # ...the browser deletes an entry while the model is thinking...
    store.delete_memory_entry(core_id, doomed)
    # ...and the pass applies what it learned, re-resolving under the lock.
    store.mutate_memory_core(core_id, lambda c: memory.apply_operations(
        c, [{"op": "add", "text": "learned something"}]))

    assert _texts(store.get_memory_core(core_id)) == ["learned something"]


def test_mutate_memory_core_reports_a_core_deleted_midflight(store):
    core_id = store.add_memory_core("c")["id"]
    store.delete_memory_core(core_id)
    mc, result = store.mutate_memory_core(core_id, lambda c: "ran")
    assert mc is None and result is None


def test_concurrent_passes_and_edits_lose_nothing(store):
    """Twenty threads adding memories through the locked path while another twenty add
    entries by hand. Every single one must be present at the end."""
    core_id = store.add_memory_core("c")["id"]

    def pass_thread(i):
        store.mutate_memory_core(core_id, lambda c: memory.apply_operations(
            c, [{"op": "add", "text": f"ai-{i}"}]))

    def edit_thread(i):
        store.upsert_memory_entry(core_id, {"text": f"user-{i}"})

    threads = ([threading.Thread(target=pass_thread, args=(i,)) for i in range(20)]
               + [threading.Thread(target=edit_thread, args=(i,)) for i in range(20)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    texts = set(_texts(store.get_memory_core(core_id)))
    assert texts == {f"ai-{i}" for i in range(20)} | {f"user-{i}" for i in range(20)}


# =====================================================================
# Store — CRUD
# =====================================================================

def test_add_memory_core_persists_and_defaults_the_name(store):
    mc = store.add_memory_core("")
    assert mc["name"] == "New memory core"
    assert store_mod.core.MEMORY_CORES_FILE.exists()


def test_update_memory_core_patches_only_the_scalars(store):
    mc = store.add_memory_core("c")
    store.upsert_memory_entry(mc["id"], {"text": "keep me"})
    store.update_memory_core(mc["id"], {
        "name": "Renamed", "auto_extract": False, "extract_every": 3, "inject_limit": 12,
        "entries": [], "id": "hacked",
    })
    saved = store.get_memory_core(mc["id"])
    assert saved is not None                      # id was not patchable
    assert (saved["name"], saved["auto_extract"]) == ("Renamed", False)
    assert (saved["extract_every"], saved["inject_limit"]) == (3, 12)
    assert _texts(saved) == ["keep me"]            # entries untouched


def test_update_memory_core_clamps_out_of_range_values(store):
    mc = store.add_memory_core("c")
    saved = store.update_memory_core(mc["id"], {"extract_every": 0, "inject_limit": 99999})
    assert (saved["extract_every"], saved["inject_limit"]) == (1, 500)


def test_update_memory_core_returns_none_for_an_unknown_core(store):
    assert store.update_memory_core("nope", {"name": "x"}) is None


def test_deleting_a_core_clears_it_off_every_chat_that_used_it(store):
    """The invisible half of the bug: only the chat that happened to be open was
    repaired, so every other chat kept reading 'memory on' with nothing to inject."""
    mc = store.add_memory_core("c")
    chats = [store.add_chat({"id": f"chat{i}", "title": f"Chat {i}",
                             "memory_enabled": True, "memory_core_id": mc["id"],
                             "memory_turns_since": 4, "messages": []})
             for i in range(3)]
    other = store.add_chat({"id": "other", "title": "Other", "memory_enabled": True,
                            "memory_core_id": "someone-else", "messages": []})

    assert store.delete_memory_core(mc["id"])

    for ch in chats:
        saved = store.get_chat(ch["id"])
        assert saved["memory_core_id"] == ""
        assert saved["memory_enabled"] is False
        assert saved["memory_turns_since"] == 0
    # A chat pointing at a different core is left alone.
    assert store.get_chat(other["id"])["memory_core_id"] == "someone-else"


def test_delete_memory_core_returns_false_for_an_unknown_core(store):
    assert store.delete_memory_core("nope") is False


# --------------------------- entries ---------------------------

def test_upsert_creates_a_user_owned_entry(store):
    mc = store.add_memory_core("c")
    entry, reason = store.upsert_memory_entry(mc["id"], {
        "text": "  Prefers short answers.  ", "category": "style", "importance": 9,
        "pinned": True,
    })
    assert reason == ""
    assert entry["text"] == "Prefers short answers."
    assert entry["origin"] == "user"               # protected from automated deletion
    assert (entry["category"], entry["importance"], entry["pinned"]) == ("style", 9, True)


def test_upsert_edits_an_existing_entry_in_place(store):
    mc = store.add_memory_core("c")
    created, _ = store.upsert_memory_entry(mc["id"], {"text": "old"})
    edited, reason = store.upsert_memory_entry(mc["id"], {
        "id": created["id"], "text": "new", "importance": 2,
    })
    assert reason == "" and edited["id"] == created["id"]
    assert len(store.get_memory_core(mc["id"])["entries"]) == 1
    assert (edited["text"], edited["importance"]) == ("new", 2)


def test_upsert_will_not_erase_text_with_a_blank_patch(store):
    mc = store.add_memory_core("c")
    created, _ = store.upsert_memory_entry(mc["id"], {"text": "keep"})
    edited, _ = store.upsert_memory_entry(mc["id"], {"id": created["id"], "text": "  "})
    assert edited["text"] == "keep"


def test_upsert_reports_an_entry_deleted_somewhere_else(store):
    """An edit naming an id that is gone used to silently reappear as a new user entry —
    what a second tab deleting it out from under this one looked like."""
    mc = store.add_memory_core("c")
    entry, reason = store.upsert_memory_entry(mc["id"], {"id": "vanished", "text": "x"})
    assert (entry, reason) == (None, "gone")
    assert store.get_memory_core(mc["id"])["entries"] == []


def test_upsert_reports_blank_text_and_a_missing_core_distinctly(store):
    mc = store.add_memory_core("c")
    assert store.upsert_memory_entry(mc["id"], {"text": "   "}) == (None, "text")
    assert store.upsert_memory_entry("nope", {"text": "x"}) == (None, "core")


def test_delete_memory_entry_reports_a_miss(store):
    mc = store.add_memory_core("c")
    assert store.delete_memory_entry(mc["id"], "nope") is False
    assert store.delete_memory_entry("nope", "nope") is False


# --------------------------- export / import ---------------------------

def test_export_import_round_trips_with_fresh_ids(store):
    mc = store.add_memory_core("Travel")
    store.upsert_memory_entry(mc["id"], {"text": "Likes trains.", "category": "interests"})
    store.upsert_memory_entry(mc["id"], {"text": "Hates flying.", "importance": 8})
    envelope = store.export_memory_cores([mc["id"]])
    assert envelope["kind"] == "memory_cores" and len(envelope["cores"]) == 1

    imported, count = store.import_memory_cores(envelope)
    assert count == 1
    new = imported[0]
    assert new["id"] != mc["id"]
    assert new["name"] == "Travel (imported)"                  # clashing name suffixed
    assert sorted(_texts(new)) == ["Hates flying.", "Likes trains."]
    # Every entry id is reassigned, so an import can never overwrite an existing memory.
    assert not set(_ids(new)) & set(_ids(store.get_memory_core(mc["id"])))


def test_export_all_takes_every_core(store):
    store.add_memory_core("a")
    store.add_memory_core("b")
    assert len(store.export_memory_cores()["cores"]) == 2


def test_import_rejects_a_file_that_is_not_a_core_export(store):
    with pytest.raises(ValueError):
        store.import_memory_cores({"kind": "chats", "cores": "nope"})
    with pytest.raises(ValueError):
        store.import_memory_cores({})


def test_import_normalizes_hostile_input(store):
    imported, count = store.import_memory_cores({"cores": [
        {"name": "  ", "inject_limit": 99999, "entries": [
            {"text": "fine", "importance": 300, "category": "nope"},
            {"text": ""},
            "junk",
        ]},
        "not a core",
    ]})
    assert count == 1
    mc = imported[0]
    assert mc["name"] == "Untitled core" and mc["inject_limit"] == 500
    assert _texts(mc) == ["fine"]
    assert mc["entries"][0]["importance"] == 10 and mc["entries"][0]["category"] == "other"


# --------------------------- incognito ---------------------------

def test_incognito_never_writes_cores_to_disk(store):
    store.add_memory_core("real")                       # create the file first
    before = store_mod.core.MEMORY_CORES_FILE.read_bytes()
    store.incognito = True
    store.add_memory_core("ephemeral")
    store.upsert_memory_entry(store.memory_cores[-1]["id"], {"text": "secret"})
    assert store_mod.core.MEMORY_CORES_FILE.read_bytes() == before
    # ...but it is live in memory for the session, which is why the tab warns instead
    # of pretending the edit failed.
    assert [c["name"] for c in store.memory_cores] == ["real", "ephemeral"]


# =====================================================================
# HTTP routes
# =====================================================================

def _make_core(client, name="Test"):
    r = client.post("/api/memory/cores", json={"name": name})
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["core"]


def test_state_ships_cores_and_categories(client):
    st = client.get("/api/state").get_json()
    assert st["memory_cores"] == []
    assert st["memory_categories"] == memory.CATEGORIES


def test_get_cores_is_the_refresh_path(client):
    _make_core(client, "Mine")
    body = client.get("/api/memory/cores").get_json()
    assert [c["name"] for c in body["cores"]] == ["Mine"]
    assert body["categories"] == memory.CATEGORIES


def test_create_returns_the_core_and_the_whole_list(client):
    body = client.post("/api/memory/cores", json={"name": "Mine"}).get_json()
    assert body["core"]["name"] == "Mine"
    assert [c["id"] for c in body["cores"]] == [body["core"]["id"]]


def test_patch_updates_and_404s_on_an_unknown_core(client):
    mc = _make_core(client)
    body = client.patch(f"/api/memory/cores/{mc['id']}",
                        json={"name": "Renamed", "inject_limit": 7}).get_json()
    assert body["core"]["name"] == "Renamed" and body["core"]["inject_limit"] == 7
    assert client.patch("/api/memory/cores/nope", json={"name": "x"}).status_code == 404


def test_delete_returns_the_remaining_cores_and_the_chats(client):
    mc = _make_core(client)
    r = client.delete(f"/api/memory/cores/{mc['id']}")
    body = r.get_json()
    assert r.status_code == 200 and body["cores"] == []
    assert "chats" in body                       # so the browser sees the scrubbed chats
    assert client.delete(f"/api/memory/cores/{mc['id']}").status_code == 404


def test_entry_upsert_returns_the_decorated_core(client):
    mc = _make_core(client)
    body = client.post(f"/api/memory/cores/{mc['id']}/entries",
                       json={"entry": {"text": "Likes tea.", "category": "preferences"}}).get_json()
    assert body["entry"]["origin"] == "user"
    entry = body["core"]["entries"][0]
    assert entry["injected"] is True
    assert body["core"]["est_tokens"] > 0


def test_entry_upsert_distinguishes_its_failures(client):
    mc = _make_core(client)
    assert client.post("/api/memory/cores/nope/entries",
                       json={"entry": {"text": "x"}}).status_code == 404
    assert client.post(f"/api/memory/cores/{mc['id']}/entries",
                       json={"entry": {"text": "  "}}).status_code == 400
    gone = client.post(f"/api/memory/cores/{mc['id']}/entries",
                       json={"entry": {"id": "vanished", "text": "x"}})
    assert gone.status_code == 409
    assert "deleted" in gone.get_json()["error"]


def test_entry_delete_and_its_404(client):
    mc = _make_core(client)
    entry = client.post(f"/api/memory/cores/{mc['id']}/entries",
                        json={"entry": {"text": "x"}}).get_json()["entry"]
    r = client.delete(f"/api/memory/cores/{mc['id']}/entries/{entry['id']}")
    assert r.status_code == 200 and r.get_json()["core"]["entries"] == []
    assert client.delete(
        f"/api/memory/cores/{mc['id']}/entries/{entry['id']}").status_code == 404


def test_cores_payload_marks_what_is_below_the_injection_limit(client):
    """The tab used to recompute this rule in JavaScript and could drift from it."""
    mc = _make_core(client)
    client.patch(f"/api/memory/cores/{mc['id']}", json={"inject_limit": 1})
    for text, imp in (("loud", 9), ("quiet", 1)):
        client.post(f"/api/memory/cores/{mc['id']}/entries",
                    json={"entry": {"text": text, "importance": imp}})
    entries = client.get("/api/memory/cores").get_json()["cores"][0]["entries"]
    assert {e["text"]: e["injected"] for e in entries} == {"loud": True, "quiet": False}


# --------------------------- extract route ---------------------------

def test_extract_404s_without_a_chat_or_a_core(client):
    mc = _make_core(client)
    assert client.post("/api/memory/extract",
                       json={"chat_id": "nope", "core_id": mc["id"]}).status_code == 404
    assert client.post("/api/memory/extract",
                       json={"chat": {"id": "c"}, "core_id": "nope"}).status_code == 404


def test_extract_skips_a_private_chat_on_the_automatic_trigger(client):
    mc = _make_core(client)
    body = client.post("/api/memory/extract", json={
        "chat": {"id": "c", "private": True, "messages": [{"role": "user", "content": "hi"}]},
        "core_id": mc["id"], "mode": "auto",
    }).get_json()
    assert body["ok"] is False and body["skipped"] == "private"


def test_a_manual_extract_overrides_the_private_skip(client):
    """Pressing the button in a private chat is the user asking for it. It gets past
    eligibility and stops at the model check instead."""
    mc = _make_core(client)
    body = client.post("/api/memory/extract", json={
        "chat": {"id": "c", "private": True, "messages": [{"role": "user", "content": "hi"}]},
        "core_id": mc["id"], "mode": "manual",
    }).get_json()
    assert "skipped" not in body
    assert body["ok"] is False and "No model available" in body["error"]


def test_extract_reports_a_missing_model_rather_than_failing_silently(client):
    mc = _make_core(client)
    body = client.post("/api/memory/extract", json={
        "chat": {"id": "c", "messages": [{"role": "user", "content": "hi"}]},
        "core_id": mc["id"], "mode": "auto",
    }).get_json()
    assert body["ok"] is False and "No model available" in body["error"]


def test_consolidate_404s_and_reports_a_missing_model(client):
    mc = _make_core(client)
    assert client.post("/api/memory/cores/nope/consolidate", json={}).status_code == 404
    body = client.post(f"/api/memory/cores/{mc['id']}/consolidate", json={}).get_json()
    assert body["ok"] is False and "No model available" in body["error"]


def test_build_404s_on_an_unknown_core(client):
    assert client.post("/api/memory/cores/nope/build", json={}).status_code == 404


# =====================================================================
# Extraction passes driven by a stub model
# =====================================================================
# Everything above stops at the "no model configured" guard, so `_memory_pass` — the
# function the whole feature is built on — never executed. These drive it for real.

@pytest.fixture
def memory_model(client, monkeypatch):
    """Configure the helper model the memory passes use, and install a stub adapter
    for it. Call with the replies the model should give, in order."""
    def install(*replies):
        r = client.post("/api/settings", json={"rewrite_model": "test-model"})
        assert r.status_code == 200, r.get_data(as_text=True)
        return use_adapter(monkeypatch, StubAdapter(replies))
    return install


def _ops(*operations):
    """A well-formed extractor reply."""
    return json.dumps({"operations": list(operations)})


def _seed_chat(client, title="Chat", messages=None, created=None, model="test-model"):
    """A stored chat with a transcript the extractor can read.

    ``model=""`` leaves the chat with no model of its own, which — with no configured
    rewrite model — is the "nothing can run a pass" state, since ``_memory_model``
    falls back to the chat's own model.
    """
    chat = client.post("/api/chats", json={"title": title, "model": model}) \
                 .get_json()["chat"]
    chat["title"] = title
    chat["messages"] = messages if messages is not None else [
        {"role": "user", "content": "I always want short answers."},
        {"role": "assistant", "content": "Understood."},
    ]
    if created:
        chat["created"] = created
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})
    return chat


def _extract(client, core_id, chat_id, mode="auto"):
    return client.post("/api/memory/extract",
                       json={"chat_id": chat_id, "core_id": core_id, "mode": mode})


def _entry_texts(core):
    return [e["text"] for e in core["entries"]]


# --------------------------- the pass writes what the model returned ---------------

def test_an_extraction_pass_writes_what_the_model_returned(client, memory_model):
    mc = _make_core(client)
    memory_model(_ops({"op": "add", "category": "preferences",
                       "text": "Prefers concise answers.", "importance": 8}))
    chat = _seed_chat(client)

    body = _extract(client, mc["id"], chat["id"]).get_json()

    assert body["ok"] is True and body["text"] == "1 added"
    e = body["core"]["entries"][0]
    assert (e["text"], e["category"], e["importance"]) == ("Prefers concise answers.", "preferences", 8)
    assert e["origin"] == "ai"                       # not user-written, so not protected
    assert e["source_chat_id"] == chat["id"]


def test_an_extraction_pass_can_update_merge_and_delete_existing_memories(client, memory_model):
    mc = _make_core(client)
    ids = [client.post(f"/api/memory/cores/{mc['id']}/entries",
                       json={"entry": {"text": t}}).get_json()["entry"]["id"]
           for t in ("one", "two", "three")]
    # Entries added by hand are origin="user" and protected, so make them AI-owned the
    # only way a test can: extract them instead.
    client.delete(f"/api/memory/cores/{mc['id']}/entries/{ids[0]}")
    client.delete(f"/api/memory/cores/{mc['id']}/entries/{ids[1]}")
    client.delete(f"/api/memory/cores/{mc['id']}/entries/{ids[2]}")
    memory_model(_ops({"op": "add", "text": "one"}, {"op": "add", "text": "two"},
                      {"op": "add", "text": "three"}),
                 None)
    chat = _seed_chat(client)
    core = _extract(client, mc["id"], chat["id"]).get_json()["core"]
    a, b, c = [e["id"] for e in core["entries"]]

    memory_model(_ops({"op": "update", "id": a, "text": "ONE"},
                      {"op": "merge", "ids": [b, c], "text": "two and three"}))
    body = _extract(client, mc["id"], chat["id"]).get_json()

    assert body["summary"] == {"added": 0, "updated": 1, "merged": 1, "deleted": 0}
    assert _entry_texts(body["core"]) == ["ONE", "two and three"]


@pytest.mark.parametrize("wrap", [
    "{body}",
    "```json\n{body}\n```",
    "Sure — here are the operations:\n\n{body}\n\nLet me know if that helps.",
])
def test_the_reply_is_parsed_out_of_fences_and_prose(client, memory_model, wrap):
    """Local models rarely return bare JSON even when asked; rewrite._extract_json is
    what makes the feature usable, so pin it through the route."""
    mc = _make_core(client)
    memory_model(wrap.format(body=_ops({"op": "add", "text": "Lives in Ohio."})))
    chat = _seed_chat(client)

    core = _extract(client, mc["id"], chat["id"]).get_json()["core"]
    assert _entry_texts(core) == ["Lives in Ohio."]


@pytest.mark.parametrize("reply", [
    "",                                     # model returned nothing
    "I'm afraid I can't help with that.",   # a refusal, no JSON at all
    "{}",                                   # valid JSON, no operations key
    '{"operations": "not a list"}',
    '{"operations": [{"op": "add"}]}',      # an operation with no text
    '{"operations": null}',
    "{ this is not valid json",
    RuntimeError("model exploded"),         # the call itself failed
])
def test_a_failed_or_unusable_pass_is_a_silent_no_op(client, memory_model, reply):
    """A pass rides along behind an ordinary send. Whatever the model does, it must not
    break the request or corrupt the core."""
    mc = _make_core(client)
    memory_model(reply)
    chat = _seed_chat(client)

    r = _extract(client, mc["id"], chat["id"])
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["text"] == ""
    assert body["core"]["entries"] == []


def test_an_empty_transcript_never_reaches_the_model(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops({"op": "add", "text": "invented"}))
    chat = _seed_chat(client, messages=[])

    body = _extract(client, mc["id"], chat["id"]).get_json()
    assert adapter.calls == 0
    assert body["core"]["entries"] == []


# --------------------------- what the extractor is shown ---------------------------

def test_the_extractor_sees_the_system_prompt_and_the_transcript(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops())
    chat = _seed_chat(client, messages=[
        {"role": "system", "content": "SYSTEM-PROMPT-TEXT"},
        {"role": "user", "content": "I live in Ohio."},
        {"role": "assistant", "content": "DRAFT-ANSWER", "intermediate": True},
        {"role": "assistant", "content": "Noted."},
    ])
    _extract(client, mc["id"], chat["id"])

    assert adapter.system() == memory._EXTRACT_SYS
    prompt = adapter.prompt()
    assert "User: I live in Ohio." in prompt and "Assistant: Noted." in prompt
    # Multi-Pass intermediates and the system prompt are not part of the conversation.
    assert "DRAFT-ANSWER" not in prompt and "SYSTEM-PROMPT-TEXT" not in prompt


def test_the_extractor_is_shown_memories_that_are_below_the_injection_limit(client, memory_model):
    """render_core_for_prompt ignores inject_limit on purpose: a memory the model can't
    see is one it will happily add all over again as a duplicate."""
    mc = _make_core(client)
    client.patch(f"/api/memory/cores/{mc['id']}", json={"inject_limit": 1})
    entries = [client.post(f"/api/memory/cores/{mc['id']}/entries",
                           json={"entry": {"text": t, "importance": imp}}).get_json()["entry"]
               for t, imp in (("Loud memory.", 9), ("Quiet memory.", 1))]
    loud, quiet = entries
    assert loud["id"] and quiet["id"]

    adapter = memory_model(_ops())
    chat = _seed_chat(client)
    core = _extract(client, mc["id"], chat["id"]).get_json()["core"]

    # Only the loud one is injected into a real chat...
    assert [e["injected"] for e in core["entries"]] == [True, False]
    # ...but the extractor is shown both, with their ids, so it can refine either.
    prompt = adapter.prompt()
    assert loud["id"] in prompt and quiet["id"] in prompt
    assert "Quiet memory." in prompt
    # And it is told which ones it may not delete.
    assert "[user-written, do not delete]" in prompt


# --------------------------- the turn counter (§3e) ---------------------------

def _turns_since(client, chat_id):
    return client.get(f"/api/chats/{chat_id}").get_json()["chat"].get("memory_turns_since")


def _set_turns(client, chat, n):
    chat["memory_turns_since"] = n
    client.post(f"/api/chats/{chat['id']}/persist", json={"chat": chat})


def test_a_pass_that_runs_zeroes_the_turn_counter_server_side(client, memory_model):
    """The count used to live only in the browser: a reload mid-cycle lost it and two
    tabs on one chat double-counted."""
    mc = _make_core(client)
    memory_model(_ops())
    chat = _seed_chat(client)
    _set_turns(client, chat, 5)

    _extract(client, mc["id"], chat["id"])
    assert _turns_since(client, chat["id"]) == 0


def test_the_turn_counter_is_untouched_when_no_model_is_configured(client):
    mc = _make_core(client)
    chat = _seed_chat(client, model="")
    _set_turns(client, chat, 5)

    body = _extract(client, mc["id"], chat["id"]).get_json()
    assert body["ok"] is False and "No model available" in body["error"]
    assert _turns_since(client, chat["id"]) == 5


def test_a_skipped_private_chat_never_reaches_the_model(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops({"op": "add", "text": "should not exist"}))

    body = client.post("/api/memory/extract", json={
        "chat": {"id": "p1", "private": True,
                 "messages": [{"role": "user", "content": "secret"}]},
        "core_id": mc["id"], "mode": "auto",
    }).get_json()

    assert body["skipped"] == "private"
    assert adapter.calls == 0
    assert body["core"]["entries"] == []


def test_a_manual_extract_on_a_private_chat_really_writes(client, memory_model):
    """The override exists because pressing the button is the user asking for it. It
    now runs a real pass — assert the memory actually lands."""
    mc = _make_core(client)
    memory_model(_ops({"op": "add", "text": "Learned in a private chat."}))

    body = client.post("/api/memory/extract", json={
        "chat": {"id": "p1", "private": True,
                 "messages": [{"role": "user", "content": "I prefer tea."}]},
        "core_id": mc["id"], "mode": "manual",
    }).get_json()

    assert body["ok"] is True
    assert _entry_texts(body["core"]) == ["Learned in a private chat."]
    # Confirmed against the stored core, not just the response.
    stored = client.get("/api/memory/cores").get_json()["cores"][0]
    assert _entry_texts(stored) == ["Learned in a private chat."]


# --------------------------- consolidation chaining (§4a) ---------------------------

def _over_limit_core(client, n=4, limit=2):
    """A core past inject_limit * 1.5, so the next pass chains a consolidation."""
    mc = _make_core(client)
    client.patch(f"/api/memory/cores/{mc['id']}", json={"inject_limit": limit})
    for i in range(n):
        client.post(f"/api/memory/cores/{mc['id']}/entries",
                    json={"entry": {"text": f"memory {i}"}})
    return mc


def test_an_over_limit_core_chains_a_consolidation_pass(client, memory_model):
    mc = _over_limit_core(client)
    adapter = memory_model(_ops(), _ops())
    chat = _seed_chat(client)

    _extract(client, mc["id"], chat["id"])

    assert adapter.calls == 2
    assert adapter.system(0) == memory._EXTRACT_SYS
    assert adapter.system(1) == memory._CONSOLIDATE_SYS


def test_a_core_under_the_line_does_not_chain_one(client, memory_model):
    mc = _over_limit_core(client, n=2, limit=10)
    adapter = memory_model(_ops(), _ops())
    chat = _seed_chat(client)

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 1


def test_a_consolidation_that_compacts_nothing_is_not_retried(client, memory_model):
    """The retry storm: an over-limit core whose refine returns no operations used to
    pay for a second model call on every extraction, forever."""
    mc = _over_limit_core(client)
    adapter = memory_model(_ops(), _ops(), _ops(), _ops())
    chat = _seed_chat(client)

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 2               # extract + consolidate

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 3               # extract only — the backoff holds


def test_growing_the_core_re_arms_consolidation(client, memory_model):
    mc = _over_limit_core(client)
    adapter = memory_model(_ops(), _ops(),
                           _ops({"op": "add", "text": "something new"}), _ops())
    chat = _seed_chat(client)

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 2

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 4               # grew past the mark, so it tries again


def test_a_chained_consolidation_actually_compacts_the_core(client, memory_model):
    """One request, both halves: the extraction pushes the core over the line and the
    consolidation it chains merges what it finds. The consolidator's reply is computed
    from the prompt it is handed, so it targets the ids the extraction just created."""
    def merge_the_first_three(adapter):
        ids = re.findall(r"id=(\w+)", adapter.prompt(-1))
        return _ops({"op": "merge", "ids": ids[:3], "text": "all three"})

    mc = _make_core(client)
    client.patch(f"/api/memory/cores/{mc['id']}", json={"inject_limit": 2})
    adapter = memory_model(_ops(*[{"op": "add", "text": f"memory {i}"} for i in range(4)]),
                           merge_the_first_three)
    chat = _seed_chat(client)

    body = _extract(client, mc["id"], chat["id"]).get_json()

    assert adapter.calls == 2
    assert body["summary"] == {"added": 4, "updated": 0, "merged": 1, "deleted": 0}
    assert _entry_texts(body["core"]) == ["all three", "memory 3"]


def test_a_consolidation_the_model_never_answered_is_not_recorded(client, memory_model):
    """A failed call is not an attempt. Recording it would let one flaky request
    suppress compaction until the core grew past its old size."""
    mc = _over_limit_core(client)
    adapter = memory_model(_ops(), RuntimeError("grader offline"), _ops(), _ops())
    chat = _seed_chat(client)

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 2                   # extract + a consolidation that failed

    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 4                   # so it is tried again, not backed off


# --------------------------- the retroactive build (§4b, §2) ---------------------------

def _build(client, core_id, chat_ids, reread=False, run_id="build-run"):
    return sse_frames(client.post(f"/api/memory/cores/{core_id}/build",
                                  json={"chat_ids": chat_ids, "run_id": run_id,
                                        "reread": reread}))


def test_the_build_reads_every_chat_oldest_first_and_records_them(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops({"op": "add", "text": "from the older chat"}),
                           _ops({"op": "add", "text": "from the newer chat"}))
    older = _seed_chat(client, "Older", created="2024-01-01")
    newer = _seed_chat(client, "Newer", created="2025-01-01")

    # Deliberately passed newest-first: the route sorts, the caller doesn't.
    frames = _build(client, mc["id"], [newer["id"], older["id"]])

    assert events(frames)[0] == "start" and events(frames)[-1] == "done"
    statuses = [d["message"] for d in all_of(frames, "status")]
    assert "Older" in statuses[0] and "Newer" in statuses[1]
    assert [d["done"] for d in all_of(frames, "progress")] == [1, 2]

    built = first(frames, "built")
    assert built["text"] == "2 added"
    assert _entry_texts(built["core"]) == ["from the older chat", "from the newer chat"]
    assert set(built["core"]["built_chat_ids"]) == {older["id"], newer["id"]}
    assert adapter.calls == 2


def test_a_second_build_reads_nothing_and_never_calls_the_model(client, memory_model):
    """The skip that stops a re-run re-paying for the whole history."""
    mc = _make_core(client)
    adapter = memory_model(_ops({"op": "add", "text": "learned once"}), _ops())
    chat = _seed_chat(client)

    _build(client, mc["id"], [chat["id"]])
    assert adapter.calls == 1

    frames = _build(client, mc["id"], [chat["id"]], run_id="second")
    assert adapter.calls == 1                       # the model was never asked again
    assert "already learned from" in first(frames, "status")["message"]
    built = first(frames, "built")
    assert built["text"] == "" and _entry_texts(built["core"]) == ["learned once"]


def test_rereading_takes_the_whole_history_again(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops({"op": "add", "text": "first pass"}),
                           _ops({"op": "add", "text": "second pass"}))
    chat = _seed_chat(client)

    _build(client, mc["id"], [chat["id"]])
    _build(client, mc["id"], [chat["id"]], reread=True, run_id="second")

    assert adapter.calls == 2
    stored = client.get("/api/memory/cores").get_json()["cores"][0]
    assert _entry_texts(stored) == ["first pass", "second pass"]


def test_the_build_skips_private_chats(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops(), _ops())
    kept = _seed_chat(client, "Public")
    private = client.post("/api/chats", json={"private": True}).get_json()["chat"]

    frames = _build(client, mc["id"], [kept["id"], private["id"]])
    assert adapter.calls == 1
    assert [d["total"] for d in all_of(frames, "progress")] == [1]


def test_deleting_the_core_mid_build_stops_the_run(client, memory_model):
    """The headline concurrency fix. Without the abort the loop keeps spending model
    calls on an orphan and writes a core list that no longer contains it."""
    mc = _make_core(client)
    chats = [_seed_chat(client, f"Chat {i}", created=f"202{i}-01-01") for i in range(3)]

    def delete_the_core_then_answer(adapter):
        client.delete(f"/api/memory/cores/{mc['id']}")
        return _ops({"op": "add", "text": "lands in an orphan"})

    adapter = memory_model(_ops({"op": "add", "text": "first"}),
                           delete_the_core_then_answer,
                           _ops({"op": "add", "text": "never reached"}))

    frames = _build(client, mc["id"], [c["id"] for c in chats])

    assert adapter.calls == 2                       # stopped, did not read the third
    assert "deleted" in first(frames, "error")["message"]
    assert "built" not in events(frames)
    assert client.get("/api/memory/cores").get_json()["cores"] == []


def test_the_build_reports_when_there_is_nothing_eligible(client, memory_model):
    mc = _make_core(client)
    adapter = memory_model(_ops())
    frames = _build(client, mc["id"], [])
    assert adapter.calls == 0
    assert "No eligible chats" in first(frames, "status")["message"]
    assert first(frames, "built")["text"] == ""


def test_the_build_reports_a_missing_model_instead_of_running(client):
    mc = _make_core(client)
    chat = _seed_chat(client, model="")
    frames = _build(client, mc["id"], [chat["id"]])
    assert "No model available" in first(frames, "error")["message"]


# --------------------------- refine route ---------------------------

def test_the_refine_route_runs_a_consolidation_pass(client, memory_model):
    mc = _make_core(client)
    ids = [client.post(f"/api/memory/cores/{mc['id']}/entries",
                       json={"entry": {"text": t}}).get_json()["entry"]["id"]
           for t in ("likes tea", "drinks tea daily")]
    adapter = memory_model(_ops({"op": "update", "id": ids[0], "text": "Drinks tea daily."}))

    body = client.post(f"/api/memory/cores/{mc['id']}/consolidate",
                       json={"model": "test-model"}).get_json()

    assert adapter.system(0) == memory._CONSOLIDATE_SYS
    assert body["ok"] is True and body["text"] == "1 refined"
    assert "Drinks tea daily." in _entry_texts(body["core"])


def test_refining_records_the_attempt_so_it_backs_off(client, memory_model):
    mc = _over_limit_core(client)
    memory_model(_ops(), _ops())
    client.post(f"/api/memory/cores/{mc['id']}/consolidate", json={"model": "test-model"})

    # The extraction that follows must not chain a second consolidation: one already
    # ran at this size and achieved nothing.
    adapter = memory_model(_ops())
    chat = _seed_chat(client)
    _extract(client, mc["id"], chat["id"])
    assert adapter.calls == 1


# --------------------------- incognito ---------------------------

# =====================================================================
# /api/memory/eval — scoring the extractor itself
# =====================================================================
# The eval needs a real model to mean anything; what is testable here is that it runs
# the PRODUCTION extractor rather than a paraphrase of it, and that it never leaks a
# scratch core into the store.

def _score(client, rows, run_id="score-run", **over):
    project = {"rows": rows, "gen_model": "test-model", "grader_model": "test-model",
               "gen_server_url": "http://localhost:11434",
               "grader_server_url": "http://localhost:11434", **over}
    return client.post("/api/memory/eval", json={"eval": project, "run_id": run_id})


def _grade(**scores):
    return json.dumps({k: {"score": v, "reasoning": "because"} for k, v in scores.items()})


_ALL_TEN = {c["label"]: 10 for c in evals.MEMORY_CRITERIA}


def test_the_seed_dataset_covers_the_cases_the_rules_argue_about(client):
    body = client.get("/api/memory/eval/seed").get_json()
    assert len(body["rows"]) >= 4
    assert all(r["Transcript"].strip() for r in body["rows"])
    assert [c["label"] for c in body["criteria"]] == \
        ["Durability", "Non-duplication", "Categorisation", "Faithfulness"]
    # At least one row starts from existing memories — the duplication case needs them.
    assert any(r["ExistingMemories"].strip() for r in body["rows"])


def test_the_eval_runs_the_production_extractor(client, memory_model):
    """The point of the route: not a hand-copied paraphrase of the prompt, the real one."""
    adapter = memory_model(_ops({"op": "add", "text": "Prefers tea.", "importance": 7}),
                           _grade(**_ALL_TEN))
    frames = sse_frames(_score(client, [
        {"Transcript": "User: I only ever drink tea.\n\nAssistant: Noted.",
         "ExistingMemories": "Lives in Ohio.", "Note": "a preference"},
    ]))

    # The extractor call used the production system prompt and was shown the existing
    # memory, exactly as a real pass would be.
    assert adapter.system(0) == memory._EXTRACT_SYS
    assert "Lives in Ohio." in adapter.prompt(0)
    assert "I only ever drink tea." in adapter.prompt(0)

    row = first(frames, "row_result")
    assert row["ungraded"] is False
    assert row["operations"] == [{"op": "add", "text": "Prefers tea.", "importance": 7}]
    assert "Prefers tea." in row["profile"]
    assert first(frames, "summary")["aggregate"]["overall"] == 100.0


def test_the_judge_is_shown_the_operations_and_the_resulting_profile(client, memory_model):
    adapter = memory_model(_ops({"op": "add", "text": "Prefers tea."}), _grade(**_ALL_TEN))
    sse_frames(_score(client, [{"Transcript": "User: tea please", "ExistingMemories": ""}]))

    grader_prompt = adapter.prompt(1)
    assert "=== RESPONSE TO GRADE ===" in grader_prompt
    assert "Prefers tea." in grader_prompt
    assert "Resulting memory profile:" in grader_prompt
    for label in _ALL_TEN:
        assert label in grader_prompt          # the rubric the grader is scored on


def test_an_eval_never_leaves_a_scratch_core_behind(client, memory_model):
    """A scratch core entering the store would show up in the tab and be injected into
    chats. It must exist only for the length of the row."""
    memory_model(_ops({"op": "add", "text": "should not persist"}), _grade(**_ALL_TEN))
    sse_frames(_score(client, [{"Transcript": "User: hi", "ExistingMemories": ""}]))
    assert client.get("/api/memory/cores").get_json()["cores"] == []


def test_recording_nothing_is_scored_not_skipped(client, memory_model):
    """An empty operations list is the correct answer for a conversation with nothing
    durable in it, so the row still has to reach the judge."""
    adapter = memory_model(_ops(), _grade(**_ALL_TEN))
    frames = sse_frames(_score(client, [
        {"Transcript": "User: convert 40F to C\n\nAssistant: 4.4C", "ExistingMemories": ""},
    ]))
    assert adapter.calls == 2                          # extracted AND graded
    row = first(frames, "row_result")
    assert row["operations"] == [] and row["ungraded"] is False


def test_a_row_with_an_empty_transcript_is_reported_ungraded(client, memory_model):
    adapter = memory_model(_grade(**_ALL_TEN))
    frames = sse_frames(_score(client, [{"Transcript": "   ", "ExistingMemories": ""}]))
    assert adapter.calls == 0
    assert first(frames, "row_result")["ungraded"] is True
    assert first(frames, "summary")["aggregate"]["graded"] == 0


def test_a_grader_that_returns_nothing_usable_leaves_the_row_ungraded(client, memory_model):
    """An unparseable grade must not become a zero — that would drag the average down
    with an invented number."""
    memory_model(_ops({"op": "add", "text": "x"}), "I'd rather not grade that.")
    frames = sse_frames(_score(client, [{"Transcript": "User: hi", "ExistingMemories": ""}]))
    assert first(frames, "row_result")["ungraded"] is True
    assert first(frames, "summary")["aggregate"]["graded"] == 0


def test_a_failed_extraction_still_produces_a_graded_row(client, memory_model):
    """The extractor failing IS a result worth scoring — it recorded nothing."""
    memory_model(RuntimeError("model exploded"), _grade(**_ALL_TEN))
    frames = sse_frames(_score(client, [{"Transcript": "User: hi", "ExistingMemories": ""}]))
    row = first(frames, "row_result")
    assert row["operations"] == [] and row["ungraded"] is False


def test_every_row_is_reported_and_progress_counts_up(client, memory_model):
    memory_model(*([_ops(), _grade(**_ALL_TEN)] * 3))
    rows = [{"Transcript": f"User: message {i}", "ExistingMemories": ""} for i in range(3)]
    frames = sse_frames(_score(client, rows))
    assert len(all_of(frames, "row_result")) == 3
    assert [d["done"] for d in all_of(frames, "progress")] == [1, 2, 3]
    assert events(frames)[-1] == "done"


@pytest.mark.parametrize("project,message", [
    ({"rows": []}, "at least one transcript"),
    ({"grader_model": ""}, "grader model"),
    ({"gen_model": ""}, "model whose extraction"),
])
def test_the_eval_refuses_an_incomplete_request(client, project, message):
    rows = [{"Transcript": "User: hi", "ExistingMemories": ""}]
    r = _score(client, project.pop("rows", rows), **project)
    assert r.status_code == 400 and message in r.get_json()["error"]


def test_existing_memories_are_seeded_as_protected_user_entries(client, memory_model):
    """A pinned or user-written memory is one the extractor is told not to delete —
    the seed rows rely on that to test whether it obeys."""
    adapter = memory_model(_ops(), _grade(**_ALL_TEN))
    sse_frames(_score(client, [
        {"Transcript": "User: hi", "ExistingMemories": "One.\nTwo."},
    ]))
    prompt = adapter.prompt(0)
    assert prompt.count("[user-written, do not delete]") == 2


def test_an_incognito_session_writes_no_memories_to_disk(client, memory_model):
    memory_model(_ops({"op": "add", "text": "ephemeral"}))
    client.post("/api/profiles/data/incognito", json={"seed": "blank"})
    mc = _make_core(client)
    chat = _seed_chat(client)

    body = _extract(client, mc["id"], chat["id"]).get_json()
    assert body["ok"] is False and body["skipped"] == "incognito"
    assert not store_mod.core.MEMORY_CORES_FILE.exists()


# =====================================================================
# built_chat_ids — the retroactive build must not re-pay for the history
# =====================================================================
# The skip itself is asserted against the real route further down
# (test_a_second_build_reads_nothing_and_never_calls_the_model). It used to be checked
# against a copy of the route's filter written here, which would have kept passing if
# the route's own filter broke.

def test_built_chat_ids_survive_a_save_and_reload(store):
    """The field has to be persisted and normalised, or the skip silently stops working
    the next time the app starts."""
    mc = store.add_memory_core("c")
    store.mutate_memory_core(mc["id"], lambda c: c["built_chat_ids"].append("chat-1"))
    reloaded = store_mod.Store()
    assert reloaded.get_memory_core(mc["id"])["built_chat_ids"] == ["chat-1"]


def test_normalize_core_repairs_a_junk_built_chat_ids():
    mc = memory.normalize_core({"built_chat_ids": "not a list"})
    assert mc["built_chat_ids"] == []
    mc = memory.normalize_core({"built_chat_ids": ["a", None, {"x": 1}, "b"]})
    assert mc["built_chat_ids"] == ["a", "b"]
