#!/usr/bin/env python3
"""Tests for the persona feature: definition storage, the pipeline engine, the memory
store's weight blending, and import/export safety.

The bug that motivated most of this: ``PersonaService.save`` accepted definitions that
``from_xml`` refuses to read back. Saving a persona with its name cleared wrote a
persona.xml nothing could parse — ``load`` raised and ``list_all`` silently skipped the
folder, so the persona vanished from the UI while its knowledge base and memories stayed
on disk, unreachable. ``save`` now validates by round-tripping before it writes.

Everything runs against a throwaway tree with the vector store stubbed out; no DuckDB /
LanceDB / Ollama is involved.
"""

import json
import zipfile

import pytest

from app import core, persona as persona_mod, persona_io, persona_store, pipeline

from conftest import StubAdapter, sse_frames, use_adapter


# --------------------------- fixtures ---------------------------

@pytest.fixture
def psvc(tmp_path, monkeypatch):
    """A PersonaService pointed at an empty, isolated personas dir."""
    monkeypatch.setattr(core, "PERSONAS_DIR", tmp_path / "personas")
    return persona_mod.PersonaService()


@pytest.fixture
def fake_rag(monkeypatch):
    """Replace the shared retrieval layer with an in-memory record of what was written.
    The persona stores are thin wrappers over rag.py, which is covered by its own tests."""
    calls = {"upserts": [], "deletes": [], "results": []}

    def upsert_item(source_type, source_id, item_id, content, embed_fn, model, **kw):
        calls["upserts"].append({"type": source_type, "sid": source_id,
                                 "item": item_id, "content": content,
                                 "meta": kw.get("meta") or {}})
        return 1

    monkeypatch.setattr(persona_store.rag, "upsert_item", upsert_item)
    monkeypatch.setattr(persona_store.rag, "delete_item",
                        lambda st, sid, iid: calls["deletes"].append((st, sid, iid)))
    monkeypatch.setattr(persona_store.rag, "retrieve",
                        lambda *a, **k: [dict(r) for r in calls["results"]])
    monkeypatch.setattr(persona_store.rag, "list_items", lambda st, sid: [])
    return calls


# --------------------------- definition round trip ---------------------------

def test_xml_round_trip_preserves_every_field(psvc):
    p = persona_mod.default_persona("Ada Lovelace", chat_model="llama3")
    p["profile"].update({"bio": "Analyst.", "role": "Mathematician"})
    p["speaking"].update({
        "tone": "precise", "quirks": "asks questions",
        "variants": [{"name": "Terse", "description": "Two sentences, no more."}],
        "examples": [{"user": "hi", "reply": "Good morning."}],
    })
    p["models"]["temperature"] = 0.0
    p["stores"]["retrieval"] = "keyword"
    p["stores"]["prompt_reword"] = False

    back = persona_mod.from_xml(persona_mod.to_xml(p))

    assert back["profile"]["name"] == "Ada Lovelace"
    assert back["profile"]["role"] == "Mathematician"
    assert back["speaking"]["variants"] == [{"name": "Terse", "description": "Two sentences, no more."}]
    assert back["speaking"]["examples"] == [{"user": "hi", "reply": "Good morning."}]
    assert back["models"]["temperature"] == 0.0     # a deliberate 0 must survive
    assert back["stores"]["retrieval"] == "keyword"
    assert back["stores"]["prompt_reword"] is False
    assert [s["id"] for s in back["pipeline"]] == [s["id"] for s in p["pipeline"]]
    assert back["pipeline"][0]["schema"] == p["pipeline"][0]["schema"]


def test_to_xml_tolerates_none_in_stores(psvc):
    """A None store value used to raise TypeError out of ElementTree — a 500 on save."""
    p = persona_mod.default_persona("Null Store")
    p["stores"]["embedding_model_used"] = None
    p["stores"]["knowledge"] = None
    back = persona_mod.from_xml(persona_mod.to_xml(p))
    assert back["stores"]["knowledge"] == "knowledge/"
    assert back["stores"]["embedding_model_used"] == ""


@pytest.mark.parametrize("mangle, needle", [
    (lambda x: x.replace('version="1"', 'version="9"'), "version"),
    (lambda x: x.replace("<name>N</name>", "<name></name>"), "name"),
    (lambda x: x.replace('knowledge="knowledge/"', 'knowledge="../../etc"'), "Unsafe"),
    (lambda x: x.replace('type="llm"', 'type="shell"'), "step type"),
    (lambda x: x.replace("<schema>", "<schema>{{{"), "schema"),
    (lambda x: "<notpersona/>", "Root element"),
    (lambda x: "<persona", "valid XML"),
])
def test_from_xml_rejects_malformed_definitions(mangle, needle):
    xml = persona_mod.to_xml(persona_mod.default_persona("N"))
    with pytest.raises(persona_mod.PersonaError) as e:
        persona_mod.from_xml(mangle(xml))
    assert needle.lower() in str(e.value).lower()


def test_from_xml_falls_back_on_unknown_retrieval_mode():
    """An unknown mode is recoverable, unlike the failures above."""
    xml = persona_mod.to_xml(persona_mod.default_persona("N")).replace(
        'retrieval="hybrid"', 'retrieval="telepathy"')
    assert persona_mod.from_xml(xml)["stores"]["retrieval"] == "hybrid"


# --------------------------- save() validation (the data-loss bug) ---------------------------

def test_save_rejects_blank_name_and_leaves_the_stored_definition_intact(psvc):
    p = psvc.create("Keeper")
    pid = p["id"]
    p["profile"]["name"] = "   "

    with pytest.raises(persona_mod.PersonaError):
        psvc.save(p)

    # The persona is still loadable and still listed — not stranded on disk.
    assert psvc.load(pid)["profile"]["name"] == "Keeper"
    assert [x["id"] for x in psvc.list_all()] == [pid]


def test_save_rejects_a_definition_the_read_path_would_refuse(psvc):
    """Anything from_xml validates, save must reject too — the two used to disagree."""
    p = psvc.create("Keeper")
    p["stores"]["knowledge"] = "../../escape"
    with pytest.raises(persona_mod.PersonaError):
        psvc.save(p)
    assert psvc.load(p["id"])["stores"]["knowledge"] == "knowledge/"


def test_save_rejects_an_unknown_step_type(psvc):
    p = psvc.create("Keeper")
    p["pipeline"].append({"id": "evil", "type": "shell", "prompt": "", "schema": None})
    with pytest.raises(persona_mod.PersonaError):
        psvc.save(p)


def test_save_normalizes_a_stale_version_field(psvc):
    """We always write the current schema; a stale version must not brick the save."""
    p = psvc.create("Keeper")
    p["version"] = "0"
    assert psvc.save(p)["version"] == persona_mod.SCHEMA_VERSION


def test_save_round_trip_keeps_edits(psvc):
    p = psvc.create("Keeper")
    p["profile"]["role"] = "Archivist"
    psvc.save(p)
    assert psvc.load(p["id"])["profile"]["role"] == "Archivist"


# --------------------------- listing ---------------------------

def test_list_all_reports_a_broken_persona_instead_of_hiding_it(psvc):
    """A corrupted persona.xml used to be skipped, leaving an invisible folder that
    still held the user's sources and still occupied its id."""
    good = psvc.create("Good")
    bad = psvc.create("Bad")
    core.write_text(persona_mod.persona_path(bad["id"]) / "persona.xml", "<persona")

    rows = {r["id"]: r for r in psvc.list_all()}
    assert rows[good["id"]].get("broken") is None
    assert rows[bad["id"]]["broken"] is True
    assert rows[bad["id"]]["error"]
    # And it is still deletable.
    assert psvc.delete(bad["id"]) is True
    assert bad["id"] not in {r["id"] for r in psvc.list_all()}


def test_list_all_skips_a_folder_with_no_definition(psvc):
    psvc.create("Real")
    (persona_mod.personas_dir() / "leftover").mkdir()
    assert [r["id"] for r in psvc.list_all()] == ["real"]


# --------------------------- ids ---------------------------

def test_unique_id_suffixes_collisions(psvc):
    assert psvc.create("Same")["id"] == "same"
    assert psvc.create("Same")["id"] == "same-2"
    assert psvc.create("Same")["id"] == "same-3"


def test_slugify_never_returns_empty():
    assert persona_mod.slugify("!!! ???") == "persona"
    assert persona_mod.slugify("Ada Lovelace") == "ada-lovelace"


@pytest.mark.parametrize("bad", ["", "..", ".", "a/b", "a\\b", "c:evil", "   "])
def test_persona_path_rejects_traversal(psvc, bad):
    with pytest.raises(persona_mod.PersonaError):
        persona_mod.persona_path(bad)


def test_create_requires_a_name(psvc):
    with pytest.raises(persona_mod.PersonaError):
        psvc.create("   ")


# --------------------------- pipeline engine ---------------------------

def _llm_persona(steps, **kw):
    p = persona_mod.default_persona("Ada")
    p["id"] = "ada"
    p["pipeline"] = steps
    p.update(kw)
    return p


def _step(sid, prompt="{user_message}", schema=None, stype="llm"):
    return {"id": sid, "type": stype, "use_history": False, "model": "",
            "prompt": prompt, "schema": schema}


DRAFT_SCHEMA = {"type": "object", "properties": {"draft": {"type": "string"}},
                "required": ["draft"]}


def test_render_template_leaves_unknown_braces_alone():
    out = pipeline.render_template('Reply as {name} using {"k": 1}', {"name": "Ada"})
    assert out == 'Reply as Ada using {"k": 1}'


def test_structured_step_retries_then_succeeds_and_clears_the_failed_raw():
    replies = iter(["not json at all", '{"draft": "ok"}'])
    eng = pipeline.PipelineEngine(_llm_persona([_step("s", schema=DRAFT_SCHEMA)]),
                                  llm_complete=lambda m, msgs, sc: next(replies))
    run = eng.start("hello")
    assert run.status == "complete"
    assert run.steps[0]["output"] == {"draft": "ok"}
    # The first attempt's text must not linger — the UI shows `raw` as an error detail.
    assert run.steps[0]["raw"] is None


def test_structured_step_pauses_when_retries_are_exhausted():
    eng = pipeline.PipelineEngine(_llm_persona([_step("s", schema=DRAFT_SCHEMA)]),
                                  llm_complete=lambda m, msgs, sc: "still not json",
                                  max_retries=2)
    run = eng.start("hello")
    assert run.status == "paused"
    assert run.steps[0]["status"] == pipeline.StepStatus.FAILED
    assert run.steps[0]["error"]
    assert run.steps[0]["raw"] == "still not json"


def test_max_retries_zero_still_attempts_once():
    """A misconfigured max_retries used to skip the loop and fail with error=None."""
    eng = pipeline.PipelineEngine(_llm_persona([_step("s", schema=DRAFT_SCHEMA)]),
                                  llm_complete=lambda m, msgs, sc: '{"draft": "ok"}',
                                  max_retries=0)
    assert eng.start("hello").status == "complete"


def test_pipeline_ending_in_a_structured_step_still_produces_an_answer():
    """Only free-text steps set run.final, so this used to complete with an empty
    answer and the UI left a bubble stuck on the placeholder."""
    eng = pipeline.PipelineEngine(_llm_persona([_step("s", schema=DRAFT_SCHEMA)]),
                                  llm_complete=lambda m, msgs, sc: '{"draft": "the answer"}')
    run = eng.start("hello")
    assert run.status == "complete"
    assert run.final == "the answer"


def test_pipeline_ending_in_a_retrieval_step_falls_back_to_the_last_text():
    steps = [_step("say"), _step("look", stype="knowledge_retrieval")]
    eng = pipeline.PipelineEngine(
        _llm_persona(steps),
        llm_complete=lambda m, msgs, sc: "spoken answer",
        knowledge_search=lambda q: [])
    run = eng.start("hello")
    assert run.status == "complete"
    assert run.final == "spoken answer"


def test_rerunning_the_last_step_keeps_an_answer():
    """run_from(last) re-executes nothing, and run.final is cleared first — so the
    corrected output was thrown away and the answer went blank."""
    steps = [_step("analyze", schema=DRAFT_SCHEMA), _step("stylize", "{draft}")]
    calls = iter(['{"draft": "d"}', "styled"])
    eng = pipeline.PipelineEngine(_llm_persona(steps),
                                  llm_complete=lambda m, msgs, sc: next(calls))
    assert eng.start("hello").final == "styled"

    run = eng.run_from(1, "hand written answer")
    assert run.status == "complete"
    assert run.final == "hand written answer"


def test_run_from_replays_upstream_context_and_invalidates_downstream():
    seen = []

    def complete(model, messages, schema):
        seen.append(messages[-1]["content"])
        return "final text"

    steps = [_step("analyze", schema=DRAFT_SCHEMA), _step("stylize", "Draft was: {draft}")]
    eng = pipeline.PipelineEngine(_llm_persona(steps), llm_complete=complete)
    eng.run.steps[0].update({"output": {"draft": "original"},
                             "status": pipeline.StepStatus.DONE, "type": "llm"})
    eng._user_message, eng._history = "hello", ""

    eng.run_from(0, {"draft": "edited"})
    assert "Draft was: edited" in seen[-1]
    # The downstream step was re-executed, not left with its stale output.
    assert eng.run.steps[1]["output"] == "final text"


def test_should_stop_halts_between_steps():
    """Stop used to be honored only inside the final streaming step, so a cancelled
    run kept marching through the rest of the pipeline."""
    ran = []
    stop = {"now": False}

    def complete(model, messages, schema):
        ran.append(1)
        stop["now"] = True         # cancel arrives while step 0 is in flight
        return "text"

    steps = [_step("one"), _step("two"), _step("three")]
    eng = pipeline.PipelineEngine(_llm_persona(steps), llm_complete=complete,
                                  should_stop=lambda: stop["now"])
    run = eng.start("hello")
    assert run.status == "stopped"
    assert len(ran) == 1
    assert run.final == "text"     # whatever was produced is still shown


def test_selected_variant_reaches_the_speaking_style():
    """The variant picker was populated and then read by nothing."""
    p = _llm_persona([_step("s", "{speaking_style}|{variant}")])
    p["speaking"]["variants"] = [{"name": "Terse", "description": "Two sentences."}]
    seen = []
    eng = pipeline.PipelineEngine(p, variant="terse",   # matched case-insensitively
                                  llm_complete=lambda m, msgs, sc: seen.append(msgs[-1]["content"]) or "x")
    eng.start("hello")
    assert "Terse" in seen[0] and "Two sentences." in seen[0]


def test_unknown_variant_does_not_break_the_run():
    p = _llm_persona([_step("s", "{speaking_style}")])
    eng = pipeline.PipelineEngine(p, variant="ghost",
                                  llm_complete=lambda m, msgs, sc: "x")
    assert eng.start("hello").status == "complete"


def test_chat_system_prompt_is_appended_to_the_persona_identity():
    seen = []
    eng = pipeline.PipelineEngine(
        _llm_persona([_step("s")]), extra_system="Always answer in French.",
        llm_complete=lambda m, msgs, sc: seen.append(msgs[0]) or "x")
    eng.start("hello")
    assert seen[0]["role"] == "system"
    assert "You are Ada." in seen[0]["content"]
    assert "Always answer in French." in seen[0]["content"]


def test_retrieval_step_falls_back_to_the_rewrite_service_for_queries():
    got = {}
    steps = [_step("look", stype="knowledge_retrieval")]
    eng = pipeline.PipelineEngine(
        _llm_persona(steps), llm_complete=lambda *a: "",
        knowledge_search=lambda q: got.setdefault("queries", q) or [],
        rewrite_queries=lambda msg, hist: {"variants": ["who is ada"], "keywords": ["ada"]})
    eng.start("who is she?")
    assert got["queries"] == ["who is ada", "ada"]


def test_step_exception_pauses_the_run_rather_than_raising():
    def boom(model, messages, schema):
        raise RuntimeError("model is down")

    eng = pipeline.PipelineEngine(_llm_persona([_step("s")]), llm_complete=boom)
    run = eng.start("hello")
    assert run.status == "paused"
    assert "model is down" in run.steps[0]["error"]


# --------------------------- memories ---------------------------

def test_emotional_weight_is_clamped(psvc, fake_rag):
    psvc.create("Ada")
    m = persona_store.MemoryService()
    assert m.save_memory("ada", {"title": "t", "emotional_weight": 99}, None, "e")["emotional_weight"] == 10
    assert m.save_memory("ada", {"title": "t", "emotional_weight": -4}, None, "e")["emotional_weight"] == 1
    assert m.save_memory("ada", {"title": "t", "emotional_weight": "nope"}, None, "e")["emotional_weight"] == 5


def test_memory_round_trips_through_disk(psvc, fake_rag):
    psvc.create("Ada")
    m = persona_store.MemoryService()
    saved = m.save_memory("ada", {"title": "First light", "description": "d",
                                  "emotional_weight": 8, "tags": ["x"]}, None, "e")
    listed = m.list_memories("ada")
    assert [x["id"] for x in listed] == [saved["id"]]
    assert listed[0]["title"] == "First light"
    assert m.get_memory("ada", saved["id"])["emotional_weight"] == 8
    # The weight travels in the vector row's meta, which is what the blend reads.
    assert fake_rag["upserts"][-1]["meta"]["weight"] == 8


def test_listing_memories_does_not_create_a_persona_folder(psvc, fake_rag):
    """Reading used to mkdir, so listing memories for an id that doesn't exist
    scaffolded a folder and left an orphan persona directory behind."""
    m = persona_store.MemoryService()
    assert m.list_memories("no-such-persona") == []
    assert not (persona_mod.personas_dir() / "no-such-persona").exists()


def test_weight_blend_reorders_and_never_inflates_a_negative_score(psvc, fake_rag):
    fake_rag["results"] = [
        {"item_id": "low", "score": 0.50, "meta": json.dumps({"weight": 1})},
        {"item_id": "high", "score": 0.45, "meta": json.dumps({"weight": 10})},
        {"item_id": "neg", "score": -0.90, "meta": json.dumps({"weight": 10})},
    ]
    out = persona_store.MemoryService().search("ada", [], "e", 3, weight_influence=0.35)
    # A weight-10 memory outranks a slightly better-matching weight-1 one...
    assert [r["item_id"] for r in out][:2] == ["high", "low"]
    # ...but a negative cosine is floored, not scaled further down the list.
    assert out[-1]["item_id"] == "neg"
    assert out[-1]["score"] == 0.0
    assert out[-1]["base_score"] == -0.90


def test_delete_memory_drops_both_representations(psvc, fake_rag):
    psvc.create("Ada")
    m = persona_store.MemoryService()
    saved = m.save_memory("ada", {"title": "t"}, None, "e")
    m.delete_memory("ada", saved["id"])
    assert m.list_memories("ada") == []
    assert fake_rag["deletes"] == [(persona_store.ST_MEMORY, "ada", saved["id"])]


# --------------------------- import / export ---------------------------

@pytest.mark.parametrize("bad", [
    "../outside.txt", "/etc/passwd", "C:/Windows/system.ini", "sources/../../escape",
    "..\\outside.txt", "other/file.txt", "memories/elsewhere.json",
])
def test_safe_member_rejects_paths_outside_the_persona_folder(bad):
    assert persona_io._safe_member(bad) is False


@pytest.mark.parametrize("ok", ["persona.xml", "sources/book.pdf", "memories/entries/a.json"])
def test_safe_member_accepts_the_three_allowed_locations(ok):
    assert persona_io._safe_member(ok) is True


def test_import_bundle_rejects_a_traversing_archive(psvc, fake_rag, tmp_path):
    buf = tmp_path / "evil.zip"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("persona.xml", persona_mod.to_xml(persona_mod.default_persona("Evil")))
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(persona_mod.PersonaError):
        persona_io.import_bundle(buf.read_bytes(), psvc, None, None,
                                 embed_url="http://localhost", embed_model_default="e")
    assert not (tmp_path / "escape.txt").exists()


def test_import_bundle_requires_a_definition(psvc, fake_rag, tmp_path):
    buf = tmp_path / "empty.zip"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sources/a.txt", "text")
    with pytest.raises(persona_mod.PersonaError) as e:
        persona_io.import_bundle(buf.read_bytes(), psvc, None, None,
                                 embed_url="http://localhost", embed_model_default="e")
    assert "persona.xml" in str(e.value)


def test_bundle_round_trip_preserves_definition_sources_and_memories(psvc, fake_rag):
    p = psvc.create("Traveller")
    kb, mem = persona_store.KnowledgeService(), persona_store.MemoryService()
    kb.add_text(p["id"], "notes.txt", "the knowledge text", None, "e")
    mem.save_memory(p["id"], {"title": "A day", "description": "it rained",
                              "emotional_weight": 9}, None, "e")

    data = persona_io.export_bundle(p["id"])
    names = set(zipfile.ZipFile(__import__("io").BytesIO(data)).namelist())
    assert "persona.xml" in names and "sources/notes.txt" in names
    assert any(n.startswith("memories/entries/") for n in names)

    imported = persona_io.import_bundle(data, psvc, kb, mem, embed_url="http://localhost",
                                        embed_model_default="e", wants_vectors=False)
    assert imported["id"] != p["id"]              # a fresh, non-colliding id
    assert imported["profile"]["name"] == "Traveller"
    copied = mem.list_memories(imported["id"])
    assert [m["title"] for m in copied] == ["A day"]
    assert copied[0]["emotional_weight"] == 9
    assert (persona_mod.persona_path(imported["id"]) / "sources" / "notes.txt").is_file()


def test_export_xml_of_a_missing_persona_raises(psvc):
    with pytest.raises(persona_mod.PersonaError):
        persona_io.export_xml("ghost")


def test_duplicate_copies_sources_and_memories_under_a_new_id(psvc, fake_rag):
    p = psvc.create("Original")
    persona_store.KnowledgeService().add_text(p["id"], "notes.txt", "text", None, "e")
    persona_store.MemoryService().save_memory(p["id"], {"title": "m"}, None, "e")

    dup = psvc.duplicate(p["id"], "Clone")
    assert dup["id"] == "clone" and dup["profile"]["name"] == "Clone"
    assert (persona_mod.persona_path("clone") / "sources" / "notes.txt").is_file()
    assert len(persona_store.MemoryService().list_memories("clone")) == 1
    # The original is untouched.
    assert psvc.load(p["id"])["profile"]["name"] == "Original"


# --------------------------- routes ---------------------------

# `client` comes from tests/conftest.py. No Ollama is reachable, which is exactly the
# state these routes have to behave sanely in.


def test_create_list_and_delete_a_persona_over_http(client):
    r = client.post("/api/personas", json={"name": "Ada"})
    assert r.status_code == 200
    pid = r.get_json()["persona"]["id"]
    assert [p["id"] for p in client.get("/api/personas").get_json()["personas"]] == [pid]

    assert client.delete(f"/api/personas/{pid}").status_code == 200
    assert client.get("/api/personas").get_json()["personas"] == []


def test_put_with_a_blank_name_is_rejected_and_changes_nothing(client):
    pid = client.post("/api/personas", json={"name": "Ada"}).get_json()["persona"]["id"]
    p = client.get(f"/api/personas/{pid}").get_json()["persona"]
    p["profile"]["name"] = ""

    r = client.put(f"/api/personas/{pid}", json={"persona": p})
    assert r.status_code == 400
    assert "name" in r.get_json()["error"].lower()
    # Still readable, still listed — the failure mode this whole change exists to stop.
    assert client.get(f"/api/personas/{pid}").status_code == 200
    assert len(client.get("/api/personas").get_json()["personas"]) == 1


def test_put_with_an_unsafe_store_path_is_rejected(client):
    pid = client.post("/api/personas", json={"name": "Ada"}).get_json()["persona"]["id"]
    p = client.get(f"/api/personas/{pid}").get_json()["persona"]
    p["stores"]["sources"] = "../../../elsewhere"
    assert client.put(f"/api/personas/{pid}", json={"persona": p}).status_code == 400
    assert client.get(f"/api/personas/{pid}").get_json()["persona"]["stores"]["sources"] == "sources/"


@pytest.mark.parametrize("path", ["memories", "knowledge"])
def test_sub_resources_404_for_an_unknown_persona_without_creating_it(client, path):
    r = client.get(f"/api/personas/ghost/{path}")
    assert r.status_code == 404
    assert not (persona_mod.personas_dir() / "ghost").exists()


def test_rerun_of_an_unknown_run_is_a_404(client):
    r = client.post("/api/runs/nope/rerun", json={"index": 0})
    assert r.status_code == 404
    assert "run not found" in r.get_json()["error"]


# --------------------------- the chat route, end to end ---------------------------

@pytest.fixture
def stub_model(monkeypatch):
    """Install conftest's StubAdapter with a list of canned replies and hand it back."""
    def install(replies):
        return use_adapter(monkeypatch, StubAdapter(replies))
    return install


def _single_step_pipeline(schema=None, prompt="{user_message}"):
    return [{"id": "only", "type": "llm", "use_history": False, "model": "",
             "prompt": prompt, "schema": schema}]


_sse_events = sse_frames        # one parser, in tests/conftest.py


def test_persona_chat_runs_the_pipeline_and_streams_a_final_answer(client, stub_model):
    pid = client.post("/api/personas", json={"name": "Ada"}).get_json()["persona"]["id"]
    p = client.get(f"/api/personas/{pid}").get_json()["persona"]
    # Two steps: one structured, one free text that echoes the style.
    p["pipeline"] = [
        {"id": "analyze", "type": "llm", "use_history": False, "model": "", "prompt": "{user_message}",
         "schema": {"type": "object", "properties": {"draft": {"type": "string"}}, "required": ["draft"]}},
        {"id": "stylize", "type": "llm", "use_history": False, "model": "",
         "prompt": "{speaking_style}||{draft}", "schema": None},
    ]
    p["speaking"]["variants"] = [{"name": "Terse", "description": "Two sentences."}]
    p["models"]["temperature"] = 0.0
    assert client.put(f"/api/personas/{pid}", json={"persona": p}).status_code == 200

    adapter = stub_model(['{"draft": "the draft"}', "the styled answer"])
    r = client.post(f"/api/personas/{pid}/chat", json={
        "run_id": "run1", "variant": "Terse",
        "chat": {"id": "c1", "model": "m", "server_url": "http://x", "system_on": True,
                 "system_prompt": "Answer in French.",
                 "messages": [{"role": "user", "content": "who are you?"}]},
    })
    assert r.status_code == 200
    events = dict((e, d) for e, d in _sse_events(r))
    assert events["run_complete"]["final"] == "the styled answer"

    # The chat's system prompt rides along with the persona identity...
    system = adapter.seen[0][1][0]
    assert "You are Ada." in system["content"] and "Answer in French." in system["content"]
    # ...the selected variant reached the stylize prompt...
    assert "Two sentences." in adapter.seen[-1][1][-1]["content"]
    # ...and the persona's temperature was actually applied.
    assert adapter.seen[-1][2]["temperature"] == 0.0


def test_persona_chat_with_a_structured_last_step_still_answers(client, stub_model):
    """The final-answer fallback, over the wire."""
    pid = client.post("/api/personas", json={"name": "Ada"}).get_json()["persona"]["id"]
    p = client.get(f"/api/personas/{pid}").get_json()["persona"]
    p["pipeline"] = _single_step_pipeline(
        schema={"type": "object", "properties": {"draft": {"type": "string"}},
                "required": ["draft"]})
    client.put(f"/api/personas/{pid}", json={"persona": p})

    stub_model(['{"draft": "structured answer"}'])
    r = client.post(f"/api/personas/{pid}/chat", json={
        "run_id": "run2",
        "chat": {"id": "c1", "model": "m", "server_url": "http://x",
                 "messages": [{"role": "user", "content": "hi"}]}})
    events = dict((e, d) for e, d in _sse_events(r))
    assert events["run_complete"]["final"] == "structured answer"


def test_rerun_index_out_of_range_is_a_400(client, stub_model):
    pid = client.post("/api/personas", json={"name": "Ada"}).get_json()["persona"]["id"]
    p = client.get(f"/api/personas/{pid}").get_json()["persona"]
    p["pipeline"] = _single_step_pipeline()
    client.put(f"/api/personas/{pid}", json={"persona": p})
    stub_model(["answer"])
    client.post(f"/api/personas/{pid}/chat", json={
        "run_id": "run3",
        "chat": {"id": "c1", "model": "m", "server_url": "http://x",
                 "messages": [{"role": "user", "content": "hi"}]}})
    r = client.post("/api/runs/run3/rerun", json={"index": 99})
    assert r.status_code == 400
    assert "out of range" in r.get_json()["error"]
