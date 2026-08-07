#!/usr/bin/env python3
"""Batch tab: prompt templating, filename resolution, source resolution, and the
/api/batch/* routes.

The destructive case this suite exists for is ``beside_source``: writing the AI's
answer next to the file it just read is one bad path away from overwriting the user's
documents. Those guards are tested from both ends — the pure function and the route.
"""

import json

import pytest

from app import batch, ingest, youtube
from tests.conftest import all_of, events, first, sse_frames, use_adapter, StubAdapter


# --------------------------- render_prompt ---------------------------

def _item(**kw):
    base = {"item_id": "i0", "title": "A Title", "content": "BODY TEXT",
            "kind": "folder", "source_path": "", "source_url": "", "chars": 9}
    base.update(kw)
    return base


def test_render_prompt_substitutes_placeholders():
    out = batch.render_prompt("Summarise {{title}} from {{url}}:\n{{content}}",
                              _item(source_url="http://x/y"))
    assert out == "Summarise A Title from http://x/y:\nBODY TEXT"


def test_render_prompt_is_case_and_space_tolerant():
    assert batch.render_prompt("{{ CONTENT }}", _item()) == "BODY TEXT"


def test_render_prompt_appends_content_when_no_placeholder():
    """The plain 'instruction, then the document' case must work with no placeholder,
    which is what makes the old pre-prompt behaviour the natural fallback."""
    assert batch.render_prompt("Summarise this.", _item()) == "Summarise this.\n\nBODY TEXT"


def test_render_prompt_empty_template_is_just_the_content():
    assert batch.render_prompt("", _item()) == "BODY TEXT"


def test_render_prompt_source_prefers_path_then_url():
    # No {{content}} here, so the fallback also appends the body — assert the
    # substitution itself rather than the whole string.
    assert batch.render_prompt("{{source}}|{{content}}",
                               _item(source_path="C:/a/b.txt")) == "C:/a/b.txt|BODY TEXT"
    assert batch.render_prompt("{{source}}|{{content}}",
                               _item(source_url="http://u")) == "http://u|BODY TEXT"


# --------------------------- sanitize_filename ---------------------------

@pytest.mark.parametrize("raw, expected", [
    ('a/b\\c:d*e?f"g<h>i|j', "a b c d e f g h i j"),
    ("  spaced   out  ", "spaced out"),
    ("trailing dots...", "trailing dots"),
    (".hidden", "hidden"),
    ("", "untitled"),
    ("   ", "untitled"),
])
def test_sanitize_filename(raw, expected):
    assert batch.sanitize_filename(raw) == expected


def test_sanitize_filename_escapes_windows_device_names():
    """CON.md is unopenable on Windows regardless of extension."""
    assert batch.sanitize_filename("CON") == "CON_"
    assert batch.sanitize_filename("lpt1") == "lpt1_"


def test_sanitize_filename_strips_newlines_from_an_llm_answer():
    assert batch.sanitize_filename("Title\nwith a second line") == "Title with a second line"


def test_sanitize_filename_caps_length():
    assert len(batch.sanitize_filename("x" * 400)) <= 120


# --------------------------- unique_path ---------------------------

def test_unique_path_avoids_clobbering(tmp_path):
    p = tmp_path / "a.md"
    assert batch.unique_path(p) == p
    p.write_text("first", encoding="utf-8")
    second = batch.unique_path(p)
    assert second.name == "a (2).md"
    second.write_text("second", encoding="utf-8")
    assert batch.unique_path(p).name == "a (3).md"
    assert p.read_text(encoding="utf-8") == "first"


# --------------------------- resolve_output_path ---------------------------

def _project(**kw):
    p = batch.new_project()
    p.update(kw)
    return p


def test_output_path_uses_source_stem_by_default(tmp_path):
    proj = _project(output_dir=str(tmp_path), ext=".md")
    item = _item(source_path=str(tmp_path / "report.pdf"), title="Embedded Title")
    assert batch.resolve_output_path(item, "", proj).name == "report.md"


def test_output_path_uses_llm_title_when_asked(tmp_path):
    proj = _project(output_dir=str(tmp_path), name_mode="llm", ext=".txt")
    item = _item(source_path=str(tmp_path / "report.pdf"))
    assert batch.resolve_output_path(item, "Quarterly Sales Summary", proj).name \
        == "Quarterly Sales Summary.txt"


def test_output_path_falls_back_to_source_name_when_llm_title_is_blank(tmp_path):
    proj = _project(output_dir=str(tmp_path), name_mode="llm")
    item = _item(source_path=str(tmp_path / "report.pdf"))
    assert batch.resolve_output_path(item, "   ", proj).name == "report.md"


def test_output_path_applies_prefix_and_suffix(tmp_path):
    proj = _project(output_dir=str(tmp_path), name_prefix="ai_", name_suffix="_v2")
    item = _item(source_path=str(tmp_path / "report.pdf"))
    assert batch.resolve_output_path(item, "", proj).name == "ai_report_v2.md"


def test_output_path_normalises_ext_without_a_dot(tmp_path):
    proj = _project(output_dir=str(tmp_path), ext="txt")
    assert batch.resolve_output_path(_item(title="x"), "", proj).suffix == ".txt"


def test_output_path_requires_an_output_dir():
    with pytest.raises(batch.BatchError, match="output folder"):
        batch.resolve_output_path(_item(title="x"), "", _project(output_dir=""))


# ---- the destructive-overwrite guards ----

def test_beside_source_refuses_without_prefix_or_suffix(tmp_path):
    """Without an append value the output path IS the input path."""
    src = tmp_path / "doc.md"
    src.write_text("original", encoding="utf-8")
    proj = _project(export_mode="beside_source", name_prefix="", name_suffix="")
    with pytest.raises(batch.BatchError, match="prefix or suffix"):
        batch.resolve_output_path(_item(source_path=str(src)), "", proj)


def test_beside_source_writes_next_to_the_original(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    src = sub / "doc.md"
    src.write_text("original", encoding="utf-8")
    proj = _project(export_mode="beside_source", name_suffix="_ai", ext=".md")
    out = batch.resolve_output_path(_item(source_path=str(src)), "", proj)
    assert out == sub / "doc_ai.md"
    assert src.read_text(encoding="utf-8") == "original"


def test_beside_source_refuses_when_the_name_collides_with_the_source(tmp_path):
    """An LLM title can land exactly on the source name even with a suffix set."""
    src = tmp_path / "doc_ai.md"
    src.write_text("original", encoding="utf-8")
    proj = _project(export_mode="beside_source", name_suffix="_ai",
                    name_mode="llm", ext=".md")
    with pytest.raises(batch.BatchError, match="overwritten"):
        batch.resolve_output_path(_item(source_path=str(src)), "doc", proj)


def test_beside_source_refuses_for_an_item_with_no_file(tmp_path):
    proj = _project(export_mode="beside_source", name_suffix="_ai")
    item = _item(kind="youtube", source_url="http://y", source_path="")
    with pytest.raises(batch.BatchError, match="no file on disk"):
        batch.resolve_output_path(item, "", proj)


# --------------------------- export bodies ---------------------------

def test_export_body_defaults_to_the_response_alone():
    proj = _project()
    assert batch.render_export_body(_item(), "PROMPT", "ANSWER", proj) == "ANSWER"


def test_export_body_can_include_source_and_prompt():
    proj = _project(include_source=True, include_prompt=True)
    body = batch.render_export_body(_item(source_url="http://u"), "PROMPT", "ANSWER", proj)
    assert "BODY TEXT" in body and "PROMPT" in body and "ANSWER" in body
    assert body.index("BODY TEXT") < body.index("PROMPT") < body.index("ANSWER")


def test_export_body_strip_markdown_toggle():
    proj = _project(strip_markdown=True)
    out = batch.render_export_body(_item(), "p", "# Heading\n\n**bold**", proj)
    assert "**" not in out and "#" not in out


def test_write_item_and_combined(tmp_path):
    proj = _project(output_dir=str(tmp_path))
    p1 = batch.write_item(_item(title="One"), "p", "first", proj)
    assert p1.read_text(encoding="utf-8") == "first"

    results = [{"item": _item(title="One"), "prompt": "p", "response": "first"},
               {"item": _item(title="Two"), "prompt": "p", "response": "second"}]
    combined = batch.write_combined(results, proj)
    text = combined.read_text(encoding="utf-8")
    assert "# One" in text and "# Two" in text
    assert text.index("first") < text.index("second")


# --------------------------- validate_project ---------------------------

def test_validate_requires_model_and_sources():
    errs = " ".join(batch.validate_project(batch.new_project()))
    assert "model" in errs.lower() and "input source" in errs.lower()


def test_validate_flags_beside_source_without_append(tmp_path):
    proj = _project(model="m", output_mode="files", export_mode="beside_source",
                    sources=[{"kind": "folder", "path": str(tmp_path)}])
    assert any("prefix or append" in e or "prefix or suffix" in e
               for e in batch.validate_project(proj))


def test_validate_flags_beside_source_with_no_folder_input():
    proj = _project(model="m", output_mode="files", export_mode="beside_source",
                    name_suffix="_ai",
                    sources=[{"kind": "search", "query": "q"}])
    assert any("only works for folder" in e for e in batch.validate_project(proj))


def test_validate_flags_a_missing_folder():
    proj = _project(model="m", output_mode="chat",
                    sources=[{"kind": "folder", "path": "Z:/definitely/not/here"}])
    assert any("does not exist" in e for e in batch.validate_project(proj))


def test_validate_passes_a_sane_project(tmp_path):
    proj = _project(model="m", output_mode="files", export_mode="per_item",
                    output_dir=str(tmp_path),
                    sources=[{"kind": "folder", "path": str(tmp_path)}])
    assert batch.validate_project(proj) == []


# --------------------------- resolve_sources ---------------------------

def test_resolve_folder_source_non_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("gamma", encoding="utf-8")

    proj = _project(sources=[{"kind": "folder", "path": str(tmp_path), "recursive": False}])
    items, errors = batch.resolve_sources(proj)
    assert sorted(i["title"] for i in items) == ["a", "b"]
    assert all(i["source_path"] for i in items)


def test_resolve_folder_source_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    (sub / "c.txt").write_text("gamma", encoding="utf-8")

    proj = _project(sources=[{"kind": "folder", "path": str(tmp_path), "recursive": True}])
    items, _ = batch.resolve_sources(proj)
    assert sorted(i["title"] for i in items) == ["a", "c"]


def test_resolve_folder_source_ignores_unreadable_types(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    proj = _project(sources=[{"kind": "folder", "path": str(tmp_path), "recursive": False}])
    items, _ = batch.resolve_sources(proj)
    assert [i["title"] for i in items] == ["a"]


def test_resolve_truncates_long_items(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
    proj = _project(max_item_chars=100,
                    sources=[{"kind": "folder", "path": str(tmp_path)}])
    items, _ = batch.resolve_sources(proj)
    assert items[0]["chars"] < 200 and "truncated" in items[0]["content"]


def test_resolve_item_ids_are_unique_across_same_named_files(tmp_path):
    """Two files called report.txt in different subfolders must not collide — the
    parallel engine keys its item map on item_id."""
    for sub in ("one", "two"):
        d = tmp_path / sub
        d.mkdir()
        (d / "report.txt").write_text(sub, encoding="utf-8")
    proj = _project(sources=[{"kind": "folder", "path": str(tmp_path), "recursive": True}])
    items, _ = batch.resolve_sources(proj)
    assert len(items) == 2
    assert len({i["item_id"] for i in items}) == 2


def test_resolve_search_source(monkeypatch):
    def fake_crawl(query, sites=None, max_results=5, should_stop=None):
        yield {"type": "progress", "done": 1, "target": 2, "url": "http://a", "ok": True}
        yield {"type": "result", "attempted": 2, "errors": ["http://c: boom"],
               "pages": [{"title": "Page A", "url": "http://a", "text": "aaa"},
                         {"title": "Page B", "url": "http://b", "text": "bbb"}]}
    monkeypatch.setattr(batch.core, "crawl_search", fake_crawl)

    seen = []
    proj = _project(sources=[{"kind": "search", "query": "q", "max_results": 2}])
    items, errors = batch.resolve_sources(proj, emit=lambda e, d: seen.append((e, d)))
    assert [i["title"] for i in items] == ["Page A", "Page B"]
    assert [i["source_url"] for i in items] == ["http://a", "http://b"]
    assert errors == ["http://c: boom"]
    assert any(e == "progress" for e, _ in seen)


def test_resolve_search_clamps_max_results(monkeypatch):
    """``core.crawl_search`` only bounds this from below, and the field's `max`
    attribute is not enforced against a typed value — so a mistyped 500 was 500 page
    fetches against a metered API. The Batch ceiling is deliberately higher than the
    chat/Resources one (bulk work is the point), but it is still a ceiling."""
    asked = []

    def fake_crawl(query, sites=None, max_results=5, should_stop=None):
        asked.append(max_results)
        yield {"type": "result", "attempted": 0, "errors": [], "pages": []}
    monkeypatch.setattr(batch.core, "crawl_search", fake_crawl)

    for typed, expected in [(500, batch.core.MAX_BATCH_CRAWL_PAGES),
                            (0, 5),      # falsy = unset, so the shipped default
                            (-3, 1),     # a real value, clamped up to the floor
                            (7, 7)]:
        batch.resolve_sources(
            _project(sources=[{"kind": "search", "query": "q", "max_results": typed}]))
        assert asked[-1] == expected, f"typed {typed}"
    assert batch.core.MAX_BATCH_CRAWL_PAGES > batch.core.MAX_CRAWL_PAGES


def test_resolve_youtube_source(monkeypatch):
    monkeypatch.setattr(batch.youtube, "fetch_video", lambda url, **kw: {
        "url": url, "title": "Vid " + url[-1], "text": "transcript " + url[-1]})
    proj = _project(sources=[{"kind": "youtube", "urls": "http://y/1\n\nhttp://y/2\n"}])
    items, _ = batch.resolve_sources(proj)
    assert [i["title"] for i in items] == ["Vid 1", "Vid 2"]
    assert [i["content"] for i in items] == ["transcript 1", "transcript 2"]


def test_resolve_playlist_source(monkeypatch):
    monkeypatch.setattr(batch.youtube, "fetch_playlist", lambda url, **kw: [
        {"video_id": "a" * 11, "url": "http://y/1", "title": "First"},
        {"video_id": "b" * 11, "url": "http://y/2", "title": "Second"}])
    monkeypatch.setattr(batch.youtube, "fetch_video", lambda url, **kw: {
        "url": url, "title": "", "text": "body " + url[-1]})
    proj = _project(sources=[{"kind": "playlist", "url": "http://p", "limit": 0}])
    items, _ = batch.resolve_sources(proj)
    assert [i["content"] for i in items] == ["body 1", "body 2"]


def test_resolve_one_bad_video_does_not_lose_the_others(monkeypatch):
    def flaky(url, **kw):
        if url.endswith("2"):
            raise youtube.YouTubeError("no captions")
        return {"url": url, "title": "ok", "text": "fine"}
    monkeypatch.setattr(batch.youtube, "fetch_video", flaky)
    proj = _project(sources=[{"kind": "youtube", "urls": "http://y/1\nhttp://y/2\nhttp://y/3"}])
    items, errors = batch.resolve_sources(proj)
    assert len(items) == 2
    assert any("no captions" in e for e in errors)


def test_resolve_mixes_source_kinds(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    monkeypatch.setattr(batch.youtube, "fetch_video", lambda url, **kw: {
        "url": url, "title": "Vid", "text": "transcript"})
    proj = _project(sources=[
        {"kind": "folder", "path": str(tmp_path)},
        {"kind": "youtube", "urls": "http://y/1"},
    ])
    items, _ = batch.resolve_sources(proj)
    assert [i["kind"] for i in items] == ["folder", "youtube"]


# --------------------------- routes ---------------------------

def test_batch_project_crud(client):
    r = client.post("/api/batch/projects", json={"name": "My batch", "sources": []})
    assert r.status_code == 200
    pid = r.get_json()["project"]["id"]
    assert pid

    assert client.get("/api/batch/projects").get_json()["projects"][0]["name"] == "My batch"
    assert client.get(f"/api/batch/projects/{pid}").get_json()["project"]["name"] == "My batch"

    client.post("/api/batch/projects", json={"id": pid, "name": "Renamed", "sources": []})
    assert client.get(f"/api/batch/projects/{pid}").get_json()["project"]["name"] == "Renamed"

    assert client.delete(f"/api/batch/projects/{pid}").get_json()["ok"] is True
    assert client.get(f"/api/batch/projects/{pid}").status_code == 404


def test_state_exposes_batch_defaults(client):
    st = client.get("/api/state").get_json()
    assert st["batch_projects"] == []
    assert st["default_batch_project"]["name_prompt"] == batch.DEFAULT_FILENAME_PROMPT
    assert "folder" in st["batch_source_kinds"]
    assert ".pdf" in st["batch_exts"]


def test_batch_run_rejects_an_invalid_project(client):
    r = client.post("/api/batch/run", json={"project": batch.new_project()})
    assert r.status_code == 400
    assert "model" in r.get_json()["error"].lower()


def test_batch_run_rejects_beside_source_without_append(client, tmp_path):
    proj = _project(model="test-model", output_mode="files",
                    export_mode="beside_source",
                    sources=[{"kind": "folder", "path": str(tmp_path)}])
    r = client.post("/api/batch/run", json={"project": proj})
    assert r.status_code == 400
    assert "overwrite" in r.get_json()["error"].lower()


def test_batch_preview_lists_items(client, tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    proj = _project(sources=[{"kind": "folder", "path": str(tmp_path)}])
    frames = sse_frames(client.post("/api/batch/preview", json={"project": proj}))
    items = first(frames, "items")
    assert items["total"] == 2
    assert sorted(i["title"] for i in items["items"]) == ["a", "b"]
    # The browser never needs the content, and shipping it would be wasteful.
    assert all("content" not in i for i in items["items"])


def test_batch_run_end_to_end_writes_one_file_per_item(client, monkeypatch, tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    (src / "b.txt").write_text("beta", encoding="utf-8")

    use_adapter(monkeypatch, StubAdapter(replies=["ANSWER A", "ANSWER B"]))
    proj = _project(model="test-model", pre_prompt="Summarise:\n{{content}}",
                    output_mode="files", export_mode="per_item",
                    output_dir=str(out), ext=".md",
                    sources=[{"kind": "folder", "path": str(src)}])

    frames = sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert "export" in events(frames)
    exp = first(frames, "export")
    assert exp["count"] == 2 and len(exp["files"]) == 2
    assert sorted(p.name for p in out.iterdir()) == ["a.md", "b.md"]
    assert {p.read_text(encoding="utf-8") for p in out.iterdir()} == {"ANSWER A", "ANSWER B"}
    # The originals are untouched.
    assert (src / "a.txt").read_text(encoding="utf-8") == "alpha"


def test_batch_run_renders_the_template_into_the_prompt(client, monkeypatch, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    stub = use_adapter(monkeypatch, StubAdapter(replies=["ok"]))
    proj = _project(model="test-model", system_on=True, system_prompt="SYS",
                    pre_prompt="Summarise {{title}}:\n{{content}}",
                    output_mode="chat",
                    sources=[{"kind": "folder", "path": str(src)}])
    sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert "Summarise a:\nalpha" in stub.prompt(0)
    assert stub.system(0) == "SYS"
    # The template must not also be appended a second time as a pre-prompt.
    assert stub.prompt(0).count("Summarise a:") == 1


def test_batch_multipass_uses_the_projects_evaluation_prompt(client, monkeypatch, tmp_path):
    """The Batch tab's Multi-Pass now has its own evaluation prompt, the way the chat
    tab does, so the user can say what the refinement pass should look for. The plumbing
    existed (new_project -> _batch_item_chat -> build_eval_messages); only the control
    was missing, so every batch refinement silently used the shipped default."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    stub = use_adapter(monkeypatch, StubAdapter(replies=["FIRST", "REFINED"]))
    proj = _project(model="test-model", output_mode="chat",
                    pre_prompt="Summarise:\n{{content}}",
                    multi_pass=True, passes=1,
                    eval_prompt="CHECK THE FACTS ONLY.\n[input prompt]\n[Response]",
                    sources=[{"kind": "folder", "path": str(src)}])
    frames = sse_frames(client.post("/api/batch/run", json={"project": proj}))

    assert stub.calls == 2, "one initial answer plus one refinement pass"
    refinement = stub.prompt(1)
    assert "CHECK THE FACTS ONLY." in refinement
    # The [tag] vocabulary is filled, not passed through literally.
    assert "[Response]" not in refinement and "FIRST" in refinement
    assert "[input prompt]" not in refinement and "Summarise:" in refinement
    assert first(frames, "item_done")["content"] == "REFINED"


def test_batch_multipass_falls_back_to_the_default_evaluation_prompt(
        client, monkeypatch, tmp_path):
    """An older saved project has eval_prompt="" — fill_eval_prompt has always fallen
    back to the shipped template, and that must keep working."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    stub = use_adapter(monkeypatch, StubAdapter(replies=["FIRST", "REFINED"]))
    proj = _project(model="test-model", output_mode="chat", pre_prompt="Do it:\n{{content}}",
                    multi_pass=True, passes=1, eval_prompt="",
                    sources=[{"kind": "folder", "path": str(src)}])
    sse_frames(client.post("/api/batch/run", json={"project": proj}))

    from app import logic
    marker = logic.DEFAULT_EVAL_PROMPT.split("[input prompt]")[0].strip()
    assert marker and marker in stub.prompt(1)


def test_batch_run_combined_export_keeps_source_order(client, monkeypatch, tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    for name in ("a", "b", "c"):
        (src / f"{name}.txt").write_text(name, encoding="utf-8")

    use_adapter(monkeypatch, StubAdapter(replies=["AAA", "BBB", "CCC"]))
    proj = _project(model="test-model", output_mode="files", export_mode="combined",
                    output_dir=str(out), combined_name="all",
                    sources=[{"kind": "folder", "path": str(src)}])
    frames = sse_frames(client.post("/api/batch/run", json={"project": proj}))
    combined = first(frames, "export")["combined"]
    assert combined
    text = open(combined, encoding="utf-8").read()
    assert text.index("# a") < text.index("# b") < text.index("# c")


def test_batch_run_beside_source_leaves_originals_intact(client, monkeypatch, tmp_path):
    src = tmp_path / "in" / "nested"
    src.mkdir(parents=True)
    (src / "doc.txt").write_text("ORIGINAL", encoding="utf-8")

    use_adapter(monkeypatch, StubAdapter(replies=["PROCESSED"]))
    proj = _project(model="test-model", output_mode="files",
                    export_mode="beside_source", name_suffix="_ai", ext=".md",
                    sources=[{"kind": "folder", "path": str(tmp_path / "in"),
                              "recursive": True}])
    sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert (src / "doc.txt").read_text(encoding="utf-8") == "ORIGINAL"
    assert (src / "doc_ai.md").read_text(encoding="utf-8") == "PROCESSED"


def test_batch_run_llm_naming_uses_a_second_call(client, monkeypatch, tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")

    stub = use_adapter(monkeypatch, StubAdapter(replies=["THE ANSWER", "Quarterly Report"]))
    proj = _project(model="test-model", output_mode="files", export_mode="per_item",
                    output_dir=str(out), name_mode="llm", ext=".md",
                    sources=[{"kind": "folder", "path": str(src)}])
    sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert stub.calls == 2                       # generation + naming
    assert (out / "Quarterly Report.md").read_text(encoding="utf-8") == "THE ANSWER"


def test_batch_run_naming_failure_falls_back_to_the_source_name(client, monkeypatch, tmp_path):
    """A naming hiccup must never cost the user the response."""
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")

    use_adapter(monkeypatch, StubAdapter(replies=["THE ANSWER", RuntimeError("naming died")]))
    proj = _project(model="test-model", output_mode="files", export_mode="per_item",
                    output_dir=str(out), name_mode="llm",
                    sources=[{"kind": "folder", "path": str(src)}])
    sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert (out / "a.md").read_text(encoding="utf-8") == "THE ANSWER"


def test_batch_run_chat_only_writes_nothing(client, monkeypatch, tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")

    use_adapter(monkeypatch, StubAdapter(replies=["ANSWER"]))
    proj = _project(model="test-model", output_mode="chat", output_dir=str(out),
                    sources=[{"kind": "folder", "path": str(src)}])
    frames = sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert first(frames, "export")["files"] == []
    assert list(out.iterdir()) == []
    assert first(frames, "item_done")["content"] == "ANSWER"


def test_batch_run_reports_no_readable_items(client, tmp_path):
    proj = _project(model="test-model", output_mode="chat",
                    sources=[{"kind": "folder", "path": str(tmp_path)}])
    frames = sse_frames(client.post("/api/batch/run", json={"project": proj}))
    assert any("No items" in d.get("message", "") for e, d in frames if e == "error")


def test_batch_projects_are_isolated_per_data_profile(client):
    """The four-place per-profile plumbing exists to stop exactly this leak: a new data
    file that forgets set_active_data_profile() writes to the legacy location and shows
    up in every profile."""
    pid = client.post("/api/batch/projects",
                      json={"name": "In profile A", "sources": []}).get_json()["project"]["id"]

    r = client.post("/api/profiles/data", json={"name": "Profile B"})
    assert r.status_code == 200, r.get_data(as_text=True)
    profiles = r.get_json()["profiles"]["data"]["profiles"]
    bid = next(p["id"] for p in profiles if p["name"] == "Profile B")
    aid = next(p["id"] for p in profiles if p["name"] != "Profile B")

    assert client.post(f"/api/profiles/data/{bid}/activate").status_code == 200
    assert client.get("/api/batch/projects").get_json()["projects"] == []
    assert client.get(f"/api/batch/projects/{pid}").status_code == 404

    assert client.post(f"/api/profiles/data/{aid}/activate").status_code == 200
    back = client.get("/api/batch/projects").get_json()["projects"]
    assert [p["name"] for p in back] == ["In profile A"]


def test_batch_projects_file_follows_the_active_profile(store, tmp_path):
    """core.BATCH_PROJECTS_FILE must be reassigned by set_active_data_profile, not
    left pointing at the legacy flat path."""
    from app import core
    other = tmp_path / "elsewhere"
    core.set_active_data_profile(other)
    try:
        assert core.BATCH_PROJECTS_FILE == other / "batch_projects.json"
    finally:
        core.set_active_data_profile(tmp_path / "data")


def test_old_chat_batch_route_still_exists(client, tmp_path):
    """The composer's 📂 Batch button is deliberately untouched by the Batch tab."""
    r = client.post("/api/batch/start", json={"chat": {}, "folder": str(tmp_path)})
    # No model in the chat dict -> its own 400, proving the route still routes.
    assert r.status_code == 400


# --------------------------- images ---------------------------
# A folder of pictures where each image IS the item, reference images sent with every
# item, and exporting whatever the model draws back.

pil = pytest.importorskip("PIL", reason="batch image tests need Pillow")


def _png(size=(80, 60), fmt="PNG"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buf, fmt)
    return buf.getvalue()


def _image_folder(tmp_path, names=("one.png", "two.jpg", "notes.txt")):
    d = tmp_path / "pics"
    d.mkdir()
    for n in names:
        if n.endswith(".txt"):
            (d / n).write_text("not an image")
        else:
            (d / n).write_bytes(_png(fmt="JPEG" if n.endswith(".jpg") else "PNG"))
    return d


def test_an_image_source_makes_one_item_per_picture(store, tmp_path):
    folder = _image_folder(tmp_path)
    items, errors = batch.resolve_sources(
        _project(sources=[{"kind": "images", "path": str(folder), "recursive": False}]))

    assert [i["title"] for i in items] == ["one", "two"]      # .txt ignored
    assert errors == []
    for i in items:
        assert i["content"] == ""            # the picture IS the input, not its text
        assert len(i["image_ids"]) == 1
        assert i["source_path"].endswith((".png", ".jpg"))


def test_an_image_item_gets_the_template_without_a_trailing_blank(store, tmp_path):
    """render_prompt appends {{content}} when the template has no placeholder; an
    image item has none, so appending would leave a dangling blank line."""
    folder = _image_folder(tmp_path, names=("one.png",))
    items, _ = batch.resolve_sources(
        _project(sources=[{"kind": "images", "path": str(folder)}]))
    assert batch.render_prompt("Describe this image.", items[0]) == "Describe this image."


def test_an_image_source_needs_a_folder_that_exists():
    problems = batch.validate_project(
        _project(model="m", sources=[{"kind": "images", "path": "/no/such/dir"}]))
    assert any("does not exist" in p for p in problems)
    problems = batch.validate_project(
        _project(model="m", sources=[{"kind": "images", "path": ""}]))
    assert any("no folder was chosen" in p for p in problems)


def test_beside_source_export_is_allowed_for_an_image_source(tmp_path):
    """It was rejected for everything but 'folder'; an image source has real files
    on disk too, so the guard has to know about it."""
    problems = batch.validate_project(_project(
        model="m", output_mode="files", export_mode="beside_source", name_suffix="_ai",
        sources=[{"kind": "images", "path": str(tmp_path)}]))
    assert not [p for p in problems if "only works for folder sources" in p]


def test_reference_images_and_the_item_image_both_reach_the_model(client, monkeypatch,
                                                                  tmp_path):
    import io
    folder = _image_folder(tmp_path, names=("one.png",))
    ref = client.post("/api/images/upload",
                      data={"files": (io.BytesIO(_png()), "style-guide.png")},
                      content_type="multipart/form-data").get_json()["images"][0]
    adapter = use_adapter(monkeypatch, StubAdapter(["described"] * 4))

    r = client.post("/api/batch/run", json={"project": _project(
        model="test-model", output_mode="chat", pre_prompt="Describe {{title}}",
        reference_images=[ref],
        sources=[{"kind": "images", "path": str(folder)}])})
    frames = sse_frames(r)
    assert "item_done" in events(frames), frames

    user = [m for m in adapter.seen[0][1] if m.get("role") == "user"][-1]
    assert user["content"] == "Describe one"
    assert len(user["images"]) == 2          # reference first, then the item's own
    assert all(im.get("b64") for im in user["images"])


def test_returned_images_are_written_beside_the_text_with_the_same_naming(
        client, monkeypatch, tmp_path):
    import base64
    folder = _image_folder(tmp_path, names=("one.png",))
    out = tmp_path / "out"
    out.mkdir()
    use_adapter(monkeypatch, StubAdapter([[
        ("content", "here it is"),
        ("image", {"b64": base64.b64encode(_png()).decode(), "media_type": "image/png"}),
    ]] * 4))

    frames = sse_frames(client.post("/api/batch/run", json={"project": _project(
        model="test-model", output_mode="files", export_mode="per_item",
        name_prefix="ai_", ext=".md", output_dir=str(out),
        pre_prompt="Describe {{title}}",
        sources=[{"kind": "images", "path": str(folder)}])}))

    written = sorted(p.name for p in out.iterdir())
    assert written == ["ai_one.md", "ai_one.png"]
    assert (out / "ai_one.md").read_text(encoding="utf-8") == "here it is"
    assert (out / "ai_one.png").read_bytes() == _png()
    assert first(frames, "export")["files"]


def test_an_unwritable_image_is_skipped_rather_than_raising(store, tmp_path):
    """write_item_images runs after the text export has already been written, so a
    picture that can't be placed must not cost the user the response."""
    from app import images
    out = tmp_path / "out"
    out.mkdir()
    proj = _project(output_dir=str(out), ext=".md")
    item = _item(source_path=str(tmp_path / "one.png"))

    # Nothing stored under this id.
    assert batch.write_item_images(item, [{"id": "img_" + "0" * 12,
                                           "media_type": "image/png"}], proj) == []
    # A record with no id at all.
    assert batch.write_item_images(item, [{"media_type": "image/png"}], proj) == []
    # beside_source with no prefix or suffix: resolve_output_path refuses.
    rec = images.store(_png(), "image/png")
    assert batch.write_item_images(
        item, [rec], _project(export_mode="beside_source")) == []
    assert list(out.iterdir()) == []


def test_several_returned_images_do_not_overwrite_each_other(store, tmp_path):
    from app import images
    out = tmp_path / "out"
    out.mkdir()
    proj = _project(output_dir=str(out), ext=".md")
    item = _item(source_path=str(tmp_path / "one.png"))
    recs = [images.store(_png(size=(10 + n, 10)), "image/png") for n in range(3)]

    paths = batch.write_item_images(item, recs, proj)
    assert [p.name for p in paths] == ["one.png", "one (2).png", "one (3).png"]


def test_the_batch_full_res_toggle_reaches_the_item_chat(client, monkeypatch, tmp_path):
    import base64
    import io
    from PIL import Image
    big = tmp_path / "pics"
    big.mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (3000, 1500), (10, 10, 10)).save(buf, "JPEG")
    (big / "wide.jpg").write_bytes(buf.getvalue())
    client.post("/api/settings", json={"image_max_dim": 512})

    for full_res, expected in ((False, 512), (True, 3000)):
        adapter = use_adapter(monkeypatch, StubAdapter(["ok"] * 4))
        sse_frames(client.post("/api/batch/run", json={"project": _project(
            model="test-model", output_mode="chat", pre_prompt="Describe",
            image_full_res=full_res,
            sources=[{"kind": "images", "path": str(big)}])}))
        sent = [m for m in adapter.seen[0][1] if m.get("role") == "user"][-1]
        img = Image.open(io.BytesIO(base64.b64decode(sent["images"][0]["b64"])))
        assert max(img.size) == expected, (full_res, img.size)


def test_a_preview_reports_the_image_ids_it_resolved(client, tmp_path):
    folder = _image_folder(tmp_path, names=("one.png", "two.png"))
    frames = sse_frames(client.post("/api/batch/preview", json={"project": _project(
        sources=[{"kind": "images", "path": str(folder)}])}))
    items = first(frames, "items")["items"]
    assert len(items) == 2
    assert all(len(i["image_ids"]) == 1 for i in items)
