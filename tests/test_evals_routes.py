#!/usr/bin/env python3
"""Integration tests for the Evaluation tab's SSE routes (/api/evals/*).

These drive the real Flask app against a throwaway data+settings tree, with a fake
provider adapter standing in for a model server. No network, no Ollama, no touching
the user's profiles.

They exist for the failure modes that only show up once the route's generator is
actually run:

* a generation that errors must be reported ungraded, not handed to the grader —
  otherwise the grader scores the error text and that invented number lands in the
  average;
* every stream must end with a ``done`` frame even when it fails, because that is
  what re-enables the Run buttons in the browser.
"""

import pytest

from app import providers        # use_per_server patches the same seam use_adapter does

from conftest import all_of, events, first, sse_frames, use_adapter


# --------------------------- harness ---------------------------
# `client` and the SSE helpers come from tests/conftest.py. FakeAdapter stays here:
# routing on the grader marker is specific to this suite's two-model runs.

GRADER_MARKER = "=== RESPONSE TO GRADE ==="


class FakeAdapter:
    """Stands in for a provider client.

    `gen` maps a substring of a *generation* prompt to a reply string or an Exception
    to raise; `grade` is the grader's reply (a string, or a callable taking the graded
    response). The two are kept apart deliberately: a grader prompt quotes the whole
    generation prompt inside it, so a single substring table would let a generation
    rule swallow the grading call.
    """

    def __init__(self, gen=None, grade="{}", models=("test-model",)):
        self.gen = gen or {}
        self.grade = grade
        self._models = list(models)
        self.gen_seen = []          # generation prompts, in order
        self.grade_seen = []        # grader prompts, in order

    def list_models(self):
        return list(self._models)

    def model_capabilities(self, model):
        return []       # not a reasoning model, no tool support

    def chat_stream(self, model, messages, options, stop_event, think=False,
                    tools=None, tool_executor=None):
        prompt = "\n".join(m.get("content", "") for m in messages)
        if GRADER_MARKER in prompt:
            self.grade_seen.append(prompt)
            graded = prompt.split(GRADER_MARKER, 1)[1].split("===", 1)[0].strip()
            reply = self.grade(graded) if callable(self.grade) else self.grade
            if isinstance(reply, Exception):
                raise reply
            yield ("content", reply)
            return
        self.gen_seen.append(prompt)
        for needle, reply in self.gen.items():
            if needle in prompt:
                if isinstance(reply, Exception):
                    raise reply
                yield ("content", reply)
                return
        yield ("content", "default reply")


PROJECT = {
    "id": "t1", "name": "T",
    "columns": ["Input", "Response"],
    "rows": [{"Input": "one", "Response": ""}, {"Input": "two", "Response": ""}],
    "input_columns": ["Input"],
    "output_column": "Response",
    "prompt_template": "Answer: {Input}",
    "gen_server_url": "http://localhost:11434",
    "gen_model": "test-model",
    "grader_server_url": "http://localhost:11434",
    "grader_model": "test-model",
    "criteria": [{"label": "Accuracy", "guidance": "right?",
                  "mode": "score", "min": 1, "max": 10}],
    "batch_models": [],
}


# --------------------------- happy path ---------------------------

def test_run_grades_every_row_and_reports_an_aggregate(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter(
        gen={"Answer:": "a fine answer"},
        grade='{"Accuracy": {"score": 5, "reasoning": "ok"}}'))
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r1"}))
    rows = all_of(frames, "row_result")
    assert len(rows) == 2
    assert all(r["grades"]["Accuracy"]["score"] == 5.0 for r in rows)

    agg = first(frames, "model_done")["aggregate"]
    assert agg["overall"] == 50.0          # 5/10 reads 50%, not the old 44%
    assert agg["graded"] == 2 and agg["total"] == 2
    assert events(frames)[-1] == "done"


def test_run_streams_the_generated_response_back_for_write_back(client, monkeypatch):
    """The client fills the output column from these — they must carry real text."""
    use_adapter(monkeypatch, FakeAdapter(gen={"Answer:": "generated text"},
                                         grade='{"Accuracy": 9}'))
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r2"}))
    assert [r["response"] for r in all_of(frames, "row_result")] == \
        ["generated text", "generated text"]


def test_run_fills_the_template_per_row(client, monkeypatch):
    adapter = FakeAdapter(grade='{"Accuracy": 9}')
    use_adapter(monkeypatch, adapter)
    # The body has to be consumed: stream_with_context is lazy, so the route's
    # generator never runs until something reads it.
    sse_frames(client.post("/api/evals/run", json={"eval": PROJECT, "run_id": "r3"}))
    assert "Answer: one" in adapter.gen_seen[0]
    assert "Answer: two" in adapter.gen_seen[1]


def test_the_grader_sees_the_response_that_was_generated(client, monkeypatch):
    adapter = FakeAdapter(gen={"Answer: one": "first answer",
                               "Answer: two": "second answer"},
                          grade='{"Accuracy": 9}')
    use_adapter(monkeypatch, adapter)
    sse_frames(client.post("/api/evals/run", json={"eval": PROJECT, "run_id": "r3b"}))
    assert "first answer" in adapter.grade_seen[0]
    assert "second answer" in adapter.grade_seen[1]


# --------------------------- failed generation ---------------------------

def test_a_failed_generation_is_ungraded_not_scored(client, monkeypatch):
    """The regression this guards: the sequential path used to turn an error into the
    string "[Generation error: ...]", which is non-empty, so it flowed on to the
    grader and the grader's score for that error text entered the average."""
    adapter = FakeAdapter(
        gen={"Answer: one": RuntimeError("connection refused"),
             "Answer: two": "a fine answer"},
        grade='{"Accuracy": {"score": 10}}')
    use_adapter(monkeypatch, adapter)
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r4"}))

    rows = all_of(frames, "row_result")
    assert rows[0]["ungraded"] is True
    assert rows[0]["response"] == ""
    assert "connection refused" in rows[0]["error"]
    assert rows[0]["grades"]["Accuracy"]["score"] is None

    # The failed row was never sent to the grader.
    assert len(adapter.grade_seen) == 1
    assert "connection refused" not in adapter.grade_seen[0]
    assert "Generation error" not in adapter.grade_seen[0]

    # ...and it doesn't dilute the score, but it is visible as an ungraded row.
    agg = first(frames, "model_done")["aggregate"]
    assert agg["overall"] == 100.0
    assert agg["graded"] == 1 and agg["total"] == 2


def test_an_unparseable_grader_reply_leaves_the_row_ungraded(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter(gen={"Answer:": "a fine answer"},
                                         grade="I'd rather not answer in JSON."))
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r5"}))
    agg = first(frames, "model_done")["aggregate"]
    assert agg["overall"] is None
    assert agg["graded"] == 0 and agg["total"] == 2


def test_a_grader_that_raises_does_not_abort_the_run(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter(gen={"Answer:": "a fine answer"},
                                         grade=RuntimeError("grader offline")))
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r5b"}))
    assert len(all_of(frames, "row_result")) == 2      # every row still reported
    assert first(frames, "model_done")["aggregate"]["graded"] == 0
    assert events(frames)[-1] == "done"


def test_an_out_of_range_grader_score_is_clamped(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter(gen={"Answer:": "a fine answer"},
                                         grade='{"Accuracy": 99}'))
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r6"}))
    agg = first(frames, "model_done")["aggregate"]
    assert agg["per_criterion"]["Accuracy"]["avg"] == 10.0
    assert agg["overall"] == 100.0


# --------------------------- stream teardown ---------------------------

def test_every_stream_ends_with_done_even_when_it_errors(client, monkeypatch):
    """The browser re-enables the Run buttons on `done`. An error frame with no `done`
    after it left the tab wedged until a page reload."""
    class Exploding(FakeAdapter):
        def chat_stream(self, *a, **kw):
            raise RuntimeError("boom")
            yield  # pragma: no cover - generator marker

    use_adapter(monkeypatch, Exploding())
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": PROJECT, "run_id": "r7"}))
    assert events(frames)[-1] == "done"


def test_gen_data_stream_ends_with_done_when_it_errors(client, monkeypatch):
    class Exploding(FakeAdapter):
        def chat_stream(self, *a, **kw):
            raise RuntimeError("boom")
            yield  # pragma: no cover - generator marker

    use_adapter(monkeypatch, Exploding())
    project = {**PROJECT, "gen_instructions": {"Input": "a word"}}
    frames = sse_frames(client.post("/api/evals/gen-data",
                                    json={"eval": project, "run_id": "r8", "num_rows": 1}))
    assert events(frames)[-1] == "done"


# --------------------------- gen-data ---------------------------

def test_gen_data_marks_rows_whose_json_did_not_parse(client, monkeypatch):
    """`ok: False` is what stops the client appending a mysterious blank row."""
    use_adapter(monkeypatch, FakeAdapter(gen={"synthetic test data": "not json"}))
    project = {**PROJECT, "gen_instructions": {"Input": "a word"}}
    frames = sse_frames(client.post("/api/evals/gen-data",
                                    json={"eval": project, "run_id": "r9", "num_rows": 2}))
    rows = all_of(frames, "row_result")
    assert len(rows) == 2
    assert all(r["ok"] is False for r in rows)


def test_gen_data_returns_indexed_rows(client, monkeypatch):
    """The client places rows by index; under parallel lanes they arrive out of order."""
    use_adapter(monkeypatch, FakeAdapter(gen={"synthetic test data": '{"Input": "hello"}'}))
    project = {**PROJECT, "gen_instructions": {"Input": "a word"}}
    frames = sse_frames(client.post("/api/evals/gen-data",
                                    json={"eval": project, "run_id": "r10", "num_rows": 3}))
    rows = all_of(frames, "row_result")
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert all(r["ok"] is True and r["row"]["Input"] == "hello" for r in rows)


def test_gen_data_refuses_without_an_instruction(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter())
    r = client.post("/api/evals/gen-data",
                    json={"eval": PROJECT, "run_id": "r11", "num_rows": 2})
    assert r.status_code == 400


# --------------------------- validation ---------------------------

@pytest.mark.parametrize("patch,expected", [
    ({"rows": []}, "no rows"),
    ({"prompt_template": "  "}, "prompt"),
    ({"criteria": []}, "criterion"),
    ({"gen_model": "", "grader_model": ""}, "grader"),
])
def test_run_rejects_an_incomplete_project(client, monkeypatch, patch, expected):
    use_adapter(monkeypatch, FakeAdapter())
    r = client.post("/api/evals/run", json={"eval": {**PROJECT, **patch}, "run_id": "x"})
    assert r.status_code == 400
    assert expected in r.get_json()["error"].lower()


def test_batch_run_needs_at_least_one_model(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter())
    r = client.post("/api/evals/run",
                    json={"eval": PROJECT, "run_id": "x", "batch": True})
    assert r.status_code == 400


# --------------------------- batch ---------------------------

def test_batch_reports_the_server_that_actually_ran_each_model(client, monkeypatch):
    use_adapter(monkeypatch, FakeAdapter(gen={"Answer:": "a fine answer"},
                                         grade='{"Accuracy": 8}',
                                         models=("test-model", "other-model")))
    project = {**PROJECT, "batch_models": [
        {"server_url": "http://a:1", "model": "test-model"},
        {"server_url": "http://b:2", "model": "other-model"},
    ]}
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": project, "run_id": "r12", "batch": True}))
    done = all_of(frames, "model_done")
    assert len(done) == 2
    # Parallel processing is off in a fresh config, so each model runs on its own
    # server -- the label must be that server, not a global lane.
    assert done[0]["server"] == "http://a:1" and done[0]["model"] == "test-model"
    assert done[1]["server"] == "http://b:2" and done[1]["model"] == "other-model"

    summary = first(frames, "summary")
    assert [m["model"] for m in summary["models"]] == ["test-model", "other-model"]


# --------------------------- parallel lane selection ---------------------------

def enable_parallel(client, lanes):
    """Turn on Parallel Processing with the given [{base_url, model}] lane list."""
    r = client.put("/api/parallel/config", json={"parallel_enabled": True,
                                                 "parallel_servers": lanes,
                                                 "parallel_mode": "balanced"})
    assert r.status_code == 200, r.get_data(as_text=True)


class PerServerAdapter(FakeAdapter):
    """One adapter shared by every server, but each server reports its own model list
    and every generation records which server it ran on."""

    def __init__(self, installed, **kw):
        super().__init__(**kw)
        self.installed = installed          # {base_url: [models]}
        self.current = None
        self.ran_on = []

    def for_server(self, url):
        self.current = url
        return self

    def list_models(self):
        return list(self.installed.get(self.current, []))

    def chat_stream(self, model, messages, options, stop_event, think=False, **kw):
        prompt = "\n".join(m.get("content", "") for m in messages)
        if GRADER_MARKER not in prompt:
            self.ran_on.append(self.current)
        yield from super().chat_stream(model, messages, options, stop_event, think=think, **kw)


def use_per_server(monkeypatch, adapter):
    monkeypatch.setattr(providers, "get_client",
                        lambda server: adapter.for_server(server.get("base_url")))


def test_parallel_skips_servers_that_do_not_have_the_model(client, monkeypatch):
    """The batch entry's server choice is honoured: rows fan only across lanes that
    actually host the model, instead of being forced onto every configured lane."""
    adapter = PerServerAdapter(
        installed={"http://a:1": ["test-model"],
                   "http://b:2": ["something-else"],
                   "http://c:3": ["test-model"]},
        gen={"Answer:": "ok"}, grade='{"Accuracy": 8}')
    use_per_server(monkeypatch, adapter)
    enable_parallel(client, [{"base_url": "http://b:2"}, {"base_url": "http://c:3"}])

    project = {**PROJECT, "gen_server_url": "http://a:1"}
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": project, "run_id": "p1"}))

    assert "http://b:2" not in adapter.ran_on        # never had the model
    assert set(adapter.ran_on) <= {"http://a:1", "http://c:3"}
    assert first(frames, "model_done")["servers"] == ["http://a:1", "http://c:3"]


def test_parallel_falls_back_to_sequential_when_only_one_server_qualifies(client, monkeypatch):
    adapter = PerServerAdapter(
        installed={"http://a:1": ["test-model"], "http://b:2": ["something-else"]},
        gen={"Answer:": "ok"}, grade='{"Accuracy": 8}')
    use_per_server(monkeypatch, adapter)
    enable_parallel(client, [{"base_url": "http://b:2"}])

    project = {**PROJECT, "gen_server_url": "http://a:1"}
    frames = sse_frames(client.post("/api/evals/run",
                                    json={"eval": project, "run_id": "p2"}))
    assert first(frames, "model_done")["servers"] == ["http://a:1"]
    assert set(adapter.ran_on) == {"http://a:1"}


# --------------------------- CRUD ---------------------------

def test_upsert_then_fetch_round_trips_the_project(client):
    r = client.post("/api/evals", json={"eval": {**PROJECT, "id": ""}})
    assert r.status_code == 200
    ev = r.get_json()["eval"]
    assert ev["id"]                                   # a fresh id was minted
    got = client.get(f"/api/evals/{ev['id']}").get_json()["eval"]
    assert got["prompt_template"] == PROJECT["prompt_template"]
    assert got["rows"] == PROJECT["rows"]


def test_delete_removes_it_from_the_listing(client):
    ev = client.post("/api/evals", json={"eval": {**PROJECT, "id": ""}}).get_json()["eval"]
    left = client.delete(f"/api/evals/{ev['id']}").get_json()["evals"]
    assert all(e["id"] != ev["id"] for e in left)
    assert client.get(f"/api/evals/{ev['id']}").status_code == 404
