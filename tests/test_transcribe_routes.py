#!/usr/bin/env python3
"""Route-level tests for local transcription (app/server.py).

``transcribe.transcribe_file``/``transcribe_url`` are monkeypatched throughout — the
engine itself is covered by test_transcribe.py, and what matters here is the SSE frame
sequence, the library append, and that one bad file doesn't cost the user the rest.

The native file picker is patched too: it shells out to a tkinter subprocess, which
would block a test run for its full 300-second timeout.
"""

import pytest

from app import core, transcribe
from conftest import all_of, events, first, sse_frames


@pytest.fixture
def fake_media(monkeypatch):
    """Stand in for the picker and the transcriber, recording what was asked for."""
    from app import native_dialog

    seen = {"picked": [], "transcribed": [], "urls": []}
    state = {"paths": [], "fail": set()}

    def pick(title="", filetypes_key=None):
        seen["picked"].append(filetypes_key)
        return list(state["paths"])

    def fake_file(path, settings=None, on_progress=None, should_stop=None):
        seen["transcribed"].append(str(path))
        if str(path) in state["fail"]:
            raise transcribe.TranscribeError("the decoder gave up")
        if on_progress:
            on_progress("whisper", done=30.0, total=60.0, unit="seconds", device="cuda")
        return {"text": f"transcript of {path}", "cues": [], "language": "en",
                "duration": 60.0, "device": "cuda", "compute_type": "float16",
                "model": "large-v3", "fallback": False, "stopped": False,
                "chars": len(f"transcript of {path}")}

    def fake_url(url, settings=None, on_progress=None, should_stop=None):
        seen["urls"].append(url)
        out = fake_file(url, settings=settings, on_progress=on_progress,
                        should_stop=should_stop)
        out["url"] = url
        return out

    monkeypatch.setattr(native_dialog, "pick_files", pick)
    monkeypatch.setattr(transcribe, "transcribe_file", fake_file)
    monkeypatch.setattr(transcribe, "transcribe_url", fake_url)
    seen["state"] = state
    return seen


def _media(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"stub")
        out.append(str(p))
    return out


# --------------------------- status ---------------------------

def test_status_reports_availability_without_loading_anything(client):
    body = client.get("/api/transcribe/status").get_json()
    assert set(body) >= {"available", "cuda_usable", "loaded", "model", "exts"}
    assert body["loaded"] is False
    assert ".mp3" in body["exts"]


def test_reset_returns_the_fresh_status(client):
    body = client.post("/api/transcribe/reset").get_json()
    assert body["loaded"] is False


# --------------------------- composer files ---------------------------

def test_files_route_emits_one_frame_per_file(client, tmp_path, fake_media):
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3", "b.m4a")
    frames = sse_frames(client.post("/api/transcribe/files"))
    assert events(frames)[:2] == ["start", "begin"]
    assert first(frames, "begin")["total"] == 2
    files = all_of(frames, "file")
    assert [f["name"] for f in files] == ["a.mp3", "b.m4a"]
    assert all(f["text"].startswith("transcript of") for f in files)
    assert first(frames, "complete")["ok"] == 2


def test_the_picker_is_filtered_to_media(client, tmp_path, fake_media):
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3")
    sse_frames(client.post("/api/transcribe/files"))
    assert fake_media["picked"] == ["media"]


def test_a_non_media_pick_is_reported_and_never_transcribed(client, tmp_path, fake_media):
    fake_media["state"]["paths"] = _media(tmp_path, "notes.pdf", "a.mp3")
    frames = sse_frames(client.post("/api/transcribe/files"))
    assert fake_media["transcribed"] == [str(tmp_path / "a.mp3")]
    assert any("notes.pdf" in e for e in first(frames, "complete")["errors"])


def test_one_bad_file_does_not_cost_the_others(client, tmp_path, fake_media):
    paths = _media(tmp_path, "a.mp3", "bad.mp3", "c.mp3")
    fake_media["state"]["paths"] = paths
    fake_media["state"]["fail"] = {paths[1]}
    frames = sse_frames(client.post("/api/transcribe/files"))
    assert [f["name"] for f in all_of(frames, "file")] == ["a.mp3", "c.mp3"]
    assert first(frames, "file_error")["name"] == "bad.mp3"
    done = first(frames, "complete")
    assert done["ok"] == 2 and len(done["errors"]) == 1


def test_progress_frames_carry_the_files_position(client, tmp_path, fake_media):
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3", "b.mp3")
    frames = sse_frames(client.post("/api/transcribe/files"))
    prog = all_of(frames, "progress")
    assert [p["index"] for p in prog] == [1, 2]
    assert all(p["count"] == 2 for p in prog)


def test_the_file_position_survives_a_whisper_frames_own_totals(client, tmp_path,
                                                                fake_media):
    """A whisper frame carries done/total in SECONDS. If the position used `total` too,
    the seconds would overwrite it and the UI would read "1 of 60 files"."""
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3")
    prog = all_of(sse_frames(client.post("/api/transcribe/files")), "progress")
    whisper = [p for p in prog if p["phase"] == "whisper"][0]
    assert whisper["count"] == 1          # one file
    assert whisper["total"] == 60.0       # sixty seconds of audio
    assert whisper["unit"] == "seconds"


def test_a_cancelled_pick_completes_cleanly(client, fake_media):
    fake_media["state"]["paths"] = []
    frames = sse_frames(client.post("/api/transcribe/files"))
    assert first(frames, "begin")["total"] == 0
    assert first(frames, "complete")["ok"] == 0


def test_the_run_id_is_unique_per_invocation(client, tmp_path, fake_media):
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3")
    a = first(sse_frames(client.post("/api/transcribe/files")), "start")["run_id"]
    b = first(sse_frames(client.post("/api/transcribe/files")), "start")["run_id"]
    assert a != b
    assert a.startswith("transcribe-chat-")


# --------------------------- library files ---------------------------

def test_library_media_route_appends_one_audio_item_per_file(client, tmp_path, fake_media):
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    fake_media["state"]["paths"] = _media(tmp_path, "ep1.mp3", "ep2.mp3")
    frames = sse_frames(client.post(f"/api/libraries/{lib['id']}/add-media-files"))
    done = first(frames, "complete")
    items = done["library"]["items"]
    assert [i["type"] for i in items] == ["audio", "audio"]
    assert [i["label"] for i in items] == ["ep1.mp3", "ep2.mp3"]
    # The path goes in filename, the way a document item's name does.
    assert items[0]["filename"] == str(tmp_path / "ep1.mp3")


def test_library_media_route_404s_on_a_missing_library(client, fake_media):
    assert client.post("/api/libraries/nope/add-media-files").status_code == 404


def test_library_media_run_id_names_the_library(client, tmp_path, fake_media):
    lib = client.post("/api/libraries", json={"name": "Shows"}).get_json()["library"]
    fake_media["state"]["paths"] = _media(tmp_path, "a.mp3")
    run = first(sse_frames(
        client.post(f"/api/libraries/{lib['id']}/add-media-files")), "start")["run_id"]
    assert run.startswith(f"transcribe-library-{lib['id']}-")


# --------------------------- url ---------------------------

def test_url_route_transcribes_and_returns_un_stored(client, fake_media):
    frames = sse_frames(client.get("/api/transcribe/url?url=https://x.test/ep-9.mp3"))
    done = first(frames, "complete")
    assert done["title"] == "ep-9.mp3"
    assert done["text"].startswith("transcript of")
    assert fake_media["urls"] == ["https://x.test/ep-9.mp3"]
    # Nothing was written anywhere.
    assert client.get("/api/libraries").get_json()["libraries"] == []


def test_url_route_needs_a_url(client):
    r = client.get("/api/transcribe/url")
    assert r.status_code == 400


def test_url_route_reports_a_failure_as_an_error_not_a_dead_stream(client, fake_media,
                                                                  monkeypatch):
    def boom(*a, **k):
        raise transcribe.TranscribeError("pip install faster-whisper")

    monkeypatch.setattr(transcribe, "transcribe_url", boom)
    frames = sse_frames(client.get("/api/transcribe/url?url=https://x.test/a.mp3"))
    assert "pip install faster-whisper" in first(frames, "file_error")["message"]
    assert first(frames, "complete")["text"] == ""
