#!/usr/bin/env python3
"""Tests for local speech-to-text (app/transcribe.py).

Nothing here may touch real faster-whisper: loading large-v3 downloads ~3 GB of weights
and then pins the GPU for minutes. Every test that reaches the model installs
``fake_whisper``, which puts a stub module in ``sys.modules`` under the real name, so the
lazy ``import faster_whisper`` inside ``_build`` finds it. That is also what lets the
CUDA-fallback tests exist at all — the failure they reproduce is a *runtime* kernel error
on hardware we can't require a test machine to have or lack.

``conftest.no_whisper`` is autouse and replaces ``transcribe_file``/``transcribe_url``
with raisers, so this file overrides it (``real_transcribe``) wherever it means to run
the real thing against the stub.
"""

import importlib.machinery
import sys
import threading
import time
import types

import pytest

from app import transcribe


# --------------------------- harness ---------------------------

class _Segment:
    def __init__(self, text, end):
        self.text = text
        self.end = end


class _Info:
    def __init__(self, duration=12.0, language="en"):
        self.duration = duration
        self.language = language


class _FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, name, **kw):
        self.name = name
        self.kw = kw
        self.calls = []
        _RECORD["models"].append(self)
        boom = _RECORD.get("load_raises")
        if boom and kw.get("device") == "cuda":
            raise RuntimeError(boom)

    def transcribe(self, path, **kw):
        self.calls.append((path, kw))
        _RECORD["transcribe_calls"].append((self.kw.get("device"), path, kw))
        boom = _RECORD.get("run_raises")
        if boom and self.kw.get("device") == "cuda":
            raise RuntimeError(boom)
        hook = _RECORD.get("on_transcribe")
        if hook:
            hook()
        segs = _RECORD.get("segments") or [
            _Segment("Hello there.", 4.0), _Segment("Second line.", 9.0)]
        return iter(segs), _Info()


class _FakeBatched:
    def __init__(self, model=None):
        self.model = model
        self.kw = model.kw

    def transcribe(self, path, **kw):
        return self.model.transcribe(path, **kw)


# BatchedInferencePipeline is matched by class NAME in transcribe._run (to decide whether
# batch_size is a legal kwarg), so the stub has to carry the real one.
_FakeBatched.__name__ = "BatchedInferencePipeline"

_RECORD = {}


@pytest.fixture
def fake_whisper(monkeypatch):
    """Install a stub ``faster_whisper`` module and reset the model singleton.

    Also puts the real ``transcribe_file``/``transcribe_url`` back on the module, undoing
    conftest's autouse ``no_whisper`` block. That matters beyond convenience:
    ``transcribe_url`` calls ``transcribe_file`` through the module global, so with the
    block in place it would reach the raiser instead of the code under test.
    """
    _RECORD.clear()
    _RECORD.update({"models": [], "transcribe_calls": []})
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _FakeModel
    mod.BatchedInferencePipeline = _FakeBatched
    # A bare ModuleType has __spec__ = None, which makes importlib.util.find_spec raise —
    # and is_available() would then report faster-whisper as absent.
    mod.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setattr(transcribe, "_cuda_present", lambda: True)
    monkeypatch.setattr(transcribe, "transcribe_file", _REAL_FILE)
    monkeypatch.setattr(transcribe, "transcribe_url", _REAL_URL)
    transcribe.reset_model()
    yield _RECORD
    transcribe.reset_model()


# Captured at import, before conftest's autouse fixture can swap them out.
_REAL_FILE = transcribe.transcribe_file
_REAL_URL = transcribe.transcribe_url


@pytest.fixture
def media(tmp_path):
    p = tmp_path / "episode.mp3"
    p.write_bytes(b"not really audio - the stub never decodes it")
    return p


# --------------------------- cue cleaning ---------------------------

def test_clean_transcript_strips_non_speech_cues():
    assert transcribe.clean_transcript("♪♪ Hello [MUSIC] world (music)") == "Hello world"


def test_clean_transcript_repairs_the_missing_space_after_punctuation():
    assert transcribe.clean_transcript("Stop.Then go") == "Stop. Then go"


def test_clean_transcript_collapses_the_hard_wraps_srt_cues_carry():
    # A 42-char wrap mid-sentence is what an SRT actually looks like; rejoining it is
    # the whole reason flow_paragraphs runs cues through this.
    raw = "of the thing that we were\ntalking about earlier"
    assert transcribe.clean_transcript(raw) == "of the thing that we were talking about earlier"


def test_flow_paragraphs_with_no_speakers_is_one_block():
    cues = [{"text": "One.", "speaker": ""}, {"text": "Two.", "speaker": ""}]
    assert transcribe.flow_paragraphs(cues) == "One. Two."


def test_flow_paragraphs_opens_a_paragraph_per_speaker_turn():
    cues = [{"text": "Morning.", "speaker": "Adam"},
            {"text": "In the morning.", "speaker": "Adam"},
            {"text": "Hello.", "speaker": "John"}]
    assert transcribe.flow_paragraphs(cues) == (
        "Adam: Morning. In the morning.\n\nJohn: Hello.")


def test_flow_paragraphs_skips_empty_cues():
    cues = [{"text": "  ", "speaker": "Adam"}, {"text": "Real.", "speaker": "Adam"}]
    assert transcribe.flow_paragraphs(cues) == "Adam: Real."


def test_flow_paragraphs_handles_no_cues():
    assert transcribe.flow_paragraphs([]) == ""
    assert transcribe.flow_paragraphs(None) == ""


# --------------------------- availability ---------------------------

def test_is_supported_matches_the_extension_set():
    assert transcribe.is_supported("show.mp3")
    assert transcribe.is_supported("SHOW.MP4")
    assert not transcribe.is_supported("notes.pdf")


def test_status_loads_nothing(fake_whisper):
    st = transcribe.status()
    assert st["loaded"] is False
    assert st["model"] == ""
    assert fake_whisper["models"] == []


def test_a_missing_faster_whisper_is_an_actionable_message(monkeypatch, media):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(transcribe, "_MODEL", None)
    with pytest.raises(transcribe.TranscribeError) as e:
        _REAL_FILE(media)
    assert "pip install faster-whisper" in str(e.value)


def test_transcribe_file_rejects_a_missing_file(tmp_path):
    with pytest.raises(transcribe.TranscribeError) as e:
        _REAL_FILE(tmp_path / "nope.mp3")
    assert "No such media file" in str(e.value)


def test_transcribe_file_rejects_an_unsupported_type(tmp_path):
    p = tmp_path / "notes.pdf"
    p.write_text("x")
    with pytest.raises(transcribe.TranscribeError) as e:
        _REAL_FILE(p)
    assert "transcribe" in str(e.value)


# --------------------------- the model singleton ---------------------------

def test_the_model_loads_once_for_two_calls(fake_whisper, media):
    _REAL_FILE(media)
    _REAL_FILE(media)
    assert len(fake_whisper["models"]) == 1


def test_reset_model_forces_a_reload(fake_whisper, media):
    _REAL_FILE(media)
    transcribe.reset_model()
    _REAL_FILE(media)
    assert len(fake_whisper["models"]) == 2


def test_the_key_includes_device_and_compute_type(fake_whisper, media):
    _REAL_FILE(media, settings={"whisper_device": "cpu", "whisper_compute_type": "int8"})
    _REAL_FILE(media, settings={"whisper_device": "cpu", "whisper_compute_type": "float32"})
    assert len(fake_whisper["models"]) == 2


def test_cpu_downgrades_float16_which_ctranslate2_rejects(fake_whisper, media):
    _REAL_FILE(media, settings={"whisper_device": "cpu", "whisper_compute_type": "float16"})
    assert fake_whisper["models"][0].kw["compute_type"] == "int8"


def test_auto_picks_cuda_when_ctranslate2_sees_a_device(fake_whisper, media):
    out = _REAL_FILE(media, settings={"whisper_device": "auto"})
    assert out["device"] == "cuda"


def test_auto_picks_cpu_when_there_is_no_device(fake_whisper, media, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_present", lambda: False)
    out = _REAL_FILE(media, settings={"whisper_device": "auto"})
    assert out["device"] == "cpu"


# --------------------------- the CUDA fallback ---------------------------

def test_a_load_time_cuda_failure_falls_back_to_cpu(fake_whisper, media):
    fake_whisper["load_raises"] = "CUDA failed with error no kernel image is available"
    out = _REAL_FILE(media, settings={"whisper_device": "cuda"})
    assert out["device"] == "cpu"
    assert out["text"]


def test_a_first_decode_cuda_failure_falls_back_to_cpu(fake_whisper, media):
    """THE case that matters: ctranslate2 reports CUDA as available, the model loads,
    and the first decode dies for want of kernels for this card (sm_120 on a 4.5 build).
    The fallback has to wrap the transcribe() call, not just the load."""
    fake_whisper["run_raises"] = "no kernel image is available for execution on the device"
    frames = []
    out = _REAL_FILE(media, settings={"whisper_device": "cuda"},
                     on_progress=lambda phase, **f: frames.append((phase, f)))
    assert out["device"] == "cpu"
    assert out["fallback"] is True
    assert out["text"] == "Hello there. Second line."
    assert any(f.get("fallback") for _, f in frames)


def test_a_second_call_goes_straight_to_cpu_without_retrying_cuda(fake_whisper, media):
    fake_whisper["run_raises"] = "no kernel image is available for execution on the device"
    _REAL_FILE(media, settings={"whisper_device": "cuda"})
    fake_whisper["transcribe_calls"].clear()
    out = _REAL_FILE(media, settings={"whisper_device": "cuda"})
    assert out["device"] == "cpu"
    assert out["fallback"] is False        # no fallback needed — CUDA is already ruled out
    assert [d for d, _, _ in fake_whisper["transcribe_calls"]] == ["cpu"]


def test_reset_model_clears_the_remembered_cuda_failure(fake_whisper, media):
    fake_whisper["run_raises"] = "no kernel image is available"
    _REAL_FILE(media, settings={"whisper_device": "cuda"})
    assert transcribe.status()["cuda_error"]
    transcribe.reset_model()
    assert transcribe.status()["cuda_error"] == ""
    assert transcribe.status()["cuda_usable"] is True


def test_a_non_cuda_error_is_not_swallowed_by_the_fallback(fake_whisper, media):
    fake_whisper["run_raises"] = "the model file is corrupt"
    with pytest.raises(transcribe.TranscribeError) as e:
        _REAL_FILE(media, settings={"whisper_device": "cuda"})
    assert "corrupt" in str(e.value)


def test_a_cpu_failure_is_reported_not_retried(fake_whisper, media):
    def boom():
        raise RuntimeError("ffmpeg could not open the file")
    fake_whisper["on_transcribe"] = boom
    with pytest.raises(transcribe.TranscribeError) as e:
        _REAL_FILE(media, settings={"whisper_device": "cpu"})
    assert "could not open" in str(e.value)


# --------------------------- concurrency ---------------------------

def test_the_lock_serialises_two_threads_and_the_waiter_is_told(fake_whisper, media):
    """The stub asserts it never sees overlapping calls; the second caller must also get
    a `waiting` frame rather than a silent forty-minute stall."""
    overlap = {"max": 0, "now": 0}
    guard = threading.Lock()

    def hook():
        with guard:
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
        time.sleep(0.15)
        with guard:
            overlap["now"] -= 1

    fake_whisper["on_transcribe"] = hook
    frames = [[], []]

    def run(i):
        _REAL_FILE(media, on_progress=lambda phase, **f: frames[i].append(f))

    ts = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert overlap["max"] == 1
    assert any(f.get("waiting") for f in frames[0] + frames[1])


# --------------------------- cancellation ---------------------------

def test_a_stopped_run_returns_what_it_had_and_says_so(fake_whisper, media):
    fake_whisper["segments"] = [_Segment(f"Line {i}.", i) for i in range(20)]
    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] > 3

    out = _REAL_FILE(media, should_stop=should_stop)
    assert out["stopped"] is True
    assert len(out["text"]) < len("Line 0. " * 20)


def test_progress_frames_are_throttled_but_the_first_one_lands(fake_whisper, media):
    fake_whisper["segments"] = [_Segment(f"Line {i}.", float(i)) for i in range(50)]
    frames = []
    _REAL_FILE(media, on_progress=lambda phase, **f: frames.append((phase, f)))
    whisper_frames = [f for p, f in frames if p == "whisper"]
    # Loading + at most a couple of throttled ticks — never one per segment.
    assert len(whisper_frames) < 10


# --------------------------- URL download ---------------------------

class _FakeResp:
    def __init__(self, chunks, headers=None, status=200):
        self._chunks = chunks
        self.headers = headers or {"Content-Type": "audio/mpeg"}
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def iter_content(self, n):
        return iter(self._chunks)

    def close(self):
        pass


def test_transcribe_url_deletes_the_download(fake_whisper, monkeypatch):
    seen = {}
    monkeypatch.setattr(transcribe.requests, "get",
                        lambda *a, **k: _FakeResp([b"audio-bytes"]))
    real_unlink = transcribe._unlink

    def spy(path):
        seen["path"] = path
        real_unlink(path)

    monkeypatch.setattr(transcribe, "_unlink", spy)
    out = _REAL_URL("https://example.com/ep1.mp3")
    assert out["text"]
    assert not __import__("os").path.exists(seen["path"])


def test_transcribe_url_deletes_the_download_even_when_transcription_fails(
        fake_whisper, monkeypatch):
    seen = {}
    monkeypatch.setattr(transcribe.requests, "get",
                        lambda *a, **k: _FakeResp([b"audio-bytes"]))
    fake_whisper["on_transcribe"] = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    real_unlink = transcribe._unlink

    def spy(path):
        seen["path"] = path
        real_unlink(path)

    monkeypatch.setattr(transcribe, "_unlink", spy)
    with pytest.raises(transcribe.TranscribeError):
        _REAL_URL("https://example.com/ep1.mp3")
    assert not __import__("os").path.exists(seen["path"])


def test_a_declared_oversize_download_is_refused_before_a_byte_is_read(monkeypatch):
    monkeypatch.setattr(transcribe.requests, "get", lambda *a, **k: _FakeResp(
        [b"x"], headers={"Content-Length": str(9 * 1024 ** 3)}))
    with pytest.raises(transcribe.TranscribeError) as e:
        transcribe.download_media("https://example.com/huge.mp3")
    assert "GB limit" in str(e.value)


def test_an_undeclared_oversize_download_is_cut_off_mid_stream(monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_MEDIA_BYTES", 8)
    monkeypatch.setattr(transcribe.requests, "get",
                        lambda *a, **k: _FakeResp([b"1234", b"5678", b"9abc"]))
    with pytest.raises(transcribe.TranscribeError) as e:
        transcribe.download_media("https://example.com/huge.mp3")
    assert "limit" in str(e.value)


def test_a_stopped_download_leaves_no_temp_file(monkeypatch):
    monkeypatch.setattr(transcribe.requests, "get",
                        lambda *a, **k: _FakeResp([b"1234", b"5678"]))
    with pytest.raises(transcribe.TranscribeError):
        transcribe.download_media("https://example.com/ep.mp3", should_stop=lambda: True)


def test_the_temp_suffix_comes_from_the_url_then_the_content_type():
    assert transcribe._ext_for("https://x/ep-42.m4a") == ".m4a"
    assert transcribe._ext_for("https://x/stream?id=9", "audio/ogg; charset=x") == ".ogg"
    assert transcribe._ext_for("https://x/stream") == ".mp3"
