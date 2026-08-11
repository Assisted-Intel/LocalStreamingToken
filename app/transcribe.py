#!/usr/bin/env python3
"""
Local Streaming Token — local speech-to-text, and the shared cue-to-text cleaner.

Two jobs, in one module because they are the same job at different distances: turning
*timed speech* into flowed prose. ``clean_transcript``/``flow_paragraphs`` do it for cues
somebody else produced (a YouTube caption track, a ``<podcast:transcript>`` SRT);
``transcribe_file``/``transcribe_url`` do it for audio nobody has transcribed at all.

Keeping them together is why ``clean_transcript`` lives here rather than in youtube.py,
where it started. Podcast SRTs carry ``[MUSIC]`` and ``(laughs)`` exactly as YouTube
captions do; a second copy of that regex triple in rss.py would drift from this one
within a release. ``youtube._clean_transcript`` is now a one-line delegation and keeps
its name, so nothing about the YouTube path changed.

faster-whisper is OPTIONAL and imported lazily. ``is_available()`` uses ``find_spec``
only — never an import — because ``batch.validate_project`` calls it on every validation
pass and importing faster_whisper drags in torch, which costs seconds.

Concurrency: ONE model, ONE lock, and the lock is held across *inference*, not just the
load. large-v3 in float16 is ~3 GB of VRAM on a card that is usually also serving Ollama,
and ctranslate2 does not document concurrent ``generate`` on a single instance as safe.
Serialising costs nothing in throughput — one whisper run already saturates the GPU — and
it costs nothing in latency either, because the only realistic concurrency here is two
browser tabs or a composer fetch overlapping a Batch preview. What it must NOT do is look
like a hang, so a caller that has to wait gets a ``waiting`` progress frame first.

Profile switching deliberately does NOT reset the model. Unlike ``app/images.py``'s LRU,
nothing here is user data keyed by id — it is a few gigabytes of published weights named
by a model string, identical across every profile — so dropping it on a profile switch
would just cost the next profile a 30-second reload for no privacy gain. (Read that
against youtube_cache.py's "no in-memory state is what lets profile switching need no
hook", which invites the opposite inference.) Changing a ``whisper_*`` SETTING is
different and must call ``reset_model()``; those live in the settings profile.

The GPU path is not assumed to work. ctranslate2 ships kernels per compute capability,
and a build older than the card reports CUDA as available, loads the model happily, and
then dies at the *first decode* with "no kernel image is available for execution on the
device" — which is why the fallback wraps the first ``transcribe()`` call and not just
the load. Once observed, CUDA is skipped for the rest of the process.

Public API:
    TranscribeError
    SUPPORTED_EXTS, MAX_MEDIA_BYTES
    is_available() -> bool
    is_supported(path) -> bool
    status() -> dict
    clean_transcript(raw) -> str
    flow_paragraphs(cues) -> str
    load_model(settings) -> (pipeline, device, compute_type)
    reset_model() -> None
    transcribe_file(path, ...) -> dict
    transcribe_url(url, ...) -> dict
"""

import gc
import importlib.util
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from . import core


class TranscribeError(Exception):
    """Raised with a user-facing, actionable message when transcription can't proceed."""


# Extensions faster-whisper can decode through PyAV. Deliberately NOT merged into
# ingest.SUPPORTED_EXTS: a folder-of-documents batch source would start advertising
# .mp3, and ingest.extract would correctly refuse it.
SUPPORTED_EXTS = {
    ".mp3", ".m4a", ".m4b", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wav", ".wma",
    ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi",
}

# A mistyped URL pointing at a disk image must not fill the drive. Generous enough for a
# 12-hour lossless recording; a real podcast episode is 50-200 MB.
MAX_MEDIA_BYTES = 2 * 1024 ** 3

DEFAULT_MODEL = "large-v3"
_DEVICES = ("auto", "cuda", "cpu")
_COMPUTE_TYPES = ("float16", "int8_float16", "bfloat16", "int8", "float32")


# --------------------------- Cue cleaning ---------------------------
# Moved here verbatim from youtube.py, which now delegates. See the module docstring.

def ensure_punctuation_spacing(text: str) -> str:
    """Insert the space a caption track omits after sentence punctuation.

    Public because youtube.py's comment cleaner wants it too — comments arrive with the
    same missing space, and it is the one piece of that cleaner worth sharing.
    """
    return re.sub(r"([.,!?;:])(?=[^\s])", r"\1 ", text or "")


# Non-speech cues that add tokens without adding meaning.
_CUE_NOISE = re.compile(r"[♪♫]+|\[.*?\]|\(music\)|\(sound effect\)", re.IGNORECASE)


def clean_transcript(raw: str) -> str:
    """Tidy a transcript assembled from timed cues.

    Deliberately narrow. A broader cleaner that also strips the words "Comments",
    "Description" and "Subtitles" — the kind a DOM-scraping fallback needs — would
    corrupt real speech here, because every source feeding this function returns caption
    text only.

    The whitespace collapse at the end is doing more work than it looks: SRT and VTT cues
    are hard-wrapped at ~42 characters mid-sentence, so joining them leaves stray line
    breaks inside sentences. Collapsing them is what turns 3,000 cues back into prose.
    """
    text = _CUE_NOISE.sub(" ", raw or "")
    text = ensure_punctuation_spacing(text)
    return re.sub(r"\s+", " ", text).strip()


def flow_paragraphs(cues) -> str:
    """Flow ``[{text, speaker}]`` cues into prose, one paragraph per speaker turn.

    Timing is dropped — an LLM reads none of it, and it is roughly 40% of an SRT file.
    Consecutive cues from one speaker join into a single paragraph; a change opens a new
    one prefixed ``Name: ``. With no speakers anywhere the result is one flowed block,
    identical in shape to what the YouTube path has always produced.
    """
    paras, current, speaker = [], [], None

    def flush():
        if not current:
            return
        body = clean_transcript(" ".join(current))
        if body:
            paras.append(f"{speaker}: {body}" if speaker else body)

    for cue in cues or []:
        who = (cue.get("speaker") or "").strip()
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        if who != (speaker or ""):
            flush()
            current, speaker = [], who or None
        current.append(text)
    flush()
    return "\n\n".join(paras)


# --------------------------- Availability ---------------------------

def is_available() -> bool:
    """Is faster-whisper importable?

    ``find_spec`` only — this must never import the package. ``batch.validate_project``
    calls it on every validation, and importing faster_whisper pulls in torch.
    """
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except Exception:
        return False


def is_supported(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


# --------------------------- Model singleton ---------------------------

# Guards the load AND every inference. See the module docstring for why inference too.
_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_KEY = None            # (model, device, compute_type, cpu_threads)
_MODEL_DEVICE = ""
_MODEL_COMPUTE = ""
# Set once a CUDA kernel failure is observed, so the rest of the process stops paying to
# rediscover it. Cleared by reset_model(), which is what makes "install a newer
# ctranslate2, press Reset" work without restarting the app.
_CUDA_UNUSABLE = ""


def _settings_or_defaults(settings):
    s = settings or {}
    device = (s.get("whisper_device") or "auto").strip().lower()
    if device not in _DEVICES:
        device = "auto"
    compute = (s.get("whisper_compute_type") or "float16").strip().lower()
    if compute not in _COMPUTE_TYPES:
        compute = "float16"
    return {
        "model": (s.get("whisper_model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "device": device,
        "compute_type": compute,
        "batch_size": max(1, min(32, int(s.get("whisper_batch_size") or 8))),
        "language": (s.get("whisper_language") or "").strip(),
        "vad": bool(s.get("whisper_vad", True)),
        "beam_size": max(1, min(10, int(s.get("whisper_beam_size") or 5))),
        "cpu_threads": max(0, min(64, int(s.get("whisper_cpu_threads") or 0))),
    }


def _cuda_present() -> bool:
    """Does ctranslate2 see a CUDA device? Cheap, and does NOT import torch."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# The shapes a card-too-new-for-the-build failure takes. OOM is arguably "try a smaller
# model" rather than "use the CPU", but a slow answer beats no answer — fall back and say
# so in the progress frame.
_CUDA_FAIL = re.compile(
    r"no kernel image|invalid device function|unsupported gpu architecture|sm_\d+|"
    r"cuda failed with error|cudnn|cublas|out of memory|device-side assert",
    re.IGNORECASE)


def _is_cuda_failure(exc) -> bool:
    return bool(_CUDA_FAIL.search(str(exc) or ""))


def _import_faster_whisper():
    try:
        import faster_whisper
    except Exception as e:
        raise TranscribeError(
            "Local transcription needs faster-whisper. Install it with:  "
            f"pip install faster-whisper\n({e})")
    return faster_whisper


def _build(cfg, force_cpu=False):
    """Construct the pipeline. Caller holds _MODEL_LOCK."""
    fw = _import_faster_whisper()

    device = "cpu" if force_cpu else cfg["device"]
    compute = cfg["compute_type"]
    if device == "auto":
        device = "cuda" if (_cuda_present() and not _CUDA_UNUSABLE) else "cpu"
    if device == "cuda" and _CUDA_UNUSABLE:
        device = "cpu"
    if device == "cpu":
        # float16 on CPU is not merely slow, ctranslate2 rejects it outright.
        if compute in ("float16", "bfloat16", "int8_float16"):
            compute = "int8"

    kw = {"device": device, "compute_type": compute}
    if device == "cpu" and cfg["cpu_threads"]:
        kw["cpu_threads"] = cfg["cpu_threads"]

    try:
        model = fw.WhisperModel(cfg["model"], **kw)
    except Exception as e:
        if device == "cuda":
            # Load-time CUDA failure — the easy half of the problem. Retry on CPU rather
            # than making the user go and change a setting.
            globals()["_CUDA_UNUSABLE"] = str(e)
            return _build(cfg, force_cpu=True)
        raise TranscribeError(_actionable(e, cfg, device))

    # Batched inference is a large speedup on long audio and is what makes a 3-hour
    # episode finish in minutes. It is a wrapper, so a version without it degrades to the
    # plain model rather than failing.
    pipeline = model
    batched = getattr(fw, "BatchedInferencePipeline", None)
    if batched is not None and cfg["batch_size"] > 1:
        try:
            pipeline = batched(model=model)
        except Exception:
            pipeline = model
    return pipeline, device, compute


def _actionable(exc, cfg, device) -> str:
    if device == "cuda" or _is_cuda_failure(exc):
        return (f"Transcription failed on the GPU: {exc}\n\n"
                "This is usually a ctranslate2 build with no kernels for your card's "
                "compute capability (Blackwell / RTX 50xx needs ctranslate2>=4.6). "
                "Either set Settings → Transcription → Device to 'cpu' and Compute type "
                "to 'int8', or upgrade with:  pip install -U ctranslate2")
    return (f"Transcription failed: {exc}\n\n"
            f"Model '{cfg['model']}' on device '{device}'. Check the model name in "
            "Settings → Transcription, and that there is disk space for the weights.")


def load_model(settings=None):
    """The cached pipeline for these settings, loading it if the key changed.

    Returns ``(pipeline, device, compute_type)``. Caller must hold ``_MODEL_LOCK``.
    """
    global _MODEL, _MODEL_KEY, _MODEL_DEVICE, _MODEL_COMPUTE
    cfg = _settings_or_defaults(settings)
    key = (cfg["model"], cfg["device"], cfg["compute_type"], cfg["cpu_threads"])
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE

    _drop()
    _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE = _build(cfg)
    _MODEL_KEY = key
    return _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE


def _drop():
    """Release the model and its VRAM. Caller holds _MODEL_LOCK (or is reset_model)."""
    global _MODEL, _MODEL_KEY, _MODEL_DEVICE, _MODEL_COMPUTE
    _MODEL, _MODEL_KEY, _MODEL_DEVICE, _MODEL_COMPUTE = None, None, "", ""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass          # a courtesy, never a requirement


def reset_model():
    """Drop the loaded model and forget any observed CUDA failure.

    Takes the lock, so an in-flight transcription finishes rather than having the model
    yanked out from under it. Called when a ``whisper_*`` setting changes — without it,
    editing the model in Settings would do nothing until restart.
    """
    global _CUDA_UNUSABLE
    with _MODEL_LOCK:
        _drop()
        _CUDA_UNUSABLE = ""


def status() -> dict:
    """What the Settings card and the RSS panels show. Loads nothing."""
    available = is_available()
    cuda = _cuda_present() if available else False
    return {
        "available": available,
        "cuda_present": cuda,
        "cuda_usable": bool(cuda and not _CUDA_UNUSABLE),
        "cuda_error": _CUDA_UNUSABLE,
        "loaded": _MODEL is not None,
        "model": (_MODEL_KEY or ("", "", "", 0))[0],
        "device": _MODEL_DEVICE,
        "compute_type": _MODEL_COMPUTE,
        "exts": sorted(SUPPORTED_EXTS),
    }


# --------------------------- Transcription ---------------------------

def _run(pipeline, path, cfg, on_progress, should_stop, device):
    """Consume the segment generator, reporting progress and honouring cancellation.

    faster-whisper yields lazily, so this is where the time actually goes — and where a
    stop has to be checked. Progress frames are throttled to one a second: they are what
    lets ``_compile_sse`` notice a disconnected browser (its GeneratorExit only arrives
    when the generator is next resumed, i.e. when a frame is written), so going silent
    for six minutes would mean a six-minute-late cancellation.
    """
    kw = {"beam_size": cfg["beam_size"], "vad_filter": cfg["vad"]}
    if cfg["language"]:
        kw["language"] = cfg["language"]
    # batch_size is a BatchedInferencePipeline argument; the plain model rejects it.
    if type(pipeline).__name__ == "BatchedInferencePipeline":
        kw["batch_size"] = cfg["batch_size"]

    segments, info = pipeline.transcribe(str(path), **kw)

    total = float(getattr(info, "duration", 0.0) or 0.0)
    language = getattr(info, "language", "") or cfg["language"]
    cues, stopped, last = [], False, 0.0
    for seg in segments:
        if should_stop and should_stop():
            stopped = True
            break
        text = (getattr(seg, "text", "") or "").strip()
        if text:
            cues.append({"text": text, "speaker": ""})
        now = time.monotonic()
        if on_progress and (now - last) >= 1.0:
            last = now
            on_progress("whisper", done=float(getattr(seg, "end", 0.0) or 0.0),
                        total=total, unit="seconds", device=device,
                        model=cfg["model"], waiting=False)
    return cues, language, total, stopped


def transcribe_file(path, *, settings=None, on_progress=None, should_stop=None) -> dict:
    """Transcribe a local media file.

    Returns ``{text, cues, language, duration, device, compute_type, model, fallback,
    stopped, chars}``. A stopped run returns what it had with ``stopped=True`` — callers
    must not cache that (see rss_cache.put rule 3): it is the first N minutes and, once
    on disk, indistinguishable from a complete transcript.
    """
    global _CUDA_UNUSABLE
    p = Path(path)
    if not p.is_file():
        raise TranscribeError(f"No such media file: {path}")
    if not is_supported(p):
        raise TranscribeError(
            f"{p.suffix or 'That file'} is not a media type we can transcribe. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    cfg = _settings_or_defaults(settings)

    # Make contention visible instead of looking like a forty-minute hang.
    if not _MODEL_LOCK.acquire(blocking=False):
        if on_progress:
            on_progress("whisper", waiting=True,
                        message="Waiting for the transcriber (another transcription is "
                                "running)…")
        _MODEL_LOCK.acquire()
    try:
        if on_progress:
            on_progress("whisper", waiting=False, message="Loading the speech model…",
                        model=cfg["model"])
        pipeline, device, compute = load_model(settings)
        fallback = False
        try:
            cues, language, duration, stopped = _run(
                pipeline, p, cfg, on_progress, should_stop, device)
        except Exception as e:
            # THE case that matters: the load succeeded because ctranslate2 reported CUDA
            # as available, and the first decode died for want of kernels for this card.
            if device != "cuda" or not _is_cuda_failure(e):
                raise TranscribeError(_actionable(e, cfg, device))
            _CUDA_UNUSABLE = str(e)
            _drop()
            if on_progress:
                on_progress("whisper", device="cpu", fallback=True,
                            reason=str(e)[:200],
                            message="The GPU couldn't run the model — falling back to "
                                    "the CPU. This will be slower.")
            pipeline, device, compute = load_model(settings)
            fallback = True
            try:
                cues, language, duration, stopped = _run(
                    pipeline, p, cfg, on_progress, should_stop, device)
            except Exception as e2:
                raise TranscribeError(_actionable(e2, cfg, device))
    finally:
        _MODEL_LOCK.release()

    text = flow_paragraphs(cues)
    return {"text": text, "cues": cues, "language": language, "duration": duration,
            "device": device, "compute_type": compute, "model": cfg["model"],
            "fallback": fallback, "stopped": stopped, "chars": len(text)}


def _ext_for(url: str, content_type: str = "") -> str:
    """A suffix PyAV can dispatch on: from the URL path, else the Content-Type."""
    name = unquote(urlparse(url or "").path).rsplit("/", 1)[-1]
    ext = Path(name).suffix.lower()
    if ext in SUPPORTED_EXTS:
        return ext
    ct = (content_type or "").split(";")[0].strip().lower()
    return {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg",
            "audio/opus": ".opus", "audio/flac": ".flac", "audio/wav": ".wav",
            "audio/x-wav": ".wav", "video/mp4": ".mp4", "video/webm": ".webm",
            "video/x-matroska": ".mkv"}.get(ct, ".mp3")


def download_media(url, *, on_progress=None, should_stop=None, timeout=45) -> str:
    """Stream a media URL to a temp file and return its path. Caller unlinks it.

    The temp file goes in the SYSTEM temp dir, never a data profile: these bytes are not
    user data we keep, and putting them in a profile would drag profile merge and the
    incognito wipe into a file that exists for ninety seconds. It is also the only
    unencrypted user-derived file the app writes — see ARCHITECTURE's plaintext
    exceptions.
    """
    resp = requests.get(url, stream=True, timeout=timeout,
                        headers={"User-Agent": core._DEFAULT_UA})
    resp.raise_for_status()

    declared = int(resp.headers.get("Content-Length") or 0)
    if declared and declared > MAX_MEDIA_BYTES:
        resp.close()
        raise TranscribeError(
            f"That media file is {declared / 1024**3:.1f} GB, over the "
            f"{MAX_MEDIA_BYTES / 1024**3:.0f} GB limit.")

    fd, tmp = tempfile.mkstemp(prefix="lst-media-",
                              suffix=_ext_for(url, resp.headers.get("Content-Type", "")))
    done, last = 0, 0.0
    try:
        with os.fdopen(fd, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                if should_stop and should_stop():
                    raise TranscribeError("stopped")
                if not chunk:
                    continue
                done += len(chunk)
                if done > MAX_MEDIA_BYTES:
                    raise TranscribeError(
                        f"That media file exceeded the "
                        f"{MAX_MEDIA_BYTES / 1024**3:.0f} GB limit while downloading.")
                fh.write(chunk)
                now = time.monotonic()
                if on_progress and (now - last) >= 1.0:
                    last = now
                    on_progress("download", done=done, total=declared, unit="bytes")
    except BaseException:
        _unlink(tmp)
        raise
    finally:
        resp.close()
    return tmp


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def transcribe_url(url, *, settings=None, on_progress=None, should_stop=None) -> dict:
    """Download a media URL, transcribe it, and delete the download.

    The unlink is in a ``finally`` so it also covers a stopped run and a failed one —
    decision 6 is that no audio is ever kept, and "no audio is kept unless something went
    wrong" is not that.
    """
    tmp = download_media(url, on_progress=on_progress, should_stop=should_stop)
    try:
        result = transcribe_file(tmp, settings=settings, on_progress=on_progress,
                                 should_stop=should_stop)
    finally:
        _unlink(tmp)
    result["url"] = url
    return result
