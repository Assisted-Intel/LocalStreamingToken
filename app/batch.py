#!/usr/bin/env python3
"""
Local Streaming Token — Batch tab orchestration.

The chat composer's older Batch button treats every file in a folder as a *prompt*.
This module powers the Batch tab, which inverts that: an input item is *content*, and
one configured prompt template runs against each item. Inputs come from YouTube videos,
a YouTube playlist, a web search, or a directory tree of documents; outputs go to an
in-tab batch chat and/or to exported files.

Nothing here talks to Flask or to an LLM — the route in ``server.py`` owns the SSE
plumbing and generation goes through the shared ``generate_one`` seam. This module is
the part that is genuinely new: turning sources into items, filling the template,
choosing a filename, and writing the result.

Ingestion is deliberately delegated rather than reimplemented:
    folder   -> ingest.extract_many  (parallel, cancellable, PDF/EPUB/DOCX/text)
    youtube  -> youtube.fetch_video  (transcript + optional comments)
    playlist -> youtube.fetch_playlist -> youtube.fetch_video per video
    search   -> core.crawl_search    (Brave discovery + fetch fallback chain)

Public API:
    DEFAULT_FILENAME_PROMPT
    BatchError
    new_project(name) -> dict
    validate_project(project) -> [str]            # fail-fast errors for the route
    resolve_sources(project, emit, should_stop) -> [item]
    render_prompt(template, item) -> str
    sanitize_filename(name) -> str
    unique_path(path) -> Path
    resolve_output_path(item, chosen_title, project) -> Path
    render_export_body(item, prompt, response, project) -> str
    write_item(item, prompt, response, project, chosen_title) -> Path
    write_item_images(item, image_records, project, chosen_title) -> [Path]
    write_combined(results, project) -> Path
"""

import datetime
import re
import uuid
from pathlib import Path

from . import core, images, ingest, youtube

# Item content is truncated to this many characters before it reaches the model, so a
# 600-page PDF or a 4-hour transcript can't silently blow the context window. Mirrors
# the cap the Library already applies to a scraped page.
DEFAULT_MAX_ITEM_CHARS = core.LIBRARY_PAGE_CHARS

# Shipped as the editable default for "name the file from the content". Uses the same
# {{content}} placeholder as the main template so the two read alike.
DEFAULT_FILENAME_PROMPT = """Read the text below and write a short, descriptive file name for it.
Rules: 3-8 words, Title Case, no file extension, no quotes, and no
punctuation other than spaces and hyphens. Reply with the file name only.

{{content}}"""

SOURCE_KINDS = ("youtube", "playlist", "search", "folder", "images")


class BatchError(Exception):
    """Raised with a user-facing, actionable message when a batch can't proceed."""


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_project(name="Untitled batch"):
    """A batch project with every field at its default. The project IS the config —
    there are no batch keys in DEFAULT_SETTINGS."""
    return {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "created": _now(),
        "updated": _now(),

        # ---- inputs ----
        "sources": [],
        "yt_comments": False,
        "yt_max_comments": youtube.DEFAULT_MAX_COMMENTS,
        "max_item_chars": DEFAULT_MAX_ITEM_CHARS,
        # Images attached to EVERY item — a style guide, a reference chart, the
        # thing each input is being compared against. Records, not ids, so the tab
        # can show them without a round trip.
        "reference_images": [],
        "image_full_res": False,        # send images unscaled (see Settings → Images)

        # ---- generation (same knobs as a chat) ----
        "server_url": "",
        "model": "",
        "num_ctx": 0,
        "system_prompt": "",
        "system_on": False,
        "pre_prompt": "",
        "pre_on": True,
        "library_ids": [],
        "library_strict": False,
        "multi_pass": False,
        "passes": 2,
        "pass_use_system": True,
        "eval_prompt": "",

        # ---- outputs ----
        "output_mode": "both",          # chat | files | both
        "export_mode": "per_item",      # per_item | combined | beside_source
        "output_dir": "",
        "combined_name": "batch-results",
        "name_mode": "source",          # source | llm
        "name_prefix": "",
        "name_suffix": "",
        "name_prompt": DEFAULT_FILENAME_PROMPT,
        "ext": ".md",
        "include_source": False,
        "include_prompt": False,
        "strip_markdown": False,
    }


# --------------------------- Validation ---------------------------

def validate_project(project):
    """Return a list of user-facing problems that would make a run fail or, worse,
    quietly destroy data. The route calls this before starting so the user gets a 400
    with a real message instead of a half-finished run."""
    errors = []
    if not project.get("model"):
        errors.append("No model selected.")
    if not (project.get("sources") or []):
        errors.append("Add at least one input source.")

    for i, src in enumerate(project.get("sources") or [], 1):
        kind = (src.get("kind") or "").strip()
        if kind not in SOURCE_KINDS:
            errors.append(f"Source {i}: unknown input type '{kind}'.")
            continue
        if kind == "youtube" and not (src.get("urls") or "").strip():
            errors.append(f"Source {i}: no YouTube URLs were given.")
        if kind == "playlist" and not (src.get("url") or "").strip():
            errors.append(f"Source {i}: no playlist URL was given.")
        if kind == "search" and not (src.get("query") or "").strip():
            errors.append(f"Source {i}: no search query was given.")
        if kind in ("folder", "images"):
            path = (src.get("path") or "").strip()
            if not path:
                errors.append(f"Source {i}: no folder was chosen.")
            elif not Path(path).is_dir():
                errors.append(f"Source {i}: folder does not exist — {path}")

    if _exports_files(project):
        mode = project.get("export_mode") or "per_item"
        if mode == "beside_source":
            # The whole point of the append text: without it the output path IS the
            # input path and the run would overwrite the user's source documents.
            if not (project.get("name_prefix") or "").strip() and \
               not (project.get("name_suffix") or "").strip():
                errors.append(
                    "Saving next to the original files needs a name prefix or suffix, "
                    "otherwise the AI output would overwrite the files it just read.")
            kinds = {(s.get("kind") or "") for s in (project.get("sources") or [])}
            if not kinds & {"folder", "images"}:
                errors.append(
                    "Saving next to the original files only works for folder sources "
                    "— YouTube, playlist and search items have no file on disk.")
        else:
            out = (project.get("output_dir") or "").strip()
            if not out:
                errors.append("Choose an output folder for the exported files.")
            elif not Path(out).is_dir():
                errors.append(f"Output folder does not exist — {out}")
    return errors


def _exports_files(project):
    return (project.get("output_mode") or "both") in ("files", "both")


# --------------------------- Source resolution ---------------------------

def _truncate(text, limit):
    text = text or ""
    if limit and len(text) > limit:
        return text[:limit] + "\n\n[… truncated …]"
    return text


def _iter_folder(src, readable=None):
    """Files under a folder source, filtered to the extensions we can actually read.
    ``recursive`` walks subdirectories; the per-source ``exts`` list narrows further.
    ``readable`` is the vocabulary of extensions this source kind understands —
    documents by default, images for an image source."""
    readable = set(readable if readable is not None else ingest.SUPPORTED_EXTS)
    root = Path(src.get("path") or "")
    want = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (src.get("exts") or []) if e}
    allowed = (want & readable) if want else readable
    it = root.rglob("*") if src.get("recursive") else root.iterdir()
    return sorted(
        (p for p in it if p.is_file() and p.suffix.lower() in allowed),
        key=lambda p: str(p).lower(),
    )


def resolve_sources(project, emit=None, should_stop=None):
    """Turn a project's sources into a flat, ordered list of items.

    Each item is ``{item_id, title, content, kind, source_path, source_url, chars}``,
    plus ``image_ids`` for an image source (whose ``content`` is empty — the picture
    IS the input, and it travels as image parts on the user turn, not as text).
    ``emit(event, data)`` receives ``("progress", {done, total, name, phase})`` frames so
    the caller can drive a progress bar; ``should_stop()`` is polled between items.

    A source that fails (a dead video, an unreadable PDF) contributes an item-shaped
    error entry rather than aborting — one bad input must not cost the user the rest.
    """
    emit = emit or (lambda *_a, **_k: None)
    stopped = (lambda: bool(should_stop and should_stop()))
    limit = int(project.get("max_item_chars") or DEFAULT_MAX_ITEM_CHARS)
    items = []
    errors = []

    def _add(title, content, kind, source_path="", source_url="", image_ids=()):
        content = _truncate(content, limit)
        item = {
            # Index-based so two files with the same name in different subdirectories
            # (or a repeated URL) can't collide in the parallel engine's item map.
            "item_id": f"i{len(items)}",
            "title": title or f"Item {len(items) + 1}",
            "content": content,
            "kind": kind,
            "source_path": source_path,
            "source_url": source_url,
            "chars": len(content),
        }
        if image_ids:
            item["image_ids"] = list(image_ids)
        items.append(item)

    for src in (project.get("sources") or []):
        if stopped():
            break
        kind = (src.get("kind") or "").strip()

        if kind in ("youtube", "playlist"):
            if kind == "youtube":
                urls = [u.strip() for u in (src.get("urls") or "").splitlines() if u.strip()]
                videos = [{"url": u, "title": ""} for u in urls]
            else:
                emit("progress", {"phase": "playlist", "name": src.get("url") or "",
                                  "done": 0, "total": 0})
                try:
                    videos = youtube.fetch_playlist(
                        src.get("url") or "", limit=int(src.get("limit") or 0),
                        should_stop=should_stop)
                except Exception as e:
                    errors.append(f"Playlist: {e}")
                    continue

            total = len(videos)
            for n, vid in enumerate(videos, 1):
                if stopped():
                    break
                url = vid.get("url") or ""
                emit("progress", {"phase": "youtube", "done": n, "total": total,
                                  "name": vid.get("title") or url})
                try:
                    meta = youtube.fetch_video(
                        url,
                        include_comments=bool(project.get("yt_comments")),
                        max_comments=int(project.get("yt_max_comments")
                                         or youtube.DEFAULT_MAX_COMMENTS),
                        should_stop=should_stop)
                except Exception as e:
                    errors.append(f"{url}: {e}")
                    continue
                _add(meta.get("title") or vid.get("title") or url,
                     meta.get("text") or "", kind, source_url=meta.get("url") or url)

        elif kind == "search":
            query = (src.get("query") or "").strip()
            # Clamped here because ``core.crawl_search`` only bounds this from below and
            # the field's `max` attribute is not enforced against a typed value. The
            # ceiling is the Batch tab's own, higher than the chat/Resources one — bulk
            # work is the point here — but it is still a ceiling.
            max_results = max(1, min(core.MAX_BATCH_CRAWL_PAGES,
                                     int(src.get("max_results") or 5)))
            emit("progress", {"phase": "search", "done": 0, "total": max_results,
                              "name": query})
            pages = []
            try:
                for ev in core.crawl_search(query, sites=src.get("sites") or [],
                                            max_results=max_results,
                                            should_stop=should_stop):
                    if ev.get("type") == "progress":
                        emit("progress", {"phase": "search", "done": ev.get("done") or 0,
                                          "total": ev.get("target") or max_results,
                                          "name": ev.get("title") or ev.get("url") or ""})
                    elif ev.get("type") == "result":
                        pages = ev.get("pages") or []
                        errors.extend(ev.get("errors") or [])
            except Exception as e:
                errors.append(f"Search '{query}': {e}")
            for page in pages:
                _add(page.get("title") or page.get("url") or query,
                     page.get("text") or "", kind, source_url=page.get("url") or "")

        elif kind == "folder":
            paths = _iter_folder(src)
            total = len(paths)
            if not total:
                errors.append(f"No readable documents in {src.get('path')}")
                continue
            emit("progress", {"phase": "folder", "done": 0, "total": total,
                              "name": src.get("path") or ""})
            results = ingest.extract_many(
                [str(p) for p in paths],
                on_progress=lambda done, tot, name: emit(
                    "progress", {"phase": "folder", "done": done, "total": tot, "name": name}),
                should_stop=should_stop)
            for res in results:
                if res.get("ok"):
                    p = Path(res.get("path") or "")
                    _add(res.get("title") or p.stem, res.get("text") or "",
                         kind, source_path=str(p))
                else:
                    errors.append(f"{Path(res.get('path') or '').name}: {res.get('error')}")

        elif kind == "images":
            paths = _iter_folder(src, images.SUPPORTED_EXTS)
            total = len(paths)
            if not total:
                errors.append(f"No readable images in {src.get('path')}")
                continue
            for n, p in enumerate(paths, 1):
                if stopped():
                    break
                emit("progress", {"phase": "images", "done": n, "total": total,
                                  "name": p.name})
                try:
                    # Stored at full resolution; the send-time clamp decides what
                    # actually goes on the wire.
                    rec = images.store_file(p)
                except Exception as e:
                    errors.append(f"{p.name}: {e}")
                    continue
                _add(p.stem, "", kind, source_path=str(p), image_ids=[rec["id"]])

        else:
            errors.append(f"Unknown input type '{kind}'.")

    return items, errors


# --------------------------- Prompt template ---------------------------

# Tolerant of spacing and case: {{content}}, {{ Content }}, {{CONTENT}} all match.
_PLACEHOLDER = re.compile(r"\{\{\s*(content|title|url|source)\s*\}\}", re.IGNORECASE)


def render_prompt(template, item):
    """Fill a prompt template with one item's fields.

    Supports ``{{content}}``, ``{{title}}``, ``{{url}}`` and ``{{source}}``. A template
    with no ``{{content}}`` gets the item text appended after a blank line, which makes
    the plain "instruction, then the document" case work with no placeholder at all.
    """
    template = template or ""
    values = {
        "content": item.get("content") or "",
        "title": item.get("title") or "",
        "url": item.get("source_url") or "",
        "source": item.get("source_path") or item.get("source_url") or "",
    }
    has_content = bool(re.search(r"\{\{\s*content\s*\}\}", template, re.IGNORECASE))
    filled = _PLACEHOLDER.sub(lambda m: values[m.group(1).lower()], template)
    if not template.strip():
        return values["content"]
    if not has_content:
        # An image item has no text, so appending it would leave a trailing blank
        # line where the document would have been.
        return f"{filled}\n\n{values['content']}" if values["content"] else filled
    return filled


# --------------------------- Filenames ---------------------------

# Characters Windows forbids in a name, plus the path separators and control chars.
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Device names that are illegal as a bare filename on Windows regardless of extension.
_RESERVED = {"con", "prn", "aux", "nul",
             *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}
_MAX_STEM = 120


def sanitize_filename(name):
    """Make an arbitrary string safe to use as a file stem on Windows and POSIX.

    An LLM-generated name is untrusted input: it can contain path separators, quotes,
    newlines, or be a reserved device name, any of which would either fail to write or
    escape the output folder.
    """
    stem = _BAD_CHARS.sub(" ", str(name or ""))
    stem = re.sub(r"\s+", " ", stem).strip()
    # A leading dot hides the file; trailing dots/spaces are silently stripped by
    # Windows, which would let two different names collide.
    stem = stem.strip(". ")
    if not stem:
        stem = "untitled"
    if stem.lower() in _RESERVED:
        stem = f"{stem}_"
    return stem[:_MAX_STEM].strip() or "untitled"


def unique_path(path):
    """``path`` if free, else the same name with ' (2)', ' (3)' … appended. Keeps a run
    from overwriting its own earlier output when two items resolve to the same name."""
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem} ({uuid.uuid4().hex[:6]}){path.suffix}")


def resolve_output_path(item, chosen_title, project, ext=None):
    """Where one item's response should be written.

    ``chosen_title`` is the LLM-generated name when ``name_mode`` is 'llm'; it is
    ignored otherwise. ``ext`` overrides the project's extension, which is how a
    returned image lands beside the .md it belongs to under the same naming rules.
    Raises BatchError rather than returning a path that would overwrite the source
    document.
    """
    mode = project.get("export_mode") or "per_item"
    prefix = (project.get("name_prefix") or "").strip()
    suffix = (project.get("name_suffix") or "").strip()
    ext = ext or project.get("ext") or ".md"
    if not ext.startswith("."):
        ext = "." + ext

    source_path = Path(item["source_path"]) if item.get("source_path") else None
    if project.get("name_mode") == "llm" and (chosen_title or "").strip():
        base = chosen_title
    elif source_path is not None:
        base = source_path.stem
    else:
        base = item.get("title") or "untitled"

    stem = sanitize_filename(f"{prefix}{sanitize_filename(base)}{suffix}")

    if mode == "beside_source":
        if source_path is None:
            raise BatchError(
                f"'{item.get('title')}' has no file on disk, so it cannot be saved "
                f"next to its original. Use an output folder instead.")
        if not prefix and not suffix:
            raise BatchError(
                "Saving next to the original files needs a name prefix or suffix.")
        target = source_path.parent / f"{stem}{ext}"
        # Belt and braces: even with a suffix set, a same-extension collision could
        # still land exactly on the input file.
        if target.resolve() == source_path.resolve():
            raise BatchError(
                f"The output name for '{source_path.name}' matches the source file. "
                f"Change the prefix or suffix so the original isn't overwritten.")
        return unique_path(target)

    out_dir = (project.get("output_dir") or "").strip()
    if not out_dir:
        raise BatchError("No output folder was chosen.")
    return unique_path(Path(out_dir) / f"{stem}{ext}")


# --------------------------- Export bodies ---------------------------

def render_export_body(item, prompt, response, project):
    """Assemble what actually goes in the file: the response, optionally preceded by
    the source text and/or the prompt that produced it, optionally de-marked-down."""
    body = response or ""
    if project.get("strip_markdown"):
        body = core.strip_markdown(body) or body

    parts = []
    if project.get("include_source"):
        origin = item.get("source_path") or item.get("source_url") or ""
        head = f"# Source: {item.get('title') or ''}"
        if origin:
            head += f"\n{origin}"
        parts.append(f"{head}\n\n{item.get('content') or ''}")
    if project.get("include_prompt"):
        parts.append(f"# Prompt\n\n{prompt or ''}")
    if parts:
        parts.append("# Response\n\n" + body)
        return "\n\n---\n\n".join(parts)
    return body


def write_item(item, prompt, response, project, chosen_title=""):
    """Write one item's export file. Returns the path written."""
    path = resolve_output_path(item, chosen_title, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_export_body(item, prompt, response, project), encoding="utf-8")
    return path


def write_item_images(item, image_records, project, chosen_title=""):
    """Write the images a model returned for one item. Returns the paths written.

    Uses the same naming rules as the text export, with the image's own extension, so
    a run's pictures land beside their .md under matching names. ``unique_path``
    inside ``resolve_output_path`` separates several images from one item.
    """
    written = []
    for rec in image_records or []:
        image_id = rec.get("id") if isinstance(rec, dict) else rec
        if not image_id:
            continue
        try:
            ext = images.ext_for((rec or {}).get("media_type"))
            path = resolve_output_path(item, chosen_title, project, ext=ext)
            written.append(images.write_out(image_id, path))
        except (BatchError, images.ImageError):
            # A picture that can't be placed must not lose the user the text export
            # that was already written for this item.
            continue
    return written


def write_combined(results, project):
    """Write every result into one file under ``output_dir``. ``results`` is a list of
    ``{item, prompt, response}`` in run order. Returns the path written."""
    out_dir = (project.get("output_dir") or "").strip()
    if not out_dir:
        raise BatchError("No output folder was chosen.")
    ext = project.get("ext") or ".md"
    if not ext.startswith("."):
        ext = "." + ext
    stem = sanitize_filename(project.get("combined_name") or "batch-results")
    path = unique_path(Path(out_dir) / f"{stem}{ext}")

    chunks = []
    for r in results:
        item = r.get("item") or {}
        chunks.append(
            f"# {item.get('title') or 'Untitled'}\n\n"
            + render_export_body(item, r.get("prompt"), r.get("response"), project))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
    return path
