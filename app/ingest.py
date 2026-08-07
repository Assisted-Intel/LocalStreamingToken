#!/usr/bin/env python3
"""
Local Streaming Token — document ingestion.

Turns a file on disk into plain text (plus light metadata) so it can be stored as a
Library item and fed to the RAG chunker. Rich formats (PDF/EPUB/DOCX) are parsed with
optional third-party libraries that are imported lazily — if a parser dependency isn't
installed, only that format raises a clear, actionable error; plain text/markdown always
works with no dependencies.

Public API:
    SUPPORTED_EXTS                     -> set of lowercase extensions we can read
    is_supported(path) -> bool
    extract(path) -> {text, title, pages, format, ...}   # metadata dict
    extract_text(path) -> str                            # convenience: just the text
    extract_many(paths, workers, on_progress) -> [{ok, path, ...}]  # parallel + progress
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# Plain-text formats need no external parser.
_TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".rst"}
_RICH_EXTS = {".pdf", ".epub", ".docx"}
SUPPORTED_EXTS = _TEXT_EXTS | _RICH_EXTS


class IngestError(Exception):
    """Raised with a user-facing, actionable message when a file can't be read."""


def is_supported(path) -> bool:
    """True if this file's extension is one we can extract text from. Cheap check on
    the name only — the file need not exist."""
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def _missing(dep: str, ext: str) -> "IngestError":
    """Build the error for an uninstalled optional parser, naming the exact pip
    command. Returned (not raised) so call sites read as ``raise _missing(...)``."""
    return IngestError(
        f"Reading {ext} files needs the '{dep}' package. "
        f"Install it with:  pip install {dep}")


# --------------------------- Format parsers ---------------------------

def _read_text(path: Path) -> dict:
    """Plain text and markdown. Decoding errors are replaced rather than raised, so a
    file with a few bad bytes still ingests instead of failing outright."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"text": text, "title": path.stem, "pages": None, "format": path.suffix.lower().lstrip(".")}


def _read_pdf(path: Path) -> dict:
    """PDF via pypdf. Pages that fail to parse contribute an empty string rather than
    aborting the document — a partially extractable PDF is still worth indexing.
    Prefers the embedded title over the filename."""
    try:
        from pypdf import PdfReader
    except Exception:
        raise _missing("pypdf", ".pdf")
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise IngestError(f"Could not open PDF '{path.name}': {e}")
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    title = path.stem
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception:
        pass
    return {"text": "\n\n".join(parts).strip(), "title": title,
            "pages": len(reader.pages), "format": "pdf"}


def _read_epub(path: Path) -> dict:
    """EPUB via ebooklib, with BeautifulSoup stripping the XHTML of each chapter.
    ``pages`` reports the chapter count, an EPUB having no fixed pagination."""
    try:
        import ebooklib
        from ebooklib import epub
    except Exception:
        raise _missing("ebooklib", ".epub")
    try:
        from bs4 import BeautifulSoup
    except Exception:
        raise _missing("beautifulsoup4", ".epub")
    try:
        book = epub.read_epub(str(path))
    except Exception as e:
        raise IngestError(f"Could not open EPUB '{path.name}': {e}")
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            txt = soup.get_text(separator="\n").strip()
            if txt:
                parts.append(txt)
        except Exception:
            continue
    title = path.stem
    try:
        meta = book.get_metadata("DC", "title")
        if meta:
            title = meta[0][0]
    except Exception:
        pass
    return {"text": "\n\n".join(parts).strip(), "title": title,
            "pages": len(parts), "format": "epub"}


def _read_docx(path: Path) -> dict:
    """DOCX via python-docx. Table cells are pulled out alongside paragraphs and
    joined with " | " — tabular content is often the substance of a document, and
    python-docx does not include it in .paragraphs."""
    try:
        import docx  # python-docx
    except Exception:
        raise _missing("python-docx", ".docx")
    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise IngestError(f"Could not open DOCX '{path.name}': {e}")
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    # Include table cell text too — often carries real content.
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return {"text": "\n".join(parts).strip(), "title": path.stem,
            "pages": None, "format": "docx"}


_PARSERS = {
    ".pdf": _read_pdf,
    ".epub": _read_epub,
    ".docx": _read_docx,
}


# --------------------------- Public API ---------------------------

def extract(path) -> dict:
    """Parse ``path`` into {text, title, pages, format}. Raises IngestError with an
    actionable message on unsupported types, missing parsers, or unreadable files."""
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        raise IngestError(f"File not found: {p}")
    if ext in _TEXT_EXTS:
        return _read_text(p)
    parser = _PARSERS.get(ext)
    if parser is None:
        raise IngestError(
            f"Unsupported file type '{ext or '(none)'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
    result = parser(p)
    if not (result.get("text") or "").strip():
        raise IngestError(
            f"No extractable text found in '{p.name}' "
            f"(it may be a scanned/image-only document).")
    return result


def extract_text(path) -> str:
    """Convenience wrapper around extract() for callers that only want the text.
    Raises IngestError on the same conditions."""
    return extract(path)["text"]


# --------------------------- Bulk parsing ---------------------------

def _extract_one(path_str: str) -> dict:
    """Process-pool entry point: must be a module-level function so it pickles under
    Windows' spawn start method. Returns a result envelope instead of raising, because
    an exception crossing the pool boundary loses its type."""
    try:
        out = extract(path_str)
        out["path"] = path_str
        out["ok"] = True
        return out
    except Exception as e:
        return {"ok": False, "path": path_str, "error": str(e)}


def extract_many(paths, workers: int = None, on_progress=None, should_stop=None) -> list:
    """Parse many documents at once. Returns ``[{ok, path, ...}]`` in INPUT order.

    Uses a process pool: pypdf/ebooklib text extraction is almost entirely pure-Python,
    so it is GIL-bound and threads barely help — processes scale it across cores, which
    is the difference between minutes and seconds on a shelf of ebooks. Falls back to a
    thread pool when a process pool can't start (frozen builds, restricted sandboxes),
    which still keeps the UI responsive and progress flowing.

    ``on_progress(done, total, name)`` fires as each file lands, in completion order.
    A file that fails to parse yields ``{"ok": False, "error": …}`` rather than
    aborting the batch — one broken PDF must not cost the user the other nine.

    ``should_stop()`` is polled as each file lands; when it goes true the remaining
    work is cancelled and those entries come back ``{"ok": False, "error": "cancelled"}``.
    Without it the Stop button on the file-parse progress bar was decorative — a shelf
    of ebooks kept parsing for minutes after the user called it off.
    """
    paths = [str(p) for p in (paths or [])]
    if not paths:
        return []
    stopped = False

    def halt():
        nonlocal stopped
        if not stopped and should_stop is not None and should_stop():
            stopped = True
        return stopped

    total = len(paths)
    results = [None] * total
    index = {p: i for i, p in enumerate(paths)}
    done = 0

    def land(res):
        nonlocal done
        i = index.get(res.get("path"))
        if i is not None:
            results[i] = res
        done += 1
        if on_progress is not None:
            try:
                on_progress(done, total, Path(res.get("path", "")).name)
            except Exception:
                pass

    if workers is None:
        workers = min(os.cpu_count() or 2, total, 8)
    workers = max(1, int(workers))

    if halt():
        pass
    elif workers > 1 and total > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_extract_one, p) for p in paths]
                for fut in as_completed(futures):
                    land(fut.result())
                    if halt():
                        # Kills queued tasks; the ones already running still finish,
                        # which is why the caller sees the stop within ~one file.
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
        except Exception:
            # Process pool unavailable — redo whatever is still missing on threads.
            # Not if we stopped on purpose, though: retrying there would defeat Stop.
            todo = [] if stopped else [p for p in paths if results[index[p]] is None]
            done = total - len(todo)
            with ThreadPoolExecutor(max_workers=min(workers, len(todo) or 1)) as ex:
                futures = [ex.submit(_extract_one, p) for p in todo]
                for fut in as_completed(futures):
                    land(fut.result())
                    if halt():
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        for p in paths:
            if halt():
                break
            land(_extract_one(p))

    filler = "cancelled" if stopped else "not parsed"
    return [r if r is not None else {"ok": False, "path": p, "error": filler}
            for p, r in zip(paths, results)]
