#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Native OS file/folder pickers via tkinter. Because tkinter must own its own event
loop and does not cooperate with Flask's request threads, the dialogs are launched
in a *separate subprocess* (``python -m app.native_dialog <mode>``). The subprocess
opens the picker, prints the chosen path(s) as a ``RESULT:<json>`` line to stdout,
and exits. The server (below, in the helper functions) shells out and parses that.

The browser therefore only ever exchanges *paths* with the server — file contents
are read/written directly on disk by the server, never uploaded.

Modes:
    folder      -> ask for a directory            -> "" or "/path"
    open-files  -> ask for one or more files      -> [] or ["/a", "/b"]
    save-file   -> ask for a save destination     -> "" or "/path"
"""

import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_PREFIX = "RESULT:"


# =====================================================================
# Subprocess entry point (runs the actual tkinter dialog)
# =====================================================================

def _run_dialog(mode, title=None, default_name=None, filetypes_key=None):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    # Nudge focus so the dialog appears in front of the browser.
    try:
        root.update()
    except Exception:
        pass

    xml_types = [("XML library files", "*.xml"), ("All files", "*.*")]
    json_types = [("JSON files", "*.json"), ("All files", "*.*")]
    text_types = [
        ("Text/Markdown/CSV/JSON", "*.txt *.md *.csv *.json"),
        ("All files", "*.*"),
    ]
    doc_types = [
        ("Documents", "*.txt *.md *.csv *.json *.pdf *.epub *.docx"),
        ("PDF", "*.pdf"), ("EPUB", "*.epub"), ("Word", "*.docx"),
        ("Text/Markdown/CSV/JSON", "*.txt *.md *.csv *.json"),
        ("All files", "*.*"),
    ]

    image_types = [
        ("Images", "*.png *.jpg *.jpeg *.gif *.webp *.bmp *.tif *.tiff "
                   "*.heic *.heif *.avif *.ico"),
        ("PNG / JPEG", "*.png *.jpg *.jpeg"),
        ("Camera photos (HEIC/HEIF)", "*.heic *.heif"),
        ("All files", "*.*"),
    ]

    # Mirrors transcribe.SUPPORTED_EXTS. Kept as a literal rather than imported: this
    # function runs in the tkinter subprocess, which must not pay for app imports.
    media_types = [
        ("Audio / video", "*.mp3 *.m4a *.m4b *.aac *.ogg *.oga *.opus *.flac *.wav "
                          "*.wma *.mp4 *.m4v *.mkv *.webm *.mov *.avi"),
        ("Audio", "*.mp3 *.m4a *.m4b *.aac *.ogg *.oga *.opus *.flac *.wav *.wma"),
        ("Video", "*.mp4 *.m4v *.mkv *.webm *.mov *.avi"),
        ("All files", "*.*"),
    ]

    def _open_ftypes():
        if filetypes_key == "xml":
            return xml_types
        if filetypes_key == "json":
            return json_types
        if filetypes_key == "documents":
            return doc_types
        if filetypes_key == "images":
            return image_types
        if filetypes_key == "media":
            return media_types
        if filetypes_key == "exe":
            return [("Programs", "*.exe"), ("All files", "*.*")]
        return text_types

    result = ""
    if mode == "folder":
        result = filedialog.askdirectory(
            title=title or "Choose a folder", mustexist=True)
    elif mode == "open-files":
        paths = filedialog.askopenfilenames(
            title=title or "Choose file(s)", filetypes=_open_ftypes())
        result = list(paths)
    elif mode == "save-file":
        if filetypes_key == "xml":
            ftypes, defext = xml_types, ".xml"
        elif filetypes_key == "json":
            ftypes, defext = json_types, ".json"
        else:
            ftypes, defext = [("All files", "*.*")], ""
        result = filedialog.asksaveasfilename(
            title=title or "Save as",
            defaultextension=defext,
            initialfile=default_name or "",
            filetypes=ftypes,
        )

    try:
        root.destroy()
    except Exception:
        pass

    print(RESULT_PREFIX + json.dumps(result))


def _main(argv):
    mode = argv[0] if argv else "folder"
    title = None
    default_name = None
    filetypes_key = None
    # Simple positional/keyword parsing: --title=, --default=, --filetypes=
    for a in argv[1:]:
        if a.startswith("--title="):
            title = a[len("--title="):]
        elif a.startswith("--default="):
            default_name = a[len("--default="):]
        elif a.startswith("--filetypes="):
            filetypes_key = a[len("--filetypes="):]
    _run_dialog(mode, title=title, default_name=default_name, filetypes_key=filetypes_key)


# =====================================================================
# Server-side helpers (call the subprocess, parse the result)
# =====================================================================

def _invoke(mode, title=None, default_name=None, filetypes_key=None, timeout=300):
    args = [sys.executable, "-m", "app.native_dialog", mode]
    if title:
        args.append(f"--title={title}")
    if default_name:
        args.append(f"--default={default_name}")
    if filetypes_key:
        args.append(f"--filetypes={filetypes_key}")
    try:
        proc = subprocess.run(
            args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(f"Native dialog failed to launch: {e}")

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except Exception:
                break
    # No result line -> treat as cancelled/empty.
    return "" if mode != "open-files" else []


def pick_folder(title="Choose a folder"):
    """Return a chosen directory path, or '' if cancelled."""
    return _invoke("folder", title=title) or ""


def pick_files(title="Choose file(s)", filetypes_key=None):
    """Return a list of chosen file paths (possibly empty)."""
    res = _invoke("open-files", title=title, filetypes_key=filetypes_key)
    return list(res) if isinstance(res, list) else []


def save_file(title="Save as", default_name="", filetypes_key=None):
    """Return a chosen save path, or '' if cancelled."""
    return _invoke("save-file", title=title, default_name=default_name,
                   filetypes_key=filetypes_key) or ""


if __name__ == "__main__":
    _main(sys.argv[1:])
