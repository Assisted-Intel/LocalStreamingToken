#!/usr/bin/env python3
"""
Local Streaming Token — persona import / export.

Export:
  * XML-only  — just persona.xml (definition travels; empty knowledge/memories).
  * Bundle    — a .zip of persona.xml + sources/ originals + memories/entries/*.json.
                LanceDB/DuckDB rows are NEVER shipped (embeddings are model-specific).

Import re-derives the vector store locally: bundle sources are re-ingested and memories
re-embedded with the *local* embedding model. All archive paths are validated (no
absolute paths, no ``..`` traversal) and the upload is size-capped.
"""

import io
import json
import os
import zipfile
from pathlib import Path

from . import core, ingest, persona as persona_mod, persona_store, rag

MAX_BUNDLE_BYTES = 500 * 1024 * 1024          # 500 MB uncompressed cap
_ALLOWED_PREFIXES = ("persona.xml", "sources/", "memories/entries/")


# --------------------------- Export ---------------------------

def export_xml(persona_id: str) -> bytes:
    """The definition alone, as plaintext persona.xml bytes. The recipient gets the
    profile, speaking style, and pipeline but an empty knowledge base — use
    export_bundle() when the documents and memories should travel too."""
    d = persona_mod.persona_path(persona_id)
    xmlf = d / "persona.xml"
    if not xmlf.is_file():
        raise persona_mod.PersonaError(f"Persona {persona_id!r} not found.")
    # Exports are plaintext + portable: decrypt the at-rest file before shipping it.
    return core.read_bytes(xmlf)


def export_bundle(persona_id: str) -> bytes:
    """Zip persona.xml + sources/ + memories/entries/*.json into an in-memory archive."""
    d = persona_mod.persona_path(persona_id)
    if not (d / "persona.xml").is_file():
        raise persona_mod.PersonaError(f"Persona {persona_id!r} not found.")
    # Exports are plaintext + portable: decrypt each at-rest file (via core.read_bytes)
    # into the archive rather than copying the encrypted bytes.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("persona.xml", core.read_bytes(d / "persona.xml"))
        src = d / "sources"
        if src.is_dir():
            for f in src.iterdir():
                if f.is_file():
                    zf.writestr(f"sources/{f.name}", core.read_bytes(f))
        entries = d / "memories" / "entries"
        if entries.is_dir():
            for f in entries.glob("*.json"):
                zf.writestr(f"memories/entries/{f.name}", core.read_bytes(f))
    return buf.getvalue()


# --------------------------- Import safety ---------------------------

def _safe_member(name: str) -> bool:
    """Whitelist check for one archive entry, applied to EVERY member before anything
    is extracted. Rejects absolute paths, drive letters, and ``..`` traversal, and
    accepts only the three locations a bundle is allowed to contain. This is the guard
    against a malicious .zip writing outside the persona folder."""
    n = name.replace("\\", "/")
    if n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return False
    if ".." in n.split("/"):
        return False
    return n == "persona.xml" or n.startswith("sources/") or n.startswith("memories/entries/")


# --------------------------- Import ---------------------------

def _finalize(persona: dict, psvc: "persona_store") -> dict:
    """Assign a unique id from the persona's name and persist persona.xml."""
    persona["id"] = ""
    pid = psvc._unique_id(persona.get("profile", {}).get("name", "persona"))
    persona["id"] = pid
    return persona


def import_xml(xml_bytes: bytes, psvc) -> dict:
    """Import a definition-only persona under a fresh, non-colliding id.
    ``embedding_model_used`` is cleared because nothing has been indexed yet."""
    persona = persona_mod.from_xml(xml_bytes.decode("utf-8", errors="replace"))
    persona = _finalize(persona, psvc)
    # XML-only: no knowledge attached yet.
    persona.setdefault("stores", {})["embedding_model_used"] = ""
    return psvc.save(persona)


def import_bundle(zip_bytes: bytes, psvc, kbsvc, memsvc, *, embed_url, embed_model_default,
                  wants_vectors=True, contextualize=None) -> dict:
    """Recreate a persona from a bundle, re-ingesting sources and re-embedding memories
    with the local embedding model. Returns the saved persona dict.

    Vectors are rebuilt rather than shipped because embeddings are only comparable
    within the model that produced them — a bundle from a machine using a different
    embedding model would otherwise retrieve nonsense.

    The archive is validated before anything is written: total uncompressed size is
    capped (zip-bomb guard) and every member must pass ``_safe_member``. Individual
    source files that fail to ingest are skipped so one unreadable document doesn't
    lose the whole import; structural problems raise PersonaError.

    Pass ``wants_vectors=False`` to import the files without embedding them (fast,
    and useful when no embedding model is available yet)."""
    if len(zip_bytes) > MAX_BUNDLE_BYTES:
        raise persona_mod.PersonaError("Bundle too large.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise persona_mod.PersonaError("Not a valid .zip bundle.")
    names = zf.namelist()
    total = 0
    for info in zf.infolist():
        if not _safe_member(info.filename):
            raise persona_mod.PersonaError(f"Unsafe path in bundle: {info.filename!r}")
        total += info.file_size
        if total > MAX_BUNDLE_BYTES:
            raise persona_mod.PersonaError("Bundle contents exceed size cap.")
    if "persona.xml" not in names:
        raise persona_mod.PersonaError("Bundle is missing persona.xml.")

    persona = persona_mod.from_xml(zf.read("persona.xml").decode("utf-8", errors="replace"))
    persona = _finalize(persona, psvc)
    embed_model = persona.get("models", {}).get("embedding_model") or embed_model_default

    from . import core
    _embed = core.OllamaClient(embed_url)

    def embed_fn(texts):
        try:
            return _embed.embed(embed_model, texts)
        except Exception:
            return [[] for _ in texts]

    # Persist the definition (creates the folder + subdirs).
    psvc.save(persona)
    pid = persona["id"]
    d = persona_mod.persona_path(pid)

    # Sources → write into sources/ then re-ingest each.
    for name in names:
        if name.startswith("sources/") and not name.endswith("/"):
            fname = os.path.basename(name)
            if not fname:
                continue
            core.write_bytes(d / "sources" / fname, zf.read(name))
    for f in (d / "sources").iterdir() if (d / "sources").is_dir() else []:
        if f.is_file() and ingest.is_supported(f):
            try:
                kbsvc.ingest_file(pid, f, embed_fn if wants_vectors else None,
                                  embed_model, contextualize=contextualize)
            except Exception:
                pass

    # Memories → write JSON + re-embed.
    for name in names:
        if name.startswith("memories/entries/") and name.endswith(".json"):
            try:
                mem = json.loads(zf.read(name).decode("utf-8", errors="replace"))
            except Exception:
                continue
            memsvc.save_memory(pid, mem, embed_fn if wants_vectors else None, embed_model)

    if wants_vectors:
        persona.setdefault("stores", {})["embedding_model_used"] = embed_model
        psvc.save(persona)
    return persona


def import_path(path, psvc, kbsvc, memsvc, *, embed_url, embed_model_default,
                wants_vectors=True, contextualize=None) -> dict:
    """Import from a path on disk, dispatching on the extension: .zip is a bundle,
    anything else is treated as definition-only XML."""
    p = Path(path)
    data = p.read_bytes()
    if p.suffix.lower() == ".zip":
        return import_bundle(data, psvc, kbsvc, memsvc, embed_url=embed_url,
                             embed_model_default=embed_model_default,
                             wants_vectors=wants_vectors, contextualize=contextualize)
    return import_xml(data, psvc)
