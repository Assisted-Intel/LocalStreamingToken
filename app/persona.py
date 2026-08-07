#!/usr/bin/env python3
"""
Local Streaming Token — Persona model + service.

A persona is a portable folder under the active data profile:

    personas/<id>/
        persona.xml          ← versioned definition (this module reads/writes it)
        sources/             ← original ingested documents (for re-export/re-index)
        knowledge/           ← (DuckDB rows are namespaced in the shared rag store, not here)
        memories/entries/    ← one JSON file per memory (portable source of truth)

Only ``persona.xml`` + ``sources/`` + ``memories/entries/`` are meant to travel; the
vector rows live in the shared ``rag.duckdb`` namespaced by persona id (see rag.py),
so they're a rebuildable cache and never shipped.

Personas are represented in Python as plain dicts (matching the rest of the app) and
serialized to versioned XML via stdlib ElementTree — no lxml. Store paths are always
relative and validated (no ``..`` / absolute) so a malicious persona can't escape its
folder.
"""

import json
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from . import core

SCHEMA_VERSION = "1"


class PersonaError(Exception):
    """Raised with a clear message on invalid persona XML or unsafe paths."""


# --------------------------- id / path safety ---------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe folder id. Never returns empty —
    a name with no usable characters becomes "persona"."""
    s = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return s or "persona"


def _safe_id(persona_id: str) -> str:
    """Reject anything that isn't a plain folder name (defense against traversal)."""
    pid = (persona_id or "").strip()
    if not pid or pid in (".", "..") or "/" in pid or "\\" in pid or ":" in pid:
        raise PersonaError(f"Invalid persona id: {persona_id!r}")
    return pid


def _rel_ok(path: str) -> bool:
    """A store path must be relative and stay inside the persona folder."""
    if not path:
        return False
    p = str(path).replace("\\", "/")
    if p.startswith("/") or ".." in p.split("/") or (len(p) > 1 and p[1] == ":"):
        return False
    return True


def personas_dir() -> Path:
    """The personas folder of the ACTIVE data profile, created if missing.
    Reads core.PERSONAS_DIR at call time so it follows profile switches."""
    d = Path(core.PERSONAS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def persona_path(persona_id: str) -> Path:
    """Folder for one persona. Raises PersonaError if the id isn't a plain name."""
    return personas_dir() / _safe_id(persona_id)


# --------------------------- default definition ---------------------------

def default_pipeline() -> list:
    """The default 5-step chain-of-thought (architecture §7). llm steps carry a JSON
    schema (structured output); retrieval steps are direct queries (no LLM)."""
    analyze_schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "knowledge_queries": {"type": "array", "items": {"type": "string"}},
            "memory_queries": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "response_goals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["intent", "knowledge_queries"],
    }
    synth_schema = {
        "type": "object",
        "properties": {
            "draft": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["draft"],
    }
    return [
        {"id": "analyze", "type": "llm", "use_history": True, "model": "",
         "schema": analyze_schema,
         "prompt": (
             "You are the reasoning core of {persona_name}. Analyze the user's message "
             "and conversation to determine intent and what to retrieve. Produce 2-3 "
             "self-contained retrieval queries for the knowledge base and (if relevant) "
             "for personal memories, resolving pronouns from history, plus keywords and "
             "the goals a good response should meet.\n\nUser message:\n{user_message}")},
        {"id": "retrieve_knowledge", "type": "knowledge_retrieval", "use_history": False,
         "model": "", "schema": None, "prompt": ""},
        {"id": "retrieve_memories", "type": "memory_retrieval", "use_history": False,
         "model": "", "schema": None, "prompt": ""},
        {"id": "synthesize", "type": "llm", "use_history": True, "model": "",
         "schema": synth_schema,
         "prompt": (
             "Using the retrieved knowledge and memories below, draft an accurate answer "
             "to the user's message. Cite which sources support each claim; if the answer "
             "isn't supported, say so.\n\nRetrieved knowledge:\n{knowledge}\n\n"
             "Retrieved memories:\n{memories}\n\nUser message:\n{user_message}")},
        {"id": "stylize", "type": "llm", "use_history": True, "model": "",
         "schema": None,
         "prompt": (
             "Rewrite the draft in the authentic voice of {persona_name}, applying the "
             "speaking style below. Keep the substance; change only the delivery.\n\n"
             "{speaking_style}\n\nDraft:\n{draft}")},
    ]


def default_persona(name: str, chat_model: str = "", embedding_model: str = "nomic-embed-text") -> dict:
    """A complete, valid persona dict for a brand-new persona. The id is left empty
    and assigned by PersonaService.create(); empty model names mean "use the chat's
    model" so a persona stays portable across machines with different models."""
    return {
        "id": "",                     # assigned on create()
        "version": SCHEMA_VERSION,
        "profile": {"name": name, "bio": "", "role": "", "avatar": ""},
        "speaking": {
            "tone": "", "formality": "", "vocabulary": "", "quirks": "",
            "variants": [],           # [{"name","description"}]
            "examples": [],           # [{"user","reply"}]
        },
        "models": {"chat_model": chat_model, "embedding_model": embedding_model,
                   "temperature": 0.7},
        "stores": {
            "knowledge": "knowledge/", "memories": "memories/", "sources": "sources/",
            "retrieval": "hybrid", "prompt_reword": True,
            "embedding_model_used": "",   # recorded on first ingest (re-index on mismatch)
        },
        "pipeline": default_pipeline(),
        "updated": datetime.utcnow().isoformat(),
    }


# --------------------------- XML (de)serialization ---------------------------

def _text(parent, tag, value):
    """Append a <tag>value</tag> child. None becomes an empty element, so a round trip
    through XML never turns a missing field into the string "None"."""
    el = ET.SubElement(parent, tag)
    el.text = "" if value is None else str(value)
    return el


def to_xml(persona: dict) -> str:
    """Serialize a persona dict to versioned, indented persona.xml text.
    Inverse of from_xml(); the ``id`` is NOT stored (the folder name is the id)."""
    root = ET.Element("persona", {"version": str(persona.get("version") or SCHEMA_VERSION)})

    prof = persona.get("profile", {})
    pe = ET.SubElement(root, "profile")
    for k in ("name", "bio", "role", "avatar"):
        _text(pe, k, prof.get(k, ""))

    sp = persona.get("speaking", {})
    se = ET.SubElement(root, "speaking")
    for k in ("tone", "formality", "vocabulary", "quirks"):
        _text(se, k, sp.get(k, ""))
    for v in sp.get("variants", []):
        ve = ET.SubElement(se, "variant", {"name": str(v.get("name", ""))})
        ve.text = str(v.get("description", ""))
    for ex in sp.get("examples", []):
        ee = ET.SubElement(se, "example")
        _text(ee, "user", ex.get("user", ""))
        _text(ee, "reply", ex.get("reply", ""))

    md = persona.get("models", {})
    me = ET.SubElement(root, "models")
    _text(me, "chat_model", md.get("chat_model", ""))
    _text(me, "embedding_model", md.get("embedding_model", ""))
    _text(me, "temperature", md.get("temperature", 0.7))

    st = persona.get("stores", {})
    # ET rejects a non-str attribute value with a TypeError, so a None anywhere in
    # ``stores`` would 500 the save instead of failing validation cleanly below.
    def _attr(key, default):
        return str(st.get(key) or default)
    ET.SubElement(root, "stores", {
        "knowledge": _attr("knowledge", "knowledge/"),
        "memories": _attr("memories", "memories/"),
        "sources": _attr("sources", "sources/"),
        "retrieval": _attr("retrieval", "hybrid"),
        "prompt_reword": "true" if st.get("prompt_reword", True) else "false",
        "embedding_model_used": str(st.get("embedding_model_used") or ""),
    })

    pipe = ET.SubElement(root, "pipeline")
    for step in persona.get("pipeline", []):
        attrs = {"id": str(step.get("id", "")), "type": str(step.get("type", "llm")),
                 "use_history": "true" if step.get("use_history") else "false"}
        if step.get("model"):
            attrs["model"] = str(step["model"])
        se2 = ET.SubElement(pipe, "step", attrs)
        pr = ET.SubElement(se2, "prompt")
        pr.text = step.get("prompt", "") or ""
        schema = step.get("schema")
        if schema is not None:
            sc = ET.SubElement(se2, "schema")
            sc.text = json.dumps(schema)

    _indent(root)
    return ET.tostring(root, encoding="unicode")


def _indent(elem, level=0):
    """Pretty-print an ElementTree in place. Stdlib ET has no indent() before Python
    3.9, and persona.xml is meant to stay human-editable, so we do it by hand."""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def from_xml(xml_text: str) -> dict:
    """Parse persona.xml into a persona dict, validating as it goes. Inverse of
    to_xml(). Raises PersonaError on malformed XML, a version mismatch, a missing
    profile name, an unknown step type, a bad JSON schema, or a store path that
    would escape the persona folder. An unknown retrieval mode falls back to
    "hybrid" rather than failing — that is recoverable; the rest are not."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise PersonaError(f"persona.xml is not valid XML: {e}")
    if root.tag != "persona":
        raise PersonaError("Root element must be <persona>.")
    version = root.get("version") or ""
    if version != SCHEMA_VERSION:
        raise PersonaError(f"Unsupported persona version {version!r} (expected {SCHEMA_VERSION}).")

    def gtext(parent, tag, default=""):
        el = parent.find(tag) if parent is not None else None
        return (el.text if el is not None and el.text is not None else default)

    prof_el = root.find("profile")
    if prof_el is None or not (gtext(prof_el, "name").strip()):
        raise PersonaError("Persona requires a <profile> with a non-empty <name>.")
    profile = {k: gtext(prof_el, k, "") for k in ("name", "bio", "role", "avatar")}

    sp_el = root.find("speaking")
    speaking = {"tone": "", "formality": "", "vocabulary": "", "quirks": "",
                "variants": [], "examples": []}
    if sp_el is not None:
        for k in ("tone", "formality", "vocabulary", "quirks"):
            speaking[k] = gtext(sp_el, k, "")
        for ve in sp_el.findall("variant"):
            speaking["variants"].append({"name": ve.get("name", ""), "description": ve.text or ""})
        for ee in sp_el.findall("example"):
            speaking["examples"].append({"user": gtext(ee, "user", ""), "reply": gtext(ee, "reply", "")})

    md_el = root.find("models")
    try:
        temp = float(gtext(md_el, "temperature", "0.7") or 0.7)
    except Exception:
        temp = 0.7
    models = {"chat_model": gtext(md_el, "chat_model", ""),
              "embedding_model": gtext(md_el, "embedding_model", "nomic-embed-text"),
              "temperature": temp}

    st_el = root.find("stores")
    st_attr = st_el.attrib if st_el is not None else {}
    for key in ("knowledge", "memories", "sources"):
        val = st_attr.get(key, key + "/")
        if not _rel_ok(val):
            raise PersonaError(f"Unsafe store path for {key}: {val!r}")
    retrieval = (st_attr.get("retrieval") or "hybrid").lower()
    if retrieval not in ("vector", "keyword", "hybrid"):
        retrieval = "hybrid"
    stores = {
        "knowledge": st_attr.get("knowledge", "knowledge/"),
        "memories": st_attr.get("memories", "memories/"),
        "sources": st_attr.get("sources", "sources/"),
        "retrieval": retrieval,
        "prompt_reword": str(st_attr.get("prompt_reword", "true")).lower() != "false",
        "embedding_model_used": st_attr.get("embedding_model_used", ""),
    }

    pipeline = []
    pipe_el = root.find("pipeline")
    if pipe_el is not None:
        for step_el in pipe_el.findall("step"):
            stype = step_el.get("type", "llm")
            if stype not in ("llm", "knowledge_retrieval", "memory_retrieval"):
                raise PersonaError(f"Unknown pipeline step type: {stype!r}")
            schema = None
            sc_el = step_el.find("schema")
            if sc_el is not None and (sc_el.text or "").strip():
                try:
                    schema = json.loads(sc_el.text)
                except Exception as e:
                    raise PersonaError(f"Step {step_el.get('id')!r} has invalid JSON schema: {e}")
            pipeline.append({
                "id": step_el.get("id", ""),
                "type": stype,
                "model": step_el.get("model", ""),
                "use_history": str(step_el.get("use_history", "false")).lower() == "true",
                "prompt": gtext(step_el, "prompt", ""),
                "schema": schema,
            })

    return {"id": "", "version": SCHEMA_VERSION, "profile": profile, "speaking": speaking,
            "models": models, "stores": stores, "pipeline": pipeline,
            "updated": datetime.utcnow().isoformat()}


# --------------------------- Service ---------------------------

class PersonaService:
    """CRUD for personas, backed by ``personas/<id>/persona.xml`` under the active data
    profile. Reads ``core.PERSONAS_DIR`` at call time so it follows profile switches."""

    def _dir(self, persona_id: str) -> Path:
        """Validated folder path for a persona id."""
        return persona_path(persona_id)

    def list_all(self) -> list:
        """Lightweight summaries for every persona folder that has a persona.xml.

        A folder whose XML no longer parses is reported with ``broken`` set rather than
        skipped. Hiding it made a corrupted persona unreachable from the UI — invisible,
        but still holding its sources and memories, and still occupying its id. Listing
        it keeps Delete (and a readable error) available."""
        out = []
        base = personas_dir()
        for child in sorted(base.iterdir()) if base.exists() else []:
            xmlf = child / "persona.xml"
            if not xmlf.is_file():
                continue
            try:
                p = from_xml(core.read_text(xmlf))
            except Exception as e:
                out.append({"id": child.name, "name": child.name, "role": "",
                            "variants": [], "broken": True, "error": str(e)})
                continue
            p["id"] = child.name
            out.append({"id": child.name, "name": p["profile"]["name"],
                        "role": p["profile"].get("role", ""),
                        "variants": [v["name"] for v in p["speaking"]["variants"]]})
        return out

    def load(self, persona_id: str) -> dict:
        """Read one persona in full. Raises PersonaError if it doesn't exist or its
        XML is invalid."""
        d = self._dir(persona_id)
        xmlf = d / "persona.xml"
        if not xmlf.is_file():
            raise PersonaError(f"Persona {persona_id!r} not found.")
        p = from_xml(core.read_text(xmlf))
        p["id"] = _safe_id(persona_id)
        return p

    def save(self, persona: dict) -> dict:
        """Write persona.xml, scaffolding the standard subfolders. Derives the id from
        the name when absent, stamps ``updated``, and returns the persona with both
        fields set. Overwrites an existing definition at the same id.

        The XML is validated by round-tripping it through ``from_xml`` BEFORE anything
        touches disk. ``from_xml`` rejects an empty name, an unsafe store path, an
        unknown step type, and a malformed schema — but ``save`` used to accept all of
        them, so saving a persona with its name cleared wrote a file that could never be
        read back: ``load`` raised and ``list_all`` skipped it, stranding the persona's
        knowledge base in an invisible folder. A write the read path would reject now
        raises PersonaError instead, leaving the previous definition intact."""
        if not (persona.get("profile", {}).get("name") or "").strip():
            raise PersonaError("Persona name is required.")
        pid = _safe_id(persona.get("id") or slugify(persona.get("profile", {}).get("name", "")))
        persona["id"] = pid
        # We always write the current schema, so don't let a stale version field on the
        # incoming dict produce a file from_xml would refuse.
        persona["version"] = SCHEMA_VERSION
        persona["updated"] = datetime.utcnow().isoformat()
        try:
            xml = to_xml(persona)
            from_xml(xml)
        except PersonaError:
            raise
        except Exception as e:
            raise PersonaError(f"Persona could not be serialized: {e}")
        d = self._dir(pid)
        # Scaffold the standard subfolders.
        for sub in ("sources", "memories/entries"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        core.write_text(d / "persona.xml", xml)
        return persona

    def _unique_id(self, base_slug: str) -> str:
        """Slug of ``base_slug``, suffixed -2, -3, … until it names a free folder."""
        base = slugify(base_slug)
        d = personas_dir()
        if not (d / base).exists():
            return base
        i = 2
        while (d / f"{base}-{i}").exists():
            i += 1
        return f"{base}-{i}"

    def create(self, name: str, chat_model: str = "", embedding_model: str = "nomic-embed-text") -> dict:
        """Create and persist a new persona with the default pipeline. Raises
        PersonaError on an empty name."""
        if not (name and name.strip()):
            raise PersonaError("Persona name is required.")
        persona = default_persona(name.strip(), chat_model, embedding_model)
        persona["id"] = self._unique_id(name)
        return self.save(persona)

    def duplicate(self, persona_id: str, new_name: str = "") -> dict:
        """Copy a persona whole — sources and memories included — under a new id.
        The vector rows are NOT copied; they are namespaced by persona id in the
        shared RAG store and get rebuilt on the copy's first use."""
        src = self.load(persona_id)
        src_dir = self._dir(persona_id)
        new_name = (new_name or (src["profile"]["name"] + " copy")).strip()
        new_id = self._unique_id(new_name)
        dst_dir = self._dir(new_id)
        # Copy the whole folder (sources + memories travel), then rewrite the definition.
        shutil.copytree(src_dir, dst_dir)
        src["id"] = new_id
        src["profile"]["name"] = new_name
        core.write_text(dst_dir / "persona.xml", to_xml(src))
        src["updated"] = datetime.utcnow().isoformat()
        return src

    def delete(self, persona_id: str) -> bool:
        """Remove a persona's folder and everything in it. Returns False if it was
        already gone. Callers are responsible for dropping its rows from the RAG
        store (see persona_store)."""
        d = self._dir(persona_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False
