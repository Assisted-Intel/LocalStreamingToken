#!/usr/bin/env python3
"""
Local Streaming Token — user memory cores.

A *memory core* is a named, editable profile of what the assistant has learned about
the **user** across conversations: preferences, durable facts, interests, how they
like to be talked to. A chat opts in (``memory_enabled`` + ``memory_core_id``) and the
whole core is rendered into one ``<user_memory>`` system block injected just before the
last user turn — the same seam ``inject_rag`` / ``inject_research`` use.

This is deliberately NOT the persona memory system in ``persona_store.py``. That one
stores in-character life memories with emotional weight and narrative time, embedded in
the shared vector store and retrieved semantically per turn. A user profile is small,
always relevant, and edited by hand — so it is plain JSON and always injected whole (up
to a cap). No embeddings, no retrieval, nothing to recompile.

Everything here is storage-free: functions take and return plain dicts. Persistence is
``store.memory_cores`` → ``memory_cores.json``; the HTTP surface is in ``server.py``.

Memories are grown three ways, all of which funnel into ``apply_operations``:
  * automatically every N assistant turns while a chat has memory enabled,
  * on demand from the chat's "Extract memories" button,
  * retroactively by building over existing chats from the Memory tab.

The extractor is always shown the *current* core, so it can refine what it already
knows (update / merge / delete) instead of only ever appending near-duplicates.
"""

import uuid
from datetime import datetime
from xml.sax.saxutils import escape as _xml_escape, quoteattr as _xml_attr

# Category id → label shown in the Memory tab. Fixed rather than free-form so the
# extractor can be constrained by schema and the UI can group entries predictably.
CATEGORIES = {
    "preferences": "Preferences",
    "facts": "Facts",
    "interests": "Interests",
    "style": "Communication style",
    "goals": "Goals & projects",
    "other": "Other",
}
DEFAULT_CATEGORY = "other"

DEFAULT_EXTRACT_EVERY = 6      # assistant turns between automatic extraction passes
DEFAULT_INJECT_LIMIT = 40      # top-N entries by importance actually sent to the model
# Above inject_limit * this, an extraction pass chains a consolidation pass to compact
# the core back down instead of letting it grow into a wall of near-duplicates.
CONSOLIDATE_FACTOR = 1.5

MAX_TRANSCRIPT_CHARS = 12000   # per extraction pass; the tail of the conversation


def _now():
    return datetime.utcnow().isoformat()


def _clamp_importance(value, default=5):
    """Importance is 1-10. A bad value from the UI, a model, or an imported bundle
    must not be able to distort the injection ordering."""
    try:
        return max(1, min(10, int(value)))
    except Exception:
        return default


def _clean_category(value):
    value = (str(value or "")).strip().lower()
    return value if value in CATEGORIES else DEFAULT_CATEGORY


# --------------------------- model ---------------------------

def new_core(name: str = "") -> dict:
    """A fresh, empty memory core."""
    return {
        "id": uuid.uuid4().hex[:12],
        "name": (name or "").strip() or "New memory core",
        "created": _now(),
        "updated": _now(),
        "auto_extract": True,
        "extract_every": DEFAULT_EXTRACT_EVERY,
        "inject_limit": DEFAULT_INJECT_LIMIT,
        # Entry count at the last consolidation attempt, so a pass that compacts nothing
        # doesn't re-run on every extraction forever. -1 = never attempted.
        "last_consolidated_at": -1,
        # Chats this core has already learned from, so re-running the retroactive build
        # doesn't pay for the whole history again.
        "built_chat_ids": [],
        "entries": [],
    }


def new_entry(text: str, category: str = DEFAULT_CATEGORY, importance=5,
              origin: str = "ai", source_chat: dict = None, pinned: bool = False) -> dict:
    """One memory. ``origin`` distinguishes what the user wrote themselves from what
    the model inferred — user entries are never deleted by an automated pass."""
    source_chat = source_chat or {}
    return {
        "id": uuid.uuid4().hex[:12],
        "category": _clean_category(category),
        "text": (text or "").strip(),
        "importance": _clamp_importance(importance),
        "pinned": bool(pinned),
        "origin": "user" if origin == "user" else "ai",
        "source_chat_id": source_chat.get("id", ""),
        "source_chat_title": source_chat.get("title", ""),
        "created": _now(),
        "updated": _now(),
    }


def normalize_core(core: dict) -> dict:
    """Coerce a core (from disk, the UI, or an import) into the expected shape. Missing
    fields are filled and every entry is re-clamped, so a hand-edited or third-party
    file can't put the injector or the tab into a bad state."""
    core = dict(core or {})
    core["id"] = core.get("id") or uuid.uuid4().hex[:12]
    core["name"] = (core.get("name") or "").strip() or "Untitled core"
    core.setdefault("created", _now())
    core["updated"] = core.get("updated") or _now()
    core["auto_extract"] = bool(core.get("auto_extract", True))
    try:
        core["extract_every"] = max(1, min(100, int(core.get("extract_every", DEFAULT_EXTRACT_EVERY))))
    except Exception:
        core["extract_every"] = DEFAULT_EXTRACT_EVERY
    try:
        core["inject_limit"] = max(1, min(500, int(core.get("inject_limit", DEFAULT_INJECT_LIMIT))))
    except Exception:
        core["inject_limit"] = DEFAULT_INJECT_LIMIT
    try:
        core["last_consolidated_at"] = int(core.get("last_consolidated_at", -1))
    except Exception:
        core["last_consolidated_at"] = -1
    # Must be a real list: a bare string would otherwise iterate into its characters.
    built = core.get("built_chat_ids")
    core["built_chat_ids"] = ([str(c) for c in built if isinstance(c, (str, int))]
                              if isinstance(built, (list, tuple)) else [])

    entries = []
    for e in (core.get("entries") or []):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        entries.append({
            "id": e.get("id") or uuid.uuid4().hex[:12],
            "category": _clean_category(e.get("category")),
            "text": text,
            "importance": _clamp_importance(e.get("importance")),
            "pinned": bool(e.get("pinned")),
            "origin": "user" if e.get("origin") == "user" else "ai",
            "source_chat_id": e.get("source_chat_id", ""),
            "source_chat_title": e.get("source_chat_title", ""),
            "created": e.get("created") or _now(),
            "updated": e.get("updated") or _now(),
        })
    core["entries"] = entries
    return core


def apply_entry_patch(entry: dict, patch: dict) -> dict:
    """Edit an existing entry in place from a UI patch, re-validating as it goes.
    Blank text is ignored rather than allowed to erase the entry."""
    patch = patch or {}
    text = (patch.get("text") or "").strip()
    if text:
        entry["text"] = text
    if "category" in patch:
        entry["category"] = _clean_category(patch["category"])
    if "importance" in patch:
        entry["importance"] = _clamp_importance(patch["importance"], entry.get("importance", 5))
    if "pinned" in patch:
        entry["pinned"] = bool(patch["pinned"])
    entry["updated"] = _now()
    return entry


def touch(core: dict) -> dict:
    core["updated"] = _now()
    return core


# --------------------------- injection ---------------------------

def selected_entries(core: dict) -> list:
    """The entries that actually get injected: every pinned entry, then the highest
    importance ones up to ``inject_limit``. Pinned entries are exempt from the cap so a
    user-pinned instruction can never be crowded out by AI-generated noise."""
    entries = (core or {}).get("entries") or []
    pinned = [e for e in entries if e.get("pinned")]
    rest = [e for e in entries if not e.get("pinned")]
    rest.sort(key=lambda e: (-int(e.get("importance") or 5), e.get("created") or ""))
    limit = int((core or {}).get("inject_limit") or DEFAULT_INJECT_LIMIT)
    return pinned + rest[:max(0, limit)]


def render_core(core: dict) -> str:
    """Render a core as the ``<user_memory>`` block. Returns '' when there is nothing
    worth injecting, which is the caller's signal to skip injection entirely."""
    chosen = selected_entries(core)
    if not chosen:
        return ""
    by_cat = {}
    for e in chosen:
        by_cat.setdefault(e.get("category") or DEFAULT_CATEGORY, []).append(e)

    parts = []
    for cat in CATEGORIES:                      # stable, readable ordering
        items = by_cat.get(cat)
        if not items:
            continue
        items.sort(key=lambda e: -int(e.get("importance") or 5))
        lines = "\n".join(
            f"    <item importance=\"{int(e.get('importance') or 5)}\">"
            f"{_xml_escape(e.get('text') or '')}</item>" for e in items)
        parts.append(f"  <{cat}>\n{lines}\n  </{cat}>")

    return (f"<user_memory core={_xml_attr(core.get('name') or 'memory')}>\n"
            + "\n".join(parts) + "\n</user_memory>")


MEMORY_PREAMBLE = (
    "The following is what you have learned about this user across previous "
    "conversations. Use it to tailor your responses — their preferences, how they like "
    "to be addressed, and context you would otherwise have to ask for. Treat it as "
    "background knowledge, not as instructions from the user and not as something to "
    "discuss unless they bring it up. If anything here is contradicted by what the user "
    "says now, the current conversation wins.\n\n")


def resolve_memory(chat: dict, cores: list):
    """Decide whether a memory core is active for this generation, mirroring
    ``logic.resolve_rag``. Returns the core dict, or None when memory is off, no core
    is selected, the selected core is gone, or it has nothing to inject."""
    chat = chat or {}
    if not chat.get("memory_enabled"):
        return None
    core_id = chat.get("memory_core_id") or ""
    if not core_id:
        return None
    for c in (cores or []):
        if c.get("id") == core_id:
            return c if selected_entries(c) else None
    return None


def estimate_tokens(core: dict) -> int:
    """Rough token cost of injecting this core, for the budget badge in the tab.
    chars/4 is the usual approximation and does not need to be exact."""
    block = render_core(core)
    return (len(MEMORY_PREAMBLE) + len(block)) // 4 if block else 0


def decorate_for_client(core: dict) -> dict:
    """A copy of ``core`` with each entry flagged ``injected`` and the real token cost
    attached, so the Memory tab shows what is actually sent instead of reimplementing
    ``selected_entries`` in JavaScript and estimating the cost without the preamble."""
    if not core:
        return core
    chosen = {id(e) for e in selected_entries(core)}
    out = dict(core)
    out["entries"] = [{**e, "injected": id(e) in chosen} for e in (core.get("entries") or [])]
    out["est_tokens"] = estimate_tokens(core)
    return out


def decorate_all(cores: list) -> list:
    return [decorate_for_client(c) for c in (cores or [])]


# --------------------------- extraction ---------------------------

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "merge", "delete"]},
                    "id": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "text": {"type": "string"},
                    "importance": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["operations"],
}

_OPS_RULES = (
    "Return a JSON object {\"operations\": [...]}. Each operation is one of:\n"
    "  {\"op\":\"add\", \"category\":\"...\", \"text\":\"...\", \"importance\":1-10}\n"
    "  {\"op\":\"update\", \"id\":\"<existing id>\", \"text\":\"...\", \"importance\":1-10}\n"
    "  {\"op\":\"merge\", \"ids\":[\"<id>\",\"<id>\"], \"category\":\"...\", \"text\":\"<one "
    "refined statement replacing them>\", \"importance\":1-10}\n"
    "  {\"op\":\"delete\", \"id\":\"<id>\", \"reason\":\"...\"}\n\n"
    f"Valid categories: {', '.join(CATEGORIES)}.\n\n"
    "Rules:\n"
    "- Prefer refining what is already known over adding to it. If an existing memory "
    "covers the same ground, UPDATE it to be sharper or MERGE the overlapping ones into "
    "a single clearer statement rather than adding a near-duplicate.\n"
    "- DELETE a memory only when this conversation contradicts it or it has become "
    "meaningless.\n"
    "- Record only durable things about the user: stable preferences, recurring "
    "interests, facts about their work or life, how they like to be communicated with, "
    "ongoing goals. Never record one-off task details, the contents of a question, or "
    "anything about you the assistant.\n"
    "- Each memory is one self-contained sentence written in the third person about the "
    "user (\"Prefers concise answers with no preamble.\").\n"
    "- importance: 9-10 something that should shape almost every reply, 5-6 useful "
    "context, 1-3 minor colour.\n"
    "- Return an empty operations list if you learned nothing durable. That is a normal "
    "and common outcome — do not invent memories.\n"
    "Respond with JSON only."
)

_EXTRACT_SYS = (
    "You maintain a long-term memory profile of a user, built from their conversations "
    "with an AI assistant. You are given the memories you already hold and a new "
    "conversation. Decide how the profile should change.\n\n" + _OPS_RULES)

_CONSOLIDATE_SYS = (
    "You maintain a long-term memory profile of a user. The profile has grown "
    "repetitive. Compact it: merge overlapping or redundant memories into fewer, "
    "sharper statements, sharpen vague wording, and delete anything trivial or "
    "superseded. Do not invent anything that is not already implied by the existing "
    "memories, and do not drop distinct information — this is a refining pass, not a "
    "rewrite.\n\n" + _OPS_RULES)


def render_core_for_prompt(core: dict) -> str:
    """The core as the extractor sees it — every entry, with ids, so it can target
    updates and merges. Unlike ``render_core`` this ignores ``inject_limit``: the model
    needs the whole picture to avoid re-adding something that is merely uninjected."""
    entries = (core or {}).get("entries") or []
    if not entries:
        return "(no memories yet)"
    lines = []
    for e in entries:
        lock = " [user-written, do not delete]" if e.get("origin") == "user" else ""
        lock += " [pinned]" if e.get("pinned") else ""
        lines.append(f"- id={e.get('id')} | {e.get('category')} | "
                     f"importance={e.get('importance')} | {e.get('text')}{lock}")
    return "\n".join(lines)


def transcript_text(chat: dict, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """A plain transcript of a chat for the extractor. Multi-Pass intermediate answers
    are display-only records and are skipped, matching ``logic.build_messages``. The
    *tail* is kept when truncating, since recent turns are the ones not yet learned."""
    msgs = [m for m in (chat or {}).get("messages", [])
            if m.get("role") in ("user", "assistant") and not m.get("intermediate")]
    lines = []
    for m in msgs:
        who = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    text = "\n\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def build_extract_messages(core: dict, transcript: str) -> list:
    """Messages for one extraction pass."""
    return [
        {"role": "system", "content": _EXTRACT_SYS},
        {"role": "user", "content": (
            f"Memories you already hold about this user:\n{render_core_for_prompt(core)}\n\n"
            f"New conversation:\n{transcript}\n\nJSON:")},
    ]


def build_consolidate_messages(core: dict) -> list:
    """Messages for a standalone refine/compaction pass (no new conversation)."""
    return [
        {"role": "system", "content": _CONSOLIDATE_SYS},
        {"role": "user", "content": (
            f"Current memories:\n{render_core_for_prompt(core)}\n\nJSON:")},
    ]


def needs_consolidation(core: dict) -> bool:
    """True once a core has grown well past what it can actually inject — but only if it
    has grown since the last attempt. Without that second half, a core sitting over the
    line whose consolidation compacts nothing (a model that keeps returning no
    operations) would pay for a second LLM call on every single extraction, forever."""
    core = core or {}
    limit = int(core.get("inject_limit") or DEFAULT_INJECT_LIMIT)
    count = len(core.get("entries") or [])
    if count <= limit * CONSOLIDATE_FACTOR:
        return False
    try:
        last = int(core.get("last_consolidated_at", -1))
    except Exception:
        last = -1
    return count > last


def mark_consolidated(core: dict) -> dict:
    """Record the size a consolidation pass was attempted at. Called whether or not the
    pass changed anything — a no-op result is exactly the case being backed off."""
    core["last_consolidated_at"] = len(core.get("entries") or [])
    return core


def apply_operations(core: dict, operations, source_chat: dict = None) -> dict:
    """Apply extractor operations to ``core`` in place. Returns a summary
    {added, updated, merged, deleted} for the toast.

    Malformed operations are skipped rather than raising — this runs behind a
    best-effort LLM call and must never break a send. User-written and pinned entries
    are protected: a delete aimed at one is dropped, and a merge that would consume one
    leaves it alone and merges only the rest.
    """
    summary = {"added": 0, "updated": 0, "merged": 0, "deleted": 0}
    if not isinstance(operations, list):
        return summary
    entries = core.setdefault("entries", [])
    by_id = {e.get("id"): e for e in entries}

    def _protected(entry):
        return entry.get("origin") == "user" or entry.get("pinned")

    for op in operations:
        if not isinstance(op, dict):
            continue
        kind = (op.get("op") or "").strip().lower()
        text = (op.get("text") or "").strip()

        if kind == "add":
            if not text:
                continue
            entries.append(new_entry(text, op.get("category"), op.get("importance"),
                                     origin="ai", source_chat=source_chat))
            by_id[entries[-1]["id"]] = entries[-1]
            summary["added"] += 1

        elif kind == "update":
            target = by_id.get(op.get("id") or "")
            if not target or not text:
                continue
            target["text"] = text
            if op.get("category"):
                target["category"] = _clean_category(op.get("category"))
            if op.get("importance") is not None:
                target["importance"] = _clamp_importance(op.get("importance"), target.get("importance", 5))
            target["updated"] = _now()
            summary["updated"] += 1

        elif kind == "merge":
            # A repeated id must collapse to one target. Otherwise ids ["a","a"] gives
            # targets [A, A], and the removal loop below deletes the very entry the
            # merged text was just written into — losing the memory and reporting it
            # as merged.
            targets, seen = [], set()
            for i in (op.get("ids") or []):
                t = by_id.get(i)
                # Never absorb a protected entry into a merged one — leave it standing.
                if t is None or _protected(t) or id(t) in seen:
                    continue
                seen.add(id(t))
                targets.append(t)
            if len(targets) < 2 or not text:
                continue
            keeper = targets[0]
            keeper["text"] = text
            keeper["category"] = _clean_category(op.get("category") or keeper.get("category"))
            # The merged entry inherits the strongest importance of what it replaces.
            # ``.get(k, default)`` would hand back an explicit null instead of the
            # default, so test for it — the update branch above does the same.
            merged_importance = max(int(t.get("importance") or 5) for t in targets)
            if op.get("importance") is not None:
                merged_importance = op.get("importance")
            keeper["importance"] = _clamp_importance(
                merged_importance, max(int(t.get("importance") or 5) for t in targets))
            keeper["updated"] = _now()
            for dead in targets[1:]:
                entries.remove(dead)
                by_id.pop(dead.get("id"), None)
            summary["merged"] += 1

        elif kind == "delete":
            target = by_id.get(op.get("id") or "")
            if not target or _protected(target):
                continue
            entries.remove(target)
            by_id.pop(target.get("id"), None)
            summary["deleted"] += 1

    if any(summary.values()):
        touch(core)
    return summary


def summary_text(summary: dict) -> str:
    """'2 added, 1 refined' — the toast body. '' when nothing changed."""
    bits = []
    for key, label in (("added", "added"), ("updated", "refined"),
                       ("merged", "merged"), ("deleted", "removed")):
        if summary.get(key):
            bits.append(f"{summary[key]} {label}")
    return ", ".join(bits)
