#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

GUI-free prompt-assembly logic. These functions were extracted from the original
wxPython frame (``_start_assistant_generation``, ``_build_library_context``,
``_build_batch_messages``, ``_inject_research``, ``_execute_tool``) and rewritten
to take plain settings dicts (a chat dict) + the libraries list instead of reading
UI widgets. Behaviour matches the original exactly.
"""

import json
import re
import uuid
from datetime import datetime
from xml.sax.saxutils import escape as _xml_escape, quoteattr as _xml_attr

from . import core
from . import memory as mem_mod
from .core import MIN_CRAWLED_PAGES, WEB_SEARCH_TOOL, web_search

# Default Multi-Pass evaluation prompt. [input prompt] and [Response] are replaced
# (case-insensitively) with the original prompt and the previous pass's answer.
DEFAULT_EVAL_PROMPT = (
    "This is what the user wanted: [input prompt]\n\n"
    "Make sure this response meets all the requirements they asked for. Here is the "
    "response to evaluate:\n[Response]\n\n"
    "Reply with the fixed response so it meets the requirements the user wanted. Don't "
    "add any pre or post comments — only reply with the updated version."
)

_TAG_INPUT = re.compile(r"\[input prompt\]", re.IGNORECASE)
_TAG_RESPONSE = re.compile(r"\[response\]", re.IGNORECASE)


# --------------------------- Chat dict factory ---------------------------

def create_chat_dict(title="New Chat", server_url=None, model=None, pre_prompt="",
                     num_ctx=None, private=False, default_num_ctx=4096, group_id=None):
    """Create a new chat dict with every per-chat setting the app tracks."""
    now = datetime.utcnow().isoformat()
    return {
        "id": uuid.uuid4().hex[:12],
        "group_id": group_id,      # sidebar tab; None/"default" = the built-in "My Chats" tab
        "title": title,
        "server_url": server_url or core.DEFAULT_LOCAL_URL,
        "model": model or "",
        "system_prompt": "",       # sent as a `system` message when system_on
        "system_on": False,
        "pre_prompt": pre_prompt,  # prepended to the last user turn when pre_on
        "pre_on": bool((pre_prompt or "").strip()),
        "num_ctx": num_ctx or default_num_ctx,
        "messages": [],
        "created": now,
        "updated": now,
        "private": private,
        "isolated": False,
        "reasoning": False,        # legacy; kept for back-compat (no longer the display control)
        "hide_thinking": False,    # when True, reasoning is still generated but not shown/saved
        "web_search": False,
        "crawl_pages": MIN_CRAWLED_PAGES,
        "library_ids": [],
        "library_strict": False,
        "pre_as_system": True,     # legacy; superseded by system_on/pre_on (kept for back-compat)
        "multi_pass": False,       # iterative self-refinement
        "passes": 2,               # refinement rounds after the first answer
        "eval_prompt": DEFAULT_EVAL_PROMPT,
        "pass_use_system": True,   # include system prompt/library context in refinement passes
        "rag_enabled": False,      # retrieve top-k relevant chunks instead of dumping full context
        "rag_auto": False,         # auto-enable RAG when the message exceeds rag_threshold words
        "rag_threshold": 400,      # word count (question + staged data) that triggers auto-RAG
        "rag_scope": "attachments",  # what RAG searches besides libraries — see RAG_SCOPES
        "memory_enabled": False,   # inject a user memory core, and grow it from this chat
        "memory_core_id": "",      # which core (per-chat, see app/memory.py)
        "memory_turns_since": 0,   # assistant turns since the last extraction pass
        "persona_on": False,       # answer through a persona's pipeline (see app/persona.py)
        "persona_id": "",          # which persona
        "persona_variant": "",     # selected speaking variant ("" = the persona's default)
        # Pinned composer sources: material that belongs to this conversation rather
        # than a reusable library. [{id, type, label, content, source}] where type
        # matches the library vocabulary (write|file|url|youtube|search).
        "attachments": [],
    }


# --------------------------- Library context ---------------------------

def _item_label(it: dict) -> str:
    """Human label for a library item (label → filename → type default)."""
    return (it.get("label") or it.get("filename")
            or ("Write-in" if it.get("type") == "write" else "File"))


def _item_attrs(it: dict) -> str:
    """Serialize a library item's identity as XML attributes (no content)."""
    parts = [f"id={_xml_attr(it.get('id') or '')}",
             f"label={_xml_attr(_item_label(it))}",
             f"type={_xml_attr(it.get('type') or 'file')}"]
    if it.get("filename"):
        parts.append(f"file={_xml_attr(it['filename'])}")
    return " ".join(parts)


def _selected_libs(chat: dict, libraries: list) -> list:
    ids = (chat or {}).get("library_ids") or []
    if not ids:
        return []
    return [l for l in (libraries or []) if l.get("id") in ids]


def _library_preamble(chat: dict, kind: str) -> str:
    """Instruction line prefixed to the library block. ``kind`` is 'full' (content
    included) or 'index' (manifest + separate excerpts)."""
    strict = bool((chat or {}).get("library_strict", False))
    if kind == "index":
        base = ("The following <library> block is an index of the reference materials "
                "available to you. Each <item> lists its id, label and filename but not "
                "its content; the most relevant passages are provided separately below as "
                "<excerpt> elements, correlated to these items by the item id.")
    else:
        base = ("The following <library> block contains the reference materials available "
                "to you, one <item> per document.")
    if strict:
        return (base + " You must answer using ONLY these reference materials. If the answer "
                "is not contained in them, say you don't have that information. Do not use "
                "outside knowledge.")
    return base + " Use them when relevant, alongside your own knowledge."


def build_library_context(chat: dict, libraries: list) -> str:
    """Build the full-content reference-materials system message as per-item XML
    (``<library name=…><item id label file>…content…</item></library>``). Used on the
    RAG-unavailable fallback path — its per-item tags double as the index there.
    Returns '' when nothing is selected/available.
    """
    libs = _selected_libs(chat, libraries)
    if not libs:
        return ""
    parts = []
    for lib in libs:
        item_els = []
        for it in lib.get("items", []):
            content = (it.get("content") or "").strip()
            if not content:
                continue
            item_els.append(f"  <item {_item_attrs(it)}>\n{_xml_escape(content)}\n  </item>")
        if item_els:
            parts.append(f"<library name={_xml_attr(lib.get('name', 'Untitled'))}>\n"
                         + "\n".join(item_els) + "\n</library>")
    materials = "\n\n".join(parts).strip()
    if not materials:
        return ""
    return _library_preamble(chat, "full") + "\n\n" + materials


def build_library_manifest(chat: dict, libraries: list) -> str:
    """Build the lightweight XML *index* of the selected libraries — one
    self-closing ``<item id label file/>`` per item, with **no content**. Sent
    alongside the RAG excerpts so the model knows what exists and can correlate an
    excerpt back to its source item by id. Returns '' when nothing is selected.
    """
    libs = _selected_libs(chat, libraries)
    if not libs:
        return ""
    parts = []
    for lib in libs:
        item_els = []
        for it in lib.get("items", []):
            if not (it.get("content") or "").strip():
                continue  # mirror the content builder: skip empty items
            item_els.append(f"  <item {_item_attrs(it)}/>")
        if item_els:
            parts.append(f"<library name={_xml_attr(lib.get('name', 'Untitled'))}>\n"
                         + "\n".join(item_els) + "\n</library>")
    manifest = "\n\n".join(parts).strip()
    if not manifest:
        return ""
    return _library_preamble(chat, "index") + "\n\n" + manifest


def _item_label_map(libraries: list) -> dict:
    """Map stable item id → human label, for annotating retrieved excerpts."""
    m = {}
    for lib in libraries or []:
        for it in lib.get("items", []):
            if it.get("id"):
                m[it["id"]] = _item_label(it)
    return m


# --------------------------- Interactive chat messages ---------------------------

def _clean_msg(m):
    """Strip a stored message down to what the model should see (role + content,
    plus any attached images). Drops app-only fields like ``reasoning`` so saved
    chain-of-thought is never fed back into the model on later turns.

    ``images`` survives as a *sibling* of ``content`` holding image records; it is
    swapped for inline bytes by ``images.hydrate_messages`` at the last moment and
    reshaped into each provider's wire format by the adapters. ``content`` therefore
    stays a plain string all the way through, which the pre-prompt fold below and
    the context tracker both depend on."""
    out = {"role": m.get("role"), "content": m.get("content", "")}
    imgs = m.get("images")
    if imgs and out["role"] in ("user", "assistant"):
        out["images"] = list(imgs)
    return out


def _resolve_prompts(chat: dict):
    """Return ``(system_text, pre_text)`` for a chat, honoring the independent
    ``system_on`` / ``pre_on`` toggles. Falls back to the legacy single
    ``pre_prompt`` + ``pre_as_system`` model for any chat not yet migrated."""
    chat = chat or {}
    if "system_on" in chat or "pre_on" in chat:
        sys_p = (chat.get("system_prompt") or "").strip() if chat.get("system_on") else ""
        pre = (chat.get("pre_prompt") or "").strip() if chat.get("pre_on") else ""
        return sys_p, pre
    pre_raw = (chat.get("pre_prompt") or "").strip()
    if pre_raw and bool(chat.get("pre_as_system", True)):
        return pre_raw, ""
    return "", pre_raw


def build_messages(chat: dict, libraries: list, skip_library_dump: bool = False,
                   history_window: int = None) -> list:
    """Assemble the message list for an interactive generation, mirroring
    ``_start_assistant_generation``: pre-prompt (system or folded into the last
    user turn), library context, isolation, and truncation to the last user
    message so regenerations are fresh answers to the same prompt.

    When ``skip_library_dump`` is True the full library content is omitted and only
    the lightweight XML *manifest* is attached — RAG owns the reference material and
    injects the relevant chunks as excerpts instead.

    ``history_window`` caps how many trailing non-intermediate messages are sent
    verbatim; ``None`` sends the whole thread. Thread-scope RAG sets it, because
    retrieving excerpts from a conversation that is also being sent in full spends the
    context window twice on the same words. The recent tail stays intact — recency and
    conversational flow are exactly what retrieval is bad at — and older turns arrive as
    excerpts only when they're relevant.
    """
    # Intermediate Multi-Pass answers are display-only records — only each turn's
    # final refined answer should feed future context.
    chat_messages = [m for m in chat.get("messages", []) if not m.get("intermediate")]
    sys_p, pre = _resolve_prompts(chat)

    full_messages = []
    if sys_p:
        full_messages.append({"role": "system", "content": sys_p})

    lib_ctx = (build_library_manifest(chat, libraries) if skip_library_dump
               else build_library_context(chat, libraries))
    if lib_ctx:
        full_messages.append({"role": "system", "content": lib_ctx})

    # Find last user and take only up to it (for fresh generation).
    last_user_idx = -1
    for i, msg in enumerate(chat_messages):
        if msg.get("role") == "user":
            last_user_idx = i

    isolated = bool(chat.get("isolated", False))
    if isolated and last_user_idx >= 0:
        full_messages.append(_clean_msg(chat_messages[last_user_idx]))
    elif last_user_idx >= 0:
        history = chat_messages[:last_user_idx + 1]
        # Isolation already sends one turn, so the window only applies here. Always keep
        # at least the final user turn — windowing it away would leave nothing to answer.
        if history_window is not None:
            history = history[-max(1, int(history_window)):]
        full_messages.extend([_clean_msg(m) for m in history])
    else:
        full_messages.extend([_clean_msg(m) for m in chat_messages])

    # The pre-prompt is folded into the last user turn (the system prompt, if any, was
    # already emitted as a leading system message above).
    if pre:
        for m in reversed(full_messages):
            if m.get("role") == "user":
                m["content"] = f"{pre}\n\n{m['content']}"
                break

    return full_messages


def resolve_web_search(chat: dict, config: dict, search_query: str, tools_supported):
    """Decide how web search should behave for this generation (mirrors the logic
    in ``_start_assistant_generation``). Returns a dict with:
        worker_query   -> str|None  (app-side crawl+inject, works with any model)
        tools          -> list|None (model-driven autonomous search; needs tools)
        min_pages      -> int
        allowed_domains-> list|None
    """
    web_on = bool(chat.get("web_search", False))
    min_pages = int(chat.get("crawl_pages") or MIN_CRAWLED_PAGES)

    allowed_domains = None
    if config.get("restrict_to_approved") and config.get("approved_domains"):
        allowed_domains = list(config["approved_domains"])

    query = (search_query or "").strip()
    worker_query = None
    tools = None
    if web_on and query:
        worker_query = query
    elif web_on and tools_supported is not False:
        tools = [WEB_SEARCH_TOOL]

    return {
        "worker_query": worker_query,
        "tools": tools,
        "min_pages": min_pages,
        "allowed_domains": allowed_domains,
    }


def inject_research(messages: list, query: str, research: str) -> list:
    """Insert crawled web research as a system message just before the final user
    turn so the model answers using it (mirrors ``_inject_research``).
    """
    msgs = list(messages)
    ctx = {
        "role": "system",
        "content": (
            f"The user asked to search the web for: \"{query}\".\n"
            "Below is the full text of web pages that were crawled for this query. "
            "Use this information to answer the user's message, and cite sources by "
            "their URL when you rely on them.\n\n" + research
        ),
    }
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    if last_user >= 0:
        msgs.insert(last_user, ctx)
    else:
        msgs.append(ctx)
    return msgs


ATTACHMENT_PREAMBLE = (
    "The following <attached> block holds material the user pinned to this "
    "conversation — documents, pages, videos or searches they added alongside their "
    "messages. Treat it as reference the user expects you to have read, and use it "
    "when relevant.\n\n"
)


def build_attachment_block(chat: dict) -> str:
    """Render a chat's pinned attachments as an XML block, or '' when there are none.

    Pinned attachments are re-sent on every turn, so they're emitted as a system
    message rather than folded into the user's text — that way they don't accumulate
    a fresh copy in the history with each message the way an inline <Data> block does.
    """
    items = (chat or {}).get("attachments") or []
    els = []
    for it in items:
        content = (it.get("content") or "").strip()
        if not content:
            continue
        attrs = [f"label={_xml_attr(it.get('label') or 'Attachment')}",
                 f"type={_xml_attr(it.get('type') or 'write')}"]
        if it.get("source"):
            attrs.append(f"source={_xml_attr(it['source'])}")
        els.append(f"  <item {' '.join(attrs)}>\n{_xml_escape(content)}\n  </item>")
    if not els:
        return ""
    return ATTACHMENT_PREAMBLE + "<attached>\n" + "\n".join(els) + "\n</attached>"


def attachment_text(chat: dict) -> str:
    """Plain concatenation of the pinned attachments, for the RAG corpus."""
    parts = []
    for it in (chat or {}).get("attachments") or []:
        content = (it.get("content") or "").strip()
        if content:
            label = it.get("label") or "Attachment"
            parts.append(f"{label}\n{content}")
    return "\n\n".join(parts)


def resolve_attachments(chat: dict, rag_active: bool = False, scope: str = None) -> str:
    """The pinned-attachment block to inject for this generation, or ''.

    Mirrors ``skip_library_dump``: when RAG is running, retrieval owns the pinned
    material outright (it is in the corpus) and the full block is dropped. There used to
    be a size threshold below which the block was sent *as well as* being retrieved,
    which spent the context twice on the same text for no benefit.

    ``scope`` narrows that: under ``"thread"`` the attachments are NOT in the corpus, so
    the block has to come back or the pinned material would vanish from the turn
    entirely — retrieved by nothing and sent by nobody. ``scope=None`` keeps the
    pre-scope behaviour for callers that don't know about it.
    """
    if rag_active and (scope or "attachments") != "thread":
        return ""
    return build_attachment_block(chat)


def attachment_images(chat: dict) -> list:
    """Image refs (``[{"id": ...}]``) among a chat's pinned attachments.

    Pinned images can't ride the ``<attached>`` XML block the way documents do —
    ``build_attachment_block`` renders text, and an image has none. They travel as
    real image parts on the user turn instead, which is why they need their own
    collector and injector."""
    out = []
    for it in (chat or {}).get("attachments") or []:
        if (it.get("type") == "image" or it.get("kind") == "image") and it.get("id"):
            out.append({"id": it["id"]})
    return out


def inject_images(messages: list, refs: list) -> list:
    """Append image refs to the last user turn (mirrors the other injectors' seam).

    Appended rather than prepended so the images the user attached to *this* message
    stay first — pinned material is background, the new attachment is the question.
    An empty list is a no-op."""
    if not refs:
        return messages
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            m = dict(msgs[i])
            m["images"] = list(m.get("images") or []) + list(refs)
            msgs[i] = m
            return msgs
    return msgs


def inject_attachments(messages: list, block: str) -> list:
    """Insert the pinned-attachment block as a system message just before the final
    user turn (mirrors ``inject_research``). An empty block is a no-op."""
    if not block:
        return messages
    msgs = list(messages)
    ctx = {"role": "system", "content": block}
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    if last_user >= 0:
        msgs.insert(last_user, ctx)
    else:
        msgs.append(ctx)
    return msgs


def inject_memory(messages: list, block: str) -> list:
    """Insert the user's memory core as a system message just before the final user
    turn (mirrors ``inject_research``). ``block`` is built by ``memory.render_core``;
    an empty block is a no-op so the caller doesn't have to check twice."""
    if not block:
        return messages
    msgs = list(messages)
    ctx = {"role": "system", "content": mem_mod.MEMORY_PREAMBLE + block}
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    if last_user >= 0:
        msgs.insert(last_user, ctx)
    else:
        msgs.append(ctx)
    return msgs


# --------------------------- RAG (Retrieval Augmented Generation) ---------------------------

# The inline data wrapper emitted by the frontend (``buildDataXml`` in app.js):
# staged data blocks are wrapped in <Data>…</Data> and prepended to the user's text.
_DATA_BLOCK = re.compile(r"<Data>(.*?)</Data>", re.DOTALL | re.IGNORECASE)


def split_inline_data(user_content: str):
    """Split a user message into (data_text, question_text) around the
    <Data>…</Data> block. Returns ('', content) when no data block is present."""
    content = user_content or ""
    blocks = _DATA_BLOCK.findall(content)
    data_text = "\n\n".join(b.strip() for b in blocks).strip()
    question = _DATA_BLOCK.sub("", content).strip()
    return data_text, question


# What RAG searches BESIDES the selected libraries. A library is always retrieved when
# selected; this chooses the chat's own corpus:
#   attachments -> pinned attachments + the staged <Data> blocks (the original behaviour)
#   thread      -> the conversation itself, so a long chat can outlive its context window
#   both        -> the union
RAG_SCOPES = ("attachments", "thread", "both")

# Short turns ("ok", "thanks", "do that") carry no retrievable content but score well on
# keyword search, where they'd displace real excerpts. Below this they aren't indexed.
_MIN_THREAD_CHARS = 40


def rag_scope(chat: dict) -> str:
    """The chat's RAG corpus scope, normalised. Chats written before the control existed
    (and anything unrecognised) read as 'attachments' — the pre-scope behaviour."""
    scope = ((chat or {}).get("rag_scope") or "").strip().lower()
    return scope if scope in RAG_SCOPES else "attachments"


def rag_is_transient(chat: dict) -> bool:
    """True when this chat's own corpus must NOT be persisted to the vector store.

    Private chats: the default LanceDB backend keeps chunk text and vectors in PLAINTEXT
    on disk (see app/vectorstore/__init__.py), so indexing a private chat would write out
    exactly what the private flag exists to keep off the disk. There is no reliable
    "chat closed" hook to clean up after, so it is never written in the first place —
    those chats fall back to the in-memory retrieval path instead.

    Synthetic chats (``rag_ephemeral``): Batch and the queue build per-item chats that
    either carry the REAL chat's id — indexing them would overwrite and then prune the
    real chat's index once per file — or a throwaway uid that nothing will ever delete.
    """
    return bool((chat or {}).get("private")
                or (chat or {}).get("rag_ephemeral")
                or not (chat or {}).get("id"))


def attachment_items(chat: dict) -> list:
    """The pinned attachments + staged ``<Data>`` blocks as indexable items.

    Returns ``rag.upsert_items``-shaped ``[(item_id, content, meta)]``. Attachments key
    on their own stable id; a ``<Data>`` block keys on the turn that carries it. Honours
    ``isolated`` exactly as ``collect_rag_inputs`` does — an isolated chat contributes
    only the current turn's data, while pinned attachments belong to the chat rather
    than to a turn and so are never hidden by isolation.
    """
    items = []
    for it in (chat or {}).get("attachments") or []:
        content = (it.get("content") or "").strip()
        if not content or not it.get("id"):
            continue
        label = it.get("label") or "Attachment"
        items.append((str(it["id"]), content,
                      {"label": label, "type": it.get("type") or "write",
                       "kind": "attachment"}))

    msgs = [m for m in (chat or {}).get("messages", []) if not m.get("intermediate")]
    isolated = bool((chat or {}).get("isolated", False))
    user_idx = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    wanted = user_idx[-1:] if isolated else user_idx
    for i in wanted:
        data, _q = split_inline_data(msgs[i].get("content", ""))
        if data:
            items.append((f"data:{i}", data,
                          {"label": f"Attached data (turn {i + 1})", "type": "data",
                           "kind": "data"}))
    return items


def thread_items(chat: dict, exclude_last_user: bool = True) -> list:
    """The conversation itself as indexable items — ``[(item_id, content, meta)]``.

    ``item_id`` is ``f"m{index}"`` over the RAW ``chat['messages']`` list. That index is
    a stable key because messages are only ever appended, popped from the tail
    (regenerate), edited in place, or cleared wholesale — nothing splices or inserts. So
    an edit keeps its id and changes only its fingerprint, and a regenerate drops ids off
    the end where ``prune_items`` reaps them.

    ``exclude_last_user`` drops the just-posted user turn: it IS the retrieval query, so
    indexing it makes the search return the question as its own best-matching excerpt.
    ``<Data>`` blocks are stripped — those belong to the attachment corpus, and leaving
    them here would index a 200k-character transcript twice under two scopes.
    """
    messages = (chat or {}).get("messages") or []
    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user" and not m.get("intermediate"):
            last_user = i

    items = []
    for i, m in enumerate(messages):
        if m.get("intermediate"):
            continue                      # display-only Multi-Pass drafts
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        if exclude_last_user and i == last_user:
            continue
        content = m.get("content", "")
        if role == "user":
            _data, content = split_inline_data(content)
        content = (content or "").strip()
        if len(content) < _MIN_THREAD_CHARS:
            continue
        items.append((f"m{i}", content,
                      {"label": f"Turn {i + 1} ({role})", "role": role, "index": i,
                       "kind": "thread"}))
    return items


def thread_text(chat: dict, exclude_last_user: bool = True) -> str:
    """``thread_items`` flattened to one string, for the transient (private-chat)
    retrieval path, which chunks text rather than indexing items."""
    return "\n\n".join(f"{meta['label']}\n{content}"
                       for _id, content, meta in thread_items(chat, exclude_last_user))


def collect_rag_inputs(chat: dict):
    """Isolation-aware gathering of the RAG query and inline-data corpus.

    Returns (question, data_text). Isolated chats scope the data to the current
    turn only; non-isolated chats concatenate the data blocks of every user turn
    in the thread (matching how ``build_messages`` scopes history). Pinned
    attachments join the corpus in both cases — they belong to the chat, not to a
    turn, so isolation doesn't hide them.
    """
    msgs = [m for m in (chat or {}).get("messages", []) if not m.get("intermediate")]
    isolated = bool((chat or {}).get("isolated", False))
    pinned = attachment_text(chat)
    last_user = None
    for m in msgs:
        if m.get("role") == "user":
            last_user = m
    if last_user is None:
        return "", pinned
    last_data, question = split_inline_data(last_user.get("content", ""))
    if isolated:
        data_text = last_data
    else:
        parts = []
        for m in msgs:
            if m.get("role") == "user":
                d, _q = split_inline_data(m.get("content", ""))
                if d:
                    parts.append(d)
        data_text = "\n\n".join(parts).strip()
    if pinned:
        data_text = (data_text + "\n\n" + pinned).strip() if data_text else pinned
    return question, data_text


def resolve_rag(chat: dict, config: dict):
    """Decide whether RAG is active for this generation and what it retrieves over.

    Returns None when RAG is off, else a dict:
        {active, query, data_text, use_libraries, top_k}
    Active when a library is selected (a selected library is ALWAYS served via RAG),
    or rag_enabled is set, or rag_auto is set and the message (question + staged data)
    meets the per-chat word-count threshold.
    """
    enabled = bool((chat or {}).get("rag_enabled", False))
    auto = bool((chat or {}).get("rag_auto", False))
    has_library = bool((chat or {}).get("library_ids"))
    if not enabled and not auto and not has_library:
        return None

    question, data_text = collect_rag_inputs(chat)
    if enabled or has_library:
        active = True
    else:
        threshold = int((chat or {}).get("rag_threshold") or 400)
        word_count = len((question + " " + data_text).split())
        active = word_count >= threshold
    if not active:
        return None

    query = question.strip()
    if not query and data_text:
        query = " ".join(data_text.split()[:60])

    # Retrieval mode: per-chat override wins, else the global setting, else hybrid.
    mode = ((chat or {}).get("rag_retrieval_mode")
            or (config or {}).get("rag_retrieval_mode") or "hybrid").lower()
    if mode not in ("vector", "keyword", "hybrid"):
        mode = "hybrid"

    # Prompt Reword (retrieval query rewrite): per-chat override, else config (default on).
    if "rag_query_rewrite" in (chat or {}):
        query_rewrite = bool(chat.get("rag_query_rewrite"))
    else:
        query_rewrite = bool((config or {}).get("rag_query_rewrite", True))

    scope = rag_scope(chat)
    use_thread = scope in ("thread", "both")

    return {
        "active": True,
        "query": query,
        # Base query variants. When query_rewrite is on, server._rag_retrieve expands
        # this to 2-3 reworded variants (+ keywords) via rewrite.rewrite_queries before
        # retrieval, and the hybrid layer RRF-merges them.
        "queries": [query] if query else [],
        "mode": mode,
        "query_rewrite": query_rewrite,
        # Kept alongside the scope flags: the transient path (private chats) chunks this
        # text directly rather than retrieving indexed items.
        "data_text": data_text,
        "use_libraries": bool((chat or {}).get("library_ids")),
        "scope": scope,
        "use_attachments": scope in ("attachments", "both"),
        "use_thread": use_thread,
        # Retrieving over the thread is pointless while the thread is also sent in full —
        # the excerpts would duplicate text already in the prompt. Under thread scope the
        # verbatim history is capped and retrieval supplies what falls outside.
        "history_window": (int((config or {}).get("rag_thread_window") or 8)
                           if use_thread else None),
        "top_k": int((config or {}).get("rag_top_k") or 6),
    }


def _chunk_meta(r: dict) -> dict:
    """A retrieved chunk's ``meta`` as a dict. Both vector-store backends hand it back as
    a JSON string, and a row written before ``meta`` existed has none at all."""
    meta = r.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = None
    return meta if isinstance(meta, dict) else {}


def _excerpt_label(r: dict, label_map: dict) -> str:
    """Human label for a retrieved chunk.

    The library map wins where it applies — a library item's label can be edited after
    it was compiled, so the live name beats the one frozen into the chunk. Otherwise the
    chunk's own persisted ``meta`` carries it, which is how a chat excerpt reads as
    "Turn 4 (assistant)" rather than the raw ``m3``."""
    item_id = r.get("item_id") or r.get("source_id") or "excerpt"
    if label_map.get(item_id):
        return label_map[item_id]
    label = _chunk_meta(r).get("label")
    if label:
        return str(label)
    return str(item_id)


def inject_rag(messages: list, retrieved: list, libraries: list = None,
               strip_data: bool = True) -> list:
    """Insert retrieved chunks as a system message before the last user turn, and
    strip the raw <Data> block from every user turn so the full payload isn't sent
    alongside the retrieved excerpts (mirrors ``inject_research``).

    Excerpts are emitted as XML ``<excerpt item=… label=… score=…>`` elements whose
    ``item`` id ties each back to an ``<item>`` in the library manifest where one
    exists. ``libraries`` is used to resolve friendly labels for library-sourced
    excerpts.

    ``strip_data=False`` leaves the ``<Data>`` blocks in place. Thread scope needs that:
    the staged data is deliberately not in the corpus there, so stripping it would send
    it neither retrieved nor verbatim — the user's attachment would simply disappear."""
    msgs = []
    for m in messages:
        mm = dict(m)
        if strip_data and m.get("role") == "user":
            d, q = split_inline_data(m.get("content", ""))
            if d:
                mm["content"] = q if q else "(reference data provided separately)"
        msgs.append(mm)

    if not retrieved:
        return msgs

    label_map = _item_label_map(libraries)
    blocks = []
    for r in retrieved:
        item_id = r.get("item_id") or r.get("source_id") or "excerpt"
        label = _excerpt_label(r, label_map)
        score = r.get("score")
        score_attr = (f" score=\"{score:.3f}\""
                      if isinstance(score, (int, float)) and score >= 0 else "")
        blocks.append(
            f"  <excerpt item={_xml_attr(str(item_id))} label={_xml_attr(str(label))}"
            f"{score_attr}>\n{_xml_escape(r.get('content', ''))}\n  </excerpt>")
    ctx = {
        "role": "system",
        "content": (
            "The following excerpts were retrieved as the most relevant to the user's "
            "question via semantic search over their reference materials, attached data "
            "and earlier turns of this conversation. Where an excerpt's item id matches "
            "an <item> in the library index above, it comes from that item. "
            "Use them to answer the user's message; if the answer is not contained in "
            "them, say you don't have that information.\n\n"
            "<retrieved>\n" + "\n".join(blocks) + "\n</retrieved>"
        ),
    }
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    if last_user >= 0:
        msgs.insert(last_user, ctx)
    else:
        msgs.append(ctx)
    return msgs


# --------------------------- Retrieved-source citations ---------------------------
#
# ``inject_rag`` renders the retrieved chunks for the MODEL. ``describe_sources`` renders
# the same chunks for the USER: the Sources panel under an answer, where each excerpt
# links back to the document it was taken from.

# Display kind per stored ``source_type``. Mirrors the constants in compile.py, which
# can't be imported here — compile imports logic, not the other way round.
_SOURCE_KINDS = {
    "library": "library",
    "chat_attach": "attachment",
    "chat_thread": "thread",
    "persona_knowledge": "persona",
    "persona_memory": "persona",
}


def _split_chunk_id(chunk_id: str):
    """(source_type, chunk_index) read off a stored chunk id.

    ``rag.upsert_items`` builds ids as ``f"{source_type}:{source_id}:{item_id}:{ci}"``.
    Only the two ends are wanted, so this reads from the edges and leaves whatever an id
    containing colons puts in the middle alone. Returns ("", None) for anything that
    isn't one — inline retrieval mints no id."""
    parts = str(chunk_id or "").split(":")
    if len(parts) < 4:
        return "", None
    try:
        index = int(parts[-1])
    except (TypeError, ValueError):
        index = None
    return parts[0], index


def describe_sources(retrieved: list, libraries: list = None) -> list:
    """Describe the chunks injected this turn so the UI can show and link them.

    One entry per excerpt, labelled exactly as ``inject_rag`` labels it, plus enough
    identity for the client to navigate back: the display ``kind``, the owning library,
    and the item within it. The chunk text rides along verbatim because that is what the
    client searches for in the source document — nothing in the store records a chunk's
    position (``rag.chunk_text_semantic`` discards the chunker's offsets), so the passage
    is located at click time rather than looked up.
    """
    label_map = _item_label_map(libraries)
    libs_by_id = {lib.get("id"): lib for lib in (libraries or []) if lib.get("id")}
    out = []
    for r in retrieved or []:
        id_type, id_index = _split_chunk_id(r.get("id"))
        source_type = r.get("source_type") or id_type
        source_id = r.get("source_id") or ""
        item_id = r.get("item_id") or ""
        chunk_index = r.get("chunk_index")
        if chunk_index is None:
            chunk_index = id_index
        score = r.get("score")
        src = {
            "id": r.get("id") or "",
            "kind": _SOURCE_KINDS.get(source_type, "inline"),
            "source_type": source_type,
            "source_id": source_id,
            "item_id": item_id,
            "chunk_index": chunk_index,
            "label": _excerpt_label(r, label_map),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "content": r.get("content") or "",
        }
        if src["kind"] == "library":
            # source_id IS the library id; the lookup only supplies the live name, and
            # its absence means the library was deleted since compiling. The row stays
            # visible either way — the client decides whether it can be linked.
            src["library_id"] = source_id
            src["library_name"] = (libs_by_id.get(source_id) or {}).get("name") or ""
        elif src["kind"] == "thread":
            index = _chunk_meta(r).get("index")
            if isinstance(index, int):
                src["message_index"] = index
        out.append(src)
    return out


# --------------------------- Multi-Pass evaluation ---------------------------

def fill_eval_prompt(template: str, input_prompt: str, last_response: str) -> str:
    """Substitute the [input prompt] and [Response] tags (case-insensitive)."""
    text = template or DEFAULT_EVAL_PROMPT
    text = _TAG_INPUT.sub(lambda _m: input_prompt or "", text)
    text = _TAG_RESPONSE.sub(lambda _m: last_response or "", text)
    return text


def build_eval_messages(chat: dict, input_prompt: str, last_response: str,
                        libraries: list, use_system: bool,
                        skip_library_dump: bool = False) -> list:
    """Assemble one refinement pass: (optional system prompt + library context) +
    the filled evaluation prompt as a single isolated user turn (no prior history).

    When ``skip_library_dump`` is True the full library content is replaced by the
    lightweight XML manifest — the caller injects RAG excerpts separately."""
    filled = fill_eval_prompt(chat.get("eval_prompt"), input_prompt, last_response)
    sys_p, pre = _resolve_prompts(chat)
    messages = []
    if use_system:
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        lib_ctx = (build_library_manifest(chat, libraries) if skip_library_dump
                   else build_library_context(chat, libraries))
        if lib_ctx:
            messages.append({"role": "system", "content": lib_ctx})
        # Fold the pre-prompt into the eval turn.
        if pre:
            filled = f"{pre}\n\n{filled}"
    messages.append({"role": "user", "content": filled})
    return messages


# --------------------------- Batch messages ---------------------------

def build_batch_messages(prompt: str, history: list, snap: dict, lib_block: str = None) -> list:
    """Assemble the message list for one batch prompt, mirroring the chat's
    pre-prompt + library + isolation logic (see ``_build_batch_messages``).

    ``lib_block`` selects the library system message: pass the XML manifest when RAG
    excerpts are injected separately, or leave None to use the full-content dump in
    ``snap['lib_ctx']`` (the fallback path)."""
    sys_p = snap.get("system_prompt") or ""
    pre = snap.get("pre") or ""
    messages = []
    if sys_p:
        messages.append({"role": "system", "content": sys_p})
    lib_ctx = lib_block if lib_block is not None else snap.get("lib_ctx")
    if lib_ctx:
        messages.append({"role": "system", "content": lib_ctx})
    # Fold the pre-prompt into the user turn (the system prompt is a system message).
    user_content = f"{pre}\n\n{prompt}" if pre else prompt
    if snap.get("isolated"):
        messages.append({"role": "user", "content": user_content})
    else:
        messages.extend(_clean_msg(m) for m in history)
        messages.append({"role": "user", "content": user_content})
    return messages


# --------------------------- Tool executor ---------------------------

def make_tool_executor(stop_event, min_pages=MIN_CRAWLED_PAGES, allowed_domains=None,
                       on_toast=None):
    """Return a ``tool_executor(name, arguments)`` callable for chat_stream that
    runs the web_search tool (mirrors ``_execute_tool``).
    """
    def tool_executor(name, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"query": arguments}
        arguments = arguments or {}

        if name == "web_search":
            query = arguments.get("query", "")
            if on_toast:
                on_toast(f"🌐 Crawling ≥{min_pages} pages: {query[:40]}")
            return web_search(query, min_pages=min_pages,
                              should_stop=lambda: stop_event.is_set(),
                              allowed_domains=allowed_domains)
        return f"Unknown tool: {name}"

    return tool_executor
