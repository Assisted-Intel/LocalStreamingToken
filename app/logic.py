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
    """Strip a stored message down to what the model should see (role + content).
    Drops app-only fields like ``reasoning`` so saved chain-of-thought is never
    fed back into the model on later turns."""
    return {"role": m.get("role"), "content": m.get("content", "")}


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


def build_messages(chat: dict, libraries: list, skip_library_dump: bool = False) -> list:
    """Assemble the message list for an interactive generation, mirroring
    ``_start_assistant_generation``: pre-prompt (system or folded into the last
    user turn), library context, isolation, and truncation to the last user
    message so regenerations are fresh answers to the same prompt.

    When ``skip_library_dump`` is True the full library content is omitted and only
    the lightweight XML *manifest* is attached — RAG owns the reference material and
    injects the relevant chunks as excerpts instead.
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
        full_messages.extend([_clean_msg(m) for m in chat_messages[:last_user_idx + 1]])
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


def collect_rag_inputs(chat: dict):
    """Isolation-aware gathering of the RAG query and inline-data corpus.

    Returns (question, data_text). Isolated chats scope the data to the current
    turn only; non-isolated chats concatenate the data blocks of every user turn
    in the thread (matching how ``build_messages`` scopes history).
    """
    msgs = [m for m in (chat or {}).get("messages", []) if not m.get("intermediate")]
    isolated = bool((chat or {}).get("isolated", False))
    last_user = None
    for m in msgs:
        if m.get("role") == "user":
            last_user = m
    if last_user is None:
        return "", ""
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

    return {
        "active": True,
        "query": query,
        # Base query variants. When query_rewrite is on, server._rag_retrieve expands
        # this to 2-3 reworded variants (+ keywords) via rewrite.rewrite_queries before
        # retrieval, and the hybrid layer RRF-merges them.
        "queries": [query] if query else [],
        "mode": mode,
        "query_rewrite": query_rewrite,
        "data_text": data_text,
        "use_libraries": bool((chat or {}).get("library_ids")),
        "top_k": int((config or {}).get("rag_top_k") or 6),
    }


def inject_rag(messages: list, retrieved: list, libraries: list = None) -> list:
    """Insert retrieved chunks as a system message before the last user turn, and
    strip the raw <Data> block from every user turn so the full payload isn't sent
    alongside the retrieved excerpts (mirrors ``inject_research``).

    Excerpts are emitted as XML ``<excerpt item=… label=… score=…>`` elements whose
    ``item`` id ties each back to an ``<item>`` in the library manifest. ``libraries``
    is used to resolve friendly labels for library-sourced excerpts."""
    msgs = []
    for m in messages:
        mm = dict(m)
        if m.get("role") == "user":
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
        label = label_map.get(item_id) or item_id
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
            "question via semantic search over their reference materials and attached "
            "data. Each excerpt's item id matches an <item> in the library index above. "
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
