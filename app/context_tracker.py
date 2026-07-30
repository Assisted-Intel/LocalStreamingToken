#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Context-usage monitoring service. GUI-free helpers that:
- estimate token counts with a fast, dependency-free heuristic (for the live,
  in-flight bar and the per-segment system/RAG/user split),
- resolve the context-window denominator per chat/server (Ollama num_ctx, or a
  configurable per-provider window for cloud backends),
- build and persist a per-chat ``ContextHistoryEntry`` after every completed call.

Exact provider counts (Ollama ``prompt_eval_count``/``eval_count``, OpenAI/Anthropic
``usage``) are the source of truth for the totals; the heuristic only supplies the
in-flight estimate and the system/RAG/user breakdown, which no provider reports.
"""

import math
import uuid
from datetime import datetime

# Cloud providers have no user-set context window like Ollama's num_ctx. These are
# sensible defaults used as the bar denominator when the user hasn't configured a
# per-provider window in settings (config["provider_context_windows"]).
DEFAULT_PROVIDER_WINDOWS = {
    "openai": 128000,
    "anthropic": 200000,
}
FALLBACK_WINDOW = 8192

# Keep only the most recent N calls per chat so context_history.json can't grow
# without bound over a long-lived chat.
MAX_HISTORY_PER_CHAT = 200


# --------------------------- token estimation ---------------------------

def estimate_tokens(text) -> int:
    """Fast heuristic token estimate (~4 chars/token). Zero dependencies; works for
    every provider/model. Used only for the in-flight bar and the segment split —
    exact provider counts override the total after each call."""
    if not text:
        return 0
    return max(1, math.ceil(len(str(text)) / 4))


def estimate_messages(messages) -> int:
    """Estimated prompt tokens for a fully-assembled messages list."""
    return sum(estimate_tokens(m.get("content", "")) for m in (messages or []))


def breakdown_from_messages(messages) -> dict:
    """Split an assembled messages list into {system, rag, user} estimated tokens.

    The RAG segment is the system message injected by ``logic.inject_rag`` (its
    content contains the ``<retrieved>`` marker). Web-research injected context is
    folded into ``system``. Because this reads the *already-assembled* list, an
    isolated call naturally reports no excluded history (it was never in the list)."""
    system = rag = user = 0
    for m in (messages or []):
        role = m.get("role")
        content = m.get("content", "")
        toks = estimate_tokens(content)
        if role == "user":
            user += toks
        elif role == "system":
            if "<retrieved>" in (content or ""):
                rag += toks
            else:
                system += toks
        # assistant history turns count toward neither system nor rag nor user here;
        # they're captured in the exact prompt total when present.
    return {"system": system, "rag": rag, "user": user}


# --------------------------- window / denominator ---------------------------

def resolve_window(chat, server, config) -> int:
    """Context-window denominator for the usage bar. Ollama uses the chat's num_ctx
    (falling back to the global default); cloud providers use the configured
    per-provider window, then a built-in default."""
    stype = (server or {}).get("type") or "ollama"
    if stype == "ollama":
        return int(chat.get("num_ctx") or config.get("default_num_ctx", 4096) or 4096)
    windows = config.get("provider_context_windows") or {}
    return int(windows.get(stype) or DEFAULT_PROVIDER_WINDOWS.get(stype) or FALLBACK_WINDOW)


# --------------------------- history entries ---------------------------

def make_entry(*, server_id, server_name, isolation, breakdown, prompt_tokens,
               completion_tokens, num_ctx_at_time, batch_item_label=None, notes=None):
    """Build a ContextHistoryEntry dict (outline §3.2). ``breakdown`` is the
    estimated {system, rag, user}; ``prompt_tokens``/``completion_tokens`` are the
    exact provider totals (fall back to the estimate when a provider omits them)."""
    b = breakdown or {}
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "call_id": uuid.uuid4().hex[:12],
        "server_id": server_id or "",
        "server_name": server_name or server_id or "",
        "isolation": bool(isolation),
        "system_tokens": int(b.get("system", 0)),
        "rag_tokens": int(b.get("rag", 0)),
        "user_tokens": int(b.get("user", 0)),
        "assistant_tokens": int(completion_tokens or 0),
        "total_prompt_tokens": int(prompt_tokens or 0),
        "total_completion_tokens": int(completion_tokens or 0),
        "num_ctx_at_time": int(num_ctx_at_time or 0),
        "batch_item_label": batch_item_label,
        "notes": notes,
    }


def record_call(store, chat_id, entry, private=False):
    """Append ``entry`` to the chat's persisted history (newest last), capped to
    MAX_HISTORY_PER_CHAT. No-op for private/incognito chats or a missing chat_id."""
    if not chat_id or private:
        return
    with store._lock:
        hist = store.context_history.setdefault(chat_id, [])
        hist.append(entry)
        if len(hist) > MAX_HISTORY_PER_CHAT:
            del hist[:-MAX_HISTORY_PER_CHAT]
        store.save_context_history()


def get_history(store, chat_id):
    """Return the persisted history list for a chat (empty list if none)."""
    return list(store.context_history.get(chat_id, []))
