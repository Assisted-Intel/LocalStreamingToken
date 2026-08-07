#!/usr/bin/env python3
"""
Local Streaming Token — prompt / query rewriting services.

Two independent, reusable features (both used by normal chat now and by the persona
pipeline later):

1. Retrieval query rewrite (``rewrite_queries``): turn a conversational message into a
   handful of retrieval-optimized search queries (pronouns/references resolved) plus a
   keyword list. Feeds the hybrid RAG layer, whose multi-query RRF merge means one weak
   rewording can't sink retrieval. Purely improves *what is retrieved*.

2. Instruction improvement (``improve_prompt``): rewrite the user's message into a
   clearer, more effective instruction so the model returns a better answer. Backs the
   on-demand "Rewrite" button. Purely improves *the prompt the model answers*.

Both are provider-agnostic: they run one non-streaming completion through whatever
provider adapter (Ollama / Anthropic / OpenAI-compatible) the caller passes in. Every
function degrades gracefully — on any error the caller's original text is returned, so
rewriting never breaks a send.
"""

import json
import re
import threading


# --------------------------- one-shot completion over a streaming adapter ---------------------------

def run_completion(adapter, model: str, messages: list,
                   num_ctx: int = 4096, max_tokens: int = 1024, fmt=None,
                   temperature=None, stop=None) -> str:
    """Collect a full (non-streamed) assistant answer from a provider adapter's
    ``chat_stream``. ``fmt`` (a JSON schema or "json") is forwarded to Ollama for
    structured output when the adapter supports it; other adapters ignore it.

    ``stop`` is an optional caller-owned threading.Event. Without one this allocated a
    private Event that nothing could reach, which is why Stop did nothing during a
    persona pipeline's structured steps — every one of them ran to completion. The
    one-shot ``client.complete`` path can't be interrupted once it starts, so a stop
    already set short-circuits it instead."""
    stop = stop if stop is not None else threading.Event()
    if stop.is_set():
        return ""
    options = {"num_ctx": num_ctx, "max_output_tokens": max_tokens}
    if temperature is not None:
        options["temperature"] = float(temperature)
    parts = []
    # Prefer a real one-shot when the adapter exposes one (OllamaAdapter.client.complete).
    client = getattr(adapter, "client", None)
    if fmt is not None and hasattr(client, "complete"):
        try:
            return client.complete(model, messages, num_ctx=num_ctx, fmt=fmt,
                                   temperature=temperature, timeout=120)
        except Exception:
            pass
    for kind, text in adapter.chat_stream(model, messages, options, stop,
                                          think=False, tools=None, tool_executor=None):
        if kind == "content":
            parts.append(text)
    return "".join(parts).strip()


# --------------------------- helpers ---------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str):
    """Best-effort parse of a JSON object from a model reply that may wrap it in prose
    or code fences. Returns a dict or None."""
    if not text:
        return None
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Fall back to the first {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def history_text(messages: list, max_turns: int = 6, max_chars: int = 1500) -> str:
    """A compact transcript of the last few turns (excluding the final user message,
    which is passed separately) for reference resolution."""
    msgs = [m for m in (messages or [])
            if m.get("role") in ("user", "assistant") and not m.get("intermediate")]
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]                      # drop the message being rewritten
    msgs = msgs[-max_turns:]
    lines = []
    for m in msgs:
        who = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{who}: {content}")
    text = "\n".join(lines)
    return text[-max_chars:]


# --------------------------- 1. Retrieval query rewrite ---------------------------

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}

_QUERY_SYS = (
    "You rewrite a user's chat message into search queries for retrieving relevant "
    "passages from a knowledge base. Resolve pronouns and vague references using the "
    "conversation so far. Produce 2-3 diverse, self-contained query variants and a short "
    "list of important keywords. Respond ONLY with JSON of the form "
    '{"queries": ["...", "..."], "keywords": ["...", "..."]}.')


def rewrite_queries(adapter, model: str, message: str, history: str = "",
                    n_variants: int = 3) -> dict:
    """Return {"variants": [str,...], "keywords": [str,...]}. Falls back to the raw
    message as the single variant on any failure (retrieval still runs)."""
    fallback = {"variants": [message] if message else [], "keywords": []}
    if not (message and message.strip()):
        return fallback
    user = (f"Conversation so far:\n{history}\n\n" if history else "") + \
        f"User message:\n{message}\n\nJSON:"
    try:
        raw = run_completion(
            adapter, model,
            [{"role": "system", "content": _QUERY_SYS},
             {"role": "user", "content": user}],
            num_ctx=4096, max_tokens=400, fmt=_QUERY_SCHEMA)
    except Exception:
        return fallback
    parsed = _extract_json(raw) or {}
    variants = [q.strip() for q in (parsed.get("queries") or []) if isinstance(q, str) and q.strip()]
    keywords = [k.strip() for k in (parsed.get("keywords") or []) if isinstance(k, str) and k.strip()]
    variants = variants[:max(1, n_variants)]
    if message and message not in variants:
        variants = variants + [message]        # always keep the literal message as one variant
    if not variants:
        return fallback
    return {"variants": variants, "keywords": keywords}


# --------------------------- 2. Instruction improvement ---------------------------

_IMPROVE_SYS = (
    "You are an expert prompt engineer. Rewrite the user's instruction so an AI assistant "
    "produces a better, more useful, more precise result. Preserve the user's original "
    "intent, constraints, tone, and any specific details or formatting requests; make the "
    "instruction clearer and more complete, adding helpful specificity only where it "
    "reflects the obvious intent. Do NOT answer the instruction or add commentary. "
    "Return ONLY the rewritten instruction text.")


def improve_prompt(adapter, model: str, text: str, history: str = "") -> str:
    """Rewrite ``text`` into a stronger instruction. Returns the original on failure."""
    if not (text and text.strip()):
        return text or ""
    user = (f"Conversation context (for reference only):\n{history}\n\n" if history else "") + \
        f"Original instruction:\n{text}\n\nRewritten instruction:"
    try:
        out = run_completion(
            adapter, model,
            [{"role": "system", "content": _IMPROVE_SYS},
             {"role": "user", "content": user}],
            num_ctx=4096, max_tokens=1024)
    except Exception:
        return text
    out = (out or "").strip()
    # Strip a leading label the model might echo.
    out = re.sub(r"^(rewritten instruction|instruction)\s*:\s*", "", out, flags=re.IGNORECASE)
    return out or text
